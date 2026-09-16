"""Scoped adapters around the original retrieval, fact ledger and calculation DSL.

No model-generated Python, expressions, document IDs or numeric operands are executed.
Facts are candidates, not truth labels. Arithmetic requires grounded accounting context.
"""

import json
from dataclasses import asdict
from decimal import ROUND_HALF_UP, Decimal

from agent.reasoning.calculation_dsl import evaluate_calculation
from agent.reasoning.fact_ledger import compile_numeric_fact_ledger
from agent.schemas import Question, RetrievalResult

from financial_agent.contracts import SearchRequest, TaskRequest
from financial_agent.corpus import Corpus, sha256_text
from financial_agent.retrieval import ScopedRetriever
from financial_agent.statement_binding import parse_statement


def stable_id(prefix: str, value: object) -> str:
    return prefix + sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def require_keys(args: dict, required: set, optional: set | None = None):
    if (
        not isinstance(args, dict)
        or not required <= args.keys()
        or args.keys() - required - (optional or set())
    ):
        raise ValueError("Invalid tool arguments")


def id_list(value: object, *, maximum: int = 12) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ValueError("Expected a non-empty bounded ID list")
    if any(not isinstance(v, str) or not v for v in value) or len(set(value)) != len(value):
        raise ValueError("Invalid or duplicate IDs")
    return value


class ResearchTools:
    def __init__(self, corpus: Corpus, retriever: ScopedRetriever, task: TaskRequest):
        self.corpus, self.retriever, self.task = corpus, retriever, task
        self.allowed = frozenset(task.document_version_ids)
        if not self.allowed <= corpus.by_version.keys():
            raise ValueError("Unknown task document version")
        self.evidence: dict = {}
        self.facts: dict = {}
        self.calculations: dict = {}

    def read_evidence(self, evidence_id: str) -> str:
        if not isinstance(evidence_id, str) or evidence_id not in self.evidence:
            raise ValueError("Unknown evidence ID; search first")
        return self.corpus.read(self.evidence[evidence_id], self.allowed)

    def search(self, args: dict) -> dict:
        require_keys(args, {"query"}, {"top_k", "document_version_ids"})
        versions = args.get("document_version_ids")
        if versions is not None:
            # Explicit [] means no documents, never unrestricted access.
            if not isinstance(versions, list) or any(not isinstance(v, str) for v in versions):
                raise ValueError("Invalid document scope")
            versions = tuple(versions)
        top_k = args.get("top_k", 5)
        if type(top_k) is not int or not 1 <= top_k <= 8:
            raise ValueError("Runtime search top_k must be in [1,8]")
        hits = self.retriever.search(self.task, SearchRequest(args["query"], versions, top_k))
        output = []
        for hit in hits:
            self.evidence[hit.ref.evidence_id] = hit.ref
            output.append(
                {"ref": asdict(hit.ref), "text": hit.text[:1000], "truncated": len(hit.text) > 1000}
            )
        return {"hits": output, "sufficient": "not_assessed"}

    def _register_hits(self, hits):
        output = []
        # Validate the whole returned batch before publishing any new references.
        for hit in hits:
            self.corpus.read(hit.ref, self.allowed)
        for hit in hits:
            self.evidence[hit.ref.evidence_id] = hit.ref
            text = self.corpus.read(hit.ref, self.allowed)
            output.append({"ref": asdict(hit.ref), "text": text[:1000],
                           "truncated": len(text) > 1000})
        return output

    def search_documents(self, args: dict) -> dict:
        require_keys(args, {"query", "document_version_ids"}, {"top_k"})
        versions = tuple(id_list(args["document_version_ids"], maximum=4))
        top_k = args.get("top_k", 8)
        if type(top_k) is not int or not len(versions) <= top_k <= 8:
            raise ValueError("Document count <= top_k <= 8 required")
        hits = self.retriever.search_balanced(self.task, SearchRequest(args["query"], versions, top_k))
        if any(hit.ref.version_id not in versions for hit in hits):
            raise PermissionError("Balanced results exceed requested document subset")
        return {"hits": self._register_hits(hits), "sufficient": "not_assessed",
                "strategy": "document_round_robin_exact_dedup",
                "local_index_queries": len(versions), "logical_tool_calls": 1,
                "returned_per_document": {v: sum(h.ref.version_id == v for h in hits) for v in versions},
                "note": "Document coverage is not semantic sufficiency; inspect each source."}

    def expand_context(self, args: dict) -> dict:
        require_keys(args, {"evidence_id", "query"}, {"radius", "top_k"})
        self.read_evidence(args["evidence_id"])
        seed = self.evidence[args["evidence_id"]]
        radius, top_k = args.get("radius", 1), args.get("top_k", 6)
        hits = self.retriever.context_candidates(self.task, seed, args["query"], radius=radius, top_k=top_k)
        if any(hit.ref.version_id != seed.version_id for hit in hits):
            raise PermissionError("Context results must retain seed document version")
        output = self._register_hits(hits)
        for row in output:
            page = row["ref"]["physical_page"]
            row["page_distance"] = (
                None if page is None or seed.physical_page is None else abs(page - seed.physical_page))
        return {"hits": output, "seed_evidence_id": seed.evidence_id,
                "document_version_id": seed.version_id, "radius": radius,
                "local_index_queries": 1, "logical_tool_calls": 1,
                "strategy": "same_version_query_and_neighbor_candidates",
                "sufficient": "not_assessed", "accounting_context_bound": False,
                "note": "Query may find distant pages. Proximity does not prove shared entity/unit/scope/period; "
                        "read and verify. Facts and calculation permissions were not changed."}

    def read(self, args: dict) -> dict:
        require_keys(args, {"evidence_id"}, {"offset", "length"})
        text = self.read_evidence(args["evidence_id"])
        offset, length = args.get("offset", 0), args.get("length", 4000)
        if type(offset) is not int or not 0 <= offset < max(1, len(text)):
            raise ValueError("Invalid offset")
        if type(length) is not int or not 1 <= length <= 4000:
            raise ValueError("Invalid length")
        return {
            "evidence_id": args["evidence_id"],
            "offset": offset,
            "text": text[offset : offset + length],
            "has_more": offset + length < len(text),
        }

    def extract(self, args: dict) -> dict:
        require_keys(args, {"evidence_ids"})
        ids = id_list(args["evidence_ids"], maximum=8)
        results = []
        refs = {}
        for key in ids:
            text = self.read_evidence(key)
            ref = self.evidence[key]
            chunk = self.corpus.chunks[ref.chunk_id]
            refs[ref.chunk_id] = ref
            results.append(
                RetrievalResult(
                    chunk_id=ref.chunk_id,
                    doc_id=ref.doc_id,
                    domain=chunk.domain,
                    score=1.0,
                    source="version_bound",
                    query=self.task.query,
                    evidence_text=text,
                    metadata=self.corpus.source_metadata[ref.chunk_id],
                )
            )
        # Legacy Question is confined to this adapter; free-form tasks have no answer slots.
        question = Question(self.task.task_id, "financial_reports", "research", self.task.query, {}, "mcq")
        ledger = compile_numeric_fact_ledger(question, results, max_facts=36)
        output = []
        for raw in ledger["facts"]:
            ref = refs[raw["chunk_id"]]
            text = self.read_evidence(ref.evidence_id)
            context = self.corpus.source_metadata[ref.chunk_id].get("financial_context", {})
            entity = context.get("entity") if isinstance(context, dict) else None
            # Only explicit source-anchored context is accepted. Missing metadata stays unknown.
            entity = entity if isinstance(entity, str) and entity and entity in text else "unknown"
            currency = "CNY" if "人民币" in text or "CNY" in text else "unknown"
            if any(term in text for term in ("美元", "港元", "欧元", "USD", "HKD", "EUR")):
                currency = "unknown"
            period_kind = "annual" if "年度" in text else "unknown"
            if any(term in text for term in ("季度", "半年", "个月", "月度")):
                period_kind = "unknown"
            parent = any(term in text for term in ("母公司财务报表", "母公司利润表", "母公司现金流量表"))
            consolidated = any(term in text for term in ("合并利润表", "合并财务报表", "合并现金流量表"))
            scope = ("parent" if parent else "consolidated") if parent != consolidated else "unknown"
            fact = {k: v for k, v in raw.items() if k != "fact_id"}
            fact.update(
                evidence_id=ref.evidence_id,
                entity=entity,
                currency=currency,
                period_kind=period_kind,
                scope=scope,
                version_id=ref.version_id,
                status="candidate",
            )
            fact["fact_id"] = stable_id("F-", fact)
            self.facts[fact["fact_id"]] = fact
            output.append(fact)
        return {"facts": output, "fact_count": len(output)}

    def bind_statement(self, args: dict) -> dict:
        require_keys(args, {"evidence_id", "context_evidence_ids", "metric", "expected_scope"})
        context_ids = id_list(args["context_evidence_ids"], maximum=3)
        facts = parse_statement(args["evidence_id"], context_ids, args["metric"], args["expected_scope"],
                                self.evidence, self.read_evidence)
        # New immutable identities; do not overwrite or silently repair legacy facts.
        for fact in facts:
            fact["fact_id"] = stable_id("F-", fact)
        self.facts.update({f["fact_id"]: f for f in facts})
        return {"facts": facts, "fact_count": len(facts), "binding": "source_spans_reparsed",
                "human_approved": False,
                "limitations": ["Only the supported complete two-year annual-statement layout is accepted.",
                                "Requested scope must still match the user's task; model review is not gold."]}

    def _revalidate_statement_fact(self, fact: dict):
        request = fact["binding_request"]
        reconstructed = parse_statement(request["evidence_id"], request["context_evidence_ids"],
                                        request["metric"], request["expected_scope"],
                                        self.evidence, self.read_evidence)
        for candidate in reconstructed:
            candidate["fact_id"] = stable_id("F-", candidate)
        if fact not in reconstructed:
            raise ValueError("Bound statement fact changed since source binding")

    def calculate(self, args: dict) -> dict:
        require_keys(args, {"operation", "fact_ids"}, {"decimal_places"})
        operation = args["operation"]
        if operation not in {"compare", "difference", "ratio", "growth_rate"}:
            raise ValueError("Unsupported calculation operation")
        ids = id_list(args["fact_ids"], maximum=2)
        if len(ids) != 2 or any(key not in self.facts for key in ids):
            raise ValueError("Two existing facts are required")
        places = args.get("decimal_places", 2)
        if type(places) is not int or not 0 <= places <= 8:
            raise ValueError("Invalid decimal precision")
        left, right = (self.facts[key] for key in ids)
        for fact in (left, right):
            self.read_evidence(fact["evidence_id"])
            if fact.get("extraction_mode") == "statement_table_v1" or "binding_request" in fact:
                self._revalidate_statement_fact(fact)
            elif fact.get("extraction_mode") != "financial_row":
                raise ValueError("Unbound text-regex facts require bind_statement before calculation")
            if any(
                fact.get(k) in (None, "", "unknown")
                for k in ("entity", "currency", "scope", "period_kind", "year", "metric")
            ):
                raise ValueError(
                    "Accounting context incomplete; retrieve explicit entity/currency/scope/period"
                )
            if fact["unit"] not in {"元", "千元", "万元", "亿元"}:
                raise ValueError("Only explicitly normalized monetary facts are supported")
            if not Decimal(fact["normalized_value"]).is_finite():
                raise ValueError("Non-finite numeric fact")
        if any(left[k] != right[k] for k in ("currency", "scope", "period_kind")):
            raise ValueError("Incompatible currency, scope or period kind")
        if all(left[k] == right[k] for k in ("entity", "metric", "year")) and Decimal(
            left["normalized_value"]
        ) != Decimal(right["normalized_value"]):
            raise ValueError(
                "Conflicting disclosures for the same entity, metric and period; resolve before calculation"
            )
        if operation == "growth_rate":
            if left["entity"] != right["entity"] or left["metric"] != right["metric"]:
                raise ValueError("Growth requires the same entity and metric")
            if (
                not (str(left["year"]).isdigit() and str(right["year"]).isdigit())
                or int(left["year"]) != int(right["year"]) + 1
            ):
                raise ValueError("Growth currently supports adjacent annual periods only")
            if Decimal(right["normalized_value"]) <= 0:
                raise ValueError("Growth requires a positive base under this policy")
        elif left["year"] != right["year"]:
            raise ValueError("Calculation requires the same period")
        elif operation in {"compare", "difference"} and left["metric"] != right["metric"]:
            raise ValueError("Comparison requires the same metric")
        elif operation == "ratio" and left["entity"] != right["entity"]:
            raise ValueError("Ratio requires the same entity")
        result = evaluate_calculation({"facts": [left, right]}, operation=operation, operands=ids)
        if operation == "compare":
            display, unit = result["result"], "relation"
        else:
            value = Decimal(result["result"])
            unit = "%" if operation == "growth_rate" else ("CNY" if operation == "difference" else "ratio")
            if operation == "growth_rate":
                value *= 100
            display = str(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))
        result.update(
            display_value=display,
            display_unit=unit,
            decimal_places=places,
            evidence_ids=list(dict.fromkeys(
                eid for f in (left, right) for eid in f.get("evidence_ids", [f["evidence_id"]]))),
        )
        result["policy_version"] = "annual-cny-v1"
        result["calculation_id"] = stable_id("C-", result)
        self.calculations[result["calculation_id"]] = result
        return result
