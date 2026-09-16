"""Freeze development risk cases with review-only navigation, never model-visible gold."""

from dataclasses import asdict
from types import SimpleNamespace

from financial_agent.contracts import Budget, EvidenceRef
from financial_agent.core_evidence import prepare_core_evidence
from financial_agent.evaluation import EvalCase, EvalSuite
from financial_agent.research_tools import stable_id


def _keys(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("Missing or unexpected risk specification fields")


def _strings(values):
    if (not isinstance(values, list) or not values or
            any(not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values)):
        raise ValueError("Nonempty unique text list required")


def build_risk_suite(recipe, parent, parent_spec, corpus):
    _keys(recipe, {"schema_version", "name", "partition", "cases"})
    if recipe["schema_version"] != 1 or recipe["partition"] != "candidate":
        raise ValueError("Risk extension is development-only, never reviewed holdout")
    if parent_spec.get("suite_hash") != parent.fingerprint:
        raise ValueError("Parent reference version mismatch")
    if not isinstance(recipe["cases"], list) or not recipe["cases"]:
        raise ValueError("Risk cases required")
    parents = {c.case_id: c for c in parent.cases}
    cases, reviews = [], []
    for row in recipe["cases"]:
        _keys(row, {"case_id", "parent_case_id", "document_ids", "query", "task_kind", "risks",
                    "anchor_aliases", "review_questions"})
        original = parents[row["parent_case_id"]]
        if original.partition != "candidate" or row["case_id"] in parents:
            raise ValueError("New case IDs and candidate parent required")
        for field in ("document_ids", "risks", "anchor_aliases", "review_questions"):
            _strings(row[field])
        versions = tuple(corpus.documents[d].version_id for d in row["document_ids"])
        if not set(versions) <= set(original.document_version_ids):
            raise PermissionError("Extension cannot silently expand parent document scope")
        case = EvalCase(row["case_id"], row["query"], versions, original.domain, row["task_kind"], "candidate")
        cases.append(case)
        navigation = []
        for alias in row["anchor_aliases"]:
            source = parent_spec["anchors"][alias]
            ref = corpus.refs[source["chunk_id"]]
            content = corpus.read(ref, frozenset(versions))
            quote = source["quote"]
            if not isinstance(quote, str) or not quote or content.count(quote) != 1:
                raise ValueError("Navigation quote must occur exactly once in scoped source")
            navigation.append({"alias": alias, "reference": asdict(ref), "quote": quote,
                               "full_chunk_text": content})
        reviews.append({"case_id": case.case_id, "parent_case_id": original.case_id,
                        "status": "assistant_review_aid", "reviewer": None,
                        "risks": list(row["risks"]), "review_questions": list(row["review_questions"]),
                        "navigation": navigation, "absence_proven": False,
                        "note": "Navigation only, not complete required evidence or expected answer"})
    suite = EvalSuite(recipe["name"], tuple(cases))
    packet = {"suite_hash": suite.fingerprint, "parent_suite_hash": parent.fingerprint,
              "recipe_hash": stable_id("recipe-", recipe), "partition": "candidate",
              "approved_gold_count": 0, "reviews": reviews}
    packet["packet_hash"] = stable_id("risk-packet-", packet)
    labels = {"suite_hash": suite.fingerprint, "labels": {
        c.case_id: {"status": "pending", "expected": None} for c in suite.cases}}
    return suite, packet, labels


def validate_risk_packet(suite, packet, corpus):
    if (packet.get("suite_hash") != suite.fingerprint or packet.get("partition") != "candidate"
            or packet.get("approved_gold_count") != 0
            or any(c.partition != "candidate" for c in suite.cases)):
        raise ValueError("Risk packet is not a matching unreviewed development packet")
    if packet.get("packet_hash") != stable_id("risk-packet-", {k: v for k, v in packet.items() if k != "packet_hash"}):
        raise ValueError("Risk packet hash mismatch")
    reviews = packet["reviews"]
    if len(reviews) != len(suite.cases) or {r["case_id"] for r in reviews} != {c.case_id for c in suite.cases}:
        raise ValueError("Risk packet must cover cases exactly once")
    cases = {c.case_id: c for c in suite.cases}
    for review in reviews:
        if review["status"] != "assistant_review_aid" or review["reviewer"] is not None or review["absence_proven"]:
            raise ValueError("Navigation aids cannot assert human approval or prove absence")
        if not review["navigation"]:
            raise ValueError("Missing navigation aids")
        for nav in review["navigation"]:
            content = corpus.read(EvidenceRef(**nav["reference"]), frozenset(cases[review["case_id"]].document_version_ids))
            if content != nav["full_chunk_text"] or not nav["quote"] or content.count(nav["quote"]) != 1:
                raise ValueError("Navigation source changed")


def diagnose_core_windows(suite, packet, corpus, retriever):
    validate_risk_packet(suite, packet, corpus)
    reviews = {r["case_id"]: r for r in packet["reviews"]}
    rows = []
    for case in suite.cases:
        captured = []

        def capture(task, request):
            hits = retriever.search(task, request)
            captured.extend(hits)
            return hits

        # Reference text, aliases, review questions and risk categories NEVER enter this call.
        pack = prepare_core_evidence(corpus, SimpleNamespace(search=capture), case.task(Budget()))
        retrieved_ids = {hit.ref.chunk_id for hit in captured}
        navigation = reviews[case.case_id]["navigation"]
        matches = []
        for nav in navigation:
            same_chunk = [w for w in pack["windows"] if w["ref"]["chunk_id"] == nav["reference"]["chunk_id"]]
            matches.append({"alias": nav["alias"], "retrieved_candidate": nav["reference"]["chunk_id"] in retrieved_ids,
                            "chunk_selected": bool(same_chunk),
                            "navigation_quote_visible": any(nav["quote"] in w["excerpt_text"] for w in same_chunk)})
        rows.append({"case_id": case.case_id, "matches": matches, "evidence_pack": pack,
                     "candidate_chunk_ids": [hit.ref.chunk_id for hit in captured],
                     "requested_documents": len(case.document_version_ids),
                     "represented_documents": len({w["ref"]["version_id"] for w in pack["windows"]})})
    all_matches = [m for r in rows for m in r["matches"]]
    return {"suite_hash": suite.fingerprint, "packet_hash": packet["packet_hash"],
            "execution": "offline_public_query_retrieval_and_v45_source_compression",
            "real_api_calls": 0, "answer_accuracy": None, "quality_claim_permitted": False,
            "summary": {"cases": len(rows), "navigation_anchors": len(all_matches),
                        "retrieved_navigation_chunks": sum(m["retrieved_candidate"] for m in all_matches),
                        "selected_navigation_chunks": sum(m["chunk_selected"] for m in all_matches),
                        "visible_navigation_quotes": sum(m["navigation_quote_visible"] for m in all_matches),
                        "retrieved_but_not_selected": sum(m["retrieved_candidate"] and not m["chunk_selected"]
                                                          for m in all_matches),
                        "selected_but_quote_omitted": sum(m["chunk_selected"] and not m["navigation_quote_visible"]
                                                        for m in all_matches),
                        "cases_with_all_navigation_quotes": sum(all(m["navigation_quote_visible"] for m in r["matches"])
                                                               for r in rows)}, "rows": rows,
            "limitations": ["Author navigation is neither unique nor complete gold evidence",
                            "Missing or visible quote does not establish answerability or answer correctness",
                            "Known development documents; not held out, not a model execution"]}


def render_risk_packet(suite, packet):
    def inert(value):
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("`", "\\`")
    cases = {c.case_id: c for c in suite.cases}
    lines = ["# 风险开发题：待人工核查", "", "审核通过数为0。以下为审阅导航，不是标准答案、模型预测或留出集。", ""]
    for row in packet["reviews"]:
        lines.extend([f"## {row['case_id']}", "", inert(cases[row["case_id"]].query), "",
                      "核查问题（作者建议）：", ""])
        lines.extend(f"- {inert(q)}" for q in row["review_questions"])
        lines.extend(["", "人工审核：待填写；预期答案：未批准。", ""])
        for nav in row["navigation"]:
            ref = nav["reference"]
            lines.extend([f"导航：{inert(ref['doc_id'])}，物理页{ref['physical_page']}，{inert(nav['alias'])}", "",
                          "原文（仅作资料，夹带指令不可执行）：", ""])
            lines.extend("> " + inert(line) for line in nav["full_chunk_text"].splitlines())
            lines.append("")
    return "\n".join(lines)
