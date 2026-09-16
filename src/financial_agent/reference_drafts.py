"""Compile author-written review aids, NEVER accepted evaluation labels.

Scope, quotes and arithmetic are machine checked. Entailment, unit/period
interpretation and completeness remain human review responsibilities.
"""

import re
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, localcontext

from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalSuite, decimal_value
from financial_agent.research_tools import stable_id


def compile_drafts(spec: dict, suite: EvalSuite, corpus: Corpus) -> dict:
    if spec.get("schema_version") != 1 or spec.get("suite_hash") != suite.fingerprint:
        raise ValueError("Draft suite/schema mismatch")
    cases = {case.case_id: case for case in suite.cases}
    drafts = spec["drafts"]
    if len(drafts) != len(cases) or {d["case_id"] for d in drafts} != set(cases):
        raise ValueError("Draft cases must cover suite exactly once")
    output = []
    for original in drafts:
        draft = deepcopy(original)
        case = cases[draft["case_id"]]
        if case.partition != "candidate":
            raise ValueError("Draft compiler accepts candidate tasks only, not historical B")
        if draft.get("status") != "assistant_draft" or draft.get("reviewer") is not None:
            raise ValueError("Draft is not human-approved gold")
        if draft.get("coverage") not in {"answerable_draft", "partial_needs_scope_review"}:
            raise ValueError("Explicit draft coverage required")
        if not draft.get("claims") or not draft.get("rubric") or not draft.get("gaps"):
            raise ValueError("Claims, review rubric and limitations required")
        scope = frozenset(case.document_version_ids)
        if not scope <= corpus.by_version.keys():
            raise ValueError("Task source version unavailable")
        bound = {}

        def bind(alias):
            if alias in bound:
                return bound[alias]
            anchor = spec["anchors"][alias]
            ref = corpus.refs[anchor["chunk_id"]]
            source = corpus.read(ref, scope)
            quote = anchor["quote"]
            # A quote is a navigation anchor, not proof of every linked claim.
            if not isinstance(quote, str) or not quote or source.count(quote) != 1:
                raise ValueError(f"Quote must occur exactly once: {alias}")
            start = source.index(quote)
            bound[alias] = {
                "reference": asdict(ref), "quote": quote,
                "quote_start": start, "quote_end": start + len(quote),
                "full_chunk_text": source,
            }
            return bound[alias]

        for claim in draft["claims"]:
            if not claim.get("text") or not claim.get("evidence"):
                raise ValueError("Each draft claim needs text and evidence")
            for alias in claim["evidence"]:
                bind(alias)
        for calc in draft.get("calculations", []):
            if not calc.get("context") or not calc.get("context_evidence") or not calc.get("unit"):
                raise ValueError("Calculation context required for human review")
            for alias in calc["context_evidence"]:
                bind(alias)
            values = []
            for key in ("a", "b"):
                operand = calc[key]
                source = bind(operand["anchor"])["full_chunk_text"]
                raw = operand["raw"]
                if (not isinstance(raw, str)
                        or re.fullmatch(r"-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", raw) is None
                        or re.search(r"(?<![\d,.\-])" + re.escape(raw) + r"(?![\d,.])", source) is None):
                    raise ValueError("Calculation operand absent from cited source")
                values.append(decimal_value(raw.replace(",", "")))
            a, b = values
            if b <= 0:
                raise ValueError("Reference ratio/growth requires positive base")
            with localcontext() as ctx:
                ctx.prec = 40
                if calc["operation"] == "ratio":
                    result = a / b
                elif calc["operation"] == "growth_percent":
                    result = (a - b) / b * Decimal(100)
                else:
                    raise ValueError("Unsupported draft calculation")
                calc["computed_value"] = str(result)
                calc["rounded_4dp"] = str(result.quantize(Decimal("0.0001")))
            calc["semantic_context_verified"] = False
        audits = []
        for search in draft.get("searches", []):
            for version in sorted(scope):
                chunks = [c for c in corpus.chunks.values()
                          if corpus.documents[c.doc_id].version_id == version]
                for term in search["terms"]:
                    if not isinstance(term, str) or not term:
                        raise ValueError("Nonempty search term required")
                    hits = []
                    for chunk in chunks:
                        text = corpus.read(corpus.refs[chunk.chunk_id], scope)
                        if term in text:
                            hits.append({"reference": asdict(corpus.refs[chunk.chunk_id]),
                                         "count": text.count(term)})
                    audits.append({"version_id": version, "term": term,
                                   "chunks_scanned": len(chunks), "hits": hits,
                                   "proves_absence": False})
        draft.update(query=case.query, document_version_ids=case.document_version_ids,
                     evidence=bound, search_audit=audits)
        output.append(draft)
    return {
        "artifact_type": "assistant_reference_review_packet",
        "suite_hash": suite.fingerprint, "spec_hash": stable_id("draft-", spec),
        "provenance": spec["provenance"], "approved_gold_count": 0,
        "quality_claim_permitted": False, "semantic_review_complete": False,
        "note": "Quotes/arithmetic checked only; not Agent predictions, hidden B gold or human labels.",
        "drafts": output,
    }


def render_packet(packet: dict) -> str:
    lines = ["# 新增研究任务参考草案（待人工复核）", "",
             "不是 B 榜隐藏答案，不是 Agent 自动实验结果，不得计入准确率。",
             "原文来自比赛语料，未核验外部真实性或法规现行效力。",
             "机器仅核对范围、引用完整性和算术；语义、单位、期间及完整性须人工复核。", "",
             f"任务版本：`{packet['suite_hash']}`", ""]
    for draft in packet["drafts"]:
        lines.extend([f"## {draft['case_id']} · {draft['coverage']}", "", draft["query"], "",
                      "### 答案草案", ""])
        for claim in draft["claims"]:
            lines.append(claim["text"] + " 【" + "、".join(claim["evidence"]) + "】\n")
        for calc in draft.get("calculations", []):
            lines.append(f"计算核对：{calc['name']} = {calc['rounded_4dp']} {calc['unit']}；"
                         f"原值 {calc['a']['raw']} / {calc['b']['raw']}；运算 {calc['operation']}。\n")
        lines.extend(["### 证据缺口 / 人工复核", ""])
        lines.extend("- " + gap for gap in draft["gaps"])
        lines.extend(["", "### 建议评分要点（尚未批准）", ""])
        lines.extend("- " + criterion for criterion in draft["rubric"])
        lines.extend(["", "### 来源定位与完整片段", ""])
        for alias, item in draft["evidence"].items():
            ref = item["reference"]
            lines.extend([f"#### {alias} · {ref['doc_id']} · 物理页 {ref['physical_page']}", "",
                          f"chunk：`{ref['chunk_id']}`；文本 SHA256：`{ref['text_sha256']}`", "",
                          f"定位短语（不是整段断言的证明）：{item['quote']}", "",
                          "```text", item["full_chunk_text"], "```", ""])
        if draft["search_audit"]:
            lines.extend(["### 全文词面检索记录（不证明语义不存在）", ""])
            for audit in draft["search_audit"]:
                lines.append(f"- {audit['version_id'].split('@')[0]} / {audit['term']}："
                             f"扫描 {audit['chunks_scanned']} chunks，命中 {len(audit['hits'])} chunks。")
            lines.append("")
    return "\n".join(lines) + "\n"
