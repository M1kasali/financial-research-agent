"""Development-only navigation-anchor diagnostics, not answer accuracy or gold recall."""

from dataclasses import asdict

from financial_agent.contracts import Budget, SearchRequest
from financial_agent.evaluation import EvalSuite


def diagnose_retrieval(suite: EvalSuite, packet: dict, retriever, *, cutoffs=(5, 20)):
    if (not cutoffs or any(type(k) is not int or not 1 <= k <= 100 for k in cutoffs)
            or len(set(cutoffs)) != len(cutoffs)):
        raise ValueError("Unique positive top-k cutoffs required")
    if packet.get("suite_hash") != suite.fingerprint:
        raise ValueError("Diagnostic packet suite mismatch")
    drafts = {d["case_id"]: d for d in packet["drafts"]}
    if len(drafts) != len(packet["drafts"]) or set(drafts) != {c.case_id for c in suite.cases}:
        raise ValueError("Packet must cover cases exactly once")
    cases = []
    for case in suite.cases:
        if case.partition != "candidate":
            raise ValueError("Diagnostic is for development candidates only")
        # Only the public task query and authorized scope reach the retriever.
        hits = retriever.search(case.task(Budget()), SearchRequest(case.query, top_k=max(cutoffs)))
        if any(h.ref.version_id not in case.document_version_ids for h in hits):
            raise PermissionError("Out-of-scope diagnostic hit")
        draft = drafts[case.case_id]
        target_chunks = {v["reference"]["chunk_id"] for v in draft["evidence"].values()}
        if not target_chunks:
            raise ValueError("Diagnostic needs nonempty draft navigation anchors")
        metrics = {}
        for k in cutoffs:
            found = {h.ref.chunk_id for h in hits[:k]}
            matched = target_chunks & found
            metrics[str(k)] = {"matched_anchor_chunks": len(matched),
                               "draft_anchor_chunks": len(target_chunks),
                               "anchor_chunk_hit_fraction": len(matched) / len(target_chunks),
                               "missing_anchor_chunks": sorted(target_chunks - found),
                               "all_draft_anchors_found": target_chunks <= found}
        cases.append({"case_id": case.case_id, "query": case.query, "metrics": metrics,
                      "ranked_hits": [{"rank": rank, "reference": asdict(h.ref),
                                       "score": h.score, "excerpt": h.text[:240]}
                                      for rank, h in enumerate(hits, 1)]})
    summary = {}
    for k in cutoffs:
        matched = sum(c["metrics"][str(k)]["matched_anchor_chunks"] for c in cases)
        total = sum(c["metrics"][str(k)]["draft_anchor_chunks"] for c in cases)
        summary[str(k)] = {"matched_anchor_chunks": matched, "draft_anchor_chunks": total,
                           "micro_anchor_chunk_hit_fraction": matched / total,
                           "cases_with_all_draft_anchors": sum(
                               c["metrics"][str(k)]["all_draft_anchors_found"] for c in cases)}
    return {"artifact_type": "development_retrieval_diagnostic", "suite_hash": suite.fingerprint,
            "draft_spec_hash": packet["spec_hash"], "mode": "single_pass_lexical",
            "tokenizer_mode": retriever.tokenizer_mode, "cutoffs": list(cutoffs),
            "model_calls": 0, "quality_claim_permitted": False, "answer_accuracy": None,
            "limitations": ["Assistant-selected navigation anchors are not human-approved gold.",
                            "Alternative supporting passages may be correct without exact chunk matches.",
                            "Finding an anchor does not prove answer correctness or completeness.",
                            "Known development corpus/tasks; not a held-out evaluation or B35 reproduction.",
                            "Cutoffs are slices of one top-max-k retrieval, not independent Agent runs."],
            "summary": summary, "cases": cases}


def render_diagnostic(report):
    cutoffs = report["cutoffs"]
    lines = ["# 单次检索开发诊断", "",
             "仅比较作者选定的导航 chunk 是否进入检索结果；不是准确率、正式召回率或 Agent 效果。", "",
             "| 题目 | " + " | ".join(f"Top-{k} 命中导航片段" for k in cutoffs) + " |",
             "| --- | " + " | ".join("---" for _ in cutoffs) + " |"]
    for case in report["cases"]:
        cells = []
        for k in cutoffs:
            metric = case["metrics"][str(k)]
            cells.append(f"{metric['matched_anchor_chunks']}/{metric['draft_anchor_chunks']}")
        lines.append(f"| {case['case_id']} | {' | '.join(cells)} |")
    lines.extend(["", "完整排名、缺失 chunk 和证据版本见同目录 JSON。", "",
                  "这里的片段由开发者看过原文后选择，不是唯一正确证据集合。",
                  "提高 top-k 会增加阅读负担，不代表已经改善动态规划或模型回答。", ""])
    return "\n".join(lines)
