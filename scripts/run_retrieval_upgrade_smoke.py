"""Offline tool-capability experiment, NOT model-driven planning or answer evaluation.

Same public query and top-8 output cap for global vs per-document retrieval.
Generic accounting-context probes are separately labeled, with larger total evidence
and extra tool/index work; they are not a same-budget model comparison.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from financial_agent.contracts import Budget
from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalSuite
from financial_agent.reference_drafts import compile_drafts
from financial_agent.research_tools import ResearchTools
from financial_agent.retrieval import ScopedRetriever

CONTEXT_QUERY = "记账本位币 编制本财务报表 货币 单位"


def run(suite, packet, corpus, retriever):
    rows = []
    for case, draft in zip(suite.cases, packet["drafts"], strict=True):
        if case.case_id != draft["case_id"]:
            raise ValueError("Diagnostic case ordering mismatch")
        task = case.task(Budget())
        baseline = ResearchTools(corpus, retriever, task)
        upgraded = ResearchTools(corpus, retriever, task)
        start = time.perf_counter()
        global_result = baseline.search({"query": case.query, "top_k": 8})
        global_seconds = time.perf_counter() - start
        start = time.perf_counter()
        balanced = upgraded.search_documents({"query": case.query, "top_k": 8,
                                              "document_version_ids": list(case.document_version_ids)})
        balanced_seconds = time.perf_counter() - start
        probes = []
        if case.domain == "financial_reports":
            # Public domain + a generic predeclared currency/unit query only.
            # No draft anchors, answers or desired page numbers drive these probes.
            for version in case.document_version_ids:
                seed = next((h for h in balanced["hits"] if h["ref"]["version_id"] == version), None)
                if seed:
                    start = time.perf_counter()
                    result = upgraded.expand_context({"evidence_id": seed["ref"]["evidence_id"],
                                                      "query": CONTEXT_QUERY, "radius": 1, "top_k": 6})
                    probes.append({"query": CONTEXT_QUERY, "result": result,
                                   "elapsed_seconds": time.perf_counter() - start})
        # Only after all retrieval calls, compare manually selected draft navigation chunks.
        anchors = {v["reference"]["chunk_id"] for v in draft["evidence"].values()}
        global_ids = {h["ref"]["chunk_id"] for h in global_result["hits"]}
        balanced_ids = {h["ref"]["chunk_id"] for h in balanced["hits"]}
        expanded_ids = {ref.chunk_id for ref in upgraded.evidence.values()}
        rows.append({"case_id": case.case_id, "query": case.query,
                     "document_version_ids": list(case.document_version_ids), "anchor_count": len(anchors),
                     "global": {"hits": global_result["hits"], "anchor_matches": len(anchors & global_ids),
                                "local_index_queries": 1, "elapsed_seconds": global_seconds},
                     "balanced": {**balanced, "anchor_matches": len(anchors & balanced_ids),
                                  "elapsed_seconds": balanced_seconds},
                     "context_probes": probes,
                     "balanced_plus_context": {"unique_evidence_count": len(expanded_ids),
                                               "anchor_matches": len(anchors & expanded_ids),
                                               "missing_anchors": sorted(anchors - expanded_ids),
                                               "logical_tool_calls": 1 + len(probes),
                                               "local_index_queries": len(case.document_version_ids) + len(probes)}})
        print(f"Finished {case.case_id}: global={len(anchors & global_ids)}, "
              f"balanced={len(anchors & balanced_ids)}, plus_context={len(anchors & expanded_ids)} "
              f"of {len(anchors)} draft navigation chunks", flush=True)
    summary = {"cases": len(rows), "draft_anchor_count": sum(r["anchor_count"] for r in rows),
               "global_top8_anchor_matches": sum(r["global"]["anchor_matches"] for r in rows),
               "balanced_top8_anchor_matches": sum(r["balanced"]["anchor_matches"] for r in rows),
               "balanced_plus_context_anchor_matches": sum(r["balanced_plus_context"]["anchor_matches"] for r in rows),
               "context_probe_calls": sum(len(r["context_probes"]) for r in rows),
               "global_local_index_queries": len(rows),
               "balanced_local_index_queries": sum(r["balanced"]["local_index_queries"] for r in rows),
               "balanced_regression_cases": [r["case_id"] for r in rows
                                              if r["balanced"]["anchor_matches"] < r["global"]["anchor_matches"]],
               "model_calls": 0, "answer_accuracy": None}
    return {"suite_hash": suite.fingerprint, "draft_spec_hash": packet["spec_hash"],
            "execution_label": "deterministic_offline_tool_probe_not_model_agent",
            "tokenizer_mode": retriever.tokenizer_mode, "network_disabled": True,
            "context_probe_query": CONTEXT_QUERY, "quality_claim_permitted": False,
            "limitations": ["Assistant draft anchors, not approved gold; alternatives may also support answers.",
                            "Same top-8 cap does not imply equal local search work or runtime.",
                            "Context branch adds calls and up to six results per seed, not an equal-budget comparison.",
                            "No automatic fact-context binding, no calculation success or planning quality claim.",
                            "Known development cases; not a held-out evaluation or original B35 reproduction."],
            "summary": summary, "cases": rows}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-directory", type=Path, required=True)
    args = parser.parse_args()
    with patch("socket.socket", side_effect=AssertionError("Network disabled in offline probe")):
        config = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / config["chunks"]).resolve())
        suite = EvalSuite.load(args.suite_directory / "cases.json")
        spec = json.loads((args.suite_directory / "reference-drafts.json").read_text(encoding="utf-8"))
        packet = compile_drafts(spec, suite, corpus)
        report = run(suite, packet, corpus, ScopedRetriever(corpus))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "experiments/runs" / f"retrieval-upgrade-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 检索工具离线对照", "", "不是模型准确率或自主规划评测；导航片段由开发者选择。", "",
             "| 题目 | 原始Top-8 | 分文档Top-8 | 分文档+额外上下文 | 参考片段数 |",
             "| --- | --- | --- | --- | --- |"]
    for row in report["cases"]:
        lines.append(f"| {row['case_id']} | {row['global']['anchor_matches']} | "
                     f"{row['balanced']['anchor_matches']} | {row['balanced_plus_context']['anchor_matches']} | "
                     f"{row['anchor_count']} |")
    lines.extend(["", "分文档检索使用更多本地索引查询；上下文列还增加工具调用和证据量，不是同预算对照。", "",
                  "完整请求、版本、证据、耗时及退步题目见 report.json。", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), **report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
