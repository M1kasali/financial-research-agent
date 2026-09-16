"""Known-development-source end-to-end fixed-workflow smoke; no preselected pages or LLM."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from financial_agent.corpus import Corpus
from financial_agent.financial_workflow import (
    FinancialQuestion,
    FinancialStatementWorkflow,
    render_financial_report,
)
from financial_agent.retrieval import ScopedRetriever


def main():
    root = Path(__file__).resolve().parents[1]
    # These are public task parameters, not answer/evidence locators. No chunk IDs/pages.
    cases = [
        ("byd_growth", "annual_byd_2025_report", "consolidated", "growth_rate", "营业收入", None, 12),
        ("byd_parent_revenue", "annual_byd_2025_report", "parent", "extract", "营业收入", None, 12),
        ("byd_parent_profit", "annual_byd_2025_report", "parent", "extract", "净利润", None, 12),
        ("byd_cashflow_ratio", "annual_byd_2025_report", "consolidated", "ratio", "经营活动产生的现金流量净额", "净利润", 12),
        ("midea_layout_probe", "annual_midea_2025_report", "consolidated", "growth_rate", "营业收入", None, 12),
        ("budget_probe", "annual_byd_2025_report", "consolidated", "growth_rate", "营业收入", None, 2),
    ]
    results = {}
    with patch("socket.socket", side_effect=AssertionError("Network disabled in offline workflow smoke")):
        config = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / config["chunks"]).resolve())
        retriever = ScopedRetriever(corpus, tokenizer_mode="mixed")
        for case_id, doc_id, scope, operation, metric, denominator, budget in cases:
            question = FinancialQuestion(corpus.documents[doc_id].version_id, 2025, scope, operation, metric, denominator)
            runner = FinancialStatementWorkflow(corpus, retriever, question, max_tool_calls=budget)
            assert not runner.tools.evidence and not runner.tools.facts
            result = runner.run()
            results[case_id] = result
            print(f"{case_id}: {result['status']} / {result['reason_code']}, calls={result['tool_calls']}", flush=True)
        # Checks are applied AFTER execution and never passed into workflow selection.
        assert results["byd_growth"]["calculation"]["display_value"] == "3.46"
        assert results["byd_parent_revenue"]["selected_facts"][0]["raw_value"] == "1,525,075"
        assert results["byd_parent_profit"]["selected_facts"][0]["raw_value"] == "4,206,012"
        assert results["byd_cashflow_ratio"]["calculation"]["display_value"] == "1.75"
        assert results["budget_probe"]["reason_code"] == "tool_budget_exhausted"
        assert all(result["tool_calls"] <= result["max_tool_calls"] for result in results.values())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "experiments/runs" / f"financial-workflow-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    for name, result in results.items():
        (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / f"{name}.md").write_text(render_financial_report(result), encoding="utf-8")
    summary = {"execution_label": "fixed_workflow_known_development_smoke_not_model_agent",
               "input_includes_evidence_ids_or_pages": False, "network_disabled": True, "model_calls": 0,
               "status_counts": dict(Counter(r["status"] for r in results.values())), "cases": {
                   k: {"status": r["status"], "reason_code": r["reason_code"], "tool_calls": r["tool_calls"]}
                   for k, r in results.items()},
               "answer_accuracy": None, "held_out": False,
               "limitations": ["Structured user requirements, not unrestricted natural-language understanding.",
                               "Known source/layout development cases; status counts are not benchmark accuracy.",
                               "Not a real-model planning experiment or B35 reproduction."]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
