"""REAL development corpus/tools + SCRIPTED model routing; not real model quality."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.core_answer import render_core_answer
from financial_agent.core_research import CoreResearchEngine
from financial_agent.corpus import Corpus
from financial_agent.financial_workflow import FinancialQuestion
from financial_agent.retrieval import ScopedRetriever
from financial_agent.simulation import ScriptedSend, fake_client


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "experiments/runs" / (
        "core-research-real-tools-simulated-model-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={output}", flush=True)
    summary = {"execution": "real_development_corpus_and_tools_SCRIPTED_MODEL_not_quality_eval",
               "real_api_calls": 0, "answer_accuracy": None, "held_out": False, "cases": {}}
    with patch("socket.socket", side_effect=AssertionError("No network in scripted routing smoke")):
        config = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / config["chunks"]).resolve())
        retriever = ScopedRetriever(corpus, tokenizer_mode="mixed")
        version = corpus.documents["annual_byd_2025_report"].version_id
        for case_id, scope, operation, metric, denominator in [
            ("growth", "consolidated", "growth_rate", "营业收入", None),
            ("parent_profit", "parent", "extract", "净利润", None),
            ("cashflow_ratio", "consolidated", "ratio", "经营活动产生的现金流量净额", "净利润"),
        ]:
            question = FinancialQuestion(version, 2025, scope, operation, metric, denominator)
            task = TaskRequest(question.query, (version,), budget=Budget(16, 8, 200_000))
            send = ScriptedSend([
                {"decision": "needs_calculation", "claims": [], "gaps": ["模拟转交财务工具"], "option_judgments": []},
                {"kind": "financial", "spec": asdict(question)},
                {"request_matches_spec": True, "covers_request": True, "gaps": []},
            ])
            result = CoreResearchEngine(corpus, retriever, task, fake_client(send),
                                        execution_label=summary["execution"]).run()
            assert result["status"] == "answered", result["gaps"]
            assert result["usage"]["model_attempts"] == 3
            assert result["usage"]["tool_calls"] == result["financial_result"]["tool_calls"] + 1
            assert all(c["ref"]["version_id"] == version for c in result["claims"][0]["citations"])
            (output / f"{case_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            (output / f"{case_id}.md").write_text(render_core_answer(result))
            summary["cases"][case_id] = {"status": result["status"], "answer": result["answer"],
                                         "tool_calls": result["usage"]["tool_calls"],
                                         "simulated_model_attempts": result["usage"]["model_attempts"]}
            print(json.dumps({case_id: summary["cases"][case_id]}, ensure_ascii=False), flush=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
