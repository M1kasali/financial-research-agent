"""Explicitly approved two-case development pilot. No automatic reruns/resume."""

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.financial_workflow import FinancialQuestion, FinancialStatementWorkflow
from financial_agent.model import ModelConfig, RecordedModelClient
from financial_agent.retrieval import ScopedRetriever
from financial_agent.runtime import ResearchRuntime
from financial_agent.spending import BatchSpendingGuard
from financial_agent.storage import SQLiteTaskStore


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("DRY PLAN: Qwen3.7 Flash, 2 BYD development questions, <=16 attempts each, "
              "<=1 CNY batch reservation; use --execute only after approval. No API call.")
        return
    os.umask(0o077)
    root = Path(__file__).resolve().parents[1]
    env = dotenv_values(root / ".env")
    config = ModelConfig(model=env.get("QWEN_MODEL", ""), api_key=env.get("DASHSCOPE_API_KEY") or "",
                         base_url=env.get("QWEN_BASE_URL", ""), enabled=True,
                         timeout_seconds=45, max_total_attempts=16)
    if not config.api_key or config.model != "qwen3.7-flash":
        raise ValueError("Explicit current key and qwen3.7-flash configuration are required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "experiments/runs" / f"live-pilot-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={output}", flush=True)
    guard = BatchSpendingGuard(output / "spending.jsonl")
    manifest = {"model": config.model, "endpoint": config.base_url, "held_out": False,
                "answer_accuracy": None, "max_attempts_per_task": 16, "batch_cap_cny": 1,
                "price_source": guard.PRICE_SOURCE, "price_checked": "2026-09-14",
                "execution": "real_model_json_action_policy",
                "limitations": ["Known development document, no gold accuracy evaluation",
                                "Fixed workflow control is not the original B35 solver",
                                "Model verification is not independent ground truth"], "cases": {}}
    save(output / "manifest.json", manifest)
    datasets = json.loads((root / "configs/datasets.example.json").read_text())
    corpus = Corpus.from_jsonl((root / datasets["chunks"]).resolve())
    retriever = ScopedRetriever(corpus, tokenizer_mode="mixed")
    version = corpus.documents["annual_byd_2025_report"].version_id
    results = {}
    for case_id, scope, operation, metric in [
        ("byd_growth", "consolidated", "growth_rate", "营业收入"),
        ("byd_parent_profit", "parent", "extract", "净利润"),
    ]:
        question = FinancialQuestion(version, 2025, scope, operation, metric)
        task = TaskRequest(question.query, (version,), task_id=case_id,
                           budget=Budget(max_tool_calls=12, max_model_attempts=16,
                                         max_estimated_tokens=300_000))
        manifest["cases"][case_id] = asdict(task)
        save(output / "manifest.json", manifest)
        # Control output is saved for AFTER-run comparison, never given to the policy.
        control = FinancialStatementWorkflow(corpus, retriever, question).run()
        save(output / f"{case_id}-fixed.json", control)

        def record(event, name=case_id):
            guard.record(name, event)
            print(f"{name}: attempt={event['attempt']} status={event['status']} "
                  f"usage={event['usage_status']}", flush=True)

        client = RecordedModelClient(config, record=record)
        runtime = ResearchRuntime(corpus, task, client, retriever=retriever,
                                  execution_label="real_qwen_development_pilot",
                                  store=SQLiteTaskStore(output / f"{case_id}.sqlite3"))
        guard.attach(runtime)
        print(f"START={case_id}", flush=True)
        result = runtime.run()
        save(output / f"{case_id}-dynamic.json", result)
        results[case_id] = {"status": result["status"], "reason": result["reason"],
                            "budget": result["budget"], "control_status": control["status"]}
        save(output / "summary.json", {"cases": results, "spending": guard.summary()})
        print(f"END={case_id} status={result['status']} reason={result['reason']}", flush=True)
        # Connection/auth/provider/unknown-use failures stop the batch; no blind continuation.
        if any(e["status"] != "ok" for e in client.attempts):
            print("Batch stopped after provider/transport error; no further tasks sent.", flush=True)
            break
    summary = {"cases": results, "spending": guard.summary(), "held_out": False,
               "answer_accuracy": None, "planned_cases": 2, "executed_cases": len(results)}
    save(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
