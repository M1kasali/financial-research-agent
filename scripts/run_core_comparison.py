"""Four predeclared development runs; explicit opt-in, no reruns or resume."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.core_answer import CoreAnswerEngine, render_core_answer
from financial_agent.core_research import CoreResearchEngine
from financial_agent.corpus import Corpus
from financial_agent.durability import implementation_hash
from financial_agent.financial_workflow import FinancialQuestion
from financial_agent.model import ModelConfig, RecordedModelClient
from financial_agent.retrieval import ScopedRetriever
from financial_agent.spending import BatchSpendingGuard

CASES = (("byd_growth", "consolidated", "growth_rate", "营业收入"),
         ("byd_parent_profit", "parent", "extract", "净利润"))
CONDITIONS = (("single_pass", CoreAnswerEngine), ("bounded_loop", CoreResearchEngine))
RUN_CAP_UNITS = 2_500_000


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ResponseJournalClient(RecordedModelClient):
    """Keep even invalid JSON before engine validation; never log request credentials."""

    def __init__(self, config, journal, **kwargs):
        super().__init__(config, **kwargs)
        self.journal = journal
        with journal.open("x", encoding="utf-8"):
            pass
        journal.chmod(0o600)

    def chat(self, messages, *, max_tokens=800):
        response = super().chat(messages, max_tokens=max_tokens)
        stage = json.loads(messages[-1]["content"]).get("stage", "unknown")
        entry = {"stage": stage, "attempt": self.total_attempts, **response}
        # Defense in depth if a provider unexpectedly echoes a credential.
        encoded = json.dumps(entry, ensure_ascii=False)
        if self.config.api_key:
            encoded = encoded.replace(self.config.api_key, "[REDACTED]")
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print("DRY PLAN: 2 BYD development questions x single-pass/bounded-loop; "
              "qwen3.7-flash; <=0.25 CNY reservation/run, <=1 CNY batch; "
              "same 16-tool/8-attempt/200000-token-proxy caps. No API call or .env read.")
        return
    os.umask(0o077)
    root = Path(__file__).resolve().parents[1]
    env = dotenv_values(root / ".env")
    config = ModelConfig(model=env.get("QWEN_MODEL", ""), api_key=env.get("DASHSCOPE_API_KEY") or "",
                         base_url=env.get("QWEN_BASE_URL", ""), enabled=True,
                         timeout_seconds=45, max_attempts_per_call=1, max_total_attempts=8)
    if not config.api_key or config.model != "qwen3.7-flash":
        raise ValueError("Current key and qwen3.7-flash required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "experiments/runs" / f"core-comparison-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={output}", flush=True)
    batch_guard = BatchSpendingGuard(output / "batch-spending.jsonl")
    manifest = {"model": config.model, "endpoint": config.base_url,
                "implementation_hash": implementation_hash(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "held_out": False, "answer_accuracy": None, "planned_runs": 4,
                "run_cap_cny": 0.25, "batch_cap_cny": 1, "retries": 0,
                "order": [f"{c[0]}-{mode}" for c in CASES for mode, _ in CONDITIONS],
                "limitations": ["Two known development questions, not gold evaluation or B35 replay",
                                "Capability ablation, not a pure planning-style comparison",
                                "Initial answer prompt forbids new arithmetic in both conditions",
                                "Same-model review is not independent semantic verification",
                                "Fixed order, one run per condition, no statistical inference"], "tasks": {}}
    save(output / "manifest.json", manifest)
    datasets = json.loads((root / "configs/datasets.example.json").read_text())
    corpus = Corpus.from_jsonl((root / datasets["chunks"]).resolve())
    retriever = ScopedRetriever(corpus, tokenizer_mode="mixed")
    version = corpus.documents["annual_byd_2025_report"].version_id
    summary = {"runs": {}, "held_out": False, "answer_accuracy": None, "planned_runs": 4}
    for case_id, scope, operation, metric in CASES:
        question = FinancialQuestion(version, 2025, scope, operation, metric)
        for mode, engine_cls in CONDITIONS:
            run_id = f"{case_id}-{mode}"
            task = TaskRequest(question.query, (version,), task_id=run_id, budget=Budget(16, 8, 200_000))
            manifest["tasks"][run_id] = asdict(task)
            save(output / "manifest.json", manifest)
            run_guard = BatchSpendingGuard(output / f"{run_id}-spending.jsonl", cap_units=RUN_CAP_UNITS)

            def record(event, name=run_id, guard=run_guard):
                guard.record(name, event)
                batch_guard.record(name, event)
                print(f"{name}: attempt={event['attempt']} status={event['status']}", flush=True)

            client = ResponseJournalClient(config, output / f"{run_id}-responses.jsonl", record=record)
            engine = engine_cls(corpus, retriever, task, client, execution_label="real_qwen_core_comparison")
            run_guard.attach(engine)
            batch_guard.attach(engine)
            print(f"START={run_id}", flush=True)
            result = engine.run()
            save(output / f"{run_id}.json", result)
            (output / f"{run_id}.md").write_text(render_core_answer(result), encoding="utf-8")
            summary["runs"][run_id] = {"status": result["status"], "answer": result["answer"],
                                       "gaps": result["gaps"], "usage": result["usage"],
                                       "spending": run_guard.summary()}
            summary.update(executed_runs=len(summary["runs"]), spending=batch_guard.summary())
            save(output / "summary.json", summary)
            print(f"END={run_id} status={result['status']}", flush=True)
            if any(e["status"] != "ok" or e["usage_status"] != "reported" for e in client.attempts):
                print("Batch stopped after provider/transport/unknown-usage event. No rerun.", flush=True)
                return
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
