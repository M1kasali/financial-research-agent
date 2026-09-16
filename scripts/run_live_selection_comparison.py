"""Four predeclared live development runs. No follow-up, retries, repair or resume."""

import argparse
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values
from run_core_comparison import ResponseJournalClient, save

from financial_agent.contracts import Budget, SearchRequest, TaskRequest
from financial_agent.core_answer import CoreAnswerEngine, render_core_answer
from financial_agent.corpus import Corpus
from financial_agent.durability import implementation_hash
from financial_agent.evaluation import EvalSuite
from financial_agent.model import ModelConfig
from financial_agent.retrieval import ScopedRetriever
from financial_agent.spending import BatchSpendingGuard

CASE_SOURCES = (
    ("data/local/evaluation-risk-v3-20260914T045248Z-1fb80cc6/cases.json", "risk-v3-005"),
    ("data/local/evaluation-v2-20260913T175250Z-415e313a/cases.json", "dev-v2-010"),
)
STRATEGIES = ("legacy", "coverage")
RUN_CAP_UNITS = 2_500_000


class FrozenCandidates:
    """Both conditions receive the same ordered public-query retrieval; no draft inputs."""

    def __init__(self, task, hits):
        self.query, self.versions, self.hits = task.query, task.document_version_ids, deepcopy(hits)

    def search(self, task, request):
        if (task.query != self.query or task.document_version_ids != self.versions or request.query != self.query
                or request.top_k != 40 or request.document_version_ids not in (None, self.versions)):
            raise ValueError("Frozen candidate query/scope mismatch")
        return deepcopy(self.hits)


def load_cases(root):
    cases = []
    for relative, case_id in CASE_SOURCES:
        suite = EvalSuite.load(root / relative)
        case = next(c for c in suite.cases if c.case_id == case_id)
        if case.partition != "candidate":
            raise ValueError("Only predeclared development cases allowed")
        cases.append((case, suite.fingerprint))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print("DRY PLAN: insurance comparison risk-v3-005 and research summary dev-v2-010; "
              "legacy/coverage each once, same cached candidates; qwen3.7-flash; "
              "1 prepare + <=2 model attempts/run; <=0.25 CNY/run, <=1 CNY batch; "
              "no follow-up/retry. No API call or .env read.")
        return
    os.umask(0o077)
    root = Path(__file__).resolve().parents[1]
    cases = load_cases(root)
    env = dotenv_values(root / ".env")
    config = ModelConfig(model=env.get("QWEN_MODEL", ""), api_key=env.get("DASHSCOPE_API_KEY") or "",
                         base_url=env.get("QWEN_BASE_URL", ""), enabled=True, timeout_seconds=45,
                         max_attempts_per_call=1, max_total_attempts=2)
    if not config.api_key or config.model != "qwen3.7-flash":
        raise ValueError("Current key and qwen3.7-flash required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "experiments/runs" / f"live-selection-comparison-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={output}", flush=True)
    batch = BatchSpendingGuard(output / "batch-spending.jsonl")
    source_hash = implementation_hash()
    manifest = {"model": config.model, "endpoint": config.base_url, "implementation_hash": source_hash,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "journal_helper_sha256": hashlib.sha256((root / "scripts/run_core_comparison.py").read_bytes()).hexdigest(),
                "held_out": False, "answer_accuracy": None, "run_cap_cny": 0.25, "batch_cap_cny": 1,
                "engine": "CoreAnswerEngine", "follow_up": False, "retries": 0, "planned_runs": 4,
                "limits": {"candidate_k": 40, "max_chunks": 8, "max_chars": 6500, "max_documents": 4,
                           "max_tool_calls": 1, "max_model_attempts": 2, "max_estimated_tokens": 100000},
                "cases": [{"case": asdict(c), "suite_hash": h} for c, h in cases],
                "order": [f"{c.case_id}-{mode}" for c, _ in cases for mode in STRATEGIES],
                "limitations": ["Known development cases selected for improvement and regression, not representative",
                                "Only evidence selection changes; single-pass plus same-model review, not full Agent loop",
                                "Reference packets and review questions are not loaded or sent",
                                "Fixed order, one run per condition; no causal or statistical superiority claim",
                                "No human gold; provider billing authoritative"], "tasks": {}}
    save(output / "manifest.json", manifest)
    datasets = json.loads((root / "configs/datasets.example.json").read_text())
    corpus = Corpus.from_jsonl((root / datasets["chunks"]).resolve())
    retriever = ScopedRetriever(corpus, tokenizer_mode="mixed")
    summary = {"runs": {}, "planned_runs": 4, "executed_runs": 0, "answer_accuracy": None, "held_out": False}
    for case, _ in cases:
        public_task = case.task(Budget(1, 2, 100_000))
        hits = retriever.search(public_task, SearchRequest(public_task.query, top_k=40))
        frozen = FrozenCandidates(public_task, hits)
        save(output / f"{case.case_id}-candidates.json", [asdict(h) for h in hits])
        for mode in STRATEGIES:
            if implementation_hash() != source_hash:
                raise RuntimeError("Implementation changed; stop without another model call")
            run_id = f"{case.case_id}-{mode}"
            task = TaskRequest(case.query, case.document_version_ids, task_id=run_id, budget=Budget(1, 2, 100_000))
            manifest["tasks"][run_id] = asdict(task)
            save(output / "manifest.json", manifest)
            guard = BatchSpendingGuard(output / f"{run_id}-spending.jsonl", cap_units=RUN_CAP_UNITS)

            def record(event, name=run_id, run_guard=guard):
                run_guard.record(name, event)
                batch.record(name, event)
                print(f"{name}: attempt={event['attempt']} status={event['status']}", flush=True)

            client = ResponseJournalClient(config, output / f"{run_id}-responses.jsonl", record=record)
            engine = CoreAnswerEngine(corpus, frozen, task, client, selection_strategy=mode,
                                      execution_label="real_qwen_single_pass_selection_ablation")
            guard.attach(engine)
            batch.attach(engine)
            print(f"START={run_id}", flush=True)
            result = engine.run()
            save(output / f"{run_id}.json", result)
            (output / f"{run_id}.md").write_text(render_core_answer(result), encoding="utf-8")
            summary["runs"][run_id] = {"status": result["status"], "answer": result["answer"],
                                       "usage": result["usage"], "spending": guard.summary()}
            summary.update(executed_runs=len(summary["runs"]), spending=batch.summary())
            save(output / "summary.json", summary)
            print(f"END={run_id} status={result['status']}", flush=True)
            if any(e["status"] != "ok" or e["usage_status"] != "reported" for e in client.attempts):
                print("Stopped after provider/transport/unknown-use event. No rerun.", flush=True)
                return
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
