"""One known insurance development question, two citation modes, no retry or follow-up."""

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
from run_live_selection_comparison import FrozenCandidates

from financial_agent.contracts import Budget, EvidenceRef, SearchRequest, TaskRequest
from financial_agent.core_answer import CoreAnswerEngine, render_core_answer
from financial_agent.core_evidence import prepare_core_evidence
from financial_agent.corpus import Corpus
from financial_agent.durability import implementation_hash
from financial_agent.evaluation import EvalSuite
from financial_agent.model import ModelConfig
from financial_agent.retrieval import ScopedRetriever
from financial_agent.spending import BatchSpendingGuard

MODES = ("verbatim", "span_id")
RUN_CAP_UNITS = 2_500_000
BATCH_CAP_UNITS = 5_000_000
CASE_PATH = "data/local/evaluation-risk-v3-20260914T045248Z-1fb80cc6/cases.json"
CASE_ID = "risk-v3-005"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class FixedWindowEngine(CoreAnswerEngine):
    """Stop before a model call if evidence differs from the predeclared frozen pack."""

    def __init__(self, *args, expected_pack, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_pack = deepcopy(expected_pack)

    def _answer_pack(self, result, pack):
        if pack != self.expected_pack:
            raise ValueError("Evidence pack differs between citation conditions")
        for window in pack["windows"]:
            self.corpus.read(EvidenceRef(**window["ref"]), frozenset(self.task.document_version_ids))
        return super()._answer_pack(result, pack)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print("DRY PLAN: risk-v3-005; frozen coverage evidence; verbatim then span_id once each; "
              "qwen3.7-flash; <=2 attempts/run, <=4 total; <=0.25 CNY/run, <=0.5 CNY batch; "
              "no retry/follow-up/resume. No API call or .env read.")
        return
    os.umask(0o077)
    root = Path(__file__).resolve().parents[1]
    suite = EvalSuite.load(root / CASE_PATH)
    case = next(c for c in suite.cases if c.case_id == CASE_ID)
    if case.partition != "candidate":
        raise ValueError("Only the declared development candidate is allowed")
    datasets = json.loads((root / "configs/datasets.example.json").read_text())
    corpus = Corpus.from_jsonl((root / datasets["chunks"]).resolve())
    task = case.task(Budget(1, 2, 100_000))
    hits = ScopedRetriever(corpus, tokenizer_mode="mixed").search(task, SearchRequest(task.query, top_k=40))
    frozen = FrozenCandidates(task, hits)
    pack = prepare_core_evidence(corpus, frozen, task, selection_strategy="coverage")
    if not pack["windows"]:
        raise ValueError("No evidence; stop before credential loading")
    env = dotenv_values(root / ".env")
    config = ModelConfig(model=env.get("QWEN_MODEL", ""), api_key=env.get("DASHSCOPE_API_KEY") or "",
                         base_url=env.get("QWEN_BASE_URL", ""), enabled=True, timeout_seconds=45,
                         max_attempts_per_call=1, max_total_attempts=2)
    if not config.api_key or config.model != "qwen3.7-flash":
        raise ValueError("Current key and qwen3.7-flash required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "experiments/runs" / f"live-citation-comparison-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={output}", flush=True)
    batch = BatchSpendingGuard(output / "batch-spending.jsonl", cap_units=BATCH_CAP_UNITS)
    source_hash = implementation_hash()
    manifest = {"model": config.model, "endpoint": config.base_url, "implementation_hash": source_hash,
                "script_hashes": {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest()
                                  for name in (Path(__file__).name, "run_core_comparison.py",
                                               "run_live_selection_comparison.py")},
                "case": asdict(case), "suite_hash": suite.fingerprint,
                "evidence_pack_hash": fingerprint(pack), "selection_strategy": "coverage",
                "order": list(MODES), "planned_runs": 2, "tasks": {}, "held_out": False,
                "answer_accuracy": None, "run_cap_cny": 0.25, "batch_cap_cny": 0.5,
                "max_attempts_per_run": 2, "retries": 0, "follow_up": False,
                "limitations": ["Known failure development question, not independent gold or B35 evaluation",
                                "Same windows/model/review, different citation prompt and added ID catalog",
                                "ID mode adds input tokens; no equality of actual token use assumed",
                                "Fixed order, one run per condition; no statistical or general quality claim",
                                "Same-model semantic review is not independent gold; provider billing authoritative"]}
    save(output / "manifest.json", manifest)
    save(output / "frozen-evidence-pack.json", pack)
    save(output / "candidates.json", [asdict(h) for h in hits])
    summary = {"planned_runs": 2, "executed_runs": 0, "answer_accuracy": None, "held_out": False, "runs": {}}
    for mode in MODES:
        if implementation_hash() != source_hash:
            raise RuntimeError("Implementation changed; stop before next model call")
        task = TaskRequest(case.query, case.document_version_ids, task_id=f"{CASE_ID}-{mode}",
                           budget=Budget(1, 2, 100_000))
        manifest["tasks"][mode] = asdict(task)
        save(output / "manifest.json", manifest)
        guard = BatchSpendingGuard(output / f"{mode}-spending.jsonl", cap_units=RUN_CAP_UNITS)

        def record(event, name=mode, run_guard=guard):
            run_guard.record(name, event)
            batch.record(name, event)
            print(f"{name}: attempt={event['attempt']} status={event['status']}", flush=True)

        client = ResponseJournalClient(config, output / f"{mode}-responses.jsonl", record=record)
        engine = FixedWindowEngine(corpus, frozen, task, client, expected_pack=pack,
                                   selection_strategy="coverage", citation_mode=mode,
                                   execution_label="real_qwen_fixed_window_citation_comparison")
        guard.attach(engine)
        batch.attach(engine)
        print(f"START={mode}", flush=True)
        result = engine.run()
        save(output / f"{mode}.json", result)
        (output / f"{mode}.md").write_text(render_core_answer(result), encoding="utf-8")
        summary["runs"][mode] = {"status": result["status"], "answer": result["answer"],
                                 "usage": result["usage"], "spending": guard.summary()}
        summary.update(executed_runs=len(summary["runs"]), spending=batch.summary())
        save(output / "summary.json", summary)
        print(f"END={mode} status={result['status']}", flush=True)
        if any(e["status"] != "ok" or e["usage_status"] != "reported" for e in client.attempts):
            print("Stopped after provider/transport/unknown-use event. No rerun.", flush=True)
            return
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
