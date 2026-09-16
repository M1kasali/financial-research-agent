"""Freeze pending risk development cases and optionally diagnose visible evidence OFFLINE."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalSuite
from financial_agent.retrieval import ScopedRetriever
from financial_agent.risk_evaluation import build_risk_suite, diagnose_core_windows, render_risk_packet


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-directory", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=root / "docs/risk-evaluation-v3.json")
    parser.add_argument("--diagnose", action="store_true", help="Run local retrieval/compression only")
    args = parser.parse_args()
    inputs = [args.parent_directory / n for n in ("cases.json", "reference-drafts.json", "labels.pending.json")]
    inputs.append(args.recipe)
    before = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    os.umask(0o077)
    with patch("socket.socket", side_effect=AssertionError("Offline preparation forbids network")):
        datasets = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / datasets["chunks"]).resolve())
        suite, packet, labels = build_risk_suite(
            json.loads(args.recipe.read_text()), EvalSuite.load(inputs[0]), json.loads(inputs[1].read_text()), corpus)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = root / "data/local" / f"evaluation-risk-v3-{stamp}-{uuid4().hex[:8]}"
        output.mkdir(parents=True, exist_ok=False)
        print(f"OUTPUT={output}", flush=True)
        summary = {"suite_hash": suite.fingerprint, "cases": len(suite.cases), "partition": "candidate",
                   "approved_gold_count": 0, "real_api_calls": 0, "answer_accuracy": None,
                   "input_hashes": before, "diagnostic_completed": False}
        for name, value in {"cases.json": suite.export(), "review_packet.json": packet,
                            "labels.pending.json": labels, "manifest.json": summary}.items():
            (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        (output / "review_packet.md").write_text(render_risk_packet(suite, packet))
        if args.diagnose:
            print("Starting local retrieval and V45-source window diagnostic; no model.", flush=True)
            result = diagnose_core_windows(suite, packet, corpus, ScopedRetriever(corpus, tokenizer_mode="mixed"))
            (output / "core-window-diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            summary.update(diagnostic_completed=True, diagnostic=result["summary"])
        after = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        if before != after:
            raise RuntimeError("Parent inputs changed during preparation")
        summary["parent_inputs_unchanged"] = True
        (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
