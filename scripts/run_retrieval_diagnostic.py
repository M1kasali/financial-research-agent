"""Run public-query lexical retrieval; use draft anchors only after search for diagnostics."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalSuite
from financial_agent.reference_drafts import compile_drafts
from financial_agent.retrieval import ScopedRetriever
from financial_agent.retrieval_diagnostics import diagnose_retrieval, render_diagnostic


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-directory", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((root / "configs/datasets.example.json").read_text())
    corpus = Corpus.from_jsonl((root / config["chunks"]).resolve())
    suite = EvalSuite.load(args.suite_directory / "cases.json")
    spec = json.loads((args.suite_directory / "reference-drafts.json").read_text(encoding="utf-8"))
    packet = compile_drafts(spec, suite, corpus)
    report = diagnose_retrieval(suite, packet, ScopedRetriever(corpus))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "experiments/runs" / f"retrieval-diagnostic-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    (output / "diagnostic.md").write_text(render_diagnostic(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": report["summary"],
                      "model_calls": 0, "answer_accuracy": None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
