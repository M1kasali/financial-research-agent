"""Score saved predictions without any model call. Never reads credentials."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalSuite, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "labels", "predictions", "chunks"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    suite = EvalSuite.load(args.suite)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    run = json.loads(args.predictions.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else None
    report = evaluate(suite, labels, run, Corpus.from_jsonl(args.chunks), reviews=reviews)
    root = Path(__file__).resolve().parents[1]
    output = (
        root
        / "experiments/runs"
        / ("score-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"Report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
