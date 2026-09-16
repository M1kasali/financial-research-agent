"""Generate a separate review packet; never write labels.pending.json."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalSuite
from financial_agent.reference_drafts import compile_drafts, render_packet


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=root / "docs/reference-drafts-v1.json")
    args = parser.parse_args()
    config = json.loads((root / "configs/datasets.example.json").read_text())
    corpus = Corpus.from_jsonl((root / config["chunks"]).resolve())
    suite = EvalSuite.load(args.suite)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    packet = compile_drafts(spec, suite, corpus)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "experiments/runs" / f"reference-drafts-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "review_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "review_packet.md").write_text(render_packet(packet), encoding="utf-8")
    summary = {"output": str(output), "draft_count": len(packet["drafts"]),
               "answerable_drafts": sum(d["coverage"] == "answerable_draft" for d in packet["drafts"]),
               "bound_references": sum(len(d["evidence"]) for d in packet["drafts"]),
               "calculations": sum(len(d.get("calculations", [])) for d in packet["drafts"]),
               "approved_gold_count": 0, "model_calls": 0, "semantic_review_complete": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
