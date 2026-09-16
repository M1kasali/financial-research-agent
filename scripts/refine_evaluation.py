"""Generate a new candidate suite and pending reference packet, retaining v1."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalSuite
from financial_agent.evaluation_refinement import refine_candidates
from financial_agent.reference_drafts import render_packet


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-suite", type=Path, required=True)
    parser.add_argument("--parent-spec", type=Path, default=root / "docs/reference-drafts-v1.json")
    parser.add_argument("--recipe", type=Path, default=root / "docs/evaluation-refinement-v2.json")
    args = parser.parse_args()
    config = json.loads((root / "configs/datasets.example.json").read_text())
    inputs = [args.parent_suite, args.parent_spec, args.recipe,
              args.parent_suite.with_name("labels.pending.json")]
    before = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    corpus = Corpus.from_jsonl((root / config["chunks"]).resolve())
    suite, spec, packet, labels, manifest = refine_candidates(
        EvalSuite.load(args.parent_suite), json.loads(args.parent_spec.read_text(encoding="utf-8")),
        json.loads(args.recipe.read_text(encoding="utf-8")), corpus)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "data/local" / f"evaluation-v2-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    after = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    if before != after:
        raise RuntimeError("Input files changed during refinement")
    manifest["input_file_hashes"] = before
    for name, value in {"cases.json": suite.export(), "reference-drafts.json": spec,
                        "review_packet.json": packet, "labels.pending.json": labels,
                        "manifest.json": manifest}.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "review_packet.md").write_text(render_packet(packet), encoding="utf-8")
    print(json.dumps({"output": str(output), "suite_hash": suite.fingerprint,
                      "cases": len(suite.cases),
                      "changed_queries": sum(x["query_changed"] for x in manifest["lineage"]),
                      "narrowed_scopes": sum(x["scope_changed"] for x in manifest["lineage"]),
                      "drafts_needing_scope_review": sum(
                          x["coverage"] == "partial_needs_scope_review" for x in packet["drafts"]),
                      "approved_gold_count": 0, "model_calls": 0, "parent_inputs_unchanged": True},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
