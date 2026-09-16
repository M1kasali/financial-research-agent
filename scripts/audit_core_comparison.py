"""Offline artifact/source integrity checks, not independent semantic or gold review."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from financial_agent.contracts import EvidenceRef
from financial_agent.corpus import Corpus
from financial_agent.durability import implementation_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.run_dir.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["implementation_hash"] == implementation_hash()
    assert manifest["runner_sha256"] == hashlib.sha256(
        (root / "scripts/run_core_comparison.py").read_bytes()).hexdigest()
    audit = {"real_api_calls": 0, "independent_gold_review": False, "cases": {}}
    with patch("socket.socket", side_effect=AssertionError("Offline audit forbids network")):
        datasets = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / datasets["chunks"]).resolve())
        for case in ("byd_growth", "byd_parent_profit"):
            baseline = json.loads((output / f"{case}-single_pass.json").read_text())
            loop = json.loads((output / f"{case}-bounded_loop.json").read_text())
            first = loop["round_history"][0] if loop["round_history"] else loop
            assert baseline["evidence_pack"]["windows"] == first["evidence_pack"]["windows"]
            task = manifest["tasks"][f"{case}-bounded_loop"]
            scope = frozenset(task["document_version_ids"])
            checked = 0
            for claim in loop["claims"]:
                for citation in claim["citations"]:
                    ref = EvidenceRef(**citation["ref"])
                    visible = corpus.read(ref, scope)
                    assert citation["quote"] in visible
                    start, end = citation["field_start"], citation["field_end"]
                    assert ref.start <= start < end <= ref.end
                    assert corpus.chunks[ref.chunk_id].text[start:end] == citation["quote"]
                    checked += 1
            assert loop["answer"] == "\n".join(c["text"] for c in loop["claims"])
            audit["cases"][case] = {"same_initial_windows": True,
                                     "checked_source_anchors": checked,
                                     "published_answer_from_audited_claims": True}
    with (output / "offline-audit.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
