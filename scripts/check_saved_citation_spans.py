"""Build and source-check NEW span catalogs on saved windows, without model calls or answer repair."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from financial_agent.citation_spans import build_citation_catalog
from financial_agent.contracts import Budget, EvidenceRef, OutputKind, TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.durability import implementation_hash


def check_result(result, stored_task, corpus):
    task = TaskRequest(**{**stored_task, "document_version_ids": tuple(stored_task["document_version_ids"]),
                          "output_kind": OutputKind(stored_task["output_kind"]),
                          "budget": Budget(**stored_task["budget"])})
    pack = result["evidence_pack"]
    catalog = build_citation_catalog(pack, corpus, task)
    windows = {w["window_number"]: w for w in pack["windows"]}
    for span in catalog["spans"]:
        window = windows[span["window_number"]]
        ref = EvidenceRef(**window["ref"])
        source = corpus.read(ref, frozenset(task.document_version_ids))
        expected = source[span["source_start"] - ref.start:span["source_end"] - ref.start]
        visible = window["excerpt_text"][span["excerpt_start"]:span["excerpt_end"]]
        if expected != span["quote"] or visible != span["quote"] or not 2 <= len(expected) <= 400:
            raise ValueError("Span/source/display binding failed")
    return {"original_status": result["status"], "windows": len(windows),
            "spans_checked": len(catalog["spans"]),
            "spans_with_line_breaks": sum("\n" in s["quote"] or "\r" in s["quote"] for s in catalog["spans"]),
            "catalog": catalog}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the old run directory")
    args = parser.parse_args()
    original_dir, output = args.run_directory.resolve(), args.output.resolve()
    if output == original_dir or original_dir in output.parents or output.exists():
        raise ValueError("Use a new output file outside the original experiment")
    manifest_bytes = (original_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    root = Path(__file__).resolve().parents[1]
    audit = {"execution_label": "offline_new_span_catalog_on_saved_windows", "real_api_calls": 0,
             "model_id_selection_tested": False, "semantic_review_performed": False,
             "answer_accuracy": None, "old_answers_repaired": False,
             "old_implementation_hash": manifest["implementation_hash"],
             "new_implementation_hash": implementation_hash(),
             "original_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "runs": {}}
    with patch("socket.socket", side_effect=AssertionError("Offline check forbids network")):
        datasets = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / datasets["chunks"]).resolve())
        for run_id in manifest["order"]:
            if Path(run_id).name != run_id or "\\" in run_id or run_id in {".", ".."}:
                raise ValueError("Invalid run identifier")
            path = original_dir / f"{run_id}.json"
            original = path.read_bytes()
            row = check_result(json.loads(original), manifest["tasks"][run_id], corpus)
            row["original_result_sha256"] = hashlib.sha256(original).hexdigest()
            if path.read_bytes() != original:
                raise ValueError("Original result changed")
            audit["runs"][run_id] = row
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    brief = {**audit, "runs": {k: {field: value for field, value in row.items() if field != "catalog"}
                              for k, row in audit["runs"].items()}}
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
