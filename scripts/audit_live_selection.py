"""Offline audit of saved live selection results. Never repairs quotes or publishes answers."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from financial_agent.contracts import EvidenceRef, TaskRequest
from financial_agent.core_answer import parse_object, validate_candidate
from financial_agent.corpus import Corpus
from financial_agent.durability import implementation_hash


def quote_diagnostic(quote, text):
    # Diagnostic counterfactual ONLY. Do not alter source, offsets, digits, spaces or production validator.
    return {"verbatim_match": quote in text,
            "match_after_removing_source_line_breaks_only": quote in text.replace("\r", "").replace("\n", ""),
            "normalization_used_for_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    output = args.run_directory.resolve()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["implementation_hash"] != implementation_hash():
        raise ValueError("Audit validator differs from run implementation")
    audit = {"real_api_calls": 0, "reviewer_kind": "programmatic_source_and_protocol_audit",
             "human_gold": False, "answer_accuracy": None, "repairs_applied": False, "runs": {}}
    with patch("socket.socket", side_effect=AssertionError("Offline audit forbids network")):
        datasets = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / datasets["chunks"]).resolve())
        for run_id in manifest["order"]:
            if Path(run_id).name != run_id or "\\" in run_id or run_id in {".", ".."}:
                raise ValueError("Invalid run identifier")
            path = output / f"{run_id}.json"
            if not path.exists():
                audit["runs"][run_id] = {"not_executed": True}
                continue
            original = path.read_bytes()
            result = json.loads(original)
            stored_task = manifest["tasks"][run_id]
            task = TaskRequest(stored_task["query"], tuple(stored_task["document_version_ids"]))
            pack = result["evidence_pack"]
            sources = {}
            for window in pack["windows"]:
                sources[window["window_number"]] = corpus.read(
                    EvidenceRef(**window["ref"]), frozenset(task.document_version_ids))
            journal = output / f"{run_id}-responses.jsonl"
            responses = [json.loads(line) for line in journal.read_text().splitlines()]
            row = {"status": result["status"], "source_windows_revalidated": len(sources),
                   "result_sha256": hashlib.sha256(original).hexdigest(), "candidate_quotes": [],
                   "review_was_called": any(r["stage"] == "review" for r in responses)}
            for response in responses:
                if response["stage"] != "answer":
                    continue
                try:
                    candidate = parse_object(response["content"])
                    validate_candidate(candidate, pack, corpus, task)
                    row["candidate_protocol_valid"] = True
                except (ValueError, TypeError, KeyError, PermissionError) as exc:
                    row["candidate_protocol_valid"] = False
                    row["candidate_validation_error"] = str(exc)
                    candidate = parse_object(response["content"])
                row["model_decision"] = candidate.get("decision")
                windows = {w["window_number"]: w for w in pack["windows"]}
                for claim in candidate.get("claims", []):
                    for citation in claim.get("citations", []):
                        number, quote = citation.get("window_number"), citation.get("quote")
                        if type(number) is not int or number not in windows or not isinstance(quote, str):
                            row["candidate_quotes"].append({"invalid_citation_shape": True})
                            continue
                        row["candidate_quotes"].append({"claim_id": claim["claim_id"], "window_number": number,
                            "visible_window": quote_diagnostic(quote, windows[number]["excerpt_text"]),
                            "full_source": quote_diagnostic(quote, sources[number])})
            assert path.read_bytes() == original
            audit["runs"][run_id] = row
    with (output / "offline-audit.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
