"""Offline source/protocol audit of a live citation comparison; never repairs answers."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from audit_live_selection import quote_diagnostic
from run_live_citation_comparison import fingerprint

from financial_agent.citation_spans import resolve_span_candidate
from financial_agent.contracts import Budget, EvidenceRef, OutputKind, TaskRequest
from financial_agent.core_answer import parse_object, validate_candidate, validate_review
from financial_agent.corpus import Corpus
from financial_agent.durability import implementation_hash


def audit_run(result, stored_task, pack, corpus, responses):
    task = TaskRequest(**{**stored_task, "document_version_ids": tuple(stored_task["document_version_ids"]),
                          "output_kind": OutputKind(stored_task["output_kind"]),
                          "budget": Budget(**stored_task["budget"])})
    if result["evidence_pack"] != pack:
        raise ValueError("Frozen evidence mismatch")
    windows = {w["window_number"]: w for w in pack["windows"]}
    sources = {n: corpus.read(EvidenceRef(**w["ref"]), frozenset(task.document_version_ids))
               for n, w in windows.items()}
    answers = [r for r in responses if r["stage"] == "answer"]
    reviews = [r for r in responses if r["stage"] == "review"]
    if len(answers) != 1 or len(reviews) > 1:
        raise ValueError("Unexpected response stages")
    raw = parse_object(answers[0]["content"])
    candidate = raw
    row = {"status": result["status"], "source_windows_checked": len(sources),
           "quote_diagnostics": [], "published_source_spans_checked": 0,
           "review_was_called": bool(reviews), "semantic_gold": False}
    if result["citation_mode"] == "span_id":
        if raw != result["span_candidate"]:
            raise ValueError("Raw ID candidate differs from journal")
        candidate = resolve_span_candidate(raw, result["citation_catalog"], pack, corpus, task)
    try:
        validate_candidate(candidate, pack, corpus, task)
        row["candidate_protocol_valid"] = True
    except (ValueError, TypeError, KeyError, PermissionError):
        row["candidate_protocol_valid"] = False
        if result["status"] != "validation_failed" or reviews or result["answer"] or result["claims"]:
            raise ValueError("Invalid candidate must not be reviewed or published") from None
    for claim in candidate.get("claims", []):
        for citation in claim["citations"]:
            number, quote = citation["window_number"], citation["quote"]
            row["quote_diagnostics"].append({"claim_id": claim["claim_id"], "window_number": number,
                "visible": quote_diagnostic(quote, windows[number]["excerpt_text"]),
                "source": quote_diagnostic(quote, sources[number])})
    if row["candidate_protocol_valid"] and candidate != result["candidate"]:
        raise ValueError("Materialized candidate differs from stored result")
    if reviews:
        review = parse_object(reviews[0]["content"])
        if review != result["review"]:
            raise ValueError("Review journal differs from stored result")
        row["model_review_accepted"] = validate_review(review, candidate)
    if result["status"] == "answered":
        if not row["candidate_protocol_valid"] or not row.get("model_review_accepted"):
            raise ValueError("Answer lacks protocol or model-review acceptance")
        if result["answer"] != "\n".join(c["text"] for c in candidate["claims"]):
            raise ValueError("Published answer differs from reviewed candidate")
        if len(result["claims"]) != len(candidate["claims"]):
            raise ValueError("Published claim count mismatch")
        for published, approved in zip(result["claims"], candidate["claims"], strict=True):
            if {k: published[k] for k in ("claim_id", "text", "kind")} != {
                    k: approved[k] for k in ("claim_id", "text", "kind")}:
                raise ValueError("Published claim mismatch")
            for citation, original in zip(published["citations"], approved["citations"], strict=True):
                n = original["window_number"]
                if citation["ref"] != windows[n]["ref"] or any(citation[k] != original[k] for k in original):
                    raise ValueError("Published citation differs from approved source")
                if result["citation_mode"] == "span_id":
                    span = next(s for s in result["citation_catalog"]["spans"] if s["span_id"] == citation["span_id"])
                    bounds = citation["source_span"]
                    if (bounds != {"start": span["source_start"], "end": span["source_end"],
                                   "offsets": result["citation_catalog"]["offsets"]}
                            or citation["catalog_id"] != result["citation_catalog"]["catalog_id"]
                            or citation["quote"] != span["quote"] or n != span["window_number"]):
                        raise ValueError("Published span binding mismatch")
                    row["published_source_spans_checked"] += 1
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    output = args.run_directory.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["implementation_hash"] != implementation_hash():
        raise ValueError("Audit implementation differs from live run")
    pack = json.loads((output / "frozen-evidence-pack.json").read_text())
    if manifest["evidence_pack_hash"] != fingerprint(pack):
        raise ValueError("Frozen pack changed")
    root = Path(__file__).resolve().parents[1]
    audit = {"execution_label": "offline_programmatic_protocol_and_source_audit", "real_api_calls": 0,
             "human_gold": False, "answer_accuracy": None, "repairs_applied": False,
             "evidence_pack_hash": fingerprint(pack), "runs": {}}
    with patch("socket.socket", side_effect=AssertionError("No network in audit")):
        datasets = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / datasets["chunks"]).resolve())
        for mode in manifest["order"]:
            if mode not in ("verbatim", "span_id"):
                raise ValueError("Unknown mode")
            original = (output / f"{mode}.json").read_bytes()
            journal = (output / f"{mode}-responses.jsonl").read_bytes()
            row = audit_run(json.loads(original), manifest["tasks"][mode], pack, corpus,
                            [json.loads(line) for line in journal.decode().splitlines()])
            row.update(result_sha256=hashlib.sha256(original).hexdigest(),
                       journal_sha256=hashlib.sha256(journal).hexdigest())
            audit["runs"][mode] = row
    with (output / "offline-audit.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
