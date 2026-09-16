"""Same-candidate, same-cap OFFLINE selection ablation on known development suites."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from financial_agent.contracts import Budget, SearchRequest
from financial_agent.core_evidence import pack_core_evidence
from financial_agent.corpus import Corpus
from financial_agent.durability import implementation_hash
from financial_agent.evaluation import EvalSuite
from financial_agent.reference_drafts import compile_drafts
from financial_agent.retrieval import ScopedRetriever
from financial_agent.risk_evaluation import validate_risk_packet


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def measure(pack, navigation):
    return {"visible_navigation_quotes": sum(any(w["ref"]["chunk_id"] == nav["reference"]["chunk_id"]
            and nav["quote"] in w["excerpt_text"] for w in pack["windows"]) for nav in navigation),
            "navigation_count": len(navigation), "visible_document_count": len({w["ref"]["version_id"] for w in pack["windows"]}),
            "rendered_chars": pack["rendered_chars"], "windows": len(pack["windows"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-directory", type=Path, required=True)
    parser.add_argument("--regression-directory", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "experiments/runs" / f"window-selection-comparison-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={output}", flush=True)
    inputs = [args.risk_directory / "cases.json", args.risk_directory / "review_packet.json",
              args.regression_directory / "cases.json", args.regression_directory / "reference-drafts.json"]
    before = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    result = {"execution": "offline_same_candidates_selection_ablation", "real_api_calls": 0,
              "answer_accuracy": None, "held_out": False, "input_hashes": before,
              "implementation_hash": implementation_hash(), "limits": {"candidate_k": 40, "max_chunks": 8,
                                                                          "max_chars": 6500, "max_documents": 4},
              "groups": {}, "limitations": ["Author navigation is not complete or unique gold evidence",
                                               "No model answers or independent semantic evaluation",
                                               "Both suites are known development material"]}
    with patch("socket.socket", side_effect=AssertionError("Offline comparison forbids network")):
        config = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / config["chunks"]).resolve())
        risk_suite = EvalSuite.load(inputs[0])
        risk_packet = json.loads(inputs[1].read_text())
        validate_risk_packet(risk_suite, risk_packet, corpus)
        reg_suite = EvalSuite.load(inputs[2])
        reg_packet = compile_drafts(json.loads(inputs[3].read_text()), reg_suite, corpus)
        suites = [("risk_v3", risk_suite, {r["case_id"]: r["navigation"] for r in risk_packet["reviews"]}),
                  ("regression_v2", reg_suite, {r["case_id"]: list(r["evidence"].values()) for r in reg_packet["drafts"]})]
        retriever = ScopedRetriever(corpus, tokenizer_mode="mixed")
        for name, suite, references in suites:
            rows = []
            for case in suite.cases:
                task = case.task(Budget())
                hits = retriever.search(task, SearchRequest(case.query, top_k=40))
                # One search, identical ordered candidate objects for BOTH selectors. No references passed.
                packs = {mode: pack_core_evidence(corpus, task, hits, selection_strategy=mode,
                                                 max_chars=6500, max_chunks=8, max_documents=4)
                         for mode in ("legacy", "coverage")}
                metrics = {mode: measure(pack, references[case.case_id]) for mode, pack in packs.items()}
                row = {"case_id": case.case_id, "requested_documents": len(case.document_version_ids),
                       "candidate_chunk_ids": [h.ref.chunk_id for h in hits], "metrics": metrics, "packs": packs}
                rows.append(row)
                print(json.dumps({"case_id": case.case_id, "metrics": metrics}, ensure_ascii=False), flush=True)
            summary = {}
            for mode in ("legacy", "coverage"):
                summary[mode] = {"visible_navigation_quotes": sum(r["metrics"][mode]["visible_navigation_quotes"] for r in rows),
                                 "navigation_count": sum(r["metrics"][mode]["navigation_count"] for r in rows),
                                 "cases_all_documents_visible": sum(r["metrics"][mode]["visible_document_count"] == r["requested_documents"]
                                                                    for r in rows)}
            deltas = [r["metrics"]["coverage"]["visible_navigation_quotes"] - r["metrics"]["legacy"]["visible_navigation_quotes"] for r in rows]
            summary.update(improved_navigation_cases=sum(d > 0 for d in deltas),
                           regressed_navigation_cases=sum(d < 0 for d in deltas),
                           unchanged_navigation_cases=sum(d == 0 for d in deltas))
            result["groups"][name] = {"suite_hash": suite.fingerprint, "summary": summary, "rows": rows}
            save(output / "comparison.json", result)
        assert before == {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        assert result["implementation_hash"] == implementation_hash()
        result["inputs_and_implementation_unchanged"] = True
        save(output / "comparison.json", result)
        brief = {k: v for k, v in result.items() if k != "groups"}
        brief["groups"] = {k: v["summary"] for k, v in result["groups"].items()}
        save(output / "summary.json", brief)
        print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
