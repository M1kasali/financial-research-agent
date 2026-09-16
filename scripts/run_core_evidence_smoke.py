"""Real local corpus, network-disabled released-component smoke, not answer evaluation."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from financial_agent.contracts import EvidenceRef, TaskRequest
from financial_agent.core_evidence import prepare_core_evidence
from financial_agent.corpus import Corpus
from financial_agent.retrieval import ScopedRetriever


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "experiments/runs" / (
        "v45-core-evidence-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={output}", flush=True)
    cases = [
        ("growth", "核查2025年度合并营业收入相对2024年的变化，确认原值、单位和来源。", {}),
        ("parent_profit", "核查2025年度母公司（公司单体）净利润，确认币种、单位和来源。", {}),
        ("claim_coverage", "核查两类利润表的口径及其披露内容。",
         {"A": "合并利润表披露集团净利润。", "B": "公司利润表披露公司单体净利润。"}),
    ]
    summaries = {}
    with patch("socket.socket", side_effect=AssertionError("Network forbidden in source-component smoke")):
        config = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / config["chunks"]).resolve())
        retriever = ScopedRetriever(corpus, tokenizer_mode="mixed")
        version = corpus.documents["annual_byd_2025_report"].version_id
        for case_id, query, options in cases:
            task = TaskRequest(query, (version,))
            result = prepare_core_evidence(corpus, retriever, task, options=options)
            assert result["rendered_chars"] <= 6500 and result["model_calls"] == 0
            assert result["windows"] and not result["answer_generated"]
            for window in result["windows"]:
                corpus.read(EvidenceRef(**window["ref"]), frozenset({version}))
            (output / f"{case_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            summaries[case_id] = {"status": result["status"], "candidate_count": result["candidate_count"],
                                  "window_count": result["selected_window_count"],
                                  "rendered_chars": result["rendered_chars"],
                                  "source_scope_and_hashes_verified": True}
            print(json.dumps({case_id: summaries[case_id]}, ensure_ascii=False), flush=True)
    summary = {"cases": summaries, "network_disabled": True, "model_calls": 0,
               "answer_accuracy": None, "held_out": False,
               "note": "Known development document; evidence preparation only, no correctness claim."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
