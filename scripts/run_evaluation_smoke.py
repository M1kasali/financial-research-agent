"""Offline evaluation pipeline and optional real-corpus annotation pack, not an A/B quality claim."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from financial_agent.benchmark import run_dynamic, run_fixed_rag
from financial_agent.contracts import Budget, SearchRequest
from financial_agent.corpus import Corpus
from financial_agent.durability import implementation_hash
from financial_agent.evaluation import EvalCase, EvalSuite, compare_reports, evaluate
from financial_agent.retrieval import ScopedRetriever
from financial_agent.simulation import FakeResponse, fake_client, finance_corpus, finance_decision


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-suite", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (
        root
        / "experiments/runs"
        / ("evaluation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    output.mkdir(parents=True, exist_ok=False)
    with (
        patch("socket.socket.connect", side_effect=AssertionError("Offline experiment")),
        patch("socket.create_connection", side_effect=AssertionError("Offline experiment")),
    ):
        corpus = finance_corpus()
        suite = EvalSuite(
            "evaluation-protocol-fixture-v1",
            (
                EvalCase(
                    "fixture-growth",
                    "甲公司2025年度营业收入同比变化是多少？",
                    tuple(corpus.by_version),
                    "financial_reports",
                    "calculation",
                    "fixture",
                ),
            ),
        )
        labels = {
            "suite_hash": suite.fingerprint,
            "labels": {
                "fixture-growth": {
                    "status": "synthetic_fixture",
                    "provenance": "synthetic_protocol_test_not_benchmark",
                    "expected": {"kind": "number", "value": "20", "unit": "%", "tolerance": "0.01"},
                }
            },
        }
        retriever = ScopedRetriever(corpus, tokenizer_mode="char")
        budget = Budget(12, 24, 400_000)

        def fixed_response(payload):
            hits = json.loads(payload["messages"][-1]["content"])["evidence"]["hits"]
            return FakeResponse(
                {
                    "decision": "answered",
                    "answer": "甲公司合并营业收入同比增长20%。",
                    "structured": {"value": "20", "unit": "%"},
                    "evidence_ids": [hits[0]["ref"]["evidence_id"]],
                }
            )

        fixed = run_fixed_rag(suite.cases[0], corpus, retriever, fake_client(fixed_response), budget)
        dynamic, trace = run_dynamic(
            suite.cases[0],
            corpus,
            retriever,
            fake_client(lambda p: FakeResponse(finance_decision(json.loads(p["messages"][-1]["content"])))),
            budget,
            execution_label="simulated_model",
        )
        settings = {
            "model": "qwen-simulated",
            "tokenizer": "char",
            "budget": asdict(budget),
            "scope": "case_document_versions",
        }
        reports = []
        write(output / "fixture.cases.json", suite.export())
        write(output / "fixture.labels.json", labels)
        (output / "fixture.chunks.jsonl").write_text(
            "\n".join(
                json.dumps({**asdict(c), "metadata": corpus.source_metadata[c.chunk_id]}, ensure_ascii=False)
                for c in corpus.chunks.values()
            )
            + "\n",
            encoding="utf-8",
        )
        for name, pred in (("fixed_rag_v1", fixed), ("dynamic_agent_v1", dynamic)):
            run = {
                "suite_hash": suite.fingerprint,
                "system_id": name,
                "mode": "simulation",
                "settings": settings,
                "predictions": [pred],
            }
            report = evaluate(suite, labels, run, corpus)
            assert report["summary"]["exact_field_accuracy"] == 1
            assert report["summary"]["human_answer_accuracy"] is None
            write(output / f"{name}.predictions.json", run)
            write(output / f"{name}.report.json", report)
            reports.append(report)
        write(output / "dynamic.trace.json", trace)
        comparison = compare_reports(*reports)
        write(output / "comparison.json", comparison)
        candidate_summary = None
        if args.candidate_suite:
            suite = EvalSuite.load(args.candidate_suite)
            config = json.loads((root / "configs/datasets.example.json").read_text(encoding="utf-8"))
            corpus = Corpus.from_jsonl((root / config["chunks"]).resolve())
            retriever = ScopedRetriever(corpus, tokenizer_mode="mixed")
            pack = []
            for case in suite.cases:
                hits = retriever.search(case.task(Budget()), SearchRequest(case.query, top_k=5))
                for hit in hits:
                    corpus.read(hit.ref, frozenset(case.document_version_ids))
                pack.append(
                    {
                        "case": asdict(case),
                        "retrieved_candidates_not_gold": [asdict(h) for h in hits],
                        "reviewer": None,
                        "answerability": None,
                        "reference_answer": None,
                        "approved_evidence_ids": [],
                        "review_instructions": "核对全文、口径、例外与缺失条件；检索候选不是正确证据标签。确认后再独立划分开发/留出集。",
                    }
                )
            write(
                output / "annotation_pack.json",
                {"suite_hash": suite.fingerprint, "status": "awaiting_human_review", "cases": pack},
            )
            empty_run = {
                "suite_hash": suite.fingerprint,
                "system_id": "annotation_preparation",
                "mode": "offline_retrieval",
                "settings": {"tokenizer": "mixed"},
                "predictions": [],
            }
            pending = {
                "suite_hash": suite.fingerprint,
                "labels": {c.case_id: {"status": "pending", "expected": None} for c in suite.cases},
            }
            report = evaluate(suite, pending, empty_run, corpus)
            assert report["summary"]["exact_field_accuracy"] is None
            write(output / "candidate.report.json", report)
            candidate_summary = {
                "cases": len(pack),
                "cases_with_retrieval_candidates": sum(
                    bool(c["retrieved_candidates_not_gold"]) for c in pack
                ),
                "citation_refs_checked": sum(len(c["retrieved_candidates_not_gold"]) for c in pack),
                "approved_gold_count": 0,
                "answer_accuracy": None,
            }
    summary = {
        "mode": "offline_evaluation_protocol_test",
        "external_model_requests": 0,
        "implementation_hash": implementation_hash(),
        "comparison": comparison,
        "candidate_preparation": candidate_summary,
        "historical_rank35_reproduced": False,
        "limitations": [
            "Fixture outputs are simulated; scores validate the scorer only.",
            "Fixed RAG is newly implemented, not the original highest-scoring competition solver.",
            "Candidate tasks still need human review and independent partitioning.",
        ],
    }
    write(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
