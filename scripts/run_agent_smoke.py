"""Offline runtime integration experiments. Fake decisions, NEVER accuracy scores."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from run_retrieval_smoke import read_questions

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.retrieval import ScopedRetriever
from financial_agent.runtime import ResearchRuntime
from financial_agent.simulation import FakeResponse, fake_client, run_synthetic, target, tool


def real_corpus_decision(context):
    if context["stage"] == "plan":
        return {"targets": [target(question="检索原始研究问题的支撑证据")]}
    if not context["recent_actions"] or not context["evidence_catalog"]:
        if context["last_observation"].get("hits") == []:
            return {"kind": "ask_user", "question": "模拟执行结束：检索未返回证据，未启用真实模型。"}
        return tool("search", {"query": context["task"]["query"], "top_k": 5})
    if "offset" not in context["last_observation"]:
        return tool("read", {"evidence_id": context["evidence_catalog"][0]["evidence_id"]})
    return {"kind": "ask_user", "question": "模拟执行链检查已结束；未启用真实模型，不生成或评价答案。"}


def no_network(*args, **kwargs):
    raise AssertionError("Network is forbidden during offline agent smoke")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-corpus", action="store_true")
    parser.add_argument("--per-domain", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.per_domain <= 20:
        parser.error("--per-domain must be in [1,20]")
    run_id = "agent-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = root / "experiments/runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    inputs, results, corpus_summary = [], [], {}
    started = time.perf_counter()
    with patch("socket.socket.connect", no_network), patch("socket.create_connection", no_network):
        for scenario in ("finance", "contract"):
            result = run_synthetic(scenario)
            if result["status"] != "completed":
                raise AssertionError(f"Synthetic {scenario} failed: {result['reason']}")
            results.append({"scenario": scenario, "data_kind": "synthetic", "result": result})
        if args.real_corpus:
            config_path = root / "configs/datasets.example.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            chunks_path = (root / config["chunks"]).resolve()
            questions_dir = (root / config["questions_b"]).resolve()
            corpus = Corpus.from_jsonl(chunks_path)
            retriever = ScopedRetriever(corpus, tokenizer_mode="mixed")
            corpus_summary = {
                "documents": len(corpus.documents),
                "chunks": len(corpus.chunks),
                "tokenizer": "mixed",
            }
            inputs = [
                config_path,
                chunks_path,
                *(p for p in sorted(questions_dir.iterdir()) if p.suffix in {".json", ".jsonl"}),
            ]
            counts = Counter()
            for q in sorted(read_questions(questions_dir), key=lambda q: (q["domain"], q["qid"])):
                domain = q["domain"]
                if counts[domain] >= args.per_domain:
                    continue
                counts[domain] += 1
                versions = tuple(v.version_id for v in corpus.documents.values() if v.domain == domain)
                query = q["question"] + "\n" + "\n".join(str(v) for v in q.get("options", {}).values())
                task = TaskRequest(query, versions, budget=Budget(6, 8, 400_000))
                client = fake_client(
                    lambda payload: FakeResponse(
                        real_corpus_decision(json.loads(payload["messages"][-1]["content"]))
                    )
                )
                result = ResearchRuntime(
                    corpus, task, client, retriever=retriever, execution_label="simulated_model_real_corpus"
                ).run()
                if result["status"] != "needs_input" or result["answers"]:
                    raise AssertionError("Real corpus smoke must stop without fabricated answers")
                for ref in result["evidence"]:
                    original = corpus.refs[ref["chunk_id"]]
                    assert original.evidence_id == ref["evidence_id"]
                    corpus.read(original, frozenset(versions))
                assert any(e.get("action", {}).get("tool") == "read" for e in result["events"])
                results.append(
                    {"qid": q["qid"], "domain": domain, "data_kind": "real_corpus", "result": result}
                )
            corpus_summary["domain_counts"] = dict(counts)
    report = {
        "run_id": run_id,
        "mode": "offline_simulated_agent_integration",
        "external_model_requests": 0,
        "answer_accuracy": None,
        "semantic_verification_accuracy": None,
        "synthetic_scenarios": 2,
        "real_corpus_scenarios": sum(r["data_kind"] == "real_corpus" for r in results),
        "simulated_model_attempts": sum(r["result"]["budget"]["model_attempts"] for r in results),
        "tool_calls": sum(r["result"]["budget"]["tool_calls"] for r in results),
        "elapsed_seconds": time.perf_counter() - started,
        "corpus": corpus_summary,
        "python": platform.python_version(),
        "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (root / "src", root / "vendor", root / "scripts")
            for p in sorted(directory.rglob("*.py"))
        },
        "limitations": [
            "Fake planner/reviewer outputs validate plumbing only, not LLM autonomy or correctness.",
            "Real questions lack official gold labels; no accuracy calculation.",
            "No reproduction of the competition rank or score.",
        ],
        "results": results,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"results", "source_hashes", "inputs"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
