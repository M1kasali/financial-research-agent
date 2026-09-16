"""Small real-corpus retrieval experiment; no answer generation or model requests."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from financial_agent.contracts import SearchRequest, TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.retrieval import ScopedRetriever


def read_questions(directory):
    questions = []
    for path in sorted(directory.iterdir()):
        if path.suffix == ".jsonl":
            questions.extend(
                json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
            )
        elif path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            questions.extend(data if isinstance(data, list) else data["questions"])
    return questions


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/datasets.example.json")
    parser.add_argument("--per-domain", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.per_domain <= 20:
        parser.error("--per-domain must be in [1,20]")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    chunks_path = (root / config["chunks"]).resolve()
    questions_dir = (root / config["questions_b"]).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = root / "experiments/runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    corpus = Corpus.from_jsonl(chunks_path)
    retriever = ScopedRetriever(corpus, tokenizer_mode="mixed")
    build_seconds = time.perf_counter() - start
    questions = read_questions(questions_dir)
    counts = Counter()
    results = []
    for q in sorted(questions, key=lambda q: (q["domain"], q["qid"])):
        domain = q["domain"]
        if counts[domain] >= args.per_domain:
            continue
        counts[domain] += 1
        versions = tuple(d.version_id for d in corpus.documents.values() if d.domain == domain)
        query = q["question"] + "\n" + "\n".join(str(x) for x in q.get("options", {}).values())
        task = TaskRequest(query, versions)
        started = time.perf_counter()
        hits = retriever.search(task, SearchRequest(query, top_k=5))
        for hit in hits:
            assert corpus.read(hit.ref, frozenset(versions)) == hit.text
        results.append(
            {
                "qid": q["qid"],
                "domain": domain,
                "hit_count": len(hits),
                "query_seconds": time.perf_counter() - started,
                "all_references_valid": True,
                "hits": [{"ref": asdict(h.ref), "score": h.score} for h in hits],
            }
        )
    input_paths = [
        chunks_path,
        *(p for p in sorted(questions_dir.iterdir()) if p.suffix in {".json", ".jsonl"}),
    ]
    report = {
        "run_id": run_id,
        "mode": "offline_retrieval_smoke",
        "model_calls": 0,
        "answer_accuracy": None,
        "document_count": len(corpus.documents),
        "chunk_count": len(corpus.chunks),
        "index_build_seconds": build_seconds,
        "question_count": len(results),
        "domain_counts": dict(counts),
        "questions_with_hits": sum(bool(r["hit_count"]) for r in results),
        "reference_integrity_passed": all(r["all_references_valid"] for r in results),
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in input_paths
        ],
        "configuration": {
            "tokenizer": "mixed",
            "scoring": "bm25f_lite",
            "top_k": 5,
            "scope": "question_domain",
            "sample": "first qids per domain",
        },
        "python": platform.python_version(),
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (root / "src", root / "vendor", root / "scripts")
            for p in sorted(directory.rglob("*.py"))
        },
        "limitations": [
            "Hits are not necessarily relevant or sufficient.",
            "No gold answer scoring, no generation, no autonomous planning.",
            "Corpus not confirmed equivalent to highest-scoring submission.",
        ],
        "results": results,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "run_id",
                    "question_count",
                    "questions_with_hits",
                    "reference_integrity_passed",
                    "index_build_seconds",
                    "model_calls",
                )
            },
            ensure_ascii=False,
        )
    )
    print(f"Report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
