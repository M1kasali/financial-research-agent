"""Financial Agent CLI; network stays off unless answer --execute is explicit."""

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

from financial_agent.contracts import Budget, SearchRequest, TaskRequest
from financial_agent.core_answer import CoreAnswerEngine, render_core_answer
from financial_agent.core_evidence import prepare_core_evidence
from financial_agent.core_research import CoreResearchEngine
from financial_agent.corpus import Corpus
from financial_agent.financial_workflow import (
    FinancialQuestion,
    FinancialStatementWorkflow,
    render_financial_report,
)
from financial_agent.model import ModelConfig, RecordedModelClient
from financial_agent.retrieval import ScopedRetriever
from financial_agent.statement_binding import METRICS
from financial_agent.storage import RecoveryError, SQLiteTaskStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog", help="List content-bound document versions")
    catalog.add_argument("--chunks", type=Path, required=True)
    search = commands.add_parser("search", help="Retrieve evidence only; no answer generation")
    search.add_argument("--chunks", type=Path, required=True)
    search.add_argument("--query", required=True)
    scope = search.add_mutually_exclusive_group(required=True)
    scope.add_argument("--doc", action="append", dest="doc_ids")
    scope.add_argument("--all-documents", action="store_true")
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--tokenizer", choices=["mixed", "char", "word"], default="mixed")
    prepare = commands.add_parser("prepare", help="V45-source evidence windows for a free task; no model call")
    prepare.add_argument("--chunks", type=Path, required=True)
    prepare.add_argument("--query", required=True)
    prepare.add_argument("--doc", action="append", dest="doc_ids", required=True)
    prepare.add_argument("--options-json", default="{}", help='Optional candidate claims, e.g. {"A":"..."}')
    prepare.add_argument("--candidate-k", type=int, default=40)
    prepare.add_argument("--max-chars", type=int, default=6500)
    prepare.add_argument("--max-documents", type=int, default=4)
    prepare.add_argument("--max-chunks", type=int, default=8)
    prepare.add_argument("--selection-strategy", choices=["legacy", "coverage"], default="legacy")
    prepare.add_argument("--tokenizer", choices=["mixed", "char", "word"], default="mixed")
    answer = commands.add_parser("answer", help="V45-source evidence answer; model disabled unless --execute")
    answer.add_argument("--chunks", type=Path, required=True)
    answer.add_argument("--query", required=True)
    answer.add_argument("--doc", action="append", dest="doc_ids", required=True)
    answer.add_argument("--options-json", default="{}")
    answer.add_argument("--tokenizer", choices=["mixed", "char", "word"], default="mixed")
    answer.add_argument("--execute", action="store_true", help="Explicitly authorize current .env model requests")
    answer.add_argument("--output-dir", type=Path, help="Required NEW private directory for real execution")
    answer.add_argument("--no-follow-up", action="store_true", help="Single-pass comparison; no refinement/financial routing")
    answer.add_argument("--selection-strategy", choices=["legacy", "coverage"], default="legacy")
    answer.add_argument("--citation-mode", choices=["verbatim", "span_id"], default="verbatim",
                        help="Optional exact source-span IDs; does not bypass semantic review")
    financial = commands.add_parser("financial", help="Rule-based annual financial workflow; no model planning")
    financial.add_argument("--chunks", type=Path, required=True)
    financial.add_argument("--doc", required=True)
    financial.add_argument("--year", type=int, required=True)
    financial.add_argument("--scope", choices=["parent", "consolidated"], required=True)
    financial.add_argument("--operation", choices=["extract", "growth_rate", "ratio"], required=True)
    financial.add_argument("--metric", choices=sorted(METRICS), required=True)
    financial.add_argument("--denominator-metric", choices=sorted(METRICS))
    financial.add_argument("--max-tool-calls", type=int, default=12)
    financial.add_argument("--tokenizer", choices=["mixed", "char", "word"], default="mixed")
    financial.add_argument("--output-dir", type=Path, help="New directory for JSON trace and Markdown report")
    state = commands.add_parser(
        "state", help="Inspect local persisted task status; never resumes or sends requests"
    )
    state.add_argument("--db", type=Path, required=True)
    state.add_argument("--task", required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        if args.command == "state":
            if not args.db.is_file():
                raise ValueError("State database does not exist")
            with SQLiteTaskStore(args.db).exclusive() as store:
                result = store.inspect(args.task)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        corpus = Corpus.from_jsonl(args.chunks)
        if args.command == "answer":
            if set(args.doc_ids) - corpus.documents.keys():
                raise ValueError("Unknown document ID; run catalog first")
            if args.output_dir is not None and args.output_dir.exists():
                raise ValueError("Output directory exists; no automatic overwrite or paid resume")
            if args.execute and args.output_dir is None:
                raise ValueError("Real execution requires a new --output-dir")
            from financial_agent.core_evidence import validate_options

            options = json.loads(args.options_json)
            validate_options(options)
            task = TaskRequest(args.query, tuple(corpus.documents[d].version_id
                                                 for d in dict.fromkeys(args.doc_ids)),
                               budget=Budget(1, 4, 100_000) if args.no_follow_up else Budget(16, 8, 200_000))
            config = ModelConfig("qwen3.7-flash", enabled=False)
            guard = None
            if args.execute:
                from dotenv import dotenv_values

                from financial_agent.spending import BatchSpendingGuard

                env = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
                if (env.get("QWEN_MODEL") != "qwen3.7-flash" or not env.get("DASHSCOPE_API_KEY")
                        or not env.get("QWEN_BASE_URL")):
                    raise ValueError("Current .env needs explicit Qwen3.7 Flash, official endpoint and key")
                config = ModelConfig(env["QWEN_MODEL"], api_key=env["DASHSCOPE_API_KEY"],
                                     base_url=env["QWEN_BASE_URL"], enabled=True, timeout_seconds=45,
                                     max_total_attempts=task.budget.max_model_attempts)
            if args.output_dir is not None:
                args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
                (args.output_dir / "task.json").write_text(json.dumps(asdict(task), ensure_ascii=False, indent=2))
            if args.execute:
                guard = BatchSpendingGuard(args.output_dir / "spending.jsonl", cap_units=2_500_000)
            client = RecordedModelClient(config, record=(
                (lambda event: guard.record(task.task_id, event)) if guard else None))
            engine_class = CoreAnswerEngine if args.no_follow_up else CoreResearchEngine
            engine = engine_class(corpus, ScopedRetriever(corpus, tokenizer_mode=args.tokenizer),
                                  task, client, options=options, selection_strategy=args.selection_strategy,
                                  citation_mode=args.citation_mode,
                                  execution_label="single_pass_core_answer" if args.no_follow_up else "bounded_core_research")
            if guard:
                guard.attach(engine)
            result = engine.run()
            if guard:
                result["spending"] = guard.summary()
            if args.output_dir is not None:
                (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                (args.output_dir / "report.md").write_text(render_core_answer(result))
                print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir.resolve()),
                                  "usage": result["usage"], "spending": result.get("spending")},
                                 ensure_ascii=False, indent=2))
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "answered" else 3
        if args.command == "prepare":
            if set(args.doc_ids) - corpus.documents.keys():
                raise ValueError("Unknown document ID; run catalog first")
            task = TaskRequest(args.query, tuple(corpus.documents[d].version_id
                                                 for d in dict.fromkeys(args.doc_ids)))
            result = prepare_core_evidence(
                corpus, ScopedRetriever(corpus, tokenizer_mode=args.tokenizer), task,
                options=json.loads(args.options_json), candidate_k=args.candidate_k,
                max_chars=args.max_chars, max_documents=args.max_documents, max_chunks=args.max_chunks,
                selection_strategy=args.selection_strategy,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "financial":
            if args.doc not in corpus.documents:
                raise ValueError("Unknown document ID; run catalog first")
            if args.output_dir is not None and args.output_dir.exists():
                raise ValueError("Output directory already exists; choose a new one")
            question = FinancialQuestion(corpus.documents[args.doc].version_id, args.year, args.scope,
                                         args.operation, args.metric, args.denominator_metric)
            result = FinancialStatementWorkflow(corpus, ScopedRetriever(corpus, tokenizer_mode=args.tokenizer),
                                                question, max_tool_calls=args.max_tool_calls).run()
            if args.output_dir is not None:
                args.output_dir.mkdir(parents=True, exist_ok=False)
                (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                           encoding="utf-8")
                (args.output_dir / "report.md").write_text(render_financial_report(result), encoding="utf-8")
                result = {"status": result["status"], "reason_code": result["reason_code"],
                          "output_directory": str(args.output_dir.resolve()), "model_calls": 0,
                          "tool_calls": result["tool_calls"]}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "completed" else 3
        if args.command == "catalog":
            result = {
                "document_count": len(corpus.documents),
                "chunk_count": len(corpus.chunks),
                "documents": [asdict(doc) for doc in corpus.documents.values()],
            }
        else:
            selected = args.doc_ids if args.doc_ids else list(corpus.documents)
            if set(selected) - corpus.documents.keys():
                raise ValueError("Unknown document ID; run catalog first")
            task = TaskRequest(
                args.query, tuple(corpus.documents[d].version_id for d in dict.fromkeys(selected))
            )
            retriever = ScopedRetriever(corpus, tokenizer_mode=args.tokenizer)
            hits = retriever.search(task, SearchRequest(args.query, top_k=args.top_k))
            result = {
                "mode": "offline_retrieval",
                "status": "evidence_found" if hits else "no_evidence",
                "task_id": task.task_id,
                "model_calls": 0,
                "answer_generated": False,
                "evidence": [asdict(hit) for hit in hits],
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, PermissionError, OSError, RecoveryError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
