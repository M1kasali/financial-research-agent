"""Synthetic scripted-model protocol smoke; ZERO real API calls, not model accuracy."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from agent.schemas import Chunk

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.core_answer import CoreAnswerEngine, render_core_answer
from financial_agent.corpus import Corpus
from financial_agent.retrieval import ScopedRetriever
from financial_agent.simulation import ScriptedSend, fake_client


def answer(context):
    window = next(w for w in context["windows"] if "2025年营业收入120万元" in w["excerpt_text"])
    return {"decision": "answered", "claims": [
        {"claim_id": "c1", "text": "甲公司2025年营业收入120万元。", "kind": "extraction",
         "citations": [{"window_number": window["window_number"], "quote": "2025年营业收入120万元"}]}],
        "gaps": [], "option_judgments": []}


def review(_):
    return {"claim_verdicts": [{"claim_id": "c1", "supported": True, "reason": "模拟判断，不是模型效果"}],
            "sufficient": True, "gaps": []}


def bad_quote(context):
    result = answer(context)
    result["claims"][0]["citations"][0]["quote"] = "营业收入999万元"
    return result


def nonanswer(decision, gap):
    return {"decision": decision, "claims": [], "gaps": [gap], "option_judgments": []}


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "experiments/runs" / (
        "core-answer-simulated-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("answered", "甲公司2025年营业收入是多少？", [answer, review], 4, "answered"),
        ("abstained", "甲公司收入变化原因？", [nonanswer("abstained", "没有披露收入变化原因")], 4, "abstained"),
        ("clarification", "请核查公司收入", [nonanswer("clarification", "请指定需要核查的年度")], 4, "clarification"),
        ("calculation", "计算甲公司收入同比", [nonanswer("needs_calculation", "需要财务计算工具")], 4, "needs_calculation"),
        ("forged_quote", "甲公司2025年营业收入是多少？", [bad_quote], 4, "validation_failed"),
        ("budget", "甲公司2025年营业收入是多少？", [answer], 1, "budget_exhausted"),
        ("review_rejected", "甲公司2025年营业收入是多少？", [answer, {
            "claim_verdicts": [{"claim_id": "c1", "supported": False, "reason": "模拟口径冲突"}],
            "sufficient": False, "gaps": ["核查合并/母公司口径"]}], 4, "needs_evidence"),
    ]
    summaries = {}
    with patch("socket.socket", side_effect=AssertionError("Network disabled for synthetic smoke")):
        corpus = Corpus([Chunk("a1", "a", "financial_reports", 1, "营业收入", "",
                               "甲公司2025年营业收入120万元。"),
                         Chunk("a2", "a", "financial_reports", 2, "营业收入", "",
                               "甲公司2024年营业收入100万元。")])
        retriever = ScopedRetriever(corpus, tokenizer_mode="char")
        for name, query, entries, attempts, expected in cases:
            task = TaskRequest(query, tuple(corpus.by_version), budget=Budget(1, attempts, 100_000))
            result = CoreAnswerEngine(corpus, retriever, task, fake_client(ScriptedSend(entries)),
                                      execution_label="simulated_model_synthetic_data_NOT_quality_eval").run()
            assert result["status"] == expected
            assert result["status"] == "answered" or not result["answer"]
            (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            (output / f"{name}.md").write_text(render_core_answer(result))
            summaries[name] = {"status": result["status"],
                               "simulated_model_attempts": result["usage"]["model_attempts"]}
    summary = {"execution": "synthetic_scripted_model_protocol_checks", "real_api_calls": 0,
               "network_disabled": True, "answer_accuracy": None, "cases": summaries}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
