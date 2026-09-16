"""Deterministic FAKE model for integration tests/demos, never quality evaluation.

It shares the production transport/JSON/runtime interfaces but does not call an API.
"""

import json

from agent.schemas import Chunk

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.model import ModelConfig, RecordedModelClient
from financial_agent.runtime import ResearchRuntime


def target(key="t1", question="研究子问题", deps=()):
    return {"target_id": key, "question": question, "depends_on": list(deps), "required_facets": []}


def tool(name, args, target_id="t1"):
    return {"kind": "tool", "target_id": target_id, "tool": name, "args": args}


def verdict(accepted=True):
    return {
        "supported": accepted,
        "sufficient": accepted,
        "failure_tags": [] if accepted else ["missing_exception"],
        "reason": "模拟审阅结果，不代表真实模型判断",
        "missing_evidence": "" if accepted else "缺少例外条款",
        "next_search_goal": "" if accepted else "重大违约的通知例外",
    }


class FakeResponse:
    status_code = 200

    def __init__(self, value):
        self.value = value

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self.value, ensure_ascii=False)}}]}


class ScriptedSend:
    """Callable entries inspect the actual preceding observation instead of guessing IDs."""

    def __init__(self, entries):
        self.entries = iter(entries)
        self.calls = []

    def __call__(self, payload):
        context = json.loads(payload["messages"][-1]["content"])
        self.calls.append(context)
        entry = next(self.entries)
        return FakeResponse(entry(context) if callable(entry) else entry)


def fake_client(send):
    return RecordedModelClient(
        ModelConfig(
            "qwen-simulated",
            api_key="FAKE_NOT_A_SECRET",
            enabled=True,
            max_attempts_per_call=1,
            max_total_attempts=100,
        ),
        send=send,
    )


def finance_corpus():
    return Corpus(
        [
            Chunk(
                "row",
                "demo-annual",
                "financial_reports",
                1,
                "合并利润表",
                "",
                "甲公司合并利润表，人民币万元，2025年度营业收入120；2024年度营业收入100。",
                metadata={
                    "financial_context": {"entity": "甲公司"},
                    "financial_row": {
                        "metric": "营业收入",
                        "unit": "万元",
                        "raw_row": "营业收入 120 100",
                        "cells": [
                            {"year": "2025", "raw_value": "120", "unit": "万元"},
                            {"year": "2024", "raw_value": "100", "unit": "万元"},
                        ],
                    },
                },
            )
        ]
    )


def evidence_ids(context):
    return [r["evidence_id"] for r in context["evidence_catalog"]]


def finance_decision(context):
    if context["stage"] == "plan":
        return {"targets": [target(question="计算甲公司营业收入同比变化")]}
    if context["stage"] == "verify":
        return verdict()  # Synthetic protocol test ONLY, not semantic verification.
    if "overall" in context["verified_answers"]:
        return {"kind": "finish"}
    ready = context["ready_target_ids"][0]
    if not context["evidence_catalog"]:
        return tool("search", {"query": "营业收入"})
    if not context["facts"]:
        return tool("extract", {"evidence_ids": evidence_ids(context)})
    if not context["calculations"]:
        facts = sorted(context["facts"], key=lambda f: f["year"], reverse=True)
        return tool("calculate", {"operation": "growth_rate", "fact_ids": [f["fact_id"] for f in facts]})
    calc = context["calculations"][0]
    return tool(
        "verify",
        {
            "answer": f"甲公司2025年度合并营业收入同比增长{calc['display_value']}%。",
            "evidence_ids": evidence_ids(context),
            "calculation_ids": [calc["calculation_id"]],
        },
        ready,
    )


def contract_decision(context):
    if context["stage"] == "plan":
        return {"targets": [target(question="核验提前解除合同的通知要求及例外")]}
    if context["stage"] == "verify":
        return verdict(any("不适用" in e["text"] for e in context["evidence"]))
    if "overall" in context["verified_answers"]:
        return {"kind": "finish"}
    if not context["evidence_catalog"]:
        return tool("search", {"query": "提前30日通知", "top_k": 1})
    if context["last_observation"].get("accepted") is False:
        return tool("search", {"query": "重大违约解除不适用提前通知", "top_k": 1})
    answer = "解除合同需要提前30日通知。"
    if len(context["evidence_catalog"]) > 1:
        answer += "重大违约解除不适用该通知要求。"
    return tool(
        "verify", {"answer": answer, "evidence_ids": evidence_ids(context)}, context["ready_target_ids"][0]
    )


def run_synthetic(scenario):
    if scenario == "finance":
        corpus, decision = finance_corpus(), finance_decision
        query = "甲公司2025年度营业收入同比变化是多少？"
    elif scenario == "contract":
        corpus = Corpus(
            [
                Chunk(
                    "notice",
                    "demo-contract",
                    "financial_contracts",
                    1,
                    "通知",
                    "",
                    "解除合同需要提前30日通知。",
                ),
                Chunk(
                    "exception",
                    "demo-contract",
                    "financial_contracts",
                    2,
                    "例外",
                    "",
                    "重大违约解除合同，不适用提前通知要求。",
                ),
            ]
        )
        decision, query = contract_decision, "核验解除合同的通知要求及例外。"
    else:
        raise ValueError("Unknown synthetic scenario")
    task = TaskRequest(query, tuple(corpus.by_version), budget=Budget(12, 24, 400_000))
    client = fake_client(
        lambda payload: FakeResponse(decision(json.loads(payload["messages"][-1]["content"])))
    )
    return ResearchRuntime(corpus, task, client, execution_label="simulated_model_synthetic_data").run()
