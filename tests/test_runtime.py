import json

import pytest
import requests

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.model import ModelConfig, RecordedModelClient
from financial_agent.policy import BudgetExceeded, TaskMeter
from financial_agent.runtime import ResearchRuntime
from financial_agent.simulation import (
    FakeResponse,
    ScriptedSend,
    evidence_ids,
    fake_client,
    run_synthetic,
    target,
    tool,
    verdict,
)


def runtime(corpus, entries, *, budget=None):
    send = ScriptedSend(entries)
    task = TaskRequest(
        "研究营业收入", (corpus.documents["a"].version_id,), budget=budget or Budget(12, 20, 400_000)
    )
    return ResearchRuntime(corpus, task, fake_client(send), execution_label="simulated_model"), send


def verify_action(context, target_id="t1"):
    return tool(
        "verify",
        {"answer": "甲公司2025年营业收入120万元。", "evidence_ids": evidence_ids(context)},
        target_id,
    )


def test_finance_loop_uses_original_dsl():
    result = run_synthetic("finance")
    assert result["status"] == "completed", result["reason"]
    assert result["calculations"][0]["display_value"] == "20.00"
    assert result["calculations"][0]["result"] == "0.2"
    assert {"t1", "overall"} == result["answers"].keys()
    assert result["budget"]["model_attempts"] == 9
    assert result["budget"]["tool_calls"] == 5


def test_failed_review_changes_search_and_preserves_citations():
    result = run_synthetic("contract")
    assert result["status"] == "completed", result["reason"]
    actions = [e["action"] for e in result["events"] if "action" in e]
    searches = [a["args"]["query"] for a in actions if a.get("tool") == "search"]
    assert len(searches) == 2 and searches[0] != searches[1]
    assert any(e.get("observation", {}).get("accepted") is False for e in result["events"])
    assert len(result["answers"]["overall"]["evidence_ids"]) == 2


def test_premature_finish_and_dependency_are_rejected(corpus):
    runner, _ = runtime(
        corpus,
        [
            {"targets": [target()]},
            {"kind": "finish"},
            tool("search", {"query": "收入"}, "overall"),
            {"kind": "ask_user", "question": "需要明确期间。"},
        ],
    )
    result = runner.run()
    assert result["status"] == "needs_input"
    assert len([e for e in result["events"] if e.get("kind") == "rejected"]) == 2
    assert not result["answers"]


@pytest.mark.parametrize(
    "bad_plan",
    [
        [],
        [target("a", deps=("b",))],
        [target("a", deps=("b",)), target("b", deps=("a",))],
        [target("overall")],
        [target(), target()],
    ],
)
def test_invalid_plan_fails_closed(corpus, bad_plan):
    runner, send = runtime(corpus, [{"targets": bad_plan}])
    result = runner.run()
    assert result["status"] == "failed"
    assert len(send.calls) == 1 and not result["evidence"]


@pytest.mark.parametrize(
    "action",
    [
        tool("shell", {"cmd": "rm -rf /"}),
        tool("read", {"evidence_id": "invented"}),
        tool("search", {"query": "收入", "document_version_ids": ["outside"]}),
        tool("search", {"query": "收入", "top_k": True}),
        tool("verify", {"answer": "空引用", "evidence_ids": []}),
    ],
)
def test_invalid_tools_are_rejected_and_charged(corpus, action):
    runner, _ = runtime(
        corpus, [{"targets": [target()]}, action, {"kind": "ask_user", "question": "请补充资料"}]
    )
    result = runner.run()
    assert result["status"] == "needs_input"
    assert result["budget"]["tool_calls"] == 1
    assert not result["answers"]
    assert result["events"][1]["kind"] == "rejected"


def test_extend_plan_cannot_remove_initial_goals(corpus):
    runner, _ = runtime(
        corpus,
        [
            {"targets": [target(question="营业收入")]},
            {"kind": "extend_plan", "targets": [target("t2", "补充现金流", ("t1",))]},
            {"kind": "ask_user", "question": "需要现金流数据"},
        ],
    )
    result = runner.run()
    assert result["unresolved_target_ids"] == ["t1", "t2", "overall"]
    assert runner.targets["overall"].depends_on == ("t1", "t2")


def test_repeated_tool_stops_under_budget(corpus):
    action = tool("search", {"query": "收入"})
    runner, _ = runtime(
        corpus, [{"targets": [target()]}, action, action, action, action], budget=Budget(3, 10, 200_000)
    )
    result = runner.run()
    assert result["status"] == "partial"
    assert result["budget"]["tool_calls"] == 3
    assert any("Repeated call" in e.get("observation", {}).get("error", "") for e in result["events"])


def test_token_budget_blocks_before_send(corpus):
    runner, send = runtime(corpus, [], budget=Budget(3, 10, 1))
    result = runner.run()
    assert result["status"] == "partial" and not send.calls
    assert result["budget"]["model_attempts"] == 0


def test_attempt_budget_counts_planner_and_verifier(corpus):
    runner, send = runtime(
        corpus,
        [{"targets": [target()]}, tool("search", {"query": "收入"}), verify_action],
        budget=Budget(5, 3, 200_000),
    )
    result = runner.run()
    assert result["status"] == "partial" and len(send.calls) == 3
    assert not result["answers"]  # Review call did not fit, so answer cannot become complete.


def test_retry_shares_reservation_and_attempt_budget():
    meter = TaskMeter(Budget(5, 1, 200_000))
    sends = []
    response = FakeResponse({})
    response.status_code = 503
    client = RecordedModelClient(
        ModelConfig("qwen-test", api_key="fake", enabled=True),
        send=lambda p: sends.append(p) or response,
        before_attempt=meter.before_attempt,
        sleep=lambda _: None,
    )
    with pytest.raises(BudgetExceeded):
        client.chat([{"role": "user", "content": "test"}])
    assert len(sends) == client.total_attempts == meter.model_attempts == 1
    assert meter.summary(client)["unknown_usage_attempts"] == 1


def test_timeout_fails_without_retry(corpus):
    runner, _ = runtime(corpus, [])

    def timeout(_):
        raise requests.Timeout("secret service error must not leak")

    runner.client._send = timeout
    result = runner.run()
    assert result["status"] == "failed" and result["budget"]["model_attempts"] == 1
    assert "secret service error" not in json.dumps(result)


@pytest.mark.parametrize("review", [{**verdict(), "supported": "true"}, {"sufficient": True}, verdict(False)])
def test_invalid_or_negative_verification_never_completes(corpus, review):
    runner, _ = runtime(
        corpus,
        [
            {"targets": [target()]},
            tool("search", {"query": "收入"}),
            verify_action,
            review,
            {"kind": "ask_user", "question": "证据不足"},
        ],
    )
    result = runner.run()
    assert not result["answers"] and result["status"] == "needs_input"


def test_single_use_runtime(corpus):
    runner, _ = runtime(corpus, [{"targets": [target()]}, {"kind": "ask_user", "question": "请明确"}])
    runner.run()
    with pytest.raises(ValueError, match="single-use"):
        runner.run()


def test_overall_review_is_required_even_after_subtask_passes(corpus):
    runner, _ = runtime(
        corpus,
        [
            {"targets": [target()]},
            tool("search", {"query": "收入"}),
            verify_action,
            verdict(),
            {"kind": "finish"},
            {"kind": "ask_user", "question": "整体问题还需核验"},
        ],
    )
    result = runner.run()
    assert "t1" in result["answers"] and "overall" not in result["answers"]
    assert result["status"] == "needs_input"
    assert any("Cannot finish" in e.get("observation", {}).get("error", "") for e in result["events"])


def test_fabricated_calculation_never_reaches_reviewer(corpus):
    def fabricated(context):
        action = verify_action(context)
        action["args"]["calculation_ids"] = ["invented"]
        return action

    runner, send = runtime(
        corpus,
        [
            {"targets": [target()]},
            tool("search", {"query": "收入"}),
            fabricated,
            {"kind": "ask_user", "question": "缺计算依据"},
        ],
    )
    result = runner.run()
    assert all(c["stage"] != "verify" for c in send.calls)
    assert not result["answers"]


def test_disabled_model_never_sends(corpus):
    runner, send = runtime(corpus, [])
    runner.client.config = ModelConfig("qwen-test")
    result = runner.run()
    assert result["status"] == "failed" and not send.calls
    assert result["budget"]["model_attempts"] == 0
