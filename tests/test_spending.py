import json
from types import SimpleNamespace

import pytest

from financial_agent.contracts import Budget
from financial_agent.model import ModelConfig, RecordedModelClient
from financial_agent.policy import BudgetExceeded, TaskMeter
from financial_agent.spending import BatchSpendingGuard


def payload():
    return {"model": "qwen3.7-flash", "messages": [{"role": "user", "content": "财报"}], "max_tokens": 1600}


def test_reserve_precedes_send_and_retains_task_hook(tmp_path):
    guard = BatchSpendingGuard(tmp_path / "cost.jsonl")
    meter = TaskMeter(Budget(max_model_attempts=1))
    client = SimpleNamespace(before_attempt=meter.before_attempt)
    guard.attach(SimpleNamespace(client=client))
    client.before_attempt(payload())
    assert meter.model_attempts == 1
    with pytest.raises(BudgetExceeded):
        client.before_attempt(payload())
    assert guard.reservations == 2  # Local denial retained conservatively; never refunded.
    assert meter.model_attempts == 1


def test_cap_refuses_before_send_and_reuse_refused(tmp_path):
    path = tmp_path / "cost.jsonl"
    guard = BatchSpendingGuard(path, cap_units=1)
    with pytest.raises(BudgetExceeded):
        guard.reserve(payload())
    assert guard.reservations == 0
    with pytest.raises(FileExistsError):
        BatchSpendingGuard(path)


def test_retry_each_reserved_unknown_retained_no_secret(tmp_path):
    path = tmp_path / "cost.jsonl"
    guard = BatchSpendingGuard(path)
    responses = iter([SimpleNamespace(status_code=503), SimpleNamespace(
        status_code=200, json=lambda: {"choices": [{"message": {"content": "{}"}}],
                                      "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                                                "total_tokens": 110}})])
    client = RecordedModelClient(ModelConfig("qwen3.7-flash", api_key="TEST_SECRET", enabled=True),
                                 send=lambda p: next(responses), sleep=lambda _: None,
                                 before_attempt=guard.reserve, record=lambda e: guard.record("test", e))
    client.chat(payload()["messages"], max_tokens=1600)
    summary = guard.summary()
    assert summary["reservations"] == 2
    assert summary["unknown_usage_attempts"] == 1
    assert summary["reported_usage_cost_estimate_cny"] == 0.000028
    assert "TEST_SECRET" not in path.read_text()
    assert "财报" not in path.read_text()
    assert len([json.loads(line) for line in path.read_text().splitlines()]) == 5


@pytest.mark.parametrize("updates", [{"model": "qwen-other"}, {"max_tokens": 1601}, {"max_tokens": True}])
def test_reject_unpriced_or_excessive_output(tmp_path, updates):
    guard = BatchSpendingGuard(tmp_path / "cost.jsonl")
    with pytest.raises(ValueError):
        guard.reserve({**payload(), **updates})


def test_large_input_and_unknown_summary(tmp_path):
    guard = BatchSpendingGuard(tmp_path / "cost.jsonl")
    with pytest.raises(BudgetExceeded):
        guard.reserve({**payload(), "messages": [{"role": "user", "content": "x" * 100_000}]})
    assert guard.summary()["reported_tokens"] is None
