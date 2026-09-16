import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from financial_agent.model import ModelConfig
from financial_agent.policy import BudgetExceeded
from financial_agent.spending import BatchSpendingGuard

spec = importlib.util.spec_from_file_location(
    "core_comparison", Path(__file__).resolve().parents[1] / "scripts/run_core_comparison.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_dry_plan_does_not_load_key_or_corpus(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry plan must not read configuration or corpus")
    monkeypatch.setattr(runner, "dotenv_values", forbidden)
    monkeypatch.setattr(runner.Corpus, "from_jsonl", forbidden)
    runner.main([])
    assert "No API call" in capsys.readouterr().out
    assert len(runner.CASES) * len(runner.CONDITIONS) * runner.RUN_CAP_UNITS == 10_000_000


def test_response_journal_preserves_invalid_json_but_not_key(tmp_path):
    journal = tmp_path / "responses.jsonl"
    response = SimpleNamespace(status_code=200, json=lambda: {
        "choices": [{"message": {"content": "invalid JSON TEST_SECRET"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    client = runner.ResponseJournalClient(
        ModelConfig("qwen3.7-flash", api_key="TEST_SECRET", enabled=True), journal,
        send=lambda _: response)
    result = client.chat([{"role": "user", "content": '{"stage":"answer"}'}])
    assert result["content"] == "invalid JSON TEST_SECRET"
    saved = json.loads(journal.read_text())
    assert saved["stage"] == "answer"
    assert saved["content"] == "invalid JSON [REDACTED]"
    assert journal.stat().st_mode & 0o777 == 0o600


def test_both_spending_caps_precede_network(tmp_path):
    calls = []
    client = runner.ResponseJournalClient(
        ModelConfig("qwen3.7-flash", api_key="test", enabled=True), tmp_path / "responses.jsonl",
        send=lambda p: calls.append(p), before_attempt=lambda p: None)
    engine = SimpleNamespace(client=client)
    per_run = BatchSpendingGuard(tmp_path / "run.jsonl", cap_units=1)
    batch = BatchSpendingGuard(tmp_path / "batch.jsonl")
    per_run.attach(engine)
    batch.attach(engine)
    with pytest.raises(BudgetExceeded):
        client.chat([{"role": "user", "content": '{"stage":"answer"}'}])
    assert not calls
    assert batch.reservations == 1
    assert per_run.reservations == 0
