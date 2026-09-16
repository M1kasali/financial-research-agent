import json
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import requests

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.runtime import ResearchRuntime
from financial_agent.simulation import FakeResponse, fake_client, finance_corpus, finance_decision
from financial_agent.storage import RecoveryError, SQLiteTaskStore


def build(path, *, corpus=None, task=None, config=None, decision=finance_decision):
    corpus = corpus or finance_corpus()
    task = task or TaskRequest(
        "甲公司2025年度营业收入同比变化是多少？",
        tuple(corpus.by_version),
        task_id="durable-test",
        session_id="session",
        budget=Budget(12, 24, 400_000),
    )
    sends = []

    def send(payload):
        context = json.loads(payload["messages"][-1]["content"])
        sends.append(context)
        return FakeResponse(decision(context))

    client = fake_client(send)
    if config:
        client.config = config
    runner = ResearchRuntime(
        corpus, task, client, store=SQLiteTaskStore(path), execution_label="simulated_durable"
    )
    return runner, sends


def test_pause_resume_reuses_search_and_preserves_budget(tmp_path):
    path = tmp_path / "state.sqlite3"
    first, sends1 = build(path)
    paused = first.run(stop_after_steps=1)
    assert paused["status"] == "paused" and len(sends1) == 2
    second, sends2 = build(path)

    def no_search(*args):
        raise AssertionError("Completed search must not rerun")

    second.tools.search = no_search
    result = second.run()
    assert result["status"] == "completed", result["reason"]
    assert len(sends2) == 7
    assert result["budget"]["model_attempts"] == 9
    assert result["budget"]["tool_calls"] == 5
    assert result["calculations"][0]["display_value"] == "20.00"
    assert sum(e.get("action", {}).get("tool") == "search" for e in result["events"]) == 1


@pytest.mark.parametrize("pause_steps", [2, 3, 4, 5])
def test_resume_across_each_tool_boundary(tmp_path, pause_steps):
    path = tmp_path / "state.sqlite3"
    first, _ = build(path)
    first.run(stop_after_steps=pause_steps)
    resumed, _ = build(path)
    result = resumed.run()
    baseline, _ = build(tmp_path / "baseline.sqlite3")
    original = baseline.run()
    for key in ("status", "answers", "evidence", "facts", "calculations", "events", "budget"):
        if key in {"facts", "calculations"}:
            assert sorted(result[key], key=str) == sorted(original[key], key=str)
        else:
            assert result[key] == original[key], key


def test_completed_result_is_cached_without_calls(tmp_path):
    path = tmp_path / "state.sqlite3"
    first, _ = build(path)
    expected = first.run()
    second, sends = build(path)
    assert second.run() == expected
    assert sends == []


@pytest.mark.parametrize(
    "field,value", [("query", "另一研究问题"), ("session_id", "other"), ("budget", Budget(30, 40, 900_000))]
)
def test_task_identity_change_refused(tmp_path, field, value):
    path = tmp_path / "state.sqlite3"
    runner, _ = build(path)
    runner.run(stop_after_steps=1)
    changed, sends = build(path, task=replace(runner.task, **{field: value}))
    with pytest.raises(RecoveryError, match="changed"):
        changed.run()
    assert sends == []


def test_changed_document_version_refused(tmp_path):
    path = tmp_path / "state.sqlite3"
    first, _ = build(path)
    first.run(stop_after_steps=1)
    chunk = finance_corpus().chunks["row"]
    changed_corpus = Corpus([replace(chunk, text=chunk.text + "更新披露")])
    resumed, sends = build(path, corpus=changed_corpus)
    with pytest.raises(RecoveryError, match="changed"):
        resumed.run()
    assert not sends


def test_changed_model_refused_but_credential_rotation_allowed(tmp_path):
    path = tmp_path / "state.sqlite3"
    first, _ = build(path)
    first.run(stop_after_steps=1)
    changed, sends = build(path, config=replace(first.client.config, model="qwen-other"))
    with pytest.raises(RecoveryError, match="changed"):
        changed.run()
    assert not sends
    secret = "credential-rotation-test-do-not-store-879321"
    rotated, _ = build(path, config=replace(first.client.config, api_key=secret))
    assert rotated.run()["status"] == "completed"
    assert secret.encode() not in path.read_bytes()


def test_code_fingerprint_change_refused(tmp_path, monkeypatch):
    path = tmp_path / "state.sqlite3"
    first, _ = build(path)
    first.run(stop_after_steps=1)
    monkeypatch.setattr("financial_agent.durability.implementation_hash", lambda: "new-code")
    second, sends = build(path)
    with pytest.raises(RecoveryError, match="changed"):
        second.run()
    assert not sends


def test_lock_refuses_concurrent_runner(tmp_path):
    path = tmp_path / "state.sqlite3"
    owner = SQLiteTaskStore(path)
    with owner.exclusive():
        runner, sends = build(path)
        with pytest.raises(RecoveryError, match="Another runner"):
            runner.run()
        assert not sends
    runner, _ = build(path)
    assert runner.run()["status"] == "completed"


def test_non_agent_database_is_not_overwritten(tmp_path):
    path = tmp_path / "other.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated(value TEXT)")
        db.execute("INSERT INTO unrelated VALUES ('keep me')")
    store = SQLiteTaskStore(path)
    with pytest.raises(RecoveryError, match="Not a financial-agent"):
        with store.exclusive():
            pass
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT value FROM unrelated").fetchone()[0] == "keep me"


def test_schema_version_refused(tmp_path):
    path = tmp_path / "state.sqlite3"
    with SQLiteTaskStore(path).exclusive():
        pass
    with sqlite3.connect(path) as db:
        db.execute("UPDATE store_meta SET value='999'")
    with pytest.raises(RecoveryError, match="Unsupported state schema"):
        with SQLiteTaskStore(path).exclusive():
            pass


def test_timeout_is_needs_attention_and_never_auto_retried(tmp_path):
    path = tmp_path / "state.sqlite3"
    first, _ = build(path)

    def timeout(_):
        raise requests.Timeout("private diagnostic")

    first.client._send = timeout
    result = first.run()
    assert result["status"] == "needs_attention"
    assert result["budget"]["unknown_usage_attempts"] == 1
    second, sends = build(path)
    assert second.run() == result and not sends
    assert "private diagnostic".encode() not in path.read_bytes()


def test_attempt_budget_does_not_reset_after_resume(tmp_path):
    path = tmp_path / "state.sqlite3"
    corpus = finance_corpus()
    task = TaskRequest(
        "收入", tuple(corpus.by_version), task_id="limited", session_id="s", budget=Budget(10, 3, 400_000)
    )
    first, _ = build(path, task=task)
    first.run(stop_after_steps=1)
    second, sends = build(path, task=task)
    result = second.run()
    assert result["status"] == "partial" and len(sends) == 1
    assert result["budget"]["model_attempts"] == 3


@pytest.mark.parametrize(
    "mode,new_sends", [("after_search", 7), ("after_action_response", 7), ("after_verify_response", 3)]
)
def test_real_process_crash_recovery(tmp_path, mode, new_sends):
    path = tmp_path / "crashed.sqlite3"
    worker = Path(__file__).with_name("recovery_worker.py")
    killed = subprocess.run(
        [sys.executable, str(worker), str(path), mode], capture_output=True, text=True, timeout=30
    )
    assert killed.returncode == 23, killed.stderr
    completed = subprocess.run(
        [sys.executable, str(worker), str(path), "resume"], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["new_sends"] == new_sends
    result = payload["result"]
    assert result["status"] == "completed", result["reason"]
    assert result["budget"]["model_attempts"] == 9
    assert result["budget"]["tool_calls"] == 5


def test_real_process_loss_during_request_blocks_resend(tmp_path):
    path = tmp_path / "uncertain.sqlite3"
    worker = Path(__file__).with_name("recovery_worker.py")
    killed = subprocess.run(
        [sys.executable, str(worker), str(path), "in_send"], capture_output=True, timeout=30
    )
    assert killed.returncode == 23
    resumed = subprocess.run(
        [sys.executable, str(worker), str(path), "resume"], capture_output=True, text=True, timeout=30
    )
    assert resumed.returncode == 0, resumed.stderr
    payload = json.loads(resumed.stdout)
    assert payload["new_sends"] == 0
    assert payload["result"]["status"] == "needs_attention"
    assert payload["result"]["budget"]["model_attempts"] == 1
    assert payload["result"]["budget"]["unknown_usage_attempts"] == 1


def test_store_requires_lock(tmp_path):
    with pytest.raises(RecoveryError, match="exclusive"):
        SQLiteTaskStore(tmp_path / "state.sqlite3").ledger("x")


@pytest.mark.parametrize("limit", [0, True, -1, 101])
def test_invalid_pause_limit(tmp_path, limit):
    runner, sends = build(tmp_path / "state.sqlite3")
    with pytest.raises(ValueError, match="stop_after_steps"):
        runner.run(stop_after_steps=limit)
    assert not sends


def test_uncommitted_local_tool_can_be_recomputed_without_model_resend(tmp_path):
    path = tmp_path / "state.sqlite3"
    first, _ = build(path)
    save = first.store.save

    def failed_write(task_id, checkpoint, **kwargs):
        if checkpoint["phase"] == "act" and checkpoint["step"] == 2:
            raise OSError("Simulated disk interruption before result commit")
        return save(task_id, checkpoint, **kwargs)

    first.store.save = failed_write
    with pytest.raises(OSError):
        first.run()
    second, sends = build(path)
    result = second.run()
    assert result["status"] == "completed"
    assert len(sends) == 7 and result["budget"]["model_attempts"] == 9
    assert result["budget"]["tool_calls"] == 5


def test_state_cli_is_non_executing(tmp_path, capsys):
    from financial_agent.cli import main

    path = tmp_path / "state.sqlite3"
    first, _ = build(path)
    first.run(stop_after_steps=1)
    assert main(["state", "--db", str(path), "--task", "durable-test"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "paused" and result["phase"] == "act"
    assert result["ledger"]["attempts"] == 2
    assert main(["state", "--db", str(path), "--task", "missing"]) == 2
    assert "Unknown task" in capsys.readouterr().err


def test_state_cli_missing_path_does_not_create_database(tmp_path, capsys):
    from financial_agent.cli import main

    path = tmp_path / "missing" / "state.sqlite3"
    assert main(["state", "--db", str(path), "--task", "x"]) == 2
    assert not path.exists() and not path.parent.exists()
