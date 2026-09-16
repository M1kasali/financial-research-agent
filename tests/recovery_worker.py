"""Child-process crash injection worker; never connects to a network."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.runtime import ResearchRuntime
from financial_agent.simulation import FakeResponse, fake_client, finance_corpus, finance_decision
from financial_agent.storage import SQLiteTaskStore


def main():
    path, mode = Path(sys.argv[1]), sys.argv[2]
    corpus = finance_corpus()
    task = TaskRequest(
        "甲公司2025年度营业收入同比变化是多少？",
        tuple(corpus.by_version),
        task_id="crash-test",
        session_id="session-test",
        budget=Budget(12, 24, 400_000),
    )
    sends = []

    def send(payload):
        sends.append(payload)
        if mode == "in_send":
            os._exit(23)
        return FakeResponse(finance_decision(json.loads(payload["messages"][-1]["content"])))

    store = SQLiteTaskStore(path)
    runner = ResearchRuntime(
        corpus, task, fake_client(send), store=store, execution_label="simulated_process_recovery"
    )
    checkpoint = runner._checkpoint

    def crash_checkpoint():
        checkpoint()
        if mode == "after_search" and runner._phase == "act" and runner._step == 2:
            os._exit(23)

    runner._checkpoint = crash_checkpoint
    finish_call = store.finish_call

    def crash_response(task_id, call_id, **kwargs):
        finish_call(task_id, call_id, **kwargs)
        if (mode == "after_action_response" and call_id == "1:act") or (
            mode == "after_verify_response" and call_id == "4:verify"
        ):
            os._exit(23)

    store.finish_call = crash_response
    with (
        patch("socket.socket.connect", side_effect=AssertionError("No network")),
        patch("socket.create_connection", side_effect=AssertionError("No network")),
    ):
        result = runner.run()
    print(json.dumps({"new_sends": len(sends), "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
