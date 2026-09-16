"""Kill and restart offline child processes, recording auditable recovery results."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from financial_agent.durability import implementation_hash
from financial_agent.storage import SQLiteTaskStore


def main():
    root = Path(__file__).resolve().parents[1]
    run_id = "recovery-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = root / "experiments/runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    worker = root / "tests/recovery_worker.py"
    results = []
    for mode, expected in (
        ("after_search", 7),
        ("after_action_response", 7),
        ("after_verify_response", 3),
        ("in_send", 0),
    ):
        database = output / f"{mode}.sqlite3"
        crashed = subprocess.run(
            [sys.executable, str(worker), str(database), mode], capture_output=True, text=True, timeout=30
        )
        if crashed.returncode != 23:
            raise AssertionError(f"Expected injected process exit in {mode}: {crashed.stderr}")
        resumed = subprocess.run(
            [sys.executable, str(worker), str(database), "resume"], capture_output=True, text=True, timeout=30
        )
        if resumed.returncode != 0:
            raise AssertionError(resumed.stderr)
        payload = json.loads(resumed.stdout)
        assert payload["new_sends"] == expected
        assert payload["result"]["status"] == ("needs_attention" if mode == "in_send" else "completed")
        with SQLiteTaskStore(database).exclusive() as store:
            inspection = store.inspect("crash-test")
        (output / f"{mode}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append(
            {
                "scenario": mode,
                "injected_exit_code": crashed.returncode,
                "status": payload["result"]["status"],
                "new_simulated_attempts_on_resume": payload["new_sends"],
                "total_model_attempts": payload["result"]["budget"]["model_attempts"],
                "tool_calls": payload["result"]["budget"]["tool_calls"],
                "pending_calls": inspection["ledger"]["pending_calls"],
                "checkpoint_revision": inspection["revision"],
                "database": str(database),
            }
        )
    report = {
        "run_id": run_id,
        "mode": "offline_real_process_crash_injection",
        "external_model_requests": 0,
        "scenario_count": len(results),
        "all_checks_passed": True,
        "implementation_hash": implementation_hash(),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "limitations": [
            "Synthetic data and fake model; not real API or model quality evaluation.",
            "SQLite on local WSL filesystem; not distributed execution or exactly-once external requests.",
            "Uncertain calls stop; external reconciliation UI is not implemented.",
        ],
        "results": results,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
