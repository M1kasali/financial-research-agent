"""Single-process, non-resumable live-batch reservation ledger; never a billing promise."""

import json
import os
from pathlib import Path

from financial_agent.policy import BudgetExceeded


class BatchSpendingGuard:
    # CNY / 10,000,000 units. Highest Beijing input/output tier, no cache discounts.
    PRICE_SOURCE = "https://help.aliyun.com/zh/model-studio/qwen3-7-flash"

    def __init__(self, journal: Path, *, cap_units: int = 10_000_000):
        if type(cap_units) is not int or not 0 < cap_units <= 10_000_000:
            raise ValueError("Live batch cap must be positive and at most 1 CNY")
        self.cap_units = cap_units
        self.reserved_units = 0
        self.reservations = 0
        self.reported_cost_units = 0
        self.reported_tokens = 0
        self.reported_attempts = 0
        self.unknown_attempts = 0
        self.journal = Path(journal)
        # Refuse reuse: a crashed/uncertain batch must not silently reset its allowance.
        with self.journal.open("x", encoding="utf-8"):
            pass
        self.journal.chmod(0o600)
        self._append({"kind": "config", "cap_cny": cap_units / 10_000_000,
                      "price_source": self.PRICE_SOURCE, "price_checked": "2026-09-14",
                      "reservation_input_cny_per_million": 1.2,
                      "reservation_output_cny_per_million": 4.8,
                      "resume_supported": False})

    def _append(self, event):
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reserve(self, payload):
        if payload.get("model") != "qwen3.7-flash":
            raise ValueError("Pricing guard only supports qwen3.7-flash")
        output = payload.get("max_tokens")
        if type(output) is not int or not 1 <= output <= 1600:
            raise ValueError("Live output allowance must be 1–1600 tokens")
        # Deliberately inflated UTF-8 proxy, not exact server tokenization.
        input_proxy = len(json.dumps(payload["messages"], ensure_ascii=False).encode("utf-8")) + 4096
        reserve_units = input_proxy * 12 + output * 48
        if input_proxy > 100_000 or self.reserved_units + reserve_units > self.cap_units:
            raise BudgetExceeded("Batch CNY reservation or per-request input ceiling exhausted")
        self._append({"kind": "reservation", "number": self.reservations + 1,
                      "input_token_proxy": input_proxy, "output_limit": output,
                      "reserved_cny": reserve_units / 10_000_000,
                      "cumulative_reserved_cny": (self.reserved_units + reserve_units) / 10_000_000})
        self.reservations += 1
        self.reserved_units += reserve_units

    def record(self, case_id, event):
        # Events contain accounting only, never payloads, credentials or HTTP headers.
        usage = event.get("usage", {}).get("reported")
        if event.get("usage_status") == "reported" and usage:
            prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
            input_rate, output_rate = (2, 8) if prompt <= 32_000 else (
                (6, 24) if prompt <= 256_000 else (12, 48))
            self.reported_cost_units += prompt * input_rate + completion * output_rate
            self.reported_tokens += usage["total_tokens"]
            self.reported_attempts += 1
        else:
            self.unknown_attempts += 1
        self._append({"kind": "attempt_result", "case_id": case_id, **event})

    def attach(self, runtime):
        # Install AFTER runtime/durability hooks; retain both task budget and WAL.
        original = runtime.client.before_attempt

        def before(payload):
            self.reserve(payload)
            original(payload)

        runtime.client.before_attempt = before

    def summary(self):
        return {"cap_cny": self.cap_units / 10_000_000,
                "reserved_cny": self.reserved_units / 10_000_000,
                "reservations": self.reservations,
                "reported_attempts": self.reported_attempts,
                "reported_tokens": self.reported_tokens if self.reported_attempts else None,
                "reported_usage_cost_estimate_cny": (
                    self.reported_cost_units / 10_000_000 if self.reported_attempts else None),
                "unknown_usage_attempts": self.unknown_attempts,
                "billing_authoritative": False,
                "note": "No reservations refunded, including unknown or locally rejected attempts. "
                        "UTF-8 reservation is an estimate; provider billing is authoritative."}
