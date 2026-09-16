"""Explicit-opt-in Qwen transport with per-attempt accounting.

No implicit .env loading. Timeout/connection uncertainty is not automatically retried.
This is transport infrastructure, not an autonomous research implementation.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Callable
from urllib.parse import urlparse

import requests


@dataclass(frozen=True)
class ModelConfig:
    model: str
    api_key: str = field(default="", repr=False)
    enabled: bool = False
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout_seconds: float = 60
    max_attempts_per_call: int = 2
    max_total_attempts: int = 16

    def __post_init__(self):
        url = urlparse(self.base_url)
        if (
            url.scheme != "https"
            or url.hostname != "dashscope.aliyuncs.com"
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.port not in (None, 443)
            or url.path.rstrip("/") != "/compatible-mode/v1"
        ):
            raise ValueError("Only the configured official HTTPS Qwen endpoint is supported")
        if not self.model.startswith("qwen"):
            raise ValueError("An explicit Qwen model is required")
        for value in (self.max_attempts_per_call, self.max_total_attempts):
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError("Attempt limits must be integers in [1,100]")
        if not 0 < self.timeout_seconds <= 300:
            raise ValueError("Invalid timeout")


@dataclass(frozen=True)
class Usage:
    status: str
    reported: dict[str, int] | None
    estimated: dict[str, int] | None


def parse_usage(raw: object, input_text: str, output_text: str) -> Usage:
    names = ("prompt_tokens", "completion_tokens", "total_tokens")
    if isinstance(raw, dict) and all(type(raw.get(k)) is int and raw[k] >= 0 for k in names):
        if raw["total_tokens"] > 0 and raw["total_tokens"] == raw["prompt_tokens"] + raw["completion_tokens"]:
            return Usage("reported", {k: raw[k] for k in names}, None)
    prompt, completion = max(1, len(input_text) // 2), max(1, len(output_text) // 2)
    return Usage(
        "estimated",
        None,
        {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    )


class ModelCallError(RuntimeError):
    pass


class RecordedModelClient:
    def __init__(
        self,
        config: ModelConfig,
        *,
        send: Callable | None = None,
        record: Callable[[dict], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        before_attempt: Callable[[dict], None] | None = None,
    ):
        self.config = config
        self._send = send or self._http_send
        self._record_sink = record
        self._sleep = sleep
        self.before_attempt = before_attempt
        self.attempts: list[dict] = []
        self.total_attempts = 0

    def _http_send(self, payload: dict):
        return requests.post(
            self.config.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json=payload,
            timeout=self.config.timeout_seconds,
            allow_redirects=False,
        )

    def _record(self, event: dict):
        # Deliberately excludes API key, request headers, prompts and response bodies.
        self.attempts.append(event)
        if self._record_sink:
            self._record_sink(event)

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 800) -> dict:
        if not self.config.enabled:
            raise ModelCallError("Model requests are disabled; explicit opt-in is required")
        if not self.config.api_key:
            raise ModelCallError("Missing API key")
        if not messages or any(
            not isinstance(m.get("content"), str) or m.get("role") not in {"system", "user", "assistant"}
            for m in messages
        ):
            raise ValueError("Invalid messages")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError("Invalid generation token limit")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "enable_thinking": False,
        }
        for call_attempt in range(1, self.config.max_attempts_per_call + 1):
            if self.total_attempts >= self.config.max_total_attempts:
                raise ModelCallError("Shared model attempt budget exhausted")
            if self.before_attempt:
                self.before_attempt(payload)
            self.total_attempts += 1
            event = {"attempt": self.total_attempts, "call_attempt": call_attempt, "model": self.config.model}
            try:
                response = self._send(payload)
            except requests.RequestException as exc:
                self._record(
                    {
                        **event,
                        "status": "uncertain",
                        "usage_status": "unknown",
                        "error_type": type(exc).__name__,
                    }
                )
                raise ModelCallError("Request outcome uncertain; automatic retry stopped") from None
            if response.status_code != 200:
                self._record(
                    {
                        **event,
                        "status": "http_error",
                        "http_status": response.status_code,
                        "usage_status": "unknown",
                    }
                )
                if (
                    response.status_code in {429, 502, 503, 504}
                    and call_attempt < self.config.max_attempts_per_call
                ):
                    self._sleep(min(2.0, 0.25 * 2 ** min(call_attempt - 1, 3)))
                    continue
                raise ModelCallError(f"Model HTTP failure: {response.status_code}")
            data = None
            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("Invalid content")
            except (ValueError, KeyError, IndexError, TypeError):
                usage = parse_usage(data.get("usage"), "", "") if isinstance(data, dict) else None
                if usage is not None and usage.status == "reported":
                    self._record(
                        {
                            **event,
                            "status": "invalid_response",
                            "usage_status": "reported",
                            "usage": asdict(usage),
                        }
                    )
                else:
                    self._record({**event, "status": "invalid_response", "usage_status": "unknown"})
                raise ModelCallError("Invalid model response; not retried") from None
            usage = parse_usage(data.get("usage"), "\n".join(m["content"] for m in messages), content)
            self._record({**event, "status": "ok", "usage_status": usage.status, "usage": asdict(usage)})
            return {"content": content, "usage": asdict(usage)}
        raise ModelCallError("Model attempts exhausted")
