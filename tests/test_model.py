import json

import pytest
import requests

from financial_agent.model import ModelCallError, ModelConfig, RecordedModelClient, parse_usage


class Response:
    def __init__(self, status=200, usage=None):
        self.status_code = status
        self.usage = usage

    def json(self):
        return {"choices": [{"message": {"content": "answer"}}], "usage": self.usage}


MESSAGES = [{"role": "user", "content": "test"}]


def test_default_has_no_model_requests():
    client = RecordedModelClient(ModelConfig("qwen-test"), send=lambda _: pytest.fail("Must not call"))
    with pytest.raises(ModelCallError, match="disabled"):
        client.chat(MESSAGES)
    assert client.total_attempts == 0


def test_missing_key_has_no_request():
    client = RecordedModelClient(ModelConfig("qwen-test", enabled=True))
    with pytest.raises(ModelCallError, match="Missing"):
        client.chat(MESSAGES)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"total_tokens": 5},
        {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 5},
        {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
    ],
)
def test_estimates_are_not_reported_usage(raw):
    usage = parse_usage(raw, "input", "output")
    assert usage.status == "estimated" and usage.reported is None
    assert usage.estimated["total_tokens"] > 0


def test_original_usage_preserved():
    raw = {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
    client = RecordedModelClient(
        ModelConfig("qwen-test", api_key="dummy-secret", enabled=True), send=lambda _: Response(usage=raw)
    )
    assert client.chat(MESSAGES)["usage"]["reported"] == raw
    assert "dummy-secret" not in json.dumps(client.attempts)
    assert "dummy-secret" not in repr(client.config)


def test_timeout_unknown_no_retry():
    def timeout(_):
        raise requests.Timeout("potentially secret response")

    client = RecordedModelClient(ModelConfig("qwen-test", api_key="test", enabled=True), send=timeout)
    with pytest.raises(ModelCallError, match="uncertain"):
        client.chat(MESSAGES)
    assert client.total_attempts == 1
    assert client.attempts[0]["usage_status"] == "unknown"
    assert "secret" not in json.dumps(client.attempts)


def test_shared_retry_budget():
    client = RecordedModelClient(
        ModelConfig("qwen-test", api_key="test", enabled=True, max_total_attempts=2),
        send=lambda _: Response(status=429),
    )
    with pytest.raises(ModelCallError):
        client.chat(MESSAGES)
    assert len(client.attempts) == 2
    with pytest.raises(ModelCallError, match="budget"):
        client.chat(MESSAGES)
    assert len(client.attempts) == 2


def test_invalid_success_response_recorded():
    class Invalid:
        status_code = 200

        def json(self):
            return {"choices": []}

    client = RecordedModelClient(
        ModelConfig("qwen-test", api_key="test", enabled=True), send=lambda _: Invalid()
    )
    with pytest.raises(ModelCallError, match="Invalid"):
        client.chat(MESSAGES)
    assert client.attempts[0]["status"] == "invalid_response"


@pytest.mark.parametrize(
    "url",
    [
        "http://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://evil.example/v1",
        "https://dashscope.aliyuncs.com@evil.example/v1",
    ],
)
def test_endpoint_restriction(url):
    with pytest.raises(ValueError):
        ModelConfig("qwen-test", base_url=url)


def test_bad_content_does_not_discard_reported_usage():
    class Invalid:
        status_code = 200

        def json(self):
            return {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}}

    client = RecordedModelClient(
        ModelConfig("qwen-test", api_key="test", enabled=True), send=lambda _: Invalid()
    )
    with pytest.raises(ModelCallError):
        client.chat(MESSAGES)
    assert client.attempts[0]["usage"]["reported"]["total_tokens"] == 12


def test_http_transport_disables_redirects(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(requests, "post", fake_post)
    client = RecordedModelClient(ModelConfig("qwen-test", api_key="test", enabled=True))
    client.chat(MESSAGES)
    assert captured["allow_redirects"] is False
