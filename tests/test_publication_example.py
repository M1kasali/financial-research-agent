"""The documented quickstart must work without local competition materials."""

import json
from pathlib import Path

from financial_agent.cli import main
from financial_agent.corpus import Corpus


def test_bundled_example_prepare_is_offline(monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / "data/example/chunks.jsonl"
    corpus = Corpus.from_jsonl(path)
    assert len(corpus.documents) == 3
    assert all(chunk.metadata.get("synthetic") is True for chunk in corpus.chunks.values())

    def forbidden(*args, **kwargs):
        raise AssertionError("Quickstart must not load credentials or call a model")

    monkeypatch.setattr("dotenv.dotenv_values", forbidden)
    monkeypatch.setattr("financial_agent.model.RecordedModelClient.chat", forbidden)
    assert main([
        "prepare", "--chunks", str(path),
        "--query", "比较甲乙合同的提前终止通知期限及例外条件",
        "--doc", "example_contract_a", "--doc", "example_contract_b",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "prepared"
    serialized = json.dumps(result, ensure_ascii=False)
    assert "30日" in serialized and "15日" in serialized
