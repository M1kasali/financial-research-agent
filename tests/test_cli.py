import json
from dataclasses import asdict

from financial_agent.cli import main


def test_cli_catalog_and_scoped_search(corpus, tmp_path, capsys):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in corpus.chunks.values()), encoding="utf-8"
    )
    assert main(["catalog", "--chunks", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["document_count"] == 2
    assert main(["search", "--chunks", str(path), "--query", "营业收入", "--doc", "a"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_calls"] == 0 and result["answer_generated"] is False
    assert result["evidence"]
    assert all(item["ref"]["doc_id"] == "a" for item in result["evidence"])


def test_cli_unknown_doc(corpus, tmp_path, capsys):
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(asdict(c)) for c in corpus.chunks.values()), encoding="utf-8")
    assert main(["search", "--chunks", str(path), "--query", "收入", "--doc", "missing"]) == 2
    assert "Unknown document" in capsys.readouterr().err
