from dataclasses import replace

import pytest
from agent.schemas import Chunk

from financial_agent.corpus import Corpus


@pytest.mark.parametrize(
    "change", [{"text": 123}, {"text": ""}, {"doc_id": ""}, {"page": 0}, {"page": True}, {"domain": None}]
)
def test_invalid_chunk_fields(change):
    original = Chunk("c1", "d1", "financial_reports", 1, "", "", "营业收入120万元")
    with pytest.raises(ValueError):
        Corpus([replace(original, **change)])


def test_old_evidence_not_reusable_after_source_update():
    original = Chunk("c1", "d1", "financial_reports", 1, "", "", "营业收入120万元")
    old = Corpus([original])
    new = Corpus([replace(original, text="营业收入125万元")])
    with pytest.raises(PermissionError):
        new.read(old.refs["c1"], frozenset(new.by_version))


def test_bad_json_has_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        Corpus.from_jsonl(path)
