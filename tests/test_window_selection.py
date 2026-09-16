import json
from dataclasses import asdict, replace

import pytest
from agent.schemas import Chunk
from test_core_research import complete_answer, contract_setup, good_review, nonanswer, refine

from financial_agent.cli import main
from financial_agent.contracts import EvidenceHit, TaskRequest
from financial_agent.core_evidence import pack_core_evidence
from financial_agent.corpus import Corpus
from financial_agent.window_selection import clause_mentions, requested_facets


def setup(query="比较第48—51条与5.3、5.4的补充材料规定。"):
    chunks = [Chunk(f"n{i}", "a", "insurance", i, "", "", "保险合同材料申请和核定。" + str(i)) for i in range(1, 9)]
    chunks += [Chunk("clauses", "a", "insurance", 9, "", "", "第四十八条 补充材料。第四十九条 三十日核定。第五十条 拒赔。第五十一条 先付。"),
               Chunk("other", "b", "insurance", 1, "", "", "5.3 申请材料。5.4 复杂核定的三十日不包括补充材料期间。")]
    corpus = Corpus(chunks)
    task = TaskRequest(query, tuple(corpus.by_version))
    hits = [EvidenceHit(corpus.refs[c.chunk_id], "FORGED_BODY", 100-i, "FORGED_SECTION") for i, c in enumerate(chunks)]
    return corpus, task, hits


@pytest.mark.parametrize("query,expected", [
    ("第48—51条", {"article:48", "article:49", "article:50", "article:51"}),
    ("第四十八至第五十一条", {"article:48", "article:49", "article:50", "article:51"}),
    ("第四十六条及9.1、9.2和10.2", {"article:46", "clause:9.1", "clause:9.2", "clause:10.2"}),
    ("2025年度第1至999条", set()),
])
def test_clause_normalization_is_bounded(query, expected):
    assert set(clause_mentions(query)) == expected


def test_lower_rank_clauses_and_both_documents_kept_from_same_candidates():
    corpus, task, hits = setup()
    result = pack_core_evidence(corpus, task, hits, selection_strategy="coverage", max_chunks=2)
    assert {w["ref"]["chunk_id"] for w in result["windows"]} == {"clauses", "other"}
    assert result["coverage_audit"]["unrepresented_document_versions"] == []
    assert result["rendered_chars"] <= 6500 and result["model_calls"] == 0
    assert "FORGED" not in result["evidence_text"]
    assert hits[0].text == "FORGED_BODY"  # Caller data not mutated.


@pytest.mark.parametrize("chars,slots,docs", [(256, 8, 4), (500, 2, 1), (1000, 1, 4), (6500, 8, 4)])
def test_limits_and_missing_coverage_are_reported_not_guaranteed(chars, slots, docs):
    corpus, task, hits = setup()
    result = pack_core_evidence(corpus, task, hits, selection_strategy="coverage", max_chars=chars,
                                max_chunks=slots, max_documents=docs)
    assert len(result["evidence_text"]) <= chars
    assert len(result["windows"]) <= slots
    assert len({w["ref"]["doc_id"] for w in result["windows"]}) <= docs
    actual = {w["ref"]["version_id"] for w in result["windows"]}
    assert set(result["coverage_audit"]["unrepresented_document_versions"]) == set(task.document_version_ids) - actual
    assert result["sufficiency"] == "not_assessed"


def test_exact_duplicates_removed_but_conflicting_values_retained():
    chunks = [Chunk("x", "a", "financial_reports", 1, "", "", "营业收入100万元"),
              Chunk("copy", "a", "financial_reports", 2, "", "", "营业收入100万元"),
              Chunk("conflict", "a", "financial_reports", 3, "", "", "营业收入200万元"),
              Chunk("b", "b", "financial_reports", 1, "", "", "营业收入100万元")]
    corpus = Corpus(chunks)
    task = TaskRequest("核查营业收入", tuple(corpus.by_version))
    hits = [EvidenceHit(corpus.refs[c.chunk_id], c.text, 1, "") for c in chunks]
    result = pack_core_evidence(corpus, task, hits, selection_strategy="coverage")
    assert {w["ref"]["chunk_id"] for w in result["windows"]} == {"x", "conflict", "b"}
    assert result["coverage_audit"]["exact_duplicate_bodies_removed"] == 1


def test_options_use_exact_legacy_fallback():
    corpus, task, hits = setup()
    options = {"A": "复杂核定排除补材料时间", "B": "收到材料即可给付"}
    legacy = pack_core_evidence(corpus, task, hits, options=options)
    coverage = pack_core_evidence(corpus, task, hits, options=options, selection_strategy="coverage")
    assert legacy["windows"] == coverage["windows"] and legacy["evidence_text"] == coverage["evidence_text"]
    assert coverage["effective_selection_strategy"] == "legacy"


def test_table_cell_separators_must_not_be_removed_for_deduplication():
    chunks = [Chunk("left", "a", "financial_reports", 1, "", "", "营业收入 1 20"),
              Chunk("right", "a", "financial_reports", 2, "", "", "营业收入 12 0")]
    corpus = Corpus(chunks)
    task = TaskRequest("核查营业收入", tuple(corpus.by_version))
    hits = [EvidenceHit(corpus.refs[c.chunk_id], c.text, 1, "") for c in chunks]
    result = pack_core_evidence(corpus, task, hits, selection_strategy="coverage")
    assert len(result["windows"]) == 2
    assert result["coverage_audit"]["exact_duplicate_bodies_removed"] == 0


def test_unknown_strategy_and_out_of_scope_fail_before_selection():
    corpus, task, hits = setup()
    with pytest.raises(ValueError):
        pack_core_evidence(corpus, task, hits, selection_strategy="gold")
    with pytest.raises(PermissionError):
        pack_core_evidence(corpus, replace(task, document_version_ids=(hits[0].ref.version_id,)), hits,
                           selection_strategy="coverage")
    empty = pack_core_evidence(corpus, task, [], selection_strategy="coverage")
    assert empty["windows"] == [] and empty["status"] == "no_evidence"


def test_no_answer_ids_are_accepted_and_financial_facets_are_only_lexical():
    assert "financial_unit_declaration" in requested_facets("核查2025年合并营业收入，注明单位")
    corpus, task, hits = setup()
    with pytest.raises(TypeError):
        pack_core_evidence(corpus, task, hits, selection_strategy="coverage", required_chunk_ids=["clauses"])


def test_refinement_keeps_optional_strategy_and_recomputes_accumulation():
    runner, _, _ = contract_setup([nonanswer("abstained"), refine, complete_answer, good_review])
    runner.pack_limits["selection_strategy"] = "coverage"
    result = runner.run()
    assert result["status"] == "answered" and len(result["evidence_pack"]["windows"]) == 2
    assert result["evidence_pack"]["coverage_audit"]["accumulated"] is True
    assert result["evidence_pack"]["latest_refinement_selection_audit"] is not None


@pytest.mark.parametrize("command", ["prepare", "answer"])
def test_cli_opt_in_no_model_or_key(command, tmp_path, monkeypatch, capsys):
    import dotenv
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a: pytest.fail("No key access"))
    monkeypatch.setattr("socket.socket", lambda *a, **k: pytest.fail("No network"))
    corpus, _, _ = setup()
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in corpus.chunks.values()))
    status = main([command, "--chunks", str(chunks), "--doc", "a", "--doc", "b", "--query", "核查第48—51条和5.4",
                   "--selection-strategy", "coverage", "--tokenizer", "char"])
    result = json.loads(capsys.readouterr().out)
    pack = result if command == "prepare" else result["evidence_pack"]
    assert pack["effective_selection_strategy"] == "coverage"
    assert status == (0 if command == "prepare" else 3)
