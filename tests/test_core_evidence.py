import json
from dataclasses import asdict, replace

import pytest
from agent.schemas import RetrievalResult
from agent_team_b1.option_coverage_v13 import option_coverage_evidence

from financial_agent.cli import main
from financial_agent.contracts import EvidenceHit, SearchRequest, TaskRequest
from financial_agent.core_evidence import pack_core_evidence, prepare_core_evidence
from financial_agent.retrieval import ScopedRetriever


def context(corpus):
    task = TaskRequest("2025年营业收入", (corpus.documents["a"].version_id,))
    retriever = ScopedRetriever(corpus, tokenizer_mode="char")
    return task, retriever, retriever.search(task, SearchRequest(task.query))


def test_exact_released_component_parity_and_no_answer(corpus):
    task, _, hits = context(corpus)
    options = {"A": "2025年收入120万元", "B": "2024年收入100万元"}
    result = pack_core_evidence(corpus, task, hits, options=options)
    legacy = [RetrievalResult(h.ref.chunk_id, h.ref.doc_id, corpus.chunks[h.ref.chunk_id].domain,
                              h.score, "test", task.query, h.text,
                              {"page": h.ref.physical_page, "section": h.section}) for h in hits]
    expected, selected = option_coverage_evidence(legacy, question_text=task.query, options=options,
                                                 max_chars=6500, max_documents=4, max_chunks=8)
    assert result["evidence_text"] == expected
    assert [w["ref"]["chunk_id"] for w in result["windows"]] == [r.chunk_id for r in selected]
    assert result["model_calls"] == 0 and not result["answer_generated"]
    assert result["sufficiency"] == "not_assessed"


def test_re_reads_source_not_supplied_hit_text(corpus):
    task, _, hits = context(corpus)
    result = pack_core_evidence(corpus, task, [replace(hits[0], text="FORGED", section="FORGED")])
    assert "FORGED" not in result["evidence_text"]


def test_out_of_scope_and_modified_reference_fail_closed(corpus):
    task, _, hits = context(corpus)
    with pytest.raises(PermissionError):
        pack_core_evidence(corpus, task, [EvidenceHit(corpus.refs["b1"], "", 1, "")])
    with pytest.raises(ValueError):
        pack_core_evidence(corpus, task, [replace(hits[0], ref=replace(hits[0].ref, start=1))])
    corpus.chunks[hits[0].ref.chunk_id].text += "changed"
    with pytest.raises(ValueError):
        pack_core_evidence(corpus, task, hits)


@pytest.mark.parametrize("limits", [{"max_chars": 255}, {"max_chars": True}, {"max_chunks": 0},
                                   {"max_documents": 41}, {"options": {"X": "abc"}},
                                   {"options": {"A": ""}}, {"options": []},
                                   {"options": {"A": "one", "B": "two"}, "max_chunks": 1}])
def test_invalid_limits_and_options(corpus, limits):
    task, _, hits = context(corpus)
    with pytest.raises(ValueError):
        pack_core_evidence(corpus, task, hits, **limits)


def test_no_candidates_is_not_answer_and_no_seed_parameter(corpus):
    task, _, _ = context(corpus)
    result = pack_core_evidence(corpus, task, [])
    assert result["status"] == "no_evidence" and result["windows"] == []
    with pytest.raises(TypeError):
        pack_core_evidence(corpus, task, [], locked_answers=["A"])


def test_free_task_entry_does_not_require_competition_id(corpus):
    task, retriever, _ = context(corpus)
    result = prepare_core_evidence(corpus, retriever, task, max_chars=400)
    assert result["rendered_chars"] <= 400 and "qid" not in result
    assert all(w["ref"]["doc_id"] == "a" for w in result["windows"])


def test_prepare_cli_offline_and_bad_options(corpus, tmp_path, capsys):
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in corpus.chunks.values()))
    args = ["prepare", "--chunks", str(path), "--doc", "a", "--query", "营业收入", "--tokenizer", "char"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_calls"] == 0 and result["selected_window_count"] > 0
    assert main(args + ["--options-json", "[]"]) == 2
    assert "error" in json.loads(capsys.readouterr().err)
