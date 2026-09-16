from dataclasses import replace

import pytest
from agent.schemas import Chunk

from financial_agent.contracts import SearchRequest, TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.retrieval import ScopedRetriever


def test_search_and_read_source(corpus):
    version = corpus.documents["a"].version_id
    task = TaskRequest("营业收入", (version,))
    hits = ScopedRetriever(corpus, tokenizer_mode="char").search(task, SearchRequest("营业收入"))
    assert hits and all(hit.ref.doc_id == "a" for hit in hits)
    assert all(corpus.read(hit.ref, frozenset({version})) == hit.text for hit in hits)


def test_empty_tool_scope_is_not_full_corpus(corpus):
    task = TaskRequest("营业收入", (corpus.documents["a"].version_id,))
    assert ScopedRetriever(corpus).search(task, SearchRequest("营业收入", ())) == []


def test_model_cannot_expand_scope(corpus):
    task = TaskRequest("终止", (corpus.documents["a"].version_id,))
    with pytest.raises(PermissionError):
        ScopedRetriever(corpus).search(task, SearchRequest("终止", (corpus.documents["b"].version_id,)))


def test_unknown_version_rejected(corpus):
    with pytest.raises(ValueError):
        ScopedRetriever(corpus).search(TaskRequest("收入", ("unknown",)), SearchRequest("收入"))


def test_evidence_cannot_be_modified(corpus):
    ref = corpus.refs["a1"]
    with pytest.raises(ValueError):
        corpus.read(replace(ref, end=1), frozenset({ref.version_id}))
    with pytest.raises(PermissionError):
        corpus.read(ref, frozenset({corpus.documents["b"].version_id}))


def test_source_mutation_detected(corpus):
    ref = corpus.refs["a1"]
    corpus.chunks["a1"].text = "changed"
    with pytest.raises(ValueError):
        corpus.read(ref, frozenset({ref.version_id}))


def test_versions_content_bound_and_input_order_independent():
    def make(text):
        return [
            Chunk("a1", "a", "financial_reports", 1, "", "", text),
            Chunk("a2", "a", "financial_reports", 2, "", "", "附注"),
        ]

    first = Corpus(make("120万元"))
    reversed_corpus = Corpus(list(reversed(make("120万元"))))
    assert first.documents == reversed_corpus.documents
    assert first.documents != Corpus(make("121万元")).documents


def test_duplicate_chunk_rejected():
    chunk = Chunk("a1", "a", "x", 1, "", "", "text")
    with pytest.raises(ValueError):
        Corpus([chunk, chunk])


def test_postfilter_catches_faulty_backend(corpus, monkeypatch):
    retriever = ScopedRetriever(corpus)
    wrong = retriever.index.result_from_chunk(corpus.chunks["b1"], score=1, source="test", query="收入")
    monkeypatch.setattr(retriever.index, "search", lambda *a, **k: [wrong])
    with pytest.raises(PermissionError):
        retriever.search(TaskRequest("收入", (corpus.documents["a"].version_id,)), SearchRequest("收入"))


def test_recovered_legacy_dependencies_import():
    from agent_team_b1.retrieval_v1 import B1Retriever
    from agent_team_v7.adaptive_blind import AdaptiveBlindRetriever
    from agent_team_v9.option_coverage_index import OptionCoverageIndex

    assert B1Retriever and AdaptiveBlindRetriever and OptionCoverageIndex
