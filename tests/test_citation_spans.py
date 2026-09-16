"""Offline protocol/source tests, not model-quality or semantic accuracy evaluation."""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest
from agent.schemas import Chunk
from test_core_answer import engine, review
from test_core_research import contract_setup, good_review, nonanswer, refine

from financial_agent.citation_spans import (
    build_citation_catalog,
    public_spans,
    resolve_span_candidate,
    segment_ranges,
    visible_region,
)
from financial_agent.contracts import Budget, TaskRequest
from financial_agent.core_answer import SPAN_ANSWER_INSTRUCTIONS, render_core_answer, validate_candidate
from financial_agent.corpus import Corpus


def setup_pack(source="期限为30日。\n\n补充提供证明和资料的期间不计入。", excerpt=None):
    corpus = Corpus([Chunk("a1", "a", "financial_contracts", 1, "", "", source)])
    task = TaskRequest("核查期限和例外", (corpus.documents["a"].version_id,))
    pack = {"windows": [{"window_number": 1, "ref": asdict(corpus.refs["a1"]),
                         "excerpt_text": source if excerpt is None else excerpt}], "options": {}}
    return corpus, task, pack


def answer_with_ids(ids):
    return {"decision": "answered", "claims": [{"claim_id": "c1", "kind": "extraction",
            "text": "模拟结论；不用于语义评估。", "citations": [{"span_id": sid} for sid in ids]}],
            "gaps": [], "option_judgments": []}


def span_answer(context):
    return answer_with_ids([context["citation_spans"][0]["span_id"]])


def test_exact_newlines_and_offsets_including_unicode():
    source = "页首😀\n期限为30日。\n\n补充资料不计入。\n未展示尾部"
    corpus, task, pack = setup_pack(source, "…期限为30日。\n\n补充资料不计入。…")
    catalog = build_citation_catalog(pack, corpus, task)
    assert len(catalog["spans"]) == 1
    span = catalog["spans"][0]
    assert span["source_start"] == source.index("期限") and span["excerpt_start"] == 1
    assert source[span["source_start"]:span["source_end"]] == span["quote"]
    assert "\n\n" in span["quote"] and "尾部" not in span["quote"]
    raw = answer_with_ids([span["span_id"]])
    resolved = resolve_span_candidate(raw, catalog, pack, corpus, task)
    validate_candidate(resolved, pack, corpus, task)
    assert raw["claims"][0]["citations"] == [{"span_id": span["span_id"]}]
    assert public_spans(catalog) == [{k: span[k] for k in ("span_id", "window_number", "quote")}]
    assert catalog["semantic_support"] == "not_assessed"


@pytest.mark.parametrize("excerpt,source,expected", [
    ("…原文…", "页首原文页尾", (1, 3, 2)),
    ("…原文…", "真实…原文…结束", (0, 4, 2)),
    ("重复", "重复重复", None),
    ("aaa", "aaaa", None),  # Overlapping repeats are ambiguous too.
    ("数字1 20", "数字12 0", None),
    ("期限30日", "期限\n30日", None),
    ("…", "…", None),
])
def test_region_only_unique_exact_body(excerpt, source, expected):
    assert visible_region(excerpt, source) == expected


def test_ambiguous_window_gets_no_ids():
    corpus, task, pack = setup_pack("重复重复", "重复")
    catalog = build_citation_catalog(pack, corpus, task)
    assert catalog["spans"] == [] and len(catalog["unavailable_windows"]) == 1


@pytest.mark.parametrize("source", [
    "项目   2025  2024\n收入   1 20   12 0\n金额  1,234.56   -20.50%",
    "甲" * 398 + "123456789.25%" + "说明",
    "1" * 500 + "\n表尾",
    "甲" * 500 + "\n" + "乙" * 500,
    "条款。\r\n" * 200,
])
def test_segments_preserve_table_spaces_and_do_not_split_numeric_tokens(source):
    spans = list(segment_ranges(source))
    assert spans
    numeric = "0123456789.,+-/%"
    for start, end in spans:
        assert 2 <= end - start <= 400
        assert not (end < len(source) and source[end - 1] in numeric and source[end] in numeric)
        assert not (start > 0 and source[start - 1] in numeric and source[start] in numeric)
    if source.startswith("项目"):
        assert source[spans[0][0]:spans[0][1]] == source
        assert "1 20" in source and "12 0" in source


@pytest.mark.parametrize("limit", [0, 1, -1, True])
def test_invalid_segment_limit(limit):
    with pytest.raises(ValueError):
        list(segment_ranges("原文", limit))


@pytest.mark.parametrize("change", [
    lambda p: p["windows"][0].update(excerpt_text="不存在的文字"),
    lambda p: p["windows"][0].update(window_number=2),
    lambda p: p.update(options={"A": "新命题"}),
])
def test_changed_window_or_options_rejects_old_catalog(change):
    corpus, task, pack = setup_pack()
    catalog = build_citation_catalog(pack, corpus, task)
    raw = answer_with_ids([catalog["spans"][0]["span_id"]])
    change(pack)
    with pytest.raises(ValueError):
        resolve_span_candidate(raw, catalog, pack, corpus, task)


@pytest.mark.parametrize("change", [{"task_id": "other"}, {"session_id": "other"}, {"query": "新问题"}])
def test_task_bound_ids(change):
    corpus, task, pack = setup_pack()
    catalog = build_citation_catalog(pack, corpus, task)
    raw = answer_with_ids([catalog["spans"][0]["span_id"]])
    other = replace(task, **change)
    assert catalog["catalog_id"] != build_citation_catalog(pack, corpus, other)["catalog_id"]
    with pytest.raises(ValueError):
        resolve_span_candidate(raw, catalog, pack, corpus, other)


def test_scope_source_and_catalog_tampering():
    corpus, task, pack = setup_pack()
    catalog = build_citation_catalog(pack, corpus, task)
    raw = answer_with_ids([catalog["spans"][0]["span_id"]])
    with pytest.raises(PermissionError):
        resolve_span_candidate(raw, catalog, pack, corpus, replace(task, document_version_ids=("outside",)))
    tampered = deepcopy(catalog)
    tampered["spans"][0].update(quote="伪造", source_start=999)
    with pytest.raises(ValueError):
        resolve_span_candidate(raw, tampered, pack, corpus, task)
    corpus.chunks["a1"].text += "改动"
    with pytest.raises(ValueError):
        resolve_span_candidate(raw, catalog, pack, corpus, task)


@pytest.mark.parametrize("citation", [
    {"span_id": "sp-unknown"}, {"span_id": True}, {"span_id": []},
    {"span_id": "CURRENT", "quote": "伪造"},
    {"span_id": "CURRENT", "source_start": 0},
    {"window_number": 1, "quote": "期限为30日。"},
])
def test_unknown_ids_and_model_text_or_offsets_cannot_bypass(citation):
    corpus, task, pack = setup_pack()
    catalog = build_citation_catalog(pack, corpus, task)
    raw = answer_with_ids([catalog["spans"][0]["span_id"]])
    citation = {k: catalog["spans"][0]["span_id"] if v == "CURRENT" else v for k, v in citation.items()}
    raw["claims"][0]["citations"] = [citation]
    with pytest.raises(ValueError):
        resolve_span_candidate(raw, catalog, pack, corpus, task)


def test_repeated_id_rejected_and_unseen_tail_not_catalogued():
    corpus, task, pack = setup_pack("期限为30日。未展示尾部999万元。", "期限为30日。…")
    catalog = build_citation_catalog(pack, corpus, task)
    assert all("999" not in s["quote"] for s in catalog["spans"])
    sid = catalog["spans"][0]["span_id"]
    with pytest.raises(ValueError):
        resolve_span_candidate(answer_with_ids([sid, sid]), catalog, pack, corpus, task)


def test_catalog_limits():
    corpus, task, pack = setup_pack()
    pack["windows"] *= 25
    with pytest.raises(ValueError, match="24"):
        build_citation_catalog(pack, corpus, task)
    pack["windows"] = pack["windows"][:2]
    with pytest.raises(ValueError, match="Unique"):
        build_citation_catalog(pack, corpus, task)
    corpus, task, pack = setup_pack("甲" * 28001)
    with pytest.raises(ValueError, match="excerpt"):
        build_citation_catalog(pack, corpus, task)
    corpus, task, pack = setup_pack("甲" * 28000)
    pack["windows"] = [{**pack["windows"][0], "window_number": n} for n in range(1, 5)]
    with pytest.raises(ValueError, match="budget"):
        build_citation_catalog(pack, corpus, task)


def test_engine_materializes_before_review_and_publishes_offsets(corpus):
    def inspect_review(context):
        assert "citation_spans" not in context
        citation = context["candidate"]["claims"][0]["citations"][0]
        assert set(citation) == {"window_number", "quote"}
        return review(context)
    runner, send = engine(corpus, [span_answer, inspect_review])
    runner.citation_mode = "span_id"
    result = runner.run()
    assert result["status"] == "answered", result["gaps"]
    assert len(send.calls) == 2
    citation = result["claims"][0]["citations"][0]
    bounds = citation["source_span"]
    assert corpus.chunks[citation["ref"]["chunk_id"]].text[bounds["start"]:bounds["end"]] == citation["quote"]
    assert result["span_candidate"]["claims"][0]["citations"] == [{"span_id": citation["span_id"]}]
    assert "Unicode" in render_core_answer(result)
    assert '"window_number":1,"quote"' not in SPAN_ANSWER_INSTRUCTIONS
    assert "每段quote为" not in SPAN_ANSWER_INSTRUCTIONS


@pytest.mark.parametrize("kind", ["invalid_id", "rejected_review", "source_change", "budget"])
def test_engine_failure_never_publishes(corpus, kind):
    def review_action(context):
        if kind == "source_change":
            corpus.chunks["a1"].text += "改变"
        value = review(context)
        if kind == "rejected_review":
            value.update(sufficient=False, gaps=["结论不支持"])
        return value
    first = (lambda _: answer_with_ids(["sp-invented"])) if kind == "invalid_id" else span_answer
    runner, send = engine(corpus, [first, review_action], budget=Budget(4, 1 if kind == "budget" else 4, 40000))
    runner.citation_mode = "span_id"
    result = runner.run()
    assert result["status"] == {"invalid_id": "validation_failed", "rejected_review": "needs_evidence",
                                "source_change": "validation_failed", "budget": "budget_exhausted"}[kind]
    assert not result["answer"] and not result["claims"]
    assert result["span_candidate"] is not None
    assert len(send.calls) == (1 if kind in {"invalid_id", "budget"} else 2)


@pytest.mark.parametrize("reuse_old_id", [False, True])
def test_refinement_rebuilds_catalog_and_keeps_history(reuse_old_id):
    old_ids = []
    def first(context):
        old_ids.extend(s["span_id"] for s in context["citation_spans"])
        return nonanswer("abstained")
    def second(context):
        ids = [s["span_id"] for s in context["citation_spans"]]
        assert not set(ids) & set(old_ids)
        return answer_with_ids(old_ids if reuse_old_id else ids)
    runner, send, _ = contract_setup([first, refine, second, good_review])
    runner.citation_mode = "span_id"
    result = runner.run()
    assert result["status"] == ("validation_failed" if reuse_old_id else "answered")
    assert result["round_history"][0]["span_candidate"]["decision"] == "abstained"
    assert result["round_history"][0]["citation_catalog"] != result["citation_catalog"]
    assert len(send.calls) == (3 if reuse_old_id else 4)
