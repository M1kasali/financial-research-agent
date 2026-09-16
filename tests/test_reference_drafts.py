from copy import deepcopy
from dataclasses import replace

import pytest
from agent.schemas import Chunk

from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalCase, EvalSuite, validate_label
from financial_agent.reference_drafts import compile_drafts, render_packet


@pytest.fixture
def sample():
    corpus = Corpus([Chunk("c1", "d1", "financial_reports", 1, "", "",
                           "2025收入120万元；2024收入100万元。"),
                     Chunk("c2", "d2", "financial_reports", 2, "", "", "其他300万元")])
    case = EvalCase("draft-001", "收入同比？", (corpus.documents["d1"].version_id,),
                    "financial_reports", "calculation", "candidate")
    suite = EvalSuite("test", (case,))
    spec = {"schema_version": 1, "suite_hash": suite.fingerprint, "provenance": "assistant draft",
            "anchors": {"a": {"chunk_id": "c1", "quote": "2025收入120万元"}},
            "drafts": [{"case_id": "draft-001", "status": "assistant_draft", "reviewer": None,
                        "coverage": "answerable_draft", "claims": [{"text": "增长20%", "evidence": ["a"]}],
                        "gaps": ["待人工核对口径"], "rubric": ["口径一致"],
                        "calculations": [{"name": "同比", "operation": "growth_percent",
                                          "a": {"raw": "120", "anchor": "a"},
                                          "b": {"raw": "100", "anchor": "a"}, "unit": "%",
                                          "context": "人民币万元", "context_evidence": ["a"]}],
                        "searches": [{"terms": ["2028"]}]}]}
    return spec, suite, corpus


def test_compile_is_separate_immutable_draft(sample):
    spec, suite, corpus = sample
    original = deepcopy(spec)
    packet = compile_drafts(spec, suite, corpus)
    assert spec == original
    assert packet["approved_gold_count"] == 0
    assert not packet["quality_claim_permitted"]
    draft = packet["drafts"][0]
    assert draft["calculations"][0]["computed_value"] == "20.0"
    assert not draft["calculations"][0]["semantic_context_verified"]
    evidence = draft["evidence"]["a"]
    assert evidence["full_chunk_text"][evidence["quote_start"]:evidence["quote_end"]] == evidence["quote"]
    assert draft["search_audit"][0]["hits"] == []
    assert not draft["search_audit"][0]["proves_absence"]
    assert "不是 B 榜隐藏答案" in render_packet(packet)
    with pytest.raises(ValueError, match="Untrusted gold"):
        validate_label(draft)


@pytest.mark.parametrize("change", ["suite", "duplicate", "unknown", "quote", "ambiguous", "scope",
                                   "status", "reviewer", "operand", "substring", "operation", "context"])
def test_reject_invalid_drafts(sample, change):
    spec, suite, corpus = sample
    draft = spec["drafts"][0]
    calc = draft["calculations"][0]
    if change == "suite":
        spec["suite_hash"] = "wrong"
    elif change == "duplicate":
        spec["drafts"].append(deepcopy(draft))
    elif change == "unknown":
        draft["case_id"] = "missing"
    elif change == "quote":
        spec["anchors"]["a"]["quote"] = "不存在"
    elif change == "ambiguous":
        spec["anchors"]["a"]["quote"] = "收入"
    elif change == "scope":
        spec["anchors"]["a"] = {"chunk_id": "c2", "quote": "其他300万元"}
    elif change == "status":
        draft["status"] = "human_reviewed"
    elif change == "reviewer":
        draft["reviewer"] = "pretend reviewer"
    elif change == "operand":
        calc["a"]["raw"] = "130"
    elif change == "substring":
        calc["a"]["raw"] = "12"
    elif change == "operation":
        calc["operation"] = "arbitrary_code"
    elif change == "context":
        calc["context_evidence"] = []
    with pytest.raises((ValueError, PermissionError)):
        compile_drafts(spec, suite, corpus)


def test_never_compile_historical_b(sample):
    spec, suite, corpus = sample
    suite = EvalSuite("B", (replace(suite.cases[0], partition="historical_replay"),))
    spec["suite_hash"] = suite.fingerprint
    with pytest.raises(ValueError, match="candidate"):
        compile_drafts(spec, suite, corpus)


def test_source_tampering_rejected(sample):
    spec, suite, corpus = sample
    corpus.chunks["c1"].text += "篡改"
    with pytest.raises(ValueError, match="changed"):
        compile_drafts(spec, suite, corpus)
