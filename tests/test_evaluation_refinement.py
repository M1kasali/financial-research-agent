from copy import deepcopy

import pytest
from agent.schemas import Chunk

from financial_agent.contracts import EvidenceHit
from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalCase, EvalSuite
from financial_agent.evaluation_refinement import refine_candidates
from financial_agent.reference_drafts import compile_drafts
from financial_agent.retrieval_diagnostics import diagnose_retrieval, render_diagnostic


@pytest.fixture
def source():
    corpus = Corpus([Chunk("c1", "d1", "insurance", 1, "", "", "保险人及时通知补充材料。"),
                     Chunk("c2", "d2", "insurance", 1, "", "", "第二份文件"),
                     Chunk("c3", "outside", "insurance", 1, "", "", "范围外文件")])
    suite = EvalSuite("v1", (EvalCase("old-1", "哪些材料？", tuple(
        corpus.documents[d].version_id for d in ("d1", "d2")), "insurance", "extraction", "candidate"),))
    spec = {"schema_version": 1, "suite_hash": suite.fingerprint, "provenance": "draft",
            "anchors": {"a": {"chunk_id": "c1", "quote": "及时通知"}},
            "drafts": [{"case_id": "old-1", "status": "assistant_draft", "reviewer": None,
                        "coverage": "partial_needs_scope_review",
                        "claims": [{"text": "需通知补充", "evidence": ["a"]}],
                        "gaps": ["只读了部分条文"], "rubric": ["正确通知"]}]}
    recipe = {"schema_version": 1, "parent_suite_hash": suite.fingerprint,
              "new_suite_name": "v2", "id_prefix": "v2-", "provenance": "development draft",
              "changes": [{"case_id": "old-1", "query": "保险人如何通知补充材料？", "reason": "限定通知",
                           "document_ids": ["d1"], "draft": {"coverage": "answerable_draft"}}]}
    return corpus, suite, spec, recipe


def test_revision_is_immutable_pending_and_auditable(source):
    corpus, suite, spec, recipe = source
    original_spec, original_recipe = deepcopy(spec), deepcopy(recipe)
    new, new_spec, packet, labels, manifest = refine_candidates(suite, spec, recipe, corpus)
    assert spec == original_spec and recipe == original_recipe
    assert suite.cases[0].case_id == "old-1"
    assert new.fingerprint != suite.fingerprint
    assert new.cases[0].case_id == "v2-001"
    assert len(new.cases[0].document_version_ids) == 1
    assert new.cases[0].partition == "candidate"
    assert new_spec["suite_hash"] == new.fingerprint
    assert packet["approved_gold_count"] == 0
    assert labels["labels"]["v2-001"]["expected"] is None
    assert labels["labels"]["v2-001"]["status"] == "pending"
    assert not manifest["comparable_to_parent_scores"]
    assert manifest["lineage"][0]["scope_changed"]


@pytest.mark.parametrize("change", ["parent", "duplicate", "unknown", "expansion", "empty_scope",
                                   "reviewer", "identity", "anchor_overwrite", "name", "reason"])
def test_revision_rejects_bad_inputs(source, change):
    corpus, suite, spec, recipe = source
    update = recipe["changes"][0]
    if change == "parent":
        recipe["parent_suite_hash"] = "wrong"
    elif change == "duplicate":
        recipe["changes"].append(deepcopy(update))
    elif change == "unknown":
        update["case_id"] = "unknown"
    elif change == "expansion":
        update["document_ids"] = ["outside"]
    elif change == "empty_scope":
        update["document_ids"] = []
    elif change == "reviewer":
        update["draft"]["reviewer"] = "auto"
    elif change == "identity":
        update["draft"]["case_id"] = "other"
    elif change == "anchor_overwrite":
        recipe["added_anchors"] = {"a": {"chunk_id": "c2", "quote": "第二份文件"}}
    elif change == "name":
        recipe["new_suite_name"] = "v1"
    elif change == "reason":
        update["reason"] = ""
    with pytest.raises(ValueError):
        refine_candidates(suite, spec, recipe, corpus)


def test_narrow_scope_must_still_support_reference(source):
    corpus, suite, spec, recipe = source
    recipe["changes"][0]["document_ids"] = ["d2"]
    with pytest.raises(PermissionError):
        refine_candidates(suite, spec, recipe, corpus)


class FakeRetriever:
    tokenizer_mode = "fixture"

    def __init__(self, corpus, chunk_ids):
        self.corpus = corpus
        self.chunk_ids = chunk_ids
        self.calls = []

    def search(self, task, request):
        self.calls.append((task, request))
        return [EvidenceHit(self.corpus.refs[c], self.corpus.chunks[c].text, 1.0, "")
                for c in self.chunk_ids]


def test_diagnostic_uses_public_query_only_and_not_accuracy(source):
    corpus, suite, spec, _ = source
    packet = compile_drafts(spec, suite, corpus)
    retriever = FakeRetriever(corpus, ["c2", "c1"])
    report = diagnose_retrieval(suite, packet, retriever, cutoffs=(1, 2))
    assert report["summary"]["1"]["matched_anchor_chunks"] == 0
    assert report["summary"]["2"]["matched_anchor_chunks"] == 1
    assert report["answer_accuracy"] is None
    assert "Top-2" in render_diagnostic(report)
    assert not report["quality_claim_permitted"]
    assert len(retriever.calls) == 1
    task, request = retriever.calls[0]
    assert task.query == request.query == suite.cases[0].query
    assert not hasattr(task, "claims") and not hasattr(request, "evidence")


def test_diagnostic_missing_anchors_not_missing_answer(source):
    corpus, suite, spec, _ = source
    report = diagnose_retrieval(suite, compile_drafts(spec, suite, corpus), FakeRetriever(corpus, []))
    assert report["cases"][0]["metrics"]["5"]["missing_anchor_chunks"] == ["c1"]
    assert "不是准确率" in render_diagnostic(report)


def test_diagnostic_rejects_scope_leak(source):
    corpus, suite, spec, _ = source
    with pytest.raises(PermissionError):
        diagnose_retrieval(suite, compile_drafts(spec, suite, corpus), FakeRetriever(corpus, ["c3"]))


@pytest.mark.parametrize("cutoffs", [(), (0,), (True,), (101,), (5, 5)])
def test_diagnostic_rejects_cutoffs(source, cutoffs):
    corpus, suite, spec, _ = source
    with pytest.raises(ValueError):
        diagnose_retrieval(suite, compile_drafts(spec, suite, corpus), FakeRetriever(corpus, []),
                           cutoffs=cutoffs)
