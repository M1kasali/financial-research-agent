import copy
import json
from dataclasses import asdict, replace

import pytest

from financial_agent.benchmark import run_dynamic, run_fixed_rag
from financial_agent.contracts import Budget
from financial_agent.evaluation import EvalCase, EvalSuite, compare_reports, evaluate, prediction_hash
from financial_agent.retrieval import ScopedRetriever
from financial_agent.simulation import FakeResponse, fake_client, finance_corpus, finance_decision


@pytest.fixture
def bundle():
    corpus = finance_corpus()
    case = EvalCase(
        "q1",
        "甲公司营业收入同比变化是多少？",
        tuple(corpus.by_version),
        "financial_reports",
        "calculation",
        "fixture",
    )
    suite = EvalSuite("scorer-fixture", (case,))
    labels = {
        "suite_hash": suite.fingerprint,
        "labels": {
            "q1": {
                "status": "synthetic_fixture",
                "provenance": "unit_test_fixture",
                "expected": {"kind": "number", "value": "20", "unit": "%", "tolerance": "0.01"},
            }
        },
    }
    pred = {
        "case_id": "q1",
        "decision": "answered",
        "answer": "同比增长20%。",
        "structured": {"value": "20.00", "unit": "%"},
        "citations": [asdict(corpus.refs["row"])],
    }
    run = {
        "system_id": "fixture",
        "mode": "simulation",
        "suite_hash": suite.fingerprint,
        "settings": {"budget": "same"},
        "predictions": [pred],
    }
    return corpus, suite, labels, run


def test_numeric_and_citation_metrics_are_distinct(bundle):
    corpus, suite, labels, run = bundle
    report = evaluate(suite, labels, run, corpus)
    assert report["summary"]["exact_field_accuracy"] == 1
    assert report["summary"]["citation_integrity_rate"] == 1
    assert report["summary"]["human_answer_accuracy"] is None
    assert report["summary"]["human_supported_answer_rate"] is None
    assert report["known_reported_tokens"] is None
    assert report["mean_elapsed_seconds"] is None


@pytest.mark.parametrize(
    "structured",
    [
        {"value": "0.2", "unit": "%"},
        {"value": "20", "unit": "CNY"},
        {"value": "NaN", "unit": "%"},
        {"value": True, "unit": "%"},
        {"value": "20.02", "unit": "%"},
        {"value": "1e99999", "unit": "%"},
        {},
    ],
)
def test_wrong_number_unit_or_missing_fields_not_correct(bundle, structured):
    corpus, suite, labels, run = bundle
    run["predictions"][0]["structured"] = structured
    assert evaluate(suite, labels, run, corpus)["summary"]["exact_field_accuracy"] == 0


def test_missing_predictions_count_as_wrong_on_eligible_cases(bundle):
    corpus, suite, labels, run = bundle
    run["predictions"] = []
    summary = evaluate(suite, labels, run, corpus)["summary"]
    assert summary["scorable_count"] == summary["missing_predictions"] == 1
    assert summary["exact_field_accuracy"] == 0
    assert summary["citation_integrity_rate"] is None


def test_pending_labels_never_create_accuracy(bundle):
    corpus, suite, labels, run = bundle
    labels["labels"]["q1"] = {"status": "pending", "expected": None}
    report = evaluate(suite, labels, run, corpus)
    assert report["summary"]["scorable_count"] == 0
    assert report["summary"]["exact_field_accuracy"] is None


def test_pending_label_cannot_hide_an_answer(bundle):
    corpus, suite, labels, run = bundle
    labels["labels"]["q1"]["status"] = "pending"
    with pytest.raises(ValueError, match="Pending"):
        evaluate(suite, labels, run, corpus)


@pytest.mark.parametrize(
    "value,expected", [("BA", 1), ("AB", 1), ("Because A", 0), ("AAB", 0), ("ABC", 0), ("", 0)]
)
def test_strict_selection_parser(bundle, value, expected):
    corpus, suite, labels, run = bundle
    labels["labels"]["q1"]["expected"] = {"kind": "selection", "value": "AB"}
    run["predictions"][0]["structured"] = {"selection": value}
    assert evaluate(suite, labels, run, corpus)["summary"]["exact_field_accuracy"] == expected


def test_error_is_not_a_correct_refusal(bundle):
    corpus, suite, labels, run = bundle
    labels["labels"]["q1"]["expected"] = {"kind": "decision", "value": "abstained"}
    run["predictions"][0].update(decision="error", answer="")
    assert evaluate(suite, labels, run, corpus)["summary"]["exact_field_accuracy"] == 0
    run["predictions"][0].update(decision="abstained", answer="资料未披露，无法确认。")
    assert evaluate(suite, labels, run, corpus)["summary"]["exact_field_accuracy"] == 1


def test_forged_and_duplicate_citations_not_valid(bundle):
    corpus, suite, labels, run = bundle
    row = run["predictions"][0]
    good = row["citations"][0]
    row["citations"] += [
        copy.deepcopy(good),
        {**good, "physical_page": True},
        {**good, "version_id": "outside"},
    ]
    assert evaluate(suite, labels, run, corpus)["summary"]["citation_integrity_rate"] == 0.25


def test_legal_citation_does_not_auto_approve_wrong_claim(bundle):
    corpus, suite, labels, run = bundle
    run["predictions"][0]["answer"] = "营收下降80%，引用其实不支持这句话。"
    report = evaluate(suite, labels, run, corpus)
    assert report["summary"]["exact_field_accuracy"] == 1
    assert report["summary"]["human_answer_accuracy"] is None


def test_human_review_binds_exact_output_and_rejects_self_judgement(bundle):
    corpus, suite, labels, run = bundle
    review = {
        "case_id": "q1",
        "prediction_hash": prediction_hash(run["predictions"][0]),
        "review_kind": "human",
        "reviewer": "test-fixture-reviewer",
        "answer_correct": True,
        "citation_supported": True,
        "complete": False,
    }
    assert evaluate(suite, labels, run, corpus, reviews=[review])["summary"]["human_completeness_rate"] == 0
    review["review_kind"] = "model"
    with pytest.raises(ValueError, match="human"):
        evaluate(suite, labels, run, corpus, reviews=[review])
    review["review_kind"] = "human"
    run["predictions"][0]["answer"] = "new answer"
    with pytest.raises(ValueError, match="stale"):
        evaluate(suite, labels, run, corpus, reviews=[review])


def test_suite_and_label_fingerprints_checked(bundle, tmp_path):
    corpus, suite, labels, run = bundle
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(suite.export()), encoding="utf-8")
    assert EvalSuite.load(path).fingerprint == suite.fingerprint
    payload = suite.export()
    payload["cases"][0]["query"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        EvalSuite.load(path)
    run["suite_hash"] = "old"
    with pytest.raises(ValueError, match="version"):
        evaluate(suite, labels, run, corpus)


@pytest.mark.parametrize("mutation", ["duplicate", "unknown"])
def test_predictions_are_one_per_case(bundle, mutation):
    corpus, suite, labels, run = bundle
    if mutation == "duplicate":
        run["predictions"].append(copy.deepcopy(run["predictions"][0]))
    else:
        run["predictions"][0]["case_id"] = "other"
    with pytest.raises(ValueError, match="prediction"):
        evaluate(suite, labels, run, corpus)


def test_simulation_comparison_never_claims_quality(bundle):
    corpus, suite, labels, run = bundle
    a = evaluate(suite, labels, run, corpus)
    b = {**a, "system_id": "other"}
    assert compare_reports(a, b)["quality_claim_permitted"] is False
    b["settings"] = {"budget": "different"}
    with pytest.raises(ValueError, match="settings"):
        compare_reports(a, b)


def test_public_task_has_no_labels(bundle):
    _, suite, _, _ = bundle
    public = asdict(suite.cases[0].task(Budget()))
    assert "expected" not in public and "labels" not in public


def test_adapters_use_same_scope_but_different_policy(bundle):
    corpus, suite, _, _ = bundle
    case = suite.cases[0]
    retriever = ScopedRetriever(corpus, tokenizer_mode="char")

    def fixed_send(payload):
        evidence = json.loads(payload["messages"][-1]["content"])["evidence"]["hits"]
        return FakeResponse(
            {
                "decision": "answered",
                "answer": "同比增长20%。",
                "evidence_ids": [evidence[0]["ref"]["evidence_id"]],
                "structured": {"value": "20", "unit": "%"},
            }
        )

    fixed = run_fixed_rag(case, corpus, retriever, fake_client(fixed_send), Budget(12, 24, 400_000))
    client = fake_client(lambda p: FakeResponse(finance_decision(json.loads(p["messages"][-1]["content"]))))
    dynamic, _ = run_dynamic(
        case, corpus, retriever, client, Budget(12, 24, 400_000), execution_label="simulated_model"
    )
    assert fixed["structured"]["value"] == "20" and dynamic["structured"]["value"] == "20.00"
    assert fixed["usage"]["model_attempts"] == 1 and dynamic["usage"]["model_attempts"] == 9
    assert fixed["usage"]["reported_tokens"] is None
    assert set(r["version_id"] for r in dynamic["citations"]) <= set(case.document_version_ids)


def test_synthetic_labels_cannot_be_used_as_real_gold(bundle):
    corpus, suite, labels, run = bundle
    suite = EvalSuite("real", (replace(suite.cases[0], partition="candidate"),))
    labels["suite_hash"] = run["suite_hash"] = suite.fingerprint
    with pytest.raises(ValueError, match="Synthetic"):
        evaluate(suite, labels, run, corpus)


def test_rubric_requires_review_not_exact_string_scoring(bundle):
    corpus, suite, labels, run = bundle
    labels["labels"]["q1"]["expected"] = {"kind": "rubric", "criteria": ["核对原始数值、年度和合并口径"]}
    result = evaluate(suite, labels, run, corpus)
    assert result["summary"]["exact_field_accuracy"] is None


def test_fixture_and_real_partitions_cannot_mix(bundle):
    _, suite, _, _ = bundle
    with pytest.raises(ValueError, match="mix"):
        EvalSuite("mixed", (suite.cases[0], replace(suite.cases[0], case_id="q2", partition="candidate")))


def test_invalid_reference_before_valid_reference_does_not_hide_valid_one(bundle):
    corpus, suite, labels, run = bundle
    good = run["predictions"][0]["citations"][0]
    run["predictions"][0]["citations"] = [{**good, "physical_page": True}, good]
    assert evaluate(suite, labels, run, corpus)["summary"]["citation_integrity_rate"] == 0.5


def test_unreviewed_holdout_refused(bundle):
    corpus, suite, labels, run = bundle
    suite = EvalSuite("holdout", (replace(suite.cases[0], partition="reviewed_holdout"),))
    labels = {"suite_hash": suite.fingerprint, "labels": {}}
    run["suite_hash"] = suite.fingerprint
    with pytest.raises(ValueError, match="Holdout"):
        evaluate(suite, labels, run, corpus)


@pytest.mark.parametrize(
    "usage", [{"reported_tokens": -1}, {"reported_tokens": True}, {"model_attempts": 1.5}]
)
def test_invalid_usage_is_rejected(bundle, usage):
    corpus, suite, labels, run = bundle
    run["predictions"][0]["usage"] = usage
    with pytest.raises(ValueError, match="usage"):
        evaluate(suite, labels, run, corpus)
