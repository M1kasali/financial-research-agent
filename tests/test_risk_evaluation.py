import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from test_financial_workflow import HEADING, POLICY, TABLE, corpus_for

from financial_agent.contracts import Budget, SearchRequest, TaskRequest
from financial_agent.core_research import CoreResearchEngine
from financial_agent.evaluation import EvalCase, EvalSuite, evaluate
from financial_agent.financial_workflow import FinancialQuestion
from financial_agent.research_tools import stable_id
from financial_agent.retrieval import ScopedRetriever
from financial_agent.risk_evaluation import build_risk_suite, diagnose_core_windows, validate_risk_packet
from financial_agent.simulation import ScriptedSend, fake_client


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import dotenv
    monkeypatch.setattr("socket.socket", lambda *a, **k: pytest.fail("No network in risk evaluation tests"))
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: pytest.fail("No credential access"))


def setup(corpus):
    parent = EvalSuite("parent", (EvalCase("old", "营业收入", (corpus.documents["a"].version_id,),
                                          "financial_reports", "extraction", "candidate"),))
    spec = {"suite_hash": parent.fingerprint, "anchors": {
        "REFERENCE_ONLY_SENTINEL": {"chunk_id": "a1", "quote": "2025年营业收入120万元"}}}
    recipe = {"schema_version": 1, "name": "risks", "partition": "candidate", "cases": [{
        "case_id": "new", "parent_case_id": "old", "document_ids": ["a"], "query": "核查2025年营业收入",
        "task_kind": "extraction", "risks": ["unit_binding"], "anchor_aliases": ["REFERENCE_ONLY_SENTINEL"],
        "review_questions": ["REVIEW_ONLY_SENTINEL"]}]}
    return recipe, parent, spec


def test_build_freezes_pending_not_gold_without_mutating_inputs(corpus):
    recipe, parent, spec = setup(corpus)
    before = deepcopy((recipe, spec))
    suite, packet, labels = build_risk_suite(recipe, parent, spec, corpus)
    assert (recipe, spec) == before
    assert packet["approved_gold_count"] == 0 and packet["reviews"][0]["absence_proven"] is False
    assert labels["labels"]["new"] == {"status": "pending", "expected": None}
    report = evaluate(suite, labels, {"suite_hash": suite.fingerprint, "mode": "offline_retrieval",
                                      "system_id": "no_model", "settings": {}, "predictions": []}, corpus)
    assert report["summary"]["scorable_count"] == 0
    assert report["summary"]["exact_field_accuracy"] is None
    validate_risk_packet(suite, packet, corpus)


@pytest.mark.parametrize("change", [lambda r: r.update(partition="reviewed_holdout"),
                                    lambda r: r.update(expected="ANSWER"),
                                    lambda r: r["cases"][0].update(expected="ANSWER"),
                                    lambda r: r["cases"].append(deepcopy(r["cases"][0])),
                                    lambda r: r["cases"][0].update(case_id="old"),
                                    lambda r: r["cases"][0].update(document_ids=["a", "a"])])
def test_invalid_or_promoted_recipe_is_rejected(corpus, change):
    recipe, parent, spec = setup(corpus)
    change(recipe)
    with pytest.raises(ValueError):
        build_risk_suite(recipe, parent, spec, corpus)


def test_scope_expansion_and_forged_quote_rejected(corpus):
    recipe, parent, spec = setup(corpus)
    spec["anchors"]["REFERENCE_ONLY_SENTINEL"]["quote"] = "FORGED"
    with pytest.raises(ValueError):
        build_risk_suite(recipe, parent, spec, corpus)
    recipe, parent, spec = setup(corpus)
    other = next(d for d in corpus.documents if d != "a")
    recipe["cases"][0]["document_ids"] = [other]
    with pytest.raises(PermissionError):
        build_risk_suite(recipe, parent, spec, corpus)


def test_packet_change_and_self_approved_review_rejected(corpus):
    recipe, parent, spec = setup(corpus)
    suite, packet, _ = build_risk_suite(recipe, parent, spec, corpus)
    packet["reviews"][0]["reviewer"] = "FAKE_HUMAN"
    with pytest.raises(ValueError):
        validate_risk_packet(suite, packet, corpus)
    packet["packet_hash"] = stable_id("risk-packet-", {k: v for k, v in packet.items() if k != "packet_hash"})
    with pytest.raises(ValueError):
        validate_risk_packet(suite, packet, corpus)


def test_only_public_question_and_scope_enter_evidence_generation(corpus):
    recipe, parent, spec = setup(corpus)
    suite, packet, _ = build_risk_suite(recipe, parent, spec, corpus)
    real = ScopedRetriever(corpus, tokenizer_mode="char")
    calls = []
    def search(task, request):
        public = json.dumps({"task": asdict(task), "request": asdict(request)}, ensure_ascii=False)
        assert "REFERENCE_ONLY_SENTINEL" not in public and "REVIEW_ONLY_SENTINEL" not in public
        assert "120万元" not in public
        calls.append(request)
        return real.search(task, request)
    report = diagnose_core_windows(suite, packet, corpus, SimpleNamespace(search=search))
    assert len(calls) == 1 and report["summary"]["visible_navigation_quotes"] == 1
    assert report["answer_accuracy"] is None and report["real_api_calls"] == 0


def test_source_mutation_fails_before_retrieval(corpus):
    recipe, parent, spec = setup(corpus)
    suite, packet, _ = build_risk_suite(recipe, parent, spec, corpus)
    corpus.chunks["a1"].text += " CHANGED"
    with pytest.raises(ValueError):
        diagnose_core_windows(suite, packet, corpus, SimpleNamespace(search=lambda *a: pytest.fail("Not safe")))


@pytest.mark.parametrize("selected", [False, True])
def test_diagnostic_distinguishes_candidate_drop_from_excerpt_omission(corpus, monkeypatch, selected):
    recipe, parent, spec = setup(corpus)
    suite, packet, _ = build_risk_suite(recipe, parent, spec, corpus)
    def controlled_pack(corpus, retriever, task):
        retriever.search(task, SearchRequest(task.query))
        # Deliberately incomplete fake pack to test diagnostic accounting, not compressor quality.
        windows = [{"ref": asdict(corpus.refs["a1"]), "excerpt_text": corpus.chunks["a1"].text[:2]}] if selected else []
        return {"windows": windows}
    monkeypatch.setattr("financial_agent.risk_evaluation.prepare_core_evidence", controlled_pack)
    report = diagnose_core_windows(suite, packet, corpus, ScopedRetriever(corpus, tokenizer_mode="char"))
    assert report["summary"]["retrieved_navigation_chunks"] == 1
    assert report["summary"]["retrieved_but_not_selected"] == int(not selected)
    assert report["summary"]["selected_but_quote_omitted"] == int(selected)


@pytest.mark.parametrize("scenario,reason", [("missing_unit", "no_accounting_declaration"),
                                            ("unit_conflict", "statement_binding_rejected"),
                                            ("value_conflict", "conflicting_statement_values"),
                                            ("missing_period", "no_supported_statement")])
def test_synthetic_risks_reach_real_financial_guard_with_scripted_routing(scenario, reason):
    if scenario == "missing_unit":
        corpus = corpus_for(policy=HEADING + "会计政策未给币种和单位。")
    elif scenario == "unit_conflict":
        corpus = corpus_for(extra=("unit_conflict", 41, POLICY.replace("千元", "万元")))
    elif scenario == "value_conflict":
        corpus = corpus_for(extra=("value_conflict", 11, TABLE.replace("120,000", "150,000")))
    else:
        corpus = corpus_for()
    year = 2026 if scenario == "missing_period" else 2025
    question = FinancialQuestion(corpus.documents["d"].version_id, year, "consolidated", "growth_rate", "营业收入")
    task = TaskRequest(question.query, (question.document_version_id,), budget=Budget(16, 8, 200_000))
    send = ScriptedSend([
        {"decision": "needs_financial_tools", "claims": [], "gaps": ["模拟路由，不是模型质量"], "option_judgments": []},
        {"kind": "financial", "spec": asdict(question)},
    ])
    result = CoreResearchEngine(corpus, ScopedRetriever(corpus, tokenizer_mode="char"), task, fake_client(send),
                                execution_label="SYNTHETIC_DATA_SCRIPTED_MODEL_REAL_TOOLS").run()
    assert result["status"] != "answered" and result["answer"] == "" and result["claims"] == []
    assert result["financial_result"]["reason_code"] == reason
    assert result["financial_result"]["calculation"] is None and len(send.calls) == 2
