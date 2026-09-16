from dataclasses import asdict
from types import SimpleNamespace

import pytest
import requests
from agent.schemas import Chunk
from test_financial_workflow import corpus_for

from financial_agent.contracts import Budget, EvidenceHit, TaskRequest
from financial_agent.core_answer import render_core_answer
from financial_agent.core_research import CoreResearchEngine, guarded_financial_spec
from financial_agent.corpus import Corpus
from financial_agent.financial_workflow import FinancialQuestion
from financial_agent.model import ModelConfig
from financial_agent.retrieval import ScopedRetriever
from financial_agent.simulation import ScriptedSend, fake_client


def nonanswer(decision="needs_calculation"):
    return {"decision": decision, "claims": [], "gaps": ["需要补查或工具"], "option_judgments": []}


def finance_setup(*, scope="consolidated", operation="growth_rate", metric="营业收入", denominator=None,
                  budget=Budget(16, 8, 200_000), query=None, review=None, spec_change=None):
    corpus = corpus_for()
    question = FinancialQuestion(corpus.documents["d"].version_id, 2025, scope, operation, metric, denominator)
    spec = asdict(question)
    if spec_change:
        spec_change(spec)
    entries = [nonanswer(), {"kind": "financial", "spec": spec}, review or {
        "request_matches_spec": True, "covers_request": True, "gaps": []}]
    send = ScriptedSend(entries)
    task = TaskRequest(query or question.query, (question.document_version_id,), budget=budget)
    runner = CoreResearchEngine(corpus, ScopedRetriever(corpus, tokenizer_mode="char"), task,
                                fake_client(send), execution_label="simulated_protocol_test")
    return runner, send


def test_growth_uses_real_deterministic_tools_and_shared_budget():
    runner, send = finance_setup()
    result = runner.run()
    assert result["status"] == "answered", result["gaps"]
    assert "20.00" in result["answer"] and result["financial_result"]["calculation"]["display_value"] == "20.00"
    assert result["financial_result"]["tool_calls"] == 6 and result["usage"]["tool_calls"] == 7
    assert result["usage"]["model_attempts"] == 3
    assert [c["stage"] for c in send.calls] == ["answer", "financial_spec", "financial_review"]
    assert {c["ref"]["physical_page"] for c in result["claims"][0]["citations"]} == {10, 40}
    assert "20.00" in render_core_answer(result)
    with pytest.raises(ValueError):
        runner.run()


def test_parent_extraction_and_cashflow_ratio():
    runner, _ = finance_setup(scope="parent", operation="extract", metric="净利润")
    result = runner.run()
    assert result["status"] == "answered" and "4,000" in result["answer"]
    assert "2024年度" not in result["answer"]
    runner, _ = finance_setup(operation="ratio", metric="经营活动产生的现金流量净额", denominator="净利润")
    result = runner.run()
    assert result["status"] == "answered" and "2.00" in result["answer"]
    assert result["usage"]["tool_calls"] == 10


@pytest.mark.parametrize("change", [lambda s: s.update(document_version_id="outside"),
                                   lambda s: s.update(scope="parent"), lambda s: s.update(year=2024),
                                   lambda s: s.update(metric="净利润"), lambda s: s.update(operation="extract"),
                                   lambda s: s.update(page=10)])
def test_financial_parameter_mismatch_never_executes_tools(change):
    runner, send = finance_setup(spec_change=change)
    result = runner.run()
    assert result["status"] == "clarification" and result["financial_result"] is None
    assert result["usage"]["tool_calls"] == 1 and len(send.calls) == 2


@pytest.mark.parametrize("query", ["核查2025年营业收入同比", "核查合并营业收入同比", "核查2025年合并与母公司营业收入同比"])
def test_missing_or_ambiguous_scope_year_needs_user(query):
    runner, _ = finance_setup(query=query)
    assert runner.run()["status"] == "clarification"


def test_compound_request_not_declared_complete_after_partial_calculation():
    runner, _ = finance_setup(query="计算2025年合并营业收入同比并解释增长原因", review={
        "request_matches_spec": True, "covers_request": False, "gaps": ["没有解释增长原因"]})
    result = runner.run()
    assert result["status"] == "needs_evidence" and result["answer"] == ""
    assert result["financial_result"]["status"] == "completed"


def test_shared_budget_stops_inside_subworkflow():
    runner, send = finance_setup(budget=Budget(4, 8, 200_000))
    result = runner.run()
    assert result["status"] == "budget_exhausted" and result["answer"] == ""
    assert result["usage"]["tool_calls"] == 4 and result["financial_result"]["tool_calls"] == 3
    assert len(send.calls) == 2


def test_model_budget_stops_final_request_review():
    runner, _ = finance_setup(budget=Budget(16, 2, 200_000))
    result = runner.run()
    assert result["status"] == "budget_exhausted" and not result["answer"]
    assert result["financial_result"]["status"] == "completed"


def test_source_mutation_during_financial_review_never_publishes():
    runner, send = finance_setup()
    def mutate(_):
        runner.corpus.chunks["policy"].text += "修改"
        return {"request_matches_spec": True, "covers_request": True, "gaps": []}
    send.entries = iter([nonanswer(), {"kind": "financial", "spec": asdict(FinancialQuestion(
        runner.task.document_version_ids[0], 2025, "consolidated", "growth_rate", "营业收入"))}, mutate])
    result = runner.run()
    assert result["status"] == "validation_failed" and not result["claims"]


def contract_setup(entries, *, max_refinements=1, search=None, budget=Budget(8, 12, 200_000)):
    corpus = Corpus([Chunk("a1", "a", "financial_contracts", 1, "", "", "解除合同需要提前30日通知。"),
                     Chunk("a2", "a", "financial_contracts", 2, "", "", "重大违约解除不适用提前30日通知要求。"),
                     Chunk("x", "x", "financial_contracts", 1, "", "", "外部合同数据。")])
    task = TaskRequest("核查解除合同的通知条件和例外。", (corpus.documents["a"].version_id,), budget=budget)
    def hits(cid):
        ref = corpus.refs[cid]
        return [EvidenceHit(ref, corpus.chunks[cid].text, 1.0, "")]
    calls = []
    def retrieve(task, request):
        calls.append(request)
        return search(corpus, task, request) if search else hits("a1" if len(calls) == 1 else "a2")
    send = ScriptedSend(entries)
    runner = CoreResearchEngine(corpus, SimpleNamespace(search=retrieve), task, fake_client(send),
                                max_refinements=max_refinements, execution_label="simulated_protocol_test")
    return runner, send, calls


def refine(context):
    return {"kind": "search", "query": "重大违约解除 通知例外", "document_version_ids": context["document_version_ids"]}


def complete_answer(context):
    assert len(context["windows"]) == 2  # Initial rule MUST survive addition of its exception.
    return {"decision": "answered", "claims": [{"claim_id": "c1", "kind": "interpretation",
            "text": "通常需提前30日通知；重大违约解除不适用该要求。",
            "citations": [{"window_number": w["window_number"], "quote": w["excerpt_text"]}
                          for w in context["windows"]]}], "option_judgments": [], "gaps": []}


def good_review(_):
    return {"claim_verdicts": [{"claim_id": "c1", "supported": True, "reason": "模拟通过"}],
            "sufficient": True, "gaps": []}


def test_gap_directed_refinement_keeps_prior_evidence_and_question():
    runner, send, queries = contract_setup([nonanswer("abstained"), refine, complete_answer, good_review])
    result = runner.run()
    assert result["status"] == "answered" and result["refinement_rounds"] == 1
    assert result["usage"]["model_attempts"] == 4 and result["usage"]["tool_calls"] == 2
    assert len(queries) == 2 and queries[1].query == "重大违约解除 通知例外"
    assert send.calls[2]["query"] == runner.task.query
    assert result["round_history"][0]["status"] == "abstained"
    assert len(result["evidence_pack"]["evidence_text"]) == result["evidence_pack"]["rendered_chars"]


@pytest.mark.parametrize("versions", [[], ["outside"], [True]])
def test_refine_scope_escape_rejected_before_search(versions):
    runner, _, queries = contract_setup([nonanswer("abstained"), {
        "kind": "search", "query": "通知例外", "document_version_ids": versions}])
    result = runner.run()
    assert result["status"] == "validation_failed" and len(queries) == 1 and not result["answer"]


def test_no_new_evidence_stops_without_reasking_answer():
    def same(corpus, *_):
        return [EvidenceHit(corpus.refs["a1"], corpus.chunks["a1"].text, 1, "")]
    runner, send, _ = contract_setup([nonanswer("abstained"), refine], search=same)
    result = runner.run()
    assert result["status"] == "needs_evidence" and len(send.calls) == 2
    assert "No new visible" in result["gaps"][0]


def test_repeated_query_and_disabled_continuation_are_bounded():
    runner, send, queries = contract_setup([nonanswer("abstained"), refine, nonanswer("abstained"), refine], max_refinements=2)
    result = runner.run()
    assert result["status"] == "needs_evidence" and len(queries) == 2 and len(send.calls) == 4
    runner, send, _ = contract_setup([nonanswer("abstained")], max_refinements=0)
    assert runner.run()["status"] == "abstained" and len(send.calls) == 1


def test_refinement_timeout_does_not_retry_or_publish():
    runner, _, _ = contract_setup([nonanswer("abstained")])
    original = runner.client._send
    def send(payload):
        if runner.client.total_attempts == 2:
            raise requests.Timeout()
        return original(payload)
    runner.client._send = send
    assert runner.run()["status"] == "needs_attention"


def test_default_disabled_no_followup_requests():
    runner, send, _ = contract_setup([])
    runner.client.config = ModelConfig("qwen3.7-flash", enabled=False)
    assert runner.run()["status"] == "needs_model" and send.calls == []


def test_finance_spec_cannot_use_attributable_profit_as_parent_or_reverse_ratio():
    task = TaskRequest("核查2025年合并归属于母公司所有者的净利润", ("v",))
    spec = asdict(FinancialQuestion("v", 2025, "consolidated", "extract", "净利润"))
    with pytest.raises(ValueError):
        guarded_financial_spec(spec, task)
    task = TaskRequest("计算2025年合并经营活动产生的现金流量净额/净利润比值", ("v",))
    spec = asdict(FinancialQuestion("v", 2025, "consolidated", "ratio", "净利润", "经营活动产生的现金流量净额"))
    with pytest.raises(ValueError):
        guarded_financial_spec(spec, task)
