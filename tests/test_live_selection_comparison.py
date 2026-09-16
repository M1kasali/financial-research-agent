import importlib
from dataclasses import replace
from pathlib import Path

import pytest

from financial_agent.contracts import Budget, EvidenceHit, SearchRequest, TaskRequest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_live_selection_comparison")


def test_dry_plan_does_not_load_secrets_cases_or_network(runner, monkeypatch, capsys):
    def forbidden(*a, **k):
        pytest.fail("Dry plan may not access data, credentials or network")
    monkeypatch.setattr(runner, "load_cases", forbidden)
    monkeypatch.setattr(runner, "dotenv_values", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    runner.main([])
    assert "No API call" in capsys.readouterr().out
    assert len(runner.CASE_SOURCES) * len(runner.STRATEGIES) * runner.RUN_CAP_UNITS == 10_000_000


def test_frozen_candidates_are_identical_and_not_shared_mutable_list(runner, corpus):
    task = TaskRequest("营业收入", (corpus.documents["a"].version_id,), budget=Budget(1, 2, 100_000))
    hits = [EvidenceHit(corpus.refs["a1"], corpus.chunks["a1"].text, 1, "")]
    frozen = runner.FrozenCandidates(task, hits)
    hits.clear()
    first = frozen.search(task, SearchRequest(task.query, top_k=40))
    second = frozen.search(replace(task, task_id="second"), SearchRequest(task.query, top_k=40))
    assert first == second and len(first) == 1 and first is not second
    first.clear()
    assert len(frozen.search(task, SearchRequest(task.query, top_k=40))) == 1


@pytest.mark.parametrize("change", ["task_query", "task_scope", "request_query", "request_scope", "top_k"])
def test_frozen_candidate_contract_rejects_scope_or_query_change(runner, corpus, change):
    task = TaskRequest("营业收入", (corpus.documents["a"].version_id,))
    frozen = runner.FrozenCandidates(task, [])
    request = SearchRequest(task.query, top_k=40)
    if change == "task_query":
        task = replace(task, query="changed")
    elif change == "task_scope":
        task = replace(task, document_version_ids=("outside",))
    elif change == "request_query":
        request = replace(request, query="changed")
    elif change == "request_scope":
        request = replace(request, document_version_ids=("outside",))
    else:
        request = replace(request, top_k=100)
    with pytest.raises(ValueError):
        frozen.search(task, request)


@pytest.mark.parametrize("quote,source,verbatim,line_break_match", [
    ("情形复杂的,在30 日内作出核定。", "情形复杂的,在30 日内\n\n作出核定。", False, True),
    ("营业收入 12 0", "营业收入 1 20", False, False),
    ("净利润100", "净利润100", True, True),
])
def test_quote_diagnostic_never_grants_acceptance(runner, quote, source, verbatim, line_break_match):
    audit = importlib.import_module("audit_live_selection")
    result = audit.quote_diagnostic(quote, source)
    assert result["verbatim_match"] is verbatim
    assert result["match_after_removing_source_line_breaks_only"] is line_break_match
    assert result["normalization_used_for_acceptance"] is False
