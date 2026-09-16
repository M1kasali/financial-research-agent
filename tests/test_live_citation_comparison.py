import importlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
from test_citation_spans import span_answer
from test_core_answer import candidate, review

from financial_agent.contracts import TaskRequest
from financial_agent.core_evidence import prepare_core_evidence
from financial_agent.retrieval import ScopedRetriever
from financial_agent.simulation import ScriptedSend, fake_client


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_live_citation_comparison")


def test_dry_plan_no_data_credentials_network(runner, monkeypatch, capsys):
    def forbidden(*a, **k):
        pytest.fail("Dry run may not load data or credentials")
    monkeypatch.setattr(runner.EvalSuite, "load", forbidden)
    monkeypatch.setattr(runner, "dotenv_values", forbidden)
    runner.main([])
    assert "No API call" in capsys.readouterr().out
    assert runner.RUN_CAP_UNITS * len(runner.MODES) == runner.BATCH_CAP_UNITS == 5_000_000


@pytest.mark.parametrize("mode,change", [("verbatim", False), ("span_id", False),
                                       ("verbatim", True), ("span_id", True)])
def test_fixed_windows_checked_before_call(runner, corpus, mode, change):
    task = TaskRequest("甲公司2025年营业收入是多少？", (corpus.documents["a"].version_id,))
    retriever = ScopedRetriever(corpus, tokenizer_mode="char")
    expected = prepare_core_evidence(corpus, retriever, task, selection_strategy="coverage")
    original = deepcopy(expected)
    if change:
        expected["windows"][0]["excerpt_text"] += "changed"
    send = ScriptedSend([candidate if mode == "verbatim" else span_answer, review])
    engine = runner.FixedWindowEngine(corpus, retriever, task, fake_client(send), expected_pack=expected,
                                     selection_strategy="coverage", citation_mode=mode)
    expected.clear()  # Engine stores an isolated copy.
    result = engine.run()
    assert result["status"] == ("validation_failed" if change else "answered")
    assert len(send.calls) == (0 if change else 2)
    assert result["evidence_pack"] == original
    assert runner.fingerprint(result["evidence_pack"]) == runner.fingerprint(original)


@pytest.mark.parametrize("tamper", [None, "offset", "quote", "claim", "answer", "raw_id"])
def test_saved_result_audit_checks_published_bindings(runner, corpus, tamper):
    audit = importlib.import_module("audit_live_citations")
    task = TaskRequest("甲公司2025年营业收入是多少？", (corpus.documents["a"].version_id,))
    retriever = ScopedRetriever(corpus, tokenizer_mode="char")
    pack = prepare_core_evidence(corpus, retriever, task, selection_strategy="coverage")
    send = ScriptedSend([span_answer, review])
    engine = runner.FixedWindowEngine(corpus, retriever, task, fake_client(send), expected_pack=pack,
                                     selection_strategy="coverage", citation_mode="span_id")
    result = engine.run()
    responses = [{"stage": "answer", "content": json.dumps(result["span_candidate"])},
                 {"stage": "review", "content": json.dumps(result["review"])}]
    citation = result["claims"][0]["citations"][0]
    if tamper == "offset":
        citation["source_span"]["start"] += 1
    elif tamper == "quote":
        citation["quote"] += "不在原文"
    elif tamper == "claim":
        result["claims"][0]["text"] += "额外结论"
    elif tamper == "answer":
        result["answer"] += "额外总结"
    elif tamper == "raw_id":
        result["span_candidate"]["claims"][0]["citations"][0]["span_id"] = "changed"
    if tamper:
        with pytest.raises(ValueError):
            audit.audit_run(result, asdict(task), pack, corpus, responses)
    else:
        checked = audit.audit_run(result, asdict(task), pack, corpus, responses)
        assert checked["candidate_protocol_valid"] and checked["model_review_accepted"]
        assert checked["published_source_spans_checked"] == 1 and not checked["semantic_gold"]
