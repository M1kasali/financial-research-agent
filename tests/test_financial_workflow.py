import json
from dataclasses import asdict

import pytest
from agent.schemas import Chunk

from financial_agent.cli import main
from financial_agent.corpus import Corpus
from financial_agent.financial_workflow import (
    FinancialQuestion,
    FinancialStatementWorkflow,
    render_financial_report,
)
from financial_agent.retrieval import ScopedRetriever

HEADING = "甲股份有限公司2025 年年度报告全文\n"
TABLE = HEADING + "\n2、合并利润表\n附注七 2025 年 2024 年\n营业收入 45 120,000 100,000\n净利润 20,000 18,000\n"
PARENT = HEADING + "\n6、公司利润表\n附注十九 2025 年 2024 年\n营业收入 4 15,000 10,000\n净利润 4,000 3,000\n"
CASH = HEADING + "\n4、合并现金流量表\n附注七 2025 年 2024 年\n经营活动产生的现金流量净额 62 40,000 35,000\n"
POLICY = HEADING + "\n重要会计政策\n会计期间\n本集团会计年度采用公历年度,即每年自1 月1 日起至12 月31 日止。\n记账本位币\n本公司记账本位币和编制本财务报表所采用的货币均为人民币,除有特别说明外,均以人民币千元为单位表示。"


def corpus_for(*, policy=POLICY, extra=None):
    rows = [("income", 10, TABLE), ("parent", 14, PARENT), ("cash", 12, CASH),
            ("policy", 40, policy), ("decoy", 90, "某客户营业收入900,000，报告讨论合并利润表。")]
    if extra:
        rows.append(extra)
    chunks = [Chunk(cid, "d", "financial_reports", page, "", "", text) for cid, page, text in rows]
    chunks.append(Chunk("outside", "other", "financial_reports", 10, "", "", TABLE.replace("120,000", "990,000")))
    return Corpus(chunks)


def run_question(corpus, operation="growth_rate", metric="营业收入", scope="consolidated", denominator=None, budget=12):
    retriever = ScopedRetriever(corpus, tokenizer_mode="char")
    question = FinancialQuestion(corpus.documents["d"].version_id, 2025, scope, operation, metric, denominator)
    workflow = FinancialStatementWorkflow(corpus, retriever, question, max_tool_calls=budget)
    return workflow, workflow.run()


def test_question_to_growth_without_preloaded_evidence():
    workflow, result = run_question(corpus_for())
    assert result["status"] == "completed", result["reason"]
    assert result["calculation"]["display_value"] == "20.00"
    assert result["model_calls"] == 0
    assert result["tool_calls"] <= 12
    assert {r["doc_id"] for r in result["evidence"]} == {"d"}
    names = [e["tool"] for e in result["events"] if "tool" in e]
    assert names == ["search", "read", "expand_context", "read", "bind_statement", "calculate"]
    assert "20.00" in render_financial_report(result)
    for event in result["events"]:
        if event.get("tool") in {"search", "expand_context"}:
            assert "120,000" not in event["args"]["query"]
            assert "40" not in event["args"]["query"]
    with pytest.raises(ValueError, match="single-use"):
        workflow.run()


def test_ratio_reuses_policy_but_retrieves_two_statement_types():
    _, result = run_question(corpus_for(), "ratio", "经营活动产生的现金流量净额", denominator="净利润")
    assert result["status"] == "completed", result["reason"]
    assert result["calculation"]["display_value"] == "2.00"
    assert result["tool_calls"] == 9
    assert len(result["calculation"]["evidence_ids"]) == 3


def test_parent_request_does_not_use_consolidated_numbers():
    _, result = run_question(corpus_for(), "extract", "营业收入", scope="parent")
    assert result["status"] == "completed"
    assert result["selected_facts"][0]["raw_value"] == "15,000"
    assert all(f["scope"] == "parent" for f in result["selected_facts"])
    assert result["calculation"] is None


def test_missing_context_never_guesses_unit():
    _, result = run_question(corpus_for(policy=HEADING + "会计政策说明缺失。"))
    assert result["status"] == "needs_evidence"
    assert result["reason_code"] == "no_accounting_declaration"
    assert not result["selected_facts"] and result["calculation"] is None


def test_unsupported_statement_has_explicit_reason():
    corpus = Corpus([Chunk("unknown", "d", "financial_reports", 1, "", "", "收入资料但无报表标题")])
    _, result = run_question(corpus)
    assert result["reason_code"] == "no_supported_statement"
    assert result["model_calls"] == 0


def test_requested_year_not_silently_substituted():
    corpus = corpus_for()
    question = FinancialQuestion(corpus.documents["d"].version_id, 2026, "consolidated", "extract", "营业收入")
    result = FinancialStatementWorkflow(corpus, ScopedRetriever(corpus, tokenizer_mode="char"), question).run()
    assert result["reason_code"] == "no_supported_statement"


def test_conflicting_retrieved_statement_not_cherry_picked():
    corpus = corpus_for(extra=("conflict", 11, TABLE.replace("120,000", "150,000")))
    _, result = run_question(corpus)
    assert result["reason_code"] == "conflicting_statement_values"
    assert result["calculation"] is None


def test_conflicting_unit_declarations_not_cherry_picked():
    corpus = corpus_for(extra=("conflict_policy", 41, POLICY.replace("千元", "万元")))
    _, result = run_question(corpus)
    assert result["reason_code"] == "statement_binding_rejected"
    assert "Conflicting accounting units" in result["reason"]


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5])
def test_every_step_respects_shared_tool_budget(budget):
    _, result = run_question(corpus_for(), budget=budget)
    assert result["status"] == "partial" and result["reason_code"] == "tool_budget_exhausted"
    assert result["tool_calls"] == budget
    assert result["calculation"] is None


def test_changed_context_drops_stale_result():
    corpus = corpus_for()
    question = FinancialQuestion(corpus.documents["d"].version_id, 2025, "consolidated", "growth_rate", "营业收入")
    workflow = FinancialStatementWorkflow(corpus, ScopedRetriever(corpus, tokenizer_mode="char"), question)
    calculate = workflow.tools.calculate
    def mutate(args):
        result = calculate(args)
        corpus.chunks["policy"].text += "changed"
        return result
    workflow.tools.calculate = mutate
    result = workflow.run()
    assert result["reason_code"] == "source_integrity_failed"
    assert result["calculation"] is None and result["selected_facts"] == []


@pytest.mark.parametrize("scope,operation,metric,denominator", [
    ("unknown", "extract", "营业收入", None),
    ("consolidated", "eval", "营业收入", None),
    ("consolidated", "ratio", "营业收入", None),
    ("consolidated", "ratio", "营业收入", "营业收入"),
    ("consolidated", "extract", "营业收入", "净利润"),
    ("parent", "extract", "归属于母公司所有者的净利润", None),
])
def test_structured_question_rejects_ambiguous_or_unsupported_requests(scope, operation, metric, denominator):
    with pytest.raises(ValueError):
        FinancialQuestion("version", 2025, scope, operation, metric, denominator)


def test_cli_writes_new_report_and_trace_without_overwriting(tmp_path, capsys):
    corpus = corpus_for()
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in corpus.chunks.values()), encoding="utf-8")
    output = tmp_path / "report"
    argv = ["financial", "--chunks", str(path), "--doc", "d", "--year", "2025", "--scope", "consolidated",
            "--operation", "growth_rate", "--metric", "营业收入", "--tokenizer", "char", "--output-dir", str(output)]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert "20.00" in (output / "report.md").read_text(encoding="utf-8")
    before = (output / "result.json").read_bytes()
    assert main(argv) == 2
    assert "already exists" in capsys.readouterr().err
    assert before == (output / "result.json").read_bytes()


def test_cli_noncompletion_returns_distinct_exit_code(tmp_path, capsys):
    corpus = corpus_for()
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in corpus.chunks.values()), encoding="utf-8")
    assert main(["financial", "--chunks", str(path), "--doc", "d", "--year", "2025", "--scope", "consolidated",
                 "--operation", "growth_rate", "--metric", "营业收入", "--max-tool-calls", "1", "--tokenizer", "char"]) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "partial"
