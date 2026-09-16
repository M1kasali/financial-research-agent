from copy import deepcopy

import pytest
from agent.schemas import Chunk

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.research_tools import ResearchTools
from financial_agent.retrieval import ScopedRetriever
from financial_agent.runtime import ResearchRuntime
from financial_agent.simulation import ScriptedSend, fake_client, target, tool
from financial_agent.storage import SQLiteTaskStore

HEADING = "甲股份有限公司2025 年年度报告全文\n"
TABLE = HEADING + "\n2、合并利润表\n\n附注七 2025 年 2024 年\n(经重述)\n\n一、 营业收入 45 120,000 100,000\n净利润 20,000 18,000\n"
POLICY = HEADING + "\n重要会计政策\n会计期间\n本集团会计年度采用公历年度,即每年自1 月1 日起至12 月31 日止。\n记账本位币\n本公司记账本位币和编制本财务报表所采用的货币均为人民币,除有特别说明外,均以人民币千元为单位表示。"


def prepared(table=TABLE, policy=POLICY, *, policy_doc="d"):
    corpus = Corpus([Chunk("t", "d", "financial_reports", 10, "", "", table),
                     Chunk("p", policy_doc, "financial_reports", 40, "", "", policy)])
    tools = ResearchTools(corpus, None, TaskRequest("合并营业收入同比", tuple(corpus.by_version)))
    tools.evidence = {ref.evidence_id: ref for ref in corpus.refs.values()}
    args = {"evidence_id": corpus.refs["t"].evidence_id,
            "context_evidence_ids": [corpus.refs["p"].evidence_id],
            "metric": "营业收入", "expected_scope": "consolidated"}
    return tools, args


def test_bind_column_values_and_all_source_spans_then_calculate():
    tools, args = prepared()
    facts = tools.bind_statement(args)["facts"]
    assert [f["year"] for f in facts] == ["2025", "2024"]
    assert [f["raw_value"] for f in facts] == ["120,000", "100,000"]
    assert [f["normalized_value"] for f in facts] == ["120000000", "100000000"]
    assert all(f["scope"] == "consolidated" and f["entity"] == "甲股份有限公司" for f in facts)
    for fact in facts:
        for spans in fact["source_anchors"].values():
            for span in spans:
                assert tools.read_evidence(span["evidence_id"])[span["start"]:span["end"]] == span["quote"]
    result = tools.calculate({"operation": "growth_rate", "fact_ids": [f["fact_id"] for f in facts]})
    assert result["display_value"] == "20.00"
    assert set(result["evidence_ids"]) == {args["evidence_id"], *args["context_evidence_ids"]}
    assert len(tools.facts) == 2
    assert tools.bind_statement(args)["facts"] == facts


@pytest.mark.parametrize("table", [
    TABLE.replace("合并利润表", "子公司财务信息"),
    TABLE.replace("合并利润表", "公司利润表"),
    TABLE.replace("附注七 2025 年 2024 年", "附注七 2024 年 2025 年"),
    TABLE.replace("附注七 2025 年 2024 年", "附注七 2025 年 2023 年"),
    TABLE.replace("附注七 2025 年 2024 年", "附注七 2025 年 2024 年 2023 年"),
    TABLE.replace("120,000 100,000", "120,000 100,000 80,000"),
    TABLE.replace("120,000", "12,00"),
    TABLE.replace("120,000", "123456789012345678901"),
    TABLE.replace("120,000", "(-120,000)"),
    TABLE + "营业收入 150,000 110,000\n",
    TABLE + "单位:万元\n",
    TABLE + "单位:百万元\n",
    TABLE + "币种:美元\n",
    TABLE + "单一客户收入\n",
    TABLE + "季度数据\n",
    TABLE.replace("附注七", "乙股份有限公司\n附注七"),
    TABLE.replace("2、合并利润表", "2、合并利润表\n3、公司利润表"),
    TABLE.replace("附注七 ", ""),  # extra numeric note lacks an explicit note header
])
def test_reject_ambiguous_or_wrong_statement(table):
    tools, args = prepared(table=table)
    with pytest.raises(ValueError):
        tools.bind_statement(args)
    assert tools.facts == {}


@pytest.mark.parametrize("policy", [
    POLICY.replace("千元", "美元"),
    POLICY.replace("2025 年年度报告", "2024 年年度报告"),
    POLICY.replace("甲股份有限公司", "乙股份有限公司"),
    POLICY.replace("本公司记账本位币和编制本财务报表所采用的货币均为人民币", "可能是人民币"),
    POLICY.replace("本集团会计年度采用公历年度", "本集团会计期间另定"),
    POLICY.replace("重要会计政策", "子公司资料"),
])
def test_missing_or_conflicting_declarations_reject(policy):
    tools, args = prepared(policy=policy)
    with pytest.raises(ValueError):
        tools.bind_statement(args)
    assert not tools.facts


def test_cross_document_context_rejected_even_when_task_allows_both():
    tools, args = prepared(policy_doc="other")
    with pytest.raises(PermissionError):
        tools.bind_statement(args)


def test_parent_table_is_not_attributable_profit():
    tools, args = prepared(table=TABLE.replace("合并利润表", "公司利润表"))
    args.update(metric="净利润", expected_scope="parent")
    facts = tools.bind_statement(args)["facts"]
    assert all(f["scope"] == "parent" and f["metric"] == "净利润" for f in facts)
    assert facts[0]["normalized_value"] == "20000000"


def test_binding_does_not_upgrade_or_mutate_legacy_candidates():
    tools, args = prepared()
    tools.facts["legacy"] = {"status": "candidate", "currency": "unknown"}
    old = deepcopy(tools.facts["legacy"])
    tools.bind_statement(args)
    assert tools.facts["legacy"] == old


@pytest.mark.parametrize("field,value", [("normalized_value", "999"), ("year", "2023"),
                                       ("entity", "乙股份有限公司"), ("scope", "parent"),
                                       ("evidence_ids", [])])
def test_recheck_binding_before_calculation(field, value):
    tools, args = prepared()
    facts = tools.bind_statement(args)["facts"]
    facts[0][field] = value
    with pytest.raises(ValueError, match="changed since"):
        tools.calculate({"operation": "growth_rate", "fact_ids": [f["fact_id"] for f in facts]})


def test_context_source_mutation_invalidates_calculation():
    tools, args = prepared()
    facts = tools.bind_statement(args)["facts"]
    tools.corpus.chunks["p"].text += "单位改为万元"
    with pytest.raises(ValueError, match="changed"):
        tools.calculate({"operation": "growth_rate", "fact_ids": [f["fact_id"] for f in facts]})


def test_legacy_text_regex_facts_cannot_bypass_column_binding():
    tools, args = prepared()
    facts = tools.bind_statement(args)["facts"]
    facts[0]["extraction_mode"] = "text_regex"
    facts[0].pop("binding_request")
    with pytest.raises(ValueError, match="Unbound text-regex"):
        tools.calculate({"operation": "growth_rate", "fact_ids": [f["fact_id"] for f in facts]})


def test_explicit_value_span_handles_parenthesized_negative():
    tools, args = prepared(table=TABLE.replace("120,000", "(120,000)"))
    fact = tools.bind_statement(args)["facts"][0]
    assert fact["normalized_value"] == "-120000000"
    assert fact["source_anchors"]["value"][0]["quote"] == "(120,000)"


def test_operating_entity_word_is_not_a_currency_unit_declaration():
    tools, args = prepared(table=TABLE + "处置子公司及其他营业单位收到的现金净额 2,000 1,000\n")
    assert len(tools.bind_statement(args)["facts"]) == 2


def test_runtime_binding_calculation_restore_and_required_context_citation(tmp_path):
    tools, args = prepared()
    corpus = tools.corpus
    retriever = ScopedRetriever(corpus, tokenizer_mode="char")
    task = TaskRequest("合并收入同比", tuple(corpus.by_version), budget=Budget(12, 20, 900_000))
    def calculate_action(context):
        facts = sorted(context["facts"], key=lambda f: f["year"], reverse=True)
        return tool("calculate", {"operation": "growth_rate", "fact_ids": [f["fact_id"] for f in facts]})

    store = SQLiteTaskStore(tmp_path / "binding.sqlite3")
    runtime = ResearchRuntime(corpus, task, fake_client(ScriptedSend([
        {"targets": [target()]}, tool("bind_statement", args), calculate_action
    ])), retriever=retriever, store=store, execution_label="simulated_model")
    # Source-selected integration fixture; not an automatic retrieval claim.
    runtime.tools.evidence = tools.evidence.copy()
    result = runtime.run(stop_after_steps=2)
    assert result["status"] == "paused"
    assert result["budget"]["tool_calls"] == 2
    calc = result["calculations"][0]
    assert calc["display_value"] == "20.00"
    with pytest.raises(ValueError, match="Calculation sources"):
        runtime._verify("t1", {"answer": "增长20%", "evidence_ids": [args["evidence_id"]],
                               "calculation_ids": [calc["calculation_id"]]})
    resumed = ResearchRuntime(corpus, task, fake_client(ScriptedSend([
        {"kind": "ask_user", "question": "请人工复核来源"}
    ])), retriever=retriever, store=store, execution_label="simulated_model")
    final = resumed.run()
    assert final["status"] == "needs_input"
    assert final["facts"] == result["facts"]
    assert final["calculations"] == result["calculations"]
    replayed = resumed.tools.calculate({"operation": "growth_rate", "fact_ids": calc["operands"]})
    assert replayed == calc


def test_conflicting_policy_pages_rejected():
    corpus = Corpus([Chunk("t", "d", "financial_reports", 10, "", "", TABLE),
                     Chunk("p1", "d", "financial_reports", 40, "", "", POLICY),
                     Chunk("p2", "d", "financial_reports", 41, "", "", POLICY.replace("千元", "万元"))])
    tools = ResearchTools(corpus, None, TaskRequest("收入", tuple(corpus.by_version)))
    tools.evidence = {r.evidence_id: r for r in corpus.refs.values()}
    with pytest.raises(ValueError, match="Conflicting accounting units"):
        tools.bind_statement({"evidence_id": corpus.refs["t"].evidence_id,
                              "context_evidence_ids": [corpus.refs[c].evidence_id for c in ("p1", "p2")],
                              "metric": "营业收入", "expected_scope": "consolidated"})
