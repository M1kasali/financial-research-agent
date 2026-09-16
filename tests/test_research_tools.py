from dataclasses import replace

import pytest

from financial_agent.contracts import TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.research_tools import ResearchTools
from financial_agent.retrieval import ScopedRetriever
from financial_agent.simulation import finance_corpus


def prepared(corpus=None):
    corpus = corpus or finance_corpus()
    task = TaskRequest("营业收入同比变化", tuple(corpus.by_version))
    tools = ResearchTools(corpus, ScopedRetriever(corpus, tokenizer_mode="char"), task)
    hits = tools.search({"query": "营业收入 净利润"})["hits"]
    args = {"evidence_ids": [hit["ref"]["evidence_id"] for hit in hits]}
    tools.extract(args)
    ids = [f["fact_id"] for f in sorted(tools.facts.values(), key=lambda f: f["year"], reverse=True)]
    return tools, args, ids


def test_stable_fact_ids_and_readback():
    tools, args, ids = prepared()
    assert len(ids) == 2
    assert {f["fact_id"] for f in tools.extract(args)["facts"]} == set(ids)
    assert len(tools.facts) == 2
    ref = tools.evidence[args["evidence_ids"][0]]
    result = tools.read({"evidence_id": ref.evidence_id, "offset": 2, "length": 5})
    assert result["text"] == tools.corpus.chunks[ref.chunk_id].text[2:7]
    assert result["has_more"]


def test_metadata_snapshot_not_mutated_by_index_or_later_changes():
    corpus = finance_corpus()
    corpus.chunks["row"].metadata["financial_row"]["cells"][0]["raw_value"] = "999"
    tools, _, _ = prepared(corpus)
    assert {f["raw_value"] for f in tools.facts.values()} == {"120", "100"}


@pytest.mark.parametrize("extra", ["美元", "第一季度", "母公司利润表"])
def test_ambiguous_source_context_rejected(extra):
    chunk = finance_corpus().chunks["row"]
    corpus = Corpus([replace(chunk, text=chunk.text + extra)])
    tools, _, ids = prepared(corpus)
    with pytest.raises(ValueError, match="context incomplete"):
        tools.calculate({"operation": "growth_rate", "fact_ids": ids})


def test_missing_entity_not_guessed():
    chunk = finance_corpus().chunks["row"]
    chunk.metadata.pop("financial_context")
    tools, _, ids = prepared(Corpus([chunk]))
    assert all(f["entity"] == "unknown" for f in tools.facts.values())
    with pytest.raises(ValueError, match="context incomplete"):
        tools.calculate({"operation": "growth_rate", "fact_ids": ids})


@pytest.mark.parametrize(
    "field,value",
    [
        ("currency", "USD"),
        ("scope", "parent"),
        ("period_kind", "quarter"),
        ("entity", "乙公司"),
        ("metric", "净利润"),
        ("year", "2023"),
        ("normalized_value", "0"),
        ("normalized_value", "-100"),
        ("normalized_value", "NaN"),
        ("unit", "%"),
    ],
)
def test_incompatible_facts_rejected(field, value):
    tools, _, ids = prepared()
    tools.facts[ids[1]][field] = value  # Fault injection, not a model-accessible mutation API.
    with pytest.raises(ValueError):
        tools.calculate({"operation": "growth_rate", "fact_ids": ids})


@pytest.mark.parametrize(
    "args",
    [
        {"operation": "eval", "fact_ids": ["x", "y"]},
        {"operation": "growth_rate", "fact_ids": ["unknown", "unknown2"]},
        {"operation": "growth_rate", "fact_ids": ["x"]},
        {"operation": "growth_rate", "fact_ids": [], "code": "print(1)"},
    ],
)
def test_calculation_whitelist(args):
    tools, _, _ = prepared()
    with pytest.raises(ValueError):
        tools.calculate(args)


def test_source_mutation_rejected_before_calculation():
    tools, _, ids = prepared()
    tools.corpus.chunks["row"].text += "changed"
    with pytest.raises(ValueError, match="Source text changed"):
        tools.calculate({"operation": "growth_rate", "fact_ids": ids})


def test_empty_scope_not_full_library():
    tools, _, _ = prepared()
    assert tools.search({"query": "收入", "document_version_ids": []})["hits"] == []


def test_reversed_growth_period_rejected():
    tools, _, ids = prepared()
    with pytest.raises(ValueError, match="adjacent annual"):
        tools.calculate({"operation": "growth_rate", "fact_ids": list(reversed(ids))})


@pytest.mark.parametrize(
    "operation,expected", [("difference", "200000"), ("compare", "gt"), ("ratio", "1.2")]
)
def test_same_period_operations(operation, expected):
    # Two explicit same-year disclosures in separate rows.
    original = finance_corpus().chunks["row"]
    chunks = []
    for index, value in enumerate(("120", "100")):
        import copy

        metadata = copy.deepcopy(original.metadata)
        entity = "乙公司" if index == 1 and operation != "ratio" else "甲公司"
        metric = "净利润" if index == 1 and operation == "ratio" else "营业收入"
        metadata["financial_context"]["entity"] = entity
        metadata["financial_row"]["metric"] = metric
        metadata["financial_row"]["raw_row"] = f"{metric} {value}"
        metadata["financial_row"]["cells"] = [{"year": "2025", "raw_value": value, "unit": "万元"}]
        text = f"{entity}合并利润表，人民币万元，2025年度{metric}{value}。"
        chunks.append(replace(original, chunk_id=f"row{index}", text=text, metadata=metadata))
    tools, _, _ = prepared(Corpus(chunks))
    ids = [
        f["fact_id"]
        for f in sorted(tools.facts.values(), key=lambda f: int(f["normalized_value"]), reverse=True)
    ]
    result = tools.calculate({"operation": operation, "fact_ids": ids})
    assert result["result"] == expected
    assert result["policy_version"] == "annual-cny-v1"


def test_mixed_monetary_scales_normalize_before_growth():
    chunk = finance_corpus().chunks["row"]
    chunk.metadata["financial_row"]["cells"][1].update(raw_value="1000000", unit="元")
    chunk.text = "甲公司合并利润表，人民币，2025年度营业收入120万元；2024年度营业收入1000000元。"
    tools, _, ids = prepared(Corpus([chunk]))
    assert tools.calculate({"operation": "growth_rate", "fact_ids": ids})["display_value"] == "20.00"


def test_conflicting_disclosures_require_resolution():
    tools, _, ids = prepared()
    tools.facts[ids[1]]["year"] = "2025"
    with pytest.raises(ValueError, match="Conflicting disclosures"):
        tools.calculate({"operation": "difference", "fact_ids": ids})
