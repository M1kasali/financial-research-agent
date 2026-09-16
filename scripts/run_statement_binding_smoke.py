"""Analyst-selected real-corpus component checks, NOT autonomous retrieval or B scoring."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from financial_agent.contracts import TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.research_tools import ResearchTools


def main():
    root = Path(__file__).resolve().parents[1]
    sources = {"profit": "b77947f7:p126:0:28be31d494", "cashflow": "b77947f7:p130:0:4ea34dbb7a",
               "parent": "b77947f7:p134:0:0b41ebcda8", "policy": "b77947f7:p140:0:ff9e0e9780",
               "subsidiary": "b77947f7:p227:0:72c3cb2368", "customer": "b77947f7:p252:3001:2ffda05f6c",
               "foreign_document": "c83bc0ff:p234:0:41a5344ec1"}
    with patch("socket.socket", side_effect=AssertionError("Network disabled")):
        config = json.loads((root / "configs/datasets.example.json").read_text())
        corpus = Corpus.from_jsonl((root / config["chunks"]).resolve())
        versions = tuple(dict.fromkeys(corpus.refs[cid].version_id for cid in sources.values()))
        task = TaskRequest("组件测试：分别核对年度合并与公司报表，不混用口径。", versions)
        tools = ResearchTools(corpus, None, task)
        # Explicit preselected sources isolate the binding component. No retrieval claim.
        refs = {name: corpus.refs[cid] for name, cid in sources.items()}
        tools.evidence = {ref.evidence_id: ref for ref in refs.values()}

        def bind(source, metric, scope="consolidated", context="policy"):
            return tools.bind_statement({"evidence_id": refs[source].evidence_id,
                                         "context_evidence_ids": [refs[context].evidence_id],
                                         "metric": metric, "expected_scope": scope})["facts"]

        revenue = bind("profit", "营业收入")
        profit = bind("profit", "净利润")
        cashflow = bind("cashflow", "经营活动产生的现金流量净额")
        parent_revenue = bind("parent", "营业收入", "parent")
        parent_profit = bind("parent", "净利润", "parent")
        growth = tools.calculate({"operation": "growth_rate", "fact_ids": [f["fact_id"] for f in revenue]})
        ratio = tools.calculate({"operation": "ratio", "fact_ids": [cashflow[0]["fact_id"], profit[0]["fact_id"]]})
        assert growth["display_value"] == "3.46"
        assert ratio["display_value"] == "1.75"
        assert parent_revenue[0]["raw_value"] == "1,525,075"
        assert parent_profit[0]["raw_value"] == "4,206,012"
        rejected = []
        for name, attempt in [
            ("consolidated_as_parent", lambda: bind("profit", "营业收入", "parent")),
            ("subsidiary_as_group", lambda: bind("subsidiary", "营业收入")),
            ("single_customer_as_group", lambda: bind("customer", "营业收入")),
            ("other_document_context", lambda: bind("profit", "营业收入", context="foreign_document")),
            ("mixed_parent_consolidated_calculation", lambda: tools.calculate(
                {"operation": "ratio", "fact_ids": [cashflow[0]["fact_id"], parent_profit[0]["fact_id"]]})),
        ]:
            try:
                attempt()
            except (ValueError, PermissionError) as exc:
                rejected.append({"scenario": name, "rejected": True, "reason": str(exc)})
            else:
                raise AssertionError(f"Unsafe scenario unexpectedly accepted: {name}")
        facts = list(tools.facts.values())
        for fact in facts:
            for spans in fact["source_anchors"].values():
                for span in spans:
                    assert tools.read_evidence(span["evidence_id"])[span["start"]:span["end"]] == span["quote"]
        result = {"execution_label": "preselected_source_component_probe_not_agent",
                  "network_disabled": True, "model_calls": 0, "human_approved": False,
                  "source_selection": "analyst_selected_chunk_ids_not_automatic_retrieval",
                  "document_versions": versions, "facts": facts, "calculations": [growth, ratio],
                  "rejected_scenarios": rejected,
                  "summary": {"bound_facts": len(facts), "calculations": 2, "rejected_scenarios": len(rejected),
                              "revenue_growth_percent": growth["display_value"],
                              "cashflow_profit_ratio": ratio["display_value"],
                              "parent_revenue_cny_thousand": parent_revenue[0]["raw_value"],
                              "parent_profit_cny_thousand": parent_profit[0]["raw_value"]},
                  "limitations": ["Narrow parser family, not general PDF/table understanding.",
                                  "Source claims not externally verified; draft facts are not human gold.",
                                  "No autonomous retrieval, planning or accuracy evaluation."]}
    output = root / "experiments/runs" / ("statement-binding-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                         + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
