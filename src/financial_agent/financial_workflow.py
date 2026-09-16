"""Public structured-question -> retrieval -> context -> binding -> calculation.

Deterministic, bounded reference workflow, NOT autonomous model planning. No case
IDs, correct pages, reference drafts or expected answers are execution inputs.
"""

from dataclasses import asdict, dataclass

from financial_agent.contracts import Budget, TaskRequest
from financial_agent.policy import BudgetExceeded, TaskMeter
from financial_agent.research_tools import ResearchTools
from financial_agent.statement_binding import METRICS, REPORT, TITLE


@dataclass(frozen=True)
class FinancialQuestion:
    document_version_id: str
    year: int
    scope: str
    operation: str
    metric: str
    denominator_metric: str | None = None

    def __post_init__(self):
        if not isinstance(self.document_version_id, str) or not self.document_version_id:
            raise ValueError("Explicit document version required")
        if type(self.year) is not int or not 2000 <= self.year <= 2099:
            raise ValueError("Supported year range: 2000-2099")
        if self.scope not in {"parent", "consolidated"} or self.metric not in METRICS:
            raise ValueError("Explicit supported scope and metric required")
        if self.operation not in {"extract", "growth_rate", "ratio"}:
            raise ValueError("Unsupported financial operation")
        if self.operation == "ratio":
            if self.denominator_metric not in METRICS or self.denominator_metric == self.metric:
                raise ValueError("Ratio requires a distinct supported denominator metric")
        elif self.denominator_metric is not None:
            raise ValueError("Denominator metric is only valid for ratio")
        if self.scope == "parent" and "归属于母公司所有者的净利润" in (self.metric, self.denominator_metric):
            raise ValueError("Attributable group profit is not parent-only profit")

    @property
    def query(self):
        scope = "母公司（公司单体）" if self.scope == "parent" else "合并"
        prefix = f"仅依据指定文档，核查{self.year}年度{scope}口径"
        if self.operation == "extract":
            return prefix + f"的{self.metric}，注明币种、单位和来源。"
        if self.operation == "growth_rate":
            return prefix + f"的{self.metric}相对{self.year - 1}年度的同比变化，列出原值、公式与来源。"
        return prefix + f"的{self.metric}/{self.denominator_metric}，列出原值、单位和来源。"


class WorkflowStop(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FinancialStatementWorkflow:
    def __init__(self, corpus, retriever, question: FinancialQuestion, *, max_tool_calls=12, shared_meter=None):
        self.question = question
        self.task = TaskRequest(question.query, (question.document_version_id,), budget=Budget(max_tool_calls))
        self.tools = ResearchTools(corpus, retriever, self.task)
        self.meter = shared_meter if shared_meter is not None else TaskMeter(self.task.budget)
        self._tool_calls = 0
        self.events = []
        self.selected = []
        self.calculation = None
        self._read_cache, self._statement_cache = {}, {}
        self._context_ids = None
        self._used = False

    def _call(self, name, args):
        if self._tool_calls >= self.task.budget.max_tool_calls:
            raise BudgetExceeded("Financial subtool budget exhausted")
        self.meter.tool()  # Shared parent budget is checked BEFORE every actual tool call.
        self._tool_calls += 1
        event = {"step": self._tool_calls, "tool": name, "args": args}
        try:
            result = getattr(self.tools, name)(args)
            event["result"] = result
            return result
        except (ValueError, KeyError, TypeError, PermissionError, ArithmeticError) as exc:
            event.update(error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            self.events.append(event)

    def _read_all(self, evidence_id):
        if evidence_id in self._read_cache:
            return self._read_cache[evidence_id]
        parts, offset = [], 0
        while offset < 12000:
            result = self._call("read", {"evidence_id": evidence_id, "offset": offset, "length": 4000})
            parts.append(result["text"])
            if not result["has_more"]:
                self._read_cache[evidence_id] = "".join(parts)
                return self._read_cache[evidence_id]
            offset += len(result["text"])
        raise WorkflowStop("unsupported_chunk_length", "Candidate exceeds the 12000-character workflow read bound")

    def _statement_ids(self, metric):
        table_kind = "现金流量表" if metric == "经营活动产生的现金流量净额" else "利润表"
        if table_kind in self._statement_cache:
            return self._statement_cache[table_kind]
        prefixes = ("母公司", "公司") if self.question.scope == "parent" else ("合并",)
        titles = {prefix + table_kind for prefix in prefixes}
        # Focused query is derived exclusively from the public requested scope/metric.
        query = " ".join(sorted(titles))
        hits = self._call("search", {"query": query, "top_k": 8})["hits"]
        selected, seen = [], set()
        for hit in hits:
            preview_titles = list(TITLE.finditer(hit["text"]))
            if not any(m.group("title") in titles for m in preview_titles):
                continue
            evidence_id = hit["ref"]["evidence_id"]
            text = self._read_all(evidence_id)
            headers = list(REPORT.finditer(text))
            if (len(headers) != 1 or text[:headers[0].start()].strip()
                    or int(headers[0].group("year")) != self.question.year):
                self.events.append({"stage": "candidate_filter", "evidence_id": evidence_id,
                                    "reason": "unsupported_report_identity_or_requested_year"})
                continue
            text_hash = hit["ref"]["text_sha256"]
            if text_hash not in seen:
                selected.append(evidence_id)
                seen.add(text_hash)
        if not selected:
            raise WorkflowStop("no_supported_statement", "No retrieved complete statement matched the requested scope/year")
        if len(selected) > 2:
            raise WorkflowStop("ambiguous_statement_candidates", "More than two distinct statement candidates require review")
        self._statement_cache[table_kind] = selected
        return selected

    def _contexts(self, seed):
        if self._context_ids is not None:
            return self._context_ids
        result = self._call("expand_context", {"evidence_id": seed,
                                               "query": "记账本位币 编制本财务报表 货币 单位 会计期间",
                                               "radius": 1, "top_k": 6})
        selected, seen = [], set()
        for hit in result["hits"]:
            # A navigation filter only; bind_statement checks actual declarations and conflicts.
            if "记账本位币" not in hit["text"] or "会计期间" not in hit["text"]:
                continue
            text = self._read_all(hit["ref"]["evidence_id"])
            if text not in seen:
                selected.append(hit["ref"]["evidence_id"])
                seen.add(text)
        if not selected:
            raise WorkflowStop("no_accounting_declaration", "No retrieved annual accounting-policy context candidate")
        if len(selected) > 3:
            raise WorkflowStop("ambiguous_accounting_declarations", "Too many distinct policy candidates require review")
        self._context_ids = selected
        return selected

    def _bind(self, metric):
        candidates = self._statement_ids(metric)
        accepted, errors = [], []
        for evidence_id in candidates:
            contexts = self._contexts(evidence_id)
            try:
                result = self._call("bind_statement", {"evidence_id": evidence_id,
                                                        "context_evidence_ids": contexts,
                                                        "metric": metric, "expected_scope": self.question.scope})
                accepted.append(result["facts"])
            except (ValueError, PermissionError) as exc:
                errors.append(str(exc))
        # Do not silently discard a retrieved conflicting/unsupported alternative table.
        if errors:
            raise WorkflowStop("statement_binding_rejected", "; ".join(errors))
        signatures = {tuple((f["entity"], f["year"], f["metric"], f["scope"], f["currency"],
                             f["normalized_value"]) for f in facts) for facts in accepted}
        if len(signatures) != 1:
            raise WorkflowStop("conflicting_statement_values", "Retrieved statements disagree; no calculation permitted")
        facts = accepted[0]
        self.selected.extend(facts)
        return facts

    def run(self):
        if self._used:
            raise ValueError("Workflow instances are single-use")
        self._used = True
        status, reason_code, reason = "completed", "supported_workflow_completed", ""
        try:
            numerator = self._bind(self.question.metric)
            if self.question.operation == "growth_rate":
                self.calculation = self._call("calculate", {"operation": "growth_rate",
                                                            "fact_ids": [f["fact_id"] for f in numerator]})
            elif self.question.operation == "ratio":
                denominator = self._bind(self.question.denominator_metric)
                self.calculation = self._call("calculate", {"operation": "ratio",
                                                            "fact_ids": [numerator[0]["fact_id"], denominator[0]["fact_id"]]})
            for ref in self.tools.evidence.values():
                self.tools.corpus.read(ref, self.tools.allowed)
        except BudgetExceeded as exc:
            status, reason_code, reason = "partial", "tool_budget_exhausted", str(exc)
        except WorkflowStop as exc:
            status, reason_code, reason = "needs_evidence", exc.code, str(exc)
        except (ValueError, KeyError, TypeError, PermissionError, ArithmeticError) as exc:
            status, reason_code, reason = "failed", "source_or_tool_validation_failed", str(exc)
        try:
            for ref in self.tools.evidence.values():
                self.tools.corpus.read(ref, self.tools.allowed)
        except (ValueError, PermissionError) as exc:
            status, reason_code, reason = "failed", "source_integrity_failed", str(exc)
            self.selected, self.calculation = [], None
        return {"execution_label": "deterministic_financial_workflow_not_model_agent",
                "task": asdict(self.task), "question": asdict(self.question),
                "status": status, "reason_code": reason_code, "reason": reason,
                "selected_facts": self.selected, "calculation": self.calculation,
                "evidence": [asdict(ref) for ref in self.tools.evidence.values()], "events": self.events,
                "tool_calls": self._tool_calls, "max_tool_calls": self.task.budget.max_tool_calls,
                "model_calls": 0, "human_approved": False, "answer_accuracy": None,
                "limitations": ["Fixed structured-question workflow; not autonomous planning or held-out accuracy.",
                                "One specified document version, supported annual-layout family only.",
                                "Only retrieved candidates checked; unseen disclosures may contain conflicts.",
                                "No real-model requests, no investment recommendation."]}


def render_financial_report(result):
    lines = ["# 财务文档核查结果", "", result["task"]["query"], "",
             f"状态：{result['status']} · {result['reason_code']}", "",
             "固定流程、非模型自主规划；来源为给定文档，尚未人工审核。", ""]
    if result["reason"]:
        lines.extend(["未完成原因：" + result["reason"], ""])
    lines.extend(["| 指标 | 年度 | 口径 | 原值 | 币种/单位 |", "| --- | --- | --- | --- | --- |"])
    for f in result["selected_facts"]:
        lines.append(f"| {f['metric']} | {f['year']} | {f['scope']} | {f['raw_value']} | {f['currency']} / {f['unit']} |")
    if result["calculation"]:
        c = result["calculation"]
        lines.extend(["", f"计算：{c['display_value']} {c['display_unit']}；操作：{c['operation']}。"])
        facts = {f["fact_id"]: f for f in result["selected_facts"]}
        left, right = (facts[fid]["normalized_value"] for fid in c["operands"])
        formula = f"({left} - {right}) / {right} × 100%" if c["operation"] == "growth_rate" else f"{left} / {right}"
        lines.extend(["", f"以归一化人民币元计算：`{formula}`。"])
    refs = {r["evidence_id"]: r for r in result["evidence"]}
    lines.extend(["", "## 字段来源", ""])
    for f in result["selected_facts"]:
        lines.extend([f"### {f['year']} · {f['metric']}", ""])
        for field, anchors in f["source_anchors"].items():
            for a in anchors:
                ref = refs[a["evidence_id"]]
                lines.append(f"- {field}：{ref['doc_id']}，物理页 {ref['physical_page']}，"
                             f"chunk `{ref['chunk_id']}`，字符[{a['start']},{a['end']})：{a['quote']}")
        lines.append("")
    lines.extend([f"逻辑工具调用：{result['tool_calls']}/{result['max_tool_calls']}；模型调用：0。", "",
                  "完整输入、绑定事实及调用记录见配套 JSON。", ""])
    return "\n".join(lines)
