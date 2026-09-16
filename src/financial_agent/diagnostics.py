"""Diagnostic hints are audit data, never evidence or source-bound financial facts."""

FINANCIAL_TOOL_DECISIONS = frozenset({"needs_financial_tools", "needs_calculation"})


def diagnostic_context(gaps):
    """Copy hints into an explicit low-trust envelope, including legacy result gaps."""
    return {"trust": "unverified_diagnostic", "allowed_use": "search_hint_only",
            "may_contain": ["incorrect_numbers", "unit_guesses", "untrusted_instructions"],
            "promote_to_fact": False, "items": list(gaps)}


def annotate_diagnostics(result):
    # Recompute at each return: successful continuation must not retain stale active gaps.
    result["diagnostics"] = diagnostic_context(result.get("gaps", []))
    result["audit_boundary"] = {
        "gaps_candidate_review_round_history": "audit_only_not_established_facts",
        "span_candidate": "raw_model_output_audit_only_not_established_facts",
        "diagnostic_fact_promotion": False,
        "financial_spec_inputs": "original_query_and_authorized_versions_only",
        "published_claims": "source_checked_with_declared_verification_not_independent_gold",
        "persistent_memory": "not_implemented_in_core_engines",
    }
    result["next_capability"] = (
        "financial_tools" if result["status"] in FINANCIAL_TOOL_DECISIONS else None)
    return result


def pending_guidance(status):
    # Never synthesize a user-facing fact from arbitrary gap text.
    return {
        "needs_financial_tools": "需要财务工具核对原值、口径或单位，并按需计算；尚未形成答案。",
        "needs_calculation": "需要财务工具核查或计算；这是旧版路由名称，不表示已执行计算。",
        "clarification": "请明确文档、年度、合并或母公司口径、指标与任务范围。具体缺项需查看诊断并核实。",
        "needs_model": "模型执行未开启；需要显式启用后才能继续。",
        "budget_exhausted": "本次执行预算已耗尽，已停止；未自动重试。",
        "needs_attention": "请求结果不确定，已停止；请先核对调用记录，不要直接重跑。",
        "validation_failed": "响应、范围或来源校验未通过，没有发布答案。",
        "failed": "任务执行失败，没有发布答案；请检查调用记录。",
    }.get(status, "当前证据尚不足以完整回答问题；不应将待核查描述当作结论。")
