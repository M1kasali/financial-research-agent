import pytest

from financial_agent.task_plan import ResearchTarget, validate_plan


def test_dependency_order():
    assert validate_plan(
        (ResearchTarget("calculate", "计算变化", ("extract",)), ResearchTarget("extract", "查找事实"))
    ) == ("extract", "calculate")


@pytest.mark.parametrize(
    "targets",
    [
        (),
        (ResearchTarget("a", "x", ("missing",)),),
        (ResearchTarget("a", "x", ("b",)), ResearchTarget("b", "x", ("a",))),
        (ResearchTarget("a", "x"), ResearchTarget("a", "y")),
    ],
)
def test_invalid_plans(targets):
    with pytest.raises(ValueError):
        validate_plan(targets)
