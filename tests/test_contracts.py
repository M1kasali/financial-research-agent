import pytest

from financial_agent.contracts import Budget, OutputKind, SearchRequest, TaskRequest


@pytest.mark.parametrize("kind", list(OutputKind))
def test_free_task_does_not_require_competition_fields(kind):
    task = TaskRequest("比较两份材料", ("a@version",), output_kind=kind)
    assert task.task_id and not hasattr(task, "qid")


@pytest.mark.parametrize("query,scope", [("", ("a",)), ("x", ()), ("x", ("a", "a")), ("x", ["a"])])
def test_invalid_task(query, scope):
    with pytest.raises(ValueError):
        TaskRequest(query, scope)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_budget(value):
    with pytest.raises(ValueError):
        Budget(max_tool_calls=value)


@pytest.mark.parametrize("value", [0, 101, True])
def test_invalid_top_k(value):
    with pytest.raises(ValueError):
        SearchRequest("收入", top_k=value)
