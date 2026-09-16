"""Strict task dependency contract used by the model-driven runtime."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchTarget:
    target_id: str
    question: str
    depends_on: tuple[str, ...] = ()
    required_facets: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.target_id, str)
            or not self.target_id.strip()
            or not isinstance(self.question, str)
            or not self.question.strip()
        ):
            raise ValueError("Targets need an ID and a question")
        if not isinstance(self.depends_on, tuple) or not isinstance(self.required_facets, tuple):
            raise ValueError("Dependencies and facets must be tuples")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in (*self.depends_on, *self.required_facets)
        ):
            raise ValueError("Dependency IDs and facets must be non-empty strings")


def validate_plan(targets: tuple[ResearchTarget, ...]) -> tuple[str, ...]:
    """Return deterministic topological order; reject cycles and dangling references."""
    if not targets:
        raise ValueError("Empty plan")
    by_id = {target.target_id: target for target in targets}
    if len(by_id) != len(targets):
        raise ValueError("Duplicate target ID")
    for target in targets:
        if set(target.depends_on) - by_id.keys():
            raise ValueError("Unknown dependency")
    ordered = []
    remaining = set(by_id)
    while remaining:
        ready = sorted(key for key in remaining if set(by_id[key].depends_on).issubset(ordered))
        if not ready:
            raise ValueError("Dependency cycle")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return tuple(ordered)
