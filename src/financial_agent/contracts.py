"""Application contracts, independent of competition questions and answer slots."""

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


class OutputKind(StrEnum):
    ANSWER = "answer"
    COMPARISON = "comparison"
    REPORT = "report"


@dataclass(frozen=True)
class Budget:
    max_tool_calls: int = 12
    max_model_attempts: int = 16
    max_estimated_tokens: int = 40_000

    def __post_init__(self):
        for value in (self.max_tool_calls, self.max_model_attempts, self.max_estimated_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError("Budget limits must be positive integers")


@dataclass(frozen=True)
class TaskRequest:
    query: str
    document_version_ids: tuple[str, ...]
    output_kind: OutputKind = OutputKind.ANSWER
    task_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid4()))
    budget: Budget = field(default_factory=Budget)

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("A non-empty research query is required")
        if len(self.query) > 20_000:
            raise ValueError("Research query exceeds 20,000 characters")
        versions = self.document_version_ids
        if not isinstance(versions, tuple) or not versions:
            raise ValueError("An explicit non-empty tuple of document versions is required")
        if any(not isinstance(v, str) or not v.strip() for v in versions):
            raise ValueError("Document versions must be non-empty strings")
        if len(set(versions)) != len(versions):
            raise ValueError("Duplicate document versions")
        if not isinstance(self.output_kind, OutputKind):
            raise ValueError("output_kind must be an OutputKind")
        if not isinstance(self.budget, Budget):
            raise ValueError("budget must be a Budget")
        if any(not isinstance(value, str) or not value.strip() for value in (self.task_id, self.session_id)):
            raise ValueError("Task and session identifiers are required")


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    doc_id: str
    version_id: str
    chunk_id: str
    physical_page: int | None
    start: int
    end: int
    text_sha256: str


@dataclass(frozen=True)
class EvidenceHit:
    ref: EvidenceRef
    text: str
    score: float
    section: str


@dataclass(frozen=True)
class SearchRequest:
    query: str
    document_version_ids: tuple[str, ...] | None = None
    top_k: int = 8

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip() or len(self.query) > 20_000:
            raise ValueError("Search query must contain 1–20,000 characters")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be an integer in [1,100]")
        if self.document_version_ids is not None:
            if not isinstance(self.document_version_ids, tuple):
                raise ValueError("Search scope must be a tuple or None")
            if any(not isinstance(v, str) or not v for v in self.document_version_ids):
                raise ValueError("Invalid search scope")
