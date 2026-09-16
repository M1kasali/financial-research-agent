"""Offline evaluation contracts. Exact fields, citation integrity and HUMAN review
are separate metrics; neither retrieval hits nor an agent's self-review is gold.
"""

import json
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from financial_agent.contracts import Budget, EvidenceRef, TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.research_tools import stable_id


def decimal_value(value):
    if not isinstance(value, str) or len(value) > 100:
        raise ValueError("Numeric values must be finite decimal strings")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError("Invalid decimal") from None
    if not number.is_finite() or abs(number.adjusted()) > 100:
        raise ValueError("Non-finite decimal")
    return number


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    document_version_ids: tuple[str, ...]
    domain: str
    task_kind: str
    partition: str  # fixture / historical_replay / candidate / reviewed_holdout

    def __post_init__(self):
        TaskRequest(self.query, self.document_version_ids)
        if any(not isinstance(x, str) or not x.strip() for x in (self.case_id, self.domain, self.task_kind)):
            raise ValueError("Case identity and category are required")
        if self.partition not in {"fixture", "historical_replay", "candidate", "reviewed_holdout"}:
            raise ValueError("Unknown evaluation partition")

    def task(self, budget: Budget) -> TaskRequest:
        # Deliberately contains no answer, gold anchors or scoring criteria.
        return TaskRequest(self.query, self.document_version_ids, budget=budget)


@dataclass(frozen=True)
class EvalSuite:
    name: str
    cases: tuple[EvalCase, ...]

    def __post_init__(self):
        if not self.name or not self.cases or len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("Suite must have a name and unique cases")
        if len({c.partition for c in self.cases}) != 1:
            raise ValueError("Do not mix fixture, replay, candidate and holdout partitions")

    @property
    def fingerprint(self):
        return stable_id("suite-", {"name": self.name, "cases": [asdict(c) for c in self.cases]})

    @classmethod
    def load(cls, path: Path):
        obj = json.loads(path.read_text(encoding="utf-8"))
        cases = tuple(
            EvalCase(**{**r, "document_version_ids": tuple(r["document_version_ids"])}) for r in obj["cases"]
        )
        suite = cls(obj["name"], cases)
        if obj["suite_hash"] != suite.fingerprint:
            raise ValueError("Suite hash mismatch")
        return suite

    def export(self):
        return {"name": self.name, "suite_hash": self.fingerprint, "cases": [asdict(c) for c in self.cases]}


def prediction_hash(prediction: dict) -> str:
    return stable_id("prediction-", prediction)


def validate_label(label: dict):
    if label.get("status") == "pending":
        if label.get("expected") is not None:
            raise ValueError("Pending labels must not contain an expected answer")
        return
    if label.get("status") not in {"synthetic_fixture", "human_reviewed"}:
        raise ValueError("Untrusted gold status")
    if not isinstance(label.get("provenance"), str) or not label["provenance"].strip():
        raise ValueError("Labels need explicit provenance")
    if label["status"] == "human_reviewed" and not label.get("reviewer"):
        raise ValueError("Human labels need a reviewer")
    expected = label.get("expected")
    if not isinstance(expected, dict) or expected.get("kind") not in {
        "number",
        "selection",
        "decision",
        "text_exact",
        "rubric",
    }:
        raise ValueError("Unsupported gold type")
    kind = expected["kind"]
    if kind == "rubric":
        criteria = expected.get("criteria")
        if (
            not isinstance(criteria, list)
            or not criteria
            or any(not isinstance(v, str) or not v.strip() for v in criteria)
        ):
            raise ValueError("Rubric requires explicit review criteria")
    if kind == "number":
        decimal_value(expected.get("value"))
        if (
            decimal_value(expected.get("tolerance", "0")) < 0
            or not isinstance(expected.get("unit"), str)
            or not expected["unit"]
        ):
            raise ValueError("Invalid numeric tolerance or unit")
    elif kind == "selection":
        if (
            not isinstance(expected.get("value"), str)
            or re.fullmatch(r"[A-H]+", expected["value"]) is None
            or len(set(expected["value"])) != len(expected["value"])
        ):
            raise ValueError("Invalid expected selection")
    elif kind == "decision" and expected.get("value") not in {"answered", "abstained", "clarification"}:
        raise ValueError("Invalid expected decision")
    elif kind == "text_exact" and (
        not isinstance(expected.get("value"), str) or not expected["value"].strip()
    ):
        raise ValueError("Empty expected text")


def exact_score(prediction: dict | None, expected: dict) -> bool:
    if prediction is None or prediction["decision"] == "error":
        return False
    kind, value = expected["kind"], expected["value"]
    if kind == "decision":
        return prediction["decision"] == value
    if prediction["decision"] != "answered":
        return False
    if kind == "text_exact":
        return " ".join(prediction["answer"].split()) == " ".join(value.split())
    field = prediction.get("structured") or {}
    if kind == "selection":
        raw = field.get("selection")
        # Never mine A-H letters out of prose such as 'Because ...'.
        return (
            isinstance(raw, str)
            and re.fullmatch(r"[A-H]+", raw) is not None
            and len(raw) == len(set(raw))
            and set(raw) == set(value)
        )
    try:
        number = decimal_value(field.get("value"))
        return field.get("unit") == expected["unit"] and abs(number - decimal_value(value)) <= decimal_value(
            expected.get("tolerance", "0")
        )
    except ValueError:
        return False


def evaluate(
    suite: EvalSuite, labels: dict, run: dict, corpus: Corpus, *, reviews: list[dict] | None = None
) -> dict:
    """Score a complete run envelope. Missing predictions remain visible in denominators."""
    if labels.get("suite_hash") != suite.fingerprint or run.get("suite_hash") != suite.fingerprint:
        raise ValueError("Run/label suite version mismatch")
    if run.get("mode") not in {"real_model", "simulation", "imported_historical", "offline_retrieval"}:
        raise ValueError("An explicit run mode is required")
    if not run.get("system_id") or not isinstance(run.get("settings"), dict):
        raise ValueError("Run must identify system and experimental settings")
    cases = {c.case_id: c for c in suite.cases}
    for case in suite.cases:
        if set(case.document_version_ids) - corpus.by_version.keys():
            raise ValueError("Evaluation corpus version mismatch")
    label_rows = labels.get("labels", {})
    if set(label_rows) - cases.keys():
        raise ValueError("Label for unknown case")
    if any(c.partition == "reviewed_holdout" and c.case_id not in label_rows for c in suite.cases):
        raise ValueError("Holdout requires reviewed labels for every case")
    for key, label in label_rows.items():
        validate_label(label)
        if label["status"] == "synthetic_fixture" and cases[key].partition != "fixture":
            raise ValueError("Synthetic labels cannot grade real cases")
        if cases[key].partition == "reviewed_holdout" and label["status"] != "human_reviewed":
            raise ValueError("Holdout requires reviewed labels")
    predictions = {}
    for row in run["predictions"]:
        key = row["case_id"]
        if key not in cases or key in predictions:
            raise ValueError("Unknown or duplicate prediction")
        if row.get("decision") not in {"answered", "abstained", "clarification", "error"}:
            raise ValueError("Invalid decision")
        if not isinstance(row.get("answer"), str) or not isinstance(row.get("citations"), list):
            raise ValueError("Invalid answer/citations")
        if row["decision"] != "error" and not row["answer"].strip():
            raise ValueError("Answers and refusals need explicit text")
        seconds = row.get("elapsed_seconds")
        if seconds is not None and (
            type(seconds) not in {float, int} or seconds < 0 or not Decimal(str(seconds)).is_finite()
        ):
            raise ValueError("Invalid elapsed time")
        usage = row.get("usage")
        if usage is not None:
            for field in (
                "reported_tokens",
                "model_attempts",
                "unknown_usage_attempts",
                "estimated_usage_attempts",
            ):
                value = usage.get(field)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("Invalid usage count")
        predictions[key] = row
    review_by_id = {}
    for review in reviews or []:
        key = review["case_id"]
        if key not in predictions or key in review_by_id:
            raise ValueError("Unknown or duplicate review")
        if review.get("prediction_hash") != prediction_hash(predictions[key]):
            raise ValueError("Review is stale for this prediction")
        if (
            review.get("review_kind") != "human"
            or not isinstance(review.get("reviewer"), str)
            or not review["reviewer"].strip()
        ):
            raise ValueError("Only explicit human reviews are counted here")
        if any(type(review.get(k)) is not bool for k in ("answer_correct", "citation_supported", "complete")):
            raise ValueError("Human review fields must be booleans")
        review_by_id[key] = review
    rows = []
    for case in suite.cases:
        pred, label = predictions.get(case.case_id), label_rows.get(case.case_id)
        eligible = (
            label is not None and label["status"] != "pending" and label["expected"]["kind"] != "rubric"
        )
        correct = exact_score(pred, label["expected"]) if eligible else None
        valid, total = 0, 0
        seen = set()
        for raw in pred["citations"] if pred else []:
            total += 1
            try:
                ref = EvidenceRef(**raw)
                if ref.evidence_id in seen:
                    continue
                corpus.read(ref, frozenset(case.document_version_ids))
                if stable_id("ref-", raw) != stable_id("ref-", asdict(corpus.refs[ref.chunk_id])):
                    continue
                seen.add(ref.evidence_id)
                valid += 1
            except (TypeError, ValueError, PermissionError, KeyError):
                pass
        rows.append(
            {
                "case_id": case.case_id,
                "domain": case.domain,
                "task_kind": case.task_kind,
                "partition": case.partition,
                "present": pred is not None,
                "decision": pred["decision"] if pred else "missing",
                "exact_field_correct": correct,
                "valid_citations": valid,
                "citation_count": total,
                "human_review": review_by_id.get(case.case_id),
            }
        )

    def summarize(subset):
        eligible = [r for r in subset if r["exact_field_correct"] is not None]
        reviewed = [r for r in subset if r["human_review"] is not None]
        cited = sum(r["citation_count"] for r in subset)
        return {
            "case_count": len(subset),
            "prediction_count": sum(r["present"] for r in subset),
            "scorable_count": len(eligible),
            "exact_field_accuracy": sum(r["exact_field_correct"] for r in eligible) / len(eligible)
            if eligible
            else None,
            "human_review_count": len(reviewed),
            "human_answer_accuracy": sum(r["human_review"]["answer_correct"] for r in reviewed)
            / len(reviewed)
            if reviewed
            else None,
            "human_supported_answer_rate": sum(r["human_review"]["citation_supported"] for r in reviewed)
            / len(reviewed)
            if reviewed
            else None,
            "human_completeness_rate": sum(r["human_review"]["complete"] for r in reviewed) / len(reviewed)
            if reviewed
            else None,
            "citation_count": cited,
            "citation_integrity_rate": sum(r["valid_citations"] for r in subset) / cited if cited else None,
            "missing_predictions": sum(not r["present"] for r in subset),
            "errors": sum(r["decision"] == "error" for r in subset),
            "abstentions": sum(r["decision"] == "abstained" for r in subset),
        }

    known_tokens = [
        p["usage"]["reported_tokens"]
        for p in predictions.values()
        if p.get("usage") and p["usage"].get("reported_tokens") is not None
    ]
    durations = [p["elapsed_seconds"] for p in predictions.values() if p.get("elapsed_seconds") is not None]
    return {
        "suite_hash": suite.fingerprint,
        "label_hash": stable_id("labels-", labels),
        "system_id": run["system_id"],
        "mode": run["mode"],
        "settings": run["settings"],
        "summary": summarize(rows),
        "by_domain": {
            d: summarize([r for r in rows if r["domain"] == d]) for d in sorted({r["domain"] for r in rows})
        },
        "by_task_kind": {
            d: summarize([r for r in rows if r["task_kind"] == d])
            for d in sorted({r["task_kind"] for r in rows})
        },
        "known_reported_tokens": sum(known_tokens) if known_tokens else None,
        "runs_without_complete_reported_usage": sum(
            not p.get("usage")
            or p["usage"].get("reported_tokens") is None
            or p["usage"].get("unknown_usage_attempts") != 0
            or p["usage"].get("estimated_usage_attempts") != 0
            for p in predictions.values()
        ),
        "mean_elapsed_seconds": sum(durations) / len(durations) if durations else None,
        "latency_sample_count": len(durations),
        "monetary_cost": None,
        "rows": rows,
        "limitations": [
            "Exact fields are not full semantic correctness.",
            "Valid citation hashes do not prove support for a claim.",
            "Simulation and historical replay are not independent real-model quality evaluations.",
        ],
    }


def compare_reports(left: dict, right: dict) -> dict:
    for key in ("suite_hash", "label_hash", "settings", "mode"):
        if left[key] != right[key]:
            raise ValueError(f"Non-comparable experimental {key}")
    a, b = left["summary"], right["summary"]
    return {
        "left": left["system_id"],
        "right": right["system_id"],
        "mode": left["mode"],
        "exact_field_accuracy_delta": b["exact_field_accuracy"] - a["exact_field_accuracy"]
        if a["exact_field_accuracy"] is not None and b["exact_field_accuracy"] is not None
        else None,
        "quality_claim_permitted": False,
        "note": "Paired diagnostic only. Check label provenance, missing runs, human review and uncertainty before making quality claims.",
    }
