"""Checkpoint binding and write-ahead model journal for ResearchRuntime."""

import hashlib
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from financial_agent.contracts import EvidenceRef
from financial_agent.model import ModelCallError
from financial_agent.policy import BudgetExceeded
from financial_agent.research_tools import stable_id
from financial_agent.storage import RecoveryError, UncertainRequest
from financial_agent.task_plan import ResearchTarget


def implementation_hash():
    # Installed editable project: include all new code and imported legacy Python code.
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in (root / "src", root / "vendor"):
        if not directory.is_dir():
            raise RecoveryError("Durable mode currently requires the editable project layout")
        for path in sorted(directory.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


class DurableSession:
    def __init__(self, runtime, store):
        self.runtime, self.store = runtime, store
        self.call_id = None
        self.original_record = runtime.client._record_sink

    def identity(self):
        runtime = self.runtime
        config = asdict(runtime.client.config)
        del config["api_key"]
        return {
            "task": asdict(runtime.task),
            "model": config,
            "code": implementation_hash(),
            "corpus_versions": sorted(runtime.tools.corpus.by_version),
            "tokenizer": runtime.tools.retriever.tokenizer_mode,
            "execution_label": runtime.execution_label,
            "python": platform.python_version(),
            "dependencies": {
                name: version(name)
                for name in ("jieba", "numpy", "scipy", "requests", "PyYAML", "python-dotenv")
            },
        }

    def snapshot(self):
        r = self.runtime
        return {
            "phase": r._phase,
            "step": r._step,
            "pending_action": r._pending_action,
            "targets": [asdict(t) for key, t in r.targets.items() if key != "overall"],
            "answers": r.answers,
            "events": r.events,
            "last_observation": r.last_observation,
            "evidence": [asdict(ref) for ref in r.tools.evidence.values()],
            "facts": r.tools.facts,
            "calculations": r.tools.calculations,
            "tool_calls": r.meter.tool_calls,
            "seen_calls": sorted(r._seen_calls),
        }

    def restore(self, state):
        r = self.runtime
        if state["phase"] not in {"plan", "act", "execute", "finished"}:
            raise RecoveryError("Invalid checkpoint phase")
        r._phase, r._step, r._pending_action = state["phase"], state["step"], state["pending_action"]
        if state["targets"]:
            r._set_plan(
                tuple(
                    ResearchTarget(
                        t["target_id"], t["question"], tuple(t["depends_on"]), tuple(t["required_facets"])
                    )
                    for t in state["targets"]
                )
            )
        for row in state["evidence"]:
            ref = EvidenceRef(**row)
            r.tools.corpus.read(ref, r.tools.allowed)
            r.tools.evidence[ref.evidence_id] = ref
        r.answers, r.events, r.last_observation = state["answers"], state["events"], state["last_observation"]
        r.tools.facts, r.tools.calculations = state["facts"], state["calculations"]
        r.meter.tool_calls, r._seen_calls = state["tool_calls"], set(state["seen_calls"])
        ledger = self.store.ledger(r.task.task_id)
        r.client.attempts = ledger["events"]
        r.client.total_attempts = r.meter.model_attempts = ledger["attempts"]
        r.meter.reserved_estimated_tokens = ledger["reserved_total"]

    def save(self, *, status="running", result=None):
        self.store.save(self.runtime.task.task_id, self.snapshot(), status=status, result=result)

    def before_attempt(self, payload):
        r = self.runtime
        if self.call_id is None:
            raise RecoveryError("A durable attempt requires a journaled model call")
        r.meter.before_attempt(payload)
        self.store.begin_attempt(
            r.task.task_id,
            self.call_id,
            r.meter.model_attempts,
            r.meter.reserved_estimated_tokens,
            r.client.config.model,
        )

    def record(self, event):
        self.store.finish_attempt(self.runtime.task.task_id, event)
        if self.original_record:
            self.original_record(event)

    def exchange(self, stage, context, messages):
        r = self.runtime
        call_id = f"{r._step}:{stage}"
        # On replay only the accounting totals may advance beyond the last checkpoint.
        # State, task limits and all substantive context must still match the cached call.
        semantic_context = {k: v for k, v in context.items() if k != "budget"}
        request_hash = stable_id("request-", [stage, semantic_context])
        cached = self.store.call(r.task.task_id, call_id, request_hash)
        if cached:
            if cached["status"] == "done":
                return cached["response"]
            error = cached["error"]
            if error["kind"] == "budget":
                raise BudgetExceeded(error["message"])
            if error["kind"] == "uncertain":
                raise UncertainRequest(error["message"])
            raise ModelCallError(error["message"])
        self.call_id = call_id
        try:
            response = r.client.chat(messages, max_tokens=1600)
        except ModelCallError as exc:
            uncertain = bool(r.client.attempts and r.client.attempts[-1]["status"] == "uncertain")
            kind = "budget" if isinstance(exc, BudgetExceeded) else ("uncertain" if uncertain else "model")
            self.store.finish_call(r.task.task_id, call_id, error={"kind": kind, "message": str(exc)})
            if uncertain:
                raise UncertainRequest(str(exc)) from None
            raise
        else:
            # Raw successful transport response is durable BEFORE JSON/schema validation.
            self.store.finish_call(r.task.task_id, call_id, response=response)
            return response
        finally:
            self.call_id = None
