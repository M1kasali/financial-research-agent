"""Single-task, model-driven research loop. No hidden nested legacy model calls.

Synchronous execution with optional local checkpoints. Model review is not a truth oracle.
"""

import json
from dataclasses import asdict

from agent.reasoning.logicrag import sanitize_logic_plan
from agent.reasoning.retrieval_refiner import parse_logicrag_sufficiency_judgement
from agent.schemas import LogicNode, LogicPlan

from financial_agent.contracts import TaskRequest
from financial_agent.corpus import Corpus
from financial_agent.durability import DurableSession
from financial_agent.model import ModelCallError, RecordedModelClient
from financial_agent.policy import BudgetExceeded, JSONPolicy, TaskMeter
from financial_agent.research_tools import ResearchTools, id_list, require_keys, stable_id
from financial_agent.retrieval import ScopedRetriever
from financial_agent.storage import SQLiteTaskStore, UncertainRequest
from financial_agent.task_plan import ResearchTarget, validate_plan


def parse_targets(rows: object) -> tuple[ResearchTarget, ...]:
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        raise ValueError("Plan needs 1–8 targets")
    targets = []
    for row in rows:
        require_keys(row, {"target_id", "question", "depends_on", "required_facets"})
        if not isinstance(row["depends_on"], list) or not isinstance(row["required_facets"], list):
            raise ValueError("Plan dependencies and facets must be arrays")
        target = ResearchTarget(
            row["target_id"], row["question"], tuple(row["depends_on"]), tuple(row["required_facets"])
        )
        if len(target.target_id) > 80 or len(target.question) > 2000 or len(target.required_facets) > 8:
            raise ValueError("Plan target too large")
        if (
            any(len(f) > 300 for f in target.required_facets)
            or target.target_id == "overall"
            or "overall" in target.depends_on
        ):
            raise ValueError("Invalid target requirements or reserved overall ID")
        targets.append(target)
    return tuple(targets)


class ResearchRuntime:
    def __init__(
        self,
        corpus: Corpus,
        task: TaskRequest,
        client: RecordedModelClient,
        *,
        retriever: ScopedRetriever | None = None,
        execution_label: str = "model_driven",
        store: SQLiteTaskStore | None = None,
    ):
        if client.total_attempts or client.before_attempt is not None:
            raise ValueError("Runtime requires an unused, dedicated model client")
        self.task, self.client = task, client
        self.meter = TaskMeter(task.budget)
        self.client.before_attempt = self.meter.before_attempt
        self.policy = JSONPolicy(client)
        self.tools = ResearchTools(corpus, retriever or ScopedRetriever(corpus, tokenizer_mode="char"), task)
        self.execution_label = execution_label
        self.targets: dict[str, ResearchTarget] = {}
        self.answers: dict[str, dict] = {}
        self.events: list[dict] = []
        self.last_observation: dict = {}
        self.levels: list = []
        self._used = False
        self._seen_calls: set[str] = set()
        self._phase, self._step, self._pending_action = "plan", 0, None
        self.store = store
        self._durable = DurableSession(self, store) if store else None
        if self._durable:
            self.policy = JSONPolicy(client, exchange=self._durable.exchange)
            self.client.before_attempt = self._durable.before_attempt
            self.client._record_sink = self._durable.record

    def _set_plan(self, targets: tuple[ResearchTarget, ...]):
        validate_plan(targets)  # Reject malformed DAGs before legacy sanitization can hide defects.
        logic = LogicPlan(
            nodes=[
                LogicNode(node_id=t.target_id, text=t.question, depends_on=list(t.depends_on))
                for t in targets
            ]
        )
        logic = sanitize_logic_plan(logic, max_subproblems=8, max_ranks=8)
        actual = {n.node_id: (n.text, tuple(n.depends_on)) for n in logic.nodes}
        expected = {t.target_id: (t.question, t.depends_on) for t in targets}
        if actual != expected:
            raise ValueError("Legacy plan normalization changed required targets")
        self.levels = logic.topological_levels()
        self.targets = {t.target_id: t for t in targets}
        self.targets["overall"] = ResearchTarget(
            "overall", self.task.query, tuple(self.targets), ("完整覆盖原始用户请求",)
        )

    def _ready(self) -> list[str]:
        return [
            key
            for key, target in self.targets.items()
            if key not in self.answers and set(target.depends_on) <= self.answers.keys()
        ]

    def _observe(self) -> dict:
        return {
            "task": asdict(self.task),
            "targets": [asdict(t) for t in self.targets.values()],
            "ready_target_ids": self._ready(),
            "verified_answers": self.answers,
            "evidence_catalog": [asdict(ref) for ref in self.tools.evidence.values()],
            "facts": [self.tools.facts[key] for key in sorted(self.tools.facts)],
            "calculations": [self.tools.calculations[key] for key in sorted(self.tools.calculations)],
            "last_observation": self.last_observation,
            "recent_actions": [{k: v for k, v in e.items() if k != "observation"} for e in self.events[-6:]],
            "budget": self.meter.summary(self.client),
        }

    def _verify(self, target_id: str, args: dict) -> dict:
        require_keys(args, {"answer", "evidence_ids"}, {"calculation_ids"})
        answer = args["answer"]
        if not isinstance(answer, str) or not 1 <= len(answer.strip()) <= 4000:
            raise ValueError("Answer must contain 1–4000 characters")
        evidence_ids = id_list(args["evidence_ids"], maximum=8)
        evidence = []
        for key in evidence_ids:
            text = self.tools.read_evidence(key)
            # Explicit truncation: reviewer must not assume unseen text supports a claim.
            evidence.append(
                {"ref": asdict(self.tools.evidence[key]), "text": text[:6000], "truncated": len(text) > 6000}
            )
        calc_ids = args.get("calculation_ids", [])
        if calc_ids != []:
            id_list(calc_ids, maximum=8)
        calculations = []
        for key in calc_ids:
            if key not in self.tools.calculations:
                raise ValueError("Unknown calculation ID")
            calc = self.tools.calculations[key]
            if not set(calc["evidence_ids"]) <= set(evidence_ids):
                raise ValueError("Calculation sources must be included in citations")
            calculations.append(calc)
        review = self.policy.ask(
            "verify",
            {
                "task_query": self.task.query,
                "target": asdict(self.targets[target_id]),
                "answer": answer,
                "evidence": evidence,
                "calculations": calculations,
                "calculation_facts": [
                    self.tools.facts[key] for calc in calculations for key in calc["operands"]
                ],
            },
        )
        require_keys(
            review,
            {"supported", "sufficient", "failure_tags", "reason", "missing_evidence", "next_search_goal"},
        )
        if type(review["supported"]) is not bool or type(review["sufficient"]) is not bool:
            raise ValueError("Review verdicts must be booleans")
        if any(
            not isinstance(review[k], str) or len(review[k]) > 2000
            for k in ("reason", "missing_evidence", "next_search_goal")
        ):
            raise ValueError("Invalid review explanation")
        if not isinstance(review["failure_tags"], list) or any(
            not isinstance(v, str) for v in review["failure_tags"]
        ):
            raise ValueError("Invalid review tags")
        judgement = parse_logicrag_sufficiency_judgement(json.dumps(review, ensure_ascii=False))
        accepted = review["supported"] and judgement.sufficient
        result = {"accepted": accepted, "review": review, "review_kind": "model_judgement_not_ground_truth"}
        if accepted:
            self.answers[target_id] = {
                "answer": answer,
                "evidence_ids": evidence_ids,
                "calculation_ids": calc_ids,
                **result,
            }
        return result

    def _tool(self, action: dict) -> dict:
        require_keys(action, {"kind", "target_id", "tool", "args"})
        self.meter.tool()  # Invalid attempts also consume the tool budget.
        target_id, name = action["target_id"], action["tool"]
        if not isinstance(target_id, str) or target_id not in self._ready():
            raise ValueError("Target is not ready or is already complete")
        if not isinstance(name, str) or name not in {
            "search", "search_documents", "expand_context", "read", "extract", "bind_statement", "calculate", "verify"
        }:
            raise ValueError("Unknown tool")
        # A failed verification can be repeated after NEW evidence, but not in a no-progress loop.
        signature = stable_id(
            "call-",
            [
                action,
                sorted(self.tools.evidence),
                sorted(self.tools.facts),
                sorted(self.tools.calculations),
                sorted(self.answers),
            ],
        )
        if signature in self._seen_calls:
            raise ValueError("Repeated call without new state; change strategy or ask the user")
        self._seen_calls.add(signature)
        if name == "verify":
            return self._verify(target_id, action["args"])
        return getattr(self.tools, name)(action["args"])

    def run(self, *, stop_after_steps: int | None = None) -> dict:
        """Optionally pause at a committed step boundary; no automatic uncertain retry."""
        if stop_after_steps is not None and (
            type(stop_after_steps) is not int or not 1 <= stop_after_steps <= 100
        ):
            raise ValueError("stop_after_steps must be in [1,100]")
        if stop_after_steps is not None and self.store is None:
            raise ValueError("Pausing requires a durable store")
        if self._used:
            raise ValueError("Runtime is single-use; create a new task runtime")
        self._used = True
        if self.store is None:
            return self._run_loop(stop_after_steps)
        with self.store.exclusive():
            record = self.store.start(self.task.task_id, self._durable.identity(), self._durable.snapshot())
            self._durable.restore(record["checkpoint"])
            if record["result"] is not None:
                return record["result"]
            if self.store.ledger(self.task.task_id)["pending_calls"]:
                result = self._result(
                    "needs_attention", "Model outcome is uncertain; automatic resend is forbidden"
                )
                self._durable.save(status="needs_attention", result=result)
                return result
            return self._run_loop(stop_after_steps)

    def _checkpoint(self):
        if self._durable:
            self._durable.save()

    def _run_loop(self, stop_after_steps) -> dict:
        status, reason = "partial", "Action limit reached"
        processed_steps = 0
        try:
            if self._phase == "plan":
                plan = self.policy.ask(
                    "plan",
                    {
                        "task": asdict(self.task),
                        "documents": [
                            asdict(self.tools.corpus.by_version[v]) for v in self.task.document_version_ids
                        ],
                    },
                )
                require_keys(plan, {"targets"})
                self._set_plan(parse_targets(plan["targets"]))
                self.events.append({"kind": "plan", "targets": list(self.targets), "levels": self.levels})
                self._phase, self._step = "act", 1
                self._checkpoint()
            # Structural guard also bounds invalid JSON/actions and plan extensions.
            while self._step <= min(
                100, self.task.budget.max_model_attempts + self.task.budget.max_tool_calls
            ):
                step = self._step
                try:
                    if self._phase == "act":
                        self._pending_action = self.policy.ask("act", self._observe())
                        self._phase = "execute"
                        self._checkpoint()
                    action = self._pending_action
                    kind = action.get("kind")
                    if kind == "finish":
                        require_keys(action, {"kind"})
                        if self.targets.keys() != self.answers.keys():
                            raise ValueError("Cannot finish: unverified targets remain")
                        status, reason = (
                            "completed",
                            "All targets passed model review with version-bound citations",
                        )
                        self.events.append({"step": step, "kind": "finish"})
                        break
                    if kind == "ask_user":
                        require_keys(action, {"kind", "question"})
                        if (
                            not isinstance(action["question"], str)
                            or not 1 <= len(action["question"].strip()) <= 2000
                        ):
                            raise ValueError("Invalid clarification question")
                        status, reason = "needs_input", action["question"]
                        self.events.append({"step": step, **action})
                        break
                    if kind == "extend_plan":
                        require_keys(action, {"kind", "targets"})
                        if "overall" in self.answers:
                            raise ValueError("Cannot extend an already reviewed overall answer")
                        existing = tuple(t for k, t in self.targets.items() if k != "overall")
                        additions = parse_targets(action["targets"])
                        if len(existing) + len(additions) > 8:
                            raise ValueError("Plan capacity reached")
                        self._set_plan(existing + additions)
                        observation = {"added_targets": [t.target_id for t in additions]}
                    elif kind == "tool":
                        observation = self._tool(action)
                    else:
                        raise ValueError("Unknown action kind")
                    self.last_observation = observation
                    self.events.append({"step": step, "action": action, "observation": observation})
                except (ValueError, KeyError, TypeError, PermissionError, ArithmeticError) as exc:
                    self.last_observation = {"error": str(exc), "error_type": type(exc).__name__}
                    self.events.append(
                        {"step": step, "kind": "rejected", "observation": self.last_observation}
                    )
                self._phase, self._step, self._pending_action = "act", step + 1, None
                self._checkpoint()
                processed_steps += 1
                if stop_after_steps is not None and processed_steps >= stop_after_steps:
                    status, reason = "paused", "Paused at a committed step boundary"
                    break
        except UncertainRequest as exc:
            status, reason = "needs_attention", str(exc)
        except BudgetExceeded as exc:
            reason = str(exc)
        except ModelCallError as exc:
            status, reason = "failed", str(exc)
        except (ValueError, KeyError, TypeError) as exc:
            status, reason = "failed", f"Invalid initial plan: {exc}"
        # Re-read every retained citation before handing off; source mutation invalidates the run.
        try:
            for ref in self.tools.evidence.values():
                self.tools.corpus.read(ref, self.tools.allowed)
        except (ValueError, PermissionError) as exc:
            status, reason = "failed", str(exc)
            self.answers.clear()
        result = self._result(status, reason)
        if self._durable:
            if status not in {"paused", "needs_attention"}:
                self._phase = "finished"
            self._durable.save(status=status, result=None if status == "paused" else result)
        return result

    def _result(self, status, reason):
        return {
            "task_id": self.task.task_id,
            "session_id": self.task.session_id,
            "execution_label": self.execution_label,
            "status": status,
            "reason": reason,
            "answers": self.answers,
            "unresolved_target_ids": [k for k in self.targets if k not in self.answers],
            "evidence": [asdict(ref) for ref in self.tools.evidence.values()],
            "facts": list(self.tools.facts.values()),
            "calculations": list(self.tools.calculations.values()),
            "events": self.events,
            "budget": self.meter.summary(self.client),
            "model_attempts": self.client.attempts,
            "limitations": [
                "Model verification is not ground-truth correctness",
                "Local checkpoints; uncertain requests require attention"
                if self.store
                else "In-memory; no crash recovery",
                "No investment recommendations or trading execution",
            ],
        }
