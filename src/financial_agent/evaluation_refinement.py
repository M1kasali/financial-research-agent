"""Version development questions without relabeling drafts as approved gold."""

from copy import deepcopy
from dataclasses import replace

from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalSuite
from financial_agent.reference_drafts import compile_drafts
from financial_agent.research_tools import stable_id


def refine_candidates(parent: EvalSuite, reference_spec: dict, recipe: dict, corpus: Corpus):
    if recipe.get("schema_version") != 1 or recipe.get("parent_suite_hash") != parent.fingerprint:
        raise ValueError("Refinement parent/schema mismatch")
    # Validate the parent bindings before deriving anything, but never approve them.
    compile_drafts(reference_spec, parent, corpus)
    if recipe.get("new_suite_name") == parent.name or not recipe.get("new_suite_name"):
        raise ValueError("Revision needs a new suite name")
    if not isinstance(recipe.get("id_prefix"), str) or not recipe["id_prefix"]:
        raise ValueError("New case ID prefix required")
    changes = recipe["changes"]
    updates = {change["case_id"]: change for change in changes}
    if len(updates) != len(changes) or not set(updates) <= {c.case_id for c in parent.cases}:
        raise ValueError("Unknown or duplicate refinement cases")
    drafts = {d["case_id"]: d for d in reference_spec["drafts"]}
    cases, new_drafts, lineage = [], [], []
    for index, old in enumerate(parent.cases, 1):
        change = updates.get(old.case_id, {})
        if change and (not change.get("reason") or not change.get("query")):
            raise ValueError("Each revision needs a reason and public query")
        versions = old.document_version_ids
        if "document_ids" in change:
            requested = change["document_ids"]
            if not requested or len(requested) != len(set(requested)):
                raise ValueError("Explicit unique nonempty narrowed document scope required")
            lookup = {corpus.by_version[v].doc_id: v for v in versions}
            if not set(requested) <= lookup.keys():
                raise ValueError("Refinement must not expand source scope")
            versions = tuple(lookup[doc_id] for doc_id in requested)
        new_id = f"{recipe['id_prefix']}{index:03d}"
        if new_id in {c.case_id for c in parent.cases}:
            raise ValueError("Revision must not reuse parent case IDs")
        case = replace(old, case_id=new_id, query=change.get("query", old.query),
                       document_version_ids=versions)
        draft = deepcopy(drafts[old.case_id])
        draft_update = change.get("draft", {})
        if set(draft_update) - {"coverage", "claims", "gaps", "rubric", "searches", "calculations"}:
            raise ValueError("Refinement cannot set approval or case identity")
        draft.update(deepcopy(draft_update))
        draft["case_id"] = new_id
        cases.append(case)
        new_drafts.append(draft)
        lineage.append({"parent_case_id": old.case_id, "case_id": new_id,
                        "query_changed": case.query != old.query,
                        "scope_changed": versions != old.document_version_ids,
                        "reason": change.get("reason", "Unchanged question; carried into new development suite"),
                        "old_query": old.query, "new_query": case.query,
                        "old_scope": old.document_version_ids, "new_scope": versions})
    suite = EvalSuite(recipe["new_suite_name"], tuple(cases))
    anchors = deepcopy(reference_spec["anchors"])
    additions = recipe.get("added_anchors", {})
    if anchors.keys() & additions.keys():
        raise ValueError("New anchors must not overwrite old anchor definitions")
    anchors.update(deepcopy(additions))
    spec = {"schema_version": 1, "suite_hash": suite.fingerprint,
            "provenance": recipe["provenance"], "anchors": anchors, "drafts": new_drafts}
    packet = compile_drafts(spec, suite, corpus)
    labels = {"suite_hash": suite.fingerprint, "labels": {
        c.case_id: {"status": "pending", "expected": None, "reviewer": None,
                    "provenance": "awaiting_independent_source_review"} for c in cases}}
    manifest = {"parent_suite_hash": parent.fingerprint, "suite_hash": suite.fingerprint,
                "parent_spec_hash": stable_id("draft-", reference_spec),
                "recipe_hash": stable_id("refinement-", recipe), "lineage": lineage,
                "partition": "candidate", "usage": "development_only",
                "comparable_to_parent_scores": False, "approved_gold_count": 0}
    return suite, spec, packet, labels, manifest
