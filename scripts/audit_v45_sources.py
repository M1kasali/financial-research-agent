"""Read-only source/input audit; writes a NEW derived report, never runs legacy code."""

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_file(root, module):
    base = root.joinpath(*module.split("."))
    for candidate in [base.with_suffix(".py"), base / "__init__.py"]:
        if candidate.is_file():
            return candidate
    return None


def audit(release, historical, project):
    baseline = release / "evaluation_results_b1_v42_raw_pdf_accuracy_recovery/answer_results.jsonl"
    correction = release / "evaluation_results_b1_v45_fin14_unrounded/answer_result.json"
    manifest = json.loads((correction.parent / "submission_manifest.json").read_text())
    if digest(baseline) != manifest["sha256"]["v42_results"] or digest(correction) != manifest["sha256"]["fin14_correction"]:
        raise ValueError("Released answer-source hash mismatch")
    rows = [json.loads(line) for line in baseline.read_text().splitlines() if line.strip()]
    fix = json.loads(correction.read_text())
    if len(rows) != 100 or len({r["qid"] for r in rows}) != 100 or fix["qid"] != "fin_b_014":
        raise ValueError("Unexpected release question identity")
    rows = [fix if r["qid"] == fix["qid"] else r for r in rows]
    stages = Counter(call["stage"] for row in rows for call in row["calls"])
    categories = Counter()
    for row in rows:
        calls = {c["stage"] for c in row["calls"]}
        group = next((label for stage, label in [
            ("document_bound_answer_and_reasoning", "document_bound"),
            ("atomic_final_aggregation", "atomic_options"),
            ("v35_provenance_reasoning", "locked_provenance"),
        ] if stage in calls), "targeted_correction")
        categories[group] += 1
    total = sum(r["token_usage"]["total_tokens"] for r in rows)
    if total != manifest["token_usage"]["total_tokens"]:
        raise ValueError("Token ledger mismatch")
    # Parse only code; do not import modules (imports may initialize legacy clients).
    dependencies = {}
    modules = sorted((release / "agent_team_b1").glob("*.py"))
    for source in modules:
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imports.add(node.module)
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
        for name in sorted(imports):
            if name != "agent" and not name.startswith(("agent.", "agent_team_")):
                continue
            entry = dependencies.setdefault(name, {"imported_by": [], "release": None,
                                                    "historical": None, "vendor": None})
            entry["imported_by"].append(source.name)
            for label, root in [("release", release), ("historical", historical), ("vendor", project / "vendor")]:
                path = module_file(root, name)
                if path:
                    entry[label] = {"path": str(path), "sha256": digest(path)}
    migrations = []
    for name in ["focused_reasoning_v11.py", "option_coverage_v13.py"]:
        source, target = release / "agent_team_b1" / name, project / "vendor/agent_team_b1" / name
        migrations.append({"origin": str(source), "target": str(target.relative_to(project)),
                           "source_sha256": digest(source), "target_sha256": digest(target),
                           "exact_bytes": source.read_bytes() == target.read_bytes()})
    if not all(row["exact_bytes"] for row in migrations):
        raise ValueError("Migrated component differs from release source")
    return {"kind": "v45_source_and_final_record_audit_not_score_reproduction",
            "release_root": str(release), "historical_root": str(historical),
            "source_hashes_verified": True, "question_count": len(rows),
            "model_counts": dict(Counter(r["model"] for r in rows)),
            "final_record_categories": dict(categories), "retained_call_stages": dict(stages),
            "retained_call_count": sum(stages.values()), "retained_tokens": total,
            "manual_audit_row_count": sum(any(k.startswith("manual_audit") for k in r) for r in rows),
            "direct_local_import_dependencies": dependencies, "migrated_components": migrations,
            "missing_in_release_and_historical": [name for name, info in dependencies.items()
                                                   if not info["release"] and not info["historical"]],
            "limitations": ["Direct static imports only; not a complete transitive/runtime dependency proof",
                            "Saved final calls omit some historical generation work; not all development costs",
                            "No gold or answer rows copied into production interface",
                            "Atomic-stage source entrypoints not identified from final records alone",
                            "No provider calls, score reproduction or model-quality evaluation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new report path; previous audit results are immutable")
    report = audit(args.release.resolve(), args.historical.resolve(), Path(__file__).resolve().parents[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()),
                      "categories": report["final_record_categories"],
                      "migrations": len(report["migrated_components"]),
                      "missing_dependencies": report["missing_in_release_and_historical"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
