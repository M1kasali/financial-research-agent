"""Generate migration hashes; fail if copied source differs from its origin."""

import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[1]
release = root.parent / "afac2026-financial-longtext-agent-team"
historic = root.parent / "AFAC赛题四_团队方案_v1_20260712"
records = []
for target in sorted((root / "vendor").rglob("*.py")):
    relative = target.relative_to(root / "vendor")
    origin_root = historic if relative.parts[0] in {"agent_team_v7", "agent_team_v9"} else release
    origin = origin_root / relative
    if digest(target) != digest(origin):
        raise RuntimeError(f"Source mismatch: {relative}")
    records.append({"target": str(target.relative_to(root)), "origin": str(origin), "sha256": digest(target)})
for name in ("test_fact_ledger.py", "test_calculation_dsl.py"):
    target = root / "tests/legacy" / name
    origin = historic / "tests" / name
    if digest(target) != digest(origin):
        raise RuntimeError(f"Test source mismatch: {name}")
    records.append({"target": str(target.relative_to(root)), "origin": str(origin), "sha256": digest(target)})
commits = {
    str(path): subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    for path in (release, historic)
}
manifest = {
    "schema_version": 1,
    "source_commits": commits,
    "files": records,
    "note": "Files reflect working-tree bytes, not necessarily committed contents. No data or secrets copied.",
}
(root / "docs/source-manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"Verified {len(records)} copied source/test files against origins")
