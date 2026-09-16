"""Re-render saved comparison artifacts OFFLINE; does not rerun or reevaluate the model."""

import argparse
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from financial_agent.core_answer import render_core_answer
from financial_agent.diagnostics import annotate_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source_dir.resolve()
    os.umask(0o077)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "experiments/runs" / f"diagnostic-report-replay-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    audit = {"execution": "offline_render_of_saved_results_NOT_model_rerun",
             "real_api_calls": 0, "new_answer_accuracy": None, "source_dir": str(source), "cases": {}}
    with patch("socket.socket", side_effect=AssertionError("Offline replay forbids network")):
        manifest = json.loads((source / "manifest.json").read_text())
        for name in manifest["order"]:
            if Path(name).name != name or name in {".", ".."} or "\\" in name:
                raise ValueError("Invalid artifact name")
            path = source / f"{name}.json"
            if not path.exists():
                audit["cases"][name] = {"status": "not_executed_in_original_batch"}
                continue
            raw = path.read_bytes()
            original = json.loads(raw)
            result = annotate_diagnostics(deepcopy(original))
            assert result["answer"] == original["answer"] and result["claims"] == original["claims"]
            report = render_core_answer(result)
            if result["status"] != "answered":
                assert not result["claims"] and not result["answer"]
                assert "诊断原文（未核验" not in report  # Details remain opt-in.
                for gap in result["gaps"]:
                    assert gap not in report
            prefix = "说明：这是旧模型结果的离线展示重放；未重新运行模型或验证新版提示效果。\n\n"
            (output / f"{name}.md").write_text(prefix + report, encoding="utf-8")
            # Diagnostic-only copy is separate from original evidence/result artifacts.
            (output / f"{name}-audit.md").write_text(
                prefix + render_core_answer(result, include_unverified_details=True), encoding="utf-8")
            assert path.read_bytes() == raw
            audit["cases"][name] = {"original_status": original["status"],
                                     "original_sha256": hashlib.sha256(raw).hexdigest(),
                                     "source_result_unchanged": True,
                                     "raw_diagnostic_count": len(result["gaps"]),
                                     "original_answer_and_claims_preserved": True}
    (output / "summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(f"OUTPUT={output}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
