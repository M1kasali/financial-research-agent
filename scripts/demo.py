"""Offline, hash-pinned replay of saved real runs. No inference, .env read or source revalidation."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from financial_agent.core_answer import render_core_answer


def read_record(root, relative, expected_hash):
    path = (root / relative).resolve()
    allowed = (root / "experiments/runs").resolve()
    if not path.is_relative_to(allowed) or not path.is_file():
        raise ValueError("Missing or out-of-scope saved result; do not rerun models automatically")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("Saved result changed; stop replay")
    result = json.loads(raw)
    if not isinstance(result.get("events"), list):
        raise ValueError("Saved result lacks execution events")
    return result


def replay_report(case, result):
    steps = [e["stage"] for e in result["events"] if e.get("status") == "started"
             or e.get("stage") in {"refinement_search", "financial_workflow"}]
    return "\n".join([
        "# 历史真实运行回放（不是现场模型执行）", "",
        "本次仅校验保存文件的SHA256并重新排版；新增API调用为0，不重新检索、计算或审定来源。",
        "原结果属于已知开发案例，不是独立准确率或B榜成绩复现。", "",
        f"案例：{case['title']}", "", f"讲解重点：{case['focus']}", "",
        f"原记录：`{case['result']}`", "", f"SHA256：`{case['sha256']}`", "",
        "原运行步骤：" + " → ".join(steps), "",
        f"原运行模型请求：{result['usage']['model_attempts']}；"
        f"原运行报告token：{result['usage'].get('reported_tokens')}（不是本次回放用量）。", "",
        render_core_answer(result),
    ])


def main(argv=None, *, root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", "growth", "followup", "clauses"], default="all")
    parser.add_argument("--list", action="store_true", help="List the three saved cases without replay")
    parser.add_argument("--output-dir", type=Path, help="NEW directory under experiments/demos; auto-generated if omitted")
    args = parser.parse_args(argv)
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    os.umask(0o077)
    with patch("socket.socket", side_effect=AssertionError("Demo is offline; network forbidden")):
        catalog = json.loads((root / "configs/demo-cases.json").read_text())
        if catalog["mode"] != "recorded_real_run_replay":
            raise ValueError("Only explicit recorded replay is supported")
        cases = catalog["cases"]
        if len(cases) != 3 or {c["id"] for c in cases} != {"growth", "followup", "clauses"}:
            raise ValueError("Unexpected demo case catalog")
        if args.list:
            print("历史真实运行回放；不读密钥、不调用模型。")
            for case in cases:
                print(f"{case['id']}: {case['title']}")
            return 0
        selected = [c for c in cases if args.case == "all" or c["id"] == args.case]
        records = []
        for case in selected:
            result = read_record(root, case["result"], case["sha256"])
            failure = None
            if "failure_result" in case:
                failure = read_record(root, case["failure_result"], case["failure_sha256"])
                if failure["status"] == "answered" or failure["answer"] or failure["claims"]:
                    raise ValueError("Failure comparison must not contain published claims")
            records.append((case, result, failure))
        base = (root / "experiments/demos").resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ((root / args.output_dir) if args.output_dir else base / f"replay-{stamp}-{uuid4().hex[:8]}").resolve()
        if output == base or not output.is_relative_to(base):
            raise ValueError("Output must be a new subdirectory of experiments/demos")
        output.mkdir(parents=True, exist_ok=False)
        index = ["# 金融长文本研究 Agent：演示回放", "",
                 "模式：历史真实运行回放。新增API调用：0。不是现场推理，不是当前源码端到端重跑。", ""]
        for case, result, failure in records:
            (output / f"{case['id']}.md").write_text(replay_report(case, result), encoding="utf-8")
            index.extend([f"- [{case['title']}]({case['id']}.md)：原状态 `{result['status']}`。", ""])
            if failure:
                failure_case = {**case, "result": case["failure_result"], "sha256": case["failure_sha256"]}
                (output / f"{case['id']}-failure.md").write_text(replay_report(failure_case, failure), encoding="utf-8")
                index.extend([f"  [同题旧引用失败记录]({case['id']}-failure.md)：保留拒绝结果，不发布候选。", ""])
        (output / "README.md").write_text("\n".join(index), encoding="utf-8")
        manifest = {"mode": catalog["mode"], "new_api_calls": 0, "fresh_agent_execution": False,
                    "source_revalidated": False, "answer_accuracy": None,
                    "cases": selected, "generated_at": datetime.now(timezone.utc).isoformat()}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"REPLAY_ONLY new_api_calls=0 cases={len(records)} OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
