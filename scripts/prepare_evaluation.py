"""Freeze historical replay questions and DRAFT new tasks; never invent gold labels."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from run_retrieval_smoke import read_questions

from financial_agent.corpus import Corpus
from financial_agent.evaluation import EvalCase, EvalSuite


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_suite(directory, suite):
    directory.mkdir()
    write_json(directory / "cases.json", suite.export())
    write_json(
        directory / "labels.pending.json",
        {
            "suite_hash": suite.fingerprint,
            "labels": {
                c.case_id: {
                    "status": "pending",
                    "expected": None,
                    "provenance": "awaiting_independent_source_review",
                    "reviewer": None,
                }
                for c in suite.cases
            },
        },
    )


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/datasets.example.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    chunks = (root / config["chunks"]).resolve()
    corpus = Corpus.from_jsonl(chunks)
    questions_dir = (root / config["questions_b"]).resolve()
    questions = read_questions(questions_dir)
    by_domain = {
        domain: tuple(v.version_id for v in corpus.documents.values() if v.domain == domain)
        for domain in sorted({v.domain for v in corpus.documents.values()})
    }
    replay = []
    for q in sorted(questions, key=lambda q: q["qid"]):
        query = (
            q["question"] + "\n" + "\n".join(f"{key}. {value}" for key, value in q.get("options", {}).items())
        )
        replay.append(
            EvalCase(
                q["qid"], query, by_domain[q["domain"]], q["domain"], "competition_qa", "historical_replay"
            )
        )
    # New wording on old documents is a candidate set, NOT an independent held-out benchmark.
    recipes = [
        (
            "financial_reports",
            "extraction",
            "从指定财报中提取2025年度合并口径的营业收入、归母净利润和经营现金流净额，逐项列出期间、单位和出处；缺少的字段明确标注。",
        ),
        (
            "financial_reports",
            "calculation",
            "根据指定财报计算2025年度营业收入相对2024年度的同比变化，说明主体、币种、合并口径、原始数值和公式。口径不明时不要计算。",
        ),
        (
            "financial_reports",
            "comparison",
            "比较指定的不同公司财报中2025年度经营现金流与净利润的关系；先核对期间和口径，未披露的差异原因不能推断成事实。",
        ),
        (
            "financial_contracts",
            "clause_verification",
            "整理指定合同中提前终止的条件、通知期限、费用及例外，核查补充条款是否改变主条款，并分别引用。",
        ),
        (
            "financial_contracts",
            "extraction",
            "提取指定合同中的违约责任、免责条件和适用主体；区分一般约定与特别约定，不能仅引用标题。",
        ),
        (
            "insurance",
            "clause_verification",
            "核查指定保险条款中等待期、责任免除和理赔材料要求，说明哪些条件需要同时满足；没有找到不等于不存在。",
        ),
        (
            "insurance",
            "comparison",
            "对比指定两份保险文件对赔付条件及除外责任的约定，保留各自产品范围，不要混成统一规则。",
        ),
        (
            "regulatory",
            "clause_verification",
            "仅依据指定监管材料，列出适用主体、关键义务、生效信息及例外；区分文件中的明确条款和你无法确认的内容。",
        ),
        (
            "regulatory",
            "extraction",
            "提取指定监管文件中资料保存或报告要求的期限、起算条件、义务主体及条文出处。资料未涉及的事项明确说明。",
        ),
        (
            "research",
            "report",
            "根据指定研报写一份带引用的短研究摘要，分为已披露事实、预测假设、风险及证据缺口，不把预测写成已实现业绩。",
        ),
        (
            "research",
            "evidence_gap",
            "指定研报是否披露了其预测在2028年得到审计验证的实际结果？若证据不足，请明确说明不能据此确认，而非补造结果。",
        ),
        (
            "financial_reports",
            "scope_change",
            "本轮要求只使用母公司口径而不是合并口径，重新提取2025年度营业收入及净利润并说明出处；不要沿用不适用的合并数值。",
        ),
    ]
    candidates = []
    for index, (domain, kind, query) in enumerate(recipes, 1):
        versions = by_domain[domain][:2]
        if kind == "comparison" and domain == "financial_reports":
            selected = [
                v.version_id
                for v in corpus.documents.values()
                if v.doc_id in {"annual_byd_2025_report", "annual_midea_2025_report"}
            ]
            if len(selected) != 2:
                raise ValueError("Expected comparison documents are missing; do not silently substitute")
            versions = tuple(selected)
        candidates.append(EvalCase(f"draft-{index:03}", query, versions, domain, kind, "candidate"))
    output = (
        root
        / "data/local"
        / ("evaluation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    output.mkdir(parents=True, exist_ok=False)
    export_suite(output / "historical_b", EvalSuite("afac-b-historical-replay-v1", tuple(replay)))
    export_suite(output / "new_tasks", EvalSuite("financial-research-candidates-v1", tuple(candidates)))
    manual_path = (root / config["manual_devset"]).resolve()
    manual = [
        json.loads(line) for line in manual_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    current_ids = {q["qid"] for q in questions}
    audit = [
        {
            "qid": row["qid"],
            "provenance": row["provenance"],
            "question_present_in_configured_b": row["qid"] in current_ids,
            "missing_doc_ids": [d for d in row["required_doc_ids"] if d not in corpus.documents],
            "missing_anchor_ids": [c for c in row["required_chunk_ids"] if c not in corpus.chunks],
            "imported_as_gold": False,
        }
        for row in manual
    ]
    inputs = [
        config_path,
        chunks,
        manual_path,
        *(p for p in sorted(questions_dir.iterdir()) if p.suffix in {".json", ".jsonl"}),
    ]
    manifest = {
        "historical_cases": len(replay),
        "candidate_cases": len(candidates),
        "historical_domain_counts": dict(Counter(c.domain for c in replay)),
        "candidate_task_kinds": dict(Counter(c.task_kind for c in candidates)),
        "approved_gold_count": 0,
        "external_model_requests": 0,
        "manual_devset_audit": audit,
        "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
        "limitations": [
            "Historical B questions were used during competition development; not held out.",
            "New tasks are assistant-authored drafts on the old corpus; human validation/splitting still required.",
            "scope_change is a standalone query, not a multi-turn memory test.",
            "No official per-question B gold answers or complete end-to-end quality claims.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "document_catalog.json", [asdict(v) for v in corpus.documents.values()])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Evaluation bundle: {output}")


if __name__ == "__main__":
    main()
