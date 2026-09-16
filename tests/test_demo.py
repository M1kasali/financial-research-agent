import hashlib
import importlib
import json
from pathlib import Path

import pytest
from test_core_answer import candidate, engine, review


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("demo")


@pytest.fixture
def demo_root(tmp_path, corpus):
    runner, _ = engine(corpus, [candidate, review])
    result = runner.run()
    (tmp_path / "experiments/runs").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    raw = json.dumps(result, ensure_ascii=False).encode()
    (tmp_path / "experiments/runs/result.json").write_bytes(raw)
    cases = [{"id": key, "title": "测试案例", "focus": "测试回放协议",
              "result": "experiments/runs/result.json", "sha256": hashlib.sha256(raw).hexdigest()}
             for key in ("growth", "followup", "clauses")]
    (tmp_path / "configs/demo-cases.json").write_text(json.dumps({"mode": "recorded_real_run_replay", "cases": cases}))
    return tmp_path


def test_list_never_reads_result_or_credentials(demo, demo_root, monkeypatch, capsys):
    monkeypatch.setattr(demo, "read_record", lambda *a: pytest.fail("Unexpected result read"))
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a: pytest.fail("Unexpected secret read"))
    assert demo.main(["--list"], root=demo_root) == 0
    assert "不调用模型" in capsys.readouterr().out
    assert not (demo_root / "experiments/demos").exists()


def test_replay_marked_offline_and_refuses_overwrite(demo, demo_root, monkeypatch):
    monkeypatch.setattr("financial_agent.model.RecordedModelClient.chat", lambda *a, **k: pytest.fail("Unexpected inference"))
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a: pytest.fail("Unexpected secret read"))
    args = ["--output-dir", "experiments/demos/test"]
    assert demo.main(args, root=demo_root) == 0
    output = demo_root / "experiments/demos/test"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["new_api_calls"] == 0 and not manifest["fresh_agent_execution"]
    assert not manifest["source_revalidated"] and len(manifest["cases"]) == 3
    assert "不是现场模型执行" in (output / "growth.md").read_text()
    with pytest.raises(FileExistsError):
        demo.main(args, root=demo_root)


def test_single_case(demo, demo_root):
    demo.main(["--case", "followup", "--output-dir", "experiments/demos/one"], root=demo_root)
    assert (demo_root / "experiments/demos/one/followup.md").exists()
    assert not (demo_root / "experiments/demos/one/growth.md").exists()


@pytest.mark.parametrize("kind", ["hash", "missing", "outside", "output"])
def test_replay_rejects_bad_input_without_model_fallback(demo, demo_root, kind):
    config = demo_root / "configs/demo-cases.json"
    catalog = json.loads(config.read_text())
    if kind == "hash":
        catalog["cases"][0]["sha256"] = "0" * 64
    elif kind == "missing":
        catalog["cases"][0]["result"] = "experiments/runs/missing.json"
    elif kind == "outside":
        catalog["cases"][0]["result"] = "configs/demo-cases.json"
    config.write_text(json.dumps(catalog))
    with pytest.raises(ValueError):
        demo.main(["--output-dir", "docs/not-allowed" if kind == "output" else "experiments/demos/bad"], root=demo_root)
    assert not (demo_root / "experiments/demos/bad").exists()
