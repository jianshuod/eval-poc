from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_run_eval_salt_module():
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "run-eval-salt.py"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location("run_eval_salt", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_eval_salt"] = module
    spec.loader.exec_module(module)
    return module


def test_run_eval_still_attempts_token_stats_when_eval_fails(monkeypatch, tmp_path: Path) -> None:
    module = _load_run_eval_salt_module()

    # Fake benchmark venv binaries
    inspect_bin = tmp_path / "venv" / "bin" / "inspect"
    python_bin = tmp_path / "venv" / "bin" / "python"
    inspect_bin.parent.mkdir(parents=True, exist_ok=True)
    inspect_bin.write_text("", encoding="utf-8")
    python_bin.write_text("", encoding="utf-8")

    # Grid-search style combo directory avoids nested timestamp paths.
    combo_dir = tmp_path / "combo"
    combo_dir.mkdir(parents=True, exist_ok=True)

    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append([str(part) for part in cmd])
        if len(calls) == 1:
            # Simulate eval failure.
            return subprocess.CompletedProcess(cmd, 1)
        # Token stats subprocess should still be attempted.
        return subprocess.CompletedProcess(cmd, 0, stdout="token ok", stderr="")

    monkeypatch.setattr(module, "get_venv_inspect", lambda _benchmark: inspect_bin)
    monkeypatch.setattr(module, "_ensure_openai_min_version", lambda _benchmark: True)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    return_code = module.run_eval(
        benchmark_name="agentharm",
        task_spec="inspect_evals/agentharm",
        config={"source": "upstream"},
        model="openai/alicloud-qwen3.5-35b-a3b",
        inspect_args=[],
        task_config={"name": "agentharm"},
        run_name="token-stats-test",
        grid_search_combo_dir=combo_dir,
    )

    assert return_code == 1
    assert len(calls) >= 2
    assert calls[1][1].endswith("token_stats_generator.py")
