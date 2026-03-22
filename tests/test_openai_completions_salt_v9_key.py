from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_openai_completions_module():
    project_root = Path(__file__).resolve().parent.parent
    inspect_src = project_root / "upstream" / "inspect_ai" / "src"
    if str(inspect_src) not in sys.path:
        sys.path.insert(0, str(inspect_src))
    return importlib.import_module("inspect_ai.model._providers.openai_completions")


def test_inject_v9_safety_identifier_from_active_sample(monkeypatch) -> None:
    module = _load_openai_completions_module()
    samples_module = importlib.import_module("inspect_ai.log._samples")

    active_sample = SimpleNamespace(
        task="inspect_evals/agentharm",
        epoch=1,
        sample=SimpleNamespace(id="case-42"),
    )
    monkeypatch.setattr(samples_module, "sample_active", lambda: active_sample)

    request = {
        "extra_body": {
            "safety_lookahead": True,
            "safety_lookahead_version": "v9",
        }
    }
    module._inject_v9_safety_identifier(request)

    assert request["extra_body"]["case_id"] == "case-42"
    assert request["extra_body"]["trace_id"] == "inspect_evals/agentharm:1"
    assert request["extra_body"]["safety_state_key"] == "inspect_evals/agentharm:case-42:1"


def test_inject_v9_safety_identifier_does_not_override_explicit_key(monkeypatch) -> None:
    module = _load_openai_completions_module()
    samples_module = importlib.import_module("inspect_ai.log._samples")

    active_sample = SimpleNamespace(
        task="inspect_evals/agentharm",
        epoch=2,
        sample=SimpleNamespace(id="case-99"),
    )
    monkeypatch.setattr(samples_module, "sample_active", lambda: active_sample)

    request = {
        "extra_body": {
            "safety_lookahead": True,
            "safety_lookahead_version": "v9",
            "safety_state_key": "pre-set-key",
        }
    }
    module._inject_v9_safety_identifier(request)

    assert request["extra_body"]["safety_state_key"] == "pre-set-key"
    assert "case_id" not in request["extra_body"]



def test_inject_v9_safety_identifier_includes_run_scope(monkeypatch) -> None:
    module = _load_openai_completions_module()
    samples_module = importlib.import_module("inspect_ai.log._samples")

    active_sample = SimpleNamespace(
        task="inspect_evals/agentharm",
        epoch=1,
        sample=SimpleNamespace(id="case-42"),
    )
    monkeypatch.setattr(samples_module, "sample_active", lambda: active_sample)
    monkeypatch.setenv("SAFETY_STATE_RUN_ID", "run:20260322-120001")

    request = {
        "extra_body": {
            "safety_lookahead": True,
            "safety_lookahead_version": "v9",
        }
    }
    module._inject_v9_safety_identifier(request)

    assert request["extra_body"]["trace_id"] == "inspect_evals/agentharm:1:run-20260322-120001"
    assert (
        request["extra_body"]["safety_state_key"]
        == "inspect_evals/agentharm:case-42:1:run-20260322-120001"
    )
