from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.local.bfcl_v4.bfcl_v4 import create_tool_info_from_dict, get_type


DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "local"
    / "bfcl_v4"
    / "data"
)

SINGLE_TURN_FILES = [
    "BFCL_v4_live_simple.json",
    "BFCL_v4_live_multiple.json",
    "BFCL_v4_live_parallel.json",
    "BFCL_v4_live_parallel_multiple.json",
    "BFCL_v4_live_relevance.json",
    "BFCL_v4_simple_python.json",
    "BFCL_v4_simple_java.json",
    "BFCL_v4_simple_javascript.json",
    "BFCL_v4_multiple.json",
    "BFCL_v4_parallel.json",
    "BFCL_v4_parallel_multiple.json",
    "BFCL_v4_irrelevance.json",
    "BFCL_v4_live_irrelevance.json",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _get_sample(path: Path, sample_id: str) -> dict[str, Any]:
    for record in _load_jsonl(path):
        if record.get("id") == sample_id:
            return record
    raise AssertionError(f"Could not find sample {sample_id} in {path}")


def test_simple_javascript_empty_type_schema_no_longer_crashes() -> None:
    file_path = DATA_DIR / "BFCL_v4_simple_javascript.json"
    sample = _get_sample(file_path, "simple_javascript_23")
    function = next(f for f in sample["function"] if f["name"] == "createAuthToken")

    tool_info = create_tool_info_from_dict(function)
    schema = tool_info.model_dump()
    issuer_schema = schema["parameters"]["properties"]["options"]["properties"]["issuer"]

    assert issuer_schema["type"] == "string"


def test_unknown_type_falls_back_to_string() -> None:
    assert get_type("not_a_real_type", schema_path="test.path") == "string"
    assert get_type("", schema_path="test.path") == "string"
    assert get_type(None, schema_path="test.path") == "string"


def test_array_enum_is_relocated_to_items() -> None:
    file_path = DATA_DIR / "BFCL_v4_live_simple.json"
    sample = _get_sample(file_path, "live_simple_71-35-0")
    function = next(f for f in sample["function"] if f["name"] == "extract_parameters_v1")

    tool_info = create_tool_info_from_dict(function)
    schema = tool_info.model_dump()
    metrics = schema["parameters"]["properties"]["metrics"]

    assert metrics["type"] == "array"
    assert metrics["enum"] is None
    assert metrics["items"]["enum"] is not None


def test_single_turn_all_tool_schemas_are_constructible() -> None:
    for file_name in SINGLE_TURN_FILES:
        file_path = DATA_DIR / file_name
        for record in _load_jsonl(file_path):
            functions = record.get("function", [])
            if isinstance(functions, dict):
                functions = [functions]
            for fn in functions:
                create_tool_info_from_dict(fn)
