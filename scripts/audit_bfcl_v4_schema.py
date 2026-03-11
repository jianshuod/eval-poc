#!/usr/bin/env python3
"""
Audit BFCL v4 dataset schemas for malformed parameter definitions.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "local" / "bfcl_v4" / "data"


def load_records(file_path: Path) -> list[dict[str, Any]]:
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


def walk_schema(
    schema: dict[str, Any],
    schema_path: str,
    issues: dict[str, list[str]],
) -> None:
    value_type = schema.get("type")
    if isinstance(value_type, str) and value_type.strip() == "":
        issues["type_empty"].append(schema_path)

    if value_type == "array" and "enum" in schema:
        issues["array_with_enum"].append(schema_path)

    required = schema.get("required")
    if required is not None and not isinstance(required, list):
        issues["required_not_list"].append(schema_path)

    properties = schema.get("properties")
    if properties is not None and not isinstance(properties, dict):
        issues["properties_not_dict"].append(schema_path)
    if isinstance(properties, dict):
        for key, nested in properties.items():
            if isinstance(nested, dict):
                walk_schema(nested, f"{schema_path}.properties.{key}", issues)
            else:
                issues["property_not_dict"].append(f"{schema_path}.properties.{key}")

    items = schema.get("items")
    if items is not None and not isinstance(items, dict):
        issues["items_not_dict"].append(schema_path)
    if isinstance(items, dict):
        walk_schema(items, f"{schema_path}.items", issues)


def audit() -> int:
    files = sorted(DATA_DIR.glob("BFCL_v4*.json"))
    if not files:
        print(f"No BFCL v4 data files found under {DATA_DIR}")
        return 2

    issues: dict[str, list[str]] = defaultdict(list)

    for file_path in files:
        for idx, record in enumerate(load_records(file_path), start=1):
            functions = record.get("function", [])
            if isinstance(functions, dict):
                functions = [functions]
            if not isinstance(functions, list):
                issues["function_not_list"].append(f"{file_path.name}#{idx}")
                continue

            for fn in functions:
                if not isinstance(fn, dict):
                    issues["function_not_dict"].append(f"{file_path.name}#{idx}")
                    continue

                params = fn.get("parameters")
                if params is None:
                    continue
                if not isinstance(params, dict):
                    issues["parameters_not_dict"].append(
                        f"{file_path.name}#{idx}:{fn.get('name', 'unknown')}"
                    )
                    continue

                root = f"{file_path.name}#{idx}:{record.get('id', 'unknown')}:{fn.get('name', 'unknown')}"
                walk_schema(params, root, issues)

    print("BFCL v4 schema audit")
    print(f"Data directory: {DATA_DIR}")
    print("")
    for issue_type in sorted(issues):
        print(f"{issue_type}: {len(issues[issue_type])}")
        for example in issues[issue_type][:10]:
            print(f"  - {example}")
        if len(issues[issue_type]) > 10:
            print(f"  ... {len(issues[issue_type]) - 10} more")
    if not issues:
        print("No schema issues detected.")

    # These are known, tolerated malformations handled by runtime sanitization.
    tolerated = {"type_empty", "array_with_enum"}
    blocking_issue_types = [k for k in issues if k not in tolerated]
    return 1 if blocking_issue_types else 0


if __name__ == "__main__":
    raise SystemExit(audit())
