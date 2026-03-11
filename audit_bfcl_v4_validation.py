#!/usr/bin/env python3
"""
BFCL v4 Category Validation Results Audit Script

This script audits the results of BFCL v4 category validation runs to:
1. Check execution success for all categories
2. Compare accuracy metrics between enabled/disabled safety
3. Identify category-specific issues
4. Verify Java/JavaScript type handling
5. Check scoring distribution (C/I ratios)

Usage:
    python audit_bfcl_v4_validation.py <results_dir>

Example:
    python audit_bfcl_v4_validation.py results/experiments/grid_search/0304/bfcl-v4-category-validation/multi_/20260304-1133
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any


@dataclass
class CategoryResult:
    """Results for a single category."""
    category: str
    safety_enabled: bool
    total: int = 0
    correct: int = 0
    invalid: int = 0
    accuracy: float = 0.0
    errors: list[str] = field(default_factory=list)
    eval_file: str = ""
    combo_dir: str = ""


def read_eval_file(eval_path: str) -> dict[str, Any] | None:
    """Read an .eval file and return the header.json contents."""
    try:
        with zipfile.ZipFile(eval_path, 'r') as zf:
            with zf.open('header.json') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading {eval_path}: {e}")
        return None


def find_eval_files(results_dir: str) -> list[tuple[str, str, str]]:
    """Find all .eval files in combo directories.

    Returns:
        List of (combo_dir, eval_path, combo_name) tuples
    """
    eval_files = []
    results_path = Path(results_dir)

    # Look in combos directory
    combos_dir = results_path / "combos"
    if not combos_dir.exists():
        print(f"Error: combos directory not found at {combos_dir}")
        return []

    for combo_dir in sorted(combos_dir.iterdir()):
        if not combo_dir.is_dir():
            continue

        # Find .eval file directly in combo directory or in logs subdirectory
        eval_files = list(combo_dir.glob("*.eval"))
        if not eval_files:
            logs_dir = combo_dir / "logs"
            if logs_dir.exists():
                eval_files = list(logs_dir.glob("*.eval"))

        if not eval_files:
            print(f"Warning: No .eval file found in {combo_dir}")
            continue

        for eval_file in eval_files:
            eval_files.append((
                str(combo_dir),
                str(eval_file),
                combo_dir.name
            ))

    return eval_files


def parse_combo_name(combo_name: str) -> tuple[str, bool]:
    """Parse combo directory name to extract category and safety setting.

    Examples:
        001-BFCL-V4-SIMPLE-PYTHON-CONFIRM-V6-RECONSIDER3-N-REQ1 -> ("simple_python", True)
        002-BFCL-V4-SIMPLE-PYTHON-DISABLED -> ("simple_python", False)
    """
    parts = combo_name.split("-")

    # Find the category part (between BFCL-V4 and CONFIRM/DISABLED)
    category_parts = []
    for part in parts[2:]:  # Skip 001 and BFCL-V4
        if part in ("CONFIRM", "DISABLED"):
            break
        category_parts.append(part)

    category = "_".join(category_parts).lower()

    # Check if safety confirmation is enabled
    safety_enabled = "CONFIRM" in combo_name

    return category, safety_enabled


def extract_metrics(header: dict[str, Any]) -> tuple[int, int, float]:
    """Extract accuracy metrics from eval header.

    Returns:
        (total, correct, accuracy)
    """
    # Try to find accuracy metrics
    total = header.get("samples", 0)

    # Look for scorer metrics
    scorer_metrics = header.get("metrics", {}).get("scorer", {})

    # Check for accuracy values
    if "accuracy" in scorer_metrics:
        accuracy = scorer_metrics["accuracy"]
        correct = int(total * accuracy)
    else:
        # Try to get from other fields
        correct = scorer_metrics.get("C", 0)
        invalid = scorer_metrics.get("I", 0)
        total_calc = correct + invalid
        if total_calc > 0:
            total = total_calc
            accuracy = correct / total
        else:
            accuracy = 0.0

    return total, correct, accuracy


def audit_results(results_dir: str) -> dict[str, list[CategoryResult]]:
    """Audit all results and return organized data.

    Returns:
        Dict mapping category name to list of results (with/without safety)
    """
    eval_files = find_eval_files(results_dir)
    results_by_category: dict[str, list[CategoryResult]] = defaultdict(list)

    for combo_dir, eval_path, combo_name in eval_files:
        category, safety_enabled = parse_combo_name(combo_name)

        # Read eval file
        header = read_eval_file(eval_path)
        if header is None:
            result = CategoryResult(
                category=category,
                safety_enabled=safety_enabled,
                errors=["Failed to read eval file"],
                combo_dir=combo_dir
            )
            results_by_category[category].append(result)
            continue

        # Extract metrics
        total, correct, accuracy = extract_metrics(header)
        invalid = total - correct

        result = CategoryResult(
            category=category,
            safety_enabled=safety_enabled,
            total=total,
            correct=correct,
            invalid=invalid,
            accuracy=accuracy,
            eval_file=eval_path,
            combo_dir=combo_dir
        )

        results_by_category[category].append(result)

    return results_by_category


def generate_report(results_by_category: dict[str, list[CategoryResult]]) -> str:
    """Generate a comprehensive audit report."""
    lines = []
    lines.append("=" * 80)
    lines.append("BFCL v4 Category Validation Audit Report")
    lines.append("=" * 80)
    lines.append("")

    # Sort categories
    sorted_categories = sorted(results_by_category.keys())

    # Overall status
    total_categories = len(sorted_categories)
    all_passed = 0
    has_issues = 0

    for category in sorted_categories:
        results = results_by_category[category]
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Category: {category.upper()}")
        lines.append(f"{'─' * 80}")

        for result in results:
            safety_status = "ENABLED" if result.safety_enabled else "DISABLED"

            if result.errors:
                lines.append(f"\n  [{safety_status}] ERROR: {', '.join(result.errors)}")
                has_issues += 1
            else:
                lines.append(f"\n  [{safety_status}] Status: OK")
                lines.append(f"    Total: {result.total}")
                lines.append(f"    Correct (C): {result.correct}")
                lines.append(f"    Invalid (I): {result.invalid}")
                lines.append(f"    Accuracy: {result.accuracy:.2%}")

                # Check for issues
                issues = []
                if result.accuracy == 0:
                    issues.append("0% accuracy - potential bug!")
                elif result.accuracy < 0.3:
                    issues.append(f"Low accuracy ({result.accuracy:.1%})")
                if result.total == 0:
                    issues.append("No samples evaluated")

                if issues:
                    lines.append(f"    ⚠️  Issues: {', '.join(issues)}")
                    has_issues += 1
                else:
                    all_passed += 1

                # Show eval file location
                lines.append(f"    Eval file: {result.eval_file}")

    # Summary section
    lines.append(f"\n\n{'=' * 80}")
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total categories: {total_categories}")
    lines.append(f"Passed: {all_passed}")
    lines.append(f"With issues: {has_issues}")
    lines.append("")

    # Comparison table
    lines.append(f"\n{'=' * 80}")
    lines.append("ACCURACY COMPARISON (Safety ON vs OFF)")
    lines.append("=" * 80)
    lines.append(f"{'Category':<30} {'Disabled':<12} {'Enabled':<12} {'Delta':<10}")
    lines.append("-" * 80)

    for category in sorted_categories:
        results = results_by_category[category]
        disabled_acc = None
        enabled_acc = None

        for result in results:
            if result.safety_enabled:
                enabled_acc = result.accuracy
            else:
                disabled_acc = result.accuracy

        if disabled_acc is not None and enabled_acc is not None:
            delta = enabled_acc - disabled_acc
            delta_str = f"{delta:+.1%}"
            lines.append(f"{category:<30} {disabled_acc:>11.1%} {enabled_acc:>11.1%} {delta_str:>10}")

    # Category-specific findings
    lines.append(f"\n{'=' * 80}")
    lines.append("CATEGORY-SPECIFIC FINDINGS")
    lines.append("=" * 80)

    # Check Java/JavaScript type handling
    java_result = results_by_category.get("simple_java", [])
    javascript_result = results_by_category.get("simple_javascript", [])

    if java_result:
        for r in java_result:
            if not r.errors and r.accuracy > 0:
                lines.append(f"\n✅ simple_java: Working ({r.accuracy:.1%} accuracy)")
                break
        else:
            lines.append("\n⚠️  simple_java: May have issues (check for 0% accuracy)")

    if javascript_result:
        for r in javascript_result:
            if not r.errors and r.accuracy > 0:
                lines.append(f"✅ simple_javascript: Working ({r.accuracy:.1%} accuracy)")
                break
        else:
            lines.append("⚠️  simple_javascript: May have issues (check for 0% accuracy)")

    # Check irrelevance categories
    for irr_cat in ["irrelevance", "live_irrelevance"]:
        irr_results = results_by_category.get(irr_cat, [])
        if irr_results:
            for r in irr_results:
                if r.accuracy > 0:
                    lines.append(f"\n✅ {irr_cat}: Irrelevance fix working ({r.accuracy:.1%} accuracy)")
                    break
            else:
                lines.append(f"\n⚠️  {irr_cat}: May still have issues (expected >0% accuracy)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Audit BFCL v4 category validation results"
    )
    parser.add_argument(
        "results_dir",
        help="Path to grid search results directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Write report to file instead of stdout"
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: results directory not found: {args.results_dir}")
        sys.exit(1)

    # Audit results
    results_by_category = audit_results(args.results_dir)

    # Generate report
    report = generate_report(results_by_category)

    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
