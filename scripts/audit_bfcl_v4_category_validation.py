#!/usr/bin/env python3
"""
BFCL v4 Category Validation Audit Script

This script audits the results of the BFCL v4 category validation grid search.
It checks for:
1. Execution success status
2. Accuracy metrics by category
3. Category-specific issues (especially Java/JavaScript type handling)
4. Scoring distribution
5. Comparison between safety-confirmation enabled/disabled
"""

import csv
import json
import os
import sys
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CategoryResult:
    """Results for a single category configuration"""
    combo_name: str
    category: str
    safety_enabled: bool
    status: str
    exit_code: int
    eval_file: Optional[str] = None
    accuracy: Optional[float] = None
    error: Optional[str] = None
    metrics: Dict = field(default_factory=dict)


@dataclass
class AuditReport:
    """Audit report for BFCL v4 category validation"""
    base_dir: Path
    results: List[CategoryResult] = field(default_factory=list)

    def add_result(self, result: CategoryResult):
        self.results.append(result)

    def get_results_by_category(self, category: str) -> List[CategoryResult]:
        return [r for r in self.results if r.category == category]

    def get_enabled_disabled_pair(self, category: str) -> tuple:
        """Get the enabled/disabled pair for a category"""
        results = self.get_results_by_category(category)
        enabled = next((r for r in results if r.safety_enabled), None)
        disabled = next((r for r in results if not r.safety_enabled), None)
        return enabled, disabled

    def print_summary(self):
        """Print a summary of the audit"""
        print("=" * 80)
        print("BFCL v4 Category Validation Audit Report")
        print("=" * 80)
        print()

        # Count by status
        success_count = sum(1 for r in self.results if r.status == "success")
        failed_count = sum(1 for r in self.results if r.status == "failed")
        zero_accuracy_count = sum(1 for r in self.results if r.accuracy == 0.0)

        print(f"Total Combinations: {len(self.results)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Zero Accuracy: {zero_accuracy_count}")
        print()

        # Categories tested
        categories = sorted(set(r.category for r in self.results))
        print(f"Categories Tested: {len(categories)}")
        for cat in categories:
            print(f"  - {cat}")
        print()

    def print_category_comparison(self):
        """Print comparison between enabled/disabled for each category"""
        print("=" * 80)
        print("Category-by-Category Results")
        print("=" * 80)
        print()

        categories = sorted(set(r.category for r in self.results))
        for cat in categories:
            enabled, disabled = self.get_enabled_disabled_pair(cat)
            print(f"{cat}:")
            if enabled:
                acc_str = f"{enabled.accuracy:.1%}" if enabled.accuracy is not None else "N/A"
                print(f"  Safety-Confirmation: {acc_str}")
            if disabled:
                acc_str = f"{disabled.accuracy:.1%}" if disabled.accuracy is not None else "N/A"
                print(f"  Baseline:            {acc_str}")
            if enabled and disabled and enabled.accuracy is not None and disabled.accuracy is not None:
                diff = enabled.accuracy - disabled.accuracy
                diff_str = f"{diff:+.1%}"
                if abs(diff) > 0.1:  # More than 10% difference
                    print(f"  Difference:          {diff_str} ⚠️  SIGNIFICANT")
                else:
                    print(f"  Difference:          {diff_str}")
            print()

    def print_issues(self):
        """Print any issues found"""
        print("=" * 80)
        print("Issues and Recommendations")
        print("=" * 80)
        print()

        issues_found = False

        # Check for zero accuracy categories
        print("Categories with 0% Accuracy:")
        zero_accuracy = [r for r in self.results if r.accuracy == 0.0]
        if zero_accuracy:
            issues_found = True
            for r in zero_accuracy:
                status = "WITH SAFETY" if r.safety_enabled else "BASELINE"
                print(f"  ⚠️  {r.category} ({status})")
        else:
            print("  ✓ None")
        print()

        # Check for Java/JavaScript type handling issues
        print("Java/JavaScript Type Handling:")
        java_results = [r for r in self.results if "java" in r.category.lower()]
        for r in java_results:
            if r.status == "success" and r.accuracy is not None and r.accuracy < 0.2:
                issues_found = True
                status = "WITH SAFETY" if r.safety_enabled else "BASELINE"
                print(f"  ⚠️  {r.category} ({status}): {r.accuracy:.1%} - possible type handling issue")
        if not any(r.accuracy < 0.2 for r in java_results if r.accuracy is not None):
            print("  ✓ No obvious type handling issues")
        print()

        # Check for execution errors
        print("Execution Errors:")
        failed = [r for r in self.results if r.status == "failed"]
        if failed:
            issues_found = True
            for r in failed:
                print(f"  ⚠️  {r.category}: {r.error}")
        else:
            print("  ✓ None")
        print()

        if not issues_found:
            print("✓ No significant issues detected. All categories appear to be working correctly.")
        else:
            print("⚠️  Issues detected. Please review the results above.")


def parse_eval_file(eval_path: Path) -> dict:
    """Parse an .eval file and extract metrics"""
    try:
        with zipfile.ZipFile(eval_path, 'r') as zf:
            # Read header.json
            with zf.open('header.json') as f:
                header = json.load(f)

            # Extract accuracy from results
            results = header.get('results', {})
            scores = results.get('scores', [])
            metrics = {}
            accuracy = None

            # Parse scores
            if scores and len(scores) > 0:
                first_score = scores[0]
                metrics = first_score.get('metrics', {})
                accuracy_metrics = metrics.get('accuracy')
                if accuracy_metrics:
                    accuracy = accuracy_metrics.get('value')

            return {
                'accuracy': accuracy,
                'metrics': metrics,
                'scores': scores
            }
    except Exception as e:
        return {'error': str(e), 'accuracy': None, 'metrics': {}, 'scores': []}


def audit_combo_dir(combo_dir: Path) -> CategoryResult:
    """Audit a single combo directory"""
    # Parse combo name
    combo_name = combo_dir.name

    # Determine category and safety settings
    # Check patterns in order from most specific to least specific
    # Note: Combo names use dashes (e.g., SIMPLE-PYTHON, LIVE-SIMPLE)
    # Live categories first (to avoid matching non-live patterns)
    if "LIVE-IRRELEVANCE" in combo_name:
        category = "live_irrelevance"
    elif "LIVE-PARALLEL-MULTIPLE" in combo_name:
        category = "live_parallel_multiple"
    elif "LIVE-PARALLEL" in combo_name:
        category = "live_parallel"
    elif "LIVE-MULTIPLE" in combo_name:
        category = "live_multiple"
    elif "LIVE-SIMPLE" in combo_name:
        category = "live_simple"
    # Non-Live categories (check JAVASCRIPT before JAVA to avoid partial match)
    elif "SIMPLE-JAVASCRIPT" in combo_name:
        category = "simple_javascript"
    elif "SIMPLE-PYTHON" in combo_name:
        category = "simple_python"
    elif "SIMPLE-JAVA" in combo_name:
        category = "simple_java"
    elif "PARALLEL-MULTIPLE" in combo_name:
        category = "parallel_multiple"
    elif "IRRELEVANCE" in combo_name:
        category = "irrelevance"
    elif "PARALLEL" in combo_name:
        category = "parallel"
    elif "MULTIPLE" in combo_name:
        category = "multiple"
    else:
        category = "unknown"

    safety_enabled = "CONFIRM" in combo_name

    # Find .eval file
    eval_files = list(combo_dir.glob("*.eval"))
    eval_file = eval_files[0] if eval_files else None

    # Find log file for errors
    log_file = combo_dir / "run.log"
    error = None
    if log_file.exists():
        try:
            with open(log_file) as f:
                log_content = f.read()
                if "error" in log_content.lower() or "failed" in log_content.lower():
                    # Extract error lines
                    lines = log_content.split('\n')
                    error_lines = [l for l in lines if 'error' in l.lower() or 'exception' in l.lower()]
                    if error_lines:
                        error = error_lines[0][:200]
        except:
            pass

    # Check status from exit code
    # Assume success if .eval file exists
    status = "success" if eval_file else "failed"
    exit_code = 0 if eval_file else 1

    # Parse eval file
    accuracy = None
    metrics = {}
    if eval_file:
        result = parse_eval_file(eval_file)
        accuracy = result['accuracy']
        metrics = result['metrics']
        if 'error' in result:
            error = result['error']
            status = "failed"

    return CategoryResult(
        combo_name=combo_name,
        category=category,
        safety_enabled=safety_enabled,
        status=status,
        exit_code=exit_code,
        eval_file=str(eval_file) if eval_file else None,
        accuracy=accuracy,
        error=error,
        metrics=metrics
    )


def main():
    """Main audit function"""
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        # Use the most recent grid search result
        base_dir = Path("/mnt/data1/workspace/djs/eval-poc-with-salt/results/experiments/grid_search")
        # Find the most recent bfcl-v4-category-validation directory
        search_dirs = list(base_dir.glob("*/bfcl-v4-category-validation/*/"))
        if not search_dirs:
            print("Error: No bfcl-v4-category-validation results found")
            sys.exit(1)
        base_dir = max(search_dirs, key=lambda p: p.stat().st_mtime)

    print(f"Auditing results from: {base_dir}")
    print()

    combos_dir = base_dir / "combos"
    if not combos_dir.exists():
        print(f"Error: No combos directory found at {combos_dir}")
        sys.exit(1)

    report = AuditReport(base_dir=base_dir)

    # Audit each combo directory
    for combo_dir in sorted(combos_dir.iterdir()):
        if combo_dir.is_dir() and combo_dir.name.startswith(('00', '01', '02', '03', '04',
                                                               '05', '06', '07', '08', '09',
                                                               '10', '11', '12', '13', '14',
                                                               '15', '16', '17', '18', '19',
                                                               '20', '21', '22', '23', '24')):
            result = audit_combo_dir(combo_dir)
            report.add_result(result)

    # Print report
    report.print_summary()
    report.print_category_comparison()
    report.print_issues()

    # Return exit code based on issues found
    if any(r.status == "failed" for r in report.results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
