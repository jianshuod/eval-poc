"""
Grid Search Module for Safety-Lookahead Evaluation

This module provides reusable grid search functionality for systematically testing
hyperparameter combinations in safety-lookahead evaluations.

Extracted from run-eval-salt-grid.py for use in run-eval-salt.py grid search mode.
"""
from __future__ import annotations

import csv
import itertools
from datetime import datetime
from pathlib import Path
from typing import Any


def load_grid_config(config_file: str, project_root: Path) -> dict:
    """
    Load grid search configuration file.

    Args:
        config_file: Path to the grid search configuration file
        project_root: Root directory of the project

    Returns:
        Dictionary containing the loaded configuration
    """
    import sys
    import yaml

    config_path = Path(config_file)

    # Try relative path from current directory
    if not config_path.is_absolute():
        # Try from current working directory
        cwd_path = Path.cwd() / config_file
        if cwd_path.exists():
            config_path = cwd_path
        # Try from script directory
        else:
            script_path = project_root / config_file
            if script_path.exists():
                config_path = script_path

    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_file}")
        print(f"  尝试的路径:")
        print(f"    - {Path.cwd() / config_file}")
        print(f"    - {project_root / config_file}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def generate_combinations(grid_config: dict) -> list[dict]:
    """
    Generate all combinations from grid search dimensions.

    Args:
        grid_config: Dictionary with grid_search.dimensions section

    Returns:
        List of dictionaries, each representing a unique combination

    Note:
        When enabled=false, removes mode, n, mask, forced, version parameters
        to avoid generating duplicate disabled combinations.
    """
    dimensions = grid_config.get("grid_search", {}).get("dimensions", {})
    safety_lookahead_dims = dimensions.get("safety_lookahead", {})

    # Build lists of values for each dimension
    param_names = []
    param_values = []

    for param_name, values in safety_lookahead_dims.items():
        if not isinstance(values, list):
            values = [values]
        param_names.append(param_name)
        param_values.append(values)

    # Generate cartesian product of all combinations
    seen_combinations = {}  # Use dict for deduplication (tuple of sorted items as key)

    for combination_values in itertools.product(*param_values):
        combo = dict(zip(param_names, combination_values))

        # Apply constraint: when enabled=false, exclude safety_lookahead-specific configs
        if combo.get("enabled") is False:
            # Remove mode, n, mask, forced for disabled safety lookahead
            combo.pop("mode", None)
            combo.pop("n", None)
            combo.pop("mask", None)
            combo.pop("forced", None)
            combo.pop("version", None)

        # Deduplicate by converting to tuple of sorted items
        combo_key = tuple(sorted(combo.items()))
        if combo_key not in seen_combinations:
            seen_combinations[combo_key] = combo

    return list(seen_combinations.values())


def combination_to_dir_name(index: int, combo: dict) -> str:
    """
    Convert a combination to a directory name (ALL-CAPS format).

    Args:
        index: Combination index (1-based)
        combo: Dictionary of parameter values

    Returns:
        Directory name like "001-REMINDER-N1-V7-FORCED-NO-MASK"

    Examples:
        001-REMINDER-N1-V7-FORCED-NO-MASK
        002-WORLD_MODEL-N3-V7-KEYWORDS
        003-DISABLED
    """
    parts = [f"{index:03d}"]

    # Check if disabled
    if combo.get("enabled") is False:
        parts.append("DISABLED")
        return "-".join(parts)

    # Mode (required for enabled)
    mode = combo.get("mode", "world_model")
    mode_display = {
        "reminder": "REMINDER",
        "spec_repeat": "SPEC_REPEAT",
        "context_analysis": "CTX_ANALYSIS",
        "world_model": "WORLD_MODEL",
    }.get(mode, mode.upper())

    parts.append(mode_display)

    # N value (n)
    if "n" in combo:
        parts.append(f"N{combo['n']}")

    # Version
    if "version" in combo:
        parts.append(combo["version"].upper())

    # Forced
    if combo.get("forced"):
        parts.append("FORCED")

    # Mask
    mask = combo.get("mask", "none")
    if mask != "none":
        mask_display = {
            "keywords": "KEYWORDS",
            "rewriting": "REWRITING",
        }.get(mask, mask.upper())
        parts.append(mask_display)
    elif combo.get("mask") == "none":
        parts.append("NO-MASK")

    return "-".join(parts)


def combination_to_cli_args(combo: dict) -> list[str]:
    """
    Convert a combination to CLI arguments for safety-lookahead.

    Args:
        combo: Dictionary of parameter values

    Returns:
        List of CLI argument strings (e.g., ["--with-safety-lookahead", "--safety-n", "3"])
    """
    args = []

    # Handle enabled flag
    enabled = combo.get("enabled", True)

    if not enabled:
        # Don't add --with-safety-lookahead, skip other safety params
        return args

    # Safety lookahead is enabled
    args.append("--with-safety-lookahead")

    # Version
    if "version" in combo:
        args.extend(["--safety-version", str(combo["version"])])

    # Mode
    if "mode" in combo:
        args.extend(["--safety-mode", str(combo["mode"])])

    # N (number of candidates)
    if "n" in combo:
        args.extend(["--safety-n", str(combo["n"])])

    # Mask strategy
    if "mask" in combo:
        args.extend(["--safety-mask", str(combo["mask"])])

    # Forced flag
    if combo.get("forced", False):
        args.append("--safety-forced")

    return args


def create_temp_config(base_config: dict, combo: dict, output_dir: Path) -> Path:
    """
    Create a temporary config file for a specific combination.

    Args:
        base_config: Base configuration from grid search file
        combo: Dictionary of parameter values for this combination
        output_dir: Directory to write the temp config

    Returns:
        Path to the temporary config file
    """
    import yaml

    # Copy base config
    temp_config = base_config.copy()

    # Remove grid_search section from temp config
    temp_config.pop("grid_search", None)

    # Update safety_lookahead section with combination values
    if "safety_lookahead" not in temp_config:
        temp_config["safety_lookahead"] = {}

    for key, value in combo.items():
        if value is not None:
            temp_config["safety_lookahead"][key] = value

    # Write temp config
    temp_config_path = output_dir / "config.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(temp_config, f, default_flow_style=False, allow_unicode=True)

    return temp_config_path


def aggregate_token_stats(grid_base_dir: Path) -> dict:
    """
    Aggregate token stats from all combo directories.

    Args:
        grid_base_dir: Base grid search directory containing combos/ subdirectory

    Returns:
        Dictionary with aggregated token stats across all combos
    """
    import sys

    combos_dir = grid_base_dir / "combos"
    if not combos_dir.exists():
        return {}

    # Try to import token_stats_generator
    try:
        from token_stats_generator import generate_token_summary
    except ImportError:
        print("Warning: token_stats_generator not available, skipping token stats aggregation", file=sys.stderr)
        return {}

    aggregated = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "total_cache_read": 0,
        "total_cache_write": 0,
        "total_reasoning": 0,
        "combos_with_stats": 0,
        "combos_total": 0,
    }

    # Iterate over combo directories
    for combo_dir in sorted(combos_dir.iterdir()):
        if not combo_dir.is_dir():
            continue

        aggregated["combos_total"] += 1

        # Get token stats for this combo
        stats = generate_token_summary(combo_dir)
        if stats:
            aggregated["combos_with_stats"] += 1
            aggregated["total_input_tokens"] += stats.get("total_input_tokens", 0)
            aggregated["total_output_tokens"] += stats.get("total_output_tokens", 0)
            aggregated["total_tokens"] += stats.get("total_tokens", 0)
            aggregated["total_cache_read"] += stats.get("total_cache_read", 0)
            aggregated["total_cache_write"] += stats.get("total_cache_write", 0)
            aggregated["total_reasoning"] += stats.get("total_reasoning", 0)

    return aggregated


def write_summary_csv(results: list[dict], output_dir: Path) -> None:
    """
    Write summary CSV file with all grid search results.

    Args:
        results: List of result dictionaries from run_combination
        output_dir: Directory to write the summary file
    """
    csv_path = output_dir / "summary.csv"

    # Get all parameter names
    all_params = set()
    for result in results:
        if "combination" in result:
            all_params.update(result["combination"].keys())

    param_columns = sorted(all_params)

    # Define CSV columns
    fieldnames = [
        "index",
        "dir_name",
        "status",
        "exit_code",
        "output_dir",
        "log_file",
    ] + param_columns

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "index": result.get("index", ""),
                "dir_name": result.get("dir_name", ""),
                "status": result.get("status", ""),
                "exit_code": result.get("exit_code", ""),
                "output_dir": result.get("output_dir", ""),
                "log_file": result.get("log_file", ""),
            }

            # Add parameter values
            for param in param_columns:
                row[param] = result.get("combination", {}).get(param, "")

            writer.writerow(row)

    print(f"Summary CSV written to: {csv_path}")


def print_summary(results: list[dict]) -> None:
    """
    Print a summary of grid search results.

    Args:
        results: List of result dictionaries from grid search runs
    """
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    dry_run = sum(1 for r in results if r["status"] == "dry_run")

    print()
    print("=" * 60)
    print("Grid Search Summary")
    print("=" * 60)
    print(f"Total combinations: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    print(f"Dry run: {dry_run}")
    print("=" * 60)
    print()
