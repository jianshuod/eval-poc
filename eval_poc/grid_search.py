"""
Grid Search Module for Safety-Lookahead and Safety-Confirmation Evaluation

This module provides reusable grid search functionality for systematically testing
hyperparameter combinations in safety-lookahead and safety-confirmation evaluations.

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

    Supports both safety_lookahead and safety_confirmation dimensions.
    Only one safety dimension should be specified per config.

    Supports multiple benchmarks via the "benchmarks" list key.

    Args:
        grid_config: Dictionary with grid_search.dimensions section

    Returns:
        List of dictionaries, each representing a unique combination

    Note:
        For safety_lookahead: When enabled=false, removes mode, n, mask, forced, version parameters
        For safety_confirmation: When enabled=false, removes max_reconsideration, version parameters
        For multiple benchmarks: Each combination includes a "benchmark" key
    """
    dimensions = grid_config.get("grid_search", {}).get("dimensions", {})

    # Support both single benchmark (string) and multiple benchmarks (list)
    benchmark = grid_config.get("benchmark")  # Single (legacy)
    benchmarks = grid_config.get("benchmarks", [])  # Multiple (new)

    if benchmark and benchmarks:
        raise ValueError("Cannot specify both 'benchmark' and 'benchmarks' in config")

    # Normalize to list for processing
    if benchmark:
        benchmarks = [benchmark]
    elif not benchmarks:
        benchmarks = []  # No benchmarks specified (should error later)

    # Handle models: support both single model (base) and multiple models (models)
    # Priority: models list takes precedence over base string
    model_cfg = grid_config.get("model", {})
    models = model_cfg.get("models", [])
    base_model = model_cfg.get("base")

    # Normalize to list for processing
    # models list takes precedence over base string
    if models:
        pass  # Already have models list
    elif base_model:
        models = [base_model]
    else:
        models = []  # No models specified (should error later)

    # Determine which safety dimension to use
    safety_lookahead_dims = dimensions.get("safety_lookahead", {})
    safety_confirmation_dims = dimensions.get("safety_confirmation", {})

    # Validate: only one safety dimension should be specified
    if safety_lookahead_dims and safety_confirmation_dims:
        raise ValueError("Cannot specify both safety_lookahead and safety_confirmation dimensions. Use one or the other.")

    # Choose the active dimension
    if safety_lookahead_dims:
        active_dims = {"safety_lookahead": safety_lookahead_dims}
    elif safety_confirmation_dims:
        active_dims = {"safety_confirmation": safety_confirmation_dims}
    else:
        active_dims = {}

    # Build lists of values for each dimension
    param_names = []
    param_values = []

    # Add benchmarks as a dimension if multiple specified
    is_multi_benchmark = len(benchmarks) > 1
    if is_multi_benchmark:
        param_names.append("benchmark")
        param_values.append(benchmarks)

    # Add models as a dimension if multiple specified
    is_multi_model = len(models) > 1
    if is_multi_model:
        param_names.append("model")
        param_values.append(models)

    for dim_type, dim_config in active_dims.items():
        for param_name, values in dim_config.items():
            if not isinstance(values, list):
                values = [values]
            # Add dimension type prefix to avoid name conflicts
            prefixed_name = f"{dim_type}_{param_name}" if param_name != "enabled" else param_name
            param_names.append(prefixed_name)
            param_values.append(values)

    # If no dimensions at all (not even benchmarks), return empty
    if not param_names:
        return []

    # Generate cartesian product of all combinations
    seen_combinations = {}  # Use dict for deduplication (tuple of sorted items as key)

    for combination_values in itertools.product(*param_values):
        combo = dict(zip(param_names, combination_values))

        # Unprefix the names and organize by dimension type
        organized = {}
        dim_type = None
        for prefixed_name, value in combo.items():
            if prefixed_name == "benchmark":
                # Benchmark dimension (for multi-benchmark mode)
                organized["benchmark"] = value
                if is_multi_benchmark:
                    organized["_multi_benchmark"] = True
            elif prefixed_name == "model":
                # Model dimension (for multi-model mode)
                organized["model"] = value
                if is_multi_model:
                    organized["_multi_model"] = True
            elif prefixed_name == "enabled":
                # Determine which dimension type based on which config we're using
                dim_type = "safety_lookahead" if safety_lookahead_dims else "safety_confirmation"
                organized["enabled"] = value
                organized["type"] = dim_type
            else:
                # Unprefix the name
                if prefixed_name.startswith("safety_lookahead_"):
                    dim_type = "safety_lookahead"
                    param = prefixed_name.replace("safety_lookahead_", "")
                    organized[param] = value
                    organized["type"] = dim_type
                elif prefixed_name.startswith("safety_confirmation_"):
                    dim_type = "safety_confirmation"
                    param = prefixed_name.replace("safety_confirmation_", "")
                    organized[param] = value
                    organized["type"] = dim_type

        # Apply constraints based on dimension type
        if organized.get("enabled") is False:
            if organized.get("type") == "safety_lookahead":
                # Remove mode, n, mask, forced for disabled safety lookahead
                organized.pop("mode", None)
                organized.pop("n", None)
                organized.pop("mask", None)
                organized.pop("forced", None)
                organized.pop("version", None)
            elif organized.get("type") == "safety_confirmation":
                # Remove max_reconsideration, version, and n_required for disabled safety confirmation
                organized.pop("max_reconsideration", None)
                organized.pop("version", None)
                organized.pop("n_required", None)

        # Deduplicate by converting to tuple of sorted items
        combo_key = tuple(sorted(organized.items()))
        if combo_key not in seen_combinations:
            seen_combinations[combo_key] = organized

    return list(seen_combinations.values())


def combination_to_dir_name(index: int, combo: dict) -> str:
    """
    Convert a combination to a directory name (ALL-CAPS format).

    Args:
        index: Combination index (1-based)
        combo: Dictionary of parameter values

    Returns:
        Directory name like:
        - For safety_lookahead: "001-REMINDER-N1-V7-FORCED-NO-MASK"
        - For safety_confirmation: "001-CONFIRM-V2-RECONSIDER2"
        - For disabled: "002-DISABLED"
        - For multi-benchmark: "001-STRONG_REJ-CONFIRM-V2-RECONSIDER2"

    Examples:
        001-REMINDER-N1-V7-FORCED-NO-MASK (safety_lookahead)
        001-CONFIRM-V2-RECONSIDER2 (safety_confirmation)
        003-DISABLED
        001-STRONG_REJ-CONFIRM-V2-RECONSIDER2 (multi-benchmark)
        002-AGENTHARM-DISABLED (multi-benchmark)
    """
    parts = [f"{index:03d}"]

    # Add benchmark prefix for multi-benchmark runs
    if "benchmark" in combo and combo.get("_multi_benchmark"):
        # Sanitize benchmark name for directory (use full names, no truncation)
        # e.g., "strong_reject" -> "STRONG-REJECT"
        # e.g., "agentharm:agentharm_benign" -> "AGENTHARM-BENIGN"
        # e.g., "agentharm:agentharm" -> "AGENTHARM-AGENTHARM"
        benchmark_spec = combo["benchmark"]
        if ":" in benchmark_spec:
            # Include task name for distinction
            benchmark_name, task_name = benchmark_spec.split(":", 1)
            # Sanitize both parts with filesystem-safe characters
            bench_short = benchmark_name.upper().replace('_', '-')
            # For task name, remove common prefix
            if task_name.startswith(benchmark_name + "_"):
                task_suffix = task_name[len(benchmark_name)+1:]
            else:
                task_suffix = task_name
            task_short = task_suffix.upper().replace('_', '-')
            parts.append(f"{bench_short}-{task_short}")
        else:
            benchmark_short = benchmark_spec.upper().replace('_', '-')
            parts.append(benchmark_short)

    # Check if disabled
    if combo.get("enabled") is False:
        parts.append("DISABLED")
        return "-".join(parts)

    # Handle based on dimension type
    dim_type = combo.get("type", "")

    if dim_type == "safety_confirmation":
        parts.append("CONFIRM")

        # Version
        if "version" in combo:
            parts.append(combo["version"].upper())

        # Max reconsideration
        if "max_reconsideration" in combo:
            parts.append(f"RECONSIDER{combo['max_reconsideration']}")

        # N required (number of confirmations required)
        if "n_required" in combo:
            parts.append(f"N-REQ{combo['n_required']}")

    elif dim_type == "safety_lookahead":
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

        # Depth (passes required)
        if "depth" in combo:
            parts.append(f"DEPTH{combo['depth']}")

        # Breadth (action proposals)
        if "breadth" in combo:
            parts.append(f"BREADTH{combo['breadth']}")

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
    Convert a combination to CLI arguments.

    Supports both safety-lookahead and safety-confirmation.

    Args:
        combo: Dictionary of parameter values

    Returns:
        List of CLI argument strings (e.g., ["--with-safety-lookahead", "--safety-n", "3"])
    """
    args = []

    # Handle enabled flag
    enabled = combo.get("enabled", True)

    if not enabled:
        # Don't add safety flags, skip other safety params
        return args

    # Determine dimension type
    dim_type = combo.get("type", "")

    if dim_type == "safety_confirmation":
        # Safety confirmation is enabled
        args.append("--with-safety-confirmation")

        # Version
        if "version" in combo:
            args.extend(["--safety-confirmation-version", str(combo["version"])])

        # Max reconsideration
        if "max_reconsideration" in combo:
            args.extend(["--safety-max-reconsider", str(combo["max_reconsideration"])])

        # N required (number of confirmations required)
        if "n_required" in combo:
            args.extend(["--safety-n-required", str(combo["n_required"])])

    elif dim_type == "safety_lookahead":
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

        # Depth (passes required)
        if "depth" in combo:
            args.extend(["--safety-depth", str(combo["depth"])])

        # Breadth (action proposals)
        if "breadth" in combo:
            args.extend(["--safety-breadth", str(combo["breadth"])])

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
    # Remove benchmarks list from temp config (use benchmark from combo)
    base_benchmarks = temp_config.pop("benchmarks", [])

    # Add benchmark from combo if present (multi-benchmark mode)
    # Otherwise, use first benchmark from base_config benchmarks list (single-benchmark mode)
    if "benchmark" in combo:
        temp_config["benchmark"] = combo["benchmark"]
    elif base_benchmarks:
        # Single-benchmark mode: extract first benchmark from list
        temp_config["benchmark"] = base_benchmarks[0]

    # Add model from combo if present (multi-model mode)
    if "model" in combo:
        temp_config["model"] = {"base": combo["model"]}

    # Determine dimension type and update appropriate section
    dim_type = combo.get("type", "")

    if dim_type == "safety_confirmation":
        # Update safety_confirmation section with combination values
        if "safety_confirmation" not in temp_config:
            temp_config["safety_confirmation"] = {}

        for key, value in combo.items():
            # Exclude non-safety parameters from being added to safety section
            if key not in ("type", "benchmark", "_multi_benchmark", "_multi_model", "model") and value is not None:
                temp_config["safety_confirmation"][key] = value

    elif dim_type == "safety_lookahead":
        # Update safety_lookahead section with combination values
        if "safety_lookahead" not in temp_config:
            temp_config["safety_lookahead"] = {}

        for key, value in combo.items():
            # Exclude non-safety parameters from being added to safety section
            if key not in ("type", "benchmark", "_multi_benchmark", "_multi_model", "model") and value is not None:
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
        from token_stats_generator import generate_token_summary, get_pricing_weights, calculate_weighted_cost
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
        "total_weighted_tokens": 0,
        "combos_with_stats": 0,
        "combos_total": 0,
    }

    # Get pricing weights for cost calculation
    weights = get_pricing_weights()

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

            # Add weighted token cost
            costs = calculate_weighted_cost(stats, weights)
            aggregated["total_weighted_tokens"] += costs.total_weighted_tokens

    return aggregated


def write_summary_csv(results: list[dict], output_dir: Path) -> None:
    """
    Write summary CSV file with all grid search results.

    Args:
        results: List of result dictionaries from run_combination
        output_dir: Directory to write the summary file

    Note:
        In multi-benchmark mode, the "benchmark" column appears first
        (after index and dir_name) for better readability.
    """
    csv_path = output_dir / "summary.csv"

    # Get all parameter names
    all_params = set()
    has_benchmark = False
    has_model = False
    for result in results:
        if "combination" in result:
            all_params.update(result["combination"].keys())
            if "benchmark" in result["combination"]:
                has_benchmark = True
            if "model" in result["combination"]:
                has_model = True

    param_columns = sorted(all_params)

    # Define CSV columns - put benchmark and model first if multi-mode
    base_columns = [
        "index",
        "dir_name",
    ]
    if has_benchmark:
        base_columns.append("benchmark")
    if has_model:
        base_columns.append("model")

    base_columns.extend([
        "status",
        "exit_code",
        "output_dir",
        "log_file",
    ])

    # Remove benchmark and model from param_columns if we've already added them
    if has_benchmark and "benchmark" in param_columns:
        param_columns.remove("benchmark")
    if has_model and "model" in param_columns:
        param_columns.remove("model")

    fieldnames = base_columns + param_columns

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

            # Add benchmark from combination if present
            if has_benchmark:
                row["benchmark"] = result.get("combination", {}).get("benchmark", "")

            # Add model from combination if present
            if has_model:
                row["model"] = result.get("combination", {}).get("model", "")

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
