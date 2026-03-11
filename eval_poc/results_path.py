"""
Results Path Builder Module

This module provides centralized path-building logic for results directories.
It ensures consistent directory structure across the eval-poc framework.

Directory Structure:
    results/
    ├── experiments/                       # Named experiments
    │   └── {run_name}/
    │       ├── metadata.json
    │       ├── {benchmark}/
    │       │   └── {sanitized_model}_{timestamp}/
    │       │       ├── config.yaml
    │       │       ├── eval/              # inspect_ai .eval files
    │       │       ├── safety_analysis.jsonl
    │       │       ├── safety_lookahead.log
    │       │       ├── token_stats.txt
    │       │       └── token_stats.csv
    │       └── grid_search/               # Grid search experiments (all grouped under grid_search/)
    │           └── {MMDD}/               # Date directory (MMDD format)
    │               └── {run_name}/
    │                   └── {benchmark}_{sanitized_model}/
    │                       └── {timestamp}/
    │                           ├── metadata.json
    │                           ├── grid_search_config.yaml
    │                           ├── results.json        # All combos tracked (including pending)
    │                           ├── summary.csv         # Aggregated results with token stats
    │                           └── combos/
    │                               ├── 001-DISABLED/    # Combo run directory (no extra nesting)
    │                               │   ├── metadata.json
    │                               │   ├── config.yaml
    │                               │   ├── *.eval           # .eval files directly here (not in eval/)
    │                               │   ├── safety_analysis.jsonl
    │                               │   ├── safety_lookahead.log
    │                               │   ├── token_stats.txt
    │                               │   └── token_stats.csv
    │                               └── 002-REMINDER-N1-V8-FORCED-NO-MASK/
    │                                   └── ... (same structure)
    │
    └── adhoc/                             # Unnamed runs
        └── {benchmark}_{sanitized_model}/
            └── {timestamp}/
                ├── config.yaml
                ├── eval/
                ├── safety_analysis.jsonl
                └── safety_lookahead.log

Usage:
    from eval_poc.results_path import ResultsPathBuilder

    # Named experiment
    run_dir = ResultsPathBuilder.for_experiment("exp1", "strong_reject", "safety-lookahead/qwen3-8b")

    # Adhoc (unnamed) run
    run_dir = ResultsPathBuilder.for_adhoc("strong_reject", "qwen3-8b")

    # Grid search
    grid_base = ResultsPathBuilder.for_grid_search("exp1", "strong_reject", "safety-lookahead/qwen3-8b")
    combo_dir = ResultsPathBuilder.for_grid_combo(grid_base, 1)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


# Project root detection
MODULE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = MODULE_DIR.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"


class ResultsPathBuilder:
    """Centralized path builder for results directories."""

    # Constants for subdirectory names
    EXPERIMENTS_DIR = "experiments"
    ADHOC_DIR = "adhoc"
    EVAL_DIR = "eval"
    GRID_SEARCH_DIR = "grid_search"
    COMBOS_DIR = "combos"

    @staticmethod
    def sanitize_model_name(model: str) -> str:
        """
        Convert model name to filesystem-safe format.

        Args:
            model: Model name (e.g., "safety-lookahead/qwen3-8b")

        Returns:
            Sanitized name (e.g., "safety-lookahead_qwen3-8b")
        """
        return model.replace("/", "_").replace("\\", "_")

    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp in YYYYMMDD-HHMM format.

        Returns:
            Timestamp string like "20250125-1430"
        """
        return datetime.now().strftime("%Y%m%d-%H%M")

    @staticmethod
    def get_date() -> str:
        """
        Get current date in MMDD format for directory naming.

        Returns:
            Date string like "0227" for February 27
        """
        return datetime.now().strftime("%m%d")

    @classmethod
    def for_experiment(
        cls,
        run_name: str,
        benchmark: str,
        model: str,
        timestamp: Optional[str] = None,
    ) -> Path:
        """
        Build path for named experiment run.

        Directory structure:
            results/experiments/{run_name}/{benchmark}/{sanitized_model}_{timestamp}/

        Args:
            run_name: Experiment name (e.g., "exp1", "baseline")
            benchmark: Benchmark name (e.g., "strong_reject")
            model: Model name (e.g., "safety-lookahead/qwen3-8b")
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Path to the experiment run directory

        Example:
            >>> ResultsPathBuilder.for_experiment("exp1", "strong_reject", "safety-lookahead/qwen3-8b", "20250125-1430")
            Path('results/experiments/exp1/strong_reject/safety-lookahead_qwen3-8b_20250125-1430')
        """
        ts = timestamp or cls.get_timestamp()
        safe_model = cls.sanitize_model_name(model)
        return (
            RESULTS_ROOT
            / cls.EXPERIMENTS_DIR
            / run_name
            / benchmark
            / f"{safe_model}_{ts}"
        )

    @classmethod
    def for_adhoc(
        cls,
        benchmark: str,
        model: str,
        timestamp: Optional[str] = None,
    ) -> Path:
        """
        Build path for adhoc (unnamed) run.

        Directory structure:
            results/adhoc/{benchmark}_{sanitized_model}/{timestamp}/

        Args:
            benchmark: Benchmark name (e.g., "strong_reject")
            model: Model name (e.g., "qwen3-8b")
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Path to the adhoc run directory

        Example:
            >>> ResultsPathBuilder.for_adhoc("strong_reject", "qwen3-8b", "20250125-1430")
            Path('results/adhoc/strong_reject_qwen3-8b/20250125-1430')
        """
        ts = timestamp or cls.get_timestamp()
        safe_model = cls.sanitize_model_name(model)
        return RESULTS_ROOT / cls.ADHOC_DIR / f"{benchmark}_{safe_model}" / ts

    @classmethod
    def for_grid_search(
        cls,
        run_name: str,
        benchmark: str,
        model: str,
        timestamp: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Path:
        """
        Build path for grid search experiment (single benchmark).

        Directory structure:
            results/experiments/grid_search/{date}/{run_name}/{benchmark}_{sanitized_model}/{timestamp}/

        Args:
            run_name: Experiment name (e.g., "exp1", "baseline")
            benchmark: Benchmark name (e.g., "strong_reject")
            model: Model name (e.g., "safety-lookahead/qwen3-8b")
            timestamp: Optional timestamp (defaults to current time)
            date: Optional date in MMDD format (defaults to current date)

        Returns:
            Path to the grid search base directory

        Example:
            >>> ResultsPathBuilder.for_grid_search("exp1", "strong_reject", "safety-lookahead/qwen3-8b", "20250125-1430", "0125")
            Path('results/experiments/grid_search/0125/exp1/strong_reject_safety-lookahead_qwen3-8b/20250125-1430')
        """
        ts = timestamp or cls.get_timestamp()
        dt = date or cls.get_date()
        safe_model = cls.sanitize_model_name(model)
        return (
            RESULTS_ROOT
            / cls.EXPERIMENTS_DIR
            / cls.GRID_SEARCH_DIR
            / dt
            / run_name
            / f"{benchmark}_{safe_model}"
            / ts
        )

    @classmethod
    def for_multi_benchmark_grid_search(
        cls,
        run_name: str,
        model: str,
        timestamp: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Path:
        """
        Build path for multi-benchmark grid search experiment.

        Directory structure:
            results/experiments/grid_search/{date}/{run_name}/multi_{sanitized_model}/{timestamp}/

        The "multi_" prefix indicates multiple benchmarks are included.

        Args:
            run_name: Experiment name (e.g., "exp1", "baseline")
            model: Model name (e.g., "safety-lookahead/qwen3-8b")
            timestamp: Optional timestamp (defaults to current time)
            date: Optional date in MMDD format (defaults to current date)

        Returns:
            Path to the grid search base directory

        Example:
            >>> ResultsPathBuilder.for_multi_benchmark_grid_search("exp1", "safety-lookahead/qwen3-8b", "20250125-1430", "0125")
            Path('results/experiments/grid_search/0125/exp1/multi_safety-lookahead_qwen3-8b/20250125-1430')
        """
        ts = timestamp or cls.get_timestamp()
        dt = date or cls.get_date()
        safe_model = cls.sanitize_model_name(model)
        return (
            RESULTS_ROOT
            / cls.EXPERIMENTS_DIR
            / cls.GRID_SEARCH_DIR
            / dt
            / run_name
            / f"multi_{safe_model}"
            / ts
        )

    @classmethod
    def for_grid_search_with_model(
        cls,
        run_name: str,
        benchmark: str,
        model: str,
        date: Optional[str] = None,
    ) -> Path:
        """
        Build path for grid search with model as subdirectory.

        Directory structure:
            results/experiments/grid_search/{date}/{run_name}/{benchmark}/{sanitized_model}/

        Args:
            run_name: Experiment name
            benchmark: Benchmark name (e.g., "agent_bench")
            model: Model name (e.g., "safety-lookahead/qwen3-8b")
            date: Optional date in MMDD format (defaults to current date)

        Returns:
            Path to the grid search base directory (contains combos/ subdirectory)

        Example:
            >>> ResultsPathBuilder.for_grid_search_with_model("exp1", "agent_bench", "doubao-seed-1-8-251228", "0227")
            Path('results/experiments/grid_search/0227/exp1/agent_bench/doubao-seed-1-8-251228')
        """
        dt = date or cls.get_date()
        safe_model = cls.sanitize_model_name(model)
        return (
            RESULTS_ROOT
            / cls.EXPERIMENTS_DIR
            / cls.GRID_SEARCH_DIR
            / dt
            / run_name
            / benchmark
            / safe_model
        )

    @classmethod
    def for_grid_combo(cls, grid_base: Path, combo_index: int) -> Path:
        """
        Build path for a single grid search combination.

        Directory structure:
            .../grid_search/{timestamp}/combos/{index:03d}/

        Args:
            grid_base: Base path from for_grid_search()
            combo_index: Combination index (1-based)

        Returns:
            Path to the combination directory

        Example:
            >>> base = ResultsPathBuilder.for_grid_search("exp1", "strong_reject", "safety-lookahead/qwen3-8b")
            >>> ResultsPathBuilder.for_grid_combo(base, 1)
            Path('results/experiments/grid_search/exp1/strong_reject_safety-lookahead_qwen3-8b/.../combos/001')
        """
        return grid_base / cls.COMBOS_DIR / f"{combo_index:03d}"

    @classmethod
    def get_eval_subdir(cls, run_dir: Path) -> Path:
        """
        Get the eval subdirectory path for inspect_ai .eval files.

        Args:
            run_dir: Base run directory

        Returns:
            Path to the eval subdirectory
        """
        return run_dir / cls.EVAL_DIR


def create_metadata_json(
    run_dir: Path,
    *,
    run_name: Optional[str] = None,
    benchmark: Optional[str] = None,
    benchmarks: Optional[list[str]] = None,
    model: Optional[str] = None,
    models: Optional[list[str]] = None,
    timestamp: Optional[str] = None,
    date: Optional[str] = None,
    safety_lookahead_config: Optional[dict] = None,
    git_commit: Optional[str] = None,
) -> Path:
    """
    Create metadata.json file in the run directory.

    Args:
        run_dir: Directory to write metadata.json
        run_name: Experiment name (if applicable)
        benchmark: Benchmark name (for single benchmark)
        benchmarks: List of benchmark names (for multi-benchmark)
        model: Model name (original, unsanitized)
        models: List of model names (for multi-model mode)
        timestamp: Run timestamp
        date: Run date in MMDD format (for grid search)
        safety_lookahead_config: Safety lookahead configuration dict
        git_commit: Git commit hash (optional)

    Returns:
        Path to the created metadata.json file
    """
    metadata: dict[str, Any] = {}

    if run_name:
        metadata["run_name"] = run_name
    if benchmarks:
        metadata["benchmarks"] = benchmarks
    elif benchmark:
        metadata["benchmark"] = benchmark
    if model:
        metadata["model"] = model
    if models:
        metadata["models"] = models
    if timestamp:
        metadata["timestamp"] = timestamp
    if date:
        metadata["date"] = date
    if safety_lookahead_config:
        metadata["safety_lookahead"] = safety_lookahead_config
    if git_commit:
        metadata["git_commit"] = git_commit

    metadata_path = run_dir / "metadata.json"

    import json

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata_path
