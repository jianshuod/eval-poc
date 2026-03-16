#!/usr/bin/env python3
"""
Safety-Lookahead Eval 入口脚本 - 支持 safety-lookahead 功能的统一 Eval 入口

用法:
    ./run-eval-salt.py <benchmark>[:<task>] --model <model_name> [options]

示例:
    # 基本用法 (与 run-eval.py 相同)
    ./run-eval-salt.py strong_reject --model doubao-seed-1-8-251228

    # 启用 safety-lookahead
    ./run-eval-salt.py --with-safety-lookahead --model safety-lookahead/qwen3-8b strong_reject

    # 使用单独的 world model
    ./run-eval-salt.py --with-safety-lookahead --world-model openai/gpt-4o --model safety-lookahead/qwen3-8b strong_reject

    # 启用 safety-confirmation (agent-driven safety reasoning)
    ./run-eval-salt.py --with-safety-confirmation --model safety-confirmation/qwen3-8b strong_reject

    # 启用 safety-prompting (baseline system prompt injection)
    ./run-eval-salt.py --with-safety-prompting --model safety-prompting/qwen3-8b strong_reject

    # 使用不同的 safety-confirmation 版本
    ./run-eval-salt.py --with-safety-confirmation --safety-confirmation-version v2 --model safety-confirmation/qwen3-8b strong_reject

    # 使用不同的 safety insight mode (requires v7)
    ./run-eval-salt.py --with-safety-lookahead --safety-mode reminder --safety-version v7 --model safety-lookahead/qwen3-8b strong_reject
    ./run-eval-salt.py --with-safety-lookahead --safety-mode context_analysis --safety-version v7 --model safety-lookahead/qwen3-8b strong_reject

    # Pass-through inspect_ai arguments (e.g., --limit, --model-role, --temperature)
    ./run-eval-salt.py strong_reject --model test --limit 10 --model-role grader=gpt-4o

Grid Search Mode:
    ./run-eval-salt.py --grid --config configs/grid-search.yaml
    ./run-eval-salt.py --grid --config configs/grid-search.yaml --grid-limit 10
    ./run-eval-salt.py --grid --config configs/grid-search.yaml --grid-dry-run

    Grid search requires a config file with grid_search.dimensions section.
    See configs/grid-search-template.yaml for examples.

一键测评:
    ./run-eval-salt.py --run-all --model <model_name>    # 运行所有 benchmark
    ./run-eval-salt.py --preflight                        # 仅运行预检查

环境管理:
    ./run-eval-salt.py --setup <benchmark>    # 仅设置环境，不运行
    ./run-eval-salt.py --setup-all            # 设置所有 benchmark 环境

Safety-Lookahead 选项:
    --with-safety-lookahead    启用 safety-lookahead 功能
    --world-model <model>      使用单独的模型作为 world model
    --safety-mode <mode>       Safety insight mode: reminder, spec_repeat, context_analysis, world_model (default: world_model)
    --safety-version <ver>     Safety lookahead version: v1-v7 (default: v4, v7 required for safety-mode)
    --safety-n <int>           Number of candidate actions to evaluate (default: 3)
    --safety-mask <strategy>   Tool call masking: keywords (fast), rewriting (LLM), none (default: none)
    --safety-forced            Force safety_check tool calls
    --safety-depth <n>         Safety lookahead passes required (SAFETY_LOOKAHEAD_PASSES_REQUIRED, v1)
    --safety-breadth <n>       Safety lookahead action proposals (SAFETY_LOOKAHEAD_ACTIONS_N, v1)
    --safety-timeout <sec>     API call timeout in seconds (default: 120)

Safety-Confirmation 选项:
    --with-safety-confirmation 启用 safety-confirmation 功能 (agent-driven safety reasoning)
    --safety-confirmation-version <ver> Safety confirmation version: v1, v2, v3, v4, v5, v6, v7 (default: v7)
    --safety-reasoning-guideline <mode> Safety reasoning guideline mode for v7+: general, structured, attacks, lookahead (default: structured)
    --safety-max-reconsider <n> Max reconsideration attempts (default: 2)
    --safety-n-required <n> Number of safety_confirmation calls required to unlock (v6+, default: 1)

Safety-Prompting 选项:
    --with-safety-prompting 启用 safety-prompting baseline (system prompt injection)
    --safety-prompting-version <ver> Safety prompting version: v1 (default: v1)

输出文件 (自动保存到 run_dir):
    safety_analysis.jsonl      安全分析结果 (JSONL 格式)
    safety_lookahead.log       运行时日志
    safety_confirmation.log    Safety confirmation 日志

inspect_ai 参数直接透传:
    --limit <n>              限制样本数量
    --model-role <role>      设置特定角色的模型 (e.g., grader=gpt-4o)
    --temperature <float>    设置温度
    --max-tokens <int>       设置最大 token 数
    ... (所有其他 inspect_ai 参数)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# 导入预检查模块
sys.path.insert(0, str(Path(__file__).parent / "benchmarks"))

# 导入 grid search 模块
from eval_poc.grid_search import (
    aggregate_token_stats,
    combination_to_dir_name,
    combination_to_cli_args,
    create_temp_config,
    generate_combinations,
    print_summary,
    write_summary_csv,
)
from eval_poc.results_path import (
    RESULTS_ROOT,
    ResultsPathBuilder,
    create_metadata_json,
)
from preflight import (
    JudgeModelConfig,
    run_preflight_checks,
    print_preflight_report,
    get_required_permissions,
)


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()
VENVS_DIR = PROJECT_ROOT / ".venvs"
UPSTREAM_DIR = PROJECT_ROOT / "upstream"


def resolve_safety_confirmation_path() -> Path | None:
    """Resolve the editable safety-confirmation repo path."""
    override = os.environ.get("SAFETY_CONFIRMATION_PATH")
    override_path: Path | None = None

    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return override_path.resolve()
        print(f"  警告: SAFETY_CONFIRMATION_PATH 指定的路径不存在: {override_path}")

    candidates = [
        PROJECT_ROOT.parent / "safety-confirmation-exp" / "safety-confirmation",
        PROJECT_ROOT.parent / "safety_confirmation",
        PROJECT_ROOT.parent / "safety-confirmation",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    print("  错误: 未找到 safety-confirmation 仓库")
    print("  尝试的路径:")
    if override_path is not None:
        print(f"    - {override_path} (from SAFETY_CONFIRMATION_PATH)")
    for candidate in candidates:
        print(f"    - {candidate}")
    print("  提示: 设置 SAFETY_CONFIRMATION_PATH 指向本地 safety-confirmation 仓库")
    return None


def resolve_safety_lookahead_path() -> Path | None:
    """Resolve the editable safety-lookahead repo path."""
    override = os.environ.get("SAFETY_LOOKAHEAD_PATH")
    override_path: Path | None = None

    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return override_path.resolve()
        print(f"  警告: SAFETY_LOOKAHEAD_PATH 指定的路径不存在: {override_path}")

    candidates = [
        PROJECT_ROOT.parent / "safety-confirmation-exp" / "safety-lookahead",
        PROJECT_ROOT.parent / "safety-lookahead",
        PROJECT_ROOT.parent / "safety_lookahead",
        UPSTREAM_DIR / "safety_lookahead",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    print("  错误: 未找到 safety-lookahead 仓库")
    print("  尝试的路径:")
    if override_path is not None:
        print(f"    - {override_path} (from SAFETY_LOOKAHEAD_PATH)")
    for candidate in candidates:
        print(f"    - {candidate}")
    print("  提示: 设置 SAFETY_LOOKAHEAD_PATH 指向本地 safety-lookahead 仓库，或初始化 git submodule: git submodule update --init upstream/safety_lookahead")
    return None


def is_safety_lookahead_from_path(benchmark_name: str, expected_path: Path) -> bool:
    """Check whether the safety-lookahead package in the venv resolves to expected_path."""
    if not expected_path.exists():
        return False

    python_path = get_venv_python(benchmark_name)
    result = subprocess.run(
        [str(python_path), "-c", "import safety_lookahead, os; print(os.path.abspath(safety_lookahead.__file__))"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return False

    installed_path = Path(result.stdout.strip()).resolve()
    expected_root = expected_path.resolve()
    return str(installed_path).startswith(str(expected_root))


def _version_tuple(version_str: str) -> tuple[int, int, int]:
    """Convert a version string to a 3-part integer tuple for comparison."""
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
        if len(parts) >= 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _ensure_openai_min_version(benchmark_name: str, min_version: str = "2.26.0") -> bool:
    """Ensure the benchmark venv has openai >= min_version."""
    python_path = get_venv_python(benchmark_name)
    result = subprocess.run(
        [str(python_path), "-c", "import openai; print(getattr(openai, '__version__', ''))"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  无法读取 {benchmark_name} 环境的 openai 版本，正在重装 openai...")
        return _install_openai_in_venv(benchmark_name, min_version)

    installed = result.stdout.strip()
    if not installed:
        print(f"  {benchmark_name} 环境未检测到 openai，正在安装...")
        return _install_openai_in_venv(benchmark_name, min_version)

    if _version_tuple(installed) < _version_tuple(min_version):
        print(f"  检测到 openai 版本 {installed}，低于最低要求 {min_version}，正在升级...")
        return _install_openai_in_venv(benchmark_name, min_version)

    return True


def _install_openai_in_venv(benchmark_name: str, min_version: str = "2.26.0") -> bool:
    """Install/upgrade OpenAI in the benchmark venv to satisfy minimum version."""
    venv_path = get_venv_path(benchmark_name)
    result = subprocess.run(
        ["uv", "pip", "install", "-p", str(venv_path), f"openai>={min_version}"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 安装 openai>={min_version} 失败")
        print(result.stderr)
        return False
    return True


# ==============================================================================
# YAML Configuration Support
# ==============================================================================

def load_config(config_file: str) -> dict:
    """Load YAML configuration file"""
    config_path = Path(config_file)

    # Try relative path from current directory
    if not config_path.is_absolute():
        # Try from current working directory
        cwd_path = Path.cwd() / config_file
        if cwd_path.exists():
            config_path = cwd_path
        # Try from script directory
        else:
            script_path = PROJECT_ROOT / config_file
            if script_path.exists():
                config_path = script_path

    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_file}")
        print(f"  尝试的路径:")
        print(f"    - {Path.cwd() / config_file}")
        print(f"    - {PROJECT_ROOT / config_file}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def config_to_cli_args(config: dict) -> list[str]:
    """
    Convert YAML config to inspect_ai CLI arguments.

    Maps nested YAML keys to their corresponding CLI flags.
    """
    args: list[str] = []

    # Helper to convert truthy/falsy strings to proper format
    def to_bool(value: Any) -> str | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            v_lower = value.lower()
            if v_lower in ("true", "yes", "1", "on"):
                return "true"
            if v_lower in ("false", "no", "0", "off"):
                return "false"
        return str(value)

    def add_arg(flag: str, value: Any, inverted: bool = False) -> None:
        """Add a CLI argument if value is not empty"""
        if value is None or value == "":
            return
        # Handle boolean flags that are inverted (e.g., --no-xyz)
        if isinstance(value, bool) or (isinstance(value, str) and value.lower() in ("true", "false", "yes", "no")):
            bool_val = to_bool(value)
            if bool_val == "true":
                if inverted:
                    args.append(f"--no-{flag[2:]}")  # Convert --flag to --no-flag
                else:
                    args.append(flag)
            # If false and not inverted, don't add the flag
            # If false and inverted, add the positive flag
            elif inverted:
                args.append(flag)
        else:
            if inverted:
                # For inverted boolean flags, we only add when value is false
                if not value or (isinstance(value, str) and value.lower() in ("false", "no", "0")):
                    args.append(flag)
            else:
                args.extend([flag, str(value)])

    def add_repeated(flag: str, values: list | None) -> None:
        """Add repeated CLI arguments"""
        if not values:
            return
        for v in values:
            if v and v != "":
                args.extend([flag, str(v)])

    # ------------------------------------------------------------------
    # Model Configuration
    # ------------------------------------------------------------------
    model_cfg = config.get("model", {})

    if model_cfg.get("base_url"):
        args.extend(["--model-base-url", model_cfg["base_url"]])

    if model_cfg.get("args"):
        add_repeated("-M", model_cfg["args"])

    if model_cfg.get("config"):
        args.extend(["--model-config", model_cfg["config"]])

    if model_cfg.get("roles"):
        add_repeated("--model-role", model_cfg["roles"])

    # ------------------------------------------------------------------
    # Task Execution Parameters
    # ------------------------------------------------------------------
    task_cfg = config.get("task", {})

    add_arg("--limit", task_cfg.get("limit"))
    add_arg("--sample-id", task_cfg.get("sample_id"))
    add_arg("--sample-shuffle", task_cfg.get("sample_shuffle"))
    add_arg("--epochs", task_cfg.get("epochs"))
    add_arg("--epochs-reducer", task_cfg.get("epochs_reducer"))
    add_arg("--solver", task_cfg.get("solver"))

    if task_cfg.get("solver_args"):
        add_repeated("-S", task_cfg["solver_args"])

    if task_cfg.get("solver_config"):
        args.extend(["--solver-config", task_cfg["solver_config"]])

    if task_cfg.get("task_args"):
        add_repeated("-T", task_cfg["task_args"])

    if task_cfg.get("task_config"):
        args.extend(["--task-config", task_cfg["task_config"]])

    # ------------------------------------------------------------------
    # Model Generation Parameters
    # ------------------------------------------------------------------
    gen_cfg = config.get("generation", {})

    add_arg("--temperature", gen_cfg.get("temperature"))
    add_arg("--top-p", gen_cfg.get("top_p"))
    add_arg("--top-k", gen_cfg.get("top_k"))
    add_arg("--max-tokens", gen_cfg.get("max_tokens"))
    add_arg("--stop-seqs", gen_cfg.get("stop_seqs"))
    add_arg("--frequency-penalty", gen_cfg.get("frequency_penalty"))
    add_arg("--presence-penalty", gen_cfg.get("presence_penalty"))
    add_arg("--num-choices", gen_cfg.get("num_choices"))
    add_arg("--best-of", gen_cfg.get("best_of"))
    add_arg("--seed", gen_cfg.get("seed"))
    add_arg("--logprobs", to_bool(gen_cfg.get("logprobs")))
    add_arg("--top-logprobs", gen_cfg.get("top_logprobs"))
    add_arg("--logit-bias", gen_cfg.get("logit_bias"))
    add_arg("--system-message", gen_cfg.get("system_message"))

    # Claude-specific
    add_arg("--reasoning-effort", gen_cfg.get("reasoning_effort"))
    add_arg("--reasoning-tokens", gen_cfg.get("reasoning_tokens"))
    add_arg("--reasoning-summary", gen_cfg.get("reasoning_summary"))
    add_arg("--reasoning-history", gen_cfg.get("reasoning_history"))
    add_arg("--effort", gen_cfg.get("effort"))
    add_arg("--cache-prompt", gen_cfg.get("cache_prompt"))

    # OpenAI-specific
    add_arg("--verbosity", gen_cfg.get("verbosity"))

    # Response format
    add_arg("--response-schema", gen_cfg.get("response_schema"))

    # ------------------------------------------------------------------
    # Tool Configuration
    # ------------------------------------------------------------------
    tools_cfg = config.get("tools", {})

    add_arg("--parallel-tool-calls", to_bool(tools_cfg.get("parallel_calls")))
    add_arg("--internal-tools", to_bool(tools_cfg.get("internal_tools")))
    add_arg("--max-tool-output", tools_cfg.get("max_output"))

    if tools_cfg.get("approval"):
        args.extend(["--approval", tools_cfg["approval"]])

    # ------------------------------------------------------------------
    # Concurrency and Performance
    # ------------------------------------------------------------------
    conc_cfg = config.get("concurrency", {})

    add_arg("--max-samples", conc_cfg.get("max_samples"))
    add_arg("--max-tasks", conc_cfg.get("max_tasks"))
    add_arg("--max-subprocesses", conc_cfg.get("max_subprocesses"))
    add_arg("--max-sandboxes", conc_cfg.get("max_sandboxes"))
    add_arg("--max-connections", conc_cfg.get("max_connections"))
    add_arg("--max-retries", conc_cfg.get("max_retries"))
    add_arg("--timeout", conc_cfg.get("timeout"))
    add_arg("--attempt-timeout", conc_cfg.get("attempt_timeout"))

    # ------------------------------------------------------------------
    # Task Limits
    # ------------------------------------------------------------------
    limits_cfg = config.get("limits", {})

    add_arg("--message-limit", limits_cfg.get("message"))
    add_arg("--token-limit", limits_cfg.get("token"))
    add_arg("--time-limit", limits_cfg.get("time"))
    add_arg("--working-limit", limits_cfg.get("working"))

    # ------------------------------------------------------------------
    # Caching and Batching
    # ------------------------------------------------------------------
    perf_cfg = config.get("performance", {})

    add_arg("--cache", perf_cfg.get("cache"))
    add_arg("--batch", perf_cfg.get("batch"))

    # ------------------------------------------------------------------
    # Error Handling
    # ------------------------------------------------------------------
    err_cfg = config.get("error_handling", {})

    add_arg("--fail-on-error", err_cfg.get("fail_on_error"))
    add_arg("--continue-on-fail", to_bool(err_cfg.get("continue_on_fail")))
    add_arg("--retry-on-error", err_cfg.get("retry_on_error"))
    add_arg("--debug-errors", to_bool(err_cfg.get("debug_errors")))

    # ------------------------------------------------------------------
    # Logging and Output
    # ------------------------------------------------------------------
    log_cfg = config.get("logging", {})

    add_arg("--log-level", log_cfg.get("log_level"))
    add_arg("--log-level-transcript", log_cfg.get("log_level_transcript"))
    add_arg("--log-format", log_cfg.get("log_format"))
    add_arg("--log-dir", log_cfg.get("log_dir"))

    # Inverted flags for logging
    log_samples_val = log_cfg.get("log_samples")
    if log_samples_val is not None and log_samples_val != "":
        if to_bool(log_samples_val) == "false":
            args.append("--no-log-samples")

    log_realtime_val = log_cfg.get("log_realtime")
    if log_realtime_val is not None and log_realtime_val != "":
        if to_bool(log_realtime_val) == "false":
            args.append("--no-log-realtime")

    add_arg("--log-images", to_bool(log_cfg.get("log_images")))
    add_arg("--log-buffer", log_cfg.get("log_buffer"))
    add_arg("--log-shared", log_cfg.get("log_shared"))
    add_arg("--no-score", to_bool(log_cfg.get("no_score")))
    add_arg("--no-score-display", to_bool(log_cfg.get("no_score_display")))

    # ------------------------------------------------------------------
    # Sandbox
    # ------------------------------------------------------------------
    sandbox_cfg = config.get("sandbox", {})

    if sandbox_cfg.get("type"):
        args.extend(["--sandbox", sandbox_cfg["type"]])

    add_arg("--no-sandbox-cleanup", to_bool(sandbox_cfg.get("no_cleanup")), inverted=True)

    # ------------------------------------------------------------------
    # Display Options
    # ------------------------------------------------------------------
    display_cfg = config.get("display", {})

    if display_cfg.get("format"):
        args.extend(["--display", display_cfg["format"]])

    add_arg("--no-ansi", to_bool(display_cfg.get("no_ansi")))
    add_arg("--traceback-locals", to_bool(display_cfg.get("traceback_locals")))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    meta_cfg = config.get("metadata", {})

    if meta_cfg.get("tags"):
        args.extend(["--tags", meta_cfg["tags"]])

    if meta_cfg.get("values"):
        add_repeated("--metadata", meta_cfg["values"])

    # ------------------------------------------------------------------
    # Environment Variables
    # ------------------------------------------------------------------
    env_cfg = config.get("env", {})

    if env_cfg.get("variables"):
        add_repeated("--env", env_cfg["variables"])

    # ------------------------------------------------------------------
    # Safety-Lookahead Options
    # ------------------------------------------------------------------
    salt_cfg = config.get("safety_lookahead", {})

    add_arg("--with-safety-lookahead", to_bool(salt_cfg.get("enabled")))
    add_arg("--safety-version", salt_cfg.get("version"))
    add_arg("--safety-n", salt_cfg.get("n"))
    add_arg("--safety-mask", salt_cfg.get("mask"))
    add_arg("--safety-mode", salt_cfg.get("mode"))
    add_arg("--safety-forced", to_bool(salt_cfg.get("forced")))
    add_arg("--safety-timeout", salt_cfg.get("timeout"))
    if salt_cfg.get("world_model"):
        add_arg("--world-model", salt_cfg["world_model"])

    # ------------------------------------------------------------------
    # Safety-Confirmation Options
    # ------------------------------------------------------------------
    sconf_cfg = config.get("safety_confirmation", {})

    add_arg("--with-safety-confirmation", to_bool(sconf_cfg.get("enabled")))
    add_arg("--safety-confirmation-version", sconf_cfg.get("version"))
    add_arg("--safety-reasoning-guideline", sconf_cfg.get("reasoning_guideline"))
    add_arg("--safety-max-reconsider", sconf_cfg.get("max_reconsideration"))
    add_arg("--safety-n-required", sconf_cfg.get("n_required"))

    # ------------------------------------------------------------------
    # Debug Options
    # ------------------------------------------------------------------
    debug_cfg = config.get("debug", {})

    add_arg("--debug", to_bool(debug_cfg.get("enabled")))
    add_arg("--debug-port", debug_cfg.get("port"))

    return args


def merge_cli_args(cli_args: list[str], config_args: list[str]) -> list[str]:
    """
    Merge CLI args and config args. CLI args take precedence.

    For repeated flags (like -M, -T, --model-role), the CLI values override
    config values entirely.

    For boolean flags, the last occurrence takes effect.
    """
    # Parse config args into key-value pairs
    config_flags: dict[str, list[str]] = {}
    i = 0
    while i < len(config_args):
        arg = config_args[i]
        if arg.startswith("--") or arg.startswith("-"):
            flag = arg
            # Check if next arg is a value (not a flag)
            if i + 1 < len(config_args) and not config_args[i + 1].startswith("-"):
                if flag not in config_flags:
                    config_flags[flag] = []
                config_flags[flag].append(config_args[i + 1])
                i += 2
            else:
                # Boolean flag
                config_flags[flag] = []
                i += 1
        else:
            i += 1

    # Parse CLI args - these override config
    cli_flags: dict[str, list[str]] = {}
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if arg.startswith("--") or arg.startswith("-"):
            flag = arg
            # Check if next arg is a value (not a flag)
            if i + 1 < len(cli_args) and not cli_args[i + 1].startswith("-"):
                if flag not in cli_flags:
                    cli_flags[flag] = []
                cli_flags[flag].append(cli_args[i + 1])
                i += 2
            else:
                # Boolean flag
                cli_flags[flag] = []
                i += 1
        else:
            i += 1

    # Merge: config flags + CLI flags (CLI overrides)
    merged: list[str] = []

    # First add config flags
    for flag, values in config_flags.items():
        # Only add if not overridden by CLI
        if flag not in cli_flags:
            if values:
                for v in values:
                    merged.extend([flag, v])
            else:
                merged.append(flag)

    # Then add CLI flags (these take precedence)
    for flag, values in cli_flags.items():
        if values:
            for v in values:
                merged.extend([flag, v])
        else:
            merged.append(flag)

    return merged


def apply_config_to_args(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    """
    Apply config values to default argparse Namespace values.
    Only applies if the CLI argument wasn't explicitly provided.
    """
    # Create a mutable copy of args
    result = argparse.Namespace(**vars(args))

    # Apply model config
    model_cfg = config.get("model", {})
    if model_cfg.get("base") and not hasattr(result, 'model_from_config'):
        # Store the base model from config
        result.model_from_config = model_cfg["base"]

    # Apply safety_lookahead config
    salt_cfg = config.get("safety_lookahead", {})
    if salt_cfg.get("enabled") and not result.with_safety_lookahead:
        result.with_safety_lookahead = True
    if salt_cfg.get("version") and result.safety_version is None:
        result.safety_version = salt_cfg["version"]
    if salt_cfg.get("n") and result.safety_n is None:
        result.safety_n = salt_cfg["n"]
    if salt_cfg.get("mask") and result.safety_mask is None:
        result.safety_mask = salt_cfg["mask"]
    if salt_cfg.get("mode") and result.safety_mode is None:
        result.safety_mode = salt_cfg["mode"]
    if salt_cfg.get("forced") and not result.safety_forced:
        result.safety_forced = True
    if salt_cfg.get("timeout") and result.safety_timeout is None:
        result.safety_timeout = salt_cfg["timeout"]
    if salt_cfg.get("depth") and result.safety_depth is None:
        result.safety_depth = salt_cfg["depth"]
    if salt_cfg.get("breadth") and result.safety_breadth is None:
        result.safety_breadth = salt_cfg["breadth"]
    # Note: output_dir and log_file are automatically set to run_dir, not configurable

    # Apply safety_confirmation config
    sconf_cfg = config.get("safety_confirmation", {})
    if sconf_cfg.get("enabled") and not result.with_safety_confirmation:
        result.with_safety_confirmation = True
    if sconf_cfg.get("version") and result.safety_confirmation_version == "v6":
        result.safety_confirmation_version = sconf_cfg["version"]
    if sconf_cfg.get("max_reconsideration") and result.safety_max_reconsider is None:
        result.safety_max_reconsider = sconf_cfg["max_reconsideration"]
    if sconf_cfg.get("n_required") and result.safety_n_required is None:
        result.safety_n_required = sconf_cfg["n_required"]

    # Apply world model from model config
    if model_cfg.get("world") and result.world_model is None:
        result.world_model = model_cfg["world"]

    # Apply run config
    run_cfg = config.get("run", {})
    if run_cfg.get("dry_run") and not result.dry_run:
        result.dry_run = True
    if run_cfg.get("skip_preflight") and not result.skip_preflight:
        result.skip_preflight = True
    if run_cfg.get("force_setup") and not result.force:
        result.force = True

    # Apply grid_search config
    grid_cfg = config.get("grid_search", {})
    if grid_cfg.get("parallel") and not result.grid_parallel:
        result.grid_parallel = True
    if grid_cfg.get("workers") and result.grid_workers is None:
        result.grid_workers = grid_cfg["workers"]
    if grid_cfg.get("fail_fast") and not result.grid_fail_fast:
        result.grid_fail_fast = True

    # Apply run_name from config (top-level, not in run section)
    if config.get("run_name") and result.run_name is None:
        result.run_name = config["run_name"]

    return result


def load_catalog():
    """加载 benchmark 路由配置"""
    catalog_path = PROJECT_ROOT / "benchmarks" / "catalog.yaml"
    with open(catalog_path, "r") as f:
        return yaml.safe_load(f)


def sanitize_model_name(model_name: str) -> str:
    """处理模型名称：斜杠替换为下划线"""
    return model_name.replace("/", "_")


def normalize_model_name(model_name: str) -> str:
    """
    规范化模型名称为 inspect_ai 所需格式。
    如果模型名称不包含 '/'，则默认添加 'openai/' 前缀。
    """
    if "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def get_venv_path(benchmark_name: str) -> Path:
    """获取 benchmark 的虚拟环境路径"""
    return VENVS_DIR / benchmark_name


def get_venv_python(benchmark_name: str) -> Path:
    """获取 benchmark 虚拟环境的 Python 路径"""
    return get_venv_path(benchmark_name) / "bin" / "python"


def get_venv_inspect(benchmark_name: str) -> Path:
    """获取 benchmark 虚拟环境的 inspect 命令路径"""
    return get_venv_path(benchmark_name) / "bin" / "inspect"


def setup_benchmark_env(benchmark_name: str, config: dict, force: bool = False,
                       with_safety_lookahead: bool = False,
                       with_safety_confirmation: bool = False,
                       with_safety_prompting: bool = False) -> bool:
    """
    为 benchmark 设置独立虚拟环境

    返回 True 表示成功，False 表示失败
    """
    venv_path = get_venv_path(benchmark_name)
    python_version = config.get("python", "3.10")
    extras = config.get("extras", [])

    # 检查是否已存在
    if venv_path.exists() and not force:
        inspect_path = get_venv_inspect(benchmark_name)
        if inspect_path.exists():
            # 如果启用 safety-lookahead，检查是否已安装
            if with_safety_lookahead:
                safety_lookahead_path = resolve_safety_lookahead_path()
                if safety_lookahead_path is None:
                    return False

                installed_ok = subprocess.run(
                    [str(get_venv_python(benchmark_name)), "-c",
                     "import safety_lookahead"],
                    capture_output=True
                ).returncode == 0 and is_safety_lookahead_from_path(
                    benchmark_name,
                    safety_lookahead_path
                )
                if not installed_ok:
                    print(f"  环境已存在，但 safety_lookahead 未安装或来源不匹配，正在安装...")
                    return _install_safety_lookahead(benchmark_name)
            # 如果启用 safety-confirmation，检查是否已安装
            if with_safety_confirmation:
                result = subprocess.run(
                    [str(get_venv_python(benchmark_name)), "-c",
                     "import safety_confirmation"],
                    capture_output=True
                )
                if result.returncode != 0:
                    print(f"  环境已存在，但 safety_confirmation 未安装，正在安装...")
                    return _install_safety_confirmation(benchmark_name)
            # 如果启用 safety-prompting，检查是否已安装
            if with_safety_prompting:
                result = subprocess.run(
                    [str(get_venv_python(benchmark_name)), "-c",
                     "import safety_prompting"],
                    capture_output=True
                )
                if result.returncode != 0:
                    print(f"  环境已存在，但 safety_prompting 未安装，正在安装...")
                    return _install_safety_prompting(benchmark_name)
            print(f"  环境已存在: {venv_path}")
            return True

    print(f"  创建环境: {venv_path} (Python {python_version})")

    # 创建虚拟环境
    venv_cmd = ["uv", "venv", str(venv_path), "--python", python_version]
    if force:
        venv_cmd.append("--clear")

    result = subprocess.run(
        venv_cmd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 创建虚拟环境失败")
        print(result.stderr)
        return False

    # 安装 inspect_ai
    print(f"  安装 inspect_ai...")
    result = subprocess.run(
        ["uv", "pip", "install", "-p", str(venv_path),
         "-e", str(UPSTREAM_DIR / "inspect_ai")],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 安装 inspect_ai 失败")
        print(result.stderr)
        return False

    # 安装 inspect_evals (带 extras)
    install_spec = str(UPSTREAM_DIR / "inspect_evals")
    if extras:
        extras_str = ",".join(extras)
        install_spec = f"{install_spec}[{extras_str}]"

    extras_display = f"[{','.join(extras)}]" if extras else ""
    print(f"  安装 inspect_evals{extras_display}...")
    result = subprocess.run(
        ["uv", "pip", "install", "-p", str(venv_path), "-e", install_spec],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 安装 inspect_evals 失败")
        print(result.stderr)
        return False

    # 安装 openai (必需)
    if not _install_openai_in_venv(benchmark_name=benchmark_name):
        print(f"  错误: 安装 openai 失败")
        return False

    # 本地 benchmarks 需要 benchmarks 包
    if config.get("source") == "local":
        benchmarks_path = PROJECT_ROOT / "benchmarks"
        print(f"  安装 benchmarks 包...")
        result = subprocess.run(
            ["uv", "pip", "install", "-p", str(venv_path), "-e", str(benchmarks_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  错误: 安装 benchmarks 包失败")
            print(result.stderr)
            return False

        # 安装本地 benchmark 的 extras (如 browsergym, gymnasium 等)
        if extras:
            extras_display = f"[{','.join(extras)}]" if extras else ""
            print(f"  安装本地 benchmark extras{extras_display}...")
            for extra in extras:
                result = subprocess.run(
                    ["uv", "pip", "install", "-p", str(venv_path), extra],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"  错误: 安装 {extra} 失败")
                    print(result.stderr)
                    return False

    # cve_bench 需要单独安装 cvebench 包
    if benchmark_name == "cve_bench":
        print(f"  安装 cvebench...")
        result = subprocess.run(
            ["uv", "pip", "install", "-p", str(venv_path),
             "cvebench @ git+https://github.com/Scott-Simmons/cve-bench.git@92541add2ebd89e5b15ed260eb5d0e9b5102c33e"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  错误: 安装 cvebench 失败")
            print(result.stderr)
            return False

    # 安装 safety-lookahead (如果启用)
    if with_safety_lookahead:
        if not _install_safety_lookahead(benchmark_name):
            return False

    # 安装 safety-confirmation (如果启用)
    if with_safety_confirmation:
        if not _install_safety_confirmation(benchmark_name):
            return False

    # 安装 safety-prompting (如果启用)
    if with_safety_prompting:
        if not _install_safety_prompting(benchmark_name):
            return False

    print(f"  环境设置完成")
    return True


def _install_safety_lookahead(benchmark_name: str) -> bool:
    """
    安装 safety-lookahead 到指定 benchmark 的虚拟环境

    返回 True 表示成功，False 表示失败
    """
    venv_path = get_venv_path(benchmark_name)
    safety_lookahead_path = resolve_safety_lookahead_path()

    if safety_lookahead_path is None:
        return False

    print(f"  安装 safety-lookahead...")
    result = subprocess.run(
        ["uv", "pip", "install", "-p", str(venv_path), "-e", str(safety_lookahead_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 安装 safety-lookahead 失败")
        print(result.stderr)
        return False

    print(f"  safety-lookahead 安装完成")
    return True


def _install_safety_confirmation(benchmark_name: str) -> bool:
    """
    安装 safety-confirmation 到指定 benchmark 的虚拟环境

    safety-confirmation is now independent and does not depend on safety-lookahead.

    返回 True 表示成功，False 表示失败
    """
    venv_path = get_venv_path(benchmark_name)
    safety_confirmation_path = resolve_safety_confirmation_path()

    if safety_confirmation_path is None:
        return False

    # Install safety-confirmation
    print(f"  安装 safety-confirmation...")
    result = subprocess.run(
        ["uv", "pip", "install", "-p", str(venv_path), "-e", str(safety_confirmation_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 安装 safety-confirmation 失败")
        print(result.stderr)
        return False

    print(f"  safety-confirmation 安装完成")
    return True


def _install_safety_prompting(benchmark_name: str) -> bool:
    """
    安装 safety-prompting 到指定 benchmark 的虚拟环境

    safety-prompting is a baseline safety enhancement via system prompt injection.

    返回 True 表示成功，False 表示失败
    """
    venv_path = get_venv_path(benchmark_name)
    # safety_prompting is in the parent directory of eval-poc
    safety_prompting_path = PROJECT_ROOT.parent / "safety_prompting"

    if not safety_prompting_path.exists():
        print(f"  错误: safety_prompting 路径不存在: {safety_prompting_path}")
        return False

    # Install safety-prompting
    print(f"  安装 safety-prompting...")
    result = subprocess.run(
        ["uv", "pip", "install", "-p", str(venv_path), "-e", str(safety_prompting_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 安装 safety-prompting 失败")
        print(result.stderr)
        return False

    print(f"  safety-prompting 安装完成")
    return True


def resolve_task(benchmark_spec: str, catalog: dict) -> tuple[str, str, dict, dict]:
    """
    解析 benchmark 规格，返回 (benchmark_name, task_spec, config, task_config)
    """
    if ":" in benchmark_spec:
        benchmark_name, task_name = benchmark_spec.split(":", 1)
    else:
        benchmark_name = benchmark_spec
        task_name = None

    benchmarks = catalog.get("benchmarks", {})
    if benchmark_name not in benchmarks:
        available = ", ".join(benchmarks.keys())
        print(f"错误: 未知的 benchmark '{benchmark_name}'")
        print(f"可用的 benchmarks: {available}")
        sys.exit(1)

    config = benchmarks[benchmark_name]
    tasks = config.get("tasks", [])

    if not tasks:
        print(f"错误: benchmark '{benchmark_name}' 没有定义任何 task")
        sys.exit(1)

    def find_task(name: str):
        for t in tasks:
            if isinstance(t, dict):
                if t.get("name") == name:
                    return t
            elif t == name:
                return {"name": t, "path": config["module"]}
        return None

    if task_name:
        task = find_task(task_name)
        if not task:
            available = ", ".join(
                t.get("name") if isinstance(t, dict) else t
                for t in tasks
            )
            print(f"错误: benchmark '{benchmark_name}' 中没有 task '{task_name}'")
            print(f"可用的 tasks: {available}")
            sys.exit(1)
    else:
        first_task = tasks[0]
        if isinstance(first_task, dict):
            task = first_task
        else:
            task = {"name": first_task, "path": config["module"]}

    task_spec = task["path"]
    return benchmark_name, task_spec, config, task


def run_eval(benchmark_name: str, task_spec: str, config: dict,
             model: str, inspect_args: list = None, dry_run: bool = False,
             task_config: dict = None,
             with_safety_lookahead: bool = False,
             with_safety_confirmation: bool = False,
             with_safety_prompting: bool = False,
             world_model: str = None,
             safety_version: str = None,
             safety_n: int = None,
             safety_mask: str = None,
             safety_mode: str = None,
             safety_forced: bool = False,
             safety_timeout: int = None,
             safety_depth: int = None,
             safety_breadth: int = None,
             safety_max_reconsider: int = None,
             safety_confirmation_version: str = "v6",
             safety_n_required: int = None,
             safety_reasoning_guideline: str = None,
             safety_prompting_version: str = "v1",
             run_name: str = None,
             grid_search_combo_dir: Path = None,
             env_vars: list = None,
             config_file: str = None) -> int:
    """运行评估"""

    # 确保环境存在
    inspect_path = get_venv_inspect(benchmark_name)
    if not inspect_path.exists():
        print(f"设置 {benchmark_name} 环境...")
        if not setup_benchmark_env(benchmark_name, config,
                                    with_safety_lookahead=with_safety_lookahead,
                                    with_safety_confirmation=with_safety_confirmation,
                                    with_safety_prompting=with_safety_prompting):
            return 1
    else:
        if not _ensure_openai_min_version(benchmark_name):
            return 1

        # 如果启用 safety-lookahead 但未安装，安装它
        if with_safety_lookahead:
            safety_lookahead_path = resolve_safety_lookahead_path()
            if safety_lookahead_path is None:
                return 1

            if not (
                subprocess.run(
                    [str(get_venv_python(benchmark_name)), "-c", "import safety_lookahead"],
                    capture_output=True
                ).returncode == 0 and is_safety_lookahead_from_path(
                    benchmark_name,
                    safety_lookahead_path
                )
            ):
                print(f"安装 safety-lookahead 到 {benchmark_name} 环境...")
                if not _install_safety_lookahead(benchmark_name):
                    return 1

        # 如果启用 safety-confirmation 但未安装，安装它
        if with_safety_confirmation:
            result = subprocess.run(
                [str(get_venv_python(benchmark_name)), "-c", "import safety_confirmation"],
                capture_output=True
            )
            if result.returncode != 0:
                print(f"安装 safety-confirmation 到 {benchmark_name} 环境...")
                if not _install_safety_confirmation(benchmark_name):
                    return 1

        # 如果启用 safety-prompting 但未安装，安装它
        if with_safety_prompting:
            result = subprocess.run(
                [str(get_venv_python(benchmark_name)), "-c", "import safety_prompting"],
                capture_output=True
            )
            if result.returncode != 0:
                print(f"安装 safety-prompting 到 {benchmark_name} 环境...")
                if not _install_safety_prompting(benchmark_name):
                    return 1

    # 规范化模型名称
    model_for_inspect = normalize_model_name(model)
    sanitized_model = sanitize_model_name(model)

    # 获取任务名称，构建完整的 benchmark 标识
    task_name = task_config.get("name", "") if task_config else ""
    if task_name and task_name != benchmark_name:
        benchmark_full = f"{benchmark_name}-{task_name}"
    else:
        benchmark_full = benchmark_name

    # 创建运行目录 (使用统一的 ResultsPathBuilder)
    # - Named experiment: results/experiments/{run_name}/{benchmark}/{model}_{timestamp}/
    # - Adhoc (unnamed): results/adhoc/{benchmark}_{model}/{timestamp}/
    # - Grid search combo: uses provided combo_dir directly (no nested timestamp)

    timestamp = ResultsPathBuilder.get_timestamp()

    # Grid search mode: use the combo_dir directly (no nested timestamp)
    # The combo directory name already encodes parameters (e.g., 001-REMINDER-N1-V8-FORCED-NO-MASK)
    if grid_search_combo_dir is not None:
        run_dir = grid_search_combo_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    elif run_name:
        # Named experiment: use new structure
        run_dir = ResultsPathBuilder.for_experiment(
            run_name=run_name,
            benchmark=benchmark_full,
            model=model,
            timestamp=timestamp
        )
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Adhoc (unnamed) run: use new structure
        run_dir = ResultsPathBuilder.for_adhoc(
            benchmark=benchmark_full,
            model=model,
            timestamp=timestamp
        )
        run_dir.mkdir(parents=True, exist_ok=True)

    # 复制配置文件到结果目录 (放在 timestamp 子目录下)
    if config_file:
        config_source = Path(config_file)
        if not config_source.is_absolute():
            # 尝试从当前目录、脚本目录查找
            if not (Path.cwd() / config_file).exists():
                config_source = PROJECT_ROOT / config_file
            else:
                config_source = Path.cwd() / config_file
        if config_source.exists():
            config_dest = run_dir / config_source.name
            import shutil
            # Skip copy if source and destination are the same (e.g., grid search combo)
            if config_source != config_dest:
                shutil.copy2(config_source, config_dest)
                print(f"配置已复制到: {config_dest}")

    # 设置 eval 结果目录
    # - Grid search combo: .eval files directly in run_dir (no eval/ subdirectory)
    # - Regular runs: .eval files in eval/ subdirectory
    if grid_search_combo_dir is not None:
        # Grid search: no eval/ subdirectory, files go directly in combo directory
        eval_results_dir = run_dir
    else:
        # Regular runs: use eval/ subdirectory
        eval_results_dir = ResultsPathBuilder.get_eval_subdir(run_dir)
        eval_results_dir.mkdir(parents=True, exist_ok=True)

    # Safety-lookahead 输出文件 (固定在 run_dir 下)
    # - safety_analysis.jsonl: 安全分析结果
    # - safety_lookahead.log: 运行时日志
    safety_analysis_file = run_dir / "safety_analysis.jsonl"
    log_file = run_dir / "safety_lookahead.log"

    # Safety-confirmation 输出文件 (固定在 run_dir 下)
    safety_confirmation_analysis_file = run_dir / "safety_confirmation_analysis.jsonl"
    safety_confirmation_log_file = run_dir / "safety_confirmation.log"

    # Create metadata.json with run configuration
    safety_config = None
    if with_safety_lookahead:
        safety_config = {
            "type": "lookahead",
            "enabled": True,
            "version": safety_version,
            "world_model": world_model,
            "n": safety_n,
            "mask": safety_mask,
            "mode": safety_mode,
            "forced": safety_forced,
            "timeout": safety_timeout,
            "depth": safety_depth,
            "breadth": safety_breadth,
        }
        # Remove None values
        safety_config = {k: v for k, v in safety_config.items() if v is not None}
    elif with_safety_confirmation:
        safety_config = {
            "type": "confirmation",
            "enabled": True,
            "max_reconsideration": safety_max_reconsider,
        }
        # Remove None values
        safety_config = {k: v for k, v in safety_config.items() if v is not None}

    create_metadata_json(
        run_dir,
        run_name=run_name,
        benchmark=benchmark_full,
        model=model,
        timestamp=timestamp,
        safety_lookahead_config=safety_config,
    )

    # 设置环境变量
    env = os.environ.copy()
    env["INSPECT_LOG_DIR"] = str(eval_results_dir)

    # For local benchmarks, add eval-poc directory to PYTHONPATH
    # (not benchmarks subdirectory, because imports use benchmarks.local.*)
    source = config.get("source", "upstream")
    if source == "local":
        eval_poc_dir = str(PROJECT_ROOT)  # PROJECT_ROOT is the eval-poc directory
        existing_pythonpath = env.get("PYTHONPATH", "")

        # For ST-WebAgentBench, also add the ST-WebAgentBench directory to PYTHONPATH
        # This is required for browsergym.stwebagentbench to find stwebagentbench module
        monorepo_root = PROJECT_ROOT.parent  # Go up one level to reach monorepo root
        st_bench_path = monorepo_root / "ST-WebAgentBench"

        # Build PYTHONPATH with all required directories
        pythonpath_entries = [eval_poc_dir]
        if st_bench_path.exists():
            pythonpath_entries.insert(0, str(st_bench_path))

        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)

        env["PYTHONPATH"] = ":".join(pythonpath_entries)

    # 清除可能影响 inspect_ai 缓存路径的 VSCode 扩展环境变量
    for key in ["INSPECT_WORKSPACE_ID", "INSPECT_VSCODE_EXT_VERSION"]:
        env.pop(key, None)

    # Apply environment variables from config
    # (Note: This is passed from main() via eval_config parameter)
    # Variables set via config will override system env vars
    if env_vars:
        for env_var in env_vars:
            if '=' in env_var:
                key, value = env_var.split('=', 1)
                env[key] = value

    # Safety-lookahead 环境变量
    if with_safety_lookahead:
        env["SAFETY_LOOKAHEAD_ENABLED"] = "true"

        if safety_version:
            env["SAFETY_LOOKAHEAD_VERSION"] = safety_version

        if world_model:
            # Don't normalize world model - it's used directly with OpenAI client
            env["SAFETY_LOOKAHEAD_WORLD_MODEL"] = world_model

        if safety_n is not None:
            env["SAFETY_LOOKAHEAD_N"] = str(safety_n)

        if safety_mask:
            env["SAFETY_LOOKAHEAD_MASK"] = safety_mask

        if safety_mode:
            env["SAFETY_LOOKAHEAD_MODE"] = safety_mode

        if safety_forced:
            env["SAFETY_LOOKAHEAD_FORCED"] = "true"

        if safety_timeout is not None:
            env["SAFETY_LOOKAHEAD_TIMEOUT"] = str(safety_timeout)

        if safety_depth is not None:
            env["SAFETY_LOOKAHEAD_PASSES_REQUIRED"] = str(safety_depth)

        if safety_breadth is not None:
            env["SAFETY_LOOKAHEAD_ACTIONS_N"] = str(safety_breadth)

        # Always set output paths to run_dir
        env["SAFETY_LOOKAHEAD_OUTPUT"] = str(safety_analysis_file)
        env["SAFETY_LOOKAHEAD_LOG_FILE"] = str(log_file)

    # Safety-confirmation 环境变量
    if with_safety_confirmation:
        env["SAFETY_CONFIRMATION_ENABLED"] = "true"
        env["SAFETY_CONFIRMATION_VERSION"] = safety_confirmation_version

        if safety_max_reconsider is not None:
            env["SAFETY_CONFIRMATION_MAX_RECONSIDERATION"] = str(safety_max_reconsider)

        if safety_n_required is not None:
            env["SAFETY_CONFIRMATION_N_REQUIRED"] = str(safety_n_required)

        if safety_reasoning_guideline is not None:
            env["SAFETY_CONFIRMATION_REASONING_GUIDELINE"] = str(safety_reasoning_guideline)

        # Always set output paths to run_dir
        env["SAFETY_CONFIRMATION_OUTPUT"] = str(safety_confirmation_analysis_file)
        env["SAFETY_CONFIRMATION_LOG_FILE"] = str(safety_confirmation_log_file)

        # Use same timeout setting for safety-confirmation
        if safety_timeout is not None:
            env["SAFETY_CONFIRMATION_TIMEOUT"] = str(safety_timeout)

    # Safety-prompting 环境变量
    if with_safety_prompting:
        env["SAFETY_PROMPTING_ENABLED"] = "true"
        env["SAFETY_PROMPTING_VERSION"] = safety_prompting_version

    # 构建命令
    cmd = [str(inspect_path), "eval", task_spec, "--model", model_for_inspect]

    # 添加 task_args (来自 catalog.yaml)
    if task_config:
        task_args = task_config.get("task_args", {})
        for key, value in task_args.items():
            cmd.extend(["-T", f"{key}={value}"])

    # Pass through all other inspect_ai arguments
    if inspect_args:
        cmd.extend(inspect_args)

    # 打印信息
    print(f"Benchmark: {benchmark_name}")
    print(f"Task: {task_spec}")
    print(f"Model: {model_for_inspect}")
    print(f"Run dir: {run_dir}")
    if with_safety_lookahead:
        print(f"Safety-lookahead: ENABLED")
        if safety_version:
            print(f"  Version: {safety_version}")
        if world_model:
            print(f"  World model: {world_model}")
        if safety_n is not None:
            print(f"  Candidates (N): {safety_n}")
        if safety_mask:
            print(f"  Mask strategy: {safety_mask}")
        if safety_mode:
            print(f"  Safety mode: {safety_mode}")
        if safety_depth is not None:
            print(f"  Safety passes required (depth): {safety_depth}")
        if safety_breadth is not None:
            print(f"  Safety action proposals (breadth): {safety_breadth}")
        if safety_forced:
            print(f"  Forced safety check: true")
        if safety_timeout is not None:
            print(f"  API timeout: {safety_timeout}s")
        print(f"  Safety analysis: {safety_analysis_file}")
        print(f"  Log file: {log_file}")
    elif with_safety_confirmation:
        print(f"Safety-confirmation: ENABLED")
        print(f"  Version: {safety_confirmation_version}")
        if safety_max_reconsider is not None:
            print(f"  Max reconsideration: {safety_max_reconsider}")
        if safety_n_required is not None:
            print(f"  N required: {safety_n_required}")
        if safety_reasoning_guideline is not None:
            print(f"  Reasoning guideline: {safety_reasoning_guideline}")
        if safety_timeout is not None:
            print(f"  API timeout: {safety_timeout}s")
        print(f"  Safety analysis: {safety_confirmation_analysis_file}")
        print(f"  Log file: {safety_confirmation_log_file}")
    elif with_safety_prompting:
        print(f"Safety-prompting: ENABLED")
        print(f"  Version: {safety_prompting_version}")
    print(f"Command: {' '.join(cmd)}")
    print()

    if dry_run:
        print("[Dry run - 不实际执行]")
        return 0

    # 执行命令
    result = subprocess.run(cmd, env=env)

    # Generate token stats whenever possible.
    # Even failed evals may still emit partial .eval logs that contain token usage.
    try:
        # Use subprocess to call token_stats_generator to avoid import path issues
        # Derive venv python path from inspect executable path
        # inspect_path is like: .venvs/<benchmark>/bin/inspect
        venv_python = inspect_path.parent / "python"

        if venv_python.exists():
            # Call token_stats_generator as subprocess
            token_script = PROJECT_ROOT / "token_stats_generator.py"
            if result.returncode != 0:
                print(
                    "Eval exited non-zero; attempting token stats extraction from any available logs..."
                )
            print(f"Generating token stats using {venv_python}...")
            token_result = subprocess.run(
                [str(venv_python), str(token_script), str(run_dir)],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)  # Ensure correct working directory
            )
            if token_result.returncode == 0:
                print(f"Token stats saved to: {run_dir / 'token_stats.txt'}")
                # Show summary output from token_stats_generator
                if token_result.stdout:
                    print(token_result.stdout)
            else:
                print(f"Warning: Token stats generation failed (exit code {token_result.returncode})")
                if token_result.stderr:
                    print(f"  stderr: {token_result.stderr}")
                if token_result.stdout:
                    print(f"  stdout: {token_result.stdout}")
        else:
            print(f"Warning: Could not find venv python at {venv_python}")
    except Exception as e:
        print(f"Warning: Failed to generate token stats: {e}")

    return result.returncode


# ==============================================================================
# Grid Search Functions
# ==============================================================================

def run_single_combination(
    combo: dict,
    index: int,
    base_config: dict,
    combo_dir: Path,
    args: argparse.Namespace,
    inspect_args: list,
    catalog: dict,
    dry_run: bool = False,
) -> dict:
    """
    Run a single grid search combination.

    Args:
        combo: Dictionary of parameter values for this combination
        index: Combination index (1-based)
        base_config: Base configuration from grid search file
        combo_dir: Directory for this combination's output
        args: Parsed command-line arguments
        inspect_args: Additional inspect_ai arguments
        catalog: Benchmark catalog
        dry_run: If True, print without executing

    Returns:
        Dictionary with result status and metadata
    """
    dir_name = combination_to_dir_name(index, combo)

    # Resolve benchmark from combo (for multi-benchmark mode) or base_config
    # For single-benchmark mode, get from benchmarks list
    benchmark_name = (
        combo.get("benchmark") or
        (base_config.get("benchmarks") or [])[0] if base_config.get("benchmarks") else
        base_config.get("benchmark")
    )
    if not benchmark_name:
        return {
            "index": index,
            "dir_name": dir_name,
            "combination": combo,
            "output_dir": str(combo_dir),
            "status": "failed",
            "error": "No benchmark specified"
        }

    benchmark_name, task_spec, config, task_config = resolve_task(benchmark_name, catalog)

    # Build metadata
    metadata = {
        "index": index,
        "dir_name": dir_name,
        "combination": combo,
        "combo_dir": str(combo_dir),
    }

    print(f"[{index}] Running: {dir_name}")
    print(f"    Config: {combo}")

    if dry_run:
        print(f"    [Dry run - skipping]")
        print()
        return {**metadata, "status": "dry_run", "exit_code": 0}

    # Map combination to run_eval parameters
    enabled = combo.get("enabled", True)
    dim_type = combo.get("type", "")

    # Determine which safety mode to use
    is_safety_lookahead = (dim_type == "safety_lookahead")
    is_safety_confirmation = (dim_type == "safety_confirmation")

    # Get the model to use - from combination if available (multi-model mode)
    # otherwise use global args.model (single-model mode from CLI)
    # otherwise use model from base_config (single-model mode from config)
    model_to_use = combo.get("model") or args.model
    if not model_to_use:
        # Try to get model from base_config (for single-model mode from config)
        model_cfg = base_config.get("model", {})
        model_to_use = model_cfg.get("models", [model_cfg.get("base")])[0] if model_cfg.get("models") else model_cfg.get("base")
    if enabled:
        if is_safety_lookahead and model_to_use and not model_to_use.startswith("safety-lookahead/"):
            model_to_use = f"safety-lookahead/{model_to_use}"
        elif is_safety_confirmation and model_to_use and not model_to_use.startswith("safety-confirmation/"):
            model_to_use = f"safety-confirmation/{model_to_use}"

    # Get world model from base config (only for safety_lookahead)
    world_model = None
    if is_safety_lookahead:
        world_model = combo.get("world")
        if not world_model:
            world_model = base_config.get("model", {}).get("world")

    # Build command and run
    returncode = run_eval(
        benchmark_name=benchmark_name,
        task_spec=task_spec,
        config=config,
        model=model_to_use,
        inspect_args=inspect_args,
        dry_run=False,
        task_config=task_config,
        with_safety_lookahead=enabled and is_safety_lookahead,
        with_safety_confirmation=enabled and is_safety_confirmation,
        world_model=world_model,
        safety_version=combo.get("version") if is_safety_lookahead else None,
        safety_n=combo.get("n") if is_safety_lookahead else None,
        safety_mask=combo.get("mask") if is_safety_lookahead else None,
        safety_mode=combo.get("mode") if is_safety_lookahead else None,
        safety_depth=combo.get("depth") if is_safety_lookahead else None,
        safety_breadth=combo.get("breadth") if is_safety_lookahead else None,
        safety_forced=combo.get("forced", False) if is_safety_lookahead else False,
        safety_timeout=combo.get("timeout"),
        safety_max_reconsider=combo.get("max_reconsideration") if is_safety_confirmation else None,
        safety_confirmation_version=combo.get("version") if is_safety_confirmation else "v6",
        safety_n_required=combo.get("n_required") if is_safety_confirmation else None,
        safety_reasoning_guideline=combo.get("reasoning_guideline") if is_safety_confirmation else None,
        run_name=base_config.get("run_name"),
        grid_search_combo_dir=combo_dir,
        env_vars=base_config.get("env", {}).get("variables", []),
        config_file=str(combo_dir / "config.yaml"),
    )

    status = "success" if returncode == 0 else "failed"
    print(f"    Status: {status}")
    print()

    return {
        **metadata,
        "status": status,
        "exit_code": returncode,
        "output_dir": str(combo_dir),
        "log_file": str(combo_dir / "run.log"),
    }


def run_grid_search(
    config: dict,
    args: argparse.Namespace,
    inspect_args: list[str],
    catalog: dict,
) -> int:
    """
    Execute grid search across all parameter combinations.

    Supports parallel execution with --grid-parallel flag.

    Args:
        config: Grid search configuration with grid_search.dimensions section
        args: Parsed command-line arguments
        inspect_args: Additional inspect_ai arguments
        catalog: Benchmark catalog

    Returns:
        Exit code (0 if all successful, 1 if any failed)
    """
    # 1. Generate combinations from grid_search.dimensions
    combinations = generate_combinations(config)

    if not combinations:
        print("错误: 从配置文件未生成任何组合")
        return 1

    print(f"Generated {len(combinations)} combinations from config")
    print()

    # 2. Apply --grid-limit if specified
    if args.grid_limit:
        combinations = combinations[:args.grid_limit]
        print(f"Limited to {len(combinations)} combinations")
        print()

    # 3. Create output directory using ResultsPathBuilder
    # Detect multi-benchmark and multi-model modes
    benchmarks = config.get("benchmarks", [])
    is_multi_benchmark = len(benchmarks) > 1

    # Detect multi-model mode from config
    model_cfg = config.get("model", {})
    models = model_cfg.get("models", [])
    base_model = model_cfg.get("base")

    # Normalize models list (models takes precedence over base)
    if models:
        pass  # Already have models list
    elif base_model:
        models = [base_model]
    else:
        models = []

    is_multi_model = len(models) > 1

    run_name = config.get("run_name", "grid_search")
    model = args.model or ""

    timestamp = ResultsPathBuilder.get_timestamp()
    date = ResultsPathBuilder.get_date()

    # Use different path builders based on mode
    if is_multi_model and is_multi_benchmark:
        # Both multi-model and multi-benchmark: create combined structure
        # Use first model/benchmark as reference for main output_base_dir
        # Each combination will have its own full path
        output_base_dir = (
            RESULTS_ROOT
            / ResultsPathBuilder.EXPERIMENTS_DIR
            / ResultsPathBuilder.GRID_SEARCH_DIR
            / date
            / run_name
        )
    elif is_multi_model:
        # Multi-model, single-benchmark: use benchmark as parent, model as subdirectory
        benchmark = benchmarks[0] if benchmarks else config.get("benchmark", "")
        # Use first model for the main output_base_dir
        first_model = models[0]
        output_base_dir = ResultsPathBuilder.for_grid_search_with_model(
            run_name, benchmark, first_model, date
        )
    elif is_multi_benchmark:
        output_base_dir = ResultsPathBuilder.for_multi_benchmark_grid_search(run_name, model, timestamp, date)
    else:
        # Extract single benchmark from benchmarks list
        benchmark = benchmarks[0] if benchmarks else config.get("benchmark", "")
        output_base_dir = ResultsPathBuilder.for_grid_search(run_name, benchmark, model, timestamp, date)

    output_base_dir.mkdir(parents=True, exist_ok=True)

    # For multi-model mode, combos_base_dir is not a single directory
    # Each combination gets its own base directory based on model
    # For single-model mode, use the existing combos_base_dir
    if is_multi_model:
        # Helper function to get combo directory for multi-model mode
        def get_combo_dir_for_combo(idx: int, combo: dict) -> Path:
            """Get the combo directory for a given combination in multi-model mode."""
            combo_model = combo.get("model", models[0])
            combo_benchmark = combo.get("benchmark", benchmarks[0] if benchmarks else config.get("benchmark", ""))
            model_base = ResultsPathBuilder.for_grid_search_with_model(
                run_name, combo_benchmark, combo_model, date
            )
            model_base.mkdir(parents=True, exist_ok=True)
            dir_name = combination_to_dir_name(idx, combo)
            return model_base / ResultsPathBuilder.COMBOS_DIR / dir_name
    else:
        # Create combos subdirectory for single-model mode
        combos_base_dir = output_base_dir / ResultsPathBuilder.COMBOS_DIR
        combos_base_dir.mkdir(parents=True, exist_ok=True)

        def get_combo_dir_for_combo(idx: int, combo: dict) -> Path:
            """Get the combo directory for a given combination in single-model mode."""
            dir_name = combination_to_dir_name(idx, combo)
            return combos_base_dir / dir_name

    print(f"Output directory: {output_base_dir}")
    print(f"Run name: {run_name}")
    if is_multi_benchmark:
        print(f"Benchmarks: {', '.join(benchmarks)}")
    elif benchmarks:
        print(f"Benchmark: {benchmarks[0]}")
    if is_multi_model:
        print(f"Models: {', '.join(models)}")
    elif models:
        print(f"Model: {models[0]}")
    print()

    # 4. Copy grid config to output directory
    if args.config:
        config_source = Path(args.config)
        if not config_source.is_absolute():
            if (Path.cwd() / args.config).exists():
                config_source = Path.cwd() / args.config
            else:
                config_source = PROJECT_ROOT / args.config
        if config_source.exists():
            shutil.copy2(config_source, output_base_dir / "grid_search_config.yaml")

    # 5. Create grid search metadata
    metadata_config = {
        "run_name": run_name,
        "model": args.model or "",
        "timestamp": timestamp,
        "date": date,
    }
    if is_multi_benchmark:
        metadata_config["benchmarks"] = benchmarks
    else:
        # Extract single benchmark from benchmarks list
        benchmark = benchmarks[0] if benchmarks else config.get("benchmark", "")
        metadata_config["benchmark"] = benchmark

    if is_multi_model:
        metadata_config["models"] = models

    if config.get("safety_lookahead"):
        metadata_config["safety_lookahead"] = config.get("safety_lookahead")

    create_metadata_json(output_base_dir, **metadata_config)

    # 6. Run combinations (parallel or sequential)
    results = []

    if args.grid_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        workers = args.grid_workers or (os.cpu_count() // 2)
        print(f"Running {len(combinations)} combinations with {workers} parallel workers")
        print()

        # Track results by index for incremental updates
        results_by_index = {}

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_combo = {}
                for idx, combo in enumerate(combinations, 1):
                    dir_name = combination_to_dir_name(idx, combo)
                    combo_dir = get_combo_dir_for_combo(idx, combo)
                    combo_dir.mkdir(parents=True, exist_ok=True)

                    # Create temp config for this combination
                    create_temp_config(config, combo, combo_dir)

                    # Add to results with "pending" status
                    result = {
                        "index": idx,
                        "dir_name": dir_name,
                        "combination": combo,
                        "output_dir": str(combo_dir),
                        "status": "pending",
                    }
                    results_by_index[idx] = result

                    # Save incremental results (before running)
                    with open(output_base_dir / "results.json", 'w') as f:
                        json.dump(list(results_by_index.values()), f, indent=2, default=str)

                    future = executor.submit(
                        run_single_combination,
                        combo=combo,
                        index=idx,
                        base_config=config,
                        combo_dir=combo_dir,
                        args=args,
                        inspect_args=inspect_args,
                        catalog=catalog,
                        dry_run=args.grid_dry_run,
                    )
                    future_to_combo[future] = (idx, combo, combo_dir)

                # Collect results as they complete
                for future in as_completed(future_to_combo):
                    idx, combo, combo_dir = future_to_combo[future]
                    try:
                        result = future.result()
                        results_by_index[idx] = result

                        # Update results.json incrementally
                        with open(output_base_dir / "results.json", 'w') as f:
                            json.dump(list(results_by_index.values()), f, indent=2, default=str)

                        # Fail fast if enabled
                        if args.grid_fail_fast and result["status"] == "failed":
                            print(f"\nCombination {idx} failed, stopping (--grid-fail-fast)")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

                    except Exception as e:
                        print(f"Combination {idx} raised exception: {e}")
                        results_by_index[idx] = {
                            "index": idx,
                            "status": "failed",
                            "error": str(e),
                        }
                        # Update results.json on error
                        with open(output_base_dir / "results.json", 'w') as f:
                            json.dump(list(results_by_index.values()), f, indent=2, default=str)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving partial results...")
            # Save final results before exiting
            with open(output_base_dir / "results.json", 'w') as f:
                json.dump(list(results_by_index.values()), f, indent=2, default=str)
            print(f"Results saved to: {output_base_dir / 'results.json'}")
            raise

        # Convert ordered dict to list
        results = list(results_by_index.values())

    else:
        # Sequential execution (existing behavior)
        for idx, combo in enumerate(combinations, 1):
            # Use encoded name for self-documenting directories
            dir_name = combination_to_dir_name(idx, combo)
            combo_dir = get_combo_dir_for_combo(idx, combo)
            combo_dir.mkdir(parents=True, exist_ok=True)

            # Create temp config for this combination
            create_temp_config(config, combo, combo_dir)

            # Add to results with "pending" status
            result = {
                "index": idx,
                "dir_name": dir_name,
                "combination": combo,
                "output_dir": str(combo_dir),
                "status": "pending",
            }
            results.append(result)

            # Save incremental results (before running)
            with open(output_base_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Run the evaluation
            run_result = run_single_combination(
                combo=combo,
                index=idx,
                base_config=config,
                combo_dir=combo_dir,
                args=args,
                inspect_args=inspect_args,
                catalog=catalog,
                dry_run=args.grid_dry_run,
            )

            # Update the result with actual run outcome
            results[-1] = run_result

            # Save incremental results (after running)
            with open(output_base_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)

    # 7. Write summary CSV
    write_summary_csv(results, output_base_dir)

    # 8. Aggregate and print token stats summary
    token_stats = aggregate_token_stats(output_base_dir)
    if token_stats and token_stats.get("combos_with_stats", 0) > 0:
        print()
        print("=" * 60)
        print("Grid Search Token Stats Summary")
        print("=" * 60)
        print(f"Combos with stats: {token_stats['combos_with_stats']}/{token_stats['combos_total']}")
        print(f"Total Input Tokens:  {token_stats['total_input_tokens']:,}")
        print(f"Total Output Tokens: {token_stats['total_output_tokens']:,}")
        print(f"Total Cache Read:    {token_stats['total_cache_read']:,}")
        print(f"Total Cache Write:   {token_stats['total_cache_write']:,}")
        print(f"Total Reasoning:     {token_stats['total_reasoning']:,}")
        print(f"---")
        print(f"Total Tokens:        {token_stats['total_tokens']:,}")
        print("=" * 60)
        print()

    # 9. Print summary
    print_summary(results)

    # Return non-zero if any failed
    failed_count = sum(1 for r in results if r["status"] == "failed")
    return 1 if failed_count > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Safety-Lookahead Eval 入口脚本 (支持 safety-lookahead 功能)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        help="Benchmark 名称，格式: benchmark_name 或 benchmark_name:task_name"
    )
    parser.add_argument(
        "--model", "-m",
        help="模型名称 (例如: doubao-seed-1-8-251228, safety-lookahead/qwen3-8b)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run identifier for organizing logs/results (e.g., 'exp1', 'baseline')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印命令，不实际执行"
    )
    parser.add_argument(
        "--setup",
        metavar="BENCHMARK",
        help="仅设置指定 benchmark 的环境"
    )
    parser.add_argument(
        "--setup-all",
        action="store_true",
        help="设置所有 benchmark 的环境"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新创建环境"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="运行预检查，检测环境依赖"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="运行所有 benchmark (一键测评)"
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过预检查 (不推荐)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="自动确认权限 (用于非交互式运行)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )

    # Safety-lookahead 相关参数
    parser.add_argument(
        "--with-safety-lookahead",
        action="store_true",
        help="启用 safety-lookahead model API wrapper"
    )
    parser.add_argument(
        "--world-model",
        type=str,
        default=None,
        help="单独的模型用于 world model 评估 (例如: openai/gpt-4o)"
    )
    parser.add_argument(
        "--safety-n",
        type=int,
        default=None,
        help="Number of candidate actions to evaluate (default: 3)"
    )
    parser.add_argument(
        "--safety-mask",
        type=str,
        default=None,
        choices=["keywords", "rewriting", "none"],
        help="Tool call masking strategy: keywords (fast), rewriting (LLM call), none (default: none)"
    )
    parser.add_argument(
        "--safety-mode",
        type=str,
        default=None,
        choices=["reminder", "spec_repeat", "context_analysis", "world_model"],
        help="Safety insight acquisition mode (default: world_model)"
    )
    parser.add_argument(
        "--safety-version",
        type=str,
        default=None,
        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7"],
        help="Safety lookahead version to use: v1-v7 (default: v4)"
    )
    parser.add_argument(
        "--safety-forced",
        action="store_true",
        help="Force the model to call the safety_check tool"
    )
    parser.add_argument(
        "--safety-timeout",
        type=int,
        default=None,
        help="Timeout for API calls in seconds (default: 120)"
    )
    parser.add_argument(
        "--safety-depth",
        type=int,
        default=None,
        help="Safety lookahead passes required (SAFETY_LOOKAHEAD_PASSES_REQUIRED)"
    )
    parser.add_argument(
        "--safety-breadth",
        type=int,
        default=None,
        help="Safety lookahead action proposals (SAFETY_LOOKAHEAD_ACTIONS_N)"
    )

    # Safety-confirmation 相关参数
    parser.add_argument(
        "--with-safety-confirmation",
        action="store_true",
        help="启用 safety-confirmation model API wrapper (agent-driven safety reasoning)"
    )
    parser.add_argument(
        "--safety-max-reconsider",
        type=int,
        default=None,
        help="Max reconsideration attempts for safety-confirmation (default: 2)"
    )
    parser.add_argument(
        "--safety-confirmation-version",
        type=str,
        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7"],
        default="v7",
        help="Safety confirmation version to use (default: v7)"
    )
    parser.add_argument(
        "--safety-reasoning-guideline",
        type=str,
        default=None,
        choices=["general", "structured", "attacks", "lookahead"],
        help="Safety reasoning guideline mode for safety-confirmation v7+ (default: structured)"
    )
    parser.add_argument(
        "--safety-n-required",
        type=int,
        default=None,
        help="Number of safety_confirmation calls required to unlock (v6+, default: 1)"
    )

    # Safety-prompting 相关参数
    parser.add_argument(
        "--with-safety-prompting",
        action="store_true",
        help="启用 safety-prompting baseline (system prompt injection)"
    )
    parser.add_argument(
        "--safety-prompting-version",
        type=str,
        choices=["v1"],
        default="v1",
        help="Safety prompting version to use (default: v1)"
    )

    # Grid search arguments
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Enable grid search mode (requires config with grid_search.dimensions)"
    )
    parser.add_argument(
        "--grid-parallel",
        action="store_true",
        help="Run grid search combinations in parallel (default: sequential)"
    )
    parser.add_argument(
        "--grid-workers",
        type=int,
        default=None,
        help="Number of parallel workers for grid search (default: CPU count // 2)"
    )
    parser.add_argument(
        "--grid-fail-fast",
        action="store_true",
        help="Stop grid search on first failure (only with --grid-parallel)"
    )
    parser.add_argument(
        "--grid-limit",
        type=int,
        help="Limit number of grid search combinations to run"
    )
    parser.add_argument(
        "--grid-dry-run",
        action="store_true",
        help="Dry run for grid search: print combinations without executing"
    )

    args, inspect_args = parser.parse_known_args()

    # Load config file if provided
    config: dict = {}
    env_vars: list = []
    if args.config:
        config = load_config(args.config)

        # Apply config values to args (only if not set via CLI)
        args = apply_config_to_args(args, config)

        # Extract environment variables from config
        env_vars = config.get("env", {}).get("variables", [])

        # Convert config to CLI args
        config_args = config_to_cli_args(config)

        # Merge CLI args and config args (CLI takes precedence)
        inspect_args = merge_cli_args(inspect_args, config_args)

    catalog = load_catalog()
    benchmarks = catalog.get("benchmarks", {})

    # Handle benchmark from config
    benchmark_from_config = config.get("benchmark")
    if benchmark_from_config:
        if args.benchmark is None:
            args.benchmark = benchmark_from_config

    # Handle model from config
    if hasattr(args, 'model_from_config') and args.model is None:
        args.model = args.model_from_config

    # Auto-add safety-lookahead/, safety-confirmation/, or safety-prompting/ prefix when enabled
    # and model doesn't already have the prefix
    salt_cfg = config.get("safety_lookahead", {})
    sconf_cfg = config.get("safety_confirmation", {})
    sprompt_cfg = config.get("safety_prompting", {})

    if salt_cfg.get("enabled") and args.model:
        # Check if model already has safety-lookahead prefix
        if not args.model.startswith("safety-lookahead/"):
            args.model = f"safety-lookahead/{args.model}"
    elif sconf_cfg.get("enabled") and args.model:
        # Check if model already has safety-confirmation prefix
        if not args.model.startswith("safety-confirmation/"):
            args.model = f"safety-confirmation/{args.model}"
    elif sprompt_cfg.get("enabled") and args.model:
        # Check if model already has safety-prompting prefix
        if not args.model.startswith("safety-prompting/"):
            args.model = f"safety-prompting/{args.model}"

    # Handle run_all from config
    if benchmark_from_config == "run_all":
        args.run_all = True

    # ==============================================================================
    # Grid Search Mode
    # ==============================================================================
    if args.grid:
        if not args.config:
            print("错误: --grid 需要 --config 参数指定 grid search 配置文件")
            return 1

        if "grid_search" not in config:
            print("错误: 配置文件必须包含 grid_search.dimensions 节")
            return 1

        # Ensure model is set
        # Support both model.base (legacy single model) and model.models (new multi-model)
        if not args.model:
            model_cfg = config.get("model", {})
            models = model_cfg.get("models", [])
            base_model = model_cfg.get("base")

            # Normalize to list (models takes precedence over base)
            if models:
                pass  # Already have models list
            elif base_model:
                models = [base_model]
            else:
                models = []

            if not models:
                print("错误: config.model.base 或 config.model.models 必须指定其一")
                return 1

            # For single model mode (legacy), auto-add safety-confirmation prefix
            # For multi-model mode, models are used as-is in combinations
            if len(models) == 1 and not model_cfg.get("models"):
                args.model = f"safety-confirmation/{models[0]}"

        return run_grid_search(config, args, inspect_args, catalog)

    # 仅运行预检查
    if args.preflight:
        benchmark_list = list(benchmarks.keys())
        # Use default judge model from catalog for cyberseceval_2
        default_judge = benchmarks.get("cyberseceval_2", {}).get("judge_model", "")
        judge_config = JudgeModelConfig(model_name=default_judge)
        results = run_preflight_checks(benchmark_list, judge_config)
        passed = print_preflight_report(results)

        # 显示权限确认
        permissions = get_required_permissions(benchmark_list)
        if permissions:
            print("\n需要用户确认的权限:")
            for perm in permissions:
                print(f"  • {perm}")
            print()

        return 0 if passed else 1

    # 一键测评
    if args.run_all:
        if not args.model:
            print("错误: --run-all 必须指定 --model 参数")
            return 1

        benchmark_list = list(benchmarks.keys())

        # 预检查
        if not args.skip_preflight:
            # Use default judge model from catalog for cyberseceval_2
            default_judge = benchmarks.get("cyberseceval_2", {}).get("judge_model", "")
            judge_config = JudgeModelConfig(model_name=default_judge)
            results = run_preflight_checks(benchmark_list, judge_config)
            passed = print_preflight_report(results)

            # 显示并确认权限
            permissions = get_required_permissions(benchmark_list)
            if permissions:
                print("\n需要用户确认的权限:")
                for perm in permissions:
                    print(f"  • {perm}")
                print()

                if not args.confirm:
                    try:
                        response = input("是否同意上述权限? [y/N]: ").strip().lower()
                        if response not in ("y", "yes"):
                            print("用户取消")
                            return 1
                    except (EOFError, KeyboardInterrupt):
                        print("\n用户取消")
                        return 1

            if not passed and not args.skip_preflight:
                print("\n预检查未通过。使用 --skip-preflight 可跳过检查（不推荐）")
                return 1

        # 运行所有 benchmark 的所有 tasks
        total_tasks = sum(
            len(config.get("tasks", [])) for config in benchmarks.values()
        )
        safety_info = " (with safety-lookahead)" if args.with_safety_lookahead else ""
        print(f"\n开始一键测评: {args.model}{safety_info}")
        print(f"共 {len(benchmarks)} 个 benchmark, {total_tasks} 个 tasks\n")
        print("=" * 60)

        VENVS_DIR.mkdir(parents=True, exist_ok=True)

        results_summary = []
        for name, config in benchmarks.items():
            tasks = config.get("tasks", [])
            if not tasks:
                print(f"\n[{name}]")
                print(f"  跳过: 没有定义 task")
                results_summary.append((name, None, "skipped", "no tasks"))
                continue

            # 遍历每个 task
            for task in tasks:
                if isinstance(task, dict):
                    task_name = task.get("name")
                    task_spec = task["path"]
                    task_config = task
                else:
                    task_name = task
                    task_spec = config["module"]
                    task_config = {"name": task, "path": task_spec}

                print(f"\n[{name}:{task_name}]")

                returncode = run_eval(
                    benchmark_name=name,
                    task_spec=task_spec,
                    config=config,
                    model=args.model,
                    inspect_args=inspect_args,
                    dry_run=args.dry_run,
                    task_config=task_config,
                    with_safety_lookahead=args.with_safety_lookahead,
                    with_safety_confirmation=args.with_safety_confirmation,
                    world_model=args.world_model,
                    safety_version=args.safety_version,
                    safety_n=args.safety_n,
                    safety_mask=args.safety_mask,
                    safety_mode=args.safety_mode,
                    safety_depth=args.safety_depth,
                    safety_breadth=args.safety_breadth,
                    safety_forced=args.safety_forced,
                    safety_timeout=args.safety_timeout,
                    safety_max_reconsider=args.safety_max_reconsider,
                    safety_confirmation_version=args.safety_confirmation_version,
                    safety_n_required=args.safety_n_required,
                    safety_reasoning_guideline=args.safety_reasoning_guideline,
                    with_safety_prompting=args.with_safety_prompting,
                    safety_prompting_version=args.safety_prompting_version,
                    run_name=args.run_name,
                    grid_search_combo_dir=None,
                    env_vars=env_vars,
                    config_file=args.config,
                )

                if returncode == 0:
                    results_summary.append((name, task_name, "success", ""))
                else:
                    results_summary.append((name, task_name, "failed", f"exit code {returncode}"))

        # 打印汇总
        print("\n" + "=" * 60)
        print("测评结果汇总")
        print("=" * 60)
        success_count = sum(1 for _, _, status, _ in results_summary if status == "success")
        for name, task_name, status, msg in results_summary:
            icon = "✅" if status == "success" else "❌" if status == "failed" else "⏭️"
            display_name = f"{name}:{task_name}" if task_name else name
            print(f"  {icon} {display_name}: {status} {msg}")

        print(f"\n通过: {success_count}/{len(results_summary)}")
        return 0 if success_count == len(results_summary) else 1

    # 设置所有环境
    if args.setup_all:
        print("设置所有 benchmark 环境...")
        VENVS_DIR.mkdir(parents=True, exist_ok=True)
        success = True
        for name, config in benchmarks.items():
            print(f"\n[{name}]")
            if not setup_benchmark_env(name, config, args.force,
                                       with_safety_lookahead=args.with_safety_lookahead,
                                       with_safety_confirmation=args.with_safety_confirmation,
                                       with_safety_prompting=args.with_safety_prompting):
                success = False
        return 0 if success else 1

    # 设置单个环境
    if args.setup:
        if args.setup not in benchmarks:
            print(f"错误: 未知的 benchmark '{args.setup}'")
            return 1
        print(f"设置 {args.setup} 环境...")
        VENVS_DIR.mkdir(parents=True, exist_ok=True)
        return 0 if setup_benchmark_env(args.setup, benchmarks[args.setup], args.force,
                                        with_safety_lookahead=args.with_safety_lookahead,
                                        with_safety_confirmation=args.with_safety_confirmation,
                                        with_safety_prompting=args.with_safety_prompting) else 1

    # 运行评估
    if not args.benchmark:
        parser.print_help()
        return 1

    if not args.model:
        print("错误: 必须指定 --model 参数")
        return 1

    benchmark_name, task_spec, config, task_config = resolve_task(args.benchmark, catalog)

    # 确保 .venvs 目录存在
    VENVS_DIR.mkdir(parents=True, exist_ok=True)

    return run_eval(
        benchmark_name=benchmark_name,
        task_spec=task_spec,
        config=config,
        model=args.model,
        inspect_args=inspect_args,
        dry_run=args.dry_run,
        task_config=task_config,
        with_safety_lookahead=args.with_safety_lookahead,
        with_safety_confirmation=args.with_safety_confirmation,
        world_model=args.world_model,
        safety_version=args.safety_version,
                    safety_n=args.safety_n,
                    safety_mask=args.safety_mask,
                    safety_mode=args.safety_mode,
                    safety_depth=args.safety_depth,
                    safety_breadth=args.safety_breadth,
                    safety_forced=args.safety_forced,
                    safety_timeout=args.safety_timeout,
        safety_max_reconsider=args.safety_max_reconsider,
        safety_confirmation_version=args.safety_confirmation_version,
        safety_n_required=args.safety_n_required,
        safety_reasoning_guideline=args.safety_reasoning_guideline,
        run_name=args.run_name,
        grid_search_combo_dir=None,
        env_vars=env_vars,
        config_file=args.config,
    )


if __name__ == "__main__":
    sys.exit(main())
