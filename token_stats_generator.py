#!/usr/bin/env python3
"""
Token Stats Generator Module

Automatically generates token usage statistics from evaluation runs.
Supports:
- safety-lookahead runs (via safety_analysis.jsonl)
- safety-confirmation runs (via safety_confirmation_analysis.jsonl)
- baseline runs (via inspect_ai .eval files)

Usage:
    from token_stats_generator import generate_and_save_token_summary
    generate_and_save_token_summary(run_dir)

Or directly:
    python token_stats_generator.py <run_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def format_number(num: int | None) -> str:
    """Format a number with thousands separator."""
    if num is None:
        return "N/A"
    return f"{num:,}"


# ============================================================
# Pricing Data Structures
# ============================================================

class VendorPreset(Enum):
    """Vendor-specific pricing presets for weighted token cost calculation."""
    DEFAULT = "default"      # 5× output, 0.1× cache read, 1.25× cache write
    OPENAI = "openai"        # OpenAI-style pricing (6-8× output)
    ANTHROPIC = "anthropic"  # Anthropic-style pricing (5× output)
    GOOGLE = "google"        # Google Gemini pricing (~8× output)
    CHINESE = "chinese"      # Chinese vendor pricing (1.5× output)


@dataclass
class PricingWeights:
    """Token type weights for cost calculation."""
    input_weight: float = 1.0
    output_weight: float = 5.0      # 5× based on user choice
    cache_read_weight: float = 0.1  # 90% discount
    cache_write_weight: float = 1.25  # 25% premium


# Vendor presets (weights multipliers relative to baseline)
VENDOR_PRESETS: dict[VendorPreset, PricingWeights] = {
    VendorPreset.DEFAULT: PricingWeights(1.0, 5.0, 0.1, 1.25),
    VendorPreset.OPENAI: PricingWeights(1.0, 6.0, 0.1, 1.25),      # Newer OpenAI: 6-8×
    VendorPreset.ANTHROPIC: PricingWeights(1.0, 5.0, 0.1, 1.25),   # Anthropic: 5×
    VendorPreset.GOOGLE: PricingWeights(1.0, 8.0, 0.1, 1.25),      # Gemini: ~8×
    VendorPreset.CHINESE: PricingWeights(1.0, 1.5, 0.1, 1.25),     # DeepSeek/Qwen: 1.5-4×
}


@dataclass
class CostBreakdown:
    """Structured cost breakdown for weighted tokens."""
    weighted_input_tokens: int
    weighted_output_tokens: int
    weighted_cache_read_tokens: int
    weighted_cache_write_tokens: int
    total_weighted_tokens: int


# ============================================================
# Cost Calculation Functions
# ============================================================

def get_pricing_weights() -> PricingWeights:
    """Get pricing weights from environment or config."""
    # Check for custom weights via environment
    if "TOKEN_PRICING_OUTPUT_WEIGHT" in os.environ:
        return PricingWeights(
            input_weight=float(os.environ.get("TOKEN_PRICING_INPUT_WEIGHT", "1.0")),
            output_weight=float(os.environ["TOKEN_PRICING_OUTPUT_WEIGHT"]),
            cache_read_weight=float(os.environ.get("TOKEN_PRICING_CACHE_READ_WEIGHT", "0.1")),
            cache_write_weight=float(os.environ.get("TOKEN_PRICING_CACHE_WRITE_WEIGHT", "1.25")),
        )

    # Check for vendor preset
    preset_name = os.environ.get("TOKEN_PRICING_PRESET", "default").lower()
    for preset in VendorPreset:
        if preset.value == preset_name:
            return VENDOR_PRESETS[preset]

    return VENDOR_PRESETS[VendorPreset.DEFAULT]


def _input_includes_cache_read() -> bool:
    """
    Whether total_input_tokens already includes cache-read tokens.

    Default is True (OpenAI-style accounting). Set
    TOKEN_PRICING_INPUT_INCLUDES_CACHE_READ=false for providers where
    input_tokens are already net-of-cache.
    """
    raw = os.environ.get("TOKEN_PRICING_INPUT_INCLUDES_CACHE_READ", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def calculate_weighted_cost(token_stats: dict, weights: PricingWeights) -> CostBreakdown:
    """Calculate weighted token cost from raw token stats."""
    total_input_tokens = int(token_stats.get("total_input_tokens", 0) or 0)
    total_output_tokens = int(token_stats.get("total_output_tokens", 0) or 0)
    total_cache_read = int(token_stats.get("total_cache_read", 0) or 0)
    total_cache_write = int(token_stats.get("total_cache_write", 0) or 0)

    # Avoid double-charging cache-read tokens when input already includes them.
    if _input_includes_cache_read():
        billable_input_tokens = max(total_input_tokens - total_cache_read, 0)
    else:
        billable_input_tokens = total_input_tokens

    weighted_input = int(billable_input_tokens * weights.input_weight)
    weighted_output = int(total_output_tokens * weights.output_weight)
    weighted_cache_read = int(total_cache_read * weights.cache_read_weight)
    weighted_cache_write = int(total_cache_write * weights.cache_write_weight)

    total = weighted_input + weighted_output + weighted_cache_read + weighted_cache_write

    return CostBreakdown(
        weighted_input_tokens=weighted_input,
        weighted_output_tokens=weighted_output,
        weighted_cache_read_tokens=weighted_cache_read,
        weighted_cache_write_tokens=weighted_cache_write,
        total_weighted_tokens=total,
    )


def calculate_step_costs(token_stats: dict, weights: PricingWeights) -> dict[str, float]:
    """Calculate cost breakdown for each step."""
    costs: dict[str, float] = {}

    # Safety-lookahead steps
    if token_stats.get("step1_input_tokens", 0) or token_stats.get("step1_output_tokens", 0):
        costs["step1_cost"] = (
            token_stats.get("step1_input_tokens", 0) * weights.input_weight +
            token_stats.get("step1_output_tokens", 0) * weights.output_weight
        )

    if token_stats.get("step2_input_tokens", 0) or token_stats.get("step2_output_tokens", 0):
        costs["step2_cost"] = (
            token_stats.get("step2_input_tokens", 0) * weights.input_weight +
            token_stats.get("step2_output_tokens", 0) * weights.output_weight
        )

    if token_stats.get("step3_input_tokens", 0) or token_stats.get("step3_output_tokens", 0):
        costs["step3_cost"] = (
            token_stats.get("step3_input_tokens", 0) * weights.input_weight +
            token_stats.get("step3_output_tokens", 0) * weights.output_weight
        )

    if token_stats.get("step4_input_tokens", 0) or token_stats.get("step4_output_tokens", 0):
        costs["step4_cost"] = (
            token_stats.get("step4_input_tokens", 0) * weights.input_weight +
            token_stats.get("step4_output_tokens", 0) * weights.output_weight
        )

    if token_stats.get("context_analysis_input_tokens", 0) or token_stats.get("context_analysis_output_tokens", 0):
        costs["context_analysis_cost"] = (
            token_stats.get("context_analysis_input_tokens", 0) * weights.input_weight +
            token_stats.get("context_analysis_output_tokens", 0) * weights.output_weight
        )

    if token_stats.get("rewriting_input_tokens", 0) or token_stats.get("rewriting_output_tokens", 0):
        costs["rewriting_cost"] = (
            token_stats.get("rewriting_input_tokens", 0) * weights.input_weight +
            token_stats.get("rewriting_output_tokens", 0) * weights.output_weight
        )

    # Safety-confirmation steps
    if token_stats.get("initial_generation_input_tokens", 0) or token_stats.get("initial_generation_output_tokens", 0):
        costs["initial_generation_cost"] = (
            token_stats.get("initial_generation_input_tokens", 0) * weights.input_weight +
            token_stats.get("initial_generation_output_tokens", 0) * weights.output_weight
        )

    if token_stats.get("finalization_input_tokens", 0) or token_stats.get("finalization_output_tokens", 0):
        costs["finalization_cost"] = (
            token_stats.get("finalization_input_tokens", 0) * weights.input_weight +
            token_stats.get("finalization_output_tokens", 0) * weights.output_weight
        )

    if token_stats.get("reconsideration_input_tokens", 0) or token_stats.get("reconsideration_output_tokens", 0):
        costs["reconsideration_cost"] = (
            token_stats.get("reconsideration_input_tokens", 0) * weights.input_weight +
            token_stats.get("reconsideration_output_tokens", 0) * weights.output_weight
        )

    return costs


# ============================================================
# Original Functions
# ============================================================

def load_jsonl(file_path: Path) -> list[dict]:
    """Load JSONL file and return list of records."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}", file=sys.stderr)
    return records


def aggregate_token_stats_from_jsonl(records: list[dict]) -> dict:
    """
    Aggregate token statistics from safety_analysis.jsonl records.

    Supports both safety-lookahead format (4-step process) and
    safety-confirmation format (2-step with reconsiderations).

    Returns a dict with aggregated totals.
    """
    totals = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "total_cache_read": 0,
        "total_cache_write": 0,
        "total_reasoning": 0,
        "total_image_tokens_input": 0,
        "total_image_tokens_output": 0,
        "total_audio_tokens_input": 0,
        "total_audio_tokens_output": 0,
        # Safety-lookahead step breakdown
        "step1_count": 0,
        "step1_input_tokens": 0,
        "step1_output_tokens": 0,
        "step2_count": 0,
        "step2_input_tokens": 0,
        "step2_output_tokens": 0,
        "step3_count": 0,
        "step3_input_tokens": 0,
        "step3_output_tokens": 0,
        "step4_count": 0,
        "step4_input_tokens": 0,
        "step4_output_tokens": 0,
        "context_analysis_count": 0,
        "context_analysis_input_tokens": 0,
        "context_analysis_output_tokens": 0,
        "rewriting_count": 0,
        "rewriting_input_tokens": 0,
        "rewriting_output_tokens": 0,
        # Safety-confirmation V2 step breakdown
        "initial_generation_count": 0,
        "initial_generation_input_tokens": 0,
        "initial_generation_output_tokens": 0,
        "initial_generation_reasoning": 0,
        "finalization_count": 0,
        "finalization_input_tokens": 0,
        "finalization_output_tokens": 0,
        "finalization_reasoning": 0,
        "reconsideration_count": 0,
        "reconsideration_input_tokens": 0,
        "reconsideration_output_tokens": 0,
        "reconsideration_reasoning": 0,
        "records_with_token_stats": 0,
        "total_records": len(records),
        "source": "safety_analysis.jsonl",
    }

    for record in records:
        token_stats = record.get("token_stats")
        if not token_stats:
            continue

        totals["records_with_token_stats"] += 1

        # Detect format: safety-confirmation has top-level totals (no "totals" sub-dict)
        is_safety_confirmation = "initial_generation" in token_stats

        if is_safety_confirmation:
            # Safety-confirmation V2 format
            # Top-level totals
            totals["total_input_tokens"] += token_stats.get("total_input_tokens", 0)
            totals["total_output_tokens"] += token_stats.get("total_output_tokens", 0)
            totals["total_tokens"] += token_stats.get("total_tokens", 0)
            totals["total_cache_read"] += token_stats.get("total_cache_read", 0)
            totals["total_cache_write"] += token_stats.get("total_cache_write", 0)
            totals["total_reasoning"] += token_stats.get("total_reasoning", 0)
            totals["total_image_tokens_input"] += token_stats.get("total_image_tokens_input", 0)
            totals["total_image_tokens_output"] += token_stats.get("total_image_tokens_output", 0)
            totals["total_audio_tokens_input"] += token_stats.get("total_audio_tokens_input", 0)
            totals["total_audio_tokens_output"] += token_stats.get("total_audio_tokens_output", 0)

            # Initial generation (Step 1 - safety confirmation)
            init_gen = token_stats.get("initial_generation", {})
            if init_gen:
                totals["initial_generation_count"] += 1
                totals["initial_generation_input_tokens"] += init_gen.get("input_tokens", 0)
                totals["initial_generation_output_tokens"] += init_gen.get("output_tokens", 0)
                totals["initial_generation_reasoning"] += init_gen.get("reasoning") or 0

            # Finalization (Step 2 - after safety confirmed)
            final = token_stats.get("finalization", {})
            if final:
                totals["finalization_count"] += 1
                totals["finalization_input_tokens"] += final.get("input_tokens", 0)
                totals["finalization_output_tokens"] += final.get("output_tokens", 0)
                totals["finalization_reasoning"] += final.get("reasoning") or 0

            # Reconsiderations (retry loop)
            for recon in token_stats.get("reconsiderations", []):
                totals["reconsideration_count"] += 1
                totals["reconsideration_input_tokens"] += recon.get("input_tokens", 0)
                totals["reconsideration_output_tokens"] += recon.get("output_tokens", 0)
                totals["reconsideration_reasoning"] += recon.get("reasoning") or 0

        else:
            # Safety-lookahead format (4-step process)
            # Aggregate totals from "totals" sub-dict
            totals_data = token_stats.get("totals", {})
            totals["total_input_tokens"] += totals_data.get("input_tokens", 0)
            totals["total_output_tokens"] += totals_data.get("output_tokens", 0)
            totals["total_tokens"] += totals_data.get("total_tokens", 0)
            totals["total_cache_read"] += totals_data.get("cache_read_tokens", 0)
            totals["total_cache_write"] += totals_data.get("cache_write_tokens", 0)
            totals["total_reasoning"] += totals_data.get("reasoning_tokens", 0)
            totals["total_image_tokens_input"] += totals_data.get("image_tokens_input", 0)
            totals["total_image_tokens_output"] += totals_data.get("image_tokens_output", 0)
            totals["total_audio_tokens_input"] += totals_data.get("audio_tokens_input", 0)
            totals["total_audio_tokens_output"] += totals_data.get("audio_tokens_output", 0)

            # Aggregate Step 1
            step1 = token_stats.get("step1_detection", {})
            if step1:
                totals["step1_count"] += 1
                totals["step1_input_tokens"] += step1.get("input_tokens", 0)
                totals["step1_output_tokens"] += step1.get("output_tokens", 0)

            # Aggregate Step 2 (may have multiple calls per record)
            for step2 in token_stats.get("step2_candidates", []):
                totals["step2_count"] += 1
                totals["step2_input_tokens"] += step2.get("input_tokens", 0)
                totals["step2_output_tokens"] += step2.get("output_tokens", 0)

            # Aggregate Step 3 (may have multiple calls per record)
            for step3 in token_stats.get("step3_world_model", []):
                totals["step3_count"] += 1
                totals["step3_input_tokens"] += step3.get("input_tokens", 0)
                totals["step3_output_tokens"] += step3.get("output_tokens", 0)

            # Aggregate Step 4
            step4 = token_stats.get("step4_final", {})
            if step4:
                totals["step4_count"] += 1
                totals["step4_input_tokens"] += step4.get("input_tokens", 0)
                totals["step4_output_tokens"] += step4.get("output_tokens", 0)

            # Aggregate context analysis
            context_analysis = token_stats.get("context_analysis", {})
            if context_analysis:
                totals["context_analysis_count"] += 1
                totals["context_analysis_input_tokens"] += context_analysis.get("input_tokens", 0)
                totals["context_analysis_output_tokens"] += context_analysis.get("output_tokens", 0)

            # Aggregate rewriting
            rewriting = token_stats.get("rewriting", {})
            if rewriting:
                totals["rewriting_count"] += 1
                totals["rewriting_input_tokens"] += rewriting.get("input_tokens", 0)
                totals["rewriting_output_tokens"] += rewriting.get("output_tokens", 0)

    return totals


def generate_token_stats_from_jsonl(jsonl_file: Path) -> dict | None:
    """
    Extract token stats from safety_analysis.jsonl.

    Returns None if the file doesn't exist or is empty.
    """
    if not jsonl_file.exists():
        return None

    records = load_jsonl(jsonl_file)
    if not records:
        return None

    return aggregate_token_stats_from_jsonl(records)


def generate_token_stats_from_eval(eval_file: Path) -> dict | None:
    """
    Extract token stats from inspect_ai .eval file.

    Only counts tokens from the primary tested model, excluding judge/grader models.

    Returns None if the file doesn't exist or has no token usage data.
    """
    if not eval_file.exists():
        return None

    try:
        from inspect_ai.log._file import read_eval_log
    except ImportError:
        print("Warning: inspect_ai not available for reading .eval file", file=sys.stderr)
        return None

    try:
        log = read_eval_log(eval_file, header_only=True)
    except Exception as e:
        print(f"Warning: Failed to read .eval file: {e}", file=sys.stderr)
        return None

    # Get the primary tested model (exclude judge/grader models)
    primary_model = log.eval.model
    model_usage = log.stats.model_usage  # dict[str, ModelUsage]

    if not model_usage or primary_model not in model_usage:
        return None

    # Only aggregate tokens from the primary tested model
    usage = model_usage[primary_model]
    totals = {
        "total_input_tokens": usage.input_tokens,
        "total_output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "total_cache_read": usage.input_tokens_cache_read or 0,
        "total_cache_write": usage.input_tokens_cache_write or 0,
        "total_reasoning": usage.reasoning_tokens or 0,
        "total_image_tokens_input": 0,
        "total_image_tokens_output": 0,
        "total_audio_tokens_input": 0,
        "total_audio_tokens_output": 0,
        "total_records": len(log.samples) if log.samples else 0,
        "source": f".eval file ({eval_file.name})",
        "primary_model": primary_model,
    }

    # Optionally list excluded judge models for transparency
    if log.eval.model_roles:
        excluded_models = list(log.eval.model_roles.values())
        totals["excluded_models"] = excluded_models

    # Add placeholder values for step breakdown (not available in .eval)
    for key in ["step1_count", "step1_input_tokens", "step1_output_tokens",
                "step2_count", "step2_input_tokens", "step2_output_tokens",
                "step3_count", "step3_input_tokens", "step3_output_tokens",
                "step4_count", "step4_input_tokens", "step4_output_tokens",
                "context_analysis_count", "context_analysis_input_tokens", "context_analysis_output_tokens",
                "rewriting_count", "rewriting_input_tokens", "rewriting_output_tokens",
                "records_with_token_stats"]:
        totals[key] = 0

    return totals


def generate_token_summary(run_dir: Path) -> dict | None:
    """
    Generate token stats from run directory (auto-detects source).

    Priority:
    1. safety_analysis.jsonl (safety-lookahead runs)
    2. safety_confirmation_analysis.jsonl (safety-confirmation runs)
    3. .eval file (baseline runs)

    Returns None if no token stats can be extracted.
    """
    # Check for safety-lookahead output
    safety_file = run_dir / "safety_analysis.jsonl"
    if safety_file.exists():
        stats = generate_token_stats_from_jsonl(safety_file)
        if stats:
            stats["run_dir"] = str(run_dir)
            return stats

    # Check for safety-confirmation output
    safety_confirmation_file = run_dir / "safety_confirmation_analysis.jsonl"
    if safety_confirmation_file.exists():
        stats = generate_token_stats_from_jsonl(safety_confirmation_file)
        if stats:
            stats["run_dir"] = str(run_dir)
            # Update source to indicate safety-confirmation
            stats["source"] = "safety_confirmation_analysis.jsonl"
            return stats

    # Fallback to .eval file
    # Grid search combos: .eval files directly in run_dir (no eval/ subdirectory)
    # Regular runs: .eval files in eval/ subdirectory
    eval_files = list(run_dir.glob("*.eval"))
    if not eval_files:
        # Regular run structure: check eval/ subdirectory
        eval_subdir = run_dir / "eval"
        if eval_subdir.exists():
            eval_files = list(eval_subdir.glob("*.eval"))
    if eval_files:
        # Use the first .eval file found
        stats = generate_token_stats_from_eval(eval_files[0])
        if stats:
            stats["run_dir"] = str(run_dir)
            return stats

    return None


def save_token_summary_txt(token_stats: dict, output_file: Path) -> None:
    """Save human-readable token summary to text file."""
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Token Usage Summary\n")
        f.write("=" * 60 + "\n\n")

        source = token_stats.get("source", "Unknown")
        f.write(f"Source: {source}\n")

        # Primary model info
        if "primary_model" in token_stats:
            f.write(f"Primary Model: {token_stats['primary_model']}\n")
            if "excluded_models" in token_stats and token_stats["excluded_models"]:
                excluded = ", ".join(token_stats["excluded_models"])
                f.write(f"Excluded judge models: {excluded}\n")
        f.write("\n")

        # Overall totals
        f.write("Total Tokens:\n")
        f.write(f"  Input Tokens:    {format_number(token_stats['total_input_tokens'])}\n")
        f.write(f"  Output Tokens:   {format_number(token_stats['total_output_tokens'])}\n")
        f.write(f"  Cache Read:      {format_number(token_stats['total_cache_read'])}\n")
        f.write(f"  Cache Write:     {format_number(token_stats['total_cache_write'])}\n")
        f.write(f"  Reasoning:       {format_number(token_stats['total_reasoning'])}\n")

        # Additional token types (if present)
        if token_stats.get("total_image_tokens_input", 0) > 0 or token_stats.get("total_image_tokens_output", 0) > 0:
            f.write(f"  Image Input:     {format_number(token_stats['total_image_tokens_input'])}\n")
            f.write(f"  Image Output:    {format_number(token_stats['total_image_tokens_output'])}\n")
        if token_stats.get("total_audio_tokens_input", 0) > 0 or token_stats.get("total_audio_tokens_output", 0) > 0:
            f.write(f"  Audio Input:     {format_number(token_stats['total_audio_tokens_input'])}\n")
            f.write(f"  Audio Output:    {format_number(token_stats['total_audio_tokens_output'])}\n")

        f.write(f"  ---                -------\n")
        f.write(f"  Total Tokens:    {format_number(token_stats['total_tokens'])}\n")
        f.write("\n")

        # Weighted cost section (input-equivalent units)
        show_costs = os.environ.get("TOKEN_STATS_SHOW_COSTS", "true").lower() != "false"
        if show_costs:
            weights = get_pricing_weights()
            costs = calculate_weighted_cost(token_stats, weights)

            f.write("Weighted Token Cost (input-equivalent units):\n")
            f.write(f"  Input Tokens (non-cached): {format_number(costs.weighted_input_tokens)}\n")
            f.write(f"  Output Tokens:    {format_number(costs.weighted_output_tokens)} (×{weights.output_weight})\n")
            f.write(f"  Cache Read:       {format_number(costs.weighted_cache_read_tokens)} (×{weights.cache_read_weight})\n")
            f.write(f"  Cache Write:      {format_number(costs.weighted_cache_write_tokens)} (×{weights.cache_write_weight})\n")
            f.write(f"  ---                -------\n")
            f.write(f"  Total Weighted:   {format_number(costs.total_weighted_tokens)}\n")
            f.write(f"  (Pricing preset: {os.environ.get('TOKEN_PRICING_PRESET', 'default')}, input_includes_cache_read={_input_includes_cache_read()})\n")
            f.write("\n")

        # Records info
        if "records_with_token_stats" in token_stats:
            f.write(f"Records: {token_stats['records_with_token_stats']}/{token_stats['total_records']} have token stats\n")
        else:
            f.write(f"Total samples: {token_stats['total_records']}\n")
        f.write("\n")

        # Step-by-step breakdown (only if available)
        has_safety_lookahead_steps = any(
            token_stats.get(f"step{i}_count", 0) > 0
            for i in range(1, 5)
        ) or token_stats.get("context_analysis_count", 0) > 0 or token_stats.get("rewriting_count", 0) > 0

        has_safety_confirmation_steps = (
            token_stats.get("initial_generation_count", 0) > 0 or
            token_stats.get("finalization_count", 0) > 0 or
            token_stats.get("reconsideration_count", 0) > 0
        )

        if has_safety_lookahead_steps:
            f.write("Breakdown by Step (Safety-Lookahead):\n")

            if token_stats.get("step1_count", 0) > 0:
                f.write("  Step 1 (Detection):\n")
                f.write(f"    {format_number(token_stats['step1_input_tokens'])} in / {format_number(token_stats['step1_output_tokens'])} out\n")
                f.write(f"    ({token_stats['step1_count']} calls)\n")

            if token_stats.get("step2_count", 0) > 0:
                f.write("  Step 2 (Candidates):\n")
                f.write(f"    {format_number(token_stats['step2_input_tokens'])} in / {format_number(token_stats['step2_output_tokens'])} out\n")
                f.write(f"    ({token_stats['step2_count']} calls)\n")

            if token_stats.get("step3_count", 0) > 0:
                f.write("  Step 3 (World Model):\n")
                f.write(f"    {format_number(token_stats['step3_input_tokens'])} in / {format_number(token_stats['step3_output_tokens'])} out\n")
                f.write(f"    ({token_stats['step3_count']} calls)\n")

            if token_stats.get("step4_count", 0) > 0:
                f.write("  Step 4 (Final):\n")
                f.write(f"    {format_number(token_stats['step4_input_tokens'])} in / {format_number(token_stats['step4_output_tokens'])} out\n")
                f.write(f"    ({token_stats['step4_count']} calls)\n")

            if token_stats.get("context_analysis_count", 0) > 0:
                f.write("  Context Analysis:\n")
                f.write(f"    {format_number(token_stats['context_analysis_input_tokens'])} in / {format_number(token_stats['context_analysis_output_tokens'])} out\n")
                f.write(f"    ({token_stats['context_analysis_count']} calls)\n")

            if token_stats.get("rewriting_count", 0) > 0:
                f.write("  Rewriting (masking):\n")
                f.write(f"    {format_number(token_stats['rewriting_input_tokens'])} in / {format_number(token_stats['rewriting_output_tokens'])} out\n")
                f.write(f"    ({token_stats['rewriting_count']} calls)\n")

            # Step cost breakdown
            if show_costs:
                step_costs = calculate_step_costs(token_stats, weights)
                f.write("  Step Costs (weighted tokens):\n")
                if "step1_cost" in step_costs:
                    f.write(f"    Step 1: {format_number(int(step_costs['step1_cost']))}\n")
                if "step2_cost" in step_costs:
                    f.write(f"    Step 2: {format_number(int(step_costs['step2_cost']))}\n")
                if "step3_cost" in step_costs:
                    f.write(f"    Step 3: {format_number(int(step_costs['step3_cost']))}\n")
                if "step4_cost" in step_costs:
                    f.write(f"    Step 4: {format_number(int(step_costs['step4_cost']))}\n")
                if "context_analysis_cost" in step_costs:
                    f.write(f"    Context Analysis: {format_number(int(step_costs['context_analysis_cost']))}\n")
                if "rewriting_cost" in step_costs:
                    f.write(f"    Rewriting: {format_number(int(step_costs['rewriting_cost']))}\n")

            f.write("\n")

        if has_safety_confirmation_steps:
            f.write("Breakdown by Step (Safety-Confirmation V2):\n")

            if token_stats.get("initial_generation_count", 0) > 0:
                f.write("  Initial Generation (Safety Confirmation):\n")
                f.write(f"    {format_number(token_stats['initial_generation_input_tokens'])} in / {format_number(token_stats['initial_generation_output_tokens'])} out\n")
                if token_stats.get("initial_generation_reasoning", 0) > 0:
                    f.write(f"    Reasoning: {format_number(token_stats['initial_generation_reasoning'])}\n")
                f.write(f"    ({token_stats['initial_generation_count']} calls)\n")

            if token_stats.get("finalization_count", 0) > 0:
                f.write("  Finalization (After Safety Confirmed):\n")
                f.write(f"    {format_number(token_stats['finalization_input_tokens'])} in / {format_number(token_stats['finalization_output_tokens'])} out\n")
                if token_stats.get("finalization_reasoning", 0) > 0:
                    f.write(f"    Reasoning: {format_number(token_stats['finalization_reasoning'])}\n")
                f.write(f"    ({token_stats['finalization_count']} calls)\n")

            if token_stats.get("reconsideration_count", 0) > 0:
                f.write("  Reconsiderations (Retry Loop):\n")
                f.write(f"    {format_number(token_stats['reconsideration_input_tokens'])} in / {format_number(token_stats['reconsideration_output_tokens'])} out\n")
                if token_stats.get("reconsideration_reasoning", 0) > 0:
                    f.write(f"    Reasoning: {format_number(token_stats['reconsideration_reasoning'])}\n")
                f.write(f"    ({token_stats['reconsideration_count']} calls)\n")

            # Step cost breakdown
            if show_costs:
                step_costs = calculate_step_costs(token_stats, weights)
                f.write("  Step Costs (weighted tokens):\n")
                if "initial_generation_cost" in step_costs:
                    f.write(f"    Initial Generation: {format_number(int(step_costs['initial_generation_cost']))}\n")
                if "finalization_cost" in step_costs:
                    f.write(f"    Finalization: {format_number(int(step_costs['finalization_cost']))}\n")
                if "reconsideration_cost" in step_costs:
                    f.write(f"    Reconsiderations: {format_number(int(step_costs['reconsideration_cost']))}\n")

            f.write("\n")


def save_token_summary_csv(token_stats: dict, output_file: Path) -> None:
    """Save token summary to CSV file."""
    # Define the fields we want in the CSV
    fields = [
        'total_input_tokens', 'total_output_tokens', 'total_tokens',
        'total_cache_read', 'total_cache_write', 'total_reasoning',
        'total_image_tokens_input', 'total_image_tokens_output',
        'total_audio_tokens_input', 'total_audio_tokens_output',
        'total_records', 'source',
        # Weighted token cost fields
        'weighted_input_tokens', 'weighted_output_tokens',
        'weighted_cache_read_tokens', 'weighted_cache_write_tokens',
        'total_weighted_tokens', 'pricing_preset',
        # Safety-lookahead fields
        'step1_count', 'step1_input_tokens', 'step1_output_tokens',
        'step2_count', 'step2_input_tokens', 'step2_output_tokens',
        'step3_count', 'step3_input_tokens', 'step3_output_tokens',
        'step4_count', 'step4_input_tokens', 'step4_output_tokens',
        'context_analysis_count', 'context_analysis_input_tokens', 'context_analysis_output_tokens',
        'rewriting_count', 'rewriting_input_tokens', 'rewriting_output_tokens',
        # Safety-confirmation V2 fields
        'initial_generation_count', 'initial_generation_input_tokens', 'initial_generation_output_tokens', 'initial_generation_reasoning',
        'finalization_count', 'finalization_input_tokens', 'finalization_output_tokens', 'finalization_reasoning',
        'reconsideration_count', 'reconsideration_input_tokens', 'reconsideration_output_tokens', 'reconsideration_reasoning',
        'records_with_token_stats',
    ]

    # Calculate weighted costs if enabled
    show_costs = os.environ.get("TOKEN_STATS_SHOW_COSTS", "true").lower() != "false"
    if show_costs:
        weights = get_pricing_weights()
        costs = calculate_weighted_cost(token_stats, weights)
        token_stats['weighted_input_tokens'] = costs.weighted_input_tokens
        token_stats['weighted_output_tokens'] = costs.weighted_output_tokens
        token_stats['weighted_cache_read_tokens'] = costs.weighted_cache_read_tokens
        token_stats['weighted_cache_write_tokens'] = costs.weighted_cache_write_tokens
        token_stats['total_weighted_tokens'] = costs.total_weighted_tokens
        token_stats['pricing_preset'] = os.environ.get('TOKEN_PRICING_PRESET', 'default')

    # Filter to only fields present in token_stats
    fieldnames = [f for f in fields if f in token_stats]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: token_stats[k] for k in fieldnames})


def generate_and_save_token_summary(run_dir: Path) -> bool:
    """
    Generate and save token stats for a run directory.

    Returns True if successful, False otherwise.
    """
    stats = generate_token_summary(run_dir)
    if not stats:
        print(f"Warning: No token stats found in {run_dir}", file=sys.stderr)
        return False

    txt_file = run_dir / "token_stats.txt"
    csv_file = run_dir / "token_stats.csv"

    save_token_summary_txt(stats, txt_file)
    save_token_summary_csv(stats, csv_file)

    print(f"Token stats saved:")
    print(f"  {txt_file}")
    print(f"  {csv_file}")

    return True


def print_summary(totals: dict) -> None:
    """Print a summary of token statistics to stdout."""
    print("=" * 60)
    print("Token Usage Summary")
    print("=" * 60)
    print()

    source = totals.get("source", "Unknown")
    print(f"Source: {source}")

    # Primary model info
    if "primary_model" in totals:
        print(f"Primary Model: {totals['primary_model']}")
        if "excluded_models" in totals and totals["excluded_models"]:
            excluded = ", ".join(totals["excluded_models"])
            print(f"Excluded judge models: {excluded}")
    print()

    # Overall totals
    print("Total Tokens:")
    print(f"  Input Tokens:    {format_number(totals['total_input_tokens'])}")
    print(f"  Output Tokens:   {format_number(totals['total_output_tokens'])}")
    print(f"  Cache Read:      {format_number(totals['total_cache_read'])}")
    print(f"  Cache Write:     {format_number(totals['total_cache_write'])}")
    print(f"  Reasoning:       {format_number(totals['total_reasoning'])}")

    # Additional token types (if present)
    if totals.get("total_image_tokens_input", 0) > 0 or totals.get("total_image_tokens_output", 0) > 0:
        print(f"  Image Input:     {format_number(totals['total_image_tokens_input'])}")
        print(f"  Image Output:    {format_number(totals['total_image_tokens_output'])}")
    if totals.get("total_audio_tokens_input", 0) > 0 or totals.get("total_audio_tokens_output", 0) > 0:
        print(f"  Audio Input:     {format_number(totals['total_audio_tokens_input'])}")
        print(f"  Audio Output:    {format_number(totals['total_audio_tokens_output'])}")

    print(f"  ---                -------")
    print(f"  Total Tokens:    {format_number(totals['total_tokens'])}")
    print()

    # Weighted cost section (input-equivalent units)
    show_costs = os.environ.get("TOKEN_STATS_SHOW_COSTS", "true").lower() != "false"
    if show_costs:
        weights = get_pricing_weights()
        costs = calculate_weighted_cost(totals, weights)

        print("Weighted Token Cost (input-equivalent units):")
        print(f"  Input Tokens (non-cached): {format_number(costs.weighted_input_tokens)}")
        print(f"  Output Tokens:    {format_number(costs.weighted_output_tokens)} (×{weights.output_weight})")
        print(f"  Cache Read:       {format_number(costs.weighted_cache_read_tokens)} (×{weights.cache_read_weight})")
        print(f"  Cache Write:      {format_number(costs.weighted_cache_write_tokens)} (×{weights.cache_write_weight})")
        print(f"  ---                -------")
        print(f"  Total Weighted:   {format_number(costs.total_weighted_tokens)}")
        print(f"  (Pricing preset: {os.environ.get('TOKEN_PRICING_PRESET', 'default')}, input_includes_cache_read={_input_includes_cache_read()})")
        print()

    # Records info
    if "records_with_token_stats" in totals:
        print(f"Records: {totals['records_with_token_stats']}/{totals['total_records']} have token stats")
    else:
        print(f"Total samples: {totals['total_records']}")
    print()

    # Step-by-step breakdown (only if available)
    has_safety_lookahead_steps = any(
        totals.get(f"step{i}_count", 0) > 0
        for i in range(1, 5)
    ) or totals.get("context_analysis_count", 0) > 0 or totals.get("rewriting_count", 0) > 0

    has_safety_confirmation_steps = (
        totals.get("initial_generation_count", 0) > 0 or
        totals.get("finalization_count", 0) > 0 or
        totals.get("reconsideration_count", 0) > 0
    )

    if has_safety_lookahead_steps:
        print("Breakdown by Step (Safety-Lookahead):")

        if totals.get("step1_count", 0) > 0:
            print(f"  Step 1 (Detection):")
            print(f"    {format_number(totals['step1_input_tokens'])} in / {format_number(totals['step1_output_tokens'])} out")
            print(f"    ({totals['step1_count']} calls)")

        if totals.get("step2_count", 0) > 0:
            print(f"  Step 2 (Candidates):")
            print(f"    {format_number(totals['step2_input_tokens'])} in / {format_number(totals['step2_output_tokens'])} out")
            print(f"    ({totals['step2_count']} calls)")

        if totals.get("step3_count", 0) > 0:
            print(f"  Step 3 (World Model):")
            print(f"    {format_number(totals['step3_input_tokens'])} in / {format_number(totals['step3_output_tokens'])} out")
            print(f"    ({totals['step3_count']} calls)")

        if totals.get("step4_count", 0) > 0:
            print(f"  Step 4 (Final):")
            print(f"    {format_number(totals['step4_input_tokens'])} in / {format_number(totals['step4_output_tokens'])} out")
            print(f"    ({totals['step4_count']} calls)")

        if totals.get("context_analysis_count", 0) > 0:
            print(f"  Context Analysis:")
            print(f"    {format_number(totals['context_analysis_input_tokens'])} in / {format_number(totals['context_analysis_output_tokens'])} out")
            print(f"    ({totals['context_analysis_count']} calls)")

        if totals.get("rewriting_count", 0) > 0:
            print(f"  Rewriting (masking):")
            print(f"    {format_number(totals['rewriting_input_tokens'])} in / {format_number(totals['rewriting_output_tokens'])} out")
            print(f"    ({totals['rewriting_count']} calls)")

        # Step cost breakdown
        if show_costs:
            step_costs = calculate_step_costs(totals, weights)
            print("  Step Costs (weighted tokens):")
            if "step1_cost" in step_costs:
                print(f"    Step 1: {format_number(int(step_costs['step1_cost']))}")
            if "step2_cost" in step_costs:
                print(f"    Step 2: {format_number(int(step_costs['step2_cost']))}")
            if "step3_cost" in step_costs:
                print(f"    Step 3: {format_number(int(step_costs['step3_cost']))}")
            if "step4_cost" in step_costs:
                print(f"    Step 4: {format_number(int(step_costs['step4_cost']))}")
            if "context_analysis_cost" in step_costs:
                print(f"    Context Analysis: {format_number(int(step_costs['context_analysis_cost']))}")
            if "rewriting_cost" in step_costs:
                print(f"    Rewriting: {format_number(int(step_costs['rewriting_cost']))}")

        print()

    if has_safety_confirmation_steps:
        print("Breakdown by Step (Safety-Confirmation V2):")

        if totals.get("initial_generation_count", 0) > 0:
            print(f"  Initial Generation (Safety Confirmation):")
            print(f"    {format_number(totals['initial_generation_input_tokens'])} in / {format_number(totals['initial_generation_output_tokens'])} out")
            if totals.get("initial_generation_reasoning", 0) > 0:
                print(f"    Reasoning: {format_number(totals['initial_generation_reasoning'])}")
            print(f"    ({totals['initial_generation_count']} calls)")

        if totals.get("finalization_count", 0) > 0:
            print(f"  Finalization (After Safety Confirmed):")
            print(f"    {format_number(totals['finalization_input_tokens'])} in / {format_number(totals['finalization_output_tokens'])} out")
            if totals.get("finalization_reasoning", 0) > 0:
                print(f"    Reasoning: {format_number(totals['finalization_reasoning'])}")
            print(f"    ({totals['finalization_count']} calls)")

        if totals.get("reconsideration_count", 0) > 0:
            print(f"  Reconsiderations (Retry Loop):")
            print(f"    {format_number(totals['reconsideration_input_tokens'])} in / {format_number(totals['reconsideration_output_tokens'])} out")
            if totals.get("reconsideration_reasoning", 0) > 0:
                print(f"    Reasoning: {format_number(totals['reconsideration_reasoning'])}")
            print(f"    ({totals['reconsideration_count']} calls)")

        # Step cost breakdown
        if show_costs:
            step_costs = calculate_step_costs(totals, weights)
            print("  Step Costs (weighted tokens):")
            if "initial_generation_cost" in step_costs:
                print(f"    Initial Generation: {format_number(int(step_costs['initial_generation_cost']))}")
            if "finalization_cost" in step_costs:
                print(f"    Finalization: {format_number(int(step_costs['finalization_cost']))}")
            if "reconsideration_cost" in step_costs:
                print(f"    Reconsiderations: {format_number(int(step_costs['reconsideration_cost']))}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate token usage statistics from evaluation runs",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to run directory, safety_analysis.jsonl, or safety_confirmation_analysis.jsonl file",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output in CSV format to stdout (in addition to saving files)",
    )
    parser.add_argument(
        "--pricing-preset",
        choices=[p.value for p in VendorPreset],
        default="default",
        help="Pricing preset for cost calculation (default: default)",
    )
    parser.add_argument(
        "--no-costs",
        action="store_true",
        help="Disable weighted cost reporting",
    )

    args = parser.parse_args()
    path = Path(args.path)

    # Set pricing configuration via environment
    if not args.no_costs:
        os.environ["TOKEN_PRICING_PRESET"] = args.pricing_preset
        os.environ["TOKEN_STATS_SHOW_COSTS"] = "true"
    else:
        os.environ["TOKEN_STATS_SHOW_COSTS"] = "false"

    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1

    # For directories, save files by default, then print summary
    if path.is_dir():
        # Save token_stats.txt and token_stats.csv
        if not generate_and_save_token_summary(path):
            return 1

        # Re-load stats for printing
        stats = generate_token_summary(path)
    elif path.suffix == ".jsonl":
        stats = generate_token_stats_from_jsonl(path)
        if stats:
            stats["source"] = f"jsonl file ({path.name})"
    elif path.suffix == ".eval":
        stats = generate_token_stats_from_eval(path)
        if stats:
            stats["source"] = f".eval file ({path.name})"
    else:
        print(f"Error: Unsupported file type: {path.suffix}", file=sys.stderr)
        print("Expected: directory, .jsonl, or .eval file", file=sys.stderr)
        return 1

    if not stats:
        print("Error: No token stats found", file=sys.stderr)
        return 1

    if args.csv:
        import io
        output = io.StringIO()
        fieldnames = [
            'total_input_tokens', 'total_output_tokens', 'total_tokens',
            'total_cache_read', 'total_cache_write', 'total_reasoning',
            'total_records', 'source',
        ]
        # Filter to only fields present in stats
        fieldnames = [f for f in fieldnames if f in stats]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: stats[k] for k in fieldnames})
        print(output.getvalue())
    else:
        print_summary(stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
