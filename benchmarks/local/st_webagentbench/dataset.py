"""Dataset loading for ST-WebAgentBench."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from inspect_ai.dataset import MemoryDataset, Sample

# Path to the ST-WebAgentBench dataset
# The benchmark source is relative to the monorepo root
_MONOREPO_ROOT = Path(__file__).resolve().parents[4]  # Go up 4 levels to reach monorepo root
_ST_BENCH_BASE_PATH = _MONOREPO_ROOT / "ST-WebAgentBench"
_ST_BENCH_DATA_FILE = _ST_BENCH_BASE_PATH / "stwebagentbench" / "test.raw.json"

# SuiteCRM tier task ID ranges
_SUITECRM_TIERS = {
    "easy": range(235, 255),  # 20 tasks
    "medium": range(255, 275),  # 20 tasks
    "hard": range(275, 295),  # 20 tasks
}

# Modality challenge task ID ranges
_MODALITY_TASKS = {
    "vision_advantage": range(295, 335),  # 40 tasks
    "dom_advantage": range(335, 375),  # 40 tasks
}

# Domain task ID ranges
_DOMAIN_RANGES = {
    "gitlab": range(0, 197),  # 197 tasks
    "shopping_admin": range(197, 205),  # 8 tasks
    "suitecrm": range(205, 375),  # 170 tasks
}

# Site name to domain mapping
_SITE_TO_DOMAIN = {
    "gitlab": "gitlab",
    "shopping": "shopping_admin",
}


def create_stable_id(task_id: int, prefix: str = "st_webagentbench") -> str:
    """Create a stable ID for a task.

    Args:
        task_id: The task identifier
        prefix: Prefix for the ID

    Returns:
        A stable string ID
    """
    return f"{prefix}_{task_id}"


def load_st_webagentbench_dataset(
    domain: Literal["all", "gitlab", "suitecrm", "shopping_admin"] = "all",
    tier: Literal["all", "easy", "medium", "hard"] = "all",
    task_id: int | None = None,
    shuffle: bool = False,
    limit: int | None = None,
) -> MemoryDataset:
    """Load the ST-WebAgentBench dataset.

    Args:
        domain: Filter by domain - "all", "gitlab", "suitecrm", or "shopping_admin"
        tier: Filter SuiteCRM by tier - "all", "easy", "medium", or "hard"
        task_id: Run a specific task ID (0-374). Overrides domain and tier filters.
        shuffle: Whether to shuffle samples
        limit: Maximum number of samples to return

    Returns:
        A MemoryDataset containing ST-WebAgentBench task samples

    Raises:
        FileNotFoundError: If the ST-WebAgentBench data file is not found
        ValueError: If task_id is out of valid range (0-374)
    """
    # Check if data file exists
    if not _ST_BENCH_DATA_FILE.exists():
        raise FileNotFoundError(
            f"ST-WebAgentBench data file not found at {_ST_BENCH_DATA_FILE}. "
            f"Please ensure ST-WebAgentBench is installed at {_ST_BENCH_BASE_PATH}"
        )

    # Load the raw task data
    with open(_ST_BENCH_DATA_FILE, "r") as f:
        raw_tasks = json.load(f)

    samples: list[Sample] = []

    for task_data in raw_tasks:
        task_id_val = task_data.get("task_id")

        # If specific task_id requested, skip non-matching
        if task_id is not None:
            if task_id_val != task_id:
                continue

        # Apply domain filter
        if domain != "all":
            task_domain = _get_task_domain(task_data)
            if task_domain != domain:
                continue

        # Apply tier filter (only for SuiteCRM tasks)
        if tier != "all":
            task_domain = _get_task_domain(task_data)
            if task_domain == "suitecrm":
                if task_id_val not in _SUITECRM_TIERS.get(tier, range(0)):
                    continue
            elif tier != "all":
                # Tier filter only applies to SuiteCRM
                continue

        # Build the sample
        intent = task_data.get("intent", "")
        intent_template = task_data.get("intent_template", "")
        instantiation_dict = task_data.get("instantiation_dict", {})

        # Format the goal with instantiation values
        goal = _format_goal(intent, intent_template, instantiation_dict)

        # Extract policies
        policies = task_data.get("policies", [])

        # Get evaluation configuration
        eval_config = task_data.get("eval", {})

        # Determine domain for this task
        task_domain = _get_task_domain(task_data)

        # Determine tier for SuiteCRM tasks
        task_tier = _get_task_tier(task_id_val)

        # Determine modality category
        modality = _get_modality_category(task_id_val)

        # Build metadata
        metadata = {
            "task_id": task_id_val,
            "intent_template_id": task_data.get("intent_template_id"),
            "sites": task_data.get("sites", []),
            "domain": task_domain,
            "tier": task_tier,
            "modality": modality,
            "start_url": task_data.get("start_url", ""),
            "require_login": task_data.get("require_login", False),
            "storage_state": task_data.get("storage_state", ""),
            "policies": policies,
            "eval_types": eval_config.get("eval_types", []),
            "policy_count": len(policies),
        }

        # Create sample
        sample = Sample(
            input=goal,
            target="",  # Scoring is based on environment state, not target matching
            id=create_stable_id(task_id_val, prefix="st_webagentbench"),
            metadata=metadata,
        )

        samples.append(sample)

    if shuffle:
        import random

        random.shuffle(samples)

    if limit is not None:
        samples = samples[:limit]

    return MemoryDataset(samples)


def _get_task_domain(task_data: dict) -> str:
    """Determine the domain for a task based on its sites field.

    Args:
        task_data: Raw task data dictionary

    Returns:
        Domain name: "gitlab", "shopping_admin", or "suitecrm"
    """
    sites = task_data.get("sites", [])
    if not sites:
        # Fallback to task_id ranges
        task_id = task_data.get("task_id", 0)
        if 0 <= task_id < 197:
            return "gitlab"
        elif 197 <= task_id < 205:
            return "shopping_admin"
        else:
            return "suitecrm"

    site = sites[0].lower() if sites else ""
    return _SITE_TO_DOMAIN.get(site, site)


def _get_task_tier(task_id: int) -> str:
    """Determine the tier for a SuiteCRM task.

    Args:
        task_id: The task ID

    Returns:
        Tier name: "easy", "medium", "hard", or None for non-SuiteCRM tasks
    """
    for tier_name, tier_range in _SUITECRM_TIERS.items():
        if task_id in tier_range:
            return tier_name
    return None


def _get_modality_category(task_id: int) -> str | None:
    """Determine the modality category for a task.

    Args:
        task_id: The task ID

    Returns:
        Modality category: "vision_advantage", "dom_advantage", or None
    """
    for category, category_range in _MODALITY_TASKS.items():
        if task_id in category_range:
            return category
    return None


def _format_goal(intent: str, intent_template: str, instantiation_dict: dict) -> str:
    """Format the task goal with instantiation values.

    Args:
        intent: The instantiated intent string
        intent_template: The intent template with placeholders
        instantiation_dict: Dictionary of placeholder values

    Returns:
        Formatted goal string
    """
    if intent:
        return intent
    if intent_template and instantiation_dict:
        # Simple template substitution
        goal = intent_template
        for key, value in instantiation_dict.items():
            goal = goal.replace(f"{{{{{key}}}}}", str(value))
        return goal
    return intent_template or ""
