"""
Dataset loading for Toolathlon benchmark.

Loads all 109 tasks from the Toolathlon finalpool with support for
filtering by category/service.

Task categories based on MCP servers and tool types:
- canvas: Canvas LMS tasks (grading, quizzes, notifications)
- notion: Notion tasks (databases, pages, HR)
- filesystem: File system operations
- k8s: Kubernetes operations
- git: Git repository operations
- woocommerce: WooCommerce e-commerce tasks
- email: Email-related tasks
- web: Web search and browsing tasks
- data_analysis: Data processing and analysis
- research: Academic/research tasks
- financial: Financial analysis tasks
- misc: Miscellaneous tasks
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

from inspect_ai.dataset import MemoryDataset, Sample

# Toolathlon project root (relative to this file)
_TOOLATHLON_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "Toolathlon"
_TASKS_DIR = _TOOLATHLON_ROOT / "tasks" / "finalpool"


def _create_stable_id(task_name: str, prefix: str = "toolathlon") -> str:
    """Create a stable ID for a task.

    Simple version that doesn't require inspect_evals dependency.
    """
    # Remove special characters and replace with hyphens
    clean_name = task_name.replace("_", "-").replace("/", "-")
    return f"{prefix}-{clean_name}"

# Task categories based on naming patterns and required tools
_TASK_CATEGORIES = {
    "canvas": [
        "canvas-arrange-exam",
        "canvas-art-manager",
        "canvas-art-quiz",
        "canvas-do-quiz",
        "canvas-homework-grader-python",
        "canvas-list-test",
        "canvas-new-students-notification",
        "canvas-submit-late-work",
    ],
    "notion": [
        "notion-find-job",
        "notion-hr",
        "notion-movies",
        "notion-personal-website",
    ],
    "k8s": [
        "k8s-deployment-cleanup",
        "k8s-mysql",
        "k8s-pr-preview-testing",
        "k8s-redis-helm-upgrade",
        "k8s-safety-audit",
    ],
    "git": [
        "git-bug-hunt",
        "git-milestone",
        "git-repo",
    ],
    "woocommerce": [
        "woocommerce-customer-survey",
        "woocommerce-new-product",
        "woocommerce-new-welcome",
        "woocommerce-product-recall",
        "woocommerce-stock-alert",
        "woocommerce-update-cover",
    ],
    "email": [
        "apply-phd-email",
        "email-paper-homepage",
    ],
    "web": [
        "find-alita-paper",
        "price-comparison",
        "shopping-helper",
        "travel-exchange",
        "trip-adviser",
    ],
    "data_analysis": [
        "excel-data-transformation",
        "excel-market-research",
        "flagged-transactions",
        "game-statistics",
        "gdp-cr5-analysis",
        "live-transactions",
        "machine-operating",
        "sales-accounting",
        "stock-build-position",
    ],
    "research": [
        "academic-pdf-report",
        "cvpr-research",
        "hk-top-conf",
        "latex-prompt-box",
        "paper-checker",
    ],
    "financial": [
        "investment-decision-analysis",
        "nvidia-market",
        "nvidia-stock-analysis",
        "oil-price",
        "quantitative-financial-analysis",
        "yahoo-analysis",
    ],
    "filesystem": [
        "arrange-workspace",
        "dataset-license-issue",
        "privacy-desensitization",
        "task-tracker",
    ],
    "misc": [
        # Tasks that don't fit other categories
        "ab-testing",
        "academic-warning",
        "add-bibtex",
        "cooking-guidance",
        "course-assistant",
        "course-schedule",
        "courses-ta-hws",
        "detect-revised-terms",
        "dietary-health",
        "experiments-recordings",
        "fillout-online-forms",
        "filter-low-selling-products",
        "huggingface-upload",
        "identify-all-songs",
        "imagenet",
        "inter-final-performance-analysis",
        "interview-report",
        "inventory-sync",
        "invoice-org",
        "ipad-edu-price",
        "landing-task-reminder",
        "language-school",
        "llm-training-dataset",
        "logical-datasets-collection",
        "meeting-assign",
        "merge-hf-datasets",
        "mrbeast-analysis",
        "music-analysis",
        "nhl-b2b-analysis",
        "personal-website-construct",
        "ppt-analysis",
        "profile-update-online",
        "reimbursement-form-filler",
        "search-ca-school",
        "set-conf-cr-ddl",
        "sla-timeout-monitor",
        "student-interview",
        "subway-planning",
        "sync-todo-to-readme",
        "train-ticket-plan",
        "travel-expense-reimbursement",
        "trip-itinerary-generator",
        "university-course-selection",
        "update-material-inventory",
        "upenn-campus-route",
        "verl-dataset",
        "vlm-history-completer",
        "wandb-best-score",
        "wandb-shortest-length",
        "youtube-repo",
    ],
}

# Build reverse mapping: task_name -> category
_TASK_TO_CATEGORY: dict[str, str] = {}
for category, tasks in _TASK_CATEGORIES.items():
    for task in tasks:
        _TASK_TO_CATEGORY[task] = category

# Service type categories (based on MCP servers required)
ServiceCategory = Literal[
    "canvas",
    "notion",
    "k8s",
    "git",
    "woocommerce",
    "email",
    "web",
    "filesystem",
    "data_analysis",
    "research",
    "financial",
    "misc",
    None,
]


def get_task_category(task_name: str) -> str:
    """Get the category for a given task name."""
    return _TASK_TO_CATEGORY.get(task_name, "misc")


def load_toolathlon_dataset(
    shuffle: bool = False,
    limit: int | None = None,
    task_filter: str | None = None,
    service_filter: ServiceCategory | None = None,
) -> MemoryDataset:
    """Load the Toolathlon dataset.

    Args:
        shuffle: Whether to shuffle samples
        limit: Maximum number of samples to return
        task_filter: Filter by task name substring (e.g., "canvas" for canvas tasks)
        service_filter: Filter by service category (e.g., "canvas", "notion", "k8s")

    Returns:
        A MemoryDataset containing Toolathlon benchmark samples
    """
    if not _TASKS_DIR.exists():
        raise FileNotFoundError(
            f"Toolathlon tasks directory not found: {_TASKS_DIR}\n"
            f"Please ensure Toolathlon is cloned at {_TOOLATHLON_ROOT}"
        )

    samples: list[Sample] = []

    # Get all task directories (exclude non-directories)
    task_dirs = sorted([
        d.name for d in _TASKS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ])

    for task_name in task_dirs:
        # Apply filters
        category = get_task_category(task_name)

        # Service filter
        if service_filter and service_filter != "all":
            if category != service_filter:
                continue

        # Task filter (substring match on task name or category)
        if task_filter and task_filter != "all":
            if task_filter not in task_name and task_filter != category:
                continue

        # Load task config if exists
        task_dir = _TASKS_DIR / task_name
        task_config_path = task_dir / "task_config.json"

        mcp_servers = []
        if task_config_path.exists():
            import json
            try:
                with open(task_config_path) as f:
                    config = json.load(f)
                    mcp_servers = config.get("needed_mcp_servers", [])
            except Exception:
                pass

        # Create sample
        samples.append(Sample(
            input=f"Complete the task: {task_name}",
            target=task_name,  # Target is the task identifier
            id=_create_stable_id(task_name, prefix="toolathlon"),
            metadata={
                "task_name": task_name,
                "category": category,
                "task_dir": str(task_dir),
                "mcp_servers": mcp_servers,
            },
        ))

    if shuffle:
        random.shuffle(samples)

    if limit is not None:
        samples = samples[:limit]

    return MemoryDataset(samples)


def get_all_task_names() -> list[str]:
    """Get all available task names."""
    return list(_TASK_TO_CATEGORY.keys())


def get_tasks_by_category(category: ServiceCategory) -> list[str]:
    """Get all task names for a given category."""
    if category == "all" or category is None:
        return get_all_task_names()
    return _TASK_CATEGORIES.get(category, [])
