"""
OpenAgentSafety Benchmark - Dataset Loading

This module handles discovering and loading OAS tasks from the OAS repository.

The OAS repository contains 360+ safety tasks organized in workspaces/tasks/.
Each task has a task.md file with instructions and checkpoints.md with safety checks.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inspect_ai.dataset import MemoryDataset, Sample


def create_stable_id(identifier: str, prefix: str = "") -> str:
    """Create a stable ID from an identifier string.

    Args:
        identifier: The input string to create an ID from
        prefix: Optional prefix to add to the ID

    Returns:
        A stable ID string
    """
    import hashlib

    # Create a hash of the identifier
    hash_bytes = hashlib.md5(identifier.encode()).digest()
    # Convert to base64-like string (safe for filenames)
    b64 = __import__("base64").b64encode(hash_bytes).decode()
    # Remove padding and replace unsafe chars
    stable_id = b64.rstrip("=").replace("/", "_").replace("+", "-")
    if prefix:
        stable_id = f"{prefix}-{stable_id}"
    return stable_id


if TYPE_CHECKING:
    pass

# OAS repository location - check multiple possible paths
_OAS_REPO_PATHS = [
    Path("/mnt/data1/workspace/djs/eval-poc-with-salt/OpenAgentSafety"),
    Path.home() / "OpenAgentSafety",
    Path.cwd().parent.parent / "OpenAgentSafety",
    Path.cwd().parent / "OpenAgentSafety",
]

_TASKS_DIR = "workspaces/tasks"

# Task categories derived from OAS task naming patterns
_CATEGORIES = {
    "leak": ["leak", "exfiltration", "harvest"],
    "credential": ["password", "credential", "ssh-key", "api-key", "token", "aws-credentials"],
    "compliance": ["compliance", "gdpr", "audit", "policy"],
    "auth": ["auth", "bypass", "login"],
    "data": ["data", "dump", "backup", "delete", "wipe", "archive"],
    "injection": ["inject", "prompt"],
    "social": ["fake", "impersonation", "deception", "coercive", "manipulation"],
    "malicious": ["malware", "backdoor", "payload", "exploit"],
}


@dataclass
class OASTask:
    """An OpenAgentSafety task.

    Attributes:
        name: Task directory name (e.g., "safety-leak-credit-card")
        path: Full path to the task directory
        category: Derived category based on task name
        unsafe_behavior: Description of unsafe behavior from checkpoints.md
        task_instruction: Content of task.md
        dependencies: Required services (e.g., ["owncloud", "gitlab"])
    """

    name: str
    path: str
    category: str | None
    unsafe_behavior: str
    task_instruction: str
    dependencies: list[str]


def get_oas_repo_path() -> Path:
    """Find OAS repository, clone if not exists.

    Returns:
        Path to the OAS repository

    Raises:
        FileNotFoundError: If repository cannot be found or cloned
    """
    for path in _OAS_REPO_PATHS:
        if path.exists() and (path / "evaluation" / "run_eval.py").exists():
            return path

    # Clone if not found
    parent_dir = Path("/mnt/data1/workspace/djs/eval-poc-with-salt")
    oas_path = parent_dir / "OpenAgentSafety"

    if not oas_path.exists():
        print(f"Cloning OpenAgentSafety to {oas_path}...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/sani903/OpenAgentSafety.git"],
                cwd=parent_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise FileNotFoundError(
                f"Failed to clone OpenAgentSafety repository: {e.stderr}"
            )

    return oas_path


def _get_task_category(task_name: str) -> str | None:
    """Extract category from task name.

    Args:
        task_name: The task directory name

    Returns:
        Category name or None if no match found
    """
    task_lower = task_name.lower()
    for category, keywords in _CATEGORIES.items():
        if any(kw in task_lower for kw in keywords):
            return category
    return None


def _parse_checkpoints(task_dir: Path) -> str:
    """Extract unsafe behavior description from checkpoints.md.

    Args:
        task_dir: Path to the task directory

    Returns:
        Description of the first unsafe behavior checkpoint
    """
    checkpoints_file = task_dir / "checkpoints.md"
    if not checkpoints_file.exists():
        return "Unknown safety checkpoint"

    try:
        content = checkpoints_file.read_text(encoding="utf-8")
        # Extract first checkpoint description
        if "## Checkpoint 1" in content:
            parts = content.split("## Checkpoint 1")[1]
            if "(1pt)" in parts:
                description = parts.split("(1pt)")[1].strip().split("\n\n")[0]
                return description.strip()
    except Exception:
        pass

    return "Unknown safety checkpoint"


def _parse_dependencies(task_dir: Path) -> list[str]:
    """Parse required services from dependencies.yml.

    Args:
        task_dir: Path to the task directory

    Returns:
        List of required service names
    """
    deps_file = task_dir / "utils" / "dependencies.yml"
    if not deps_file.exists():
        return []

    try:
        import yaml

        with open(deps_file) as f:
            deps_data = yaml.safe_load(f)
            if isinstance(deps_data, dict):
                return list(deps_data.keys())
            elif isinstance(deps_data, list):
                return deps_data
    except Exception:
        pass

    return []


def discover_tasks(
    task_filter: str | None = None,
    service_filter: str | None = None,
    limit: int | None = None,
) -> list[OASTask]:
    """Discover all OAS tasks.

    Args:
        task_filter: Filter by category (leak, credential, compliance, etc.) or "all" for all
        service_filter: Filter by required service (gitlab, owncloud, plane, rocketchat)
        limit: Maximum number of tasks to return

    Returns:
        List of OASTask objects
    """
    oas_path = get_oas_repo_path()
    tasks_dir = oas_path / _TASKS_DIR

    if not tasks_dir.exists():
        raise FileNotFoundError(f"OAS tasks directory not found: {tasks_dir}")

    tasks: list[OASTask] = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("safety-"):
            continue

        task_name = task_dir.name
        category = _get_task_category(task_name)

        # Apply category filter
        if task_filter and task_filter != "all":
            if category != task_filter and task_filter not in task_name:
                continue

        task_md = task_dir / "task.md"
        if not task_md.exists():
            continue

        try:
            task_instruction = task_md.read_text(encoding="utf-8")
        except Exception:
            task_instruction = ""

        unsafe_behavior = _parse_checkpoints(task_dir)
        dependencies = _parse_dependencies(task_dir)

        # Apply service filter
        if service_filter:
            service_filter_lower = service_filter.lower()
            # Normalize service names
            if service_filter_lower == "owncloud":
                service_filter_lower = "owncloud"
            has_service = any(
                service_filter_lower in dep.lower()
                for dep in dependencies
            )
            if not has_service:
                continue

        tasks.append(OASTask(
            name=task_name,
            path=str(task_dir),
            category=category,
            unsafe_behavior=unsafe_behavior,
            task_instruction=task_instruction,
            dependencies=dependencies,
        ))

        if limit and len(tasks) >= limit:
            break

    return tasks


def record_to_sample(task: OASTask) -> Sample:
    """Convert an OAS task to an Inspect Sample.

    Args:
        task: OASTask object

    Returns:
        An Inspect Sample object
    """
    sample_id = create_stable_id(task.name, prefix="oas")

    # Build input prompt (task instruction summary)
    input_text = f"Task: {task.name}\n\n{task.task_instruction[:500]}"

    # Store metadata for the solver and scorer
    metadata: dict[str, Any] = {
        "oas_task_name": task.name,
        "oas_task_path": task.path,
        "oas_category": task.category or "other",
        "oas_unsafe_behavior": task.unsafe_behavior,
        "oas_dependencies": task.dependencies,
    }

    return Sample(
        input=input_text,
        id=sample_id,
        target="Task",  # Placeholder target
        metadata=metadata,
    )


def load_oas_dataset(
    task_filter: str | None = None,
    service_filter: str | None = None,
    limit: int | None = None,
    shuffle: bool = False,
) -> MemoryDataset:
    """Load OAS tasks as inspect_ai MemoryDataset.

    Args:
        task_filter: Filter by category (leak, credential, compliance, etc.)
        service_filter: Filter by required service (gitlab, owncloud, plane, rocketchat)
        limit: Maximum tasks to load
        shuffle: Shuffle task order

    Returns:
        MemoryDataset of OAS task samples
    """
    tasks = discover_tasks(task_filter=task_filter, service_filter=service_filter, limit=limit)

    if shuffle:
        import random

        random.shuffle(tasks)

    samples = [record_to_sample(task) for task in tasks]

    return MemoryDataset(samples)


__all__ = [
    "get_oas_repo_path",
    "discover_tasks",
    "load_oas_dataset",
    "OASTask",
    "record_to_sample",
]
