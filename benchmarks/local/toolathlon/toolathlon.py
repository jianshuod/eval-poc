"""
Toolathlon: The Tool Decathlon - Benchmarking Language Agents

Evaluates LLM agents on 109 diverse, realistic, long-horizon tasks
using 600+ tools across multiple real-world environments.

Based on:
    Paper: https://arxiv.org/abs/2510.25726
    GitHub: https://github.com/hkust-nlp/Toolathlon
    Website: https://toolathlon.xyz/

Usage:
    inspect eval benchmarks/local/toolathlon/toolathlon.py@toolathlon \\
        --model openai/gpt-4o --limit 5

    # Run only Canvas-related tasks
    inspect eval benchmarks/local/toolathlon/toolathlon.py@toolathlon \\
        --model openai/gpt-4o -T service_filter="canvas"

    # Run with safety-confirmation (requires safety proxy)
    inspect eval benchmarks/local/toolathlon/toolathlon.py@toolathlon \\
        --model openai/gpt-4o -T use_safety_proxy=true --limit 5
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver

try:
    from .dataset import load_toolathlon_dataset, ServiceCategory
    from .runtime import run_toolathlon_evaluation, verify_toolathlon_setup
    from .scorer import toolathlon_scorer
except ImportError:
    from dataset import load_toolathlon_dataset, ServiceCategory
    from runtime import run_toolathlon_evaluation, verify_toolathlon_setup
    from scorer import toolathlon_scorer

if TYPE_CHECKING:
    pass

# Default generation parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 3600  # 1 hour per task
DEFAULT_MAX_STEPS = 50  # Max agent iterations


def _base_solver(
    toolathlon_root: str | Path | None = None,
    mode: str = "public_service",
    service_host: str | None = None,
    service_port: int | None = None,
    ws_proxy_port: str | None = None,
    use_safety_proxy: bool = False,
    safety_proxy_host: str = "127.0.0.1",
    safety_proxy_port: int = 8765,
    timeout: int = DEFAULT_TIMEOUT,
    max_steps: int | None = None,
    verify_setup: bool = False,
) -> Solver:
    """Create base solver for Toolathlon evaluation."""
    solvers: list[Solver] = []

    if verify_setup:
        from .runtime import verify_toolathlon_setup
        solvers.append(_verify_toolathlon(toolathlon_root, mode))

    solvers.append(
        run_toolathlon_evaluation(
            toolathlon_root=toolathlon_root,
            mode=mode,
            service_host=service_host,
            service_port=service_port,
            ws_proxy_port=ws_proxy_port,
            use_safety_proxy=use_safety_proxy,
            safety_proxy_host=safety_proxy_host,
            safety_proxy_port=safety_proxy_port,
            timeout=timeout,
            max_steps=max_steps,
        )
    )

    @solver
    def chain() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            for s in solvers:
                state = await s(state, generate)
            return state

    return chain()


@solver
def _verify_toolathlon(
    toolathlon_root: str | Path | None = None,
    mode: str = "public_service",
) -> Solver:
    """Verify Toolathlon setup before running tasks."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from .runtime import verify_toolathlon_setup
        result = verify_toolathlon_setup(toolathlon_root, mode=mode)

        if not result.get("root_exists"):
            raise RuntimeError(
                f"Toolathlon not found at {result['toolathlon_root']}. "
                f"Please clone Toolathlon to that location."
            )

        # Report warnings (not errors)
        for issue in result.get("issues", []):
            if not state.metadata.get("warning"):
                state.metadata["warning"] = []
            if isinstance(state.metadata["warning"], str):
                state.metadata["warning"] = [state.metadata["warning"]]
            state.metadata["warning"].append(issue)

        return state

    return solve


@task
def toolathlon(
    shuffle: bool = False,
    limit: int | None = None,
    task_filter: str | None = None,
    service_filter: ServiceCategory | None = None,
    toolathlon_root: str | Path | None = None,
    mode: str = "public_service",
    service_host: str | None = None,
    service_port: int | None = None,
    ws_proxy_port: str | None = None,
    use_safety_proxy: bool = False,
    safety_proxy_host: str = "127.0.0.1",
    safety_proxy_port: int = 8765,
    timeout: int = DEFAULT_TIMEOUT,
    max_steps: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verify_setup: bool = False,
) -> Task:
    """Toolathlon benchmark - all 109 tasks."""
    return Task(
        dataset=load_toolathlon_dataset(
            shuffle=shuffle,
            limit=limit,
            task_filter=task_filter,
            service_filter=service_filter,
        ),
        solver=[_base_solver(
            toolathlon_root=toolathlon_root,
            mode=mode,
            service_host=service_host,
            service_port=service_port,
            ws_proxy_port=ws_proxy_port,
            use_safety_proxy=use_safety_proxy,
            safety_proxy_host=safety_proxy_host,
            safety_proxy_port=safety_proxy_port,
            timeout=timeout,
            max_steps=max_steps,
            verify_setup=verify_setup,
        )],
        scorer=[toolathlon_scorer()],
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        version="1.0.0",
    )


@task
def toolathlon_sample(
    shuffle: bool = True,
    samples_per_category: int = 1,
    toolathlon_root: str | Path | None = None,
    mode: str = "public_service",
    service_host: str | None = None,
    service_port: int | None = None,
    ws_proxy_port: str | None = None,
    use_safety_proxy: bool = False,
    safety_proxy_host: str = "127.0.0.1",
    safety_proxy_port: int = 8765,
    timeout: int = DEFAULT_TIMEOUT,
    max_steps: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verify_setup: bool = False,
) -> Task:
    """Toolathlon benchmark - sample of tasks from each category."""
    from .dataset import get_tasks_by_category

    samples: list = []
    for category in [
        "canvas", "notion", "k8s", "git", "woocommerce",
        "email", "web", "filesystem", "data_analysis",
        "research", "financial", "misc",
    ]:
        tasks = get_tasks_by_category(category)
        if shuffle:
            import random
            random.shuffle(tasks)
        for task_name in tasks[:samples_per_category]:
            samples.append(task_name)

    return Task(
        dataset=load_toolathlon_dataset(
            shuffle=shuffle,
            task_filter=",".join(samples) if samples else None,
        ),
        solver=[_base_solver(
            toolathlon_root=toolathlon_root,
            mode=mode,
            service_host=service_host,
            service_port=service_port,
            ws_proxy_port=ws_proxy_port,
            use_safety_proxy=use_safety_proxy,
            safety_proxy_host=safety_proxy_host,
            safety_proxy_port=safety_proxy_port,
            timeout=timeout,
            max_steps=max_steps,
            verify_setup=verify_setup,
        )],
        scorer=[toolathlon_scorer()],
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        version="1.0.0",
    )


# Service-specific tasks
@task
def toolathlon_canvas(limit: int | None = None, **kwargs) -> Task:
    """Canvas LMS-specific tasks (8 tasks)."""
    return toolathlon(service_filter="canvas", limit=limit, **kwargs)


@task
def toolathlon_notion(limit: int | None = None, **kwargs) -> Task:
    """Notion-specific tasks (4 tasks)."""
    return toolathlon(service_filter="notion", limit=limit, **kwargs)


@task
def toolathlon_k8s(limit: int | None = None, **kwargs) -> Task:
    """Kubernetes-specific tasks (5 tasks)."""
    return toolathlon(service_filter="k8s", limit=limit, **kwargs)


@task
def toolathlon_git(limit: int | None = None, **kwargs) -> Task:
    """Git-specific tasks (3 tasks)."""
    return toolathlon(service_filter="git", limit=limit, **kwargs)


@task
def toolathlon_woocommerce(limit: int | None = None, **kwargs) -> Task:
    """WooCommerce-specific tasks (6 tasks)."""
    return toolathlon(service_filter="woocommerce", limit=limit, **kwargs)


@task
def toolathlon_research(limit: int | None = None, **kwargs) -> Task:
    """Research/academic tasks (5 tasks)."""
    return toolathlon(service_filter="research", limit=limit, **kwargs)


@task
def toolathlon_financial(limit: int | None = None, **kwargs) -> Task:
    """Financial analysis tasks (6 tasks)."""
    return toolathlon(service_filter="financial", limit=limit, **kwargs)
