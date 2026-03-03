"""
OpenAgentSafety: Agent Safety in Realistic High-Risk Environments

Evaluates LLM agents on 360+ safety tasks in realistic environments
with real tools (file systems, terminals, browsers, messaging).

Based on:
    https://github.com/sani903/OpenAgentSafety

Usage:
    inspect eval benchmarks/local/openagentsafety/openagentsafety.py@openagentsafety \\
        --model openai/gpt-4o --limit 5
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver

try:
    from .dataset import load_oas_dataset
    from .runtime import run_oas_task
    from .scorer import oas_scorer
except ImportError:
    from dataset import load_oas_dataset
    from runtime import run_oas_task
    from scorer import oas_scorer

if TYPE_CHECKING:
    pass

# Task categories
TaskCategory = Literal[
    "leak",
    "credential",
    "compliance",
    "auth",
    "data",
    "injection",
    "social",
    "malicious",
    None,
]

# Service categories for filtering
ServiceCategory = Literal[
    "gitlab",
    "owncloud",
    "plane",
    "rocketchat",
    None,
]

# Default generation parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 600  # 10 minutes per task
DEFAULT_MAX_ITERATIONS = 50


@solver
def run_oas_evaluation(
    agent_llm: str = "openai/gpt-4o",
    env_llm: str = "openai/gpt-4o",
    server_hostname: str = "localhost",
    timeout: int = DEFAULT_TIMEOUT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    manage_services: bool = False,
    compose_file: str | Path | None = None,
    service_timeout: int = 300,
) -> Solver:
    """Solver that runs OAS evaluation for each task.

    This solver orchestrates the OAS evaluation by:
    1. Extracting task metadata from the sample
    2. Optionally starting required Docker services
    3. Calling the OAS run_eval.py script via subprocess
    4. Storing the evaluation result in state.metadata for the scorer

    Args:
        agent_llm: LLM for the agent (e.g., "openai/gpt-4o")
        env_llm: LLM for environment/NPC interactions
        server_hostname: Hostname for OAS services
        timeout: Timeout per task in seconds
        max_iterations: Maximum agent iterations per task
        manage_services: Whether to manage Docker services from eval-poc
        compose_file: Path to docker-compose.yaml for service management
        service_timeout: Timeout for service health checks

    Returns:
        A Solver that runs OAS evaluation
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Initialize metadata if needed
        if not state.metadata:
            state.metadata = {}

        # Get task metadata
        task_path = state.metadata.get("oas_task_path")
        if not task_path:
            # Skip if no OAS task path (shouldn't happen)
            state.metadata["oas_error"] = "No OAS task path in metadata"
            return state

        # Run OAS evaluation
        result = run_oas_task(
            task_path=task_path,
            agent_llm=agent_llm,
            env_llm=env_llm,
            server_hostname=server_hostname,
            timeout=timeout,
            max_iterations=max_iterations,
            manage_services=manage_services,
            compose_file=compose_file,
            service_timeout=service_timeout,
        )

        # Store result in metadata for scorer
        state.metadata["oas_eval_result"] = result.get("eval_result", {})
        state.metadata["oas_success"] = result["success"]

        if not result["success"]:
            error = result.get("error", "Unknown error")
            state.metadata["oas_error"] = error
            # Set default unsafe score on failure
            state.metadata["oas_eval_result"] = {"final_score": {"total": 1, "result": 1}}

        return state

    return solve


@task
def openagentsafety(
    task_filter: TaskCategory | None = None,
    service_filter: ServiceCategory | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    agent_llm: str = "openai/gpt-4o",
    env_llm: str | None = None,
    server_hostname: str = "localhost",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    manage_services: bool = False,
    compose_file: str | Path | None = None,
    service_timeout: int = 300,
) -> Task:
    """OpenAgentSafety: Comprehensive agent safety benchmark.

    Evaluates agents in realistic high-risk environments with real tools.
    360+ tasks covering data leaks, credential theft, compliance violations,
    prompt injection, and more.

    The benchmark uses OpenHands runtime with:
    - Real file systems
    - Terminal access
    - Browser automation
    - Messaging platforms (RocketChat, GitLab, OwnCloud, Plane)
    - Adversarial FakeUser simulation

    Args:
        task_filter: Filter tasks by category (leak, credential, compliance, etc.)
        service_filter: Filter tasks by required service (gitlab, owncloud, plane, rocketchat)
        limit: Maximum number of tasks to evaluate
        shuffle: Whether to shuffle task order
        agent_llm: LLM for agent (default: openai/gpt-4o)
        env_llm: LLM for environment/NPC (default: same as agent_llm)
        server_hostname: Hostname for OAS services
        max_iterations: Max agent iterations per task
        timeout: Timeout per task (seconds)
        temperature: Generation temperature
        max_tokens: Maximum tokens in response
        manage_services: Whether to manage Docker services from eval-poc
        compose_file: Path to docker-compose.yaml for service management
        service_timeout: Timeout for service health checks

    Returns:
        A Task that evaluates agent safety across OAS scenarios

    Examples:
        # Run 5 random tasks
        inspect eval ...@openagentsafety --model openai/gpt-4o -T limit=5 -T shuffle=true

        # Run only leak-related tasks
        inspect eval ...@openagentsafety --model openai/gpt-4o -T task_filter="leak"

        # Run only GitLab-related tasks with service management
        inspect eval ...@openagentsafety --model openai/gpt-4o -T service_filter="gitlab" -T manage_services=true

        # Run with different env model
        inspect eval ...@openagentsafety --model openai/gpt-4o -T env_llm="openai/gpt-4o-mini"
    """
    if env_llm is None:
        env_llm = agent_llm

    return Task(
        dataset=load_oas_dataset(
            task_filter=task_filter,
            service_filter=service_filter,
            limit=limit,
            shuffle=shuffle,
        ),
        solver=[
            run_oas_evaluation(
                agent_llm=agent_llm,
                env_llm=env_llm,
                server_hostname=server_hostname,
                timeout=timeout,
                max_iterations=max_iterations,
                manage_services=manage_services,
                compose_file=compose_file,
                service_timeout=service_timeout,
            )
        ],
        scorer=[oas_scorer()],
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        version="1.0.0",
    )


__all__ = [
    "openagentsafety",
    "run_oas_evaluation",
]
