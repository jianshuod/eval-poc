"""
ST-WebAgentBench - A Benchmark for Evaluating Safety & Trustworthiness in Web Agents

Based on:
    ST-WebAgentBench: A Benchmark for Evaluating Safety & Trustworthiness in Web Agents
    https://arxiv.org/abs/2410.06703
    https://github.com/segev-shlomov/ST-WebAgentBench

Usage:
    inspect eval benchmarks/local/st_webagentbench/st_webagentbench.py@st_webagentbench --model <model>

    # With task filter (gitlab, suitecrm, shopping_admin, or specific tier)
    inspect eval benchmarks/local/st_webagentbench/st_webagentbench.py@st_webagentbench --model <model> -T domain=gitlab

    # Run specific task ID
    inspect eval benchmarks/local/st_webagentbench/st_webagentbench.py@st_webagentbench --model <model> -T task_id=47

    # Run tier-based CRM tasks (easy, medium, hard)
    inspect eval benchmarks/local/st_webagentbench/st_webagentbench.py@st_webagentbench --model <model> -T tier=easy
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.util import SandboxEnvironmentType

try:
    from .dataset import load_st_webagentbench_dataset
    from .scorer import st_webagentbench_scorer
    from .solver import st_webagentbench_solver
except ImportError:
    # Fallback to absolute imports for when loaded directly
    from benchmarks.local.st_webagentbench.dataset import load_st_webagentbench_dataset
    from benchmarks.local.st_webagentbench.scorer import st_webagentbench_scorer
    from benchmarks.local.st_webagentbench.solver import st_webagentbench_solver


# BrowserGym action space configuration
_BROWSERGYM_AVAILABLE = False
try:
    from browsergym.core.action.highlevel import HighLevelActionSet

    _BROWSERGYM_AVAILABLE = True
except ImportError:
    pass


def get_action_set(multiaction: bool = False) -> "HighLevelActionSet | None":
    """Get the BrowserGym HighLevelActionSet for ST-WebAgentBench.

    Args:
        multiaction: Whether to enable multi-action mode

    Returns:
        HighLevelActionSet if browsergym is available, None otherwise
    """
    if not _BROWSERGYM_AVAILABLE:
        return None

    def finish(message: str):
        """Call when the task is done and you want to submit your final answer.

        Examples:
            finish("I have completed the task successfully.")
            finish("The task is done. Here is the answer: 42")
        """
        from browsergym.core.action.highlevel import send_message_to_user

        send_message_to_user(message)

    return HighLevelActionSet(
        custom_actions=[finish],
        subsets=["bid", "chat", "nav", "custom"],
        strict=False,
        multiaction=multiaction,
        demo_mode="off",
    )


@task
def st_webagentbench(
    domain: Literal["all", "gitlab", "suitecrm", "shopping_admin"] = "all",
    tier: Literal["all", "easy", "medium", "hard"] = "all",
    task_id: int | None = None,
    shuffle: bool = False,
    limit: int | None = None,
    max_steps: int = 50,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    sandbox: SandboxEnvironmentType | None = None,  # Ignored - BrowserGym manages browser env
) -> Task:
    """ST-WebAgentBench: Safety & Trustworthiness evaluation for web agents.

    This benchmark measures not only whether agents complete tasks (CR - Completion Rate),
    but whether they do so while respecting safety and trustworthiness policies
    (CuP - Completion under Policy).

    Args:
        domain: Filter by domain - "all", "gitlab", "suitecrm", or "shopping_admin"
        tier: Filter SuiteCRM by tier - "all", "easy", "medium", or "hard"
        task_id: Run a specific task ID (0-374). Overrides domain and tier filters.
        shuffle: Whether to shuffle the dataset
        limit: Maximum number of samples to evaluate
        max_steps: Maximum number of agent steps per task (default: 50)
        temperature: Generation temperature for LLM-based agents
        max_tokens: Maximum tokens for LLM responses
        sandbox: Sandbox environment type (BrowserGym environments manage their own)

    Returns:
        A Task that evaluates web agent safety across 6 dimensions:
        - Boundary & Scope: Stay within permitted resources
        - Strict Execution: Execute exactly as instructed
        - User Consent: Obtain approval before sensitive actions
        - Robustness & Security: Resist prompt injection and jailbreaking
        - Hierarchy Adherence: Follow Organization > User > Task priority
        - Error Handling: Handle errors gracefully

    Benchmark Details:
        - Total Tasks: 375 (GitLab: 0-196, ShoppingAdmin: 197-204, SuiteCRM: 205-374)
        - SuiteCRM Tiers: Easy (235-254), Medium (255-274), Hard (275-294)
        - Modality Tasks: Vision-advantage (295-334), DOM-advantage (335-374)
        - Primary Metric: CuP (Completion under Policy) = CR with zero policy violations
    """
    # Get action set if browsergym is available
    action_set = get_action_set(multiaction=False)

    return Task(
        dataset=load_st_webagentbench_dataset(
            domain=domain,
            tier=tier,
            task_id=task_id,
            shuffle=shuffle,
            limit=limit,
        ),
        solver=[st_webagentbench_solver(action_set=action_set, max_steps=max_steps)],
        scorer=[st_webagentbench_scorer()],
        sandbox=sandbox,
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        version="1.0.0",
    )



# Note: The actual BrowserGym integration solver is in solver.py
# The solver.py module contains st_webagentbench_solver which handles:
# - BrowserGym environment creation and management
# - Agent loop with action execution via env.step()
# - Safety report collection from the environment
# - Metadata population for the scorer (task_success, violated_policies, safety_report)