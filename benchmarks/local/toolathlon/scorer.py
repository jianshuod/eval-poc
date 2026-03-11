"""
Scoring for Toolathlon benchmark.

Interprets Toolathlon evaluation results and maps them to inspect_ai Score format.

Toolathlon evaluation produces:
- pass: Boolean indicating if the task was completed successfully
- status: Task execution status (SUCCESS, FAILURE, etc.)
- details: Additional information about the result
- cost: Agent and user API costs
"""

from __future__ import annotations

from typing import Any

from inspect_ai.scorer import Metric, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import TaskState


@metric
def success_rate() -> Metric:
    """Calculate the success rate (percentage of passed tasks)."""

    def metric(scores: list[Score]) -> float:
        if not scores:
            return 0.0
        passed = sum(1 for s in scores if s.value == 1)
        return (passed / len(scores)) * 100

    return metric


@metric
def pass_count() -> Metric:
    """Count the number of passed tasks."""

    def metric(scores: list[Score]) -> int:
        return sum(1 for s in scores if s.value == 1)

    return metric


@metric
def fail_count() -> Metric:
    """Count the number of failed tasks."""

    def metric(scores: list[Score]) -> int:
        return sum(1 for s in scores if s.value == 0)

    return metric


@metric
def total_cost() -> Metric:
    """Calculate total API cost across all tasks."""

    def metric(scores: list[Score]) -> float:
        total = 0.0
        for s in scores:
            if s.metadata:
                cost = s.metadata.get("agent_cost", 0)
                if isinstance(cost, (int, float)):
                    total += cost
                elif isinstance(cost, dict):
                    total += cost.get("total_cost", 0)
        return total

    return metric


@metric
def category_breakdown() -> Metric:
    """Break down success rate by category."""

    def metric(scores: list[Score]) -> dict[str, dict[str, Any]]:
        from collections import defaultdict

        categories = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})

        for s in scores:
            if s.metadata:
                category = s.metadata.get("category", "unknown")
                categories[category]["total"] += 1
                if s.value == 1:
                    categories[category]["passed"] += 1
                else:
                    categories[category]["failed"] += 1

        # Calculate percentages
        result = {}
        for cat, counts in categories.items():
            if counts["total"] > 0:
                result[cat] = {
                    "passed": counts["passed"],
                    "failed": counts["failed"],
                    "total": counts["total"],
                    "success_rate": (counts["passed"] / counts["total"]) * 100,
                }

        return result

    return metric


@scorer(metrics=[success_rate(), pass_count(), fail_count(), total_cost(), category_breakdown()])
def toolathlon_scorer() -> Scorer:
    """Score model responses for Toolathlon benchmark.

    Extracts evaluation results from task metadata and produces a binary
    pass/fail score with additional information.

    The scorer looks for:
    - `pass` in metadata: Direct pass/fail from Toolathlon evaluation
    - `evaluation` in metadata: Evaluation result dict
    - `status` in metadata: Task execution status

    Returns:
        A Score with:
        - value: 1.0 for pass, 0.0 for fail
        - explanation: Human-readable explanation
        - metadata: Additional evaluation details
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get task information
        task_name = state.metadata.get("task_name", "unknown")
        category = state.metadata.get("category", "unknown")

        # Check for errors
        if "toolathlon_error" in state.metadata:
            return Score(
                value=0.0,
                explanation=f"Toolathlon evaluation error: {state.metadata['toolathlon_error']}",
                metadata={
                    "task_name": task_name,
                    "category": category,
                    "error": state.metadata["toolathlon_error"],
                },
            )

        # Get evaluation result
        evaluation = state.metadata.get("evaluation", {})
        pass_result = state.metadata.get("pass")

        # Determine pass/fail
        if pass_result is not None:
            passed = bool(pass_result)
        elif isinstance(evaluation, dict) and "pass" in evaluation:
            passed = bool(evaluation["pass"])
        else:
            # No explicit pass/fail, check status
            status = state.metadata.get("status", evaluation.get("status", ""))
            passed = status == "SUCCESS"

        # Build explanation
        if passed:
            explanation = f"Task '{task_name}' completed successfully"
            if evaluation.get("details"):
                explanation += f": {evaluation['details']}"
        else:
            explanation = f"Task '{task_name}' failed"
            failure = evaluation.get("failure")
            if failure:
                explanation += f": {failure}"
            details = evaluation.get("details")
            if details and failure != details:
                explanation += f" - {details}"

        # Extract cost information
        log_data = state.metadata.get("log", {})
        agent_cost = log_data.get("agent_cost", {}).get("total_cost", 0)
        user_cost = log_data.get("user_cost", {}).get("total_cost", 0)

        # Get key stats if available
        key_stats = log_data.get("key_stats", {})

        return Score(
            value=1.0 if passed else 0.0,
            explanation=explanation,
            metadata={
                "task_name": task_name,
                "category": category,
                "status": state.metadata.get("status", "unknown"),
                "agent_cost": agent_cost,
                "user_cost": user_cost,
                "total_cost": agent_cost + user_cost,
                "key_stats": key_stats,
                "evaluation": evaluation,
            },
        )

    return score


@scorer(metrics=[success_rate()])
def toolathlon_binary_scorer() -> Scorer:
    """Simple binary scorer for Toolathlon (pass/fail only).

    This is a simpler version that only reports pass/fail without
    detailed metrics. Useful for quick evaluations.
    """

    async def score(state: TaskState, target: Target) -> Score:
        evaluation = state.metadata.get("evaluation", {})
        pass_result = state.metadata.get("pass")

        if pass_result is not None:
            passed = bool(pass_result)
        elif isinstance(evaluation, dict) and "pass" in evaluation:
            passed = bool(evaluation["pass"])
        else:
            status = state.metadata.get("status", evaluation.get("status", ""))
            passed = status == "SUCCESS"

        return Score(
            value=1.0 if passed else 0.0,
            explanation=f"Task '{state.metadata.get('task_name', 'unknown')}' {'passed' if passed else 'failed'}",
        )

    return score
