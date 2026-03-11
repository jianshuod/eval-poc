"""
BFCL v4 Scorer with Flexible Answer Support

Implements scoring for BFCL v4 with support for:
- Multiple valid answers per parameter
- Python, Java, and JavaScript type conversions
- Optional parameters
- Parallel and multiple function calls
- AST-based evaluation (bfcl_v4_ast_scorer) for alignment with upstream
"""

import logging
from typing import Any

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState

logger = logging.getLogger(__name__)

# Import AST evaluation module (supports package and top-level import modes)
try:
    from .ast_eval import (
        ast_checker,
        Language,
        standardize_string,
    )
    AST_EVAL_AVAILABLE = True
except ImportError:
    try:
        from ast_eval import (
            ast_checker,
            Language,
            standardize_string,
        )
        AST_EVAL_AVAILABLE = True
    except ImportError:
        AST_EVAL_AVAILABLE = False
        logger.warning("AST evaluation module not available, using basic scorer only")


def normalize_value(value: Any, param_type: str | None = None) -> Any:
    """
    Normalize a value for comparison.

    Handles type conversions between Python, Java, and JavaScript representations.
    """
    if value is None or value == "":
        return ""

    # Handle boolean strings
    if isinstance(value, str):
        lower_val = value.lower().strip()
        if lower_val in ("true", "false"):
            return lower_val == "true"

    # Handle numbers
    if isinstance(value, (int, float)):
        return value

    # Handle lists/arrays - normalize to list
    if isinstance(value, list):
        return [normalize_value(v, param_type) for v in value]

    # Handle dicts/objects
    if isinstance(value, dict):
        return {k: normalize_value(v, param_type) for k, v in value.items()}

    # String values - strip whitespace
    if isinstance(value, str):
        return value.strip()

    return value


def values_match(
    actual: Any, expected: Any, param_type: str | None = None
) -> bool:
    """
    Check if two values match, accounting for multiple possible values.

    The expected value can be:
    - A single value
    - A list of possible values (any match is accepted)
    - An empty string or None (optional parameter)
    """
    # Normalize the actual value
    normalized_actual = normalize_value(actual, param_type)

    # If expected is a list, check if actual matches any element
    if isinstance(expected, list):
        for exp_val in expected:
            normalized_exp = normalize_value(exp_val, param_type)
            if normalized_actual == normalized_exp:
                return True
        return False

    # Single expected value
    normalized_expected = normalize_value(expected, param_type)
    return normalized_actual == normalized_expected


def is_optional_param(expected_value: Any) -> bool:
    """
    Check if a parameter is optional based on its expected value.

    A parameter is considered optional if:
    - The value itself is "", None, or []
    - The value is a list containing "" or None (multiple valid answers with empty as option)
    """
    if expected_value in ("", None, []):
        return True
    # Check if it's a list containing empty/None values (optional with multiple choices)
    if isinstance(expected_value, list):
        return any(v in ("", None) for v in expected_value)
    return False


def params_match(
    actual_params: dict[str, Any],
    expected_params: dict[str, Any],
) -> bool:
    """
    Check if parameter dictionaries match.

    Handles:
    - Missing optional parameters (empty string or None in expected)
    - Extra parameters in actual (ignored)
    - Multiple valid values per parameter
    """
    # Check all expected parameters
    for param_name, expected_value in expected_params.items():
        if param_name not in actual_params:
            # Missing parameter - check if it's optional
            if is_optional_param(expected_value):
                # Optional parameter not provided - OK
                continue
            logger.debug(
                f"Missing required parameter: {param_name} (expected: {expected_value})"
            )
            return False

        actual_value = actual_params[param_name]
        if not values_match(actual_value, expected_value):
            logger.debug(
                f"Parameter mismatch for {param_name}: "
                f"actual={actual_value}, expected={expected_value}"
            )
            return False

    return True


def tool_calls_match(
    actual_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
) -> bool:
    """
    Check if lists of tool calls match.

    Handles parallel and multiple function calls.
    Order doesn't matter for parallel calls.
    """
    if len(actual_calls) != len(expected_calls):
        logger.debug(
            f"Tool call count mismatch: {len(actual_calls)} vs {len(expected_calls)}"
        )
        return False

    # Create copies of lists for matching
    expected_remaining = list(expected_calls)

    for actual_call in actual_calls:
        found_match = False
        for i, expected_call in enumerate(expected_remaining):
            if single_tool_call_matches(actual_call, expected_call):
                found_match = True
                expected_remaining.pop(i)
                break

        if not found_match:
            logger.debug(f"No matching expected call for: {actual_call}")
            return False

    return True


def single_tool_call_matches(
    actual_call: dict[str, Any],
    expected_call: dict[str, Any],
) -> bool:
    """
    Check if a single tool call matches the expected call.

    Args:
        actual_call: Dict with 'function' (str) and 'arguments' (dict)
        expected_call: Dict with function name as key and params as value

    Expected format from BFCL v4:
        {"function_name": {"param1": [val1, val2], "param2": val3}}
    """
    # Get function name from actual call
    actual_function = actual_call.get("function")
    if not actual_function:
        return False

    # Expected call has function name as key
    if len(expected_call) != 1:
        return False

    expected_function = next(iter(expected_call.keys()))
    if actual_function != expected_function:
        logger.debug(
            f"Function name mismatch: {actual_function} vs {expected_function}"
        )
        return False

    # Compare parameters
    expected_params = expected_call[expected_function]
    actual_params = actual_call.get("arguments", {})

    return params_match(actual_params, expected_params)


@scorer([accuracy()])
def bfcl_v4_scorer() -> Scorer:
    """
    BFCL v4 scorer with support for multiple valid answers.

    Scoring logic:
    - "C" (Correct): Function name and all parameters match
    - "I" (Invalid): Function name or parameters don't match
    """

    async def score(state: TaskState, target: Target) -> Score:
        assistant_messages = [
            m for m in state.messages if isinstance(m, ChatMessageAssistant)
        ]

        if len(assistant_messages) == 0:
            return Score(value="I", answer="No assistant message")
        elif len(assistant_messages) != 1:
            return Score(
                value="I",
                answer=f"Expected just 1 assistant message, got {len(assistant_messages)}",
            )

        message = assistant_messages[0]

        tool_calls = message.tool_calls

        # Check if this is an irrelevance category
        category = state.metadata.get("category", "")
        is_irrelevance = "irrelevance" in category

        if tool_calls is None or len(tool_calls) == 0:
            if is_irrelevance:
                # For irrelevance: NO tool calls is CORRECT (model recognized irrelevance)
                return Score(value="C", answer="No tool calls (correct for irrelevance)")
            else:
                # For other categories: NO tool calls is INCORRECT
                return Score(value="I", answer="No tool calls")

        # Get expected calls from metadata
        ground_truth = state.metadata.get("ground_truth", [])

        # Normalize ground_truth format to list of dicts
        # The ground_truth can be either:
        # - A dict: {'function_name': {params}} for single calls
        # - A list: [{'function_name': {params}}, ...] for multiple/parallel calls
        if isinstance(ground_truth, dict):
            # Convert single dict to list format
            expected_calls = [ground_truth]
        else:
            expected_calls = ground_truth

        # Convert inspect_ai tool calls to dict format
        actual_calls = []
        for tc in tool_calls:
            actual_calls.append({
                "function": tc.function,
                "arguments": tc.arguments,
            })

        # Check if calls match
        is_correct = tool_calls_match(actual_calls, expected_calls)

        value = "C" if is_correct else "I"

        # Format answer for display
        answer = format_tool_calls(actual_calls)

        return Score(value=value, answer=answer)

    return score


def format_tool_calls(calls: list[dict[str, Any]]) -> str:
    """Format tool calls for display in answer."""
    if not calls:
        return "No tool calls"

    formatted = []
    for call in calls:
        func = call.get("function", "?")
        args = call.get("arguments", {})
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        formatted.append(f"{func}({args_str})")

    return " | ".join(formatted)


# ============================================================================
# Multi-Turn Scorer
# ============================================================================

@scorer([accuracy()])
def bfcl_v4_multi_turn_scorer() -> Scorer:
    """
    BFCL v4 scorer for multi-turn conversations.

    Scoring logic:
    - Collects all tool calls from all assistant messages across all turns
    - Compares against the full ground truth sequence
    - Supports multiple assistant messages per conversation
    - A conversation is correct if ALL tool calls match the ground truth
    """

    async def score(state: TaskState, target: Target) -> Score:
        assistant_messages = [
            m for m in state.messages if isinstance(m, ChatMessageAssistant)
        ]

        if len(assistant_messages) == 0:
            return Score(value="I", answer="No assistant messages")

        # Collect all tool calls from all assistant messages
        all_actual_calls = []
        for msg in assistant_messages:
            tool_calls = msg.tool_calls
            if tool_calls:
                for tc in tool_calls:
                    all_actual_calls.append({
                        "function": tc.function,
                        "arguments": tc.arguments,
                    })

        if not all_actual_calls:
            # Check if this is an irrelevance category
            category = state.metadata.get("category", "")
            is_irrelevance = "irrelevance" in category

            if is_irrelevance:
                # For irrelevance: NO tool calls is CORRECT (model recognized irrelevance)
                return Score(value="C", answer="No tool calls (correct for irrelevance)")
            else:
                # For other categories: NO tool calls is INCORRECT
                return Score(value="I", answer="No tool calls in conversation")

        # Get expected calls from metadata
        ground_truth = state.metadata.get("ground_truth", [])

        # Normalize ground_truth format to list of dicts
        # The ground_truth can be either:
        # - A dict: {'function_name': {params}} for single calls
        # - A list: [{'function_name': {params}}, ...] for multiple/parallel calls
        if isinstance(ground_truth, dict):
            # Convert single dict to list format
            expected_calls = [ground_truth]
        else:
            expected_calls = ground_truth

        # Convert inspect_ai tool calls to dict format
        actual_calls_dict = []
        for tc in all_actual_calls:
            actual_calls_dict.append({
                "function": tc.function,
                "arguments": tc.arguments,
            })

        # Check if calls match
        is_correct = tool_calls_match(actual_calls_dict, expected_calls)

        value = "C" if is_correct else "I"

        # Format answer for display
        answer = format_tool_calls(actual_calls_dict)

        return Score(value=value, answer=answer)

    return score


# ============================================================================
# AST-based Scorer (Upstream Aligned)
# ============================================================================


@scorer([accuracy()])
def bfcl_v4_ast_scorer() -> Scorer:
    """
    BFCL v4 AST-based scorer aligned with upstream Berkeley implementation.

    This scorer uses AST-based evaluation with:
    - Comprehensive type checking (Python, Java, JavaScript)
    - String standardization (case-insensitive, punctuation normalization)
    - Variable detection
    - Nested type validation (one level deep)
    - Detailed error classification

    Scoring logic:
    - "C" (Correct): Function name and all parameters match (AST-validated)
    - "I" (Invalid): Function name or parameters don't match

    Error types reported:
    - simple_function_checker:wrong_func_name
    - simple_function_checker:missing_required
    - simple_function_checker:unexpected_param
    - simple_function_checker:missing_optional
    - type_error:simple, type_error:java, type_error:js, type_error:nested
    - value_error:string, value_error:list/tuple, value_error:dict_key, etc.
    """

    if not AST_EVAL_AVAILABLE:
        raise ImportError(
            "AST evaluation module not available. "
            "Ensure ast_eval.py is present and imports succeed."
        )

    async def score(state: TaskState, target: Target) -> Score:
        assistant_messages = [
            m for m in state.messages if isinstance(m, ChatMessageAssistant)
        ]

        if len(assistant_messages) == 0:
            return Score(
                value="I",
                answer="No assistant message",
                error_type="no_assistant_message"
            )
        elif len(assistant_messages) != 1:
            return Score(
                value="I",
                answer=f"Expected 1 assistant message, got {len(assistant_messages)}",
                error_type="wrong_message_count"
            )

        message = assistant_messages[0]
        tool_calls = message.tool_calls

        # Check if this is an irrelevance category
        test_category = state.metadata.get("category", "simple_python")
        is_irrelevance = "irrelevance" in test_category

        if tool_calls is None or len(tool_calls) == 0:
            if is_irrelevance:
                # For irrelevance: NO tool calls is CORRECT (model recognized irrelevance)
                return Score(
                    value="C",
                    answer="No tool calls (correct for irrelevance)",
                    error_type="no_tool_calls"
                )
            else:
                # For other categories: NO tool calls is INCORRECT
                return Score(
                    value="I",
                    answer="No tool calls",
                    error_type="no_tool_calls"
                )

        # Get ground truth from metadata
        ground_truth = state.metadata.get("ground_truth", [])
        func_description = state.metadata.get("tools", [])
        test_category = state.metadata.get("category", "simple_python")
        model_name = state.metadata.get("model_name", "")

        # Convert inspect_ai tool calls to upstream format
        model_output = []
        for tc in tool_calls:
            model_output.append({
                tc.function: tc.arguments,
            })

        # Determine language from category
        if "java" in test_category:
            language = Language.JAVA
        elif "javascript" in test_category:
            language = Language.JAVASCRIPT
        else:
            language = Language.PYTHON

        # Get possible answers if available
        possible_answer = state.metadata.get("possible_answer", ground_truth)

        # Run AST checker
        result = ast_checker(
            func_description=func_description,
            model_output=model_output,
            possible_answer=possible_answer,
            language=language,
            test_category=test_category,
            model_name=model_name,
        )

        # Extract error type for detailed reporting
        error_type = result.get("error_type", "unknown")
        errors = result.get("error", [])

        value = "C" if result["valid"] else "I"

        # Format answer for display
        answer = format_tool_calls([{
            "function": tc.function,
            "arguments": tc.arguments,
        } for tc in tool_calls])

        # Build explanation with error details
        explanation = answer
        if not result["valid"] and errors:
            error_summary = errors[0] if isinstance(errors[0], str) else str(errors[0])
            explanation = f"{answer} | Error: {error_type} - {error_summary}"

        return Score(
            value=value,
            answer=answer,
            explanation=explanation,
            error_type=error_type,
            errors=errors,
        )

    return score


# ============================================================================
# Multi-Turn AST-based Scorer
# ============================================================================


@scorer([accuracy()])
def bfcl_v4_multi_turn_ast_scorer() -> Scorer:
    """
    BFCL v4 AST-based scorer for multi-turn conversations.

    Collects all tool calls from all assistant messages across all turns
    and compares against the full ground truth sequence.

    Scoring logic:
    - "C" (Correct): ALL tool calls match the ground truth (AST-validated)
    - "I" (Invalid): Any tool call doesn't match
    """

    if not AST_EVAL_AVAILABLE:
        raise ImportError(
            "AST evaluation module not available. "
            "Ensure ast_eval.py is present and imports succeed."
        )

    async def score(state: TaskState, target: Target) -> Score:
        assistant_messages = [
            m for m in state.messages if isinstance(m, ChatMessageAssistant)
        ]

        if len(assistant_messages) == 0:
            return Score(
                value="I",
                answer="No assistant messages",
                error_type="no_assistant_message"
            )

        # Collect all tool calls from all assistant messages
        all_actual_calls = []
        for msg in assistant_messages:
            tool_calls = msg.tool_calls
            if tool_calls:
                for tc in tool_calls:
                    all_actual_calls.append({
                        tc.function: tc.arguments,
                    })

        if not all_actual_calls:
            # Check if this is an irrelevance category
            test_category = state.metadata.get("category", "multi_turn_base")
            is_irrelevance = "irrelevance" in test_category

            if is_irrelevance:
                # For irrelevance: NO tool calls is CORRECT (model recognized irrelevance)
                return Score(
                    value="C",
                    answer="No tool calls (correct for irrelevance)",
                    error_type="no_tool_calls"
                )
            else:
                # For other categories: NO tool calls is INCORRECT
                return Score(
                    value="I",
                    answer="No tool calls in conversation",
                    error_type="no_tool_calls"
                )

        # Get ground truth from metadata
        ground_truth = state.metadata.get("ground_truth", [])
        func_description = state.metadata.get("tools", [])
        test_category = state.metadata.get("category", "multi_turn_base")
        model_name = state.metadata.get("model_name", "")

        # Determine language from category
        if "java" in test_category:
            language = Language.JAVA
        elif "javascript" in test_category:
            language = Language.JAVASCRIPT
        else:
            language = Language.PYTHON

        # Get possible answers if available
        possible_answer = state.metadata.get("possible_answer", ground_truth)

        # Run AST checker
        result = ast_checker(
            func_description=func_description,
            model_output=all_actual_calls,
            possible_answer=possible_answer,
            language=language,
            test_category=test_category,
            model_name=model_name,
        )

        # Extract error type for detailed reporting
        error_type = result.get("error_type", "unknown")
        errors = result.get("error", [])

        value = "C" if result["valid"] else "I"

        # Format answer for display
        answer = format_tool_calls([
            {"function": list(call.keys())[0], "arguments": list(call.values())[0]}
            for call in all_actual_calls
        ])

        # Build explanation with error details
        explanation = answer
        if not result["valid"] and errors:
            error_summary = errors[0] if isinstance(errors[0], str) else str(errors[0])
            explanation = f"{answer} | Error: {error_type} - {error_summary}"

        return Score(
            value=value,
            answer=answer,
            explanation=explanation,
            error_type=error_type,
            errors=errors,
        )

    return score
