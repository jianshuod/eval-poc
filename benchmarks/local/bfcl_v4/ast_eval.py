"""
BFCL v4 AST-based Evaluation

Ported from upstream Berkeley Function Call Leaderboard v4:
https://github.com/ShishirPatil/gorilla/tree/main/bfcl_eval/eval_checker/ast_eval

This module provides comprehensive AST-based evaluation for function calls,
including:
- String standardization (case-insensitive, punctuation normalization)
- Type checking (Python, Java, JavaScript)
- Variable detection
- Nested type validation
- Detailed error classification
"""

import logging
import re
from enum import Enum
from typing import Any

try:
    from .type_convertor import (
        java_type_converter,
        js_type_converter,
        get_possible_answer_type,
        JAVA_TYPE_CONVERSION,
        JS_TYPE_CONVERSION,
        PYTHON_TYPE_MAPPING,
        PYTHON_NESTED_TYPE_CHECK_LIST,
        NESTED_CONVERSION_TYPE_LIST,
    )
except ImportError:
    from type_convertor import (
        java_type_converter,
        js_type_converter,
        get_possible_answer_type,
        JAVA_TYPE_CONVERSION,
        JS_TYPE_CONVERSION,
        PYTHON_TYPE_MAPPING,
        PYTHON_NESTED_TYPE_CHECK_LIST,
        NESTED_CONVERSION_TYPE_LIST,
    )

logger = logging.getLogger(__name__)


# ============================================================================
# Language Enum
# ============================================================================


class Language(Enum):
    """Programming language for function call evaluation."""
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"


# ============================================================================
# String Standardization
# ============================================================================


def standardize_string(input_string: str) -> str:
    """
    Standardize strings for comparison.

    Removes spaces, commas, periods, forward slashes, hyphens,
    underscores, asterisks, and carets. Converts to lowercase and
    replaces single quotes with double quotes.

    This handles cases like "April 1, 2024" vs "April 1,2024" vs "April 1 2024".

    Args:
        input_string: String to standardize

    Returns:
        Standardized string
    """
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


# ============================================================================
# Value Checkers
# ============================================================================


def string_checker(param: str, model_output: str, possible_answer: list) -> dict:
    """
    Check if a string value matches one of the possible answers.

    Uses standardize_string() for case-insensitive, punctuation-normalized comparison.

    Args:
        param: Parameter name
        model_output: Model's string value
        possible_answer: List of possible answer values

    Returns:
        Result dict with 'valid', 'error', 'error_type' keys
    """
    standardize_possible_answer = []
    standardize_model_output = standardize_string(model_output)

    for i in range(len(possible_answer)):
        if isinstance(possible_answer[i], str):
            standardize_possible_answer.append(standardize_string(possible_answer[i]))

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. "
                f"Expected one of {possible_answer}. Case insensitive."
            ],
            "error_type": "value_error:string",
        }

    return {"valid": True, "error": []}


def list_checker(param: str, model_output: list, possible_answer: list) -> dict:
    """
    Check if a list value matches one of the possible answers.

    Standardizes string elements before comparison.

    Args:
        param: Parameter name
        model_output: Model's list value
        possible_answer: List of possible answer lists

    Returns:
        Result dict with 'valid', 'error', 'error_type' keys
    """
    # Convert tuple to list
    standardize_model_output = list(model_output)

    # Standardize string elements
    for i in range(len(standardize_model_output)):
        if isinstance(standardize_model_output[i], str):
            standardize_model_output[i] = standardize_string(model_output[i])

    # Standardize possible answers
    standardize_possible_answer = []
    for i in range(len(possible_answer)):
        standardize_possible_answer.append([])
        for j in range(len(possible_answer[i])):
            if isinstance(possible_answer[i][j], str):
                standardize_possible_answer[i].append(
                    standardize_string(possible_answer[i][j])
                )
            else:
                standardize_possible_answer[i].append(possible_answer[i][j])

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. "
                f"Expected one of {possible_answer}."
            ],
            "error_type": "value_error:list/tuple",
        }

    return {"valid": True, "error": []}


def dict_checker(param: str, model_output: dict, possible_answers: list) -> dict:
    """
    Check if a dict value matches one of the possible answers.

    Validates:
    - All keys in model_output are in possible_answer
    - All values match possible answer values
    - All required keys are present

    Args:
        param: Parameter name
        model_output: Model's dict value
        possible_answers: List of possible answer dicts

    Returns:
        Result dict with 'valid', 'error', 'error_type' keys
    """
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}

    for i in range(len(possible_answers)):
        if possible_answers[i] == "":
            continue

        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        possible_answer = possible_answers[i]

        # Check all keys in model_output
        for key, value in model_output.items():
            if key not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

            # Standardize value if it's a string
            standardize_value = value
            if isinstance(value, str):
                standardize_value = standardize_string(value)

            # Standardize possible answer values
            standardize_possible_answer = []
            for j in range(len(possible_answer[key])):
                if isinstance(possible_answer[key][j], str):
                    standardize_possible_answer.append(
                        standardize_string(possible_answer[key][j])
                    )
                else:
                    standardize_possible_answer.append(possible_answer[key][j])

            if standardize_value not in standardize_possible_answer:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {repr(key)}: {repr(value)}. "
                    f"Expected one of {standardize_possible_answer}."
                )
                result["error_type"] = "value_error:dict_value"
                flag = False
                break

        # Check all required keys are present
        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["valid"] = False
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

        if flag:
            return {"valid": True, "error": []}

    return result


def list_dict_checker(param: str, model_output: list, possible_answers: list) -> dict:
    """
    Check if a list of dicts matches one of the possible answers.

    The order of dictionaries in the list must match the order of possible answers.
    Each dictionary is validated using dict_checker().

    Args:
        param: Parameter name
        model_output: Model's list of dicts
        possible_answers: List of possible answer lists of dicts

    Returns:
        Result dict with 'valid', 'error', 'error_type' keys
    """
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}

    for answer_index in range(len(possible_answers)):
        flag = True

        # Check dictionary count matches
        if len(model_output) != len(possible_answers[answer_index]):
            result["valid"] = False
            result["error"] = ["Wrong number of dictionaries in the list."]
            result["error_type"] = "value_error:list_dict_count"
            flag = False
            continue

        # Check each dictionary
        for dict_index in range(len(model_output)):
            result = dict_checker(
                param,
                model_output[dict_index],
                [possible_answers[answer_index][dict_index]],
            )
            if not result["valid"]:
                flag = False
                break

        if flag:
            return {"valid": True, "error": []}

    return result


# ============================================================================
# Type Checker
# ============================================================================


def type_checker(
    param: str,
    value: Any,
    possible_answer: list,
    expected_type_description: str,
    expected_type_converted: type,
    nested_type_converted: type | None,
) -> dict:
    """
    Check if a value matches the expected type.

    Supports:
    - Simple type checking (int, float, str, bool)
    - Nested type checking (one level deep for arrays/tuples)
    - Variable detection (if value type != expected but == possible_answer type)

    Note: This only supports nested type checking for one level deep.
    Recursive type checking for nested types is not implemented.

    Args:
        param: Parameter name
        value: Value to check
        possible_answer: List of possible answer values
        expected_type_description: Expected type as string (e.g., "array", "float")
        expected_type_converted: Expected type as Python type
        nested_type_converted: For collections, the inner type as Python type

    Returns:
        Result dict with 'valid', 'error', 'is_variable', 'error_type' keys
    """
    result = {
        "valid": True,
        "error": [],
        "is_variable": False,
        "error_type": "type_error:simple",
    }

    is_variable = False

    # Check for variable usage: use possible_answer type as expected type
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if possible_answer_type != expected_type_converted:
            is_variable = True

    # Value matches expected type
    if type(value) == expected_type_converted:
        # No nested type checking needed
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        else:
            # Nested type checking (one level deep)
            for possible_answer_item in possible_answer:
                flag = True
                if isinstance(possible_answer_item, list):
                    for value_item in value:
                        checker_result = type_checker(
                            param,
                            value_item,
                            possible_answer_item,
                            str(nested_type_converted),
                            nested_type_converted,
                            None,
                        )
                        if not checker_result["valid"]:
                            flag = False
                            break

                if flag:
                    return {"valid": True, "error": [], "is_variable": is_variable}

            result["valid"] = False
            result["error"] = [
                f"Nested type checking failed for parameter {repr(param)}. "
                f"Expected outer type {expected_type_description} with inner type "
                f"{str(nested_type_converted)}. Parameter value: {repr(value)}."
            ]
            result["error_type"] = "type_error:nested"
            return result

    # Value doesn't match expected type - check for variable usage
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if type(value) == possible_answer_type:
            result["is_variable"] = True
            return result

    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {repr(param)}. "
        f"Expected type {expected_type_description}, got {type(value).__name__}. "
        f"Parameter value: {repr(value)}."
    )
    result["error_type"] = "type_error:simple"
    return result


# ============================================================================
# Helper Functions
# ============================================================================


def find_description(func_descriptions: list | dict, name: str) -> dict | None:
    """
    Find a function description by name.

    Args:
        func_descriptions: List of function descriptions or single dict
        name: Function name to find

    Returns:
        Function description dict or None
    """
    if isinstance(func_descriptions, list):
        for func_description in func_descriptions:
            if func_description["name"] == name:
                return func_description
        return None
    else:
        # Single dict
        return func_descriptions


def convert_func_name(function_name: str, model_name: str) -> str:
    """
    Convert function name if needed for model compatibility.

    Some models (OpenAI, Mistral, Google) don't support "." in function names,
    so they replace it with "_". This function converts it back for comparison.

    Args:
        function_name: Function name from model output
        model_name: Model identifier

    Returns:
        Converted function name
    """
    # For now, return as-is. In upstream, this checks MODEL_CONFIG_MAPPING
    # for underscore_to_dot setting and converts dots to underscores or vice versa.
    return function_name


# ============================================================================
# Function Checkers
# ============================================================================


def simple_function_checker(
    func_description: dict,
    model_output: dict,
    possible_answer: dict,
    language: Language,
    model_name: str = "",
) -> dict:
    """
    Validate a single function call.

    Checks:
    1. Function name matches
    2. All required parameters are present
    3. All parameters are valid (name, type, value)
    4. Optional parameters handled correctly

    Args:
        func_description: Function description with name and parameters
        model_output: Model output dict: {function_name: {param: value, ...}}
        possible_answer: Possible answers: {function_name: {param: [values, ...], ...}}
        language: Programming language (Python, Java, JavaScript)
        model_name: Model name for function name conversion

    Returns:
        Result dict with 'valid', 'error', 'error_type' keys
    """
    possible_answer = list(possible_answer.values())[0]

    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"].get("required", [])

    result = {
        "valid": True,
        "error": [],
        "error_type": "simple_function_checker:unclear",
    }

    func_name = convert_func_name(func_name, model_name)

    # Check function name
    if func_name not in model_output:
        result["valid"] = False
        result["error"].append(
            f"Function name {repr(func_name)} not found in model output."
        )
        result["error_type"] = "simple_function_checker:wrong_func_name"
        return result

    model_params = model_output[func_name]

    # Check required parameters
    for param in required_params:
        if param not in model_params:
            result["valid"] = False
            result["error"].append(f"Missing required parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:missing_required"
            return result

    # Validate each parameter
    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            result["valid"] = False
            result["error"].append(f"Unexpected parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:unexpected_param"
            return result

        full_param_details = param_details[param]
        expected_type_description = full_param_details["type"]
        is_variable = False
        nested_type_converted = None

        # Language-specific type handling
        if language == Language.JAVA:
            expected_type_converted = JAVA_TYPE_CONVERSION[expected_type_description]

            if expected_type_description in JAVA_TYPE_CONVERSION:
                if type(value) != str:
                    result["valid"] = False
                    result["error"].append(
                        f"Incorrect type for parameter {repr(param)}. "
                        f"Expected type String, got {type(value).__name__}. "
                        f"Parameter value: {repr(value)}."
                    )
                    result["error_type"] = "type_error:java"
                    return result

                if expected_type_description in NESTED_CONVERSION_TYPE_LIST:
                    nested_type = param_details[param]["items"]["type"]
                    nested_type_converted = JAVA_TYPE_CONVERSION[nested_type]
                    value = java_type_converter(
                        value, expected_type_description, nested_type
                    )
                else:
                    value = java_type_converter(value, expected_type_description)

        elif language == Language.JAVASCRIPT:
            expected_type_converted = JS_TYPE_CONVERSION[expected_type_description]

            if expected_type_description in JS_TYPE_CONVERSION:
                if type(value) != str:
                    result["valid"] = False
                    result["error"].append(
                        f"Incorrect type for parameter {repr(param)}. "
                        f"Expected type String, got {type(value).__name__}. "
                        f"Parameter value: {repr(value)}."
                    )
                    result["error_type"] = "type_error:js"
                    return result

                if expected_type_description in NESTED_CONVERSION_TYPE_LIST:
                    nested_type = param_details[param]["items"]["type"]
                    nested_type_converted = JS_TYPE_CONVERSION[nested_type]
                    value = js_type_converter(value, expected_type_description, nested_type)
                else:
                    value = js_type_converter(value, expected_type_description)

        elif language == Language.PYTHON:
            expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
            if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
                nested_type = param_details[param]["items"]["type"]
                nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]

        else:
            raise ValueError(f"Unsupported language: {language}")

        # Convert tuple to list for comparison
        if expected_type_description == "tuple" and isinstance(value, tuple):
            value = list(value)

        # Allow Python auto conversion from int to float
        if language == Language.PYTHON and expected_type_description == "float" and isinstance(value, int):
            value = float(value)

        # Type checking
        type_check_result = type_checker(
            param,
            value,
            possible_answer[param],
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
        )
        is_variable = type_check_result["is_variable"]
        if not type_check_result["valid"]:
            return type_check_result

        # Skip special handling if value is a variable
        if not is_variable:
            # Special handling for dicts
            if expected_type_converted == dict:
                result = dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            # Special handling for list of dicts
            elif expected_type_converted == list and nested_type_converted == dict:
                result = list_dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            # Special handling for strings
            elif expected_type_converted == str:
                result = string_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            # Special handling for lists/tuples
            elif expected_type_converted == list:
                result = list_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

        # Check if value is in possible answers
        if value not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Invalid value for parameter {repr(param)}: {repr(value)}. "
                f"Expected one of {possible_answer[param]}."
            )
            result["error_type"] = "value_error:others"
            return result

    # Check optional parameters
    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Optional parameter {repr(param)} not provided and not marked as optional."
            )
            result["error_type"] = "simple_function_checker:missing_optional"
            return result

    return result


def parallel_function_checker_no_order(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
    language: Language,
    model_name: str = "",
) -> dict:
    """
    Validate parallel function calls (order-insensitive).

    Each function call in model_output must match exactly one function call
    in possible_answers, but order doesn't matter.

    Args:
        func_descriptions: List of function descriptions
        model_output: List of model function call dicts
        possible_answers: List of possible answer dicts
        language: Programming language
        model_name: Model name for function name conversion

    Returns:
        Result dict with 'valid', 'error', 'error_type' keys
    """
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }

    matched_indices = []

    # Match each possible answer to a model output
    for i in range(len(possible_answers)):
        func_name_expected = list(possible_answers[i].keys())[0]
        func_description = find_description(func_descriptions, func_name_expected)

        all_errors = []

        for index in range(len(model_output)):
            if index in matched_indices:
                continue

            result = simple_function_checker(
                func_description,
                model_output[index],
                possible_answers[i],
                language,
                model_name,
            )

            if result["valid"]:
                matched_indices.append(index)
                break
            else:
                all_errors.append(
                    {
                        f"Model Result Index {index}": {
                            "sub_error": result["error"],
                            "sub_error_type": result["error_type"],
                            "model_output_item": model_output[index],
                            "possible_answer_item": possible_answers[i],
                        }
                    }
                )

        if not result["valid"]:
            considered_indices = [i for i in range(len(model_output)) if i not in matched_indices]
            all_errors.insert(
                0,
                f"Could not find a matching function among index {considered_indices} "
                f"of model output for index {i} of possible answers.",
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "parallel_function_checker_no_order:cannot_find_match",
            }

    return {"valid": True, "error": []}


def parallel_function_checker_enforce_order(
    func_descriptions: list,
    model_output: list,
    possible_answers: dict,
    language: Language,
    model_name: str = "",
) -> dict:
    """
    Validate parallel function calls (order-sensitive).

    Each function call in model_output must match the corresponding
    function call in possible_answers at the same index.

    Args:
        func_descriptions: List of function descriptions
        model_output: List of model function call dicts
        possible_answers: Dict of possible answer dicts
        language: Programming language
        model_name: Model name for function name conversion

    Returns:
        Result dict with 'valid', 'error', 'error_type' keys
    """
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_enforce_order:wrong_count",
        }

    func_name_list = list(possible_answers.keys())
    possible_answers_list = [{key: value} for key, value in possible_answers.items()]

    for i in range(len(possible_answers_list)):
        func_description = find_description(func_descriptions, func_name_list[i])

        result = simple_function_checker(
            func_description,
            model_output[i],
            possible_answers_list[i],
            language,
            model_name,
        )
        if not result["valid"]:
            return result

    return {"valid": True, "error": []}


def multiple_function_checker(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
    language: Language,
    model_name: str = "",
) -> dict:
    """
    Validate multiple (sequential) function calls.

    All calls must be for the same function. Only validates the first call.

    Args:
        func_descriptions: List of function descriptions
        model_output: List of model function call dicts
        possible_answers: List of possible answer dicts
        language: Programming language
        model_name: Model name for function name conversion

    Returns:
        Result dict with 'valid', 'error', 'error_type' keys
    """
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "multiple_function_checker:wrong_count",
        }

    # Only validate the first function call
    func_name_expected = list(possible_answers[0].keys())[0]
    func_description = find_description(func_descriptions, func_name_expected)

    return simple_function_checker(
        func_description,
        model_output[0],
        possible_answers[0],
        language,
        model_name,
    )


# ============================================================================
# Main AST Checker Entry Point
# ============================================================================


def ast_checker(
    func_description: list | dict,
    model_output: list,
    possible_answer: list,
    language: Language,
    test_category: str,
    model_name: str = "",
) -> dict:
    """
    Main entry point for AST-based evaluation.

    Routes to the appropriate checker based on test category:
    - "parallel" categories: parallel_function_checker_no_order
    - "multiple" categories: multiple_function_checker
    - others: simple_function_checker

    Args:
        func_description: Function description(s) - list or dict
        model_output: Model output as list of function call dicts
        possible_answer: Possible answers as list
        language: Programming language (Python, Java, JavaScript)
        test_category: Test category name
        model_name: Model name for function name conversion

    Returns:
        Result dict with 'valid', 'error', 'error_type' keys
    """
    if "parallel" in test_category:
        return parallel_function_checker_no_order(
            func_description, model_output, possible_answer, language, model_name
        )

    elif "multiple" in test_category:
        return multiple_function_checker(
            func_description, model_output, possible_answer, language, model_name
        )

    else:
        # Single function call
        if len(model_output) != 1:
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "simple_function_checker:wrong_count",
            }

        return simple_function_checker(
            func_description[0], model_output[0], possible_answer[0], language, model_name
        )


# ============================================================================
# Testing
# ============================================================================


if __name__ == "__main__":
    # Test string standardization
    print("Testing string standardization...")
    assert standardize_string("April 1, 2024") == standardize_string("april12024")
    assert standardize_string("Hello World") == "helloworld"
    print("  String standardization tests passed!")

    # Test type checker
    print("Testing type checker...")
    result = type_checker("test", 42, [42], "integer", int, None)
    assert result["valid"] == True

    result = type_checker("test", "42", [42], "integer", int, None)
    assert result["valid"] == False
    print("  Type checker tests passed!")

    print("\nAll AST evaluation tests passed!")
