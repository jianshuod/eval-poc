"""
BFCL v4 Type Converters

Ported from upstream Berkeley Function Call Leaderboard v4:
https://github.com/ShishirPatil/gorilla/tree/main/bfcl_eval/eval_checker/ast_eval/type_convertor

This module provides language-specific type converters for parsing and validating
function call arguments in Python, Java, and JavaScript formats.
"""

import re
from typing import Any

# ============================================================================
# Type Mappings (from upstream constants/type_mappings.py)
# ============================================================================

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

# Types that need recursive nested type checking
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]

# Types that represent nested collections across languages
NESTED_CONVERSION_TYPE_LIST = ["Array", "ArrayList", "array"]

JAVA_TYPE_CONVERSION = {
    "byte": int,
    "short": int,
    "integer": int,
    "float": float,
    "double": float,
    "long": int,
    "boolean": bool,
    "char": str,
    "Array": list,
    "ArrayList": list,
    "Set": set,
    "HashMap": dict,
    "Hashtable": dict,
    "Queue": list,
    "Stack": list,
    "String": str,
    "any": str,
}

JS_TYPE_CONVERSION = {
    "String": str,
    "integer": int,
    "float": float,
    "Bigint": int,
    "Boolean": bool,
    "dict": dict,
    "array": list,
    "any": str,
}


# ============================================================================
# Java Type Converter
# ============================================================================


def java_type_converter(value: str, expected_type: str, nested_type: str | None = None) -> Any:
    """
    Convert a Java string value to its Python representation.

    Handles:
    - Primitive types: byte, short, int, float, double, long, boolean, char
    - Collections: Array, ArrayList, HashMap
    - Numeric suffixes: 123L, 3.14f

    Args:
        value: String value from model output
        expected_type: Expected Java type (e.g., "integer", "float", "ArrayList")
        nested_type: For collections, the type of elements (e.g., "String", "integer")

    Returns:
        Converted Python value (int, float, str, bool, list, dict)

    Raises:
        ValueError: If type is not supported
    """
    if expected_type not in JAVA_TYPE_CONVERSION:
        raise ValueError(f"Unsupported Java type: {expected_type}")

    # Primitive numeric types
    if expected_type in ("byte", "short", "integer"):
        if not re.match(r"^-?\d+$", value):
            return str(value)  # Default to string if doesn't match pattern
        return int(value)

    elif expected_type == "float":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value):
            return str(value)
        return float(re.sub(r"[fF]$", "", value))

    elif expected_type == "double":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", value):
            return str(value)
        return float(value)

    elif expected_type == "long":
        if not re.match(r"^-?\d+[lL]$", value):
            return str(value)
        return int(re.sub(r"[lL]$", "", value))

    elif expected_type == "boolean":
        if value not in ["true", "false"]:
            return str(value)
        return value == "true"

    elif expected_type == "char":
        # Single quoted character: 'a'
        if not re.match(r"^\'[^\']\'$", value):
            return str(value)
        return value[1]  # Return the character without quotes

    # Collection types
    elif expected_type == "Array" or expected_type == "ArrayList":
        return _parse_java_collection(value, expected_type, nested_type)

    elif expected_type == "HashMap":
        return _parse_java_collection(value, expected_type, nested_type)

    elif expected_type == "Set":
        raise NotImplementedError("Set conversion is not implemented")
    elif expected_type == "Hashtable":
        raise NotImplementedError("Hashtable conversion is not implemented")
    elif expected_type in ("Queue", "Stack"):
        raise NotImplementedError(f"{expected_type} conversion is not implemented")

    elif expected_type == "String" or expected_type == "any":
        return str(value)

    else:
        raise ValueError(f"Unsupported Java type: {expected_type}")


def _parse_java_collection(input_str: str, type_str: str, nested_type: str | None = None) -> list | dict:
    """Parse a Java collection (Array, ArrayList, HashMap) from string representation."""
    if type_str == "ArrayList":
        return _parse_arraylist(input_str, nested_type)
    elif type_str == "Array":
        return _parse_java_array(input_str, nested_type)
    elif type_str == "HashMap":
        return _parse_hashmap(input_str)
    else:
        raise ValueError(f"Unsupported Java collection type: {type_str}")


def _parse_arraylist(input_str: str, nested_type: str | None = None) -> list:
    """Parse Java ArrayList: new ArrayList<>(Arrays.asList(...)) or new ArrayList<>() {{ add(...) }}"""
    # Pattern 1: new ArrayList<>(Arrays.asList("a", "b"))
    match_aslist = re.search(
        r"new\s+ArrayList<\w*>\(Arrays\.asList\((.+?)\)\)", input_str
    )
    if match_aslist:
        elements_str = match_aslist.group(1)
        elements = []
        for element_str in elements_str.split(","):
            element_str = element_str.strip()
            if nested_type == "char":
                element = element_str[1:-1]  # Remove single quotes
            elif nested_type == "String":
                element = element_str[1:-1]  # Remove double quotes
            else:
                element = (
                    java_type_converter(element_str, nested_type)
                    if nested_type
                    else _parse_java_value(element_str)
                )
            elements.append(element)
        return elements

    # Pattern 2: new ArrayList<>() {{ add("a"); add("b") }}
    match_add = re.search(
        r"new\s+ArrayList<\w*>\(\)\s*\{\{\s*(.+?)\s*\}\}", input_str, re.DOTALL
    )
    if match_add:
        adds_str = match_add.group(1)
        elements = []
        matches = re.findall(r"add\((.+?)\)", adds_str)
        for match in matches:
            value_str = match.strip()
            if nested_type == "char":
                value = value_str[1:-1]
            elif nested_type == "String":
                value = value_str[1:-1]
            else:
                value = (
                    java_type_converter(value_str, nested_type)
                    if nested_type
                    else _parse_java_value(value_str)
                )
            elements.append(value)
        return elements

    # Pattern 3: Empty ArrayList
    match_empty = re.search(r"new\s+ArrayList<\w*>\(\)", input_str)
    if match_empty:
        return []

    return input_str  # Default to string


def _parse_java_array(input_str: str, nested_type: str | None = None) -> list:
    """Parse Java array: new int[]{1, 2, 3}"""
    match = re.search(r"new\s+\w+\[\]\s*\{(.*?)\}", input_str)
    if match:
        elements_str = match.group(1)
        if not elements_str.strip():
            return []

        if nested_type:
            elements = [
                java_type_converter(x.strip(), nested_type)
                for x in elements_str.split(",")
                if x.strip()
            ]
        else:
            elements = [
                _parse_java_value(x.strip())
                for x in elements_str.split(",")
                if x.strip()
            ]
        return elements
    else:
        return input_str


def _parse_hashmap(input_str: str) -> dict:
    """Parse Java HashMap: new HashMap<>() {{ put("key", "value") }}"""
    elements = {}
    match = re.search(
        r"new\s+HashMap<.*?>\s*\(\)\s*\{\s*\{?\s*(.*?)\s*\}?\s*\}", input_str, re.DOTALL
    )
    if match:
        puts_str = match.group(1)
        if puts_str.strip():
            matches = re.findall(r"put\(\"(.*?)\",\s*(.*?)\)", puts_str)
            for match in matches:
                key = match[0]
                value = _parse_java_value(match[1].strip())
                elements[key] = value
        return elements

    # Empty HashMap
    match_empty = re.search(r"new\s+HashMap<.*?>\s*\(\)", input_str)
    if match_empty:
        return {}

    return input_str


def _parse_java_value(value_str: str) -> Any:
    """Parse a Java value without type information."""
    value_str = value_str.strip()

    if value_str == "true":
        return True
    elif value_str == "false":
        return False
    elif value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]
    elif re.match(r"^-?\d+[lL]$", value_str):
        return int(value_str[:-1])
    elif re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value_str):
        return float(re.sub(r"[fF]$", "", value_str))
    else:
        # Try to parse as int, then float, then default to string
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str


# ============================================================================
# JavaScript Type Converter
# ============================================================================


def js_type_converter(value: str, expected_type: str, nested_type: str | None = None) -> Any:
    """
    Convert a JavaScript string value to its Python representation.

    Handles:
    - Types: String, integer, float, Bigint, Boolean, dict, array
    - Constructor notation: new Array(), new Object()
    - Literal notation: [], {}

    Args:
        value: String value from model output
        expected_type: Expected JavaScript type (e.g., "String", "integer", "array")
        nested_type: For arrays, the type of elements

    Returns:
        Converted Python value (int, float, str, bool, list, dict)

    Raises:
        ValueError: If type is not supported
    """
    if expected_type not in JS_TYPE_CONVERSION:
        raise ValueError(f"Unsupported JavaScript type: {expected_type}")

    if expected_type == "String":
        # Single or double quoted string
        if not ((value.startswith('"') and value.endswith('"')) or
                (value.startswith("'") and value.endswith("'"))):
            return str(value)
        return value[1:-1]

    elif expected_type == "integer":
        if not re.match(r"^-?\d+$", value):
            return str(value)
        return int(value)

    elif expected_type == "float":
        if not re.match(r"^-?\d+(\.\d+)?$", value):
            return str(value)
        return float(value)

    elif expected_type == "Bigint":
        if not re.match(r"^-?\d+n$", value):
            return str(value)
        return int(value[:-1])

    elif expected_type == "Boolean":
        if value not in ["true", "false"]:
            return str(value)
        return value == "true"

    elif expected_type == "dict":
        return _parse_js_collection(value, "dict", nested_type)

    elif expected_type == "array":
        return _parse_js_collection(value, "array", nested_type)

    elif expected_type == "any":
        return str(value)

    else:
        raise ValueError(f"Unsupported JavaScript type: {expected_type}")


def _parse_js_collection(code: str, type_str: str, nested_type: str | None = None) -> list | dict:
    """Parse a JavaScript collection (array or object/dict)."""
    code = code.strip()

    if type_str == "array":
        return _parse_js_array(code, nested_type)
    elif type_str == "dict":
        return _parse_js_object(code)
    else:
        raise ValueError(f"Unsupported JavaScript collection type: {type_str}")


def _parse_js_array(code: str, nested_type: str | None = None) -> list:
    """Parse JavaScript array: [1, 2, 3] or new Array(1, 2, 3)"""
    code = code.strip()

    # Check for 2D array
    array_2d_pattern = r"\[\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\]|\bnew\s+Array\(\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\)"
    array_2d_match = re.match(array_2d_pattern, code)
    if array_2d_match:
        elements_str = array_2d_match.group(0)
        inner_arrays = re.findall(r"\[(.*?)\]", elements_str)
        elements = []
        for idx, inner_array_str in enumerate(inner_arrays):
            inner_array_str = inner_array_str.strip()
            if idx == 0 and inner_array_str.startswith("["):
                inner_array_str = inner_array_str[1:]
            inner_array_elements = [e.strip() for e in inner_array_str.split(",")]
            elements.append([_parse_js_value(e) for e in inner_array_elements])
        return elements

    # Check for 1D array
    array_pattern = r"\[(.*?)\]|\bnew\s+Array\((.*?)\)"
    array_match = re.match(array_pattern, code)
    if array_match:
        if array_match.group(1) is not None:
            elements_str = array_match.group(1).strip()
            elements = elements_str.split(",") if elements_str else []
        elif array_match.group(2) is not None:
            elements_str = array_match.group(2).strip()
            elements = elements_str.split(",") if elements_str else []
        else:
            elements = []

        if nested_type:
            elements = [
                (
                    js_type_converter(e.strip(), nested_type, "String")
                    if (e.strip().startswith("'") or e.strip().startswith('"'))
                    else js_type_converter(e.strip(), nested_type)
                )
                for e in elements
            ]
        else:
            elements = [_parse_js_value(e.strip()) for e in elements]
        return elements
    else:
        return code


def _parse_js_object(code: str) -> dict:
    """Parse JavaScript object: {"key": "value"} or {}"""
    code = code.strip()

    if code == "{}":
        return {}

    dict_pattern = r"\{(.*?)\}"
    dict_match = re.match(dict_pattern, code)
    if dict_match:
        try:
            content = dict_match.group(1)
            pairs = re.findall(r"([^:]+):\s*(.*?)(?:,\s*(?=[^,]+:)|$)", content)
            dictionary = {}
            for key, value in pairs:
                key = key.strip().strip("'\"")
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    dictionary[key] = _parse_js_array(value)
                elif value.startswith("{") and value.endswith("}"):
                    dictionary[key] = _parse_js_object(value)
                else:
                    dictionary[key] = _parse_js_value(value.strip("'\""))
            return dictionary
        except Exception:
            return code
    else:
        return code


def _parse_js_value(value_str: str) -> Any:
    """Parse a JavaScript value without type information."""
    value_str = value_str.strip()

    if value_str == "true":
        return True
    elif value_str == "false":
        return False
    elif (value_str.startswith('"') and value_str.endswith('"')) or \
         (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]
    else:
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str


# ============================================================================
# Python Type Helpers
# ============================================================================


def get_possible_answer_type(possible_answer: list) -> type | None:
    """
    Get the type of values in possible_answer list.

    Used for variable detection - if the model output type doesn't match
    the expected type but matches the possible_answer type, it's likely
    a variable reference.

    Args:
        possible_answer: List of possible answer values

    Returns:
        The type of non-empty values in the list, or None if all are empty
    """
    for answer in possible_answer:
        if answer != "":  # Optional parameter
            return type(answer)
    return None


# ============================================================================
# Testing
# ============================================================================


if __name__ == "__main__":
    # Test Java type converter
    print("Testing Java type converter...")

    assert java_type_converter("true", "boolean") == True
    assert java_type_converter("false", "boolean") == False
    assert java_type_converter("123", "integer") == 123
    assert java_type_converter("3.14f", "float") == 3.14
    assert java_type_converter("123L", "long") == 123
    assert java_type_converter("new int[]{1, 2, 3}", "Array") == [1, 2, 3]
    assert java_type_converter('new ArrayList<>(Arrays.asList("a", "b"))', "ArrayList") == ["a", "b"]
    assert java_type_converter('new HashMap<String, String>() {{ put("key", "value"); }}', "HashMap") == {"key": "value"}

    print("  All Java tests passed!")

    # Test JavaScript type converter
    print("Testing JavaScript type converter...")

    assert js_type_converter("true", "Boolean") == True
    assert js_type_converter("false", "Boolean") == False
    assert js_type_converter("123", "integer") == 123
    assert js_type_converter("3.14", "float") == 3.14
    assert js_type_converter("123n", "Bigint") == 123
    assert js_type_converter("[1, 2, 3]", "array") == [1, 2, 3]
    assert js_type_converter("new Array(1, 2, 3)", "array") == [1, 2, 3]
    assert js_type_converter("{'key': 'value'}", "dict") == {"key": "value"}

    print("  All JavaScript tests passed!")

    print("\nAll type converter tests passed!")
