# BFCL v4 Evaluation: Alignment with Upstream Reference Implementation

## Context

The current inspect_ai BFCL v4 implementation uses **direct structural comparison** for evaluating function calls, while the upstream Berkeley implementation uses **AST-based evaluation** with comprehensive type checking and validation. This misalignment means the inspect_ai version may produce different scores than the official BFCL v4 leaderboard.

---

## Phase 1: Current Implementation Summary

### File: `scorer.py` (Current)

The current implementation (`bfcl_v4_scorer`) uses direct dictionary comparison:

1. **Tool Call Extraction**: Extracts tool calls from `ChatMessageAssistant` messages
2. **Format Conversion**: Converts inspect_ai `ToolCall` objects to simple dict format:
   ```python
   {"function": "func_name", "arguments": {"param1": "value1", ...}}
   ```
3. **Comparison Logic**: Direct dict comparison with:
   - `params_match()`: Checks parameter dictionaries match
   - `values_match()`: Checks if values match (handles lists of possible values)
   - `tool_calls_match()`: Order-insensitive matching for parallel calls

### Current Normalization

The `normalize_value()` function handles:
- Boolean string conversion (`"true"`/`"false"` → `True`/`False`)
- Number passthrough (int, float)
- List/dict recursion
- String whitespace trimming

### Current Limitations

| Feature | Current Status |
|---------|---------------|
| **AST Parsing** | Not implemented - uses direct dict comparison |
| **Type Checking** | Basic - only checks bool/int/float/str types |
| **Variable Detection** | No - cannot distinguish variables from literals |
| **String Standardization** | Basic - only whitespace trimming |
| **Error Types** | Generic - only "C" (Correct) or "I" (Invalid) |
| **Java Support** | Basic - dict parsing, no numeric suffix handling |
| **JavaScript Support** | Basic - dict parsing, no constructor notation |
| **Nested Types** | Basic - list/dict recursion only |

---

## Phase 2: Upstream Reference Summary

### File: `upstream/.../ast_checker.py` (Reference)

The upstream implementation (`ast_checker`) uses comprehensive AST-based evaluation:

#### 2.1 Parsing Method

**Python**: Direct dict comparison (similar to current)

**Java**: TreeSitter AST parsing via regex patterns
- `new int[]{1, 2, 3}` → `[1, 2, 3]`
- `new ArrayList<>(Arrays.asList("a", "b"))` → `["a", "b"]`
- `123L` → `123`, `3.14f` → `3.14`

**JavaScript**: TreeSitter AST parsing via regex patterns
- `[1, 2, 3]` or `new Array(1, 2, 3)` → `[1, 2, 3]`
- `{"key": "value"}` or `new Object()` → `{"key": "value"}`

#### 2.2 Type Checking Mechanism

**Python Type Mapping**:
```python
PYTHON_TYPE_MAPPING = {
    "string": str, "integer": int, "float": float,
    "boolean": bool, "array": list, "tuple": list,
    "dict": dict, "any": str,
}
```

**Java Type Mapping**:
```python
JAVA_TYPE_CONVERSION = {
    "byte": int, "short": int, "integer": int,
    "float": float, "double": float, "long": int,
    "boolean": bool, "char": str,
    "Array": list, "ArrayList": list, "HashMap": dict,
    "String": str, "any": str,
}
```

**JavaScript Type Mapping**:
```python
JS_TYPE_CONVERSION = {
    "String": str, "integer": int, "float": float,
    "Bigint": int, "Boolean": bool,
    "dict": dict, "array": list, "any": str,
}
```

#### 2.3 Variable Detection

The `get_possible_answer_type()` function detects when a model uses a variable instead of a literal value:
```python
def get_possible_answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != "":  # Optional parameter
            return type(answer)
    return None
```

If the value's type differs from expected but matches the possible_answer type, it's flagged as `is_variable=True`.

#### 2.4 String Standardization

The `standardize_string()` function normalizes strings for comparison:
```python
def standardize_string(input_string: str):
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')
```

This handles cases like "April 1, 2024" vs "April 1, 2024" vs "April 1 2024".

#### 2.5 Error Classification System

**Function Errors**:
- `simple_function_checker:wrong_func_name`
- `simple_function_checker:missing_required`
- `simple_function_checker:unexpected_param`
- `simple_function_checker:missing_optional`
- `parallel_function_checker_no_order:cannot_find_match`
- `multiple_function_checker:wrong_count`

**Type Errors**:
- `type_error:simple`
- `type_error:java`
- `type_error:js`
- `type_error:nested`

**Value Errors**:
- `value_error:string`
- `value_error:list/tuple`
- `value_error:dict_key`
- `value_error:dict_value`
- `value_error:list_dict_count`
- `value_error:others`

---

## Phase 3: Gap Analysis

### Missing Features

| Feature | Current | Upstream | Impact |
|---------|---------|----------|--------|
| **AST Parsing (Java)** | No | Yes (regex) | Java collections not parsed correctly |
| **AST Parsing (JS)** | No | Yes (regex) | JS constructor notation not parsed |
| **Variable Detection** | No | Yes | Variable usage treated as error |
| **String Standardization** | Basic (trim) | Full (punctuation + case) | String comparison too strict |
| **Nested Type Checking** | Basic | One-level recursive | Nested arrays/tuples not validated |
| **Error Classification** | Generic (C/I) | Detailed (20+ types) | No diagnostic info |
| **Numeric Suffixes** | No | Yes (123L, 3.14f) | Java numeric literals fail |
| **Boolean Literals** | Yes | Yes | ✅ Supported |

### Scoring Differences

1. **Java Collections**: Current implementation may fail on `new ArrayList<>(...)` syntax
2. **JavaScript Constructor Notation**: Current implementation may fail on `new Array()` syntax
3. **String Case Sensitivity**: Current is case-sensitive, upstream is not
4. **Variable Usage**: Current treats variables as errors, upstream detects them
5. **Diagnostic Value**: Upstream provides detailed error types for analysis

### Language Support Gaps

| Language | Current | Upstream | Gap |
|----------|---------|----------|-----|
| Python | ✅ Full | ✅ Full | None |
| Java | ⚠️ Basic | ✅ Full | Collections, numeric suffixes |
| JavaScript | ⚠️ Basic | ✅ Full | Constructor notation, arrays |

---

## Implementation Plan

### Phase 1: ✅ Documentation
- [x] Create `note.md` with current vs upstream comparison
- [x] Document gap analysis

### Phase 2: Enhanced Evaluation Implementation
- [ ] Create `type_convertor.py` with Java/JS type converters
- [ ] Create `ast_eval.py` with AST checker ported from upstream

### Phase 3: Scorer Integration
- [ ] Add `bfcl_v4_ast_scorer()` to `scorer.py`
- [ ] Support detailed error type reporting

### Phase 4: Task Updates
- [ ] Update `bfcl_v4.py` to use AST-based scorer
- [ ] Bump version to 2.0.0

### Phase 5: Verification
- [ ] Unit tests for type converters
- [ ] Integration tests comparing old vs new scorers
- [ ] Validation against upstream leaderboard (if available)

---

## References

- Upstream AST Checker: `/mnt/data1/workspace/djs/eval-poc-with-salt/gorilla/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py`
- Type Converters: `/mnt/data1/workspace/djs/eval-poc-with-salt/gorilla/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/type_convertor/`
- Type Mappings: `/mnt/data1/workspace/djs/eval-poc-with-salt/gorilla/berkeley-function-call-leaderboard/bfcl_eval/constants/type_mappings.py`
