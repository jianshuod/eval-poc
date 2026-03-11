# BFCL v4: Berkeley Function Call Leaderboard v4

Comprehensive function calling benchmark evaluating LLM capability to generate correct function calls with proper parameters. BFCL v4 combines multiple releases into a unified evaluation framework.

## BFCL v4 Single-Turn vs BFCL v2

**Important:** BFCL v4 single-turn split is NOT the same as BFCL v2.

| Aspect | BFCL V2 | BFCL v4 Single-Turn |
|--------|---------|---------------------|
| **Focus** | Live data only | Live + Non-Live combined |
| **Non-Live categories** | ❌ Not included | ✅ simple_python, simple_java, simple_javascript, multiple, parallel, parallel_multiple, irrelevance |
| **Live categories** | ✅ All | ✅ All (including live_relevance) |
| **Test cases** | ~2,251 | ~3,399 |
| **Multi-turn** | ❌ No | ❌ Excluded |
| **Agentic** | ❌ No | ❌ Excluded |

**BFCL v4 single-turn = V1 (non-live) + V2 (live)**

## Version Evolution

| Version | Key Addition | Test Cases |
|---------|--------------|------------|
| **V1** | Foundation (Python + non-Python) | ~1,150 |
| **V2** | Live data (enterprise/OSS) | +2,251 |
| **V3** | Multi-turn conversations | +800 |
| **V4** | Agentic (Web Search + Memory) | +265 |

## Category Structure

### Single-Turn Categories (V1 + V2)

#### 💡 Live Categories (from V2, ~1,367 test cases)

Real-world function calls from enterprise and OSS contributions:

| Category | Tests | Description |
|----------|-------|-------------|
| `live_simple` | 258 | Live simple function calls |
| `live_multiple` | 1053 | Live multiple function calls (dependent functions) |
| `live_parallel` | 16 | Live parallel function calls (independent functions) |
| `live_parallel_multiple` | 24 | Live parallel multiple function calls |
| `live_relevance` | 16 | At least one function is relevant - don't check correctness |
| `live_irrelevance` | 882 | No functions relevant - none should be invoked (hallucination measurement) |

#### 📂 Non-Live Categories (from V1, ~1,150 test cases)

Curated function calls across multiple programming languages:

| Category | Tests | Description |
|----------|-------|-------------|
| `simple_python` | 400 | Python function calling |
| `simple_java` | 100 | Java function calling (uses HashMap, ArrayList, etc.) |
| `simple_javascript` | 50 | JavaScript function calling (uses new Array(), etc.) |
| `multiple` | 200 | Multiple function calling (dependent functions) |
| `parallel` | 200 | Parallel function calling (independent functions) |
| `parallel_multiple` | 200 | Parallel multiple function calling |
| `irrelevance` | 240 | Irrelevance detection - no functions should be invoked |

### Multi-Turn Categories (from V3, ~800 test cases)

Stateful conversations requiring tool use across multiple turns:

| Category | Tests | Description |
|----------|-------|-------------|
| `multi_turn_base` | 200 | Multi-turn baseline tasks |
| `multi_turn_miss_func` | 200 | Missing function scenarios |
| `multi_turn_miss_param` | 200 | Missing parameter scenarios |
| `multi_turn_long_context` | 200 | Long context conversations |

### Agentic Categories (from V4, ~265 test cases)

Complex tasks requiring memory and web search:

| Category | Tests | Description |
|----------|-------|-------------|
| `memory` | 133 | Memory-augmented tasks (KV, Vector, Recursive Summary) |
| `web_search` | 132 | Multi-hop reasoning with search tools |

## Scoring

BFCL v4 uses AST-based evaluation aligned with the upstream Berkeley implementation:

- **"C" (Correct)**: Function name and all parameters match
- **"I" (Invalid)**: Function name or parameters don't match

### AST-Based Evaluation Features

1. **Comprehensive Type Checking**: Python, Java, JavaScript type handling
2. **String Standardization**: Case-insensitive, punctuation-normalized comparison
3. **Variable Detection**: Identifies when model uses variables vs literal values
4. **Nested Type Validation**: One-level deep type checking for arrays/objects
5. **Detailed Error Classification**: Specific error types for debugging

### Error Types

- `simple_function_checker:wrong_func_name` - Incorrect function name
- `simple_function_checker:missing_required` - Missing required parameter
- `simple_function_checker:unexpected_param` - Unexpected parameter
- `simple_function_checker:missing_optional` - Missing optional parameter
- `type_error:simple`, `type_error:java`, `type_error:js`, `type_error:nested` - Type mismatches
- `value_error:string`, `value_error:list/tuple`, `value_error:dict_key`, etc. - Value mismatches

### Tool Definition Type Mapping

For `simple_java` and `simple_javascript` categories, function definitions use language-specific types that are mapped to JSON Schema types:

| Java Type | JSON Schema Type | JavaScript Type | JSON Schema Type |
|-----------|------------------|-----------------|------------------|
| `HashMap<K,V>` | `object` | `Array` | `array` |
| `Map<K,V>` | `object` | `Object` | `object` |
| `ArrayList<E>` | `array` | `String` | `string` |
| `List<E>` | `array` | `Number` | `number` |
| `Set<E>` | `array` | `Boolean` | `boolean` |
| `String` | `string` | | |
| `Integer`, `int`, `Long` | `integer` | | |
| `Double`, `Float` | `number` | | |
| `Boolean`, `boolean` | `boolean` | | |

This mapping allows Java and JavaScript function definitions to be properly converted to inspect_ai's ToolInfo format.

## How AST Evaluation Works

### Java/JavaScript Type Handling

Java and JavaScript function calls are represented as strings in model output:

```java
// Java (in string format)
"HashMap<String, String> params = new HashMap<>()"
// Converted to: {"key1": "value1", "key2": "value2"}
```

```javascript
// JavaScript (in string format)
"new Array('item1', 'item2')"
// Converted to: ["item1", "item2"]
```

### String Comparison

Strings are standardized for comparison:
- Removes spaces, commas, periods, slashes, hyphens, underscores, asterisks, carets
- Converts to lowercase
- Replaces single quotes with double quotes

Example: `"April 1, 2024"` == `"april12024"` == `"April 1 2024"`

## Usage

### Installation

```bash
cd eval-poc
./run-eval.py --setup bfcl_v4
```

### Running evaluations

```bash
# Run single-turn (V1 + V2 combined)
./run-eval.py bfcl_v4_single_turn --model openai/gpt-4o

# Run individual categories
./run-eval.py bfcl_v4_simple_python --model openai/gpt-4o
./run-eval.py bfcl_v4_live_simple --model openai/gpt-4o

# Run with AST-based evaluation (upstream aligned)
./run-eval.py bfcl_v4_simple_python_ast --model openai/gpt-4o

# Run multi-turn categories
./run-eval.py bfcl_v4_multi_turn_base --model openai/gpt-4o

# Run agentic categories
./run-eval.py bfcl_v4_web_search --model openai/gpt-4o
./run-eval.py bfcl_v4_memory --model openai/gpt-4o

# Run complete BFCL v4 (all categories)
./run-eval.py bfcl_v4 --model openai/gpt-4o
```

### Limit samples for testing

```bash
./run-eval.py bfcl_v4_single_turn --model openai/gpt-4o --limit 10
```

## Parameters

### `bfcl_v4_single_turn()`

Main single-turn task combining V1 + V2 categories (~3,399 test cases).

**Categories Included:**
- Live: `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`, `live_relevance`, `live_irrelevance`
- Non-Live: `simple_python`, `simple_java`, `simple_javascript`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance`

**Excluded:**
- Multi-turn: `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`
- Agentic: `memory`, `web_search`

### `bfcl_v4()`

Complete BFCL v4 task including all categories (single-turn + multi-turn + agentic).

### Category-Specific Tasks

Each category has its own task for targeted evaluation:
- `bfcl_v4_simple_python`, `bfcl_v4_simple_java`, `bfcl_v4_simple_javascript`
- `bfcl_v4_multiple`, `bfcl_v4_parallel`, `bfcl_v4_parallel_multiple`
- `bfcl_v4_irrelevance`, `bfcl_v4_live_irrelevance`
- `bfcl_v4_live_simple`, `bfcl_v4_live_multiple`, `bfcl_v4_live_parallel`, `bfcl_v4_live_parallel_multiple`
- `bfcl_v4_multi_turn_base`, `bfcl_v4_multi_turn_miss_func`, `bfcl_v4_multi_turn_miss_param`, `bfcl_v4_multi_turn_long_context`
- `bfcl_v4_memory`, `bfcl_v4_web_search`

### AST-Based Evaluation Tasks

Tasks with `_ast` suffix use AST-based evaluation aligned with upstream:
- `bfcl_v4_simple_python_ast`, `bfcl_v4_simple_java_ast`, `bfcl_v4_simple_javascript_ast`
- `bfcl_v4_parallel_ast`, `bfcl_v4_multiple_ast`
- `bfcl_v4_single_turn_ast`, `bfcl_v4_ast`
- `bfcl_v4_multi_turn_base_ast`

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `accuracy` | Percentage of correct function calls (C / (C + I)) |

## Paper

Based on the Berkeley Function Call Leaderboard:

> [Berkeley Function Call Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla)
> Shishir Patel, Tianjun Zhang, Sheng Shen, et al.
> UC Berkeley
> 2024

## References

- [Upstream BFCL Repository](https://github.com/ShishirPatil/gorilla)
- [BFCL V2 Blog Post](https://gorilla.cs.berkeley.edu/blogs/12_bfcl_v2_live.html)
- [inspect_ai Documentation](https://inspect.ai-safety.com/)
