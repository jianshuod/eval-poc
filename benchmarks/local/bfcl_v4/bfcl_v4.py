"""
BFCL v4 - Berkeley Function Call Leaderboard v4

Comprehensive function calling benchmark with:
- Single-turn tasks (Python, Java, JavaScript, parallel, multiple)
- Multi-turn tasks (stateful conversations)
- Agentic tasks (memory, web search)
"""

import logging
from typing import Any, cast, get_args

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageTool
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolInfo, ToolParam, ToolParams
from inspect_ai.util import JSONType

try:
    from .dataset import load_category_data, load_multi_turn_tools
    from .scorer import (
        bfcl_v4_scorer,
        bfcl_v4_multi_turn_scorer,
        bfcl_v4_ast_scorer,
        bfcl_v4_multi_turn_ast_scorer,
    )
except ImportError:
    from dataset import load_category_data, load_multi_turn_tools
    from scorer import (
        bfcl_v4_scorer,
        bfcl_v4_multi_turn_scorer,
        bfcl_v4_ast_scorer,
        bfcl_v4_multi_turn_ast_scorer,
    )

logger = logging.getLogger(__name__)

# Gorilla/BFCL schema type aliases to JSON schema primitive types.
_BFCL_TYPE_MAPPING: dict[str, JSONType] = {
    "dict": "object",
    "float": "number",
    "double": "number",
    "char": "string",
    "tuple": "array",
    "HashMap": "object",
    "Map": "object",
    "ArrayList": "array",
    "List": "array",
    "Set": "array",
    "String": "string",
    "Integer": "integer",
    "int": "integer",
    "long": "integer",
    "Long": "integer",
    "Double": "number",
    "Float": "number",
    "Boolean": "boolean",
    "boolean": "boolean",
    "Object": "object",
    "Array": "array",
    "Number": "number",
    "business": "object",
    "directory": "string",
    "file": "string",
}


def get_type(
    bfcl_type: str | None,
    *,
    schema_path: str = "unknown",
) -> JSONType:
    """Normalize BFCL/Gorilla type strings to safe JSON schema types."""
    if bfcl_type is None:
        return "string"

    normalized = bfcl_type.strip()
    if normalized in ("", "any"):
        return "string"

    mapped = _BFCL_TYPE_MAPPING.get(normalized)
    if mapped:
        return mapped

    # Handle Java-style generic types, e.g. HashMap<String, Integer>.
    base_type = normalized.split("<", 1)[0].strip()
    mapped_base = _BFCL_TYPE_MAPPING.get(base_type)
    if mapped_base:
        return mapped_base

    if normalized in get_args(JSONType):
        return cast(JSONType, normalized)

    logger.warning(
        "Unknown BFCL schema type %r at %s; coercing to 'string'.",
        bfcl_type,
        schema_path,
    )
    return "string"


def _sanitize_param_schema(param_dict: dict[str, Any], schema_path: str) -> dict[str, Any]:
    """Recursively sanitize malformed BFCL parameter schemas."""
    sanitized = dict(param_dict)
    raw_type = sanitized.get("type")
    if isinstance(raw_type, str):
        raw_type = raw_type.strip()
    sanitized["type"] = get_type(raw_type, schema_path=schema_path)

    # BFCL data occasionally places enum on array instead of its items schema.
    if (
        sanitized.get("type") == "array"
        and sanitized.get("enum")
        and isinstance(sanitized.get("items"), dict)
    ):
        items = dict(sanitized["items"])
        if get_type(items.get("type"), schema_path=f"{schema_path}.items") == "string":
            items["enum"] = sanitized.pop("enum")
            sanitized["items"] = items
        else:
            sanitized.pop("enum", None)

    properties = sanitized.get("properties")
    if isinstance(properties, dict):
        sanitized["properties"] = {
            key: _sanitize_param_schema(value, f"{schema_path}.properties.{key}")
            for key, value in properties.items()
            if isinstance(value, dict)
        }

    items = sanitized.get("items")
    if isinstance(items, dict):
        sanitized["items"] = _sanitize_param_schema(items, f"{schema_path}.items")

    return sanitized


def create_tool_param(
    param_dict: dict[str, Any] | None,
    *,
    schema_path: str = "unknown",
) -> ToolParam | None:
    """Helper function to create ToolParam instances recursively"""
    if param_dict is None:
        return None

    sanitized = _sanitize_param_schema(param_dict, schema_path)

    # Handle nested properties
    properties = None
    if sanitized.get("properties"):
        properties = {
            key: create_tool_param(value, schema_path=f"{schema_path}.properties.{key}")
            for key, value in sanitized["properties"].items()
            if value is not None
        }

    # Handle array items
    items = None
    if sanitized.get("items"):
        items = create_tool_param(sanitized["items"], schema_path=f"{schema_path}.items")

    return ToolParam(
        type=get_type(sanitized.get("type"), schema_path=schema_path),
        description=sanitized.get("description"),
        default=sanitized.get("default"),
        enum=sanitized.get("enum"),
        items=items,
        properties=properties,  # type: ignore
        additionalProperties=sanitized.get("additionalProperties"),
        required=sanitized.get("required"),
    )


def create_tool_info_from_dict(tool_dict: dict[str, Any]) -> ToolInfo:
    """
    Create a ToolInfo instance from a dictionary.

    Args:
        tool_dict: Dictionary containing tool information

    Returns:
        ToolInfo instance
    """
    # Create the parameters object
    parameters = None
    if "parameters" in tool_dict:
        parameters = create_tool_param(
            tool_dict["parameters"],
            schema_path=f"{tool_dict.get('name', 'unknown')}.parameters",
        )

    # Handle tools with no parameters (e.g., pwd function)
    if parameters is None or parameters.properties is None:
        tool_params = ToolParams(
            properties={},
            required=[],
        )
    else:
        assert "additionalProperties" not in parameters.properties
        tool_params = ToolParams(
            properties=parameters.properties,
            required=parameters.required or [],
        )

    # Create and return the ToolInfo instance
    return ToolInfo(
        name=tool_dict["name"],
        description=tool_dict["description"],
        parameters=tool_params,
    )


@solver
def bfcl_v4_solver() -> Solver:
    """
    Solver that sets up tool definitions from metadata and generates response.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tool_infos: list[ToolInfo] = []

        for tool_spec in state.metadata.get("tools", []):
            tool_info = create_tool_info_from_dict(tool_spec)
            tool_infos.append(tool_info)

        state.tools.extend(tool_infos)  # type: ignore
        return await generate(state, tool_calls="none")

    return solve


# ============================================================================
# Tool Result Generator
# ============================================================================

# Simple state tracking for file system operations
_file_system_state = {
    "files": set(),  # Track existing files
    "directories": set(),  # Track existing directories
}

# Simple state tracking for other operations
_execution_state = {
    "messages": [],  # Track sent messages
    "posts": [],  # Track created posts
    "tickets": [],  # Track created tickets
    "trades": [],  # Track executed trades
    "bookings": [],  # Track travel bookings
}


def generate_tool_result(function_name: str, arguments: dict[str, Any]) -> str:
    """
    Generate a realistic tool execution result based on the function and arguments.

    This function analyzes the function name and generates appropriate responses
    that provide meaningful information for continuing multi-turn conversations.
    """
    # File system operations (GorillaFileSystem)
    if function_name.startswith("file_") or function_name.startswith("dir_"):
        return _generate_filesystem_result(function_name, arguments)

    # Message operations (MessageAPI)
    elif function_name.startswith("message_") or function_name.startswith("send_"):
        return _generate_message_result(function_name, arguments)

    # Post operations (PostingAPI)
    elif function_name.startswith("post_"):
        return _generate_post_result(function_name, arguments)

    # Ticket operations (TicketAPI)
    elif function_name.startswith("ticket_"):
        return _generate_ticket_result(function_name, arguments)

    # Trading operations (TradingBot)
    elif function_name in ("buy", "sell", "get_portfolio", "get_balance"):
        return _generate_trading_result(function_name, arguments)

    # Travel booking operations (TravelBooking)
    elif function_name.startswith("travel_") or function_name in ("book", "cancel"):
        return _generate_travel_result(function_name, arguments)

    # Vehicle control operations (VehicleControl)
    elif function_name.startswith("vehicle_") or function_name in (
        "start_engine", "stop_engine", "accelerate", "brake", "turn"
    ):
        return _generate_vehicle_result(function_name, arguments)

    # Math operations
    elif function_name.startswith(("add", "subtract", "multiply", "divide", "calculate")):
        return _generate_math_result(function_name, arguments)

    # Memory operations
    elif function_name.startswith(("memory_", "store_", "retrieve_", "search_")):
        return _generate_memory_result(function_name, arguments)

    # Web search operations
    elif function_name.startswith("search_") or function_name == "web_search":
        return _generate_search_result(function_name, arguments)

    # Default fallback
    else:
        return f"{function_name} executed successfully with arguments: {arguments}"


def _generate_filesystem_result(function_name: str, arguments: dict[str, Any]) -> str:
    """Generate results for file system operations."""
    path = arguments.get("path", arguments.get("file", arguments.get("directory", "")))

    if "list" in function_name or "read" in function_name or "get" in function_name:
        # Return mock file/directory listing
        if "dir" in function_name.lower():
            return "Contents: ['file1.txt', 'file2.pdf', 'subfolder/']"
        else:
            return f"File content from {path}"

    elif "create" in function_name or "write" in function_name or "add" in function_name:
        _file_system_state["files"].add(path)
        return f"Successfully created/wrote {path}"

    elif "delete" in function_name or "remove" in function_name:
        _file_system_state["files"].discard(path)
        return f"Successfully deleted {path}"

    elif "move" in function_name or "copy" in function_name:
        src = arguments.get("source", arguments.get("src", ""))
        dst = arguments.get("destination", arguments.get("dst", ""))
        if "move" in function_name:
            _file_system_state["files"].discard(src)
            _file_system_state["files"].add(dst)
        return f"Successfully {'moved' if 'move' in function_name else 'copied'} {src} to {dst}"

    elif "exist" in function_name or "check" in function_name:
        exists = path in _file_system_state["files"]
        return f"{path} {'exists' if exists else 'does not exist'}"

    return f"File system operation {function_name} completed on {path}"


def _generate_message_result(function_name: str, arguments: dict[str, Any]) -> str:
    """Generate results for message operations."""
    recipient = arguments.get("recipient", arguments.get("to", "user"))
    content = arguments.get("content", arguments.get("message", arguments.get("body", "")))

    if "send" in function_name:
        _execution_state["messages"].append({"to": recipient, "content": content})
        return f"Message sent to {recipient}"

    elif "list" in function_name or "get" in function_name:
        return "Messages: [{'id': 1, 'from': 'alice', 'content': 'Hello'}]"

    return f"Message operation {function_name} completed"


def _generate_post_result(function_name: str, arguments: dict[str, Any]) -> str:
    """Generate results for post operations."""
    title = arguments.get("title", "New Post")
    content = arguments.get("content", arguments.get("body", ""))

    if "create" in function_name or "publish" in function_name or "post" in function_name:
        _execution_state["posts"].append({"title": title, "content": content})
        return f"Post '{title}' published successfully"

    elif "list" in function_name or "get" in function_name:
        return "Posts: [{'id': 1, 'title': 'Sample Post', 'author': 'user'}]"

    return f"Post operation {function_name} completed"


def _generate_ticket_result(function_name: str, arguments: dict[str, Any]) -> str:
    """Generate results for ticket operations."""
    title = arguments.get("title", arguments.get("subject", "Support Ticket"))
    description = arguments.get("description", arguments.get("details", ""))

    if "create" in function_name:
        ticket_id = len(_execution_state["tickets"]) + 1
        _execution_state["tickets"].append({"id": ticket_id, "title": title})
        return f"Ticket #{ticket_id} created: {title}"

    elif "list" in function_name or "get" in function_name:
        return f"Tickets: {[_['id'] for _ in _execution_state['tickets']]}"

    return f"Ticket operation {function_name} completed"


def _generate_trading_result(function_name: str, arguments: dict[str, Any]) -> str:
    """Generate results for trading operations."""
    symbol = arguments.get("symbol", arguments.get("asset", "AAPL"))
    quantity = arguments.get("quantity", arguments.get("amount", 10))

    if function_name in ("buy", "sell"):
        action = "bought" if function_name == "buy" else "sold"
        _execution_state["trades"].append({"action": function_name, "symbol": symbol, "quantity": quantity})
        return f"Successfully {action} {quantity} shares of {symbol}"

    elif function_name == "get_portfolio":
        trades = _execution_state["trades"]
        if trades:
            portfolio = ", ".join([f"{t['symbol']}: {t['quantity']}" for t in trades])
        else:
            portfolio = "Empty"
        return f"Portfolio: {portfolio}"

    elif function_name == "get_balance":
        return "Account balance: $10000.00"

    return f"Trading operation {function_name} completed"


def _generate_travel_result(function_name: str, arguments: dict[str, Any]) -> str:
    """Generate results for travel booking operations."""
    destination = arguments.get("destination", arguments.get("location", "Unknown"))
    date = arguments.get("date", arguments.get("departure_date", "2024-01-01"))

    if "book" in function_name:
        booking_id = len(_execution_state["bookings"]) + 1
        _execution_state["bookings"].append({"id": booking_id, "destination": destination, "date": date})
        return f"Booking confirmed for {destination} on {date}. Reference: #{booking_id}"

    elif "cancel" in function_name:
        return "Booking cancelled successfully"

    elif "search" in function_name or "list" in function_name:
        bookings = _execution_state["bookings"]
        if bookings:
            booking_list = [f"{b['id']}: {b['destination']}" for b in bookings]
        else:
            booking_list = ["No bookings"]
        return f"Available bookings: {booking_list}"

    return f"Travel operation {function_name} completed"


def _generate_vehicle_result(function_name: str, arguments: dict[str, Any]) -> str:
    """Generate results for vehicle control operations."""
    if "start" in function_name:
        return "Engine started. Vehicle ready."

    elif "stop" in function_name:
        return "Engine stopped. Vehicle powered down."

    elif "accelerate" in function_name:
        speed = arguments.get("speed", arguments.get("target_speed", 50))
        return f"Accelerating to {speed} km/h"

    elif "brake" in function_name:
        return "Braking engaged. Decelerating."

    elif "turn" in function_name:
        direction = arguments.get("direction", "left")
        return f"Turning {direction}"

    return f"Vehicle operation {function_name} completed"


def _generate_math_result(function_name: str, arguments: dict[str, Any]) -> str:
    """Generate results for math operations."""
    try:
        a = arguments.get("a", arguments.get("x", arguments.get("num1", 0)))
        b = arguments.get("b", arguments.get("y", arguments.get("num2", 0)))

        if "add" in function_name:
            result = float(a) + float(b)
        elif "subtract" in function_name:
            result = float(a) - float(b)
        elif "multiply" in function_name:
            result = float(a) * float(b)
        elif "divide" in function_name:
            result = float(a) / float(b) if float(b) != 0 else "Error: Division by zero"
        else:
            result = "Calculation completed"

        return str(result)
    except Exception:
        return "Calculation completed"


def _generate_memory_result(function_name: str, arguments: dict[str, Any]) -> str:
    """Generate results for memory operations."""
    key = arguments.get("key", arguments.get("name", "default_key"))
    value = arguments.get("value", arguments.get("data", "default_value"))

    if "store" in function_name or "set" in function_name or "add" in function_name:
        return f"Stored: {key} = {value}"

    elif "retrieve" in function_name or "get" in function_name:
        return f"Retrieved value for '{key}': {value}"

    elif "search" in function_name or "query" in function_name:
        return f"Search results for '{key}': Found 1 matching item"

    elif "delete" in function_name or "remove" in function_name:
        return f"Deleted key: {key}"

    return f"Memory operation {function_name} completed"


def _generate_search_result(function_name: str, arguments: dict[str, Any]) -> str:
    """Generate results for web search operations."""
    query = arguments.get("query", arguments.get("q", arguments.get("keyword", "")))
    return f"Search results for '{query}': Found 5 relevant results"


# ============================================================================
# Hybrid Solver (handles both single-turn and multi-turn)
# ============================================================================

@solver
def bfcl_v4_hybrid_solver() -> Solver:
    """
    Hybrid solver that routes to single-turn or multi-turn solver based on metadata.

    This solver checks the `is_multi_turn` flag in sample metadata and routes
    to the appropriate solver. Used for combined tasks that mix categories.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        is_multi_turn = state.metadata.get("is_multi_turn", False)

        if is_multi_turn:
            # Use multi-turn solver logic inline
            tool_infos: list[ToolInfo] = []

            for tool_spec in state.metadata.get("tools", []):
                tool_info = create_tool_info_from_dict(tool_spec)
                tool_infos.append(tool_info)

            state.tools.extend(tool_infos)  # type: ignore

            # Get multi-turn questions
            multi_turn_questions = state.metadata.get("multi_turn_questions", [])
            if not multi_turn_questions:
                # Fallback to single-turn generation
                response = await generate(state)
                return response

            # Initialize conversation state
            conversation_state = state

            # Process each turn
            for turn_idx, turn_messages in enumerate(multi_turn_questions):
                # Add user messages for this turn
                for user_msg in turn_messages:
                    if isinstance(user_msg, str):
                        conversation_state = conversation_state.copy(
                            messages=[*conversation_state.messages, ChatMessageUser(content=user_msg)]
                        )
                    elif isinstance(user_msg, dict):
                        role = user_msg.get("role", "user")
                        content = user_msg.get("content", "")
                        if role == "user":
                            conversation_state = conversation_state.copy(
                                messages=[*conversation_state.messages, ChatMessageUser(content=content)]
                            )

                # Generate model response
                max_steps_per_turn = 10  # Prevent infinite loops
                for step in range(max_steps_per_turn):
                    response = await generate(conversation_state)

                    # Check if model made any tool calls
                    assistant_messages = [
                        m for m in response.messages
                        if isinstance(m, ChatMessageAssistant)
                    ]

                    if not assistant_messages:
                        break  # No response, move to next turn

                    last_message = assistant_messages[-1]
                    tool_calls = last_message.tool_calls

                    # If no tool calls, end this turn
                    if not tool_calls or len(tool_calls) == 0:
                        break

                    # Generate realistic tool execution results
                    tool_result_messages = []
                    for tool_call in tool_calls:
                        result_content = generate_tool_result(tool_call.function, tool_call.arguments)
                        tool_result_messages.append(
                            ChatMessageTool(
                                tool_call_id=tool_call.id,
                                content=result_content,
                                error=False,
                            )
                        )

                    # Add tool results to conversation
                    conversation_state = response.copy(
                        messages=[*response.messages, *tool_result_messages]
                    )

                # Update state for next turn
                state = conversation_state

            return state
        else:
            # Single-turn: use standard solver logic
            tool_infos: list[ToolInfo] = []

            for tool_spec in state.metadata.get("tools", []):
                tool_info = create_tool_info_from_dict(tool_spec)
                tool_infos.append(tool_info)

            state.tools.extend(tool_infos)  # type: ignore

            response = await generate(state)
            return response

    return solve


# ============================================================================
# Multi-Turn Solver
# ============================================================================

@solver
def bfcl_v4_multi_turn_solver() -> Solver:
    """
    Solver for multi-turn conversations that handles sequential interaction.

    This solver:
    1. Processes each turn of user messages sequentially
    2. Maintains conversation history across turns
    3. Continues generating responses until model stops making tool calls
    4. Tracks all tool calls and tool results across the conversation
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tool_infos: list[ToolInfo] = []

        for tool_spec in state.metadata.get("tools", []):
            tool_info = create_tool_info_from_dict(tool_spec)
            tool_infos.append(tool_info)

        state.tools.extend(tool_infos)  # type: ignore

        # Get multi-turn questions
        multi_turn_questions = state.metadata.get("multi_turn_questions", [])
        if not multi_turn_questions:
            # Fallback to single-turn
            return await generate(state, tool_calls="none")

        # Initialize conversation state
        conversation_state = state

        # Process each turn
        for turn_idx, turn_messages in enumerate(multi_turn_questions):
            # Add user messages for this turn
            for user_msg in turn_messages:
                if isinstance(user_msg, str):
                    conversation_state = conversation_state.copy(
                        messages=[*conversation_state.messages, ChatMessageUser(content=user_msg)]
                    )
                elif isinstance(user_msg, dict):
                    role = user_msg.get("role", "user")
                    content = user_msg.get("content", "")
                    if role == "user":
                        conversation_state = conversation_state.copy(
                            messages=[*conversation_state.messages, ChatMessageUser(content=content)]
                        )

            # Generate model response
            max_steps_per_turn = 10  # Prevent infinite loops
            for step in range(max_steps_per_turn):
                response = await generate(conversation_state)

                # Check if model made any tool calls
                assistant_messages = [
                    m for m in response.messages
                    if isinstance(m, ChatMessageAssistant)
                ]

                if not assistant_messages:
                    break  # No response, move to next turn

                last_message = assistant_messages[-1]
                tool_calls = last_message.tool_calls

                # If no tool calls, end this turn
                if not tool_calls or len(tool_calls) == 0:
                    break

                # Generate realistic tool execution results
                tool_result_messages = []
                for tool_call in tool_calls:
                    result_content = generate_tool_result(tool_call.function, tool_call.arguments)
                    tool_result_messages.append(
                        ChatMessageTool(
                            tool_call_id=tool_call.id,
                            content=result_content,
                            error=False,
                        )
                    )

                # Add tool results to conversation
                conversation_state = response.copy(
                    messages=[*response.messages, *tool_result_messages]
                )

            # Update state for next turn
            state = conversation_state

        return state

    return solve


def record_to_sample(record: dict[str, Any], category: str) -> Sample:
    """
    Convert a BFCL v4 record to an inspect_ai Sample.

    Args:
        record: BFCL v4 data record
        category: Category name for determining format

    Returns:
        Sample for use with inspect_ai
    """
    # Extract question - BFCL v4 has nested list format
    question = record.get("question", [])
    tools = record.get("function", [])

    # Check if this is a multi-turn conversation
    is_multi_turn = _is_multi_turn_category(category)

    # For multi-turn samples, load tools from involved_classes
    if is_multi_turn and not tools:
        involved_classes = record.get("involved_classes", [])
        if involved_classes:
            tools = load_multi_turn_tools(involved_classes)

    if is_multi_turn and question and len(question) > 1:
        # Multi-turn: extract all turns
        multi_turn_questions = []
        for turn_messages in question:
            # Each turn is a list of message dicts
            turn_input = []
            for msg in turn_messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        turn_input.append(msg)
                elif isinstance(msg, str):
                    turn_input.append({"role": "user", "content": msg})
            multi_turn_questions.append(turn_input)

        # Extract ground truth - for multi-turn, this may be a sequence
        ground_truth_raw = record.get("possible_answer") or record.get("ground_truth", [])
        if ground_truth_raw and len(ground_truth_raw) > 0:
            ground_truth = ground_truth_raw[0]
        else:
            ground_truth = []

        # For multi-turn, input is just the first turn's messages (rest handled by solver)
        input_messages = question[0] if question else []

        return Sample(
            input=input_messages,
            target=_format_ground_truth(ground_truth),  # Full sequence target
            metadata={
                "tools": tools,
                "ground_truth": ground_truth,
                "category": category,
                "is_multi_turn": True,
                "multi_turn_questions": multi_turn_questions,
                "initial_config": record.get("initial_config", {}),
                "involved_classes": record.get("involved_classes", []),
            },
        )
    else:
        # Single-turn: use original logic
        if question and len(question) > 0:
            input_messages = question[0]
        else:
            input_messages = []

        # Extract ground truth - BFCL v4 has nested list format
        # For parallel calls, possible_answer contains multiple dicts
        # For single calls, it contains one dict
        ground_truth_raw = record.get("possible_answer") or record.get("ground_truth", [])
        if ground_truth_raw and len(ground_truth_raw) > 0:
            # Keep all elements for parallel/multiple calls
            ground_truth = ground_truth_raw
        else:
            ground_truth = []

        return Sample(
            input=input_messages,
            target=_format_ground_truth(ground_truth),
            metadata={
                "tools": tools,
                "ground_truth": ground_truth,
                "category": category,
                "is_multi_turn": False,
            },
        )


def _is_multi_turn_category(category: str) -> bool:
    """Check if a category requires multi-turn handling."""
    multi_turn_categories = {
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    }
    return category in multi_turn_categories


def _format_ground_truth(ground_truth: list[dict[str, Any]] | dict[str, Any]) -> str:
    """Format ground truth for target display."""
    if not ground_truth:
        return "No function calls"

    # Handle case where ground_truth is a single dict (BFCL v4 format)
    if isinstance(ground_truth, dict):
        calls = [ground_truth]
    else:
        calls = ground_truth

    formatted = []
    for call in calls:
        for func_name, params in call.items():
            # Handle params as dict or list
            if isinstance(params, dict):
                args_str = ", ".join(f"{k}={v}" for k, v in params.items())
            elif isinstance(params, list):
                args_str = ", ".join(str(v) for v in params)
            else:
                args_str = str(params)
            formatted.append(f"{func_name}({args_str})")

    return " | ".join(formatted)


def _create_dataset(category: str) -> list[Sample]:
    """Create dataset for a given category."""
    records = load_category_data(category, include_possible_answers=True)
    return [record_to_sample(record, category) for record in records]


# ============================================================================
# Combined BFCL v4 Task (All Categories)
# ============================================================================

@task
def bfcl_v4() -> Task:
    """
    Combined BFCL v4 task including all categories.

    This task combines:
    - Single-turn (20%): simple_python, simple_java, simple_javascript, parallel, multiple, parallel_multiple, irrelevance, live_*
    - Multi-turn (30%): multi_turn_base, multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context
    - Agentic (40%): memory_kv, memory_vector, memory_rec_sum, web_search_base, web_search_no_snippet
    """
    # Combine all datasets into one
    all_categories = [
        # Single-turn
        "simple_python", "simple_java", "simple_javascript",
        "parallel", "multiple", "parallel_multiple", "irrelevance",
        "live_simple", "live_parallel", "live_multiple",
        "live_parallel_multiple", "live_relevance", "live_irrelevance",
        # Multi-turn
        "multi_turn_base", "multi_turn_miss_func",
        "multi_turn_miss_param", "multi_turn_long_context",
        # Agentic
        # Note: memory and web_search use the same dataset with different task types
        "memory", "web_search",
    ]

    # Combine all datasets
    combined_dataset = []
    for category in all_categories:
        try:
            records = load_category_data(category, include_possible_answers=True)
            for record in records:
                sample = record_to_sample(record, category)
                # Add category to metadata for analysis
                sample.metadata["bfcl_v4_category"] = category
                combined_dataset.append(sample)
        except Exception as e:
            logger.warning(f"Failed to load category {category}: {e}")

    return Task(
        dataset=combined_dataset,
        solver=bfcl_v4_hybrid_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


# ============================================================================
# Combined BFCL v4 V3 - Scoring Categories Only (5,088 test cases)
# ============================================================================

@task
def bfcl_v4_v3() -> Task:
    """
    BFCL v4 V3 - Core function calling categories (4,423 test cases).

    This task combines the core BFCL v4 categories (excludes agentic tasks):
    - 🔄 Multi-Turn (30%, 800): multi_turn_base (200), multi_turn_miss_func (200),
                                multi_turn_miss_param (200), multi_turn_long_context (200)
    - 💡 Live (10%, 1,351): live_simple (258), live_multiple (1053),
                            live_parallel (16), live_parallel_multiple (24)
    - 📂 Non-Live (10%, 1,150): simple_python (400), simple_java (100), simple_javascript (50),
                                multiple (200), parallel (200), parallel_multiple (200), irrelevance (240)
    - 🧐 Hallucination (10%, 1,122): live_irrelevance (882)

    Excluded categories:
    - 🤖 Agentic (40%, 265): web_search (100), memory (155) - excluded from v3
    - 📝 Format Sensitivity (5,200): 26 configurations × 200 test cases
    - ✅ Relevance (18)

    Note: This focuses on pure function calling capability without external tool dependencies
    (web_search requires SERPAPI_KEY, memory requires additional infrastructure).
    """
    # Core scoring categories (excludes agentic, format_sensitivity, and relevance)
    scoring_categories = [
        # 🔄 Multi-Turn (30%, 800 test cases)
        "multi_turn_base",        # 200: Multi-turn baseline
        "multi_turn_miss_func",   # 200: Missing function
        "multi_turn_miss_param",  # 200: Missing parameter
        "multi_turn_long_context", # 200: Long context

        # 💡 Live (10%, 1,351 test cases)
        "live_simple",           # 258: Live simple tasks
        "live_multiple",         # 1053: Live multiple tasks
        "live_parallel",         # 16: Live parallel tasks
        "live_parallel_multiple", # 24: Live parallel multiple tasks

        # 📂 Non-Live (10%, 1,150 test cases)
        "simple_python",         # 400: Python function calling
        "simple_java",           # 100: Java function calling
        "simple_javascript",     # 50: JavaScript function calling
        "multiple",              # 200: Multiple function calling
        "parallel",              # 200: Parallel function calling
        "parallel_multiple",     # 200: Parallel multiple function calling
        "irrelevance",           # 240: Irrelevance detection

        # 🧐 Hallucination (10%, 1,122 test cases)
        "live_irrelevance",      # 882: Live irrelevance (hallucination measurement)
    ]

    # Combine all scoring datasets
    combined_dataset = []
    for category in scoring_categories:
        try:
            records = load_category_data(category, include_possible_answers=True)
            for record in records:
                sample = record_to_sample(record, category)
                # Add category to metadata for analysis
                sample.metadata["bfcl_v4_category"] = category
                # Add scoring group for weighted average calculation
                sample.metadata["bfcl_v4_scoring_group"] = _get_scoring_group(category)
                combined_dataset.append(sample)
        except Exception as e:
            logger.warning(f"Failed to load category {category}: {e}")

    return Task(
        dataset=combined_dataset,
        solver=bfcl_v4_hybrid_solver(),
        scorer=bfcl_v4_scorer(),
        version="3.0.0",
    )


def _get_scoring_group(category: str) -> str:
    """Get the scoring group for a category."""
    multi_turn = {"multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"}
    live = {"live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"}
    non_live = {"simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance"}
    hallucination = {"live_irrelevance"}

    if category in multi_turn:
        return "multi_turn"
    elif category in live:
        return "live"
    elif category in non_live:
        return "non_live"
    elif category in hallucination:
        return "hallucination"
    else:
        return "unknown"


# ============================================================================
# Combined BFCL v4 Single-Turn - Live + Non-Live + Hallucination Only
# ============================================================================

@task
def bfcl_v4_single_turn() -> Task:
    """
    BFCL v4 Single-Turn - All single-turn categories (V1 + V2).

    This task combines all single-turn function calling categories from BFCL V1 and V2:
    - 💡 Live from V2 (1,367): live_simple (258), live_multiple (1053),
                         live_parallel (16), live_parallel_multiple (24),
                         live_relevance (16)
    - 📂 Non-Live from V1 (1,150): simple_python (400), simple_java (100),
                             simple_javascript (50), multiple (200), parallel (200),
                             parallel_multiple (200), irrelevance (240)
    - 🧐 Hallucination from V2 (882): live_irrelevance (882)

    Total: ~3,399 test cases (V1 Non-Live + V2 Live + V2 Hallucination)

    Excluded categories:
    - 🔄 Multi-Turn (30%, 800): multi_turn_base, multi_turn_miss_func,
                        multi_turn_miss_param, multi_turn_long_context
    - 🤖 Agentic (40%, 265): web_search, memory
    - 📝 Format Sensitivity (5,200): 26 configurations × 200 test cases

    Note: This split equals BFCL V1 (non-live) + BFCL V2 (live), representing
    the complete single-turn function calling benchmark. It excludes the V3
    multi-turn categories and V4 agentic categories.
    """
    # Single-turn categories only (V1 Non-Live + V2 Live + V2 Hallucination)
    single_turn_categories = [
        # 💡 Live from V2 (1,367 test cases)
        "live_simple",           # 258: Live simple tasks
        "live_multiple",         # 1053: Live multiple tasks
        "live_parallel",         # 16: Live parallel tasks
        "live_parallel_multiple", # 24: Live parallel multiple tasks
        "live_relevance",        # 16: Live relevance detection (at least one function relevant)

        # 📂 Non-Live from V1 (1,150 test cases)
        "simple_python",         # 400: Python function calling
        "simple_java",           # 100: Java function calling
        "simple_javascript",     # 50: JavaScript function calling
        "multiple",              # 200: Multiple function calling
        "parallel",              # 200: Parallel function calling
        "parallel_multiple",     # 200: Parallel multiple function calling
        "irrelevance",           # 240: Irrelevance detection (no functions relevant)

        # 🧐 Hallucination from V2 (882 test cases)
        "live_irrelevance",      # 882: Live irrelevance (hallucination measurement)
    ]

    # Combine all single-turn datasets
    combined_dataset = []
    for category in single_turn_categories:
        try:
            records = load_category_data(category, include_possible_answers=True)
            for record in records:
                sample = record_to_sample(record, category)
                # Add category to metadata for analysis
                sample.metadata["bfcl_v4_category"] = category
                # Add scoring group for analysis
                sample.metadata["bfcl_v4_scoring_group"] = _get_scoring_group(category)
                combined_dataset.append(sample)
        except Exception as e:
            logger.warning(f"Failed to load category {category}: {e}")

    return Task(
        dataset=combined_dataset,
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


# ============================================================================
# Individual Category Tasks (for targeted evaluation)
# ============================================================================

@task
def bfcl_v4_simple_python() -> Task:
    """Single-turn Python function calling tasks."""
    return Task(
        dataset=_create_dataset("simple_python"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_simple_java() -> Task:
    """Single-turn Java function calling tasks."""
    return Task(
        dataset=_create_dataset("simple_java"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_simple_javascript() -> Task:
    """Single-turn JavaScript function calling tasks."""
    return Task(
        dataset=_create_dataset("simple_javascript"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_parallel() -> Task:
    """Parallel function calling tasks (multiple independent functions)."""
    return Task(
        dataset=_create_dataset("parallel"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_multiple() -> Task:
    """Multiple function calling tasks (dependent functions)."""
    return Task(
        dataset=_create_dataset("multiple"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_parallel_multiple() -> Task:
    """Parallel multiple function calling tasks."""
    return Task(
        dataset=_create_dataset("parallel_multiple"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_irrelevance() -> Task:
    """Tasks testing if model ignores irrelevant context."""
    return Task(
        dataset=_create_dataset("irrelevance"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_live_simple() -> Task:
    """Live simple function calling tasks."""
    return Task(
        dataset=_create_dataset("live_simple"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_live_parallel() -> Task:
    """Live parallel function calling tasks."""
    return Task(
        dataset=_create_dataset("live_parallel"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_live_multiple() -> Task:
    """Live multiple function calling tasks."""
    return Task(
        dataset=_create_dataset("live_multiple"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_live_parallel_multiple() -> Task:
    """Live parallel multiple function calling tasks."""
    return Task(
        dataset=_create_dataset("live_parallel_multiple"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_live_irrelevance() -> Task:
    """Live irrelevance tasks (hallucination measurement)."""
    return Task(
        dataset=_create_dataset("live_irrelevance"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_multi_turn_base() -> Task:
    """Multi-turn baseline tasks."""
    return Task(
        dataset=_create_dataset("multi_turn_base"),
        solver=bfcl_v4_multi_turn_solver(),
        scorer=bfcl_v4_multi_turn_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_multi_turn_miss_func() -> Task:
    """Multi-turn missing function tasks."""
    return Task(
        dataset=_create_dataset("multi_turn_miss_func"),
        solver=bfcl_v4_multi_turn_solver(),
        scorer=bfcl_v4_multi_turn_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_multi_turn_miss_param() -> Task:
    """Multi-turn missing parameter tasks."""
    return Task(
        dataset=_create_dataset("multi_turn_miss_param"),
        solver=bfcl_v4_multi_turn_solver(),
        scorer=bfcl_v4_multi_turn_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_multi_turn_long_context() -> Task:
    """Multi-turn long context tasks."""
    return Task(
        dataset=_create_dataset("multi_turn_long_context"),
        solver=bfcl_v4_multi_turn_solver(),
        scorer=bfcl_v4_multi_turn_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_memory() -> Task:
    """Memory tasks (KV, Vector, Recursive Summary)."""
    return Task(
        dataset=_create_dataset("memory"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


@task
def bfcl_v4_web_search() -> Task:
    """Web search tasks (multi-hop reasoning with search tools)."""
    return Task(
        dataset=_create_dataset("web_search"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_scorer(),
        version="1.0.0",
    )


# ============================================================================
# AST-based Evaluation Tasks (Aligned with Upstream)
# ============================================================================

@task
def bfcl_v4_ast() -> Task:
    """
    BFCL v4 task with AST-based evaluation (aligned with upstream).

    This version uses the AST-based scorer that matches the upstream Berkeley
    implementation, providing:
    - Comprehensive type checking (Python, Java, JavaScript)
    - String standardization (case-insensitive, punctuation normalization)
    - Variable detection
    - Nested type validation
    - Detailed error classification

    Version: 2.0.0
    """
    # Combine all datasets into one
    all_categories = [
        # Single-turn
        "simple_python", "simple_java", "simple_javascript",
        "parallel", "multiple", "parallel_multiple", "irrelevance",
        "live_simple", "live_parallel", "live_multiple",
        "live_parallel_multiple", "live_relevance", "live_irrelevance",
        # Multi-turn
        "multi_turn_base", "multi_turn_miss_func",
        "multi_turn_miss_param", "multi_turn_long_context",
        # Agentic
        "memory", "web_search",
    ]

    # Combine all datasets
    combined_dataset = []
    for category in all_categories:
        try:
            records = load_category_data(category, include_possible_answers=True)
            for record in records:
                sample = record_to_sample(record, category)
                sample.metadata["bfcl_v4_category"] = category
                combined_dataset.append(sample)
        except Exception as e:
            logger.warning(f"Failed to load category {category}: {e}")

    return Task(
        dataset=combined_dataset,
        solver=bfcl_v4_hybrid_solver(),
        scorer=bfcl_v4_ast_scorer(),
        version="2.0.0",
    )


@task
def bfcl_v4_single_turn_ast() -> Task:
    """Single-turn tasks with AST-based evaluation."""
    all_categories = [
        "simple_python", "simple_java", "simple_javascript",
        "parallel", "multiple", "parallel_multiple", "irrelevance",
        "live_simple", "live_parallel", "live_multiple",
        "live_parallel_multiple", "live_relevance", "live_irrelevance",
    ]

    combined_dataset = []
    for category in all_categories:
        try:
            records = load_category_data(category, include_possible_answers=True)
            for record in records:
                sample = record_to_sample(record, category)
                sample.metadata["bfcl_v4_category"] = category
                combined_dataset.append(sample)
        except Exception as e:
            logger.warning(f"Failed to load category {category}: {e}")

    return Task(
        dataset=combined_dataset,
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_ast_scorer(),
        version="2.0.0",
    )


@task
def bfcl_v4_simple_python_ast() -> Task:
    """Single-turn Python function calling tasks with AST-based evaluation."""
    return Task(
        dataset=_create_dataset("simple_python"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_ast_scorer(),
        version="2.0.0",
    )


@task
def bfcl_v4_simple_java_ast() -> Task:
    """Single-turn Java function calling tasks with AST-based evaluation."""
    return Task(
        dataset=_create_dataset("simple_java"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_ast_scorer(),
        version="2.0.0",
    )


@task
def bfcl_v4_simple_javascript_ast() -> Task:
    """Single-turn JavaScript function calling tasks with AST-based evaluation."""
    return Task(
        dataset=_create_dataset("simple_javascript"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_ast_scorer(),
        version="2.0.0",
    )


@task
def bfcl_v4_parallel_ast() -> Task:
    """Parallel function calling tasks with AST-based evaluation."""
    return Task(
        dataset=_create_dataset("parallel"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_ast_scorer(),
        version="2.0.0",
    )


@task
def bfcl_v4_multiple_ast() -> Task:
    """Multiple function calling tasks with AST-based evaluation."""
    return Task(
        dataset=_create_dataset("multiple"),
        solver=bfcl_v4_solver(),
        scorer=bfcl_v4_ast_scorer(),
        version="2.0.0",
    )


@task
def bfcl_v4_multi_turn_base_ast() -> Task:
    """Multi-turn baseline tasks with AST-based evaluation."""
    return Task(
        dataset=_create_dataset("multi_turn_base"),
        solver=bfcl_v4_multi_turn_solver(),
        scorer=bfcl_v4_multi_turn_ast_scorer(),
        version="2.0.0",
    )


__all__ = [
    "bfcl_v4",
    "bfcl_v4_v3",  # Official scoring categories (5,088 test cases)
    "bfcl_v4_single_turn",  # Live + Non-Live + Hallucination only (3,623 test cases)
    # Individual category tasks
    # Non-Live (10%)
    "bfcl_v4_simple_python",
    "bfcl_v4_simple_java",
    "bfcl_v4_simple_javascript",
    "bfcl_v4_parallel",
    "bfcl_v4_multiple",
    "bfcl_v4_parallel_multiple",
    "bfcl_v4_irrelevance",
    # Live (10%)
    "bfcl_v4_live_simple",
    "bfcl_v4_live_parallel",
    "bfcl_v4_live_multiple",
    "bfcl_v4_live_parallel_multiple",
    "bfcl_v4_live_irrelevance",
    # Multi-Turn (30%)
    "bfcl_v4_multi_turn_base",
    "bfcl_v4_multi_turn_miss_func",
    "bfcl_v4_multi_turn_miss_param",
    "bfcl_v4_multi_turn_long_context",
    # Agentic (40%)
    "bfcl_v4_memory",
    "bfcl_v4_web_search",
    # AST-based evaluation tasks (v2.0.0 - aligned with upstream)
    "bfcl_v4_ast",
    "bfcl_v4_single_turn_ast",
    "bfcl_v4_simple_python_ast",
    "bfcl_v4_simple_java_ast",
    "bfcl_v4_simple_javascript_ast",
    "bfcl_v4_parallel_ast",
    "bfcl_v4_multiple_ast",
    "bfcl_v4_multi_turn_base_ast",
]
