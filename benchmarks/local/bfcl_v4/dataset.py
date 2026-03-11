"""
BFCL v4 Dataset Loader

Loads BFCL v4 JSON data files from local data directory.
Supports all category types: single-turn, multi-turn, and agentic.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base directory for BFCL v4 data files
DATA_DIR = Path(__file__).parent / "data"

# Category to filename mapping
CATEGORY_FILES = {
    # Single-turn categories
    "simple_python": "BFCL_v4_simple_python.json",
    "simple_java": "BFCL_v4_simple_java.json",
    "simple_javascript": "BFCL_v4_simple_javascript.json",
    "parallel": "BFCL_v4_parallel.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel_multiple": "BFCL_v4_parallel_multiple.json",
    "irrelevance": "BFCL_v4_irrelevance.json",
    "live_simple": "BFCL_v4_live_simple.json",
    "live_parallel": "BFCL_v4_live_parallel.json",
    "live_multiple": "BFCL_v4_live_multiple.json",
    "live_parallel_multiple": "BFCL_v4_live_parallel_multiple.json",
    "live_relevance": "BFCL_v4_live_relevance.json",
    "live_irrelevance": "BFCL_v4_live_irrelevance.json",
    # Multi-turn categories
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
    # Agentic categories
    "memory": "BFCL_v4_memory.json",
    "web_search": "BFCL_v4_web_search.json",
    # Non-scoring
    "format_sensitivity": "BFCL_v4_format_sensitivity.json",
}

# Categories that have possible_answer files
CATEGORIES_WITH_POSSIBLE_ANSWERS = {
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "multiple",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
}


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file (one JSON object per line)."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {file_path}: {e}")
    return data


def load_possible_answers(category: str) -> dict[str, Any] | None:
    """
    Load possible answers for a category.

    Returns a dictionary mapping sample_id to possible answer data,
    or None if the category doesn't have possible answers.
    """
    if category not in CATEGORIES_WITH_POSSIBLE_ANSWERS:
        return None

    possible_answer_file = DATA_DIR / "possible_answer" / f"BFCL_v4_{category}.json"
    if not possible_answer_file.exists():
        logger.warning(f"Possible answer file not found: {possible_answer_file}")
        return None

    possible_answers = {}
    for entry in load_jsonl(possible_answer_file):
        sample_id = entry.get("id")
        if sample_id:
            possible_answers[sample_id] = entry
    return possible_answers


def load_category_data(
    category: str,
    include_possible_answers: bool = True,
) -> list[dict[str, Any]]:
    """
    Load data for a specific category.

    Args:
        category: The category name (e.g., "simple_python", "multi_turn_base")
        include_possible_answers: Whether to include possible answer data

    Returns:
        List of sample dictionaries with BFCL v4 data
    """
    if category not in CATEGORY_FILES:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Available categories: {list(CATEGORY_FILES.keys())}"
        )

    data_file = DATA_DIR / CATEGORY_FILES[category]
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Load the main data
    samples = load_jsonl(data_file)

    # Optionally load and merge possible answers
    if include_possible_answers:
        possible_answers = load_possible_answers(category)
        if possible_answers:
            for sample in samples:
                sample_id = sample.get("id")
                if sample_id in possible_answers:
                    sample["possible_answer"] = possible_answers[sample_id].get(
                        "ground_truth", []
                    )

    return samples


def load_multi_turn_func_doc(api_name: str) -> dict[str, Any] | None:
    """
    Load function documentation for multi-turn tasks.

    Args:
        api_name: Name of the API (e.g., "GorillaFileSystem", "MemoryAPI")

    Returns:
        Dictionary with function documentation or None
    """
    func_doc_file = DATA_DIR / "multi_turn_func_doc" / f"{api_name}.json"
    if not func_doc_file.exists():
        logger.warning(f"Function doc file not found: {func_doc_file}")
        return None

    with open(func_doc_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_memory_prereq_conversation(scenario: str) -> list[dict[str, Any]] | None:
    """
    Load prerequisite conversation for memory tasks.

    Args:
        scenario: Scenario name (e.g., "customer", "healthcare", "finance")

    Returns:
        List of conversation messages or None
    """
    prereq_file = (
        DATA_DIR / "memory_prereq_conversation" / f"{scenario}_conversation.json"
    )
    if not prereq_file.exists():
        logger.warning(f"Prereq conversation file not found: {prereq_file}")
        return None

    with open(prereq_file, "r", encoding="utf-8") as f:
        return json.load(f)


# Mapping of class names to function doc file names
CLASS_TO_FUNC_DOC = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "PostingAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelBooking": "travel_booking.json",
    "TravelAPI": "travel_booking.json",  # Alias for TravelBooking
    "VehicleControl": "vehicle_control.json",
    "VehicleControlAPI": "vehicle_control.json",  # Alias for VehicleControl
    "TwitterAPI": "message_api.json",  # Twitter uses MessageAPI
    "MemoryAPI": "memory_kv.json",  # Default to KV store
    "MemoryKV": "memory_kv.json",
    "MemoryVector": "memory_vector.json",
    "MemoryRecursiveSummary": "memory_rec_sum.json",
    "WebSearch": "web_search.json",
}


def load_multi_turn_tools(involved_classes: list[str]) -> list[dict[str, Any]]:
    """
    Load tool definitions for multi-turn conversations.

    Args:
        involved_classes: List of class names involved in the conversation

    Returns:
        List of tool definitions
    """
    tools = []

    for class_name in involved_classes:
        # Get the file name for this class
        file_name = CLASS_TO_FUNC_DOC.get(class_name)
        if not file_name:
            logger.warning(f"No function doc mapping for class: {class_name}")
            continue

        func_doc_file = DATA_DIR / "multi_turn_func_doc" / file_name
        if not func_doc_file.exists():
            logger.warning(f"Function doc file not found: {func_doc_file}")
            continue

        # Load tools from JSONL file (one JSON object per line)
        with open(func_doc_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        tool_def = json.loads(line)
                        tools.append(tool_def)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool definition in {file_name}: {e}")

    return tools
