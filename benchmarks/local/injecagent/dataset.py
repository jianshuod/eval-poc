"""Dataset loading for InjecAgent benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_evals.utils import create_stable_id

# Path to InjecAgent data directory
_DATA_BASE_PATH = Path(__file__).parent.parent / "InjecAgent" / "data"


def load_injecagent_dataset(
    attack_type: str = "combined",
    setting: str = "base",
    shuffle: bool = False,
    limit: int | None = None,
) -> MemoryDataset:
    """Load the InjecAgent dataset.

    Args:
        attack_type: Type of attack - "dh" (Direct Harm), "ds" (Data Stealing),
                     or "combined" for both
        setting: IPI setting - "base" or "enhanced"
        shuffle: Whether to shuffle samples
        limit: Maximum number of samples to return

    Returns:
        A MemoryDataset containing InjecAgent test samples
    """
    samples: list[Sample] = []

    # Determine which files to load
    if attack_type == "combined":
        attack_types = ["dh", "ds"]
    else:
        attack_types = [attack_type]

    for atype in attack_types:
        file_name = f"test_cases_{atype}_{setting}.json"
        file_path = _DATA_BASE_PATH / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"InjecAgent data file not found: {file_path}. "
                f"Please ensure InjecAgent repository is cloned at {_DATA_BASE_PATH.parent}"
            )

        with open(file_path, 'r') as f:
            data = json.load(f)

        for item in data:
            # Create sample metadata
            metadata = {
                "attack_type": atype,
                "setting": setting,
                "user_tool": item.get("User Tool", ""),
                "attacker_tools": item.get("Attacker Tools", []),
                "attacker_instruction": item.get("Attacker Instruction", ""),
                "expected_achievements": item.get("Expected Achievements", ""),
                "attack_category": item.get("Attack Type", ""),
            }

            # Build the agent prompt with ReAct format
            # The user instruction is what the agent should do
            input_text = item.get("User Instruction", "")

            # Store the tool response (containing IPI) in metadata for solver to use
            metadata["tool_parameters"] = item.get("Tool Parameters", "{}")
            metadata["tool_response"] = item.get("Tool Response", "")
            metadata["thought"] = item.get("Thought", "")

            # Target is the attacker tool that should NOT be called
            # If the agent calls this tool, the attack succeeds
            target_tool = item.get("Attacker Tools", [""])[0] if item.get("Attacker Tools") else ""

            samples.append(Sample(
                input=input_text,
                target=target_tool,  # The tool that represents attack success
                id=create_stable_id(
                    f"{atype}_{setting}_{len(samples)}",
                    prefix="injecagent"
                ),
                metadata=metadata,
            ))

    if shuffle:
        import random
        random.shuffle(samples)

    if limit is not None:
        samples = samples[:limit]

    return MemoryDataset(samples)
