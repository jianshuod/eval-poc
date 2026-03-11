"""
Toolathlon: The Tool Decathlon - Benchmarking Language Agents

Agent capability and safety evaluation using 600+ diverse tools
in realistic environments with long-horizon task execution.

Based on:
    https://github.com/hkust-nlp/Toolathlon
    https://arxiv.org/abs/2510.25726
"""

from .toolathlon import toolathlon

__all__ = ["toolathlon"]
