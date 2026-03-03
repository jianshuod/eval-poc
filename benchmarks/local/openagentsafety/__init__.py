"""
OpenAgentSafety - Agent Safety in Realistic High-Risk Environments

This benchmark evaluates LLM agents on 360+ safety tasks in realistic environments
with real tools (file systems, terminals, browsers, messaging platforms).

Based on:
    https://github.com/sani903/OpenAgentSafety
"""

from .openagentsafety import openagentsafety

__all__ = ["openagentsafety"]
