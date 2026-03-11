"""ST-WebAgentBench - A Benchmark for Evaluating Safety & Trustworthiness in Web Agents.

This benchmark evaluates not only whether agents complete tasks, but whether they do
so while respecting safety and trustworthiness policies.

Paper: https://arxiv.org/abs/2410.06703
GitHub: https://github.com/segev-shlomov/ST-WebAgentBench
"""

from .st_webagentbench import st_webagentbench

__all__ = ["st_webagentbench"]
