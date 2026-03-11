# Local benchmarks package

# Import local benchmarks so they can be discovered by inspect_ai
try:
    from . import raccoon  # noqa: F401
except ImportError:
    pass  # raccoon may have missing dependencies

try:
    from . import overthink  # noqa: F401
except ImportError:
    pass  # overthink may have missing dependencies

try:
    from . import st_webagentbench  # noqa: F401
except ImportError:
    pass  # st_webagentbench may have missing dependencies

__all__ = ["raccoon", "overthink", "st_webagentbench"]
