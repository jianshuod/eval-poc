"""
Runtime module for Toolathlon evaluation.

This module handles interfacing with Toolathlon's evaluation system.
It supports multiple modes:
1. public_service: Use Toolathlon's public evaluation service (NO local setup required!)
2. subprocess: Call Toolathlon's eval_client.py via subprocess
3. direct: Import and call Toolathlon modules directly

The public_service mode is recommended for most users as it requires NO local
Docker, MCP servers, or external service setup.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver

# Toolathlon project root
_TOOLATHLON_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "Toolathlon"


# Default public service configuration
DEFAULT_PUBLIC_SERVICE = {
    "host": "47.253.6.47",
    "port": "8080",
    "ws_proxy_port": "8081",
}


@solver
def run_toolathlon_evaluation(
    toolathlon_root: str | Path | None = None,
    mode: str = "public_service",
    service_host: str | None = None,
    service_port: int | None = None,
    ws_proxy_port: int | None = None,
    max_steps: int | None = None,
    timeout: int = 7200,  # 2 hours default (service mode can take longer)
    debug: bool = False,
    use_safety_proxy: bool = False,
    safety_proxy_host: str = "127.0.0.1",
    safety_proxy_port: int = 8765,
) -> Solver:
    """Solver that runs Toolathlon evaluation for each task.

    Args:
        toolathlon_root: Path to Toolathlon project root (auto-detected if None)
        mode: Execution mode
            - "public_service": Use Toolathlon's public evaluation service (recommended!)
            - "subprocess": Use local Toolathlon installation via eval_client.py
            - "direct": Import and run Toolathlon modules directly
        service_host: Public service host (default: 47.253.6.47)
        service_port: Public service port (default: 8080)
        ws_proxy_port: WebSocket proxy port (default: 8081)
        max_steps: Maximum steps for the agent (None uses Toolathlon default)
        timeout: Timeout per task in seconds
        debug: Enable debug output
        use_safety_proxy: Use local safety proxy for safety-confirmation
            NOTE: When using with public_service mode, the safety proxy must be
            publicly accessible since Toolathlon's service runs remotely.
            Default 127.0.0.1 will NOT work with public_service mode.
            Use safety_proxy_secure.py with authentication for public access.
        safety_proxy_host: Safety proxy host (default: 127.0.0.1)
            Use 0.0.0.0 for public access with safety_proxy_secure.py
        safety_proxy_port: Safety proxy port (default: 8765)

    Returns:
        A Solver that runs Toolathlon evaluation

    Network Architecture Note:
        When use_safety_proxy=True with mode="public_service":
        - Toolathlon public service (47.253.6.47) makes API calls to your endpoint
        - Your safety proxy must be accessible from the internet
        - Use safety_proxy_secure.py with --public --proxy-key for secure access
        - Or use SSH/Cloudflare tunnel for testing
        - For pure local evaluation, use mode="subprocess" or "direct" instead
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Initialize metadata
        if not state.metadata:
            state.metadata = {}

        # Get task information from sample
        task_name = state.metadata.get("task_name", state.target.text)
        task_dir = state.metadata.get("task_dir", "")

        if not task_name:
            state.metadata["toolathlon_error"] = "No task name found in sample"
            return state

        # Determine toolathlon root
        root = Path(toolathlon_root) if toolathlon_root else _TOOLATHLON_ROOT

        if not root.exists():
            state.metadata["toolathlon_error"] = f"Toolathlon root not found: {root}"
            return state

        try:
            # Run evaluation based on mode
            if mode == "public_service":
                result = await _run_public_service(
                    task_name=task_name,
                    root=root,
                    service_host=service_host or DEFAULT_PUBLIC_SERVICE["host"],
                    service_port=service_port or int(DEFAULT_PUBLIC_SERVICE["port"]),
                    ws_proxy_port=ws_proxy_port or int(DEFAULT_PUBLIC_SERVICE["ws_proxy_port"]),
                    max_steps=max_steps,
                    timeout=timeout,
                    debug=debug,
                    use_safety_proxy=use_safety_proxy,
                    safety_proxy_host=safety_proxy_host,
                    safety_proxy_port=safety_proxy_port,
                )
            elif mode == "subprocess":
                result = await _run_subprocess(
                    task_name=task_name,
                    root=root,
                    max_steps=max_steps,
                    timeout=timeout,
                    debug=debug,
                )
            else:
                result = await _run_direct(
                    task_name=task_name,
                    root=root,
                    max_steps=max_steps,
                    timeout=timeout,
                    debug=debug,
                )

            # Store results in metadata for scorer
            state.metadata.update(result)

            # Set completion text with result summary
            if result.get("success"):
                state.output.completion = f"Task {task_name} completed successfully"
            else:
                state.output.completion = f"Task {task_name} failed: {result.get('error', 'Unknown error')}"

        except asyncio.TimeoutError:
            state.metadata["toolathlon_error"] = f"Timeout after {timeout}s"
            state.output.completion = f"Task {task_name} timed out"
        except Exception as e:
            state.metadata["toolathlon_error"] = str(e)
            state.output.completion = f"Task {task_name} error: {str(e)}"

        return state

    return solve


async def _run_public_service(
    task_name: str,
    root: Path,
    service_host: str,
    service_port: int,
    ws_proxy_port: int,
    max_steps: int | None,
    timeout: int,
    debug: bool,
    use_safety_proxy: bool = False,
    safety_proxy_host: str = "127.0.0.1",
    safety_proxy_port: int = 8765,
) -> dict[str, Any]:
    """Run Toolathlon evaluation using the public evaluation service.

    This is the RECOMMENDED mode as it requires NO local setup:
    - No Docker/Podman needed
    - No MCP servers needed
    - No external services needed
    - Just eval_client.py and API credentials

    The public service handles all the infrastructure.

    With use_safety_proxy=True, routes requests through a local safety proxy
    that applies safety-confirmation before forwarding to the actual model API.

    CRITICAL NETWORK CONSIDERATION:
        When use_safety_proxy=True:
        - Toolathlon public service (47.253.6.47) makes API calls to YOUR endpoint
        - Default safety_proxy_host="127.0.0.1" will NOT work (localhost not accessible remotely)
        - You must either:
          1. Use safety_proxy_secure.py with --public and --proxy-key for secure access
          2. Set up SSH tunnel: ssh -R 8765:localhost:8765 your-server.com
          3. Use Cloudflare tunnel: cloudflared tunnel --url http://localhost:8765
          4. Use mode="subprocess" or "direct" for pure local evaluation
    """
    # Check if eval_client.py exists
    eval_client = root / "eval_client.py"
    if not eval_client.exists():
        return {
            "success": False,
            "error": f"eval_client.py not found at {eval_client}",
            "status": "eval_client_not_found",
        }

    # Get model configuration from environment
    base_url = os.environ.get("TOOLATHLON_OPENAI_BASE_URL", "")
    api_key = os.environ.get("TOOLATHLON_OPENAI_API_KEY", "")
    model_name = os.environ.get("TOOLATHLON_MODEL_NAME", "gpt-4o")

    # If using safety proxy, point to the local proxy instead
    if use_safety_proxy:
        proxy_url = f"http://{safety_proxy_host}:{safety_proxy_port}/v1"
        original_base_url = base_url
        base_url = proxy_url
        # Set environment variables for the safety proxy
        os.environ["SAFETY_PROXY_ACTUAL_BASE"] = original_base_url
        if api_key:
            os.environ["SAFETY_PROXY_ACTUAL_KEY"] = api_key

    if not base_url:
        return {
            "success": False,
            "error": "TOOLATHLON_OPENAI_BASE_URL environment variable not set",
            "status": "missing_config",
        }

    # Create temporary task file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"{task_name}\n")
        task_file = f.name

    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "toolathlon_output"

            # Build command for eval_client.py
            cmd = [
                "python",
                str(eval_client),
                "run",
                "--mode", "public",
                "--base-url", base_url,
                "--model-name", model_name,
                "--output-dir", str(output_dir),
                "--server-host", service_host,
                "--api-key", api_key or "dummy",  # api_key may not be needed for proxy mode
                "--server-port", str(service_port),
                "--ws-proxy-port", str(ws_proxy_port),
                "--task-list-file", task_file,
                "--workers", "1",  # Run sequentially for reliability
                "--skip-container-restart",
            ]

            if debug:
                cmd.append("--debug")

            # Run with timeout
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise

            if proc.returncode != 0:
                return {
                    "success": False,
                    "error": stderr.decode() if stderr else "Unknown error",
                    "status": "execution_failed",
                    "returncode": proc.returncode,
                    "stdout": stdout.decode() if stdout else "",
                }

            # Parse results from output directory
            # The public service writes results to {output_dir}/{task_name}/eval_res.json
            result_file = output_dir / task_name / "eval_res.json"
            log_file = output_dir / task_name / "log.json"

            result = {
                "success": True,
                "task_name": task_name,
                "mode": "public_service_with_safety_proxy" if use_safety_proxy else "public_service",
                "output_dir": str(output_dir),
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "use_safety_proxy": use_safety_proxy,
            }

            # Load evaluation results if available
            if result_file.exists():
                with open(result_file) as f:
                    eval_result = json.load(f)
                    result["evaluation"] = eval_result
                    result["pass"] = eval_result.get("pass", False)

            if log_file.exists():
                with open(log_file) as f:
                    log_data = json.load(f)
                    result["log"] = log_data
                    result["status"] = log_data.get("status", "unknown")

            return result

    finally:
        # Clean up task file
        try:
            os.unlink(task_file)
        except:
            pass


async def _run_subprocess(
    task_name: str,
    root: Path,
    max_steps: int | None,
    timeout: int,
    debug: bool,
) -> dict[str, Any]:
    """Run Toolathlon evaluation via subprocess (local installation required).

    Uses Toolathlon's scripts/run_single_containerized.sh script.
    Requires Docker/Podman and full Toolathlon setup.
    """
    eval_client = root / "eval_client.py"
    if not eval_client.exists():
        # Fall back to run_single_containerized.sh
        return await _run_containerized(task_name, root, max_steps, timeout, debug)

    # Get model configuration from environment
    base_url = os.environ.get("TOOLATHLON_OPENAI_BASE_URL", "")
    api_key = os.environ.get("TOOLATHLON_OPENAI_API_KEY", "")
    model_name = os.environ.get("TOOLATHLON_MODEL_NAME", "gpt-4o")

    if not base_url:
        return {
            "success": False,
            "error": "TOOLATHLON_OPENAI_BASE_URL environment variable not set",
            "status": "missing_config",
        }

    # Create temporary task file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"{task_name}\n")
        task_file = f.name

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "toolathlon_output"

            cmd = [
                "python",
                str(eval_client),
                "run",
                "--mode", "private",
                "--base-url", base_url,
                "--model-name", model_name,
                "--output-dir", str(output_dir),
                "--api-key", api_key or "dummy",
                "--task-list-file", task_file,
            ]

            if max_steps:
                cmd.extend(["--max-steps", str(max_steps)])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise

            if proc.returncode != 0:
                return {
                    "success": False,
                    "error": stderr.decode() if stderr else "Unknown error",
                    "status": "execution_failed",
                    "returncode": proc.returncode,
                }

            # Parse results
            result_file = output_dir / task_name / "eval_res.json"
            log_file = output_dir / task_name / "log.json"

            result = {
                "success": True,
                "task_name": task_name,
                "mode": "subprocess",
                "output_dir": str(output_dir),
            }

            if result_file.exists():
                with open(result_file) as f:
                    result["evaluation"] = json.load(f)
                    result["pass"] = result["evaluation"].get("pass", False)

            if log_file.exists():
                with open(log_file) as f:
                    result["log"] = json.load(f)
                    result["status"] = result["log"].get("status", "unknown")

            return result

    finally:
        try:
            os.unlink(task_file)
        except:
            pass


async def _run_containerized(
    task_name: str,
    root: Path,
    max_steps: int | None,
    timeout: int,
    debug: bool,
) -> dict[str, Any]:
    """Run using Toolathlon's containerized script (legacy method)."""
    script_path = root / "scripts" / "run_single_containerized.sh"

    if not script_path.exists():
        return {
            "success": False,
            "error": f"Toolathlon run script not found: {script_path}",
            "status": "script_not_found",
        }

    # Get model configuration
    model_name = os.environ.get("TOOLATHLON_MODEL_NAME", "gpt-4o")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "toolathlon_output"
        output_dir.mkdir()

        task_dir = f"finalpool/{task_name}"
        max_steps_arg = max_steps or 50

        cmd = [
            str(script_path),
            task_dir,
            "quickstart",
            str(output_dir),
            model_name,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        if proc.returncode != 0:
            return {
                "success": False,
                "error": stderr.decode() if stderr else "Unknown error",
                "status": "execution_failed",
                "returncode": proc.returncode,
            }

        # Parse results
        log_file = output_dir / "finalpool" / task_name / "log.json"
        eval_file = output_dir / "finalpool" / task_name / "eval_res.json"

        result = {
            "success": True,
            "task_name": task_name,
            "mode": "containerized",
            "output_dir": str(output_dir),
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else "",
        }

        if eval_file.exists():
            with open(eval_file) as f:
                result["evaluation"] = json.load(f)
                result["pass"] = result["evaluation"].get("pass", False)

        if log_file.exists():
            with open(log_file) as f:
                result["log"] = json.load(f)
                result["status"] = result["log"].get("status", "unknown")

        return result


async def _run_direct(
    task_name: str,
    root: Path,
    max_steps: int | None,
    timeout: int,
    debug: bool,
) -> dict[str, Any]:
    """Run Toolathlon evaluation via direct module import.

    Requires full Toolathlon Python environment setup.
    """
    try:
        import sys
        sys.path.insert(0, str(root))

        from utils.general.helper import read_json
        from utils.data_structures.task_config import TaskConfig
        from utils.task_runner.runner import TaskRunner
        from utils.evaluation.evaluator import TaskEvaluator

        # Build task config
        task_dir = root / "tasks" / "finalpool" / task_name

        eval_config = {
            "agent": {
                "model": {
                    "short_name": os.environ.get("TOOLATHLON_MODEL_NAME", "gpt-4o"),
                    "provider": "unified",
                }
            },
            "global_task_config": {
                "max_steps_under_single_turn_mode": max_steps or 50,
            }
        }

        mcp_config, agent_config, user_config = TaskRunner.load_configs(eval_config)

        task_config = TaskConfig.build(
            str(task_dir.relative_to(root)),
            agent_config.model.short_name,
            eval_config["global_task_config"],
            single_turn_mode=True,
            cn_mode=False,
        )

        # Run task with timeout
        task_status = await asyncio.wait_for(
            TaskRunner.run_single_task(
                task_config=task_config,
                agent_config=agent_config,
                user_config=user_config,
                mcp_config=mcp_config,
                debug=debug,
                allow_resume=False,
                single_turn_mode=True,
            ),
            timeout=timeout,
        )

        # Evaluate results
        eval_res = await TaskEvaluator.evaluate_from_log_file(
            task_config.log_file,
            allow_resume=False,
        )

        return {
            "success": True,
            "task_name": task_name,
            "mode": "direct",
            "status": task_status.value,
            "pass": eval_res.get("pass", False),
            "evaluation": eval_res,
            "log_file": str(task_config.log_file),
        }

    except ImportError as e:
        return {
            "success": False,
            "error": f"Failed to import Toolathlon modules: {e}",
            "status": "import_error",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "status": "execution_error",
        }


def verify_toolathlon_setup(toolathlon_root: str | Path | None = None,
                            mode: str = "public_service") -> dict[str, Any]:
    """Verify that Toolathlon is properly set up for the given mode.

    Args:
        toolathlon_root: Path to Toolathlon installation
        mode: Execution mode to verify (public_service, subprocess, direct)

    Returns:
        Dict with verification results
    """
    root = Path(toolathlon_root) if toolathlon_root else _TOOLATHLON_ROOT

    result = {
        "toolathlon_root": str(root),
        "root_exists": root.exists(),
        "mode": mode,
        "issues": [],
    }

    if not result["root_exists"]:
        result["issues"].append(f"Toolathlon root not found: {root}")
        return result

    if mode == "public_service":
        # Check for eval_client.py
        eval_client = root / "eval_client.py"
        if not eval_client.exists():
            result["issues"].append("eval_client.py not found - required for public_service mode")
        else:
            result["eval_client_exists"] = True

        # Check for required Python packages
        try:
            import httpx
            import websockets
            result["dependencies_ok"] = True
        except ImportError:
            result["dependencies_ok"] = False
            result["issues"].append("Required packages missing: httpx, websockets. Install with: pip install httpx websockets")

        # Check environment variables
        if not os.environ.get("TOOLATHLON_OPENAI_BASE_URL"):
            result["issues"].append("TOOLATHLON_OPENAI_BASE_URL environment variable not set")
        if not os.environ.get("TOOLATHLON_OPENAI_API_KEY"):
            result["issues"].append("TOOLATHLON_OPENAI_API_KEY environment variable not set")

    elif mode == "subprocess":
        # Check for eval_client.py or run_single_containerized.sh
        eval_client = root / "eval_client.py"
        script_path = root / "scripts" / "run_single_containerized.sh"

        if not eval_client.exists() and not script_path.exists():
            result["issues"].append("Neither eval_client.py nor run_single_containerized.sh found")

        # Check for Docker/Podman
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                check=True,
            )
            result["docker_available"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(
                    ["podman", "--version"],
                    capture_output=True,
                    check=True,
                )
                result["docker_available"] = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                result["docker_available"] = False
                result["issues"].append("Docker/Podman not available")

    elif mode == "direct":
        # Check if Toolathlon modules can be imported
        try:
            import sys
            sys.path.insert(0, str(root))
            from utils.data_structures import task_config
            result["dependencies_ok"] = True
        except ImportError:
            result["dependencies_ok"] = False
            result["issues"].append("Toolathlon Python dependencies not installed")

    return result
