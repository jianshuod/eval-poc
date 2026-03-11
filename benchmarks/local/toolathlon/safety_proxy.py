"""
Toolathlon Safety Proxy - Local proxy server with safety-confirmation

This proxy sits between Toolathlon's eval_client.py and your model API,
applying safety-confirmation to all requests.

Usage:
    1. Start this proxy server
    2. Configure TOOLATHLON_OPENAI_BASE_URL to point to this proxy
    3. Run Toolathlon evaluation

The proxy will:
- Receive requests from Toolathlon
- Apply safety-confirmation via the safety_confirmation package
- Forward confirmed requests to the actual model API
- Return responses back to Toolathlon
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Import safety-confirmation wrapper
try:
    from safety_confirmation import SafetyConfirmationAPI
    from inspect_ai.model import GenerateConfig
    from inspect_ai.model import ChatMessageUser, ChatMessageSystem
except ImportError:
    print("Error: safety-confirmation package not found")
    print("Install with: pip install -e ../safety_confirmation")
    sys.exit(1)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Get actual API endpoint from environment
ACTUAL_API_BASE = os.environ.get("SAFETY_PROXY_ACTUAL_BASE", "https://api.openai.com/v1")
ACTUAL_API_KEY = os.environ.get("SAFETY_PROXY_ACTUAL_KEY", os.environ.get("OPENAI_API_KEY", ""))
SAFETY_CONFIRMATION_VERSION = os.environ.get("SAFETY_CONFIRMATION_VERSION", "v6")
SAFETY_MAX_RECONSIDER = int(os.environ.get("SAFETY_MAX_RECONSIDER", "3"))
SAFETY_N_REQUIRED = int(os.environ.get("SAFETY_N_REQUIRED", "1"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    logger.info("Starting Toolathlon Safety Proxy...")
    logger.info(f"Actual API endpoint: {ACTUAL_API_BASE}")
    logger.info(f"Safety confirmation version: {SAFETY_CONFIRMATION_VERSION}")
    yield
    logger.info("Shutting down Toolathlon Safety Proxy...")


app = FastAPI(
    title="Toolathlon Safety Proxy",
    description="Local proxy with safety-confirmation for Toolathlon evaluation",
    lifespan=lifespan,
)


def create_safety_wrapper(model_name: str) -> SafetyConfirmationAPI:
    """Create a safety-confirmation API wrapper for the given model."""
    # Parse the model name to extract base model
    # e.g., "safety-confirmation/gpt-4o" -> "gpt-4o"
    # But we expect the raw model name here

    # Create wrapper with actual API endpoint
    wrapper = SafetyConfirmationAPI(
        model_name=model_name,
        base_url=ACTUAL_API_BASE,
        api_key=ACTUAL_API_KEY,
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=4096,
        ),
    )

    # Configure safety confirmation parameters
    # These are set via environment variables in the wrapper
    os.environ["SAFETY_CONFIRMATION_ENABLED"] = "true"
    os.environ["SAFETY_CONFIRMATION_VERSION"] = SAFETY_CONFIRMATION_VERSION
    os.environ["SAFETY_CONFIRMATION_MAX_RECONSIDER"] = str(SAFETY_MAX_RECONSIDER)
    os.environ["SAFETY_CONFIRMATION_N_REQUIRED"] = str(SAFETY_N_REQUIRED)

    return wrapper


# Cache wrappers per model
_wrappers: dict[str, SafetyConfirmationAPI] = {}


def get_wrapper(model_name: str) -> SafetyConfirmationAPI:
    """Get or create a wrapper for the given model."""
    if model_name not in _wrappers:
        _wrappers[model_name] = create_safety_wrapper(model_name)
    return _wrappers[model_name]


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Handle chat completions requests with safety-confirmation.

    This endpoint:
    1. Receives the request from Toolathlon
    2. Applies safety-confirmation to each message
    3. Generates the response with safety checks
    4. Returns the response in OpenAI format
    """
    try:
        body = await request.json()

        # Extract model name
        model_name = body.get("model", "gpt-4o")
        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.0)
        max_tokens = body.get("max_tokens", 4096)

        logger.info(f"Received request for model: {model_name}, messages: {len(messages)}")

        # Get safety wrapper
        wrapper = get_wrapper(model_name)

        # Convert messages to inspect_ai format
        # OpenAI format: {"role": "user"|"assistant"|"system", "content": "..."}
        # inspect_ai format: ChatMessageUser, ChatMessageAssistant, ChatMessageSystem
        inspect_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                from inspect_ai.model import ChatMessageSystem
                inspect_messages.append(ChatMessageSystem(content=content))
            elif role == "user":
                from inspect_ai.model import ChatMessageUser
                inspect_messages.append(ChatMessageUser(content=content))
            elif role == "assistant":
                from inspect_ai.model import ChatMessageAssistant
                inspect_messages.append(ChatMessageAssistant(content=content))

        # Generate with safety-confirmation
        # The wrapper will internally handle safety confirmation
        from inspect_ai.model import ChatMessageAssistant

        response_content = ""
        async for chunk in wrapper.generate(inspect_messages):
            if isinstance(chunk, ChatMessageAssistant):
                response_content += chunk.content
            elif hasattr(chunk, "text"):
                response_content += chunk.text
            elif isinstance(chunk, str):
                response_content += chunk

        # Return in OpenAI format
        import time
        return JSONResponse({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "")) for m in messages),
                "completion_tokens": len(response_content),
                "total_tokens": sum(len(m.get("content", "")) for m in messages) + len(response_content),
            },
        })

    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: Request):
    """Handle legacy completions requests."""
    # Convert to chat completions format
    body = await request.json()
    prompt = body.get("prompt", "")
    model = body.get("model", "gpt-4o")

    # Forward to chat completions
    return await chat_completions({
        "json": async lambda: {"model": model, "messages": [{"role": "user", "content": prompt}]},
        "await": lambda x: x,
    })


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Toolathlon Safety Proxy",
        "status": "running",
        "actual_api": ACTUAL_API_BASE,
        "safety_confirmation": SAFETY_CONFIRMATION_VERSION,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    parser = argparse.ArgumentParser(description="Toolathlon Safety Proxy with Safety-Confirmation")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to (default: 8765)")
    parser.add_argument("--actual-base", default=None, help="Actual API base URL (env: SAFETY_PROXY_ACTUAL_BASE)")
    parser.add_argument("--actual-key", default=None, help="Actual API key (env: SAFETY_PROXY_ACTUAL_KEY)")
    parser.add_argument("--safety-version", default="v6", help="Safety confirmation version (env: SAFETY_CONFIRMATION_VERSION)")
    parser.add_argument("--max-reconsider", type=int, default=3, help="Max reconsideration attempts (env: SAFETY_MAX_RECONSIDER)")
    parser.add_argument("--n-required", type=int, default=1, help="Number of confirmations required (env: SAFETY_N_REQUIRED)")

    args = parser.parse_args()

    # Set environment variables from args
    if args.actual_base:
        os.environ["SAFETY_PROXY_ACTUAL_BASE"] = args.actual_base
    if args.actual_key:
        os.environ["SAFETY_PROXY_ACTUAL_KEY"] = args.actual_key
    if args.safety_version:
        os.environ["SAFETY_CONFIRMATION_VERSION"] = args.safety_version
    os.environ["SAFETY_MAX_RECONSIDER"] = str(args.max_reconsider)
    os.environ["SAFETY_N_REQUIRED"] = str(args.n_required)

    # Run the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
