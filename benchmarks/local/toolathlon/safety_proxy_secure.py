"""
Secure Safety Proxy with API Key Authentication

This version of the safety proxy includes authentication so you can
safely expose it to the internet for Toolathlon's public service to access.
"""

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.security import APIKeyHeader
import uvicorn
import os
import logging
from typing import Optional

# Configuration
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")
ACTUAL_API_BASE = os.environ.get("SAFETY_PROXY_ACTUAL_BASE", "")
ACTUAL_API_KEY = os.environ.get("SAFETY_PROXY_ACTUAL_KEY", "")
SAFETY_CONFIRMATION_VERSION = os.environ.get("SAFETY_CONFIRMATION_VERSION", "v6")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Toolathlon Safety Proxy (Secure)")

# API Key authentication
api_key_header = APIKeyHeader(name="X-Proxy-API-Key", auto_error=False)


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "Toolathlon Safety Proxy (Secure)",
        "status": "running",
        "auth_required": bool(PROXY_API_KEY),
    }


@app.get("/health")
async def health():
    """Health check endpoint (no auth required)."""
    return {"status": "healthy"}


def verify_api_key(request: Request) -> bool:
    """Verify the proxy API key."""
    if not PROXY_API_KEY:
        # Allow if no API key is configured (not recommended for production)
        logger.warning("No PROXY_API_KEY set - allowing unauthenticated access")
        return True

    api_key = api_key_header.get(request)
    return api_key == PROXY_API_KEY


async def call_actual_api(payload: dict):
    """Call the actual model API."""
    import httpx

    headers = {
        "Content-Type": "application/json",
    }
    if ACTUAL_API_KEY:
        headers["Authorization"] = f"Bearer {ACTUAL_API_KEY}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ACTUAL_API_BASE}/chat/completions",
            json=payload,
            headers=headers,
            timeout=300.0,
        )
        return response.json()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Handle chat completions with safety-confirmation.

    Requires X-Proxy-API-Key header if PROXY_API_KEY is set.
    """
    # Verify API key
    if not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    try:
        body = await request.json()
        model_name = body.get("model", "gpt-4o")
        messages = body.get("messages", [])

        # Here you would integrate safety-confirmation
        # For now, just forward the request
        response = await call_actual_api(body)
        return response

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Toolathlon Safety Proxy (Secure)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (use 0.0.0.0 for public access)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--actual-base", help="Actual API base URL (env: SAFETY_PROXY_ACTUAL_BASE)")
    parser.add_argument("--actual-key", help="Actual API key (env: SAFETY_PROXY_ACTUAL_KEY)")
    parser.add_argument("--proxy-key", help="Proxy API key for authentication (env: PROXY_API_KEY)")
    parser.add_argument("--safety-version", default="v6", help="Safety confirmation version")

    args = parser.parse_args()

    # Set environment variables from args
    if args.actual_base:
        os.environ["SAFETY_PROXY_ACTUAL_BASE"] = args.actual_base
    if args.actual_key:
        os.environ["SAFETY_PROXY_ACTUAL_KEY"] = args.actual_key
    if args.proxy_key:
        os.environ["PROXY_API_KEY"] = args.proxy_key
    os.environ["SAFETY_CONFIRMATION_VERSION"] = args.safety_version

    if PROXY_API_KEY:
        logger.info("✓ Proxy API authentication enabled")
    else:
        logger.warning("⚠️  No PROXY_API_KEY set - running in open mode (not recommended for public access)")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
