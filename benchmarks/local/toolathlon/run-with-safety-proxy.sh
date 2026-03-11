#!/bin/bash
#
# Toolathlon Safety Proxy Runner (Secure Version)
#
# This script sets up and runs the SAFETY proxy server for Toolathlon evaluation
# with safety-confirmation enabled.
#
# IMPORTANT: Network Access Considerations
# ========================================
# - Default: Runs on localhost (127.0.0.1) - only accessible from this machine
# - With --public: Binds to 0.0.0.0 - accessible from anywhere
# - For public service mode: You MUST expose the proxy publicly AND set --proxy-key
#
# Usage:
#   ./run-with-safety-proxy.sh [options] [eval_args...]
#
# Examples:
#   # Local evaluation (not suitable for Toolathlon public service)
#   ./run-with-safety-proxy.sh --limit 3 --model gpt-4o
#
#   # Public access for Toolathlon public service (with API key authentication)
#   ./run-with-safety-proxy.sh --public --proxy-key your-secret-key --limit 3
#
#   # Run with custom safety confirmation settings
#   ./run-with-safety-proxy.sh --safety-version v7 --max-reconsider 5 --limit 3
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
SAFETY_PROXY_HOST="127.0.0.1"
SAFETY_PROXY_PORT=8765
SAFETY_VERSION="v6"
MAX_RECONSIDER=3
N_REQUIRED=1
MODEL_NAME="${TOOLATHLON_MODEL_NAME:-gpt-4o}"
BASE_URL="${TOOLATHLON_OPENAI_BASE_URL:-}"
API_KEY="${TOOLATHLON_OPENAI_API_KEY:-}"
PROXY_API_KEY="${PROXY_API_KEY:-}"
PROXY_PID=""
PUBLIC_ACCESS=false

# Cleanup function
cleanup() {
    echo ""
    if [ -n "$PROXY_PID" ]; then
        echo "Stopping safety proxy (PID: $PROXY_PID)..."
        kill "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
        echo "✓ Safety proxy stopped"
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Parse arguments
EVAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --safety-version)
            SAFETY_VERSION="$2"
            shift 2
            ;;
        --max-reconsider)
            MAX_RECONSIDER="$2"
            shift 2
            ;;
        --n-required)
            N_REQUIRED="$2"
            shift 2
            ;;
        --proxy-host)
            SAFETY_PROXY_HOST="$2"
            shift 2
            ;;
        --proxy-port)
            SAFETY_PROXY_PORT="$2"
            shift 2
            ;;
        --proxy-key)
            PROXY_API_KEY="$2"
            shift 2
            ;;
        --public)
            SAFETY_PROXY_HOST="0.0.0.0"
            PUBLIC_ACCESS=true
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            EVAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check required dependencies
echo "Checking dependencies..."

# Check for safety_confirmation package
if ! python -c "from safety_confirmation import SafetyConfirmationAPI" 2>/dev/null; then
    echo "❌ safety_confirmation package not found"
    echo ""
    echo "Installing safety_confirmation..."
    cd "$PROJECT_ROOT/../safety_confirmation"
    pip install -e .
    cd "$PROJECT_ROOT"
fi

# Check for fastapi and uvicorn
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing fastapi and uvicorn..."
    pip install fastapi uvicorn
fi

# Check for required environment variables
if [ -z "$BASE_URL" ]; then
    echo "❌ TOOLATHLON_OPENAI_BASE_URL not set"
    echo ""
    echo "Please set your model API endpoint:"
    echo "  export TOOLATHLON_OPENAI_BASE_URL='https://your-api-endpoint.com/v1'"
    exit 1
fi

if [ -z "$API_KEY" ]; then
    echo "⚠️  TOOLATHLON_OPENAI_API_KEY not set (may be required by your API)"
fi

# Warn about public access security
if [ "$PUBLIC_ACCESS" = true ]; then
    echo ""
    echo "⚠️  WARNING: Public Access Enabled"
    echo "=================================="
    echo "The proxy will be accessible from anywhere on 0.0.0.0:$SAFETY_PROXY_PORT"
    echo ""
    if [ -z "$PROXY_API_KEY" ]; then
        echo "❌ ERROR: --proxy-key is REQUIRED when using --public"
        echo ""
        echo "For security, you MUST set an API key to protect your proxy:"
        echo "  $0 --public --proxy-key your-secret-key [other args...]"
        echo ""
        echo "Without authentication, anyone can use your proxy and consume your API quota!"
        exit 1
    fi
    echo "✓ API key authentication enabled"
    echo ""
    echo "To use this proxy with Toolathlon public service, set:"
    echo "  TOOLATHLON_OPENAI_BASE_URL=http://<your-public-ip>:$SAFETY_PROXY_PORT/v1"
    echo "  TOOLATHLON_OPENAI_API_KEY=$PROXY_API_KEY"
    echo ""
fi

# Set environment for safety proxy
export SAFETY_PROXY_ACTUAL_BASE="$BASE_URL"
export SAFETY_PROXY_ACTUAL_KEY="$API_KEY"
export SAFETY_CONFIRMATION_VERSION="$SAFETY_VERSION"
export SAFETY_CONFIRMATION_MAX_RECONSIDER="$MAX_RECONSIDER"
export SAFETY_CONFIRMATION_N_REQUIRED="$N_REQUIRED"
export TOOLATHLON_MODEL_NAME="$MODEL_NAME"

if [ -n "$PROXY_API_KEY" ]; then
    export PROXY_API_KEY="$PROXY_API_KEY"
fi

echo ""
echo "=== Toolathlon Safety Proxy (Secure) ==="
echo "Proxy Host: $SAFETY_PROXY_HOST:$SAFETY_PROXY_PORT"
echo "Actual API: $BASE_URL"
echo "Model: $MODEL_NAME"
echo "Safety Version: $SAFETY_VERSION"
echo "Max Reconsider: $MAX_RECONSIDER"
echo "N Required: $N_REQUIRED"
if [ -n "$PROXY_API_KEY" ]; then
    echo "API Key Auth: ✓ Enabled"
fi
echo ""

# Start safety proxy in background
echo "Starting safety proxy..."
cd "$SCRIPT_DIR"
python safety_proxy_secure.py \
    --host "$SAFETY_PROXY_HOST" \
    --port "$SAFETY_PROXY_PORT" \
    --safety-version "$SAFETY_VERSION" \
    --max-reconsider "$MAX_RECONSIDER" \
    --n-required "$N_REQUIRED" > /tmp/toolathlon_safety_proxy.log 2>&1 &

PROXY_PID=$!
echo "✓ Safety proxy started (PID: $PROXY_PID)"
echo "  Logs: /tmp/toolathlon_safety_proxy.log"

# Wait for proxy to be ready
echo ""
echo "Waiting for proxy to be ready..."
for i in {1..30}; do
    if curl -s "http://$SAFETY_PROXY_HOST:$SAFETY_PROXY_PORT/health" > /dev/null 2>&1; then
        echo "✓ Proxy is ready!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 1
done

# Set Toolathlon to use safety proxy
export TOOLATHLON_OPENAI_BASE_URL="http://$SAFETY_PROXY_HOST:$SAFETY_PROXY_PORT/v1"

if [ -n "$PROXY_API_KEY" ]; then
    export TOOLATHLON_OPENAI_API_KEY="$PROXY_API_KEY"
fi

echo ""
echo "=== Running Toolathlon Evaluation ==="
echo "Using safety proxy: $TOOLATHLON_OPENAI_BASE_URL"
echo ""

# Run the evaluation
cd "$PROJECT_ROOT"
if [ ${#EVAL_ARGS[@]} -eq 0 ]; then
    # No eval args provided, show usage
    echo "No evaluation arguments provided."
    echo ""
    echo "Usage examples:"
    echo "  $0 --limit 5 --model gpt-4o"
    echo "  $0 toolathlon --model gpt-4o --limit 3"
    echo ""
    if [ "$PUBLIC_ACCESS" = false ]; then
        echo "⚠️  Note: Running on localhost. Toolathlon public service cannot access this."
        echo ""
        echo "For Toolathlon public service, use:"
        echo "  $0 --public --proxy-key your-secret-key --limit 3"
        echo ""
        echo "Then set TOOLATHLON_OPENAI_BASE_URL to your public IP."
    fi
    echo ""
    echo "Safety proxy will continue running until you press Ctrl+C"
    echo ""
    echo "To run evaluation, use:"
    echo "  cd $PROJECT_ROOT"
    if [ -n "$PROXY_API_KEY" ]; then
        echo "  TOOLATHLON_OPENAI_API_KEY=$PROXY_API_KEY \\"
        echo "  ./run-eval.py toolathlon --model $MODEL_NAME --limit 3"
    else
        echo "  ./run-eval.py toolathlon --model $MODEL_NAME --limit 3"
    fi
    echo ""
    echo "Press Ctrl+C to stop the safety proxy..."

    # Wait indefinitely (proxy will run until Ctrl+C)
    wait $PROXY_PID
else
    # Run evaluation with provided arguments
    ./run-eval.py "${EVAL_ARGS[@]}"
fi
