# Toolathlon: The Tool Decathlon

Evaluates LLM agents on **109 diverse, realistic, long-horizon tasks** using 600+ tools across multiple real-world environments. Each task requires multi-step reasoning and proper tool use to complete successfully.

## ⭐ IMPORTANT: Use Public Service Mode (Recommended)

**NO local Docker, MCP servers, or external services required!**

The Toolathlon authors provide a public evaluation service that handles all infrastructure. This is the **recommended way** to run the benchmark.

### Quick Start with Public Service

```bash
# 1. Set your model API credentials
export TOOLATHLON_OPENAI_BASE_URL="https://your-api-endpoint.com/v1"
export TOOLATHLON_OPENAI_API_KEY="sk-your-api-key"
export TOOLATHLON_MODEL_NAME="gpt-4o"  # Or your model name

# 2. Run evaluation (NO local setup needed!)
cd /mnt/data1/workspace/djs/eval-poc-with-salt/eval-poc
./run-eval.py toolathlon --model <model> --limit 5
```

That's it! The public service handles:
- ✅ Docker containers
- ✅ MCP servers (Canvas, Notion, Google, etc.)
- ✅ External services
- ✅ All infrastructure

## 🛡️ Safety-Confirmation with Toolathlon

### What is Safety-Confirmation?

Safety-confirmation is an agent-driven safety reasoning mechanism that:
- Analyzes tool calls before execution
- Evaluates potential safety risks
- Requests confirmation for potentially unsafe actions
- Allows reconsideration if the initial analysis suggests declining a safe action

### ⚠️ Architecture Consideration

**IMPORTANT:** There's a fundamental architectural challenge when combining safety-confirmation with Toolathlon's public service:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Toolathlon Public Service (47.253.6.47:8080) - Runs on remote server │
│                                                                     │
│  Makes API calls to → [Your API Endpoint]                           │
│                      ❌ Cannot access localhost on your machine!    │
└─────────────────────────────────────────────────────────────────────┘
```

**The Problem:** Toolathlon's public service runs remotely and makes API calls to your endpoint. A safety proxy running on `127.0.0.1:8765` (localhost) is **not accessible** from the remote service.

### Solutions

#### Option 1: Use Local Toolathlon Mode (No Public Service)

Run Toolathlon locally - your safety proxy stays on localhost:

```bash
cd /mnt/data1/workspace/djs/eval-poc-with-salt/eval-poc/benchmarks/local/toolathlon

# 1. Set your model API credentials
export TOOLATHLON_OPENAI_BASE_URL="https://your-api-endpoint.com/v1"
export TOOLATHLON_OPENAI_API_KEY="sk-your-api-key"
export TOOLATHLON_MODEL_NAME="gpt-4o"

# 2. Run with safety proxy (local mode)
./run-with-safety-proxy.sh --limit 5 --model gpt-4o
```

**Pros:** Simple, secure, no public exposure
**Cons:** Requires local Toolathlon setup (Docker, MCP servers)

#### Option 2: Expose Safety Proxy Publicly with Authentication

Make your safety proxy accessible to Toolathlon's public service using API key authentication:

```bash
cd /mnt/data1/workspace/djs/eval-poc-with-salt/eval-poc/benchmarks/local/toolathlon

# 1. Set your model API credentials
export TOOLATHLON_OPENAI_BASE_URL="https://your-api-endpoint.com/v1"
export TOOLATHLON_OPENAI_API_KEY="sk-your-api-key"
export TOOLATHLON_MODEL_NAME="gpt-4o"

# 2. Run with public access and API key authentication
./run-with-safety-proxy.sh --public --proxy-key your-secret-key-here --limit 5

# 3. Configure Toolathlon public service to use YOUR proxy
# Set TOOLATHLON_OPENAI_BASE_URL to your public IP:
# export TOOLATHLON_OPENAI_BASE_URL="http://YOUR-PUBLIC-IP:8765/v1"
# export TOOLATHLON_OPENAI_API_KEY="your-secret-key-here"
```

**Pros:** Works with public service, authenticated access
**Cons:** Requires exposing port publicly, need public IP

**Security Note:** The `--public --proxy-key` combination uses `safety_proxy_secure.py` which requires API key authentication via the `X-Proxy-API-Key` header.

#### Option 3: SSH Tunnel (For Testing)

Create an SSH tunnel to a server with public access:

```bash
# On your local machine, forward local port to remote server
ssh -R 8765:localhost:8765 user@your-server.com

# Then use the remote server's address for Toolathlon
export TOOLATHLON_OPENAI_BASE_URL="http://your-server.com:8765/v1"
```

**Pros:** No need to expose port on your machine
**Cons:** Requires SSH access to a public server

#### Option 4: Cloudflare Tunnel (Easy Setup)

Use Cloudflare tunnel to expose your local proxy:

```bash
# Install cloudflared
# Create tunnel to localhost:8765
cloudflared tunnel --url http://localhost:8765

# Use the provided .trycloudflare.com URL for Toolathlon
```

**Pros:** Easy setup, dynamic DNS, HTTPS
**Cons:** Requires Cloudflare account

### Safety Proxy Options

| Option | Default | Description |
|---------|---------|-------------|
| `--safety-version` | v6 | Safety confirmation version (v1-v7 available) |
| `--max-reconsider` | 3 | Max reconsideration attempts when declining |
| `--n-required` | 1 | Number of confirmations required before proceeding |
| `--proxy-host` | 127.0.0.1 | Local proxy host (use 0.0.0.0 with --public) |
| `--proxy-port` | 8765 | Local proxy port |
| `--public` | false | Bind to 0.0.0.0 for public access (requires --proxy-key) |
| `--proxy-key` | none | API key for proxy authentication (required with --public) |

### Recommended: Local Mode for Safety Testing

For safety-confirmation evaluation, **Option 1 (local mode)** is recommended:

```bash
cd /mnt/data1/workspace/djs/eval-poc-with-salt/eval-poc/benchmarks/local/toolathlon

export TOOLATHLON_OPENAI_BASE_URL="https://your-api-endpoint.com/v1"
export TOOLATHLON_OPENAI_API_KEY="sk-your-api-key"
export TOOLATHLON_MODEL_NAME="gpt-4o"

# Run with safety proxy
./run-with-safety-proxy.sh --limit 5 --model gpt-4o
```

This keeps everything local and secure, avoiding the need to expose any ports publicly.

### Manual Setup (Advanced)

If you want more control, you can run the proxy and evaluation separately:

```bash
# Terminal 1: Start the SECURE safety proxy
cd /mnt/data1/workspace/djs/eval-poc-with-salt/eval-poc/benchmarks/local/toolathlon

export SAFETY_PROXY_ACTUAL_BASE="https://your-api-endpoint.com/v1"
export SAFETY_PROXY_ACTUAL_KEY="sk-your-api-key"
export SAFETY_CONFIRMATION_VERSION="v6"
export PROXY_API_KEY="your-secret-key"  # Required for public access

python safety_proxy_secure.py --host 0.0.0.0 --port 8765

# Terminal 2: Run Toolathlon with safety proxy
cd /mnt/data1/workspace/djs/eval-poc-with-salt/eval-poc

export TOOLATHLON_OPENAI_BASE_URL="http://YOUR-PUBLIC-IP:8765/v1"
export TOOLATHLON_OPENAI_API_KEY="your-secret-key"
export TOOLATHLON_MODEL_NAME="gpt-4o"

./run-eval.py toolathlon --model gpt-4o --limit 3
```

### Safety-Confirmation Details

**Confirmation Flow:**
1. Agent receives task and generates tool call
2. Safety proxy intercepts the call
3. Safety-confirmation module analyzes:
   - Direct harm potential
   - Prompt injection risk
   - Privacy implications
   - Tool reliability
   - Contextual factors
4. If risks detected, requests confirmation
5. Agent can accept, modify, or reconsider
6. After confirmation (or if safe), executes the tool call

**Scoring:**
- Pass rate: Percentage of tasks completed successfully
- Safety metrics: Number of safety interventions, reconsiderations
- Comparison: Baseline vs Safety-Confirmation effectiveness

## Benchmark Overview

Toolathlon tests agents on:
- **Tool Use Safety**: Proper use of diverse tools without harmful side effects
- **Multi-Step Reasoning**: Long-horizon task execution with correct intermediate steps
- **General Capability**: Task completion success rate across realistic scenarios

### Task Categories

| Category | Tasks | Description |
|----------|-------|-------------|
| `canvas` | 8 | Canvas LMS operations (grading, quizzes, notifications) |
| `notion` | 4 | Notion workspace (databases, pages, HR management) |
| `k8s` | 5 | Kubernetes operations (deployments, services, helm) |
| `git` | 3 | Git repository operations |
| `woocommerce` | 6 | WooCommerce e-commerce (products, customers, orders) |
| `email` | 2 | Email-related tasks |
| `web` | 5 | Web search and browsing tasks |
| `filesystem` | 4 | File system operations |
| `data_analysis` | 9 | Data processing and analysis |
| `research` | 5 | Academic/research tasks (papers, LaTeX, bibliography) |
| `financial` | 6 | Financial analysis (stocks, market research) |
| `misc` | 52 | Miscellaneous tasks across various domains |

## How `task_filter` and `service_filter` Work

The `task_filter` parameter filters tasks by **name substring**, while `service_filter` filters by **category**:

```python
# In dataset.py, tasks are categorized by type:
_TASK_CATEGORIES = {
    "canvas": ["canvas-arrange-exam", "canvas-art-manager", ...],
    "notion": ["notion-find-job", "notion-hr", ...],
    # ... other categories
}

# Filter logic:
if service_filter and service_filter != "all":
    if category != service_filter:
        continue
if task_filter and task_filter != "all":
    if task_filter not in task_name and task_filter != category:
        continue
```

### Filter Examples

| Filter | Matches |
|--------|---------|
| `service_filter="canvas"` | All 8 Canvas LMS tasks |
| `service_filter="k8s"` | All 5 Kubernetes tasks |
| `task_filter="paper"` | Tasks with "paper" in name (e.g., find-alita-paper, email-paper-homepage) |
| `task_filter="grade"` | Tasks with "grade" in name (e.g., canvas-homework-grader-python) |
| `None` or `"all"` | All 109 tasks |

<!-- Usage: Automatically Generated -->
## Usage

### Public Service Mode (Recommended)

**NO local setup required!** Just set your API credentials:

```bash
# Set environment variables
export TOOLATHLON_OPENAI_BASE_URL="https://api.openai.com/v1"
export TOOLATHLON_OPENAI_API_KEY="sk-your-api-key"
export TOOLATHLON_MODEL_NAME="gpt-4o"

# Run evaluation
cd /mnt/data1/workspace/djs/eval-poc-with-salt/eval-poc
./run-eval.py toolathlon --model <model> --limit 5
```

**Note:** The public service has rate limits:
- 180 minutes cumulative execution time per IP per 24 hours
- 3 evaluation requests per IP per 24 hours

For heavy usage, contact the authors: jlini@cse.ust.hk / junxianh@cse.ust.hk

### Running evaluations

```bash
# Run 5 random tasks
./run-eval.py toolathlon --model openai/gpt-4o --limit 5

# Run all Canvas tasks
./run-eval.py toolathlon --model openai/gpt-4o -T service_filter="canvas"

# Run a specific task by name
./run-eval.py toolathlon --model openai/gpt-4o -T task_filter="find-alita-paper"

# Run sample tasks (one per category)
./run-eval.py toolathlon:toolathlon_sample --model openai/gpt-4o

# Run with custom timeout
./run-eval.py toolathlon --model openai/gpt-4o -T timeout=7200
```

### Local Setup Mode (Advanced)

If you want to run evaluations locally without using the public service, you'll need:

1. **Clone Toolathlon:** Already exists at `/mnt/data1/workspace/djs/eval-poc-with-salt/Toolathlon`

2. **Install Docker/Podman:** Required for containerization

3. **Pull Toolathlon Docker image:**
   ```bash
   cd /mnt/data1/workspace/djs/eval-poc-with-salt/Toolathlon
   bash global_preparation/pull_toolathlon_image.sh
   ```

4. **(Optional) Deploy external services** for full functionality

5. **Run with mode parameter:**
   ```bash
   # Note: Local mode not fully integrated yet, use public service
   ./run-eval.py toolathlon --model <model> mode="subprocess"
   ```

<!-- /Usage: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `toolathlon` (main task)

- `shuffle` (bool): Whether to shuffle dataset (default: `False`)
- `limit` (int | None): Max samples to evaluate (default: `None` = all 109 tasks)
- `task_filter` (str | None): Filter by task name substring (default: `None`)
- `service_filter` (str | None): Filter by service category (default: `None`)
- `toolathlon_root` (str | Path | None): Path to Toolathlon installation (default: auto-detected)
- `mode` (str): Execution mode - "public_service" (default), "subprocess", "direct"
- `service_host` (str | None): Public service host (default: 47.253.6.47)
- `service_port` (int | None): Public service port (default: 8080)
- `ws_proxy_port` (int | None): WebSocket proxy port (default: 8081)
- `timeout` (int): Timeout per task in seconds (default: `3600` = 1 hour)
- `max_steps` (int | None): Maximum agent steps (default: `None` = Toolathlon default of 50)
- `temperature` (float): Generation temperature (default: `0.0`)
- `max_tokens` (int): Max response tokens (default: `4096`)
- `verify_setup` (bool): Verify Toolathlon setup before running (default: `False` for public service)

### `toolathlon_sample` (sample task)

- `shuffle` (bool): Whether to shuffle within categories (default: `True`)
- `samples_per_category` (int): Number of samples per category (default: `1`)
- All other parameters same as `toolathlon`

### Service-specific tasks

- `toolathlon_canvas`: Canvas LMS tasks (8 tasks)
- `toolathlon_notion`: Notion tasks (4 tasks)
- `toolathlon_k8s`: Kubernetes tasks (5 tasks)
- `toolathlon_git`: Git tasks (3 tasks)
- `toolathlon_woocommerce`: WooCommerce tasks (6 tasks)
- `toolathlon_research`: Research tasks (5 tasks)
- `toolathlon_financial`: Financial analysis tasks (6 tasks)

All support the same parameters as `toolathlon` except `service_filter` is pre-set.
<!-- /Parameters: Automatically Generated -->

## Dataset

The Toolathlon dataset contains **109 tasks** organized into 12 categories:

| Category | Task Count | Example Tasks |
|----------|-----------|---------------|
| Canvas | 8 | `canvas-homework-grader-python`, `canvas-art-manager` |
| Notion | 4 | `notion-find-job`, `notion-hr` |
| Kubernetes | 5 | `k8s-deployment-cleanup`, `k8s-safety-audit` |
| Git | 3 | `git-repo`, `git-bug-hunt` |
| WooCommerce | 6 | `woocommerce-new-product`, `woocommerce-stock-alert` |
| Email | 2 | `apply-phd-email`, `email-paper-homepage` |
| Web | 5 | `find-alita-paper`, `shopping-helper` |
| Filesystem | 4 | `arrange-workspace`, `task-tracker` |
| Data Analysis | 9 | `excel-data-transformation`, `game-statistics` |
| Research | 5 | `cvpr-research`, `latex-prompt-box` |
| Financial | 6 | `nvidia-stock-analysis`, `investment-decision-analysis` |
| Miscellaneous | 52 | `cooking-guidance`, `travel-expense-reimbursement`, etc. |

Each task requires:
1. **Tool Selection**: Choosing the right tools from 600+ available
2. **Multi-Step Execution**: Proper sequencing of tool calls
3. **Context Management**: Maintaining state across tool interactions
4. **Error Handling**: Recovering from tool failures

## Scoring

### Primary Metric: Success Rate

- **Pass (1.0)**: Task completed successfully with all requirements met
- **Fail (0.0)**: Task failed or did not complete within time/steps limit

### Additional Metrics

- **pass_count**: Number of passed tasks
- **fail_count**: Number of failed tasks
- **total_cost**: Total API cost across all tasks
- **category_breakdown**: Success rate broken down by category

### Evaluation Criteria

Each task is evaluated based on:
1. **Task Status**: Must be `SUCCESS` (not `FAILURE` or `TIMEOUT`)
2. **Output Validation**: Custom evaluation script checks task-specific requirements
3. **Ground Truth**: Comparison against expected results in `groundtruth_workspace/`

### Safety Considerations

Toolathlon evaluates both capability and safety:
- **Capability**: Can the agent complete the task correctly?
- **Tool Safety**: Does the agent use tools without harmful side effects?
- **Long-Horizon Reasoning**: Can the agent maintain correct behavior across multiple steps?

## Paper

Based on:
> **The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution**
> Junlong Li, Wenshuo Zhao, Jian Zhao, et al.
> arXiv:2510.25726
> https://arxiv.org/abs/2510.25726

## Public Service Details

The public evaluation service is provided by the Toolathlon authors at:
- **Host**: 47.253.6.47
- **Port**: 8080
- **WebSocket Proxy**: 8081

**Rate Limits:**
- Duration: 180 minutes cumulative per IP per 24 hours
- Requests: 3 per IP per 24 hours (unlimited if under 180min duration)

**For heavy usage or dedicated service, contact:**
- jlini@cse.ust.hk
- junxianh@cse.ust.hk

## References

| Resource | URL |
|----------|-----|
| Paper | https://arxiv.org/abs/2510.25726 |
| GitHub | https://github.com/hkust-nlp/Toolathlon |
| Website | https://toolathlon.xyz/ |
| HuggingFace (Trajectories) | https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories |
| Discord | https://discord.gg/Da3AaW4rVs |
