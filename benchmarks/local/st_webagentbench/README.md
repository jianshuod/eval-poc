# ST-WebAgentBench: Safety & Trustworthiness in Web Agents

A benchmark for evaluating not only whether web agents complete tasks, but whether they do so while respecting safety and trustworthiness (ST) policies — the constraints that govern real enterprise deployments.

## Integration Modes

This benchmark supports two evaluation modes:

### Mode 1: Simplified Safety-Confirmation Testing (Default)

The default mode tests safety reasoning without requiring full browser automation infrastructure:

- **Purpose**: Test safety-confirmation and safety-lookahead mechanisms
- **Requirements**: Only LLM API access
- **Use Case**: Evaluating how models reason about safety policies before taking actions
- **Limitations**: No actual browser automation, uses simulated evaluation

This mode is useful for:
- Testing safety-confirmation meta-tool behavior
- Evaluating safety-lookahead world model reasoning
- Quick iteration without browser dependencies

### Mode 2: Full BrowserGym Integration (Optional)

For complete end-to-end evaluation with real browser automation:

- **Purpose**: Full web agent evaluation with actual browser automation
- **Requirements**: BrowserGym, Playwright, external web services
- **Use Case**: Production-quality evaluation matching the original benchmark

See "Full BrowserGym Integration" section below for setup details.

## Overview

ST-WebAgentBench contains **375 realistic enterprise tasks** across three applications (GitLab, ShoppingAdmin, SuiteCRM), paired with **policy instances** spanning six safety dimensions. Every task is evaluated along two orthogonal axes:

- **CR (Completion Rate)**: Fraction of tasks completed successfully (ignoring policies)
- **CuP (Completion under Policy)**: Fraction of tasks completed with **zero** policy violations

The gap between CR and CuP is the core measurement — agents lose up to **38%** of their raw successes when enforcing policies, revealing hidden safety gaps that task-only benchmarks miss entirely.

## How Domain Filter Works

The benchmark supports filtering by domain and tier:

```python
# Domain mapping based on task ID ranges
_DOMAIN_RANGES = {
    "gitlab": range(0, 197),      # 197 tasks
    "shopping_admin": range(197, 205),  # 8 tasks
    "suitecrm": range(205, 375),  # 170 tasks
}

# SuiteCRM tier mapping
_SUITECRM_TIERS = {
    "easy": range(235, 255),      # 20 tasks
    "medium": range(255, 275),    # 20 tasks
    "hard": range(275, 295),      # 20 tasks
}
```

### Filter Examples

| Filter | Task IDs | Description |
|--------|----------|-------------|
| `domain="all"` | 0-374 | All 375 tasks |
| `domain="gitlab"` | 0-196 | GitLab tasks only |
| `domain="suitecrm"` | 205-374 | SuiteCRM tasks only |
| `tier="easy"` | 235-254 | Easy SuiteCRM tasks |
| `tier="medium"` | 255-274 | Medium SuiteCRM tasks |
| `tier="hard"` | 275-294 | Hard SuiteCRM tasks |

## Environment Setup

**Important:** ST-WebAgentBench requires external web applications to be running before evaluation.

### 1. SuiteCRM (Docker)

```bash
cd /mnt/data1/workspace/djs/eval-poc-with-salt/ST-WebAgentBench/suitecrm_setup
docker compose up -d

# Load demo data (first time only)
cd init-db
docker exec -i suitecrm_setup-mariadb-1 mysql -u bn_suitecrm -pbitnami123 < demo_data.sql

# Verify: http://localhost:8080 (user/bitnami)
```

### 2. GitLab & ShoppingAdmin (WebArena)

Option 1: **WebArena AWS AMI** (recommended)
- Follow: https://github.com/web-arena-x/webarena/tree/main/environment_docker
- Services exposed on ports 8081 (GitLab) and 8083 (ShoppingAdmin)

Option 2: **Local Docker**
- Clone and run WebArena locally
- More complex setup, see WebArena repository

### 3. Configure URLs

Create/Edit `.env` in ST-WebAgentBench directory:

```bash
# GitLab
GITLAB_URL=http://localhost:8081

# ShoppingAdmin
SHOPPING_ADMIN_URL=http://localhost:8083

# SuiteCRM
SUITECRM_URL=http://localhost:8080
```

### 4. Verify Services

```bash
curl http://localhost:8080  # SuiteCRM
curl http://localhost:8081  # GitLab
curl http://localhost:8083  # ShoppingAdmin
```

**Note:** Unlike other agent benchmarks, ST-WebAgentBench does **not** use a local Docker Compose sandbox. BrowserGym manages the browser environment (Playwright + Chromium) and connects to the external web applications.

<!-- Usage: Automatically Generated -->
## Usage

### Installation

```bash
# 1. Install BrowserGym and Playwright
pip install browsergym==0.7.0
pip install playwright==1.52.0
playwright install chromium

# 2. Install ST-WebAgentBench package
cd /mnt/data1/workspace/djs/eval-poc-with-salt/ST-WebAgentBench/browsergym/stwebagentbench
pip install -e ./

# 3. Provision web applications (GitLab, SuiteCRM, ShoppingAdmin)
#    - GitLab/ShoppingAdmin: Use WebArena AWS AMI
#    - SuiteCRM: See ST-WebAgentBench/suitecrm_setup/README.md
#    - Configure .env with service URLs and credentials
```

### Running evaluations

```bash
# Run all tasks
./run-eval.py st_webagentbench --model <model>

# Run specific domain
./run-eval.py st_webagentbench_gitlab --model <model>
./run-eval.py st_webagentbench_suitecrm --model <model>

# Run SuiteCRM by tier
./run-eval.py st_webagentbench_easy --model <model>
./run-eval.py st_webagentbench_medium --model <model>
./run-eval.py st_webagentbench_hard --model <model>

# With limit for testing
./run-eval.py st_webagentbench --model <model> --limit 5
```
<!-- /Usage: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `st_webagentbench`

- `domain` (Literal["all", "gitlab", "suitecrm", "shopping_admin"]): Filter by domain (default: `"all"`)
- `tier` (Literal["all", "easy", "medium", "hard"]): Filter SuiteCRM by tier (default: `"all"`)
- `task_id` (int | None): Run a specific task ID 0-374 (default: `None`)
- `shuffle` (bool): Whether to shuffle the dataset (default: `False`)
- `limit` (int | None): Maximum number of samples to evaluate (default: `None`)
- `max_steps` (int): Maximum number of agent steps per task (default: `50`)
- `temperature` (float): Generation temperature (default: `0.0`)
- `max_tokens` (int): Maximum tokens for LLM responses (default: `4096`)
- `sandbox` (SandboxEnvironmentType | None): Sandbox environment type (default: `None`)
<!-- /Parameters: Automatically Generated -->

## Dataset

### Task Distribution

| Application | Task IDs | Tasks | Policies |
|:-----------|:---------|:-----:|:--------:|
| **GitLab** | 0-196 | 197 | 1,511 |
| **ShoppingAdmin** | 197-204 | 8 | 101 |
| **SuiteCRM** | 205-374 | 170 | — |
| **Total** | 0-374 | **375** | — |

### SuiteCRM Tier Structure

| Tier | Task IDs | Tasks | Avg Policies |
|:-----|:---------|:-----:|:------------:|
| **Easy** | 235-254 | 20 | 5.6 |
| **Medium** | 255-274 | 20 | 9.1 |
| **Hard** | 275-294 | 20 | 14.1 |

### Modality Challenge Tasks

| Group | Task IDs | Count | Concept |
|:------|:---------|:-----:|:--------|
| **Vision-advantage** | 295-334 | 40 | Information visible in screenshots, hidden from AXTree |
| **DOM-advantage** | 335-374 | 40 | Information in AXTree, visually obscured |

## Safety Dimensions

All policies map to one of six orthogonal safety dimensions:

| Dimension | Policies | Description |
|:----------|:--------:|:------------|
| **Boundary & Scope** | 959 | Stay within permitted resources and navigation boundaries |
| **Strict Execution** | 795 | Execute exactly as instructed — no additions or fabrications |
| **User Consent** | 274 | Obtain explicit approval before sensitive or irreversible actions |
| **Robustness & Security** | 274 | Resist prompt injection, jailbreaking, and credential leakage |
| **Hierarchy Adherence** | 132 | Follow Organization > User > Task policy priority order |
| **Error Handling** | 118 | Handle errors, missing parameters, and unexpected states gracefully |

## Scoring

### Primary Metrics

| Metric | Definition |
|:-------|:-----------|
| **CR** | Completion Rate — fraction of tasks completed successfully (ignoring policies) |
| **CuP** | Completion under Policy — fraction of tasks completed with **zero** policy violations |
| **Risk Ratio** | Per-dimension violation rate: violations / total policy instances |

### Evaluator Types

The benchmark uses 9 specialized evaluator types:

| Evaluator | What It Detects |
|:----------|:----------------|
| `is_ask_the_user` | Missing user consent before sensitive actions |
| `is_url_match` | Navigation outside permitted boundaries |
| `element_action_match` | Clicking/filling prohibited elements |
| `is_sequence_match` | Wrong action ordering |
| `is_program_html` | Unexpected page content (wrong execution path) |
| `is_input_hallucination` | Fabricated form values not in task |
| `is_sensitive_data` | Credential or PII leakage |
| `is_action_count` | Exceeding action budgets |
| `popup_error` | Error dialogs from invalid actions |

### Score Interpretation

- **CuP = 1.0 (100%)**: All tasks completed with zero policy violations — **ideal safety**
- **CuP < CR**: Agent completes tasks unsafely — **hidden safety risk**
- **CR - CuP gap > 20%**: Significant policy violations — **high risk**

## Paper

Based on:
> ST-WebAgentBench: A Benchmark for Evaluating Safety & Trustworthiness in Web Agents
> https://arxiv.org/abs/2410.06703
>
> Levy, I., Wiesel, B., Marreed, S., Oved, A., Yaeli, A., & Shlomov, S. (2025). ICLR.

### Key Findings

- Agents lose up to **38%** of raw successes when enforcing policies (CR → CuP)
- **User Consent** and **Strict Execution** are the most violated dimensions
- Policy complexity (tier progression) significantly impacts CuP but not CR
- Vision-based and DOM-based agents show systematic blind spots

## References

| Component | Path |
|:----------|:-----|
| Original Repo | `/mnt/data1/workspace/djs/eval-poc-with-salt/ST-WebAgentBench` |
| Dataset | `ST-WebAgentBench/stwebagentbench/test.raw.json` |
| BrowserGym Plugin | `ST-WebAgentBench/browsergym/stwebagentbench/` |
| Evaluation Harness | `ST-WebAgentBench/stwebagentbench/evaluation_harness/` |
| Local Integration | `/mnt/data1/workspace/djs/eval-poc-with-salt/eval-poc/benchmarks/local/st_webagentbench/` |

## Implementation Notes

### Current Status (Simplified Mode)

The current integration uses a simplified solver that:

1. **Loads task configurations** from the ST-WebAgentBench dataset
2. **Presents task goals and policies** to the model via ChatMessage objects
3. **Collects safety reasoning** without browser automation
4. **Scores responses** based on safety policy analysis

This mode is designed for testing safety-confirmation and safety-lookahead mechanisms without requiring the full BrowserGym infrastructure.

### Full BrowserGym Integration (Future Work)

To enable full browser automation, the following would be required:

1. **Install ST-WebAgentBench as a package**:
   ```bash
   cd /mnt/data1/workspace/djs/eval-poc-with-salt/ST-WebAgentBench/browsergym/stwebagentbench
   pip install -e ./
   ```

2. **Provision web services** (see Environment Setup above)

3. **Configure environment variables**:
   ```bash
   export WA_SUITECRM=http://localhost:8080
   export WA_GITLAB=http://localhost:8081
   export WA_SHOPPING_ADMIN=http://localhost:8083
   ```

4. **Modify solver.py** to use BrowserGym environments

The full BrowserGym integration is complex because:
- Requires `stwebagentbench.browser_env` module with all dependencies
- Needs external web services (GitLab, SuiteCRM, ShoppingAdmin) to be running
- Involves complex action parsing and environment management

For most safety testing purposes, the simplified mode is sufficient.

### Code Structure

```
benchmarks/local/st_webagentbench/
├── __init__.py              # Package initialization
├── st_webagentbench.py      # Main task definition
├── solver.py                # Safety reasoning solver
├── scorer.py                # CuP and CR metrics
├── dataset.py               # Task loading from test.raw.json
├── browsergym_src/          # Copied BrowserGym integration (for future use)
│   └── browsergym/stwebagentbench/
└── README.md                # This file
``` |
