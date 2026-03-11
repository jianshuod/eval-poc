#!/usr/bin/env python3
"""
预检查模块 - 在运行评测前检查环境依赖和收集配置

提供细粒度的操作指引，确保"一键测评"真正可行。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


class DependencyType(Enum):
    JUDGE_MODEL = "judge_model"
    HF_NETWORK = "hf_network"
    HF_AUTH = "hf_auth"
    HF_GATED = "hf_gated"  # 需要申请访问的 gated dataset
    DOCKER = "docker"
    K8S = "k8s"
    DATASET_DOWNLOAD = "dataset_download"  # 需要手动下载的数据集
    SERPAPI_KEY = "serpapi_key"  # SerpAPI key for web search tasks


@dataclass
class ActionItem:
    """用户需要执行的操作"""
    title: str
    command: Optional[str] = None  # 需要执行的命令
    url: Optional[str] = None  # 需要访问的链接
    description: Optional[str] = None  # 说明


@dataclass
class BenchmarkRequirement:
    """Benchmark 的依赖需求"""
    benchmark: str
    tasks: list[str]
    dependency: DependencyType
    description: str
    action: Optional[ActionItem] = None  # 具体操作指引
    optional: bool = False
    alternatives: list[DependencyType] = field(default_factory=list)


# ============================================================
# 详细的依赖配置
# ============================================================

BENCHMARK_REQUIREMENTS: list[BenchmarkRequirement] = [
    # Judge Model 依赖
    BenchmarkRequirement(
        benchmark="cyberseceval_2",
        tasks=["cyse2_interpreter_abuse", "cyse2_prompt_injection", "cyse2_vulnerability_exploit"],
        dependency=DependencyType.JUDGE_MODEL,
        description="需要配置 Judge Model 进行 LLM-as-a-judge 评分",
        action=ActionItem(
            title="配置 Judge Model",
            description="在 catalog.yaml 或命令行通过 --judge-model 指定。推荐选项:\n"
                       "  - economy:  gpt-4o-mini  (成本低)\n"
                       "  - balanced: zai-glm-4.7  (推荐)\n"
                       "  - premium:  gpt-4o       (精度高)",
        ),
    ),

    # HuggingFace Gated Dataset - xstest
    BenchmarkRequirement(
        benchmark="xstest",
        tasks=["xstest"],
        dependency=DependencyType.HF_GATED,
        description="xstest 使用受限数据集，需要 HuggingFace 认证和访问申请",
        action=ActionItem(
            title="申请 HuggingFace 数据集访问权限",
            url="https://huggingface.co/datasets/walledai/XSTest",
            description="1. 访问上述链接，点击 'Access repository' 申请访问\n"
                       "2. 等待审批通过（通常即时批准）\n"
                       "3. 设置环境变量: export HF_TOKEN=<your_token>\n"
                       "   获取 Token: https://huggingface.co/settings/tokens",
        ),
    ),

    # GitHub 数据集下载 - strong_reject
    BenchmarkRequirement(
        benchmark="strong_reject",
        tasks=["strong_reject"],
        dependency=DependencyType.DATASET_DOWNLOAD,
        description="strong_reject 需要从 GitHub 下载数据集",
        action=ActionItem(
            title="下载 strong_reject 数据集",
            url="https://raw.githubusercontent.com/alexandrasouly/strongreject/3432b2d696b428f242bd507df96d80f686571d5e/strongreject_dataset/strongreject_dataset.csv",
            command="mkdir -p ~/.cache/inspect_evals/strong_reject && \\\n"
                   "curl -L -o ~/.cache/inspect_evals/strong_reject/strongreject_dataset.csv \\\n"
                   "  'https://raw.githubusercontent.com/alexandrasouly/strongreject/3432b2d696b428f242bd507df96d80f686571d5e/strongreject_dataset/strongreject_dataset.csv'",
            description="如果自动下载超时，请手动执行上述命令下载数据集",
        ),
    ),

    # Docker 依赖 - cyse2_vulnerability_exploit
    BenchmarkRequirement(
        benchmark="cyberseceval_2",
        tasks=["cyse2_vulnerability_exploit"],
        dependency=DependencyType.DOCKER,
        description="cyse2_vulnerability_exploit 需要 Docker 运行漏洞测试沙箱",
        action=ActionItem(
            title="启动 Docker 服务",
            command="sudo systemctl start docker",
            description="确保 Docker 服务运行中。首次使用可能需要:\n"
                       "  1. 安装 Docker: https://docs.docker.com/get-docker/\n"
                       "  2. 将用户加入 docker 组: sudo usermod -aG docker $USER\n"
                       "  3. 重新登录使组权限生效",
        ),
    ),

    # Docker 依赖 - cve_bench
    BenchmarkRequirement(
        benchmark="cve_bench",
        tasks=["cve_bench"],
        dependency=DependencyType.DOCKER,
        description="cve_bench 需要 Docker 运行 CVE 漏洞环境 (支持全部 40 个 CVE)",
        action=ActionItem(
            title="启动 Docker 服务",
            command="sudo systemctl start docker",
            description="cve_bench 支持 Docker 或 Kubernetes:\n"
                       "  - Docker: 支持全部 40 个 CVE (推荐)\n"
                       "  - K8s: 仅支持 27 个 CVE\n"
                       "需要 Python >= 3.12",
        ),
        alternatives=[DependencyType.K8S],
    ),

    # PrivacyLens 数据集依赖
    BenchmarkRequirement(
        benchmark="privacylens",
        tasks=["privacylens_probing", "privacylens_action"],
        dependency=DependencyType.DATASET_DOWNLOAD,
        description="privacylens 需要下载 PrivacyLens 数据集",
        action=ActionItem(
            title="下载 PrivacyLens 数据集",
            url="https://github.com/SALT-NLP/PrivacyLens",
            command="mkdir -p benchmarks/local/pending_bench/privacylens/src/privacylens/data && "
                   "curl -sL -o benchmarks/local/pending_bench/privacylens/src/privacylens/data/main_data.json "
                   "https://raw.githubusercontent.com/SALT-NLP/PrivacyLens/main/data/main_data.json",
            description="或设置环境变量: export PRIVACYLENS_DATA_PATH=/path/to/main_data.json",
        ),
    ),

    # PrivacyLens Judge Model 依赖
    BenchmarkRequirement(
        benchmark="privacylens",
        tasks=["privacylens_action"],
        dependency=DependencyType.JUDGE_MODEL,
        description="privacylens_action 需要 Judge Model 评估泄漏和有效性",
        action=ActionItem(
            title="配置 Judge Model",
            description="通过 --judge-model 或 catalog.yaml 中的 judge_model 指定",
        ),
    ),

    # GAIA HuggingFace Gated Dataset
    BenchmarkRequirement(
        benchmark="gaia",
        tasks=["gaia", "gaia_level1", "gaia_level2", "gaia_level3"],
        dependency=DependencyType.HF_GATED,
        description="GAIA 使用受限数据集，需要 HuggingFace 认证和访问申请",
        action=ActionItem(
            title="申请 HuggingFace GAIA 数据集访问权限",
            url="https://huggingface.co/datasets/gaia-benchmark/GAIA",
            description="1. 访问上述链接，点击 'Access repository' 申请访问\n"
                       "2. 等待审批通过（通常即时批准）\n"
                       "3. 设置环境变量: export HF_TOKEN=<your_token>\n"
                       "   获取 Token: https://huggingface.co/settings/tokens",
        ),
    ),

    # BFCL v4 Web Search SerpAPI Key
    BenchmarkRequirement(
        benchmark="bfcl_v4",
        tasks=["bfcl_v4_web_search_base", "bfcl_v4_web_search_no_snippet"],
        dependency=DependencyType.SERPAPI_KEY,
        description="BFCL v4 web_search 任务需要 SerpAPI key 进行网络搜索",
        action=ActionItem(
            title="获取 SerpAPI Key",
            url="https://serpapi.com/",
            description="1. 访问 https://serpapi.com/ 注册账号\n"
                       "2. 获取 API Key (免费版每月 100 次搜索)\n"
                       "3. 设置环境变量: export SERPAPI_KEY=<your_key>\n"
                       "   或在 .env 文件中添加: SERPAPI_KEY=<your_key>",
        ),
    ),

    # GAIA Docker 依赖
    BenchmarkRequirement(
        benchmark="gaia",
        tasks=["gaia", "gaia_level1", "gaia_level2", "gaia_level3"],
        dependency=DependencyType.DOCKER,
        description="GAIA 需要 Docker 运行工具沙箱 (bash, python, web_browser)",
        action=ActionItem(
            title="启动 Docker 服务",
            command="sudo systemctl start docker",
            description="确保 Docker 服务运行中。GAIA 使用 Docker 沙箱执行工具调用。\n"
                       "首次使用可能需要:\n"
                       "  1. 安装 Docker: https://docs.docker.com/get-docker/\n"
                       "  2. 将用户加入 docker 组: sudo usermod -aG docker $USER\n"
                       "  3. 重新登录使组权限生效",
        ),
    ),

    # Docker 依赖 - agent_bench
    BenchmarkRequirement(
        benchmark="agent_bench",
        tasks=["agent_bench_os"],
        dependency=DependencyType.DOCKER,
        description="agent_bench 需要 Docker 运行 OS 交互沙箱 (bash, python 工具)",
        action=ActionItem(
            title="启动 Docker 服务",
            command="sudo systemctl start docker",
            description="确保 Docker 服务运行中。agent_bench 使用 Docker 沙箱执行 OS 操作任务。\n"
                       "首次使用可能需要:\n"
                       "  1. 安装 Docker: https://docs.docker.com/get-docker/\n"
                       "  2. 将用户加入 docker 组: sudo usermod -aG docker $USER\n"
                       "  3. 重新登录使组权限生效\n"
                       "\n"
                       "网络问题解决:\n"
                       "  如果 Docker 构建时网络超时，可设置 DEBIAN_MIRROR 环境变量:\n"
                       "  - 清华镜像: export DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian (默认)\n"
                       "  - 中科大镜像: export DEBIAN_MIRROR=https://mirrors.ustc.edu.cn/debian\n"
                       "  - 阿里镜像: export DEBIAN_MIRROR=https://mirrors.aliyun.com/debian\n"
                       "  - 官方源: export DEBIAN_MIRROR=http://deb.debian.org/debian\n"
                       "\n"
                       "  如果网络完全不可用，可跳过 FIPS apt 升级:\n"
                       "  - 跳过升级: export SKIP_FIPS_APT_UPGRADE=true\n"
                       "  (仅在非 FIPS 模式环境下使用)\n"
                       "\n"
                       "VPN 网络访问:\n"
                       "  如果宿主机有 VPN 且容器需要访问，可设置 AGENT_BENCH_NETWORK_MODE:\n"
                       "  - export AGENT_BENCH_NETWORK_MODE=host\n"
                       "\n"
                       "  警告: network_mode=host 有安全隐患:\n"
                       "  - 容器直接访问宿主机网络接口\n"
                       "  - 容器与宿主机端口可能冲突\n"
                       "  - 绕过 Docker 网络隔离\n"
                       "  - 仅在可信环境中使用\n"
                       "\n"
                       "  可选值: none (默认, 最大安全隔离), host (宿主机网络, 支持 VPN)",
        ),
    ),

    # Kubernetes 依赖 - agentdojo
    BenchmarkRequirement(
        benchmark="agentdojo",
        tasks=["agentdojo"],
        dependency=DependencyType.K8S,
        description="agentdojo 需要 Kubernetes 集群运行代理安全测试环境",
        action=ActionItem(
            title="启动/连接 Kubernetes 集群",
            command="kubectl cluster-info",
            description="确保 Kubernetes 集群可访问。agentdojo 使用 K8s 沙箱执行代理安全测试。\n"
                       "首次使用可能需要:\n"
                       "  1. 安装 kubectl: https://kubernetes.io/docs/tasks/tools/\n"
                       "  2. 配置 kubeconfig 文件 (~/.kube/config)\n"
                       "  3. 验证连接: kubectl cluster-info",
        ),
    ),

    # ST-WebAgentBench 依赖 - Web应用部署
    BenchmarkRequirement(
        benchmark="st_webagentbench",
        tasks=[
            "st_webagentbench", "st_webagentbench_gitlab", "st_webagentbench_suitecrm",
            "st_webagentbench_easy", "st_webagentbench_medium", "st_webagentbench_hard",
        ],
        dependency=DependencyType.DATASET_DOWNLOAD,
        description="ST-WebAgentBench 需要部署 Web 应用 (GitLab, SuiteCRM, ShoppingAdmin) 并安装 BrowserGym",
        action=ActionItem(
            title="部署 ST-WebAgentBench 环境依赖",
            url="https://github.com/segev-shlomov/ST-WebAgentBench",
            command="# 1. SuiteCRM (Docker)\n"
                   "cd /mnt/data1/workspace/djs/eval-poc-with-salt/ST-WebAgentBench/suitecrm_setup\n"
                   "docker compose up -d\n"
                   "cd init-db && docker exec -i suitecrm_setup-mariadb-1 mysql -u bn_suitecrm -pbitnami123 < demo_data.sql\n"
                   "\n"
                   "# 2. BrowserGym 和 Playwright\n"
                   "pip install browsergym==0.7.0 playwright==1.52.0\n"
                   "playwright install chromium\n"
                   "\n"
                   "# 3. ST-WebAgentBench 包\n"
                   "cd /mnt/data1/workspace/djs/eval-poc-with-salt/ST-WebAgentBench/browsergym/stwebagentbench\n"
                   "pip install -e ./\n"
                   "\n"
                   "# 4. 验证服务\n"
                   "curl http://localhost:8080  # SuiteCRM\n"
                   "curl http://localhost:8081  # GitLab (需要 WebArena)\n"
                   "curl http://localhost:8083  # ShoppingAdmin (需要 WebArena)",
            description="ST-WebAgentBench 环境依赖:\n"
                       "\n"
                       "**Web 应用 (必须运行)**\n"
                       "- SuiteCRM: http://localhost:8080 (Docker, 详见 suitecrm_setup/README.md)\n"
                       "- GitLab: http://localhost:8081 (需要 WebArena AWS AMI 或本地部署)\n"
                       "- ShoppingAdmin: http://localhost:8083 (需要 WebArena AWS AMI 或本地部署)\n"
                       "\n"
                       "**Python 包**\n"
                       "- browsergym==0.7.0\n"
                       "- playwright==1.52.0\n"
                       "- stwebagentbench (本地安装)\n"
                       "\n"
                       "**注意**: 与其他 agent benchmark 不同，ST-WebAgentBench 不使用本地 Docker Compose sandbox。\n"
                       "BrowserGym 通过 Playwright 管理浏览器环境，并连接到外部运行的 Web 应用。\n"
                       "\n"
                       "WebArena 设置: https://github.com/web-arena-x/webarena/tree/main/environment_docker",
        ),
    ),

    # Toolathlon 依赖 - Public Service 模式 (推荐) 或本地部署
    BenchmarkRequirement(
        benchmark="toolathlon",
        tasks=[
            "toolathlon", "toolathlon_sample",
            "toolathlon_canvas", "toolathlon_notion", "toolathlon_k8s",
            "toolathlon_git", "toolathlon_woocommerce",
            "toolathlon_research", "toolathlon_financial",
        ],
        dependency=DependencyType.DATASET_DOWNLOAD,
        description="Toolathlon 评测 - 推荐使用 Public Service 模式 (无需本地 Docker/MCP 配置)",
        action=ActionItem(
            title="设置 Toolathlon 评测环境",
            url="https://github.com/hkust-nlp/Toolathlon",
            command="# 1. 确保 Toolathlon 仓库存在 (已有)\n"
                   "# 应该位于: /mnt/data1/workspace/djs/eval-poc-with-salt/Toolathlon\n\n"
                   "# 2. 设置模型 API 凭证 (必需)\n"
                   "export TOOLATHLON_OPENAI_BASE_URL='https://your-api-endpoint.com/v1'\n"
                   "export TOOLATHLON_OPENAI_API_KEY='sk-your-api-key'\n"
                   "export TOOLATHLON_MODEL_NAME='gpt-4o'  # 或其他模型\n\n"
                   "# 3. 运行评测 (Public Service 模式，无需本地 Docker!)\n"
                   "./run-eval.py toolathlon --model <model> --limit 5\n\n"
                   "# === Public Service 模式 (推荐) ===\n"
                   "# 优点:\n"
                   "#   - 无需本地 Docker/Podman\n"
                   "#   - 无需配置 MCP 服务器\n"
                   "#   - 无需部署外部服务 (Canvas, Notion 等)\n"
                   "#   - 公共服务处理所有基础设施\n"
                   "#\n"
                   "# 限制:\n"
                   "#   - 每个 IP 每 24 小时最多 180 分钟累计执行时间\n"
                   "#   - 每个 IP 每 24 小时最多 3 个评测请求\n"
                   "#   - 如需大量使用，请联系作者: jlini@cse.ust.hk\n\n"
                   "# === 本地部署模式 (高级用户) ===\n"
                   "# 如需本地运行 (绕过公共服务限制):\n"
                   "cd /mnt/data1/workspace/djs/eval-poc-with-salt/Toolathlon\n\n"
                   "# 安装 Docker/Podman\n"
                   "sudo apt-get install docker.io  # 或 podman\n\n"
                   "# 拉取 Toolathlon 镜像\n"
                   "bash global_preparation/pull_toolathlon_image.sh\n\n"
                   "# (可选) 部署外部服务以获得完整功能\n"
                   "bash global_preparation/deploy_containers.sh",
            description="Toolathlon 支持两种评测模式:\n"
                       "\n"
                       "**Public Service 模式 (推荐)**\n"
                       "- 使用作者提供的公共评测服务\n"
                       "- 无需本地 Docker、MCP 服务器、外部服务\n"
                       "- 只需设置 API 凭证即可运行\n"
                       "- 适合快速测试和轻度使用\n"
                       "\n"
                       "**本地部署模式 (高级)**\n"
                       "- 本地运行完整 Toolathlon 环境\n"
                       "- 需要 Docker/Podman、MCP 服务器、外部服务\n"
                       "- 适合大量评测或自定义配置\n"
                       "\n"
                       "**环境变量 (Public Service 模式)**:\n"
                       "  - TOOLATHLON_OPENAI_API_KEY: 模型 API 密钥\n"
                       "  - TOOLATHLON_OPENAI_BASE_URL: 模型 API 端点\n"
                       "  - TOOLATHLON_MODEL_NAME: 模型名称\n"
                       "\n"
                       "**任务分类** (109 个任务):\n"
                       "  - Canvas: 8 个任务\n"
                       "  - Notion: 4 个任务\n"
                       "  - Kubernetes: 5 个任务\n"
                       "  - Git: 3 个任务\n"
                       "  - WooCommerce: 6 个任务\n"
                       "  - 其他: 83 个任务\n"
                       "\n"
                       "**论文**: https://arxiv.org/abs/2510.25726\n"
                       "**GitHub**: https://github.com/hkust-nlp/Toolathlon\n"
                       "**公共服务**: 47.253.6.47:8080 (由作者提供)\n"
                       "\n"
                       "**快速测试命令**:\n"
                       "./run-eval.py toolathlon --model gpt-4o --limit 3\n"
                       "./run-eval.py toolathlon:toolathlon_sample --model gpt-4o\n"
                       "./run-eval.py toolathlon --model gpt-4o -T service_filter=canvas",
        ),
    ),
]


@dataclass
class PreflightResult:
    """预检查结果"""
    passed: bool
    dependency: DependencyType
    benchmark: str
    message: str
    action: Optional[ActionItem] = None


@dataclass
class JudgeModelConfig:
    """Judge Model 配置"""
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    PRESETS = {
        "economy": ("gpt-4o-mini", "成本低，适合大规模测试"),
        "balanced": ("zai-glm-4.7", "性价比高，推荐"),
        "premium": ("gpt-4o", "精度最高，成本较高"),
    }


def check_docker() -> tuple[bool, str]:
    """检查 Docker 是否可用"""
    docker_path = shutil.which("docker")
    if not docker_path:
        return False, "Docker 未安装"

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, "Docker 服务未运行"
    except subprocess.TimeoutExpired:
        return False, "Docker 响应超时"
    except Exception as e:
        return False, f"Docker 检查失败: {e}"

    return True, "Docker 已就绪"


def check_k8s() -> tuple[bool, str]:
    """检查 Kubernetes 是否可用"""
    kubectl_path = shutil.which("kubectl")
    if not kubectl_path:
        return False, "kubectl 未安装"

    try:
        result = subprocess.run(
            ["kubectl", "cluster-info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, "无法连接 Kubernetes 集群"
    except subprocess.TimeoutExpired:
        return False, "Kubernetes 集群响应超时"
    except Exception as e:
        return False, f"K8s 检查失败: {e}"

    return True, "Kubernetes 集群已就绪"


def check_hf_token() -> tuple[bool, str]:
    """检查 HuggingFace Token"""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        return False, "未设置 HF_TOKEN 环境变量"
    return True, "HuggingFace Token 已配置"


def check_hf_gated_access(dataset_id: str) -> tuple[bool, str]:
    """检查 HuggingFace gated dataset 访问权限"""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        return False, f"需要 HF_TOKEN 并申请 {dataset_id} 访问权限"

    # 尝试访问数据集 API
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/datasets/{dataset_id}",
            headers={
                "Authorization": f"Bearer {hf_token}",
                "User-Agent": "preflight-check",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                return True, f"已有 {dataset_id} 访问权限"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, f"HF_TOKEN 无效"
        elif e.code == 403:
            return False, f"需要申请 {dataset_id} 访问权限"
    except Exception:
        pass

    return False, f"无法验证 {dataset_id} 访问权限"


def check_dataset_download(url: str, cache_path: Path) -> tuple[bool, str]:
    """检查数据集是否已下载或可下载"""
    # 检查缓存
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return True, f"数据集已缓存: {cache_path}"

    # 尝试访问 URL
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "preflight-check"})
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                return True, "数据集 URL 可访问，将自动下载"
    except Exception:
        pass

    return False, "数据集 URL 无法访问，需要手动下载"


def check_hf_network() -> tuple[bool, str]:
    """检查 HuggingFace 网络访问"""
    try:
        req = urllib.request.Request(
            "https://huggingface.co/api/health",
            headers={"User-Agent": "preflight-check"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                return True, "HuggingFace 网络访问正常"
    except Exception:
        pass
    return False, "无法访问 HuggingFace"


def check_judge_model(config: Optional[JudgeModelConfig] = None) -> tuple[bool, str]:
    """检查 Judge Model 配置"""
    if config and config.model_name:
        return True, f"Judge Model: {config.model_name}"
    return False, "未配置 Judge Model"


def check_privacylens_data() -> tuple[bool, str]:
    """检查 PrivacyLens 数据集"""
    data_path = os.environ.get("PRIVACYLENS_DATA_PATH")
    if data_path and Path(data_path).exists():
        return True, f"数据集已配置: {data_path}"

    default_path = PROJECT_ROOT / "benchmarks" / "local" / "pending_bench" / "privacylens" / "src" / "privacylens" / "data" / "main_data.json"
    if default_path.exists():
        return True, f"数据集已存在: {default_path}"

    return False, "PrivacyLens 数据集未找到"


def check_serpapi_key() -> tuple[bool, str]:
    """检查 SerpAPI Key"""
    serpapi_key = os.environ.get("SERPAPI_KEY")
    if not serpapi_key:
        return False, "未设置 SERPAPI_KEY 环境变量"
    # Basic validation - SerpAPI keys are typically 32 characters alphanumeric
    if len(serpapi_key) < 20:
        return False, "SERPAPI_KEY 格式不正确（太短）"
    return True, "SerpAPI Key 已配置"


def check_st_webagentbench() -> tuple[bool, str]:
    """检查 ST-WebAgentBench 环境依赖"""
    # 1. Check data file exists
    data_file = Path("/mnt/data1/workspace/djs/eval-poc-with-salt/ST-WebAgentBench/stwebagentbench/test.raw.json")
    if not data_file.exists():
        return False, "ST-WebAgentBench 数据文件不存在，请克隆仓库: https://github.com/segev-shlomov/ST-WebAgentBench"

    # 2. Check BrowserGym is installed
    try:
        import browsergym
        import browsergym.stwebagentbench
    except ImportError:
        return False, "BrowserGym 未安装，运行: pip install browsergym==0.7.0"

    # 3. Check Playwright browsers are installed
    try:
        from playwright.sync_api import sync_playwright
        # Try to check if chromium is installed
        import os
        from playwright._impl._driver import get_driver_env, get_executable_path
        try:
            chromium_path = get_executable_path()["chromium"]
            if not os.path.exists(chromium_path):
                return False, "Playwright Chromium 未安装，运行: playwright install chromium"
        except Exception:
            # Fallback: try importing browsergym which handles this check
            pass
    except ImportError:
        return False, "Playwright 未安装，运行: pip install playwright==1.52.0"

    # 4. Check SuiteCRM is running (optional but helpful)
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8080", method="HEAD", timeout=2)
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                suitecrm_running = True
    except Exception:
        suitecrm_running = False

    if not suitecrm_running:
        return False, "SuiteCRM 未运行，启动: cd /mnt/data1/workspace/djs/eval-poc-with-salt/ST-WebAgentBench/suitecrm_setup && docker compose up -d"

    # 5. Check for web services (GitLab/ShoppingAdmin are optional - use WebArena)
    gitlab_running = False
    shopping_running = False
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8081", method="HEAD", timeout=2)
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                gitlab_running = True
    except Exception:
        pass

    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8083", method="HEAD", timeout=2)
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                shopping_running = True
    except Exception:
        pass

    # Build status message
    status_parts = ["数据文件已就绪", "BrowserGym 已安装", "Playwright 已就绪", "SuiteCRM 已运行"]
    if not gitlab_running:
        status_parts.append("GitLab 未运行 (需要 WebArena)")
    if not shopping_running:
        status_parts.append("ShoppingAdmin 未运行 (需要 WebArena)")

    return True, ", ".join(status_parts)


def run_preflight_checks(
    benchmarks: list[str],
    judge_config: Optional[JudgeModelConfig] = None,
) -> list[PreflightResult]:
    """
    运行预检查，返回详细结果列表
    """
    results: list[PreflightResult] = []
    checked: set[tuple[str, DependencyType]] = set()

    for benchmark in benchmarks:
        for req in BENCHMARK_REQUIREMENTS:
            if req.benchmark != benchmark:
                continue

            key = (req.benchmark, req.dependency)
            if key in checked:
                continue
            checked.add(key)

            # 执行对应检查
            passed, message = False, ""

            if req.dependency == DependencyType.DOCKER:
                passed, message = check_docker()
            elif req.dependency == DependencyType.K8S:
                passed, message = check_k8s()
            elif req.dependency == DependencyType.HF_AUTH:
                passed, message = check_hf_token()
            elif req.dependency == DependencyType.HF_GATED:
                if benchmark == "xstest":
                    passed, message = check_hf_gated_access("walledai/XSTest")
                elif benchmark == "gaia":
                    passed, message = check_hf_gated_access("gaia-benchmark/GAIA")
            elif req.dependency == DependencyType.HF_NETWORK:
                passed, message = check_hf_network()
            elif req.dependency == DependencyType.JUDGE_MODEL:
                passed, message = check_judge_model(judge_config)
            elif req.dependency == DependencyType.DATASET_DOWNLOAD:
                if benchmark == "strong_reject":
                    cache_dir = Path.home() / ".cache" / "inspect_evals" / "strong_reject"
                    cache_file = cache_dir / "strongreject_dataset.csv"
                    # 也检查可能的其他缓存位置
                    for pattern in cache_dir.glob("strong_reject_*.csv"):
                        if pattern.stat().st_size > 0:
                            cache_file = pattern
                            break
                    passed, message = check_dataset_download(
                        "https://raw.githubusercontent.com/alexandrasouly/strongreject/3432b2d696b428f242bd507df96d80f686571d5e/strongreject_dataset/strongreject_dataset.csv",
                        cache_file,
                    )
                elif benchmark == "privacylens":
                    passed, message = check_privacylens_data()
                elif benchmark == "st_webagentbench":
                    passed, message = check_st_webagentbench()
            elif req.dependency == DependencyType.SERPAPI_KEY:
                passed, message = check_serpapi_key()

            results.append(PreflightResult(
                passed=passed,
                dependency=req.dependency,
                benchmark=benchmark,
                message=message,
                action=req.action if not passed else None,
            ))

    return results


def get_required_permissions(benchmarks: list[str]) -> list[str]:
    """获取需要用户确认的权限列表"""
    permissions = []
    if "cve_bench" in benchmarks:
        permissions.append(
            "cve_bench: 将在 Docker 容器中运行包含真实 CVE 漏洞的 Web 应用 (隔离环境)"
        )
    if "cyberseceval_2" in benchmarks:
        permissions.append(
            "cyse2_vulnerability_exploit: 将在 Docker 容器中编译和执行测试代码 (隔离环境)"
        )
    if "gaia" in benchmarks:
        permissions.append(
            "gaia: 将在 Docker 容器中执行 bash、python 和 web 浏览工具 (隔离环境)"
        )
    if "agent_bench" in benchmarks:
        permissions.append(
            "agent_bench: 将在 Docker 容器中执行 bash 和 python 工具进行 OS 操作 (隔离环境)"
        )
    if "agentdojo" in benchmarks:
        permissions.append(
            "agentdojo: 将在 Kubernetes Pod 中执行代理工具操作 (隔离环境)"
        )
    return permissions


def print_preflight_report(results: list[PreflightResult]) -> bool:
    """打印预检查报告，返回是否全部通过"""
    all_passed = True
    failed_actions: list[ActionItem] = []

    print("\n" + "=" * 70)
    print("预检查报告")
    print("=" * 70)

    # 按 benchmark 分组
    by_benchmark: dict[str, list[PreflightResult]] = {}
    for r in results:
        by_benchmark.setdefault(r.benchmark, []).append(r)

    for benchmark, checks in by_benchmark.items():
        print(f"\n[{benchmark}]")
        for result in checks:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.message}")
            if not result.passed:
                all_passed = False
                if result.action:
                    failed_actions.append(result.action)

    # 打印失败项的详细操作指引
    if failed_actions:
        print("\n" + "=" * 70)
        print("需要执行的操作")
        print("=" * 70)

        for i, action in enumerate(failed_actions, 1):
            print(f"\n{i}. {action.title}")
            if action.description:
                for line in action.description.split("\n"):
                    print(f"   {line}")
            if action.url:
                print(f"\n   链接: {action.url}")
            if action.command:
                print(f"\n   命令:")
                for line in action.command.split("\n"):
                    print(f"   $ {line.strip()}")

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 所有检查通过，可以开始评测")
    else:
        print("❌ 部分检查未通过，请按上述指引操作后重试")
    print("=" * 70 + "\n")

    return all_passed


def generate_setup_script(results: list[PreflightResult], output_path: Path) -> None:
    """生成设置脚本，包含所有需要的操作"""
    failed = [r for r in results if not r.passed and r.action]

    lines = [
        "#!/bin/bash",
        "# 一键测评环境设置脚本",
        "# 自动生成，请根据实际情况修改",
        "",
        "set -e",
        "",
    ]

    for r in failed:
        if not r.action:
            continue

        lines.append(f"# === {r.benchmark}: {r.action.title} ===")
        if r.action.description:
            for desc_line in r.action.description.split("\n"):
                lines.append(f"# {desc_line}")
        if r.action.command:
            lines.append(r.action.command)
        lines.append("")

    lines.append('echo "设置完成！"')

    output_path.write_text("\n".join(lines))
    output_path.chmod(0o755)


if __name__ == "__main__":
    # 测试所有检查
    all_benchmarks = [
        "strong_reject", "xstest", "cyberseceval_2", "cve_bench",
        "bbq", "truthfulqa", "agentharm", "agentdojo", "gaia",
    ]

    results = run_preflight_checks(
        all_benchmarks,
        judge_config=JudgeModelConfig(model_name="zai-glm-4.7"),
    )

    passed = print_preflight_report(results)

    # 生成设置脚本
    if not passed:
        script_path = Path("setup_env.sh")
        generate_setup_script(results, script_path)
        print(f"已生成设置脚本: {script_path}")

    # 打印权限确认
    permissions = get_required_permissions(all_benchmarks)
    if permissions:
        print("\n需要用户确认的权限:")
        for perm in permissions:
            print(f"  • {perm}")
