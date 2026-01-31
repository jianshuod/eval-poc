# 本地 Benchmark 集成指南

本目录用于存放待审核的本地 benchmark 实现。本指南说明如何将新的 benchmark 集成到测评套装中。

## 目录结构

```
pending_bench/
└── <benchmark_name>/
    ├── pyproject.toml              # 包配置和依赖
    └── src/
        └── <benchmark_name>/
            ├── __init__.py         # 导出任务函数
            ├── task.py             # 任务定义 (@task 装饰器)
            ├── dataset.py          # 数据集加载
            ├── scorers.py          # 评分器
            ├── prompts.py          # 提示模板
            └── data/
                └── *.json          # 数据文件
```

## 集成步骤

### 1. 创建目录结构

```bash
mkdir -p benchmarks/local/pending_bench/<benchmark_name>/src/<benchmark_name>/data
```

### 2. 创建 pyproject.toml

```toml
[project]
name = "<benchmark_name>"
version = "1.0.0"
description = "描述"
requires-python = ">=3.10"
dependencies = [
    "inspect_ai>=0.3.0",
    # 其他依赖...
]

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

### 3. 实现核心文件

#### `__init__.py`
```python
from .task import my_task_1, my_task_2

__all__ = ["my_task_1", "my_task_2"]
```

#### `task.py`
```python
from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

# 使用绝对导入（不要用相对导入）
from <benchmark_name>.dataset import load_dataset
from <benchmark_name>.scorers import my_scorer

@task
def my_task_1() -> Task:
    return Task(
        dataset=load_dataset(),
        solver=[generate()],
        scorer=my_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=256),
        version="1.0.0",
    )
```

**重要**：必须使用绝对导入（`from <benchmark_name>.xxx`），不要使用相对导入（`from .xxx`），因为 inspect_ai 直接加载文件时相对导入会失败。

#### `dataset.py`
```python
import json
from pathlib import Path
from inspect_ai.dataset import Dataset, Sample, MemoryDataset

def get_data_path() -> Path:
    return Path(__file__).parent / "data" / "data.json"

def load_dataset() -> Dataset:
    with open(get_data_path()) as f:
        data = json.load(f)
    samples = [Sample(input=item["input"], target=item["target"]) for item in data]
    return MemoryDataset(samples=samples, name="my_dataset")
```

#### `scorers.py`
```python
from inspect_ai.scorer import Score, Scorer, Target, metric, scorer
from inspect_ai.solver import TaskState

@scorer(metrics=[my_metric()])
def my_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # 评分逻辑
        return Score(value=1.0, answer="...", explanation="...")
    return score

@metric
def my_metric():
    def compute(scores):
        return sum(s.score.as_float() for s in scores) / len(scores)
    return compute
```

### 4. 配置 catalog.yaml

在 `benchmarks/catalog.yaml` 添加：

```yaml
  <benchmark_name>:
    source: local
    module: benchmarks/local/pending_bench/<benchmark_name>
    python: "3.10"
    judge_model: zai-glm-4.7  # 如果需要 judge model
    tasks:
      - name: my_task_1
        path: benchmarks/local/pending_bench/<benchmark_name>/src/<benchmark_name>/task.py@my_task_1
        task_args:
          param1: value1
      - name: my_task_2
        path: benchmarks/local/pending_bench/<benchmark_name>/src/<benchmark_name>/task.py@my_task_2
```

### 5. 添加预检查（可选）

如果 benchmark 有特殊依赖，在 `benchmarks/preflight.py` 添加：

```python
# 在 BENCHMARK_REQUIREMENTS 列表中添加
BenchmarkRequirement(
    benchmark="<benchmark_name>",
    tasks=["my_task_1", "my_task_2"],
    dependency=DependencyType.DATASET_DOWNLOAD,
    description="需要下载数据集",
    action=ActionItem(
        title="下载数据集",
        command="curl -o ... <url>",
    ),
),
```

### 6. 添加评分映射（可选）

如果需要统一评分，在 `score_mapper.py` 添加：

```python
@register_mapper
class MyBenchmarkMapper(ScoreMapper):
    @property
    def benchmark_name(self) -> str:
        return "my_task_1"

    @property
    def description(self) -> str:
        return "描述"

    @property
    def scoring_rationale(self) -> str:
        return """原始指标、语义、转换方向说明"""

    @property
    def score_type(self) -> ScoreType:
        return ScoreType.NUMERIC

    @property
    def score_direction(self) -> ScoreDirection:
        return ScoreDirection.HIGHER_IS_SAFER  # 或 LOWER_IS_SAFER

    @property
    def score_range(self) -> tuple:
        return (0.0, 1.0)
```

## 测试验证

```bash
# 1. 设置环境
./run-eval.py --setup <benchmark_name>

# 2. 验证模块导入
.venvs/<benchmark_name>/bin/python -c "from <benchmark_name> import my_task_1; print('OK')"

# 3. 运行测试（限制样本数）
./run-eval.py <benchmark_name>:my_task_1 --model <model> --limit 5

# 4. 检查日志
ls results/<model>/<benchmark_name>/logs/
```

## 常见问题

### ModuleNotFoundError
- 确保使用绝对导入，不要用相对导入
- 确保 `pyproject.toml` 中 `[tool.setuptools.packages.find]` 配置正确

### 数据文件找不到
- 数据文件应放在 `src/<benchmark_name>/data/` 目录
- 使用 `Path(__file__).parent / "data"` 定位

### Judge Model 问题
- 在 `catalog.yaml` 中配置 `judge_model`
- 或通过 `--judge-model` 命令行参数指定

## 参考实现

- `privacylens/` - 完整的本地 benchmark 示例
