# Tau2-CL 持续学习工作流程

本文档提供了从数据生成、训练到测试的完整工作流程。

## 目录

1. [数据准备](#1-数据准备)
2. [训练CL模型](#2-训练cl模型)
3. [评估与测试](#3-评估与测试)
4. [完整示例](#4-完整示例)

---

## 1. 数据准备

### 1.1 检查现有数据

```bash
# 查看当前数据量
tau2 cl-data-requirements --domains airline retail telecom
```

### 1.2 验证数据有效性

```bash
# 验证单个domain的数据
tau2 cl-validate-data data/tau2/domains/airline/tasks.json

# 验证所有domain
tau2 cl-validate-data data/tau2/domains/
```

### 1.3 生成数据（如需要）

#### 方式一：手动创建任务数据

创建新的任务JSON文件，遵循以下格式：

```json
{
  "id": "new_task_001",
  "description": {
    "purpose": "测试agent处理xxx的能力",
    "relevant_policies": null,
    "notes": null
  },
  "user_scenario": {
    "persona": null,
    "instructions": {
      "domain": "airline",
      "reason_for_call": "用户想要查询航班信息",
      "known_info": "You are John Doe.\nYour user id is john_doe_1234.",
      "unknown_info": null,
      "task_instructions": "询问从纽约到洛杉矶的航班。如果agent询问日期，告诉他是下周一。"
    }
  },
  "initial_state": null,
  "evaluation_criteria": {
    "actions": [
      {
        "action_id": "new_task_001_0",
        "requestor": "assistant",
        "name": "search_flights",
        "arguments": {
          "origin": "JFK",
          "destination": "LAX"
        },
        "info": null
      }
    ],
    "communicate_info": [],
    "nl_assertions": [
      "Agent should successfully search for flights",
      "Agent should communicate flight options to user"
    ],
    "reward_basis": ["DB", "COMMUNICATE"]
  }
}
```

将新任务添加到对应domain的 `tasks.json` 文件中。

#### 方式二：使用LLM辅助生成（推荐）

创建一个生成脚本 `scripts/generate_tasks.py`:

```python
import json
from pathlib import Path
from litellm import completion

def generate_task_with_llm(domain: str, base_tasks: list, num_new_tasks: int = 10):
    """使用LLM生成新任务"""

    # 加载domain策略
    policy_path = Path(f"data/tau2/domains/{domain}/policy.md")
    with open(policy_path, 'r', encoding='utf-8') as f:
        policy = f.read()

    # 选择几个示例任务
    examples = base_tasks[:3]

    prompt = f"""你是一个任务生成器，需要为{domain}领域生成新的客服对话任务。

策略文档：
{policy[:2000]}

现有任务示例：
{json.dumps(examples, indent=2, ensure_ascii=False)[:3000]}

请生成{num_new_tasks}个新的、多样化的任务，确保：
1. 任务测试不同的策略规则
2. 包含不同难度级别
3. 覆盖不同的工具调用组合
4. 用户意图表达方式多样

以JSON数组格式返回，每个任务遵循上述示例的格式。
"""

    response = completion(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    new_tasks = json.loads(response.choices[0].message.content)
    return new_tasks

# 使用示例
domain = "airline"
with open(f"data/tau2/domains/{domain}/tasks.json", 'r', encoding='utf-8') as f:
    base_tasks = json.load(f)

new_tasks = generate_task_with_llm(domain, base_tasks, num_new_tasks=50)

# 保存
all_tasks = base_tasks + new_tasks
with open(f"data/tau2/domains/{domain}/tasks_expanded.json", 'w', encoding='utf-8') as f:
    json.dump(all_tasks, f, indent=2, ensure_ascii=False)
```

运行：
```bash
python scripts/generate_tasks.py
```

### 1.4 生成CL数据划分

```bash
# 生成sequential策略的数据划分
tau2 cl-generate-splits \
    --domains airline retail telecom \
    --strategy sequential \
    --train-ratio 0.6 \
    --num-phases 3 \
    --output-dir data/tau2/cl_splits

# 生成difficulty-based策略的数据划分
tau2 cl-generate-splits \
    --domains airline retail telecom \
    --strategy difficulty \
    --train-ratio 0.7 \
    --num-phases 5 \
    --output-dir data/tau2/cl_splits
```

生成的文件结构：
```
data/tau2/cl_splits/
├── airline_cl_split.json
├── retail_cl_split.json
├── telecom_cl_split.json
└── multi_domain_cl_split.json
```

---

## 2. 训练CL模型

### 2.1 准备配置文件

创建实验配置 `configs/cl_experiments/my_experiment.yaml`:

```yaml
# 实验配置
name: "airline_retail_icl_er"
seed: 42
output_dir: "./experiments/my_experiment"

# Curriculum设置
curriculum_strategy: "SEQUENTIAL"  # SEQUENTIAL, INTERLEAVED, DIFFICULTY_BASED
domains: ["airline", "retail"]
num_tasks_per_domain: 100  # null表示使用所有任务

# Agent配置
agent_type: "ICL_ER"  # ICL_ER, PROMPT_STRATEGY, BASELINE
agent_llm: "gpt-4o-mini"
max_examples_in_prompt: 5

# Memory配置
memory_buffer_size: 1000
sampling_strategy: "DIVERSITY"  # UNIFORM, REWARD_WEIGHTED, RECENCY_WEIGHTED, DIVERSITY, SIMILARITY

# User simulator配置
user_llm: "gpt-4o-mini"

# 评估配置
eval_frequency: 10  # 每10个任务评估一次
```

### 2.2 运行实验

#### 方式一：使用配置文件（推荐）

```bash
tau2 cl-run --config configs/cl_experiments/my_experiment.yaml
```

#### 方式二：使用命令行参数

```bash
tau2 cl-run \
    --name "quick_test" \
    --domains airline retail \
    --curriculum sequential \
    --agent-type icl_er \
    --agent-llm gpt-4o-mini \
    --user-llm gpt-4o-mini \
    --max-examples 5 \
    --buffer-size 1000 \
    --num-tasks 50 \
    --eval-frequency 10 \
    --output-dir ./experiments/quick_test \
    --seed 42
```

### 2.3 监控训练进度

实验运行时会输出：
```
[2025-12-03 10:30:00] Starting CL Experiment: airline_retail_icl_er
[2025-12-03 10:30:01] Phase 1/3: airline (50 tasks)
[2025-12-03 10:30:05] Task 1/50 completed. Reward: 0.85
[2025-12-03 10:30:10] Task 2/50 completed. Reward: 0.92
...
[2025-12-03 10:35:00] Evaluation checkpoint (10 tasks)
  - Current accuracy: 0.87
  - Memory buffer: 38 experiences
...
```

### 2.4 实验输出

实验完成后会生成以下文件：

```
experiments/my_experiment/
├── config.json              # 实验配置
├── results.json             # 完整结果
├── metrics/
│   ├── accuracy_curve.png   # 准确率曲线
│   ├── forgetting_matrix.png  # 遗忘矩阵
│   └── performance_matrix.png  # 性能矩阵
├── agent_state/
│   ├── final_state.json     # 最终agent状态
│   └── memory_buffer.json   # 最终memory buffer
└── logs/
    └── experiment.log       # 详细日志
```

---

## 3. 评估与测试

### 3.1 分析实验结果

```bash
# 查看实验结果摘要
tau2 cl-analyze experiments/my_experiment/results.json
```

输出示例：
```
============================================================
Continual Learning Experiment Analysis
============================================================
Experiment: airline_retail_icl_er
Domains: ['airline', 'retail']
Curriculum: SEQUENTIAL
Agent Type: ICL_ER

Metrics:
  Average Accuracy: 0.847
  Forgetting Rate: 0.123
  Forward Transfer: 0.056
  Backward Transfer: -0.089

Per-Domain Accuracy:
  airline: 0.820
  retail: 0.874

Per-Domain Forgetting:
  airline: 0.156
  retail: 0.090
============================================================
```

### 3.2 对比多个实验

创建对比脚本 `scripts/compare_experiments.py`:

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt

def compare_experiments(exp_paths: list):
    """对比多个实验结果"""

    results = []
    for path in exp_paths:
        with open(path / "results.json", 'r') as f:
            results.append(json.load(f))

    # 对比准确率
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 平均准确率
    names = [r['config']['name'] for r in results]
    accs = [r['metrics']['average_accuracy'] for r in results]
    axes[0].bar(names, accs)
    axes[0].set_title('Average Accuracy')
    axes[0].set_ylim([0, 1])

    # 遗忘率
    forgetting = [r['metrics']['forgetting_rate'] for r in results]
    axes[1].bar(names, forgetting)
    axes[1].set_title('Forgetting Rate')

    # Forward Transfer
    transfer = [r['metrics']['forward_transfer'] for r in results]
    axes[2].bar(names, transfer)
    axes[2].set_title('Forward Transfer')

    plt.tight_layout()
    plt.savefig('experiments/comparison.png')
    print("Comparison saved to experiments/comparison.png")

# 使用
experiments = [
    Path("experiments/icl_er_sequential"),
    Path("experiments/prompt_strategy_sequential"),
    Path("experiments/baseline"),
]
compare_experiments(experiments)
```

### 3.3 测试特定场景

创建测试脚本 `scripts/test_agent.py`:

```python
from tau2.continual_learning import CLOrchestrator
from tau2.continual_learning.agents.icl_experience_replay import ICLExperienceReplayAgent
from tau2.environment.loader import get_tools, load_domain_policy

# 加载训练好的agent
agent_path = "experiments/my_experiment/agent_state/final_state.json"
agent = ICLExperienceReplayAgent(
    tools=get_tools("airline"),
    domain_policy=load_domain_policy("airline"),
    llm="gpt-4o-mini",
)
agent.load_state(agent_path)

# 在测试任务上评估
from tau2.data_model.tasks import Task
import json

# 加载测试任务
with open("data/tau2/domains/airline/tasks.json", 'r') as f:
    all_tasks = json.load(f)

with open("data/tau2/cl_splits/airline_cl_split.json", 'r') as f:
    split = json.load(f)

test_task_ids = set(split['splits']['test'])
test_tasks = [Task(**t) for t in all_tasks if t['id'] in test_task_ids]

# 运行测试
from tau2.run import run_single_task

results = []
for task in test_tasks[:10]:  # 测试前10个
    result = run_single_task(
        task=task,
        agent=agent,
        domain="airline",
    )
    results.append(result)
    print(f"Task {task.id}: reward={result['reward']:.2f}")

# 计算平均性能
avg_reward = sum(r['reward'] for r in results) / len(results)
print(f"\nTest set average reward: {avg_reward:.3f}")
```

---

## 4. 完整示例

### 4.1 快速开始实验

```bash
# 1. 验证数据
tau2 cl-validate-data data/tau2/domains/

# 2. 生成数据划分
tau2 cl-generate-splits --strategy sequential --num-phases 3

# 3. 运行quick test
tau2 cl-run \
    --name quick_test \
    --domains airline retail \
    --curriculum sequential \
    --agent-type icl_er \
    --num-tasks 20 \
    --eval-frequency 5 \
    --output-dir ./experiments/quick_test

# 4. 查看结果
tau2 cl-analyze experiments/quick_test/results.json
```

### 4.2 完整实验流程

```bash
#!/bin/bash

# 1. 数据准备
echo "Step 1: Validating data..."
tau2 cl-validate-data data/tau2/domains/

echo "Step 2: Checking data requirements..."
tau2 cl-data-requirements --domains airline retail telecom

echo "Step 3: Generating CL splits..."
tau2 cl-generate-splits \
    --strategy sequential \
    --train-ratio 0.7 \
    --num-phases 3 \
    --output-dir data/tau2/cl_splits

# 2. 运行多个实验进行对比
echo "Step 4: Running experiments..."

# ICL-ER with sequential curriculum
tau2 cl-run \
    --config configs/cl_experiments/icl_er_sequential.yaml &

# Prompt Strategy with sequential curriculum
tau2 cl-run \
    --config configs/cl_experiments/prompt_strategy.yaml &

# Baseline (no continual learning)
tau2 cl-run \
    --config configs/cl_experiments/baseline.yaml &

wait

# 3. 分析结果
echo "Step 5: Analyzing results..."
tau2 cl-analyze experiments/icl_er_sequential/results.json
tau2 cl-analyze experiments/prompt_strategy_sequential/results.json
tau2 cl-analyze experiments/baseline/results.json

# 4. 生成对比报告
python scripts/compare_experiments.py

echo "Done! Check experiments/ for results."
```

保存为 `run_full_experiment.sh` 并运行：
```bash
bash run_full_experiment.sh
```

### 4.3 Python API使用

```python
from tau2.continual_learning import (
    CLOrchestrator,
    CLExperimentConfig,
    AgentType,
)
from tau2.continual_learning.curriculum import CurriculumStrategy

# 配置实验
config = CLExperimentConfig(
    name="my_experiment",
    seed=42,
    output_dir="./experiments/my_experiment",

    # Curriculum
    curriculum_strategy=CurriculumStrategy.SEQUENTIAL,
    domains=["airline", "retail"],
    num_tasks_per_domain=100,

    # Agent
    agent_type=AgentType.ICL_ER,
    agent_llm="gpt-4o-mini",
    max_examples_in_prompt=5,

    # Memory
    memory_buffer_size=1000,

    # Evaluation
    eval_frequency=10,

    # User
    user_llm="gpt-4o-mini",
)

# 运行实验
orchestrator = CLOrchestrator(config)
result = orchestrator.run()

# 查看结果
print(f"Average Accuracy: {result.metrics.average_accuracy:.3f}")
print(f"Forgetting Rate: {result.metrics.forgetting_rate:.3f}")
print(f"Forward Transfer: {result.metrics.forward_transfer:.3f}")

# 保存
result.save(config.output_dir)
```

---

## 5. 常见问题

### Q1: 实验运行很慢怎么办？

```bash
# 使用更快的模型
tau2 cl-run --agent-llm gpt-4o-mini --user-llm gpt-4o-mini

# 减少任务数量
tau2 cl-run --num-tasks 50

# 降低评估频率
tau2 cl-run --eval-frequency 20
```

### Q2: 如何恢复中断的实验？

实验结果会定期保存，可以通过加载agent状态继续：

```python
from tau2.continual_learning import CLOrchestrator

# 加载之前的配置和状态
orchestrator = CLOrchestrator.load("experiments/my_experiment")

# 继续运行
orchestrator.resume()
```

### Q3: 如何自定义评估指标？

编辑 `src/tau2/continual_learning/metrics/metrics.py` 添加新指标：

```python
def compute_my_custom_metric(self, performance_matrix):
    """自定义指标"""
    # 实现你的指标计算
    return metric_value
```

### Q4: 如何可视化训练过程？

```python
import matplotlib.pyplot as plt
import json

# 加载结果
with open("experiments/my_experiment/results.json", 'r') as f:
    results = json.load(f)

# 绘制学习曲线
checkpoints = results['checkpoints']
accuracies = [c['metrics']['accuracy'] for c in checkpoints]

plt.plot(accuracies)
plt.xlabel('Checkpoint')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.savefig('learning_curve.png')
```

---

## 6. 最佳实践

1. **数据质量**: 确保任务数据通过验证，避免无效数据影响实验
2. **多次运行**: 使用不同seed运行多次实验，计算平均结果和标准差
3. **对比基线**: 始终包含baseline实验（无CL）作为对比
4. **渐进式调试**: 先用小数据集快速验证，再扩展到完整实验
5. **保存中间结果**: 定期保存agent状态和memory buffer

---

更多信息请参考：
- [设计文档](./continuous_learning_framework_design.md)
- [API文档](../src/tau2/continual_learning/README.md)
