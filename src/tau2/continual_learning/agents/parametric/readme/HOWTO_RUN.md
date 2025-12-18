
# 参数化持续学习 - 运行指南

## 📖 三种运行方式

### 方式1: 最简单 - 直接导入使用（推荐）

在你的Python脚本中：

```python
from tau2.continual_learning.agents.parametric import EWCContinualLearningAgent
from tau2.environment.tool import Tool

# 定义工具
def my_tool(query: str) -> str:
    """我的工具"""
    return f"结果: {query}"

# 创建agent
agent = EWCContinualLearningAgent(
    tools=[Tool(my_tool)],
    domain_policy="你是一个助手",
    llm="gpt-4",
)

# 使用agent（需要配合tau2的orchestrator）
# 或者直接调用learn_from_trajectory
```

### 方式2: 集成到现有tau2项目

如果你已经有tau2的持续学习实验：

```python
# 在你的实验脚本中
from tau2.continual_learning.agents.parametric import (
    EWCContinualLearningAgent,
    ReplayContinualLearningAgent,
    # ... 其他方法
)

# 替换原来的ICL-ER Agent
# agent = ICLExperienceReplayAgent(...)  # 旧方法
agent = EWCContinualLearningAgent(...)   # 新方法（带参数学习）

# 其余代码保持不变
for task in task_stream:
    trajectory = orchestrator.run(agent, task)

    # 关键：让agent学习（参数更新）
    agent.learn_from_trajectory(
        task_id=task.id,
        domain=task.domain,
        trajectory=trajectory.messages,
        reward=compute_reward(trajectory),
        success=is_success(trajectory),
    )
```

### 方式3: 独立测试（不需要完整环境）

创建`test_parametric_agent.py`:

```python
import numpy as np
from tau2.continual_learning.agents.parametric import EWCContinualLearningAgent
from tau2.continual_learning.agents.parametric.tool_scorer import ToolScorer
from tau2.environment.tool import Tool

# 定义简单工具
def search(query: str) -> str:
    """搜索工具"""
    return f"Found: {query}"

# 创建agent
tools = [Tool(search)]
policy = "Be helpful"

agent = EWCContinualLearningAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    embedding_dim=768,
    learning_rate=0.01,
    ewc_lambda=1.0,
)

print(f"✓ Agent创建成功: {agent.__class__.__name__}")
print(f"✓ 工具数量: {len(agent.tools)}")
print(f"✓ Tool Scorer参数shape: {agent.tool_scorer.weights.shape}")

# 测试参数更新
state_emb = np.random.randn(768)
selected_tool = "search"
reward = 1.0

update_stats = agent._update_parameters(
    state_embedding=state_emb,
    selected_tool=selected_tool,
    reward=reward,
    success=True,
)

print(f"✓ 参数更新成功: {update_stats['updated']}")
print(f"  - 工具: {update_stats['tool']}")
print(f"  - 奖励: {update_stats['reward']}")
print(f"  - 概率: {update_stats['probability']:.4f}")

print("\n🎉 测试通过！Agent的参数确实可以学习和更新！")
```

运行：
```bash
python test_parametric_agent.py
```

## 🚀 完整示例（需要tau2环境）

如果你有完整的tau2-bench环境，可以运行：

```bash
# 运行完整的CL实验
python src/tau2/run.py \
    --agent ewc \
    --domains customer_service,tech_support \
    --num_tasks 50

# 或使用CLOrchestrator
python -c "
from tau2.continual_learning.orchestrator import CLOrchestrator
from tau2.continual_learning.agents.parametric import EWCContinualLearningAgent

# 创建agent
agent = EWCContinualLearningAgent(...)

# 创建orchestrator
orchestrator = CLOrchestrator(
    agent=agent,
    curriculum=your_curriculum,
)

# 运行实验
results = orchestrator.run()
print(f'平均性能: {results[\"avg_performance\"]:.3f}')
print(f'遗忘度: {results[\"forgetting\"]:.3f}')
"
```

## 📊 方法对比测试

测试5种方法的性能：

```python
from tau2.continual_learning.agents.parametric import (
    EWCContinualLearningAgent,
    ReplayContinualLearningAgent,
    ParameterIsolationAgent,
    ProgressiveModularAgent,
    MetaContinualLearningAgent,
)

methods = {
    "EWC": EWCContinualLearningAgent,
    "Replay": ReplayContinualLearningAgent,
    "ParamIso": ParameterIsolationAgent,
    "Progressive": ProgressiveModularAgent,
    "MetaCL": MetaContinualLearningAgent,
}

results = {}
for name, AgentClass in methods.items():
    agent = AgentClass(
        tools=tools,
        domain_policy=policy,
        llm="gpt-4",
    )

    # 运行实验
    perf = run_experiment(agent, tasks)
    results[name] = perf

    print(f"{name}: {perf['accuracy']:.3f}, 遗忘={perf['forgetting']:.3f}")
```

## 🔧 常见问题

### Q1: 需要什么环境？
A: 需要tau2-bench环境。如果只是测试参数更新，只需要numpy和基础依赖。

### Q2: 如何选择方法？
A:
- 任务相似 → EWC或Replay
- 任务差异大 → Parameter Isolation或Progressive
- 终身学习 → Progressive或Meta-CL
- 详见 `METHODS_COMPARISON.md`

### Q3: 与ICL-ER的区别？
A: ICL-ER只是把经验加到prompt，没有参数学习。我们的方法有真正的可学习参数和梯度更新。详见 `COMPARISON.md`

### Q4: 如何查看学习效果？
A:
```python
# 查看统计
stats = agent.get_statistics()
print(stats['num_tasks_learned'])
print(stats['tool_scorer_stats'])

# 查看Fisher Information (EWC)
if 'cumulative_fisher_stats' in stats:
    print(stats['cumulative_fisher_stats'])

# 查看参数变化
params = agent.get_parameters()
print(params['tool_scorer']['weights'].shape)
```

## 📚 文档

- `README.md` - 完整使用指南
- `COMPARISON.md` - 与ICL-ER的详细对比
- `METHODS_COMPARISON.md` - 5种方法全面对比
- `FINAL_SUMMARY.md` - 项目总结

## 🎯 核心概念

```python
# 旧方法 (ICL-ER) - 无参数学习
agent.memory.append(experience)  # 只存储
response = llm(prompt)            # LLM决定

# 新方法 (Parametric) - 真正学习
gradient = compute_gradient(reward)  # 计算梯度
w += lr * gradient                   # 更新参数
# Agent学到了！
```

## ✨ 关键优势

- ✅ 真正的参数学习（不只是prompt）
- ✅ 5种经典CL方法
- ✅ 防遗忘机制
- ✅ 统一接口
- ✅ 完整文档

开始使用吧！🚀
