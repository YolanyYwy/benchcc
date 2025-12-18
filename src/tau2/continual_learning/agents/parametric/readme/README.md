
# Parametric Continual Learning for Agents

## 核心理念

这个模块实现了**真正的Agent层持续学习**，与传统的ICL-ER（In-Context Learning Experience Replay）有本质区别：

### ICL-ER的局限性
- ❌ 仅将历史经验加入prompt
- ❌ Agent没有可学习参数
- ❌ 没有真正的"学习"，只有"记忆"
- ❌ 本质上是few-shot prompting，不是持续学习

### 我们的方法
- ✅ **LLM参数冻结** - 持续学习发生在Agent层和Memory层
- ✅ **显式可学习参数** - Tool Scorer权重 w_i、Memory重要性权重 α_i
- ✅ **梯度更新** - 使用真正的参数更新，不只是prompt工程
- ✅ **防遗忘机制** - EWC、Replay等经典持续学习方法

---

## 架构设计

```
Agent (Parametric)
├── LLM (frozen) ❄️
│   └── 只用于语言生成和状态嵌入提取
│
├── Tool Scorer (learnable) 📊
│   ├── 参数: w_i ∈ R^{num_tools × embedding_dim}
│   ├── 功能: score(s, tool_i) = w_i^T φ(s)
│   └── 更新: 梯度下降 + 可选EWC正则化
│
├── Parametric Memory (learnable) 🧠
│   ├── 参数: α_i (importance weights)
│   ├── 功能: 可学习的经验重要性
│   └── 更新: 基于检索效用的梯度
│
└── Update Rule (method-specific) 🔄
    ├── EWC: Fisher Information正则化
    ├── Replay: 梯度混合
    └── 其他方法...
```

---

## 核心组件

### 1. ToolScorer
**文件**: `tool_scorer.py`

可学习的工具选择模块：
```python
# 工具评分
score(s, tool_i) = w_i^T φ(s)

# 参数更新（REINFORCE-style）
∇w_i = α * (reward - baseline) * ∇log π(tool_i | s)
```

**关键特性**：
- 为每个工具维护权重向量 w_i
- 支持Fisher Information计算（用于EWC）
- 支持EWC正则化惩罚

### 2. ParametricMemory
**文件**: `parametric_memory.py`

带可学习重要性权重的记忆系统：
```python
# 每条经验有三部分
m_i = (z_i, τ_i, α_i)
# z_i: 轨迹嵌入（固定）
# τ_i: 时间戳（固定）
# α_i: 重要性权重（可学习）
```

**关键特性**：
- 基于重要性的采样和检索
- 重要性权重的梯度更新
- 时间衰减机制

### 3. ParametricCLAgent
**文件**: `base.py`

参数化持续学习Agent基类：

**核心流程**：
```python
1. 提取状态嵌入 φ(s) ← LLM (frozen)
2. 工具选择 ← ToolScorer (learnable)
3. 执行动作 ← Environment
4. 参数更新 ← 子类实现
5. 存储经验 ← ParametricMemory
```

---

## 实现的方法

### 方法1: EWC (Elastic Weight Consolidation)
**文件**: `ewc_agent.py`

**核心思想**: 通过Fisher Information保护重要参数

```python
# 目标函数
L = L_task + (λ/2) * Σ_i F_i * (θ_i - θ_i*)^2

# Fisher Information
F_i = E[(∂log π(a|s) / ∂θ_i)^2]
```

**特性**：
- ✅ 在线EWC：累积多个任务的Fisher
- ✅ 自适应λ：随任务数增长
- ✅ 参数重要性分析

**使用示例**：
```python
from tau2.continual_learning.agents.parametric import EWCContinualLearningAgent

agent = EWCContinualLearningAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    embedding_dim=768,
    learning_rate=0.01,
    ewc_lambda=1.0,           # EWC强度
    online_ewc=True,          # 使用在线EWC
    ewc_lambda_growth="adaptive",  # λ增长策略
    fisher_sample_size=100,   # Fisher计算样本数
)

# 学习循环
for task in tasks:
    trajectory = run_task(task)
    stats = agent.learn_from_trajectory(
        task_id=task.id,
        domain=task.domain,
        trajectory=trajectory,
        reward=get_reward(trajectory),
        success=is_success(trajectory),
    )
    # stats包含Fisher计算和EWC统计信息
```

### 方法2: Replay-based Continual Learning
**文件**: `replay_agent.py`

**核心思想**: 通过回放旧经验的梯度来防止遗忘

```python
# 梯度混合
g_total = (1-α) * g_current + α * g_replay

# 经验检索基于可学习重要性
experiences ~ ParametricMemory.sample_by_importance(α_i)
```

**特性**：
- ✅ 参数化经验回放（不只是prompt）
- ✅ 梯度混合策略
- ✅ 可学习的记忆重要性
- ✅ 多种检索策略（重要性/相似度/混合）

**使用示例**：
```python
from tau2.continual_learning.agents.parametric import ReplayContinualLearningAgent

agent = ReplayContinualLearningAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    embedding_dim=768,
    learning_rate=0.01,
    replay_ratio=0.5,          # 回放梯度权重
    replay_batch_size=5,       # 每次回放经验数
    replay_strategy="importance",  # 检索策略
    update_memory_importance=True,  # 更新重要性
    replay_frequency=1,        # 回放频率
)

# 学习循环
for task in tasks:
    trajectory = run_task(task)
    stats = agent.learn_from_trajectory(
        task_id=task.id,
        domain=task.domain,
        trajectory=trajectory,
        reward=get_reward(trajectory),
        success=is_success(trajectory),
    )
    # stats包含回放统计和重要性更新信息
```

---

## 与ICL-ER的对比

| 特性 | ICL-ER | EWC Agent | Replay Agent |
|-----|--------|-----------|--------------|
| **可学习参数** | ❌ 无 | ✅ w_i, α_i | ✅ w_i, α_i |
| **参数更新** | ❌ 无 | ✅ 梯度+EWC | ✅ 梯度+Replay |
| **防遗忘机制** | ❌ 无 | ✅ Fisher正则化 | ✅ 经验回放 |
| **真正的学习** | ❌ 只是记忆 | ✅ 是 | ✅ 是 |
| **工具选择** | LLM决定 | Scorer决定 | Scorer决定 |
| **记忆管理** | 固定采样 | 可学习重要性 | 可学习重要性 |

---

## 完整使用示例

```python
from tau2.continual_learning.agents.parametric import (
    EWCContinualLearningAgent,
    ReplayContinualLearningAgent,
)
from tau2.continual_learning.orchestrator import CLOrchestrator
from tau2.environment.tool import Tool

# 1. 定义工具
tools = [
    Tool.from_function(search_database),
    Tool.from_function(send_email),
    Tool.from_function(create_ticket),
]

# 2. 创建Agent
agent = EWCContinualLearningAgent(
    tools=tools,
    domain_policy=load_policy("customer_service"),
    llm="gpt-4",
    embedding_dim=768,
    learning_rate=0.01,
    ewc_lambda=1.0,
    online_ewc=True,
)

# 3. 创建持续学习Orchestrator
orchestrator = CLOrchestrator(
    agent=agent,
    curriculum=SequentialCurriculum(domains=["domain1", "domain2", "domain3"]),
    eval_frequency=10,
    save_checkpoints=True,
)

# 4. 运行持续学习实验
results = orchestrator.run(
    num_tasks_per_domain=50,
    eval_on_all_domains=True,
)

# 5. 分析结果
print("平均性能:", results["avg_performance"])
print("遗忘度:", results["forgetting"])
print("前向迁移:", results["forward_transfer"])

# 6. 可视化
import matplotlib.pyplot as plt

plt.plot(results["performance_over_time"])
plt.xlabel("Task")
plt.ylabel("Performance")
plt.title("Continual Learning Performance")
plt.show()

# 7. 保存Agent
agent.save_state("checkpoints/agent_final.json")

# 8. 加载Agent
agent.load_state("checkpoints/agent_final.json")
```

---

## 状态嵌入提取

**关键问题**: 如何从冻结的LLM中提取状态嵌入 φ(s)？

**当前实现** (`base.py:_extract_state_embedding`):
```python
def _extract_state_embedding(self, messages):
    # 方法1: 使用OpenAI Embedding API
    text = extract_text(messages)
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=768,
    )
    return embedding

    # 方法2: 提取LLM隐藏层（需要模型访问）
    # hidden_states = llm.get_hidden_states(messages)
    # return hidden_states[-1].mean(dim=1)
```

**改进方向**：
- 使用模型的实际hidden states（如果可访问）
- 使用对比学习优化嵌入空间
- 任务特定的嵌入适配器

---

## 评估指标

### 1. 平均性能 (Average Performance)
所有任务的平均成功率

### 2. 遗忘度 (Forgetting)
```
F = (1/T) * Σ_t max_k(P_{t,k} - P_{T,k})
```
其中 P_{t,k} 是在学习任务t后，在任务k上的性能

### 3. 前向迁移 (Forward Transfer)
```
FWT = (1/T) * Σ_t (P_{t,t} - P_{0,t})
```
学习新任务时相比随机初始化的提升

### 4. 参数变化分析
- Fisher Information的分布
- 参数更新的幅度
- 重要性权重的演化

---

## 实验建议

### 1. Baseline对比
- **No-CL**: 每个任务独立训练
- **Joint**: 所有任务联合训练（上界）
- **ICL-ER**: 原始的prompt-based方法
- **EWC**: 我们的EWC Agent
- **Replay**: 我们的Replay Agent

### 2. 消融实验
- Tool Scorer的影响
- Memory importance的影响
- 不同λ值的影响
- 不同replay ratio的影响

### 3. 任务序列
- Sequential: 顺序学习（最难）
- Interleaved: 交错学习
- Curriculum: 由易到难

---

## 文件结构

```
parametric/
├── __init__.py              # 模块入口
├── tool_scorer.py           # 可学习工具选择器
├── parametric_memory.py     # 可学习记忆系统
├── base.py                  # 参数化CL Agent基类
├── ewc_agent.py            # EWC方法实现
├── replay_agent.py         # Replay方法实现
├── README.md               # 本文档
└── example_usage.py        # 使用示例
```

---

## 未来扩展

### 其他持续学习方法
- [ ] **Progressive Networks**: 为新任务添加新模块
- [ ] **PackNet**: 参数分配和打包
- [ ] **Meta-CL**: 学习如何学习的元参数
- [ ] **Parameter Isolation**: 任务特定参数子集

### 改进方向
- [ ] 更好的状态嵌入提取方法
- [ ] 自适应学习率调整
- [ ] 多模态状态表示
- [ ] 分层记忆系统

---

## 引用

如果使用本代码，请引用：

```bibtex
@inproceedings{tau2-parametric-cl,
  title={Parametric Continual Learning for Tool-using Agents},
  author={Your Name},
  booktitle={Proceedings of ...},
  year={2025}
}
```

---

## 核心创新总结

1. **首次**将经典持续学习方法系统地映射到Agent Tool-use层
2. **明确**区分了LLM层（冻结）和Agent层（可学习）
3. **引入**显式可学习参数（Tool Scorer权重、Memory重要性）
4. **实现**真正的梯度更新，而非仅仅prompt工程
5. **提供**统一框架，便于公平比较不同CL方法

这是真正的**Agent-level Continual Learning**，而不是伪装的few-shot learning！
