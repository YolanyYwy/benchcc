# 完整持续学习方法对比

## 方法总览

我们实现了5种参数化持续学习方法，每种方法都有不同的防遗忘机制和适用场景。

| 方法 | 核心思想 | 防遗忘机制 | 参数量 | 适用场景 |
|-----|---------|-----------|--------|---------|
| **EWC** | 保护重要参数 | Fisher Information | 基础 | 任务相似度中等 |
| **Replay** | 梯度级回放 | 经验重放 | 基础 + Memory | 需要强记忆 |
| **Parameter Isolation** | 参数隔离 | 任务子空间 | 基础 × N_tasks | 任务差异大 |
| **Progressive** | 模块扩展 | 冻结旧模块 | 基础 × N_modules | 增量学习 |
| **Meta-CL** | 学习如何学习 | 自适应策略 | 基础 + Meta | 长期学习 |

---

## 方法 1: EWC (Elastic Weight Consolidation)

### 核心原理

通过Fisher Information Matrix保护重要参数：

```
L = L_task + (λ/2) * Σ_i F_i * (θ_i - θ_i*)^2

where:
F_i = E[(∂log π(a|s) / ∂θ_i)^2]  # 参数重要性
```

### 数学推导

1. **Fisher Information计算**：
   ```python
   for each experience (s, a):
       gradient = ∂log π(a|s) / ∂θ
       F += gradient^2
   F /= num_samples
   ```

2. **EWC更新**：
   ```python
   gradient_task = ∂L_task / ∂θ
   ewc_penalty = λ * F * (θ - θ*)
   θ ← θ + α * (gradient_task - ewc_penalty)
   ```

### 优势
- ✅ 数学上严格的防遗忘
- ✅ 可量化参数重要性
- ✅ 支持在线EWC（累积多任务Fisher）
- ✅ 计算高效

### 劣势
- ❌ 假设任务相似（diagonal Fisher近似）
- ❌ 对任务顺序敏感
- ❌ 需要存储Fisher矩阵

### 使用场景
- 任务之间有一定相似性
- 需要定量分析遗忘
- 任务数量适中（<20）

### 代码示例
```python
from tau2.continual_learning.agents.parametric import EWCContinualLearningAgent

agent = EWCContinualLearningAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    ewc_lambda=1.0,           # EWC强度
    online_ewc=True,          # 累积Fisher
    ewc_lambda_growth="adaptive",  # λ自适应增长
    fisher_sample_size=100,   # Fisher计算样本数
)
```

---

## 方法 2: Replay-based Continual Learning

### 核心原理

通过回放旧经验的梯度来保持知识：

```
g_total = (1-α) * g_current + α * E[g_replay]

where:
g_replay ~ Memory.sample_by_importance(α_i)
```

### 关键机制

1. **参数化Memory**：
   - 每条经验有可学习重要性权重 α_i
   - 基于重要性采样回放经验

2. **梯度混合**：
   ```python
   g_current = ∂L_task / ∂θ
   g_replay = mean([∂L_exp / ∂θ for exp in replayed])
   g_total = (1-β) * g_current + β * g_replay
   ```

3. **重要性更新**：
   ```python
   if replay_helpful:
       α_i += lr * reward
   else:
       α_i -= lr * (1 - reward)
   ```

### 优势
- ✅ 梯度级知识保留
- ✅ 学习哪些经验重要
- ✅ 灵活的回放策略
- ✅ 不假设任务结构

### 劣势
- ❌ 需要存储经验（内存开销）
- ❌ 回放计算开销
- ❌ 采样策略影响效果

### 使用场景
- 需要强记忆保持
- 有足够内存存储经验
- 任务分布复杂

### 代码示例
```python
from tau2.continual_learning.agents.parametric import ReplayContinualLearningAgent

agent = ReplayContinualLearningAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    replay_ratio=0.5,          # 回放梯度权重
    replay_batch_size=5,       # 回放经验数
    replay_strategy="importance",  # 采样策略
    update_memory_importance=True,  # 学习重要性
    replay_frequency=1,        # 回放频率
)
```

---

## 方法 3: Parameter Isolation (Agent Adapter)

### 核心原理

为不同任务族分配独立参数子集：

```
w = [w_shared, w_task1, w_task2, ..., w_taskN]

Router: p(task_k | s) = softmax(W_route @ φ(s))
Score: score(tool_i | s, task_k) = w_i^k @ φ(s)
```

### 架构设计

```
State s
  ↓
φ(s) (shared embedding)
  ↓
Router → select task_k
  ↓
w^k (task-specific params)
  ↓
Tool scores
```

### 关键组件

1. **Task Router**：
   - 输入：state embedding
   - 输出：task probabilities
   - 可学习的路由权重

2. **Task-Specific Scorers**：
   - 每个任务族独立的参数
   - 可选：共享参数 + 任务参数

3. **自动任务分配**：
   - 检测新domain
   - 分配或重用任务族

### 优势
- ✅ 完全防遗忘（参数隔离）
- ✅ 支持任务差异大的场景
- ✅ 可扩展到多任务
- ✅ 可解释（明确的任务划分）

### 劣势
- ❌ 参数量随任务增长
- ❌ 需要任务划分
- ❌ 新任务冷启动

### 使用场景
- 任务之间差异很大
- 需要零遗忘保证
- 可以预定义任务族

### 代码示例
```python
from tau2.continual_learning.agents.parametric import ParameterIsolationAgent

agent = ParameterIsolationAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    num_task_families=5,       # 任务族数量
    use_shared_parameters=True,  # 使用共享参数
    shared_weight=0.3,         # 共享参数权重
    auto_create_tasks=True,    # 自动创建任务族
)
```

---

## 方法 4: Progressive / Modular Continual Learning

### 核心原理

逐步扩展模块，冻结旧模块：

```
for new_task:
    1. Freeze all old modules
    2. Add new module
    3. Learn routing π_route
    4. Learn new module params
    5. NO backprop to old modules
```

### 模块结构

```
Module_0 (frozen)
  ↓
Module_1 (frozen)
  ↓
Module_2 (frozen)
  ↓
Module_3 (active) ← 当前学习
```

### 关键机制

1. **Module**：
   - 一个module = 一个ToolScorer
   - 可冻结/解冻
   - 追踪使用频率

2. **Module Router**：
   - 注意力机制路由
   - 学习模块选择策略
   - 支持模块组合

3. **冻结策略**：
   - 任务切换时冻结
   - 保护旧知识
   - 只学习新模块

### 优势
- ✅ 绝对防遗忘（冻结）
- ✅ 容量随任务增长
- ✅ 模块可复用
- ✅ 适合终身学习

### 劣势
- ❌ 参数量持续增长
- ❌ 模块数量限制
- ❌ 旧模块不更新

### 使用场景
- 终身学习系统
- 持续添加新能力
- 需要增量扩展

### 代码示例
```python
from tau2.continual_learning.agents.parametric import ProgressiveModularAgent

agent = ProgressiveModularAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    freeze_on_task_change=True,  # 任务切换时冻结
    max_modules=10,              # 最大模块数
    use_attention_routing=True,  # 注意力路由
    allow_module_composition=True,  # 允许模块组合
)
```

---

## 方法 5: Meta-Continual Learning

### 核心原理

学习**如何学习**，优化学习过程本身：

```
Meta-Parameters:
- learning_rate: 基础学习率
- memory_write_threshold: 记忆写入阈值
- memory_decay_rate: 记忆衰减率
- retrieval_k: 检索数量
- replay_ratio: 回放比例
- ewc_lambda: EWC强度

Meta-Objective:
L_meta = -long_term_performance + λ * forgetting_rate
```

### 元学习过程

```
1. 执行任务，收集性能
2. 计算长期性能和遗忘率
3. 计算meta-loss
4. 更新meta-parameters
5. 使用新meta-params继续学习
```

### 自适应策略

```python
if forgetting_rate > 0.3:
    ewc_lambda += α * forgetting_rate
    replay_ratio += α * forgetting_rate

if performance < 0.6:
    learning_rate += β * (0.6 - performance)

if performance < 0.5:
    retrieval_k += 1
elif performance > 0.8:
    retrieval_k -= 0.5
```

### 优势
- ✅ 自适应学习策略
- ✅ 长期性能优化
- ✅ 结合多种方法
- ✅ 少需人工调参

### 劣势
- ❌ 元学习复杂度高
- ❌ 需要足够训练数据
- ❌ 调试困难

### 使用场景
- 长期持续学习
- 动态环境
- 希望自动调参

### 代码示例
```python
from tau2.continual_learning.agents.parametric import MetaContinualLearningAgent

agent = MetaContinualLearningAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
    meta_learning_rate=0.001,  # 元学习率
    performance_window=10,     # 性能窗口
    enable_ewc=True,           # 启用EWC
    enable_replay=True,        # 启用Replay
    adapt_frequency=5,         # 自适应频率
)
```

---

## 方法对比矩阵

### 防遗忘效果
```
EWC:              ███████░░░  70%
Replay:           ████████░░  80%
Param Isolation:  ██████████  100% (完全隔离)
Progressive:      ██████████  100% (冻结)
Meta-CL:          ████████░░  80%
```

### 参数效率
```
EWC:              ██████████  最高 (只存Fisher)
Replay:           ████████░░  高 (需存经验)
Param Isolation:  ███░░░░░░░  低 (N倍参数)
Progressive:      ██░░░░░░░░  很低 (持续增长)
Meta-CL:          █████████░  高
```

### 学习效率
```
EWC:              ████████░░  高
Replay:           ███████░░░  中 (回放开销)
Param Isolation:  █████████░  高 (无干扰)
Progressive:      ██████████  最高 (专注新任务)
Meta-CL:          ██████░░░░  中 (元学习开销)
```

### 可扩展性
```
EWC:              ███████░░░  中 (Fisher累积)
Replay:           ██████░░░░  中 (内存限制)
Param Isolation:  ████░░░░░░  低 (任务数限制)
Progressive:      ████░░░░░░  低 (模块数限制)
Meta-CL:          ██████████  最高 (自适应)
```

---

## 实验配置建议

### Scenario 1: 顺序学习3个相似domain
**推荐**: EWC 或 Replay

```python
agent = EWCContinualLearningAgent(
    ewc_lambda=1.0,
    online_ewc=True,
)
```

### Scenario 2: 顺序学习10个差异大的domain
**推荐**: Parameter Isolation 或 Progressive

```python
agent = ParameterIsolationAgent(
    num_task_families=10,
    use_shared_parameters=True,
)
```

### Scenario 3: 终身学习（持续添加新任务）
**推荐**: Progressive 或 Meta-CL

```python
agent = ProgressiveModularAgent(
    max_modules=20,
    freeze_on_task_change=True,
)
```

### Scenario 4: 需要强记忆（complex tools）
**推荐**: Replay 或 Meta-CL

```python
agent = ReplayContinualLearningAgent(
    replay_ratio=0.7,
    replay_batch_size=10,
)
```

---

## 完整对比表

| 特性 | EWC | Replay | Param Isolation | Progressive | Meta-CL |
|-----|-----|--------|----------------|-------------|---------|
| **防遗忘** | Fisher正则 | 梯度回放 | 参数隔离 | 模块冻结 | 自适应 |
| **参数量** | 1x + Fisher | 1x + Memory | Nx | Nx (增长) | 1x + Meta |
| **计算开销** | 低 | 中 | 低 | 低 | 中 |
| **内存开销** | 低 | 高 | 低 | 中 | 低 |
| **任务数限制** | 中 | 中 | 是 | 是 | 否 |
| **任务相似性** | 需要 | 不需要 | 不需要 | 不需要 | 不需要 |
| **可解释性** | 中 | 低 | 高 | 高 | 低 |
| **调参难度** | 中 (λ) | 中 (ratio) | 低 | 低 | 低 (自动) |

---

## 总结与选择指南

### 快速选择决策树

```
开始
  ↓
任务差异大?
  ├─ 是 → Parameter Isolation / Progressive
  └─ 否 ↓
     ↓
需要强记忆?
  ├─ 是 → Replay
  └─ 否 ↓
     ↓
任务数量多 (>20)?
  ├─ 是 → Meta-CL / Progressive
  └─ 否 → EWC
```

### 组合使用建议

可以组合多种方法：
- **EWC + Replay**: 结合正则化和回放
- **Parameter Isolation + Meta-CL**: 隔离 + 元学习
- **Progressive + Replay**: 模块扩展 + 回放

---

## 性能预期

基于理论分析，预期性能排序（在不同场景下）：

### 相似任务序列
```
1. EWC
2. Replay
3. Meta-CL
4. Progressive
5. Parameter Isolation
```

### 差异大的任务序列
```
1. Parameter Isolation / Progressive
2. Replay
3. Meta-CL
4. EWC
```

### 长期学习 (50+ tasks)
```
1. Meta-CL
2. Progressive
3. Parameter Isolation
4. Replay
5. EWC
```

---

*所有方法都已实现并可用，详见各自的代码文件和README*
