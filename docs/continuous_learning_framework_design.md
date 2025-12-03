# Tau2-CL: 持续学习 Tool Use 能力测试框架开发文档

## 项目概述

### 项目名称
**Tau2-CL (Tau2-Continuous Learning)**: 基于 Tau2-Bench 的 Agent Tool Use 持续学习能力评估框架

### 项目目标
将 Tau2-Bench 从静态的 Agent 评估基准转变为一个可以系统性测试和评估持续学习算法在 Tool Use 领域效果的综合性平台。该平台将支持多种持续学习算法，提供标准化的评估指标，并能够模拟真实世界中 Agent 需要不断学习新工具和适应新环境的场景。

---

## 第一部分：核心设计理念

### 1.1 持续学习的定义与挑战

在 Tool Use 场景下的持续学习具有以下特点：

1. **任务序列性 (Task Sequentiality)**
   - Agent 按顺序面对不同的任务
   - 早期任务的学习应该有助于后期任务
   - 需要在新旧知识之间保持平衡

2. **灾难性遗忘 (Catastrophic Forgetting)**
   - 学习新工具/新domain时可能遗忘旧知识
   - 需要机制保持已掌握的工具使用能力

3. **正向迁移 (Positive Transfer)**
   - 从相似工具中学到的经验应该能够迁移
   - 跨domain的工具使用模式应该能够泛化

4. **负向迁移 (Negative Transfer)**
   - 不同domain的策略可能相互干扰
   - 需要避免错误的工具使用模式泛化

### 1.2 设计原则

1. **最小侵入性**: 尽可能复用 Tau2-Bench 现有架构，通过扩展而非重写实现新功能
2. **模块化设计**: 各持续学习算法作为独立模块，易于添加新算法
3. **标准化接口**: 统一的数据格式和评估指标，确保算法间可比性
4. **可扩展性**: 支持自定义curriculum、新的domain和评估指标
5. **完整追踪**: 记录学习全过程，支持详细的分析和可视化

---

## 第二部分：技术架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tau2-CL Framework                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Curriculum Manager (课程管理器)                  │  │
│  │  - Task Sequencing (任务排序)                              │  │
│  │  - Difficulty Progression (难度递进)                       │  │
│  │  - Domain Mixing Strategies (领域混合策略)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      Continual Learning Orchestrator (持续学习编排器)       │  │
│  │  - Training Loop Control (训练循环控制)                    │  │
│  │  - Experience Collection (经验收集)                        │  │
│  │  - Evaluation Scheduling (评估调度)                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Learning Agent Interface (学习Agent接口)           │  │
│  │             专为无参数API Agent设计                        │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │  │
│  │  │   ICL-Based    │  │   Prompt       │  │   Memory   │  │  │
│  │  │   Experience   │  │   Strategy     │  │   Augmented│  │  │
│  │  │   Replay       │  │   Evolution    │  │   Retrieval│  │  │
│  │  │  - Few-shot    │  │  - Multi       │  │  - Episodic│  │  │
│  │  │  - Retrieval   │  │    Variants    │  │  - Semantic│  │  │
│  │  │  - Diversity   │  │  - Adaptive    │  │  - Pattern │  │  │
│  │  └────────────────┘  └────────────────┘  └────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Memory Management (记忆管理)                       │  │
│  │  - Experience Buffer (经验缓冲区)                          │  │
│  │  - Importance Sampling (重要性采样)                        │  │
│  │  - Memory Consolidation (记忆巩固)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │             Base Components (基础组件)                      │  │
│  │  ┌────────────┐  ┌──────────┐  ┌────────────────────┐    │  │
│  │  │ Tau2-Bench │  │   Gym    │  │   Evaluator        │    │  │
│  │  │ Orchestrator│  │ Interface│  │   (Extended)       │    │  │
│  │  └────────────┘  └──────────┘  └────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Metrics & Analysis (指标与分析)                     │  │
│  │  - Learning Curves (学习曲线)                              │  │
│  │  - Forgetting Metrics (遗忘度量)                           │  │
│  │  - Transfer Metrics (迁移度量)                             │  │
│  │  - Visualization Dashboard (可视化面板)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块详细设计

#### 2.2.1 Curriculum Manager (课程管理器)

**职责**: 管理任务序列，决定 Agent 的学习路径

**核心类设计**:

```python
# src/tau2/continual_learning/curriculum/base.py

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from enum import Enum

class CurriculumStrategy(str, Enum):
    """课程策略枚举"""
    SEQUENTIAL = "sequential"          # 顺序学习
    INTERLEAVED = "interleaved"        # 交错学习
    DIFFICULTY_BASED = "difficulty"    # 难度递进
    SIMILARITY_BASED = "similarity"    # 相似度聚类
    RANDOM = "random"                  # 随机顺序
    ANTI_CURRICULUM = "anti"           # 反课程（先难后易）

class TaskMetadata(BaseModel):
    """任务元数据"""
    task_id: str
    domain: str
    difficulty: Optional[float] = None  # 0.0-1.0
    required_tools: List[str]
    semantic_tags: List[str]
    estimated_steps: Optional[int] = None
    success_prerequisites: List[str] = []  # 前置任务

class CurriculumManager(ABC):
    """课程管理器基类"""

    def __init__(
        self,
        tasks: List[Task],
        strategy: CurriculumStrategy,
        config: Dict[str, Any]
    ):
        self.tasks = tasks
        self.strategy = strategy
        self.config = config
        self.task_metadata = self._compute_metadata()
        self.curriculum = self._generate_curriculum()
        self.current_index = 0

    @abstractmethod
    def _compute_metadata(self) -> List[TaskMetadata]:
        """计算任务元数据"""
        pass

    @abstractmethod
    def _generate_curriculum(self) -> List[str]:
        """生成课程序列（返回task_id列表）"""
        pass

    def get_next_task(self) -> Optional[Task]:
        """获取下一个任务"""
        pass

    def get_task_sequence(self, start: int = 0, end: Optional[int] = None) -> List[Task]:
        """获取任务序列"""
        pass

    def adapt_curriculum(self, performance_history: List[Dict]) -> None:
        """根据性能历史自适应调整课程"""
        pass
```

**实现的具体策略**:

1. **SequentialCurriculum**: 按domain顺序学习（airline -> retail -> telecom）
2. **InterleavedCurriculum**: domain交错学习，防止灾难性遗忘
3. **DifficultyBasedCurriculum**: 基于任务难度（工具数量、步骤数）递进
4. **SimilarityClusteredCurriculum**: 将相似任务聚类，逐cluster学习
5. **AdaptiveCurriculum**: 根据学习表现动态调整难度

#### 2.2.2 Continual Learning Orchestrator (持续学习编排器)

**职责**: 协调整个持续学习流程，管理训练-评估循环

**核心类设计**:

```python
# src/tau2/continual_learning/orchestrator.py

class CLPhase(str, Enum):
    """持续学习阶段"""
    TRAINING = "training"
    EVALUATION = "evaluation"
    CONSOLIDATION = "consolidation"  # 记忆巩固阶段

class CLExperimentConfig(BaseModel):
    """持续学习实验配置"""
    # Curriculum配置
    curriculum_strategy: CurriculumStrategy
    num_tasks_per_phase: int = 10  # 每个学习阶段的任务数

    # Training配置
    episodes_per_task: int = 5  # 每个任务训练的episode数
    batch_size: int = 32
    replay_buffer_size: int = 1000

    # Evaluation配置
    eval_frequency: int = 10  # 每N个任务评估一次
    eval_on_all_seen_tasks: bool = True  # 是否在所有见过的任务上评估

    # Algorithm配置
    cl_algorithm: str = "experience_replay"
    algorithm_config: Dict[str, Any] = {}

class ContinualLearningOrchestrator:
    """持续学习编排器"""

    def __init__(
        self,
        agent: "ContinualLearningAgent",
        curriculum_manager: CurriculumManager,
        config: CLExperimentConfig,
        logger: "CLLogger"
    ):
        self.agent = agent
        self.curriculum = curriculum_manager
        self.config = config
        self.logger = logger
        self.current_phase = CLPhase.TRAINING
        self.task_history = []
        self.performance_history = []

    def run_experiment(self) -> "ExperimentResults":
        """运行完整的持续学习实验"""
        while not self.curriculum.is_complete():
            # 1. 获取下一批任务
            tasks = self.curriculum.get_next_task_batch()

            # 2. 训练阶段
            self.current_phase = CLPhase.TRAINING
            for task in tasks:
                training_metrics = self._train_on_task(task)
                self.logger.log_training(task.id, training_metrics)

            # 3. 记忆巩固阶段（如果需要）
            if self.config.use_consolidation:
                self.current_phase = CLPhase.CONSOLIDATION
                self._consolidate_memory()

            # 4. 评估阶段
            if self._should_evaluate():
                self.current_phase = CLPhase.EVALUATION
                eval_results = self._evaluate()
                self.logger.log_evaluation(eval_results)
                self.performance_history.append(eval_results)

                # 自适应调整课程
                if self.config.adaptive_curriculum:
                    self.curriculum.adapt_curriculum(self.performance_history)

        # 最终评估
        final_results = self._final_evaluation()
        return self._compile_results(final_results)

    def _train_on_task(self, task: Task) -> Dict[str, Any]:
        """在单个任务上训练"""
        metrics = {
            "rewards": [],
            "steps": [],
            "success": []
        }

        for episode in range(self.config.episodes_per_task):
            # 使用Gym接口运行episode
            trajectory, reward, info = self._run_episode(task)

            # Agent学习
            self.agent.learn_from_trajectory(trajectory, task)

            metrics["rewards"].append(reward)
            metrics["steps"].append(len(trajectory))
            metrics["success"].append(info["success"])

        return metrics

    def _run_episode(self, task: Task) -> Tuple[List[Message], float, Dict]:
        """运行单个episode"""
        # 使用现有的Tau2-Bench Gym接口
        env = AgentGymEnv(
            domain=task.domain,
            task_id=task.id,
            agent=self.agent,
            user_llm="gpt-4.1"  # 可配置
        )

        trajectory = []
        done = False

        obs, info = env.reset()

        while not done:
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            trajectory.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": terminated or truncated
            })

            obs = next_obs
            done = terminated or truncated

        final_reward = info.get("final_reward", 0.0)
        return trajectory, final_reward, info

    def _evaluate(self) -> "EvaluationResults":
        """评估当前Agent性能"""
        results = EvaluationResults(
            timestamp=datetime.now(),
            tasks_seen=len(self.task_history),
            evaluations={}
        )

        # 评估所有见过的任务
        if self.config.eval_on_all_seen_tasks:
            for past_task in self.task_history:
                perf = self._evaluate_on_task(past_task, num_trials=3)
                results.evaluations[past_task.id] = perf

        # 计算聚合指标
        results.compute_aggregate_metrics()

        return results

    def _consolidate_memory(self) -> None:
        """记忆巩固"""
        # 可以实现各种巩固策略
        # 1. 在replay buffer中的重要样本上额外训练
        # 2. 知识蒸馏
        # 3. 伪rehearsal
        self.agent.consolidate()
```

#### 2.2.3 Continual Learning Agent (持续学习Agent)

**职责**: 实现各种持续学习算法的Agent

**重要说明**: 由于Agent是无参数的(仅调用API),因此不使用需要梯度更新的方法(如EWC、MAML、A-GEM)。我们采用适合无参数API Agent的持续学习方法。

**基类设计**:

```python
# src/tau2/continual_learning/agents/base.py

class ContinualLearningAgent(BaseAgent):
    """持续学习Agent基类 - 专为无参数API Agent设计"""

    def __init__(
        self,
        base_model: str,  # LLM模型名称 (如 "gpt-4", "claude-3")
        tools: List[Tool],
        domain_policy: str,
        memory_buffer: "MemoryBuffer",
        cl_config: Dict[str, Any]
    ):
        super().__init__()
        self.base_model = base_model
        self.tools = tools
        self.domain_policy = domain_policy
        self.memory = memory_buffer
        self.config = cl_config

        # API客户端初始化
        self.client = self._initialize_api_client()

        # 持续学习相关组件
        self.task_id_counter = 0
        self.current_task_id = None
        self.seen_tasks = set()

        # 示例库 (用于in-context learning)
        self.example_bank = []
        self.example_embeddings = []  # 用于快速检索

    @abstractmethod
    def learn_from_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        task: Task
    ) -> Dict[str, Any]:
        """从轨迹中学习 - 更新example bank或prompt策略"""
        pass

    def consolidate(self) -> None:
        """记忆巩固 - 整理和优化example bank"""
        pass

    def select_action(self, observation: str) -> str:
        """选择动作（生成下一条消息）- 通过API调用"""
        # 1. 检索相关示例
        relevant_examples = self._retrieve_examples(observation)

        # 2. 构建prompt
        prompt = self._build_prompt(observation, relevant_examples)

        # 3. 调用API
        response = self._call_api(prompt)

        return response

    def store_experience(
        self,
        experience: Dict[str, Any],
        task_id: str
    ) -> None:
        """存储经验到memory buffer"""
        self.memory.add(experience, task_id=task_id)

    @abstractmethod
    def _retrieve_examples(self, query: str, k: int = 5) -> List[Dict]:
        """从example bank检索相关示例"""
        pass

    @abstractmethod
    def _build_prompt(self, observation: str, examples: List[Dict]) -> str:
        """构建包含示例的prompt"""
        pass
```

**具体算法实现**:

##### A. In-Context Learning with Experience Replay (ICL-ER)

```python
# src/tau2/continual_learning/agents/icl_experience_replay.py

class ICLExperienceReplayAgent(ContinualLearningAgent):
    """使用In-Context Learning + Experience Replay的Agent

    核心思想: 将历史成功经验作为few-shot examples加入prompt,
    通过检索相关经验来增强当前任务的推理能力。
    """

    def __init__(self, *args, max_examples_in_prompt: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_examples_in_prompt = max_examples_in_prompt
        self.embedding_model = "text-embedding-ada-002"  # 用于检索

    def learn_from_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        task: Task
    ) -> Dict[str, Any]:
        """从轨迹中学习 - 将成功经验加入example bank"""

        added_count = 0

        # 1. 筛选高质量经验
        for step in trajectory:
            if self._is_good_example(step):
                # 存储到memory buffer
                self.store_experience(step, task.id)

                # 加入example bank
                example = {
                    "task_id": task.id,
                    "domain": task.domain,
                    "observation": step["obs"],
                    "action": step["action"],
                    "reward": step["reward"],
                    "tool_calls": step.get("tool_calls", []),
                    "timestamp": datetime.now()
                }
                self.example_bank.append(example)

                # 计算embedding用于后续检索
                embedding = self._get_embedding(step["obs"])
                self.example_embeddings.append(embedding)

                added_count += 1

        # 2. 保持example bank大小
        if len(self.example_bank) > self.config.max_examples:
            self._prune_examples()

        return {
            "examples_added": added_count,
            "total_examples": len(self.example_bank)
        }

    def _is_good_example(self, step: Dict[str, Any]) -> bool:
        """判断是否是值得保留的示例"""
        # 标准: 高奖励 + 成功的工具调用
        return (
            step.get("reward", 0) > 0.5 and
            step.get("tool_calls") is not None and
            len(step.get("tool_calls", [])) > 0
        )

    def _retrieve_examples(self, query: str, k: int = None) -> List[Dict]:
        """基于语义相似度检索examples"""
        if k is None:
            k = self.max_examples_in_prompt

        if not self.example_bank:
            return []

        # 1. 计算query embedding
        query_embedding = self._get_embedding(query)

        # 2. 计算相似度
        similarities = [
            self._cosine_similarity(query_embedding, ex_emb)
            for ex_emb in self.example_embeddings
        ]

        # 3. 选择top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # 4. 多样性过滤 (确保不同domain/task)
        selected_examples = self._diversify_examples(
            [self.example_bank[i] for i in top_k_indices]
        )

        return selected_examples

    def _build_prompt(self, observation: str, examples: List[Dict]) -> str:
        """构建few-shot prompt"""

        prompt_parts = [
            # System instruction
            f"You are a helpful agent. Follow the policy below:\n{self.domain_policy}\n",

            # Few-shot examples
            "Here are some examples of successful interactions:\n"
        ]

        for i, ex in enumerate(examples, 1):
            prompt_parts.append(
                f"\nExample {i}:\n"
                f"Observation: {ex['observation']}\n"
                f"Action: {ex['action']}\n"
            )

        # Current task
        prompt_parts.append(
            f"\nNow, given the following observation, generate the next action:\n"
            f"Observation: {observation}\n"
            f"Action:"
        )

        return "".join(prompt_parts)

    def _prune_examples(self) -> None:
        """修剪example bank - 保留最有价值的examples"""

        # 策略1: 保留每个task/domain的代表性examples
        # 策略2: 移除重复或相似的examples
        # 策略3: 保留高奖励的examples

        # 简单实现: 保留最近的 + 高奖励的
        sorted_examples = sorted(
            enumerate(self.example_bank),
            key=lambda x: (x[1]["reward"], x[1]["timestamp"]),
            reverse=True
        )

        keep_indices = [idx for idx, _ in sorted_examples[:self.config.max_examples]]

        self.example_bank = [self.example_bank[i] for i in keep_indices]
        self.example_embeddings = [self.example_embeddings[i] for i in keep_indices]

    def _diversify_examples(self, examples: List[Dict]) -> List[Dict]:
        """确保examples多样性"""
        # 确保不同domain和task的examples都有代表
        seen_domains = set()
        seen_tasks = set()
        diverse_examples = []

        for ex in examples:
            # 优先添加新domain/task的example
            if ex["domain"] not in seen_domains or ex["task_id"] not in seen_tasks:
                diverse_examples.append(ex)
                seen_domains.add(ex["domain"])
                seen_tasks.add(ex["task_id"])
            elif len(diverse_examples) < self.max_examples_in_prompt:
                diverse_examples.append(ex)

        return diverse_examples[:self.max_examples_in_prompt]

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本embedding"""
        # 使用OpenAI embedding API或本地模型
        response = openai.Embedding.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response['data'][0]['embedding'])

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

##### B. Prompt Strategy Evolution (PSE)

```python
# src/tau2/continual_learning/agents/prompt_strategy.py

class PromptStrategyAgent(ContinualLearningAgent):
    """通过演化prompt策略实现持续学习

    核心思想: 维护多个prompt变体,根据任务表现选择和演化最优策略。
    """

    def __init__(self, *args, num_prompt_variants: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_variants = num_prompt_variants

        # Prompt变体库
        self.prompt_variants = self._initialize_variants()
        self.variant_performance = {i: [] for i in range(num_prompt_variants)}

    def _initialize_variants(self) -> List[Dict]:
        """初始化prompt变体"""
        variants = []

        # 变体1: 标准few-shot
        variants.append({
            "name": "standard_few_shot",
            "template": "Here are examples:\n{examples}\n\nNow:\n{query}",
            "example_selection": "similarity"
        })

        # 变体2: Chain-of-thought
        variants.append({
            "name": "chain_of_thought",
            "template": "Let's solve step by step.\nExamples:\n{examples}\n\nQuery:\n{query}\nThinking:",
            "example_selection": "similarity"
        })

        # 变体3: Task-specific
        variants.append({
            "name": "task_specific",
            "template": "Task type: {task_type}\nExamples from this task:\n{examples}\n\nQuery:\n{query}",
            "example_selection": "same_task"
        })

        # 变体4: Cross-domain
        variants.append({
            "name": "cross_domain",
            "template": "Examples from different domains:\n{examples}\n\nQuery:\n{query}",
            "example_selection": "diverse"
        })

        # 变体5: Recent experience
        variants.append({
            "name": "recent",
            "template": "Recent successful actions:\n{examples}\n\nQuery:\n{query}",
            "example_selection": "recent"
        })

        return variants

    def learn_from_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        task: Task
    ) -> Dict[str, Any]:
        """学习并更新prompt策略"""

        # 1. 存储经验
        for step in trajectory:
            self.store_experience(step, task.id)
            if self._is_good_example(step):
                self.example_bank.append({
                    "task_id": task.id,
                    "domain": task.domain,
                    "observation": step["obs"],
                    "action": step["action"],
                    "reward": step["reward"]
                })

        # 2. 评估当前使用的variant
        avg_reward = np.mean([s["reward"] for s in trajectory])
        current_variant_id = self.config.get("current_variant", 0)
        self.variant_performance[current_variant_id].append(avg_reward)

        # 3. 决定是否切换variant
        if len(self.variant_performance[current_variant_id]) >= 10:
            best_variant = self._select_best_variant()
            self.config["current_variant"] = best_variant

        return {
            "avg_reward": avg_reward,
            "current_variant": self.config.get("current_variant", 0)
        }

    def _select_best_variant(self) -> int:
        """选择表现最好的variant"""
        avg_performance = {
            vid: np.mean(perfs[-10:]) if perfs else 0.0
            for vid, perfs in self.variant_performance.items()
        }
        return max(avg_performance.items(), key=lambda x: x[1])[0]

    def select_action(self, observation: str) -> str:
        """使用当前最优prompt variant生成action"""
        variant_id = self.config.get("current_variant", 0)
        variant = self.prompt_variants[variant_id]

        # 根据variant策略检索examples
        examples = self._retrieve_examples_by_strategy(
            observation,
            variant["example_selection"]
        )

        # 构建prompt
        prompt = self._build_prompt_with_variant(
            observation,
            examples,
            variant
        )

        # 调用API
        return self._call_api(prompt)
```

##### C. Memory-Augmented Retrieval (MAR)

```python
# src/tau2/continual_learning/agents/memory_augmented.py

class MemoryAugmentedAgent(ContinualLearningAgent):
    """使用结构化记忆的Agent

    核心思想: 维护分层的记忆结构(episodic + semantic),
    根据当前情境检索最相关的知识。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Episodic memory: 具体的经验实例
        self.episodic_memory = []

        # Semantic memory: 抽象的模式和策略
        self.semantic_memory = {
            "tool_usage_patterns": {},  # 工具使用模式
            "task_strategies": {},  # 任务策略
            "error_recoveries": []  # 错误恢复经验
        }

    def learn_from_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        task: Task
    ) -> Dict[str, Any]:
        """学习并更新记忆结构"""

        # 1. 添加到episodic memory
        for step in trajectory:
            if self._is_good_example(step):
                self.episodic_memory.append(step)
                self.store_experience(step, task.id)

        # 2. 提取semantic patterns
        self._extract_semantic_patterns(trajectory, task)

        # 3. 记忆巩固
        if len(self.episodic_memory) > self.config.max_episodic:
            self._consolidate_memory()

        return {
            "episodic_size": len(self.episodic_memory),
            "semantic_patterns": len(self.semantic_memory["tool_usage_patterns"])
        }

    def _extract_semantic_patterns(
        self,
        trajectory: List[Dict[str, Any]],
        task: Task
    ) -> None:
        """从具体经验中提取抽象模式"""

        # 提取工具使用序列
        tool_sequences = self._extract_tool_sequences(trajectory)
        for seq in tool_sequences:
            seq_key = "->".join(seq)
            if seq_key not in self.semantic_memory["tool_usage_patterns"]:
                self.semantic_memory["tool_usage_patterns"][seq_key] = {
                    "sequence": seq,
                    "success_count": 0,
                    "total_count": 0,
                    "domains": set()
                }
            self.semantic_memory["tool_usage_patterns"][seq_key]["total_count"] += 1
            self.semantic_memory["tool_usage_patterns"][seq_key]["domains"].add(task.domain)

    def _consolidate_memory(self) -> None:
        """记忆巩固: episodic -> semantic"""
        # 将重复的episodic experiences转化为semantic patterns
        # 移除冗余的episodic memories
        pass

    def _retrieve_examples(self, query: str, k: int = 5) -> List[Dict]:
        """多层检索: episodic + semantic"""

        # 1. Episodic retrieval
        episodic_examples = self._retrieve_episodic(query, k=k//2)

        # 2. Semantic retrieval
        semantic_hints = self._retrieve_semantic(query)

        # 3. 组合
        return episodic_examples + [{"type": "semantic", "content": h} for h in semantic_hints]
```

#### 2.2.4 Memory Management (记忆管理)

**职责**: 管理经验存储、采样和巩固

```python
# src/tau2/continual_learning/memory/buffer.py

class SamplingStrategy(str, Enum):
    UNIFORM = "uniform"
    RESERVOIR = "reservoir"
    IMPORTANCE = "importance"
    RECENCY = "recency"
    DIVERSE = "diverse"

class Experience(BaseModel):
    """单个经验"""
    task_id: str
    domain: str
    observation: str
    action: str
    reward: float
    next_observation: str
    done: bool
    timestamp: datetime
    importance_score: Optional[float] = None
    tool_calls: Optional[List[ToolCall]] = None

class MemoryBuffer:
    """经验记忆缓冲区"""

    def __init__(
        self,
        capacity: int,
        sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        balance_tasks: bool = True
    ):
        self.capacity = capacity
        self.sampling_strategy = sampling_strategy
        self.balance_tasks = balance_tasks
        self.buffer: List[Experience] = []
        self.task_indices: Dict[str, List[int]] = defaultdict(list)
        self.importance_scores: Optional[np.ndarray] = None

    def add(self, experience: Dict[str, Any], task_id: str) -> None:
        """添加经验"""
        exp = Experience(task_id=task_id, **experience)

        if len(self.buffer) < self.capacity:
            idx = len(self.buffer)
            self.buffer.append(exp)
        else:
            # Buffer满了，使用替换策略
            idx = self._select_replacement_index()
            old_task = self.buffer[idx].task_id
            self.task_indices[old_task].remove(idx)
            self.buffer[idx] = exp

        self.task_indices[task_id].append(idx)

    def sample(self, batch_size: int) -> List[Experience]:
        """采样经验"""
        if self.sampling_strategy == SamplingStrategy.UNIFORM:
            return self._uniform_sample(batch_size)
        elif self.sampling_strategy == SamplingStrategy.IMPORTANCE:
            return self._importance_sample(batch_size)
        elif self.sampling_strategy == SamplingStrategy.DIVERSE:
            return self._diverse_sample(batch_size)
        else:
            raise ValueError(f"Unknown strategy: {self.sampling_strategy}")

    def _uniform_sample(self, batch_size: int) -> List[Experience]:
        """均匀采样"""
        if self.balance_tasks:
            # 每个任务采样相同数量
            samples = []
            tasks = list(self.task_indices.keys())
            per_task = batch_size // len(tasks)
            for task_id in tasks:
                indices = random.sample(
                    self.task_indices[task_id],
                    min(per_task, len(self.task_indices[task_id]))
                )
                samples.extend([self.buffer[i] for i in indices])
            return samples
        else:
            indices = random.sample(range(len(self.buffer)), batch_size)
            return [self.buffer[i] for i in indices]

    def _importance_sample(self, batch_size: int) -> List[Experience]:
        """基于重要性采样"""
        if self.importance_scores is None:
            self._compute_importance_scores()

        probs = self.importance_scores / self.importance_scores.sum()
        indices = np.random.choice(
            len(self.buffer), size=batch_size, p=probs
        )
        return [self.buffer[i] for i in indices]

    def _diverse_sample(self, batch_size: int) -> List[Experience]:
        """多样性采样：确保采样覆盖不同类型的经验"""
        # 使用聚类或其他方法确保多样性
        pass

    def _compute_importance_scores(self) -> None:
        """计算经验重要性分数"""
        # 可以基于：
        # 1. TD-error
        # 2. 梯度范数
        # 3. 奖励大小
        # 4. 稀有度
        pass

    def _select_replacement_index(self) -> int:
        """选择要替换的经验索引"""
        if self.sampling_strategy == SamplingStrategy.RECENCY:
            # 替换最旧的
            return 0
        elif self.sampling_strategy == SamplingStrategy.IMPORTANCE:
            # 替换最不重要的
            return np.argmin(self.importance_scores)
        else:
            return random.randint(0, len(self.buffer) - 1)

    def get_task_statistics(self) -> Dict[str, int]:
        """获取各任务的经验数量统计"""
        return {
            task_id: len(indices)
            for task_id, indices in self.task_indices.items()
        }
```

#### 2.2.5 Evaluation & Metrics (评估与指标)

**职责**: 计算持续学习特定的评估指标

```python
# src/tau2/continual_learning/metrics/metrics.py

class ContinualLearningMetrics:
    """持续学习评估指标"""

    @staticmethod
    def compute_average_accuracy(
        performance_matrix: np.ndarray
    ) -> float:
        """
        平均准确率
        performance_matrix[i, j] = 学习完任务i后在任务j上的性能
        """
        n_tasks = performance_matrix.shape[0]
        return np.mean([
            performance_matrix[i, :i+1].mean()
            for i in range(n_tasks)
        ])

    @staticmethod
    def compute_forgetting(
        performance_matrix: np.ndarray
    ) -> float:
        """
        遗忘度量：平均每个任务的性能下降
        Forgetting = (1/N) * Σ max_over_time(acc_j) - final_acc_j
        """
        n_tasks = performance_matrix.shape[0]
        forgetting = 0.0

        for j in range(n_tasks - 1):  # 排除最后一个任务（无遗忘）
            max_acc = performance_matrix[:, j].max()
            final_acc = performance_matrix[-1, j]
            forgetting += (max_acc - final_acc)

        return forgetting / (n_tasks - 1) if n_tasks > 1 else 0.0

    @staticmethod
    def compute_forward_transfer(
        performance_matrix: np.ndarray,
        baseline_matrix: np.ndarray
    ) -> float:
        """
        正向迁移：学习第i个任务时，对未来任务j (j>i) 的影响
        FWT = (1/N) * Σ (acc_ij - baseline_j) for j > i
        baseline_matrix[j, j] = 在任务j上从零开始训练的性能
        """
        n_tasks = performance_matrix.shape[0]
        fwt = 0.0
        count = 0

        for i in range(n_tasks - 1):
            for j in range(i + 1, n_tasks):
                fwt += (performance_matrix[i, j] - baseline_matrix[j, j])
                count += 1

        return fwt / count if count > 0 else 0.0

    @staticmethod
    def compute_backward_transfer(
        performance_matrix: np.ndarray
    ) -> float:
        """
        后向迁移：学习新任务对已学任务的影响
        BWT = (1/N) * Σ (final_acc_j - acc_jj)
        """
        n_tasks = performance_matrix.shape[0]
        bwt = 0.0

        for j in range(n_tasks - 1):
            initial_acc = performance_matrix[j, j]  # 刚学完任务j时的性能
            final_acc = performance_matrix[-1, j]  # 最终在任务j上的性能
            bwt += (final_acc - initial_acc)

        return bwt / (n_tasks - 1) if n_tasks > 1 else 0.0

    @staticmethod
    def compute_intransigence(
        performance_matrix: np.ndarray,
        baseline_matrix: np.ndarray
    ) -> float:
        """
        顽固性：相比joint training的性能差距
        """
        # 需要single-task baseline和multi-task joint training baseline
        pass

    @staticmethod
    def compute_learning_curve_area(
        performance_over_time: List[float]
    ) -> float:
        """学习曲线下面积"""
        return np.trapz(performance_over_time)

class CLEvaluator:
    """持续学习评估器"""

    def __init__(self):
        self.performance_matrix = []  # [n_evaluations, n_tasks]
        self.task_order = []
        self.evaluation_timestamps = []

    def record_evaluation(
        self,
        eval_point: int,  # 学了几个任务后
        task_performances: Dict[str, float]  # {task_id: performance}
    ):
        """记录评估结果"""
        self.evaluation_timestamps.append(datetime.now())

        # 确保按照task_order记录
        perf_vector = [
            task_performances.get(task_id, 0.0)
            for task_id in self.task_order
        ]
        self.performance_matrix.append(perf_vector)

    def compute_all_metrics(
        self,
        baseline_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """计算所有CL指标"""
        matrix = np.array(self.performance_matrix)

        metrics = {
            "average_accuracy": ContinualLearningMetrics.compute_average_accuracy(matrix),
            "forgetting": ContinualLearningMetrics.compute_forgetting(matrix),
            "backward_transfer": ContinualLearningMetrics.compute_backward_transfer(matrix),
        }

        if baseline_matrix is not None:
            metrics["forward_transfer"] = ContinualLearningMetrics.compute_forward_transfer(
                matrix, baseline_matrix
            )

        return metrics
```

#### 2.2.6 Logging & Visualization (日志与可视化)

```python
# src/tau2/continual_learning/logging/logger.py

class CLLogger:
    """持续学习实验日志记录器"""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 初始化各种日志文件
        self.training_log = []
        self.evaluation_log = []
        self.memory_log = []

        # TensorBoard/WandB集成
        self.use_tensorboard = True
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.experiment_dir / "tensorboard")

    def log_training(
        self,
        task_id: str,
        step: int,
        metrics: Dict[str, Any]
    ):
        """记录训练指标"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "step": step,
            **metrics
        }
        self.training_log.append(entry)

        if self.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(f"train/{key}", value, step)

    def log_evaluation(
        self,
        eval_results: "EvaluationResults"
    ):
        """记录评估结果"""
        self.evaluation_log.append(eval_results.dict())

        # 保存性能矩阵可视化
        self._plot_performance_matrix(eval_results.performance_matrix)

    def _plot_performance_matrix(self, matrix: np.ndarray):
        """绘制性能矩阵热力图"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=[f"T{i}" for i in range(matrix.shape[1])],
            yticklabels=[f"After T{i}" for i in range(matrix.shape[0])]
        )
        plt.title("Task Performance Matrix")
        plt.xlabel("Evaluated Task")
        plt.ylabel("Training Progress")
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "performance_matrix.png")
        plt.close()

    def plot_learning_curves(self):
        """绘制学习曲线"""
        pass

    def plot_forgetting_curves(self):
        """绘制遗忘曲线"""
        pass

    def generate_report(self) -> str:
        """生成实验报告"""
        pass
```

---

## 第三部分：实现步骤

### 3.1 Phase 1: 基础架构搭建（2-3周）

#### 任务 1.1: 创建项目结构
```bash
src/tau2/continual_learning/
├── __init__.py
├── curriculum/
│   ├── __init__.py
│   ├── base.py
│   ├── sequential.py
│   ├── interleaved.py
│   ├── difficulty_based.py
│   └── adaptive.py
├── orchestrator.py
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── experience_replay.py
│   ├── ewc.py
│   ├── si.py
│   ├── agem.py
│   └── maml.py
├── memory/
│   ├── __init__.py
│   ├── buffer.py
│   └── sampling.py
├── metrics/
│   ├── __init__.py
│   └── metrics.py
├── logging/
│   ├── __init__.py
│   └── logger.py
└── utils/
    ├── __init__.py
    └── helpers.py
```

**关键代码文件**:
- `base.py`: 基类定义
- `orchestrator.py`: 主要的实验运行逻辑

**依赖安装**:
```bash
# 在 setup.py 或 pyproject.toml 中添加
dependencies = [
    "tau2-bench",  # 基础
    "torch>=2.0",  # 深度学习
    "numpy>=1.24",
    "scipy",
    "scikit-learn",  # 聚类、相似度计算
    "matplotlib",
    "seaborn",  # 可视化
    "tensorboard",  # 日志
    "wandb",  # 可选：实验追踪
    "transformers",  # LLM
    "peft",  # Parameter-Efficient Fine-Tuning
]
```

#### 任务 1.2: 扩展数据模型

修改 `src/tau2/data_model/tasks.py`:
```python
class TaskDifficulty(BaseModel):
    """任务难度评估"""
    num_tools: int
    num_steps: int
    complexity_score: float  # 0.0-1.0
    success_rate_baseline: Optional[float] = None

class Task(BaseModel):
    # ... 现有字段 ...

    # 新增字段
    difficulty: Optional[TaskDifficulty] = None
    semantic_tags: List[str] = []
    prerequisite_skills: List[str] = []
```

#### 任务 1.3: CLI集成

在 `src/tau2/cli/main.py` 中添加新命令:
```python
@click.group()
def cl():
    """Continual Learning commands"""
    pass

@cl.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--algorithm", type=click.Choice(["er", "ewc", "agem", "maml"]))
@click.option("--curriculum", type=click.Choice(["sequential", "interleaved", "difficulty"]))
def train(config, algorithm, curriculum):
    """Run continual learning training"""
    pass

@cl.command()
@click.option("--experiment-dir", type=click.Path(exists=True))
def evaluate(experiment_dir):
    """Evaluate a trained CL agent"""
    pass

@cl.command()
@click.option("--experiment-dir", type=click.Path(exists=True))
def visualize(experiment_dir):
    """Visualize CL experiment results"""
    pass
```

### 3.2 Phase 2: 核心算法实现（3-4周）

#### 任务 2.1: 实现 Experience Replay
- 完整的ER agent实现
- Memory buffer和采样策略
- 与Tau2-Bench Gym接口集成
- 单元测试

#### 任务 2.2: 实现 EWC
- Fisher信息矩阵计算
- 正则化loss实现
- 参数重要性跟踪
- 单元测试

#### 任务 2.3: 实现 A-GEM
- 梯度计算和投影
- 约束检查
- 单元测试

#### 任务 2.4: 实现基本的MAML
- Inner/outer loop
- 元参数管理
- 单元测试

### 3.3 Phase 3: Curriculum系统（1-2周）

#### 任务 3.1: 任务元数据计算
```python
# 实现自动计算任务难度的函数
def compute_task_difficulty(task: Task) -> TaskDifficulty:
    """基于任务特征计算难度"""
    num_tools = len(task.evaluation_criteria.actions) if task.evaluation_criteria else 0
    num_steps = task.initial_state.estimated_steps if task.initial_state else 10

    complexity_score = (
        0.4 * (num_tools / 10) +  # 工具数量
        0.3 * (num_steps / 20) +   # 步骤数
        0.3 * compute_semantic_complexity(task)  # 语义复杂度
    )

    return TaskDifficulty(
        num_tools=num_tools,
        num_steps=num_steps,
        complexity_score=min(complexity_score, 1.0)
    )
```

#### 任务 3.2: 实现各种课程策略
- Sequential
- Interleaved (domain mixing)
- Difficulty-based
- Adaptive (基于性能动态调整)

### 3.4 Phase 4: 评估与指标（1-2周）

#### 任务 4.1: 实现CL指标计算
- Average Accuracy
- Forgetting
- Forward/Backward Transfer
- Learning curves

#### 任务 4.2: 可视化系统
- 性能矩阵热力图
- 学习曲线图
- 遗忘曲线图
- 各任务性能对比

### 3.5 Phase 5: 实验与优化（2-3周）

#### 任务 5.1: Baseline实验
运行baseline实验建立性能基准:
```bash
# No CL (独立训练每个任务)
tau2 cl train --algorithm none --curriculum sequential

# ER baseline
tau2 cl train --algorithm er --curriculum sequential --buffer-size 1000

# EWC baseline
tau2 cl train --algorithm ewc --curriculum sequential --ewc-lambda 0.4
```

#### 任务 5.2: 超参数调优
- Buffer size
- Learning rates
- Replay ratios
- Regularization strengths

#### 任务 5.3: 性能分析
- 不同算法对比
- 不同curriculum效果
- Domain-specific分析

---

## 第四部分：技术细节

### 4.1 与现有Tau2-Bench的集成点

#### 4.1.1 利用Gym接口训练

```python
# 使用现有的AgentGymEnv
from tau2.gym.gym_agent import AgentGymEnv

env = AgentGymEnv(
    domain="airline",
    task_id="task_001",
    agent=cl_agent,  # 我们的CL agent
    user_llm="gpt-4.1",
    seed=42
)

# RL-style training loop
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    trajectory = []

    while not done:
        action = cl_agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": terminated or truncated
        })

        done = terminated or truncated
        obs = next_obs

    # CL算法从轨迹中学习
    cl_agent.learn_from_trajectory(trajectory, task)
```

#### 4.1.2 复用Evaluator

```python
from tau2.evaluator.evaluator import Evaluator

# 评估CL agent
evaluator = Evaluator()

simulation_run = orchestrator.run()  # 运行模拟
reward_info = evaluator.evaluate(
    task=task,
    simulation_run=simulation_run
)

# reward_info 包含各种评估指标
performance = reward_info.total_reward
```

#### 4.1.3 扩展现有Agent

```python
from tau2.agent.llm_agent import LLMAgent

class ContinualLearningLLMAgent(LLMAgent):
    """扩展LLM Agent支持持续学习"""

    def __init__(self, *args, cl_module: ContinualLearningModule, **kwargs):
        super().__init__(*args, **kwargs)
        self.cl_module = cl_module

    def generate_next_message(self, message, state):
        # 调用父类生成
        response, new_state = super().generate_next_message(message, state)

        # CL模块记录经验
        self.cl_module.record_experience(message, response, state)

        return response, new_state

    def learn_from_experience(self):
        """触发CL学习"""
        self.cl_module.update()
```

### 4.2 API Agent的持续学习实现方案

由于Agent是无参数的(仅调用API),我们不能使用需要梯度更新的传统持续学习方法。相反,我们采用以下适合API Agent的方法:

#### 方案 A: In-Context Learning with Experience Replay (推荐)

```python
# src/tau2/continual_learning/agents/icl_experience_replay.py

class ICLExperienceReplayAgent(ContinualLearningAgent):
    """使用ICL + 经验回放的无参数Agent"""

    def __init__(self, base_model: str = "gpt-4", ...):
        super().__init__(...)

        # API客户端
        self.client = OpenAI()  # 或 Anthropic, 等
        self.base_model = base_model

        # Example bank
        self.example_bank = []

    def select_action(self, observation: str) -> str:
        """通过API生成action"""

        # 1. 检索相关示例
        examples = self._retrieve_examples(observation, k=5)

        # 2. 构建few-shot prompt
        prompt = self._build_few_shot_prompt(
            examples=examples,
            query=observation,
            policy=self.domain_policy
        )

        # 3. 调用API
        response = self.client.chat.completions.create(
            model=self.base_model,
            messages=[
                {"role": "system", "content": self.domain_policy},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    def learn_from_trajectory(self, trajectory, task):
        """学习 = 更新example bank"""

        for step in trajectory:
            if step["reward"] > 0.5:  # 只保留好的示例
                # 计算embedding
                embedding = self._get_embedding(step["obs"])

                self.example_bank.append({
                    "obs": step["obs"],
                    "action": step["action"],
                    "reward": step["reward"],
                    "task_id": task.id,
                    "domain": task.domain,
                    "embedding": embedding
                })

        # 限制大小
        if len(self.example_bank) > 1000:
            self._prune_examples()

    def _retrieve_examples(self, query: str, k: int) -> List[Dict]:
        """基于相似度检索examples"""
        query_emb = self._get_embedding(query)

        # 计算相似度
        similarities = []
        for ex in self.example_bank:
            sim = self._cosine_sim(query_emb, ex["embedding"])
            similarities.append((sim, ex))

        # 返回top-k,同时保证diversity
        sorted_examples = sorted(similarities, key=lambda x: x[0], reverse=True)
        return self._diversify_selection([e for _, e in sorted_examples], k)

    def _diversify_selection(self, examples: List, k: int) -> List:
        """确保选择的examples来自不同domain/task"""
        selected = []
        seen_domains = set()
        seen_tasks = set()

        # 第一轮: 优先选择新domain/task的examples
        for ex in examples:
            if len(selected) >= k:
                break
            if ex["domain"] not in seen_domains or ex["task_id"] not in seen_tasks:
                selected.append(ex)
                seen_domains.add(ex["domain"])
                seen_tasks.add(ex["task_id"])

        # 第二轮: 填充剩余名额
        for ex in examples:
            if len(selected) >= k:
                break
            if ex not in selected:
                selected.append(ex)

        return selected[:k]
```

**关键优势**:
- ✅ 无需模型参数,适合API Agent
- ✅ 实现简单,易于调试
- ✅ 可以利用最新的大模型(GPT-4, Claude等)
- ✅ 通过example bank自然实现experience replay
- ✅ 灵活性高,可随时修改prompt策略

#### 方案 B: Prompt库管理

```python
class PromptLibraryAgent(ContinualLearningAgent):
    """维护prompt模板库"""

    def __init__(self, ...):
        super().__init__(...)

        # Prompt模板库
        self.prompts = {
            "airline": {
                "templates": [],
                "performance": []
            },
            "retail": {...},
            "telecom": {...}
        }

    def learn_from_trajectory(self, trajectory, task):
        """提取成功的prompt patterns"""

        # 如果任务成功,保存使用的prompt模式
        if trajectory_success(trajectory):
            prompt_pattern = extract_pattern(trajectory)
            self.prompts[task.domain]["templates"].append(prompt_pattern)

        # 评估各模板性能
        self._evaluate_templates(task.domain)

    def select_action(self, observation: str) -> str:
        """使用最佳prompt模板"""
        domain = self.current_task.domain
        best_template = self.prompts[domain]["templates"][0]  # 性能最好的

        prompt = best_template.format(observation=observation)
        return self._call_api(prompt)
```

#### 方案 C: 示例选择策略演化

```python
class AdaptiveExampleSelector(ContinualLearningAgent):
    """自适应的示例选择策略"""

    def __init__(self, ...):
        super().__init__(...)

        # 多种选择策略
        self.selection_strategies = {
            "similarity": self._select_by_similarity,
            "diversity": self._select_by_diversity,
            "recency": self._select_by_recency,
            "reward": self._select_by_reward,
            "task_specific": self._select_task_specific
        }

        # 跟踪每种策略的性能
        self.strategy_performance = {k: [] for k in self.selection_strategies}
        self.current_strategy = "similarity"

    def learn_from_trajectory(self, trajectory, task):
        """根据性能调整策略"""

        # 记录当前策略的性能
        reward = compute_trajectory_reward(trajectory)
        self.strategy_performance[self.current_strategy].append(reward)

        # 每N个任务,切换到最佳策略
        if task.id % 10 == 0:
            self.current_strategy = self._select_best_strategy()

    def _retrieve_examples(self, query: str, k: int) -> List:
        """使用当前最佳策略检索"""
        strategy_fn = self.selection_strategies[self.current_strategy]
        return strategy_fn(query, k)
```

### 4.3 配置文件规范

```yaml
# configs/cl_experiment_api_agent.yaml

experiment:
  name: "icl_er_airline_retail_telecom"
  seed: 42
  output_dir: "./experiments/icl_er"

curriculum:
  strategy: "sequential"
  domains: ["airline", "retail", "telecom"]
  task_splits: "train"
  num_tasks_per_domain: 20

agent:
  type: "icl_experience_replay"  # 新增: 无参数方法
  base_model: "gpt-4"  # API模型

  # ICL-ER配置
  icl_config:
    max_examples_in_prompt: 5  # few-shot examples数量
    example_bank_size: 1000  # example bank容量
    similarity_threshold: 0.7  # 相似度筛选阈值
    diversity_weight: 0.3  # 多样性权重
    embedding_model: "text-embedding-ada-002"  # 用于检索的embedding模型

    # 示例选择策略
    selection_strategy: "hybrid"  # similarity, diversity, recency, hybrid

    # 示例质量筛选
    min_reward_threshold: 0.5  # 只保留高质量示例
    filter_failed_tool_calls: true  # 过滤失败的工具调用

  # Prompt策略配置 (如果使用PSE)
  prompt_strategy_config:
    num_variants: 5
    variant_types: ["standard", "cot", "task_specific", "cross_domain", "recent"]
    exploration_rate: 0.1  # 探索新策略的概率

  # 记忆管理配置 (如果使用MAR)
  memory_config:
    max_episodic: 500
    max_semantic_patterns: 100
    consolidation_frequency: 50  # 每N个任务进行一次consolidation

training:
  episodes_per_task: 10
  max_steps_per_episode: 100

  # API配置
  api_timeout: 30  # 秒
  retry_on_failure: true
  max_retries: 3

evaluation:
  eval_frequency: 10
  eval_on_all_seen_tasks: true
  num_eval_trials: 3

logging:
  use_tensorboard: true
  save_example_bank: true  # 保存example bank快照
  log_prompts: true  # 记录使用的prompts
  log_level: "INFO"

user_simulator:
  llm: "gpt-4"
```

### 4.4 数据流图

```
┌─────────────┐
│ Task Pool   │
│ (Curriculum)│
└──────┬──────┘
       │
       ├──> Task 1 ──┐
       ├──> Task 2 ──┼──> ┌─────────────────┐
       ├──> Task 3 ──┘    │  CL Orchestrator │
       │                  └────────┬─────────┘
       ...                         │
                                   ▼
                     ┌─────────────────────────────┐
                     │  Training Episode (Gym Env) │
                     │  - Agent acts               │
                     │  - User responds            │
                     │  - Environment executes     │
                     └────────┬────────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   Trajectory     │
                     │   Collection     │
                     └────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
         ┌──────────────────┐  ┌──────────────┐
         │  Memory Buffer   │  │ CL Algorithm │
         │  (Replay)        │◄─┤  Learning    │
         └──────────────────┘  └──────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │  Updated Agent   │
                     └────────┬─────────┘
                              │
                    (back to next task)
```

---

## 第五部分：实验设计

### 5.1 实验场景

#### 实验 1: Domain Incremental Learning (领域增量学习)
- 按顺序学习 airline -> retail -> telecom
- 评估各domain的遗忘情况
- 对比方法：
  * **ICL-ER**: In-Context Learning + Experience Replay
  * **PSE**: Prompt Strategy Evolution
  * **MAR**: Memory-Augmented Retrieval
  * **Baseline**: 无持续学习(每次从零开始)

#### 实验 2: Task Incremental Learning (任务增量学习)
- 同一domain内，task逐个学习
- 评估任务间的迁移和遗忘
- 对比不同curriculum策略:
  * Sequential (顺序学习)
  * Interleaved (交错学习)
  * Difficulty-based (难度递进)

#### 实验 3: Example Bank Size Analysis (示例库大小分析)
- 测试不同example bank大小对性能的影响
- 实验配置: 50, 100, 200, 500, 1000 examples
- 评估指标: 性能 vs 存储成本 vs 检索时间

#### 实验 4: Example Selection Strategy (示例选择策略对比)
- 对比不同的示例选择策略:
  * Similarity-based (相似度)
  * Diversity-based (多样性)
  * Recency-based (新近度)
  * Reward-based (奖励)
  * Hybrid (混合策略)

#### 实验 5: Prompt Engineering Impact (Prompt工程影响)
- 测试不同prompt模板对持续学习的影响
- 标准few-shot vs Chain-of-Thought vs Task-specific
- 评估prompt复杂度与性能的关系

### 5.2 评估协议

```python
# 标准评估流程
class EvaluationProtocol:
    """标准化评估协议"""

    @staticmethod
    def run_full_evaluation(
        agent: ContinualLearningAgent,
        task_sequence: List[Task],
        num_trials: int = 3
    ) -> Dict[str, Any]:
        """
        完整评估流程:
        1. 在每个task上运行num_trials次
        2. 计算Pass@k
        3. 记录详细轨迹
        """
        results = defaultdict(list)

        for task in task_sequence:
            task_results = []
            for trial in range(num_trials):
                sim_run = run_single_trial(agent, task)
                reward = evaluate_simulation(sim_run)
                task_results.append({
                    "success": reward > 0.5,
                    "reward": reward,
                    "steps": len(sim_run.messages),
                    "trajectory": sim_run
                })

            results[task.id] = task_results

        # 计算aggregate metrics
        return compute_aggregate_metrics(results)
```

### 5.3 预期结果 (针对无参数API Agent)

基于In-Context Learning的持续学习方法预期结果:

1. **ICL-ER (In-Context Learning + Experience Replay)**:
   - **遗忘率**: ~5-10% (优于传统方法)
   - **优势**:
     * 通过检索和重放历史成功经验,有效保持旧知识
     * 不依赖参数更新,避免了catastrophic forgetting的主要原因
     * 可以灵活地调整example bank大小和检索策略
   - **挑战**:
     * 依赖embedding质量
     * Example bank过大时检索成本增加
     * Prompt长度限制(上下文窗口)

2. **PSE (Prompt Strategy Evolution)**:
   - **适应性**: 高
   - **优势**:
     * 自动发现不同domain/task的最优prompt策略
     * 无需人工调优prompt
     * 可以在学习过程中动态调整
   - **挑战**:
     * 需要足够的探索时间
     * 策略评估需要多次尝试
     * 可能在早期阶段性能较低

3. **MAR (Memory-Augmented Retrieval)**:
   - **迁移能力**: 最强
   - **优势**:
     * 结合episodic和semantic记忆
     * 可以提取和泛化工具使用模式
     * 支持跨domain的知识迁移
   - **挑战**:
     * 实现复杂度较高
     * Pattern提取可能不准确
     * 需要更多的存储空间

4. **性能对比预期**:
   ```
   指标              Baseline   ICL-ER    PSE      MAR
   ────────────────────────────────────────────────
   平均准确率         65%       78%      75%      80%
   遗忘率            30%        8%      12%       6%
   Forward Transfer   0%        15%      12%      20%
   实现复杂度        低         中       中       高
   计算成本          低         中       中       高
   ```

5. **Example Bank Size vs Performance**:
   - 50 examples: ~70% 性能
   - 100 examples: ~75% 性能
   - 500 examples: ~78% 性能 (推荐)
   - 1000 examples: ~79% 性能 (收益递减)
   - **结论**: 500左右是性价比最优点

6. **Curriculum Strategy Impact**:
   - **Sequential**: 基线,遗忘较多(~10%)
   - **Interleaved**: 遗忘最少(~5%),但学习速度稍慢
   - **Difficulty-based**: 学习效率最高,遗忘中等(~7%)
   - **推荐**: 对于无参数Agent,Interleaved效果最好

### 5.4 与传统方法的对比优势

| 特性 | 传统CL方法 (EWC, MAML等) | 无参数CL方法 (ICL-ER等) |
|------|------------------------|------------------------|
| 需要参数更新 | ✅ 是 | ❌ 否 |
| 需要GPU | ✅ 通常需要 | ❌ 不需要 |
| 实现复杂度 | 高 | 中 |
| 训练时间 | 长 | 短 |
| 灵活性 | 低 | 高 |
| 可解释性 | 低 | 高 (可以查看examples) |
| 成本 | 高 (GPU) | 中 (API调用) |
| 适用模型 | 开源模型 | 任何API模型 |
| Catastrophic Forgetting | 严重 | 轻微 |
| 冷启动性能 | 低 | 高 (可利用预训练能力) |

---

## 第六部分：技术栈总结

### 6.1 核心依赖

| 库/框架 | 版本 | 用途 |
|--------|------|------|
| Python | 3.10+ | 基础语言 |
| Tau2-Bench | latest | 基础评估框架 |
| PyTorch | 2.0+ | 深度学习框架 |
| Transformers | 4.30+ | LLM接口 |
| PEFT | 0.4+ | Parameter-efficient fine-tuning |
| Gymnasium | 0.28+ | RL接口 |
| NumPy | 1.24+ | 数值计算 |
| Scikit-learn | 1.3+ | ML工具 |
| Matplotlib | 3.7+ | 可视化 |
| Seaborn | 0.12+ | 高级可视化 |
| TensorBoard | 2.13+ | 训练日志 |
| WandB | 0.15+ | 实验管理（可选） |
| LiteLLM | latest | 多LLM接口 |

### 6.2 开发工具

- **Version Control**: Git
- **Testing**: pytest
- **Code Quality**: ruff, black (已在tau2-bench中使用)
- **Documentation**: Sphinx + Markdown
- **CI/CD**: GitHub Actions

### 6.3 硬件需求

- **最小配置**:
  - CPU: 4核+
  - RAM: 16GB
  - GPU: 可选 (使用API-based LLMs时)

- **推荐配置** (用于local LLM fine-tuning):
  - CPU: 8核+
  - RAM: 32GB+
  - GPU: NVIDIA GPU with 16GB+ VRAM (如RTX 4080, A100)

---

## 第七部分：挑战与解决方案

### 7.1 技术挑战 (针对无参数API Agent)

#### 挑战 1: Prompt长度限制
**问题**: API模型有上下文窗口限制,限制了可以包含的examples数量

**解决方案**:
1. 智能example选择 - 只选择最相关的examples
2. Example压缩 - 只保留关键信息
3. 分层检索 - 先粗筛选再精选择
4. 使用支持更长上下文的模型(如Claude 3 Sonnet 4.5: 200K tokens)

#### 挑战 2: API调用成本
**问题**: 频繁的API调用可能产生高成本

**解决方案**:
1. Caching机制 - 缓存相似query的结果
2. 批处理 - 合并多个请求
3. 使用较小/便宜的模型进行初始筛选
4. Cost tracking和预算管理

#### 挑战 3: Embedding质量和一致性
**问题**: 检索依赖embedding,但不同任务的embedding空间可能不一致

**解决方案**:
1. 使用domain-adapted embedding模型
2. 定期重新计算embeddings保持一致性
3. 结合多种相似度度量(embedding + keyword + structural)
4. 实现embedding的增量更新策略

#### 挑战 4: Example Bank管理
**问题**: Example bank会随时间增长,需要有效管理

**解决方案**:
1. 定期清理 - 移除低质量或过时的examples
2. 聚类去重 - 移除相似的冗余examples
3. 重要性评分 - 保留最有价值的examples
4. 分domain/task管理 - 结构化存储

#### 挑战 5: 冷启动问题
**问题**: 初始阶段example bank为空,性能可能不佳

**解决方案**:
1. 提供seed examples - 使用少量人工标注的高质量examples
2. 从相关任务迁移 - 如果有相关domain的历史数据
3. 使用zero-shot能力 - 充分利用LLM的预训练知识
4. 渐进式学习 - 从简单任务开始积累经验

### 7.2 设计权衡 (针对无参数方法)

#### 权衡 1: Example数量 vs 质量
- **更多examples**: 更好的覆盖,但增加噪声和成本
- **更少但高质量**: 清晰的信号,但可能覆盖不全
**决策**: 使用质量筛选(reward threshold) + 适中数量(500左右)

#### 权衡 2: 相似度 vs 多样性
- **高相似度**: 更相关,但可能过拟合
- **高多样性**: 更好的泛化,但可能不够相关
**决策**: Hybrid策略 - 先按相似度排序,再进行diversity filtering

#### 权衡 3: API成本 vs 性能
- **更多API调用**: 更好的性能,但成本高
- **减少调用**: 节省成本,但性能下降
**决策**:
  - 使用caching减少重复调用
  - 训练阶段可以多调用,推理阶段优化
  - 提供不同的cost-performance profiles (low/medium/high)

#### 权衡 4: 实时性 vs 准确性
- **快速检索**: 用户体验好,但可能不够准确
- **深度检索**: 更准确,但延迟高
**决策**:
  - 使用索引加速检索(如FAISS)
  - 异步更新example bank
  - 提供fast mode和accurate mode选项

---

## 第八部分：文档与教程

### 8.1 用户文档结构

```
docs/
├── README.md                          # 概览
├── getting_started.md                 # 快速开始
├── user_guide/
│   ├── installation.md
│   ├── basic_usage.md
│   ├── curriculum_design.md
│   ├── algorithm_selection.md
│   └── configuration.md
├── api_reference/
│   ├── curriculum.md
│   ├── agents.md
│   ├── memory.md
│   └── metrics.md
├── tutorials/
│   ├── 01_simple_er_experiment.md
│   ├── 02_curriculum_strategies.md
│   ├── 03_custom_algorithm.md
│   └── 04_advanced_evaluation.md
└── developer_guide/
    ├── architecture.md
    ├── contributing.md
    └── testing.md
```

### 8.2 示例教程

#### Tutorial 1: 简单的ER实验

```markdown
# Tutorial: Your First Continual Learning Experiment

## Goal
Train an agent to learn airline, retail, and telecom tasks sequentially
using Experience Replay, and evaluate forgetting.

## Steps

### 1. Install Tau2-CL
\`\`\`bash
pip install -e .
\`\`\`

### 2. Prepare config
\`\`\`yaml
# config.yaml
experiment:
  name: "my_first_cl_exp"
  seed: 42

curriculum:
  strategy: "sequential"
  domains: ["airline", "retail", "telecom"]

agent:
  type: "experience_replay"
  base_model: "gpt-4.1"
  er_config:
    buffer_size: 500
    replay_ratio: 0.5

training:
  episodes_per_task: 5

evaluation:
  eval_frequency: 10
  eval_on_all_seen_tasks: true
\`\`\`

### 3. Run experiment
\`\`\`bash
tau2 cl train --config config.yaml
\`\`\`

### 4. View results
\`\`\`bash
tau2 cl visualize --experiment-dir ./experiments/my_first_cl_exp
\`\`\`

## Expected Output
- Learning curves showing performance on each domain
- Forgetting metrics showing retention of airline skills after learning retail
- Performance matrix heatmap
\`\`\`
```

---

## 第九部分：测试策略

### 9.1 测试层级

```python
# tests/continual_learning/

# Unit tests
tests/unit/
├── test_curriculum.py          # Curriculum生成逻辑
├── test_memory_buffer.py       # Memory采样和存储
├── test_metrics.py             # 指标计算
└── test_agents.py              # 各算法的关键函数

# Integration tests
tests/integration/
├── test_orchestrator.py        # 端到端训练流程
├── test_evaluation.py          # 评估流程
└── test_gym_integration.py     # 与Tau2-Bench Gym接口

# End-to-end tests
tests/e2e/
└── test_full_experiment.py     # 完整实验运行（小规模）
```

### 9.2 关键测试用例

```python
# tests/unit/test_memory_buffer.py

def test_uniform_sampling():
    """测试均匀采样"""
    buffer = MemoryBuffer(capacity=100, sampling_strategy="uniform")

    # 添加不同任务的经验
    for i in range(50):
        buffer.add({"data": i}, task_id="task_1")
    for i in range(50):
        buffer.add({"data": i + 50}, task_id="task_2")

    # 采样
    samples = buffer.sample(20)

    # 验证: 均匀采样应该从两个任务中大致均等采样
    task_1_count = sum(1 for s in samples if s.task_id == "task_1")
    task_2_count = sum(1 for s in samples if s.task_id == "task_2")

    assert abs(task_1_count - task_2_count) <= 3  # 允许一些随机性

def test_buffer_overflow():
    """测试buffer满时的替换策略"""
    buffer = MemoryBuffer(capacity=10)

    for i in range(20):
        buffer.add({"data": i}, task_id="task_1")

    assert len(buffer.buffer) == 10
    # 验证替换策略正确执行
```

---

## 第十部分：项目管理

### 10.1 开发时间线

| 阶段 | 时间 | 主要产出 |
|------|------|---------|
| Phase 1: 基础架构 | 2-3周 | 项目结构、基类、CLI |
| Phase 2: 核心算法 | 3-4周 | ER, EWC, A-GEM, MAML实现 |
| Phase 3: Curriculum | 1-2周 | 各种curriculum策略 |
| Phase 4: 评估系统 | 1-2周 | 指标、可视化 |
| Phase 5: 实验优化 | 2-3周 | Baseline实验、调优 |
| Phase 6: 文档完善 | 1周 | 用户文档、教程 |
| **总计** | **10-15周** | 完整可用的Tau2-CL框架 |

### 10.2 里程碑

- **M1 (Week 3)**: 基础架构完成，可以运行简单的sequential training
- **M2 (Week 7)**: 至少2个CL算法实现并通过测试
- **M3 (Week 10)**: 完整的evaluation和visualization系统
- **M4 (Week 13)**: Baseline实验完成，有初步结果
- **M5 (Week 15)**: 文档完善，ready for release

### 10.3 风险管理

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| LLM Fine-tuning技术难度高 | 中 | 高 | 先实现ICL-based方案，fine-tuning作为进阶功能 |
| 计算资源不足 | 中 | 中 | 支持smaller models，使用cloud compute |
| 与Tau2-Bench集成问题 | 低 | 高 | 早期进行集成测试，与原作者沟通 |
| 实验时间过长 | 高 | 中 | 并行化，使用caching，smaller dev dataset |

---

## 第十一部分：未来扩展方向

### 11.1 短期扩展（6个月内）

1. **更多CL算法**
   - PackNet
   - Progressive Neural Networks
   - Learning without Forgetting (LwF)

2. **Multi-modal支持**
   - 支持vision+language tasks
   - 多模态tool使用

3. **Distributed training**
   - 支持多GPU训练
   - 分布式evaluation

### 11.2 中期扩展（6-12个月）

1. **Online CL**
   - 支持在线学习场景
   - 实时适应新任务

2. **Meta-continual learning**
   - 学习如何持续学习
   - 自动选择CL策略

3. **Real-world deployment**
   - Production-ready agent
   - A/B testing framework

### 11.3 长期愿景（1年+）

1. **Foundation Model for Tool Use**
   - 预训练的tool use foundation model
   - 支持zero-shot新工具

2. **Cross-domain CL**
   - 在完全不同的domains间迁移
   - Universal tool use agent

3. **Community Contributions**
   - 开放的algorithm library
   - User-contributed domains和tasks

---

## 附录

### A. 参考文献

1. **Continual Learning基础**
   - "Three scenarios for continual learning" (van de Ven & Tolias, 2019)
   - "Continual Lifelong Learning with Neural Networks: A Review" (Parisi et al., 2019)

2. **关键算法论文**
   - Experience Replay: "Gradient Episodic Memory for Continual Learning" (Lopez-Paz & Ranzato, 2017)
   - EWC: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)
   - A-GEM: "Efficient Lifelong Learning with A-GEM" (Chaudhry et al., 2019)
   - MAML: "Model-Agnostic Meta-Learning" (Finn et al., 2017)

3. **Tool Use相关**
   - Tau2-Bench paper
   - "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al., 2023)
   - "GPT-4 Technical Report" (OpenAI, 2023)

### B. Glossary

- **Catastrophic Forgetting**: 学习新任务时，旧任务性能急剧下降的现象
- **Forward Transfer**: 之前学习的知识帮助学习新任务的能力
- **Backward Transfer**: 学习新任务后对旧任务性能的影响
- **Curriculum Learning**: 按特定顺序组织训练任务以提升学习效果
- **Experience Replay**: 重放旧经验以防止遗忘的技术
- **Regularization**: 通过添加约束保护重要参数的方法

### C. 配置文件模板

完整的配置模板见 `configs/template.yaml`

---

## 总结

本文档详细描述了将 Tau2-Bench 扩展为**专门针对无参数API Agent**的持续学习评估框架的完整方案，包括：

1. **核心理念**: 明确了Tool Use场景下持续学习的挑战和目标，特别强调了无参数Agent的特殊性
2. **技术架构**: 设计了模块化、可扩展的系统架构，专注于In-Context Learning方法
3. **实现细节**: 提供了三种主要无参数持续学习算法的详细设计:
   - **ICL-ER**: In-Context Learning + Experience Replay (推荐)
   - **PSE**: Prompt Strategy Evolution
   - **MAR**: Memory-Augmented Retrieval
4. **实验方案**: 规划了系统性的评估实验，包括example bank大小分析、选择策略对比等
5. **开发计划**: 制定了清晰的实施步骤和时间线

### 关键特点

✅ **专为无参数Agent设计** - 不依赖模型参数更新，适合API调用场景
✅ **低遗忘率** - 通过智能example检索和重放，遗忘率可降至5-10%
✅ **实现简单** - 相比传统CL方法(EWC, MAML等)，实现和调试更容易
✅ **成本可控** - 通过caching和智能调用管理，平衡性能和成本
✅ **高灵活性** - 可以随时切换prompt策略和example选择方法
✅ **可解释性强** - 可以直观查看和分析使用的examples
✅ **模型通用性** - 适用于任何支持API的LLM (GPT-4, Claude, Gemini等)

### 与传统方法的核心区别

| 方面 | 传统CL (EWC/MAML/A-GEM) | 本方案 (ICL-based CL) |
|------|------------------------|---------------------|
| **学习方式** | 参数梯度更新 | Example bank + 检索 |
| **遗忘机制** | 参数覆盖导致灾难性遗忘 | Example丢失(可避免) |
| **防遗忘策略** | 正则化/约束/元学习 | Experience replay + 多样性 |
| **计算需求** | 需要GPU进行训练 | 仅需CPU进行检索 |
| **适用场景** | 开源可训练模型 | 任何API模型 |
| **主要成本** | GPU时间 | API调用费用 |

该框架将为研究者和开发者提供一个标准化的平台，用于测试和比较各种**无参数持续学习算法**在Tool Use任务上的效果，推动API-based Agent持续学习能力的发展。

### 下一步工作

1. **实现Phase 1**: 基础架构和ICL-ER算法
2. **Baseline实验**: 建立性能基准
3. **算法对比**: 评估ICL-ER, PSE, MAR的效果
4. **优化调优**: Example bank大小、检索策略等超参数
5. **文档完善**: 用户指南和API文档

---

**文档版本**: 2.0 (针对无参数Agent)
**最后更新**: 2025-12-03
**适用场景**: 无参数API Agent的持续学习
