# Copyright Sierra
"""
Meta-Continual Learning Agent

This agent implements meta-learning for continual learning by:
1. Learning meta-parameters that control the learning process
2. Optimizing for long-term performance and low forgetting
3. Adapting learning strategies based on task history
4. Using delayed rewards for meta-optimization
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from collections import deque

from tau2.environment.tool import Tool
from tau2.data_model.message import Message

from tau2.continual_learning.agents.parametric.base import (
    ParametricCLAgent,
    ParametricCLAgentState,
)
from tau2.continual_learning.agents.parametric.tool_scorer import ToolScorer
from tau2.continual_learning.agents.parametric.parametric_memory import ParametricMemory
from tau2.continual_learning.memory.buffer import Experience


AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

FEW_SHOT_INSTRUCTION = """
Here are some examples of successful interactions that may help you:
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>

{few_shot_section}

<policy>
{domain_policy}
</policy>
""".strip()


class MetaParameters:
    """
    Meta-parameters that control the learning process.

    These are learned to optimize long-term performance.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        meta_learning_rate: float = 0.001,
    ):
        """
        Initialize meta-parameters.

        Args:
            learning_rate: Base learning rate
            meta_learning_rate: Learning rate for meta-parameters
        """
        self.meta_learning_rate = meta_learning_rate

        # Meta-parameters (all learnable)
        self.base_learning_rate = learning_rate
        self.memory_write_threshold = 0.5  # Threshold for storing experiences
        self.memory_decay_rate = 0.001  # Decay rate for old memories
        self.retrieval_k = 5  # Number of memories to retrieve
        self.replay_ratio = 0.5  # Ratio for experience replay
        self.ewc_lambda = 1.0  # EWC regularization strength

        # Bounds for meta-parameters
        self.bounds = {
            "base_learning_rate": (0.001, 0.1),
            "memory_write_threshold": (0.0, 1.0),
            "memory_decay_rate": (0.0, 0.01),
            "retrieval_k": (1, 20),
            "replay_ratio": (0.0, 1.0),
            "ewc_lambda": (0.0, 10.0),
        }

        # Track gradient history for meta-optimization
        self.gradient_history: Dict[str, List[float]] = {
            param: [] for param in self.bounds.keys()
        }

        logger.info(f"Initialized MetaParameters with meta_lr={meta_learning_rate}")

    def get_parameters(self) -> Dict[str, float]:
        """Get all meta-parameters."""
        return {
            "base_learning_rate": self.base_learning_rate,
            "memory_write_threshold": self.memory_write_threshold,
            "memory_decay_rate": self.memory_decay_rate,
            "retrieval_k": int(self.retrieval_k),
            "replay_ratio": self.replay_ratio,
            "ewc_lambda": self.ewc_lambda,
        }

    def update_meta_parameter(
        self,
        param_name: str,
        gradient: float,
    ) -> Dict[str, Any]:
        """
        Update a meta-parameter using gradient.

        Args:
            param_name: Name of meta-parameter
            gradient: Gradient signal

        Returns:
            Update statistics
        """
        if param_name not in self.bounds:
            return {"updated": False, "reason": "unknown_parameter"}

        # Get current value
        old_value = getattr(self, param_name)

        # Update
        new_value = old_value + self.meta_learning_rate * gradient

        # Clip to bounds
        lower, upper = self.bounds[param_name]
        new_value = np.clip(new_value, lower, upper)

        # Set new value
        setattr(self, param_name, new_value)

        # Track gradient
        self.gradient_history[param_name].append(gradient)
        if len(self.gradient_history[param_name]) > 100:
            self.gradient_history[param_name] = self.gradient_history[param_name][-100:]

        logger.debug(
            f"Updated meta-parameter {param_name}: "
            f"{old_value:.4f} -> {new_value:.4f} (grad={gradient:.4f})"
        )

        return {
            "updated": True,
            "parameter": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "gradient": gradient,
        }

    def adapt_to_feedback(
        self,
        long_term_performance: float,
        forgetting_rate: float,
    ) -> Dict[str, Any]:
        """
        Adapt meta-parameters based on long-term feedback.

        Meta-loss:
            L_meta = -long_term_performance + λ * forgetting_rate

        Args:
            long_term_performance: Average performance over recent tasks
            forgetting_rate: Rate of forgetting on old tasks

        Returns:
            Adaptation statistics
        """
        # Compute meta-loss
        meta_loss = -long_term_performance + forgetting_rate

        # Heuristic gradients for each meta-parameter
        updates = {}

        # If forgetting is high, increase regularization
        if forgetting_rate > 0.3:
            updates["ewc_lambda"] = self.update_meta_parameter(
                "ewc_lambda",
                gradient=0.1 * forgetting_rate,
            )
            updates["replay_ratio"] = self.update_meta_parameter(
                "replay_ratio",
                gradient=0.1 * forgetting_rate,
            )

        # If performance is low, adjust learning rate
        if long_term_performance < 0.6:
            updates["base_learning_rate"] = self.update_meta_parameter(
                "base_learning_rate",
                gradient=0.01 * (0.6 - long_term_performance),
            )

        # Adaptive memory management
        if long_term_performance < 0.5:
            # Retrieve more memories
            updates["retrieval_k"] = self.update_meta_parameter(
                "retrieval_k",
                gradient=1.0,
            )
        elif long_term_performance > 0.8:
            # Can reduce retrieval
            updates["retrieval_k"] = self.update_meta_parameter(
                "retrieval_k",
                gradient=-0.5,
            )

        logger.info(
            f"Adapted meta-parameters: performance={long_term_performance:.3f}, "
            f"forgetting={forgetting_rate:.3f}, meta_loss={meta_loss:.3f}"
        )

        return {
            "meta_loss": meta_loss,
            "long_term_performance": long_term_performance,
            "forgetting_rate": forgetting_rate,
            "updates": updates,
        }


class MetaContinualLearningAgent(ParametricCLAgent):
    """
    Meta-Continual Learning Agent.

    Key features:
    1. Learns meta-parameters that control learning
    2. Optimizes for long-term performance, not just current task
    3. Adapts learning strategy based on forgetting signals
    4. Uses episode-level feedback for meta-optimization

    Meta-parameters:
    - learning_rate: Base learning rate
    - memory_write_threshold: When to store experiences
    - memory_decay_rate: How fast memories decay
    - retrieval_k: How many memories to retrieve
    - replay_ratio: How much to rely on replay
    - ewc_lambda: EWC regularization strength
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[Dict[str, Any]] = None,
        tool_scorer: Optional[ToolScorer] = None,
        parametric_memory: Optional[ParametricMemory] = None,
        embedding_dim: int = 768,
        learning_rate: float = 0.01,
        max_examples_in_prompt: int = 5,
        # Meta-learning specific
        meta_learning_rate: float = 0.001,
        performance_window: int = 10,
        enable_ewc: bool = True,
        enable_replay: bool = True,
        adapt_frequency: int = 5,
        cl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Meta-CL agent.

        Args:
            tools: Available tools
            domain_policy: Domain policy
            llm: LLM model name
            llm_args: LLM arguments
            tool_scorer: Tool scorer
            parametric_memory: Memory system
            embedding_dim: Embedding dimension
            learning_rate: Base learning rate
            max_examples_in_prompt: Max examples in prompt
            meta_learning_rate: Learning rate for meta-parameters
            performance_window: Window for computing long-term performance
            enable_ewc: Enable EWC regularization
            enable_replay: Enable experience replay
            adapt_frequency: How often to adapt meta-parameters
            cl_config: Additional config
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            tool_scorer=tool_scorer,
            parametric_memory=parametric_memory,
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            max_examples_in_prompt=max_examples_in_prompt,
            use_tool_scorer_for_selection=True,
            cl_config=cl_config,
        )

        # Meta-parameters
        self.meta_params = MetaParameters(
            learning_rate=learning_rate,
            meta_learning_rate=meta_learning_rate,
        )

        self.performance_window = performance_window
        self.enable_ewc = enable_ewc
        self.enable_replay = enable_replay
        self.adapt_frequency = adapt_frequency

        # Track performance history for meta-learning
        self.performance_history: deque = deque(maxlen=performance_window)
        self.forgetting_measurements: List[float] = []
        self.tasks_since_adapt = 0

        # Store baseline performance for each domain
        self.baseline_performance: Dict[str, float] = {}
        self.current_performance: Dict[str, float] = {}

        logger.info(
            f"Initialized MetaContinualLearningAgent with "
            f"meta_lr={meta_learning_rate}, "
            f"ewc={enable_ewc}, replay={enable_replay}"
        )

    def _update_parameters(
        self,
        state_embedding: np.ndarray,
        selected_tool: str,
        reward: float,
        success: bool,
    ) -> Dict[str, Any]:
        """
        Update parameters using current meta-parameters.

        Args:
            state_embedding: State embedding
            selected_tool: Selected tool
            reward: Reward
            success: Success flag

        Returns:
            Update statistics
        """
        # Use meta-learned learning rate
        old_lr = self.tool_scorer.learning_rate
        self.tool_scorer.learning_rate = self.meta_params.base_learning_rate

        # Standard update
        probs = self.tool_scorer.get_tool_probabilities(state_embedding)
        current_prob = probs.get(selected_tool, 0.0)
        advantage = reward - 0.5

        gradient = advantage * state_embedding * (1 - current_prob)

        # Apply EWC if enabled
        if self.enable_ewc and self.tool_scorer.fisher_information is not None:
            tool_idx = self.tool_scorer.tool_names.index(selected_tool)
            ewc_penalty = (
                self.meta_params.ewc_lambda *
                self.tool_scorer.fisher_information[tool_idx] *
                (self.tool_scorer.weights[tool_idx] - self.tool_scorer.optimal_weights[tool_idx])
            )
            gradient = gradient - ewc_penalty

        # Apply replay if enabled
        replay_grad = None
        if self.enable_replay and len(self.parametric_memory) > 0:
            # Retrieve memories
            k = int(self.meta_params.retrieval_k)
            replay_experiences = self.parametric_memory.sample_by_importance(
                n=k,
                domain=self._current_domain,
            )

            if replay_experiences:
                replay_grads = []
                for exp in replay_experiences:
                    if exp.embedding and exp.required_tools and exp.required_tools[0] == selected_tool:
                        exp_embedding = np.array(exp.embedding)
                        exp_probs = self.tool_scorer.get_tool_probabilities(exp_embedding)
                        exp_prob = exp_probs.get(selected_tool, 0.0)
                        exp_advantage = exp.reward - 0.5

                        importance = self.parametric_memory.get_importance(exp.experience_id)
                        grad = importance * exp_advantage * exp_embedding * (1 - exp_prob)
                        replay_grads.append(grad)

                if replay_grads:
                    replay_grad = np.mean(replay_grads, axis=0)

                    # Blend with current gradient
                    gradient = (
                        (1 - self.meta_params.replay_ratio) * gradient +
                        self.meta_params.replay_ratio * replay_grad
                    )

        # Update weights
        tool_idx = self.tool_scorer.tool_names.index(selected_tool)
        self.tool_scorer.weights[tool_idx] += self.tool_scorer.learning_rate * gradient

        # Restore learning rate
        self.tool_scorer.learning_rate = old_lr

        return {
            "updated": True,
            "tool": selected_tool,
            "reward": reward,
            "probability": current_prob,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "used_ewc": self.enable_ewc and self.tool_scorer.fisher_information is not None,
            "used_replay": replay_grad is not None,
            "meta_learning_rate": self.meta_params.base_learning_rate,
        }

    def _build_system_prompt_with_examples(
        self,
        examples: List[Experience]
    ) -> str:
        """Build system prompt with examples."""
        if not examples:
            few_shot_section = ""
        else:
            example_strs = []
            for i, exp in enumerate(examples, 1):
                obs_truncated = exp.observation[:500] + "..." if len(exp.observation) > 500 else exp.observation
                action_truncated = exp.action[:300] if len(exp.action) > 300 else exp.action

                example_str = f"""
<example_{i}>
Context: {obs_truncated}
Assistant action: {action_truncated}
</example_{i}>
""".strip()
                example_strs.append(example_str)

            few_shot_section = f"""
<examples>
{FEW_SHOT_INSTRUCTION}

{chr(10).join(example_strs)}
</examples>
""".strip()

        return SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION,
            few_shot_section=few_shot_section,
            domain_policy=self.domain_policy,
        )

    def learn_from_trajectory(
        self,
        task_id: str,
        domain: str,
        trajectory: List[Message],
        reward: float,
        success: bool = False,
    ) -> Dict[str, Any]:
        """
        Learn from trajectory with meta-learning.

        Args:
            task_id: Task ID
            domain: Domain name
            trajectory: Trajectory
            reward: Final reward
            success: Success flag

        Returns:
            Learning statistics including meta-learning info
        """
        # Standard learning
        learning_stats = super().learn_from_trajectory(
            task_id=task_id,
            domain=domain,
            trajectory=trajectory,
            reward=reward,
            success=success,
        )

        # Track performance
        self.performance_history.append(reward)
        self.current_performance[domain] = reward

        # Update baseline if first time seeing domain
        if domain not in self.baseline_performance:
            self.baseline_performance[domain] = reward

        # Compute forgetting
        forgetting = 0.0
        if len(self.baseline_performance) > 1:
            forgetting_per_domain = []
            for d, baseline in self.baseline_performance.items():
                if d in self.current_performance and d != domain:
                    forget = max(0, baseline - self.current_performance[d])
                    forgetting_per_domain.append(forget)

            if forgetting_per_domain:
                forgetting = np.mean(forgetting_per_domain)

        self.forgetting_measurements.append(forgetting)

        # Adapt meta-parameters periodically
        self.tasks_since_adapt += 1
        meta_stats = None

        if self.tasks_since_adapt >= self.adapt_frequency and len(self.performance_history) >= 3:
            long_term_perf = np.mean(list(self.performance_history))
            forgetting_rate = np.mean(self.forgetting_measurements[-5:]) if self.forgetting_measurements else 0.0

            meta_stats = self.meta_params.adapt_to_feedback(
                long_term_performance=long_term_perf,
                forgetting_rate=forgetting_rate,
            )

            self.tasks_since_adapt = 0

            # Apply meta-learned memory decay
            if self.meta_params.memory_decay_rate > 0:
                self.parametric_memory.importance_decay = self.meta_params.memory_decay_rate
                self.parametric_memory.apply_importance_decay()

        learning_stats["meta_learning"] = {
            "meta_parameters": self.meta_params.get_parameters(),
            "long_term_performance": np.mean(list(self.performance_history)) if self.performance_history else 0.0,
            "current_forgetting": forgetting,
            "adaptation": meta_stats,
        }

        return learning_stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics including meta-learning info."""
        stats = super().get_statistics()

        stats.update({
            "meta_parameters": self.meta_params.get_parameters(),
            "long_term_performance": np.mean(list(self.performance_history)) if self.performance_history else 0.0,
            "average_forgetting": np.mean(self.forgetting_measurements) if self.forgetting_measurements else 0.0,
            "baseline_performance": dict(self.baseline_performance),
            "current_performance": dict(self.current_performance),
        })

        return stats

    def save_state(self, path: str) -> None:
        """Save meta-CL agent state."""
        super().save_state(path)

        import json
        meta_path = path.replace('.json', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                "meta_parameters": self.meta_params.get_parameters(),
                "performance_history": list(self.performance_history),
                "forgetting_measurements": self.forgetting_measurements,
                "baseline_performance": self.baseline_performance,
                "current_performance": self.current_performance,
            }, f)

        logger.info(f"Saved meta-CL state to {meta_path}")


def create_meta_cl_agent(
    tools: List[Tool],
    domain_policy: str,
    llm: str = "gpt-4",
    embedding_dim: int = 768,
    learning_rate: float = 0.01,
    meta_learning_rate: float = 0.001,
    enable_ewc: bool = True,
    enable_replay: bool = True,
    **kwargs
) -> MetaContinualLearningAgent:
    """
    Factory function to create a Meta-CL agent.

    Args:
        tools: Available tools
        domain_policy: Domain policy
        llm: LLM model name
        embedding_dim: Embedding dimension
        learning_rate: Base learning rate
        meta_learning_rate: Meta-learning rate
        enable_ewc: Enable EWC
        enable_replay: Enable replay
        **kwargs: Additional arguments

    Returns:
        Configured MetaContinualLearningAgent
    """
    return MetaContinualLearningAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        meta_learning_rate=meta_learning_rate,
        enable_ewc=enable_ewc,
        enable_replay=enable_replay,
        **kwargs
    )
