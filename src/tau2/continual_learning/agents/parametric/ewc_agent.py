# Copyright Sierra
"""
EWC (Elastic Weight Consolidation) Continual Learning Agent

This agent implements EWC to prevent catastrophic forgetting by:
1. Computing Fisher Information Matrix after each task
2. Adding regularization penalty to protect important parameters
3. Gradually consolidating knowledge from multiple tasks
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

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


class EWCContinualLearningAgent(ParametricCLAgent):
    """
    EWC-based Continual Learning Agent.

    Key features:
    1. Computes Fisher Information Matrix F_i after each task
    2. Protects important parameters using EWC regularization:
       L = L_task + (λ/2) * Σ_i F_i * (θ_i - θ_i*)^2
    3. Supports multi-task consolidation with cumulative Fisher
    4. Gradually increases regularization strength as more tasks are learned

    This implements TRUE continual learning with parameter updates,
    not just prompt engineering like ICL-ER.
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
        # EWC-specific parameters
        ewc_lambda: float = 1.0,
        ewc_lambda_growth: str = "constant",  # "constant", "linear", "adaptive"
        fisher_sample_size: int = 100,
        online_ewc: bool = False,
        importance_threshold: float = 0.01,
        cl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize EWC agent.

        Args:
            tools: Available tools
            domain_policy: Domain policy
            llm: LLM model name
            llm_args: LLM arguments
            tool_scorer: Tool scorer (created if None)
            parametric_memory: Memory system (created if None)
            embedding_dim: Embedding dimension
            learning_rate: Learning rate
            max_examples_in_prompt: Max examples in prompt
            ewc_lambda: EWC regularization strength
            ewc_lambda_growth: How λ grows over tasks ("constant", "linear", "adaptive")
            fisher_sample_size: Number of samples for Fisher computation
            online_ewc: Whether to use online EWC (accumulate Fisher across tasks)
            importance_threshold: Threshold for considering a parameter important
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

        self.ewc_lambda = ewc_lambda
        self.ewc_lambda_growth = ewc_lambda_growth
        self.fisher_sample_size = fisher_sample_size
        self.online_ewc = online_ewc
        self.importance_threshold = importance_threshold

        # Track task history for EWC
        self._task_fisher_history: List[Dict[str, Any]] = []
        self._cumulative_fisher = None
        self._num_tasks_learned = 0
        self._current_lambda = ewc_lambda

        logger.info(
            f"Initialized EWCContinualLearningAgent with λ={ewc_lambda}, "
            f"online={online_ewc}, fisher_samples={fisher_sample_size}"
        )

    def _build_system_prompt_with_examples(
        self,
        examples: List[Experience]
    ) -> str:
        """Build system prompt with few-shot examples."""
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

    def _update_parameters(
        self,
        state_embedding: np.ndarray,
        selected_tool: str,
        reward: float,
        success: bool,
    ) -> Dict[str, Any]:
        """
        Update parameters with EWC regularization.

        Standard update:
            θ_i ← θ_i + α * gradient

        With EWC:
            θ_i ← θ_i + α * (gradient - λ * F_i * (θ_i - θ_i*))

        Args:
            state_embedding: State embedding φ(s)
            selected_tool: Selected tool
            reward: Reward signal
            success: Whether succeeded

        Returns:
            Update statistics
        """
        # Compute standard gradient
        probs = self.tool_scorer.get_tool_probabilities(state_embedding)
        current_prob = probs.get(selected_tool, 0.0)

        advantage = reward - 0.5  # Center around 0.5
        standard_grad = advantage * state_embedding * (1 - current_prob)

        # Apply EWC penalty if we have Fisher information
        tool_idx = self.tool_scorer.tool_names.index(selected_tool) if selected_tool in self.tool_scorer.tool_names else None

        if tool_idx is not None:
            if self.tool_scorer.fisher_information is not None:
                # Apply EWC penalty
                ewc_grad = self.tool_scorer.apply_ewc_penalty(
                    tool_idx=tool_idx,
                    gradient=standard_grad,
                    ewc_lambda=self._current_lambda,
                )

                # Update weights with EWC-penalized gradient
                self.tool_scorer.weights[tool_idx] += self.learning_rate * ewc_grad

                ewc_loss = self.tool_scorer.get_ewc_regularization_loss(self._current_lambda)

                logger.debug(
                    f"EWC update for {selected_tool}: reward={reward:.3f}, "
                    f"ewc_loss={ewc_loss:.4f}, λ={self._current_lambda:.3f}"
                )

                return {
                    "updated": True,
                    "tool": selected_tool,
                    "reward": reward,
                    "probability": current_prob,
                    "gradient_norm": float(np.linalg.norm(standard_grad)),
                    "ewc_penalty_norm": float(np.linalg.norm(standard_grad - ewc_grad)),
                    "ewc_loss": ewc_loss,
                    "lambda": self._current_lambda,
                }
            else:
                # No Fisher yet, standard update
                self.tool_scorer.weights[tool_idx] += self.learning_rate * standard_grad

                return {
                    "updated": True,
                    "tool": selected_tool,
                    "reward": reward,
                    "probability": current_prob,
                    "gradient_norm": float(np.linalg.norm(standard_grad)),
                    "ewc_applied": False,
                }

        return {"updated": False, "reason": "tool_not_found"}

    def _compute_fisher_for_task(
        self,
        task_id: str,
        domain: str,
    ) -> Dict[str, Any]:
        """
        Compute Fisher Information Matrix for the current task.

        Fisher is computed from recent experiences in this task:
            F_i = E[(∂log π(a|s) / ∂θ_i)^2]

        Args:
            task_id: Task ID
            domain: Domain name

        Returns:
            Fisher computation statistics
        """
        # Sample experiences from this task
        task_experiences = [
            exp for exp in self.parametric_memory._experiences
            if exp.task_id == task_id
        ]

        if not task_experiences:
            logger.warning(f"No experiences found for task {task_id}")
            return {"computed": False, "reason": "no_experiences"}

        # Limit sample size
        if len(task_experiences) > self.fisher_sample_size:
            import random
            task_experiences = random.sample(task_experiences, self.fisher_sample_size)

        # Extract state embeddings and tools
        state_embeddings = []
        selected_tools = []

        for exp in task_experiences:
            # Get embedding from experience
            if exp.embedding is not None:
                state_embeddings.append(np.array(exp.embedding))
            else:
                # Compute embedding from observation
                emb = self._extract_state_embedding([])  # Simplified
                state_embeddings.append(emb)

            # Get tool from experience
            if exp.required_tools:
                selected_tools.append(exp.required_tools[0])

        if not state_embeddings:
            logger.warning(f"No valid embeddings for task {task_id}")
            return {"computed": False, "reason": "no_embeddings"}

        # Compute Fisher Information Matrix
        fisher = self.tool_scorer.compute_fisher_information(
            state_embeddings=state_embeddings,
            selected_tools=selected_tools,
        )

        # Store task-specific Fisher
        task_fisher = {
            "task_id": task_id,
            "domain": domain,
            "fisher": fisher.copy(),
            "optimal_weights": self.tool_scorer.optimal_weights.copy(),
            "num_samples": len(state_embeddings),
        }
        self._task_fisher_history.append(task_fisher)

        # Update cumulative Fisher for online EWC
        if self.online_ewc:
            if self._cumulative_fisher is None:
                self._cumulative_fisher = fisher.copy()
            else:
                # Accumulate Fisher: F_cumulative = (n*F_old + F_new) / (n+1)
                n = self._num_tasks_learned
                self._cumulative_fisher = (n * self._cumulative_fisher + fisher) / (n + 1)

            # Update tool scorer's Fisher with cumulative
            self.tool_scorer.fisher_information = self._cumulative_fisher

        self._num_tasks_learned += 1

        # Update lambda based on growth strategy
        self._update_lambda()

        logger.info(
            f"Computed Fisher for task {task_id}: "
            f"samples={len(state_embeddings)}, "
            f"mean_fisher={fisher.mean():.6f}, "
            f"online_ewc={self.online_ewc}, "
            f"λ={self._current_lambda:.3f}"
        )

        return {
            "computed": True,
            "task_id": task_id,
            "domain": domain,
            "num_samples": len(state_embeddings),
            "mean_fisher": float(fisher.mean()),
            "max_fisher": float(fisher.max()),
            "num_important_params": int(np.sum(fisher > self.importance_threshold)),
            "lambda": self._current_lambda,
        }

    def _update_lambda(self) -> None:
        """Update EWC lambda based on growth strategy."""
        if self.ewc_lambda_growth == "constant":
            self._current_lambda = self.ewc_lambda

        elif self.ewc_lambda_growth == "linear":
            # Linear growth: λ = λ_0 * (1 + α * num_tasks)
            growth_rate = 0.1
            self._current_lambda = self.ewc_lambda * (1 + growth_rate * self._num_tasks_learned)

        elif self.ewc_lambda_growth == "adaptive":
            # Adaptive: increase λ if forgetting is detected
            # For now, use square root growth
            self._current_lambda = self.ewc_lambda * np.sqrt(1 + self._num_tasks_learned)

        logger.debug(f"Updated λ: {self._current_lambda:.3f} (strategy={self.ewc_lambda_growth})")

    def learn_from_trajectory(
        self,
        task_id: str,
        domain: str,
        trajectory: List[Message],
        reward: float,
        success: bool = False,
    ) -> Dict[str, Any]:
        """
        Learn from trajectory with EWC.

        Process:
        1. Update parameters with EWC regularization
        2. Store experiences in memory
        3. Compute Fisher Information Matrix for this task
        4. Update cumulative Fisher if online EWC

        Args:
            task_id: Task ID
            domain: Domain name
            trajectory: Message trajectory
            reward: Final reward
            success: Whether succeeded

        Returns:
            Learning statistics
        """
        # Standard parameter updates
        learning_stats = super().learn_from_trajectory(
            task_id=task_id,
            domain=domain,
            trajectory=trajectory,
            reward=reward,
            success=success,
        )

        # Compute Fisher Information for this task
        fisher_stats = self._compute_fisher_for_task(task_id, domain)

        learning_stats["fisher_computation"] = fisher_stats
        learning_stats["num_tasks_learned"] = self._num_tasks_learned
        learning_stats["current_lambda"] = self._current_lambda

        return learning_stats

    def consolidate(self) -> None:
        """
        Consolidate knowledge by updating Fisher matrices.

        This can be called periodically to:
        1. Recompute Fisher from recent experiences
        2. Prune unimportant parameters
        3. Adjust lambda based on forgetting
        """
        super().consolidate()

        if self.online_ewc and len(self._task_fisher_history) > 1:
            logger.info(
                f"Consolidating knowledge: {len(self._task_fisher_history)} tasks learned, "
                f"mean cumulative Fisher: {self._cumulative_fisher.mean():.6f}"
            )

            # Could implement Fisher pruning here
            # Remove parameters with Fisher < threshold
            if self._cumulative_fisher is not None:
                low_importance = self._cumulative_fisher < self.importance_threshold
                num_pruned = int(np.sum(low_importance))
                logger.info(f"Parameters below importance threshold: {num_pruned}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get EWC-specific statistics."""
        stats = super().get_statistics()
        stats.update({
            "num_tasks_learned": self._num_tasks_learned,
            "current_lambda": self._current_lambda,
            "fisher_history_length": len(self._task_fisher_history),
            "online_ewc": self.online_ewc,
            "has_cumulative_fisher": self._cumulative_fisher is not None,
        })

        if self._cumulative_fisher is not None:
            stats["cumulative_fisher_stats"] = {
                "mean": float(self._cumulative_fisher.mean()),
                "std": float(self._cumulative_fisher.std()),
                "max": float(self._cumulative_fisher.max()),
                "num_important": int(np.sum(self._cumulative_fisher > self.importance_threshold)),
            }

        return stats

    def save_state(self, path: str) -> None:
        """Save EWC agent state including Fisher matrices."""
        super().save_state(path)

        # Save EWC-specific state
        import pickle
        ewc_path = path.replace('.json', '_ewc.pkl')
        with open(ewc_path, 'wb') as f:
            pickle.dump({
                "task_fisher_history": self._task_fisher_history,
                "cumulative_fisher": self._cumulative_fisher,
                "num_tasks_learned": self._num_tasks_learned,
                "current_lambda": self._current_lambda,
            }, f)

        logger.info(f"Saved EWC state to {ewc_path}")

    def load_state(self, path: str) -> None:
        """Load EWC agent state including Fisher matrices."""
        super().load_state(path)

        # Load EWC-specific state
        import pickle
        ewc_path = path.replace('.json', '_ewc.pkl')
        try:
            with open(ewc_path, 'rb') as f:
                data = pickle.load(f)

            self._task_fisher_history = data["task_fisher_history"]
            self._cumulative_fisher = data["cumulative_fisher"]
            self._num_tasks_learned = data["num_tasks_learned"]
            self._current_lambda = data["current_lambda"]

            logger.info(f"Loaded EWC state from {ewc_path}")
        except FileNotFoundError:
            logger.warning(f"EWC state file not found: {ewc_path}")


def create_ewc_agent(
    tools: List[Tool],
    domain_policy: str,
    llm: str = "gpt-4",
    embedding_dim: int = 768,
    learning_rate: float = 0.01,
    ewc_lambda: float = 1.0,
    online_ewc: bool = True,
    **kwargs
) -> EWCContinualLearningAgent:
    """
    Factory function to create an EWC agent.

    Args:
        tools: Available tools
        domain_policy: Domain policy
        llm: LLM model name
        embedding_dim: Embedding dimension
        learning_rate: Learning rate
        ewc_lambda: EWC regularization strength
        online_ewc: Whether to use online EWC
        **kwargs: Additional arguments

    Returns:
        Configured EWCContinualLearningAgent
    """
    return EWCContinualLearningAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        ewc_lambda=ewc_lambda,
        online_ewc=online_ewc,
        **kwargs
    )
