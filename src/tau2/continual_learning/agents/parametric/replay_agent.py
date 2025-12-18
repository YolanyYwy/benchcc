# Copyright Sierra
"""
Replay-based Continual Learning Agent

This agent implements continual learning through experience replay with:
1. Parametric memory with learnable importance weights
2. Replay-based parameter updates (not just prompt engineering)
3. Gradient blending between current and replayed experiences
4. Dynamic replay scheduling
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


class ReplayContinualLearningAgent(ParametricCLAgent):
    """
    Replay-based Continual Learning Agent.

    Key differences from ICL-ER (In-Context Learning Experience Replay):
    1. ICL-ER: Only puts experiences in prompt, no parameter updates
    2. This: Actually updates parameters using replay experiences

    Process:
    1. For each new experience, update parameters
    2. Retrieve K relevant past experiences from parametric memory
    3. Update parameters using replayed experiences (gradient blending)
    4. Update memory importance weights based on utility

    This implements TRUE continual learning with:
    - Parametric tool selection (learnable w_i)
    - Parametric memory (learnable α_i)
    - Replay-based parameter updates (prevents forgetting)
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
        # Replay-specific parameters
        replay_ratio: float = 0.5,  # Ratio of replay vs current updates
        replay_batch_size: int = 5,  # Number of experiences to replay per update
        replay_strategy: str = "importance",  # "importance", "similarity", "mixed"
        update_memory_importance: bool = True,
        replay_frequency: int = 1,  # Replay every N steps
        use_replay_in_prompt: bool = True,  # Also include replay in prompt
        cl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Replay-based CL agent.

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
            replay_ratio: Weight for replay gradient vs current gradient
            replay_batch_size: Number of experiences to replay
            replay_strategy: Strategy for selecting replay experiences
            update_memory_importance: Whether to update memory importance weights
            replay_frequency: How often to perform replay (every N steps)
            use_replay_in_prompt: Whether to also include replay in prompt
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

        self.replay_ratio = replay_ratio
        self.replay_batch_size = replay_batch_size
        self.replay_strategy = replay_strategy
        self.update_memory_importance = update_memory_importance
        self.replay_frequency = replay_frequency
        self.use_replay_in_prompt = use_replay_in_prompt

        # Track replay statistics
        self._total_replay_updates = 0
        self._replay_update_history: List[Dict[str, Any]] = []
        self._steps_since_last_replay = 0

        logger.info(
            f"Initialized ReplayContinualLearningAgent with "
            f"replay_ratio={replay_ratio}, batch_size={replay_batch_size}, "
            f"strategy={replay_strategy}"
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

                # Include importance weight in prompt for transparency
                importance = self.parametric_memory.get_importance(exp.experience_id)

                example_str = f"""
<example_{i} importance="{importance:.2f}">
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

    def _retrieve_replay_experiences(
        self,
        current_state_embedding: np.ndarray,
        current_domain: Optional[str] = None,
    ) -> List[Experience]:
        """
        Retrieve experiences for replay.

        Args:
            current_state_embedding: Current state embedding
            current_domain: Current domain filter

        Returns:
            List of experiences to replay
        """
        if len(self.parametric_memory) == 0:
            return []

        if self.replay_strategy == "importance":
            # Sample by importance weights
            experiences = self.parametric_memory.sample_by_importance(
                n=self.replay_batch_size,
                domain=current_domain,
                temperature=1.0,
            )

        elif self.replay_strategy == "similarity":
            # Retrieve by similarity to current state
            # Convert embedding to text query (simplified)
            experiences = self.parametric_memory.retrieve_by_importance(
                k=self.replay_batch_size,
                domain=current_domain,
                similarity_weight=1.0,  # Pure similarity
            )

        elif self.replay_strategy == "mixed":
            # Blend importance and similarity
            experiences = self.parametric_memory.retrieve_by_importance(
                k=self.replay_batch_size,
                domain=current_domain,
                similarity_weight=0.5,  # 50-50 blend
            )

        else:
            # Fallback to uniform sampling
            experiences = self.parametric_memory.sample(
                n=self.replay_batch_size,
                domain=current_domain,
            )

        logger.debug(
            f"Retrieved {len(experiences)} replay experiences "
            f"using strategy={self.replay_strategy}"
        )

        return experiences

    def _update_parameters(
        self,
        state_embedding: np.ndarray,
        selected_tool: str,
        reward: float,
        success: bool,
    ) -> Dict[str, Any]:
        """
        Update parameters with replay.

        Process:
        1. Compute gradient from current experience
        2. Retrieve replay experiences
        3. Compute gradients from replay experiences
        4. Blend gradients: g_total = (1-α)*g_current + α*g_replay
        5. Update parameters with blended gradient

        Args:
            state_embedding: State embedding φ(s)
            selected_tool: Selected tool
            reward: Reward signal
            success: Whether succeeded

        Returns:
            Update statistics
        """
        if selected_tool not in self.tool_scorer.tool_names:
            return {"updated": False, "reason": "tool_not_found"}

        tool_idx = self.tool_scorer.tool_names.index(selected_tool)

        # 1. Compute current gradient
        probs = self.tool_scorer.get_tool_probabilities(state_embedding)
        current_prob = probs[selected_tool]
        advantage = reward - 0.5

        current_grad = advantage * state_embedding * (1 - current_prob)

        # 2. Replay updates (if enabled and time to replay)
        self._steps_since_last_replay += 1
        should_replay = (
            self._steps_since_last_replay >= self.replay_frequency and
            len(self.parametric_memory) > 0
        )

        replay_grad = None
        replay_experiences = []

        if should_replay:
            # Retrieve replay experiences
            replay_experiences = self._retrieve_replay_experiences(
                current_state_embedding=state_embedding,
                current_domain=self._current_domain,
            )

            if replay_experiences:
                # Compute replay gradients
                replay_grads = []
                for exp in replay_experiences:
                    if exp.embedding is None:
                        continue

                    exp_embedding = np.array(exp.embedding)
                    if exp.required_tools and exp.required_tools[0] == selected_tool:
                        # Replay experience used the same tool
                        exp_probs = self.tool_scorer.get_tool_probabilities(exp_embedding)
                        exp_prob = exp_probs.get(selected_tool, 0.0)
                        exp_advantage = exp.reward - 0.5

                        # Weight by memory importance
                        importance = self.parametric_memory.get_importance(exp.experience_id)

                        grad = importance * exp_advantage * exp_embedding * (1 - exp_prob)
                        replay_grads.append(grad)

                if replay_grads:
                    # Average replay gradients
                    replay_grad = np.mean(replay_grads, axis=0)

                    self._total_replay_updates += 1
                    self._steps_since_last_replay = 0

        # 3. Blend gradients
        if replay_grad is not None:
            # Blended gradient: (1-α)*g_current + α*g_replay
            blended_grad = (
                (1 - self.replay_ratio) * current_grad +
                self.replay_ratio * replay_grad
            )

            logger.debug(
                f"Replay update: current_grad_norm={np.linalg.norm(current_grad):.4f}, "
                f"replay_grad_norm={np.linalg.norm(replay_grad):.4f}, "
                f"blended_norm={np.linalg.norm(blended_grad):.4f}"
            )
        else:
            blended_grad = current_grad

        # 4. Update weights
        self.tool_scorer.weights[tool_idx] += self.learning_rate * blended_grad

        # 5. Update memory importance weights based on replay utility
        if self.update_memory_importance and replay_experiences:
            for exp in replay_experiences:
                # If replay helped (positive reward), increase importance
                # If replay didn't help, decrease importance
                gradient = reward - 0.5
                self.parametric_memory.update_importance(exp.experience_id, gradient)

        # Track statistics
        update_stats = {
            "updated": True,
            "tool": selected_tool,
            "reward": reward,
            "probability": current_prob,
            "gradient_norm": float(np.linalg.norm(current_grad)),
            "used_replay": replay_grad is not None,
            "num_replay_experiences": len(replay_experiences),
        }

        if replay_grad is not None:
            update_stats["replay_gradient_norm"] = float(np.linalg.norm(replay_grad))
            update_stats["blended_gradient_norm"] = float(np.linalg.norm(blended_grad))

        return update_stats

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None
    ) -> ParametricCLAgentState:
        """Get initial state with replay examples in prompt."""
        if message_history is None:
            message_history = []

        valid_messages = [
            msg for msg in message_history
            if is_valid_agent_history_message(msg)
        ]

        # Extract state embedding
        state_embedding = self._extract_state_embedding(valid_messages)

        # Retrieve examples for prompt
        if self.use_replay_in_prompt:
            examples = self.parametric_memory.retrieve_by_importance(
                k=self.max_examples_in_prompt,
                domain=self._current_domain,
                similarity_weight=0.3,  # Blend importance and similarity
            )
        else:
            examples = []

        # Build system prompt
        system_prompt = self._build_system_prompt_with_examples(examples)

        return ParametricCLAgentState(
            system_messages=[SystemMessage(role="system", content=system_prompt)],
            messages=valid_messages,
            few_shot_examples=[exp.model_dump() for exp in examples],
            current_task_id=self._current_task_id,
            current_domain=self._current_domain,
            state_embeddings=[state_embedding],
            selected_tools=[],
            rewards=[],
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
        Learn from trajectory with replay-based updates.

        Process:
        1. Extract experiences from trajectory
        2. For each experience:
           a. Update parameters (with replay)
           b. Add to parametric memory
        3. Update memory importance based on overall success

        Args:
            task_id: Task ID
            domain: Domain name
            trajectory: Message trajectory
            reward: Final reward
            success: Whether succeeded

        Returns:
            Learning statistics
        """
        logger.info(
            f"Learning from trajectory with replay: task={task_id}, "
            f"reward={reward:.3f}, success={success}"
        )

        # Use parent's learning which calls _update_parameters with replay
        learning_stats = super().learn_from_trajectory(
            task_id=task_id,
            domain=domain,
            trajectory=trajectory,
            reward=reward,
            success=success,
        )

        # Additional replay-specific statistics
        learning_stats["total_replay_updates"] = self._total_replay_updates
        learning_stats["replay_ratio"] = self.replay_ratio
        learning_stats["replay_strategy"] = self.replay_strategy

        return learning_stats

    def consolidate(self) -> None:
        """
        Consolidate memory by:
        1. Applying importance decay to old memories
        2. Pruning low-importance memories
        3. Rebalancing memory across domains
        """
        super().consolidate()

        # Apply time-based importance decay
        if self.parametric_memory.importance_decay > 0:
            decay_stats = self.parametric_memory.apply_importance_decay()
            logger.info(f"Memory importance decay: {decay_stats}")

        # Get memory statistics
        mem_stats = self.parametric_memory.get_statistics()
        logger.info(
            f"Memory consolidation: {mem_stats['total_experiences']} experiences, "
            f"importance range: [{mem_stats['importance_min']:.3f}, "
            f"{mem_stats['importance_max']:.3f}], "
            f"mean: {mem_stats['importance_mean']:.3f}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get replay-specific statistics."""
        stats = super().get_statistics()
        stats.update({
            "total_replay_updates": self._total_replay_updates,
            "replay_ratio": self.replay_ratio,
            "replay_batch_size": self.replay_batch_size,
            "replay_strategy": self.replay_strategy,
            "steps_since_last_replay": self._steps_since_last_replay,
        })

        return stats

    def save_state(self, path: str) -> None:
        """Save replay agent state."""
        super().save_state(path)

        # Save replay-specific state
        import json
        replay_path = path.replace('.json', '_replay.json')
        with open(replay_path, 'w') as f:
            json.dump({
                "total_replay_updates": self._total_replay_updates,
                "replay_ratio": self.replay_ratio,
                "replay_batch_size": self.replay_batch_size,
                "replay_strategy": self.replay_strategy,
            }, f)

        logger.info(f"Saved replay state to {replay_path}")

    def load_state(self, path: str) -> None:
        """Load replay agent state."""
        super().load_state(path)

        # Load replay-specific state
        import json
        replay_path = path.replace('.json', '_replay.json')
        try:
            with open(replay_path, 'r') as f:
                data = json.load(f)

            self._total_replay_updates = data["total_replay_updates"]
            self.replay_ratio = data.get("replay_ratio", self.replay_ratio)
            self.replay_batch_size = data.get("replay_batch_size", self.replay_batch_size)
            self.replay_strategy = data.get("replay_strategy", self.replay_strategy)

            logger.info(f"Loaded replay state from {replay_path}")
        except FileNotFoundError:
            logger.warning(f"Replay state file not found: {replay_path}")


def create_replay_agent(
    tools: List[Tool],
    domain_policy: str,
    llm: str = "gpt-4",
    embedding_dim: int = 768,
    learning_rate: float = 0.01,
    replay_ratio: float = 0.5,
    replay_batch_size: int = 5,
    replay_strategy: str = "importance",
    **kwargs
) -> ReplayContinualLearningAgent:
    """
    Factory function to create a Replay agent.

    Args:
        tools: Available tools
        domain_policy: Domain policy
        llm: LLM model name
        embedding_dim: Embedding dimension
        learning_rate: Learning rate
        replay_ratio: Weight for replay gradient vs current gradient
        replay_batch_size: Number of experiences to replay
        replay_strategy: Strategy for selecting replay experiences
        **kwargs: Additional arguments

    Returns:
        Configured ReplayContinualLearningAgent
    """
    return ReplayContinualLearningAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        replay_ratio=replay_ratio,
        replay_batch_size=replay_batch_size,
        replay_strategy=replay_strategy,
        **kwargs
    )
