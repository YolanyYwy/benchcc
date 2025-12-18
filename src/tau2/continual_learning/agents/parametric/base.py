# Copyright Sierra
"""
Parametric Continual Learning Agent Base Classes

This module provides the base classes for continual learning agents with
learnable parameters at the Agent and Memory layers, while keeping the LLM frozen.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from loguru import logger

from tau2.agent.base import ValidAgentInputMessage, is_valid_agent_history_message
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

from tau2.continual_learning.agents.base import ContinualLearningAgent, CLAgentState
from tau2.continual_learning.agents.parametric.tool_scorer import ToolScorer
from tau2.continual_learning.agents.parametric.parametric_memory import ParametricMemory


class ParametricCLAgentState(CLAgentState):
    """State for parametric continual learning agents"""

    # Store state embeddings for parameter updates
    state_embeddings: List[np.ndarray] = Field(default_factory=list)
    selected_tools: List[str] = Field(default_factory=list)
    rewards: List[float] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class ParametricCLAgent(ContinualLearningAgent, ABC):
    """
    Base class for parametric continual learning agents.

    Key differences from non-parametric CL agents:
    1. Tool selection uses a learnable ToolScorer with parameters w_i
    2. Memory has learnable importance weights α_i
    3. Parameters are updated through gradient-based learning
    4. Supports EWC and other regularization methods to prevent forgetting

    Architecture:
        Agent
        ├── LLM (frozen) - only for language generation and state embedding
        ├── Tool Scorer (parametric) - learnable tool selection
        ├── Memory System (parametric) - learnable importance weights
        └── Update Rule (method-specific) - how parameters are updated
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[Dict[str, Any]] = None,
        # Parametric components
        tool_scorer: Optional[ToolScorer] = None,
        parametric_memory: Optional[ParametricMemory] = None,
        # Configuration
        embedding_dim: int = 768,
        learning_rate: float = 0.01,
        max_examples_in_prompt: int = 5,
        use_tool_scorer_for_selection: bool = True,
        cl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize parametric CL agent.

        Args:
            tools: Available tools
            domain_policy: Domain policy
            llm: LLM model name
            llm_args: LLM arguments
            tool_scorer: Learnable tool scorer (created if None)
            parametric_memory: Learnable memory system (created if None)
            embedding_dim: Dimension of state embeddings
            learning_rate: Learning rate for parameter updates
            max_examples_in_prompt: Maximum examples in prompt
            use_tool_scorer_for_selection: Whether to use scorer for tool selection
            cl_config: Additional CL configuration
        """
        # Initialize tool scorer
        if tool_scorer is None:
            tool_scorer = ToolScorer(
                tools=tools,
                embedding_dim=embedding_dim,
                learning_rate=learning_rate,
            )

        # Initialize parametric memory
        if parametric_memory is None:
            parametric_memory = ParametricMemory(
                max_size=1000,
                embedding_dim=embedding_dim,
                learning_rate=learning_rate,
                enable_embeddings=True,
            )

        # Initialize base class with parametric memory
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            memory_buffer=parametric_memory,
            max_examples_in_prompt=max_examples_in_prompt,
            cl_config=cl_config,
        )

        self.tool_scorer = tool_scorer
        self.parametric_memory = parametric_memory
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.use_tool_scorer_for_selection = use_tool_scorer_for_selection

        # Track trajectories for parameter updates
        self._trajectory_embeddings: List[np.ndarray] = []
        self._trajectory_tools: List[str] = []
        self._trajectory_rewards: List[float] = []

        logger.info(
            f"Initialized {self.__class__.__name__} with parametric components: "
            f"embedding_dim={embedding_dim}, lr={learning_rate}, "
            f"use_tool_scorer={use_tool_scorer_for_selection}"
        )

    def _extract_state_embedding(
        self,
        messages: List[Message],
        method: str = "last_hidden",
    ) -> np.ndarray:
        """
        Extract state embedding φ(s) from messages using the frozen LLM.

        This is a CRITICAL function that bridges the frozen LLM and learnable parameters.

        Args:
            messages: Message history
            method: Embedding extraction method
                - "last_hidden": Use last hidden state (requires model access)
                - "mean_pooling": Mean pooling of token embeddings
                - "random": Random embedding (for testing)

        Returns:
            State embedding of shape (embedding_dim,)
        """
        # For now, use a simple embedding based on message content
        # In practice, you would extract embeddings from the LLM's hidden states

        if method == "random":
            # Random embedding for testing
            return np.random.randn(self.embedding_dim)

        # Extract text content
        text_parts = []
        for msg in messages[-5:]:  # Last 5 messages
            content = getattr(msg, 'content', '')
            if content:
                text_parts.append(str(content)[:200])  # Truncate

        text = " ".join(text_parts)

        # Generate embedding using OpenAI API
        try:
            if not hasattr(self, '_embedding_client'):
                import openai
                self._embedding_client = openai.OpenAI()

            response = self._embedding_client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000],
                dimensions=self.embedding_dim,
            )
            embedding = np.array(response.data[0].embedding)

            # Ensure correct dimension
            if embedding.shape[0] != self.embedding_dim:
                # Pad or truncate
                if embedding.shape[0] < self.embedding_dim:
                    embedding = np.pad(
                        embedding,
                        (0, self.embedding_dim - embedding.shape[0]),
                        mode='constant'
                    )
                else:
                    embedding = embedding[:self.embedding_dim]

            return embedding

        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}, using random embedding")
            return np.random.randn(self.embedding_dim)

    def _select_tool_with_scorer(
        self,
        state_embedding: np.ndarray,
        available_tools: Optional[List[str]] = None,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Get tool scores from the learnable scorer.

        Args:
            state_embedding: Current state embedding
            available_tools: Optional list of available tool names
            temperature: Sampling temperature

        Returns:
            Dictionary mapping tool names to probabilities
        """
        if available_tools is None:
            available_tools = [tool.name for tool in self.tools]

        return self.tool_scorer.get_tool_probabilities(
            state_embedding,
            tool_names=available_tools,
            temperature=temperature,
        )

    def _blend_scorer_with_llm(
        self,
        scorer_probs: Dict[str, float],
        llm_message: AssistantMessage,
        blend_weight: float = 0.5,
    ) -> Optional[str]:
        """
        Blend tool scorer probabilities with LLM's tool selection.

        This allows us to gradually shift from LLM-based selection
        to scorer-based selection as the agent learns.

        Args:
            scorer_probs: Probabilities from tool scorer
            llm_message: Message from LLM (may contain tool calls)
            blend_weight: Weight for scorer (0 = all LLM, 1 = all scorer)

        Returns:
            Selected tool name, or None if no tools
        """
        if not llm_message.tool_calls:
            return None

        # Get LLM's selected tool
        llm_tool = llm_message.tool_calls[0].name

        if blend_weight == 0:
            # Pure LLM selection
            return llm_tool

        if blend_weight == 1:
            # Pure scorer selection
            return max(scorer_probs.items(), key=lambda x: x[1])[0]

        # Blend: use scorer with probability = blend_weight
        if np.random.random() < blend_weight:
            # Sample from scorer probabilities
            tools = list(scorer_probs.keys())
            probs = list(scorer_probs.values())
            return np.random.choice(tools, p=probs)
        else:
            # Use LLM's selection
            return llm_tool

    @abstractmethod
    def _update_parameters(
        self,
        state_embedding: np.ndarray,
        selected_tool: str,
        reward: float,
        success: bool,
    ) -> Dict[str, Any]:
        """
        Update learnable parameters based on feedback.

        This is method-specific and must be implemented by subclasses.

        Args:
            state_embedding: State embedding φ(s)
            selected_tool: Tool that was selected
            reward: Reward signal
            success: Whether the action succeeded

        Returns:
            Update statistics
        """
        raise NotImplementedError

    def _retrieve_examples(
        self,
        observation: str,
        k: int = 5
    ) -> List[Any]:
        """
        Retrieve examples from parametric memory based on importance and similarity.

        Args:
            observation: Current observation
            k: Number of examples to retrieve

        Returns:
            List of experiences
        """
        # Use parametric memory's importance-based retrieval
        return self.parametric_memory.retrieve_by_importance(
            k=k,
            domain=self._current_domain,
        )

    def _build_prompt(
        self,
        observation: str,
        examples: List[Any]
    ) -> str:
        """
        Build prompt with examples from parametric memory.

        Args:
            observation: Current observation
            examples: Retrieved experiences

        Returns:
            Formatted prompt string
        """
        # Start with domain policy
        prompt_parts = [self.domain_policy, "\n\n"]

        # Add few-shot examples if available
        if examples:
            prompt_parts.append("Here are some relevant examples:\n\n")
            for i, exp in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:\n")
                prompt_parts.append(f"Observation: {exp.observation}\n")
                prompt_parts.append(f"Action: {exp.action}\n")
                if exp.success:
                    prompt_parts.append(f"Result: Success (reward={exp.reward:.2f})\n")
                else:
                    prompt_parts.append(f"Result: Failed\n")
                prompt_parts.append("\n")

        # Add current observation
        prompt_parts.append(f"Current task:\n{observation}\n")

        return "".join(prompt_parts)

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None
    ) -> ParametricCLAgentState:
        """Get initial state for the agent."""
        if message_history is None:
            message_history = []

        valid_messages = [
            msg for msg in message_history
            if is_valid_agent_history_message(msg)
        ]

        # Extract state embedding
        state_embedding = self._extract_state_embedding(valid_messages)

        # Retrieve examples from parametric memory
        examples = self.parametric_memory.retrieve_by_importance(
            k=self.max_examples_in_prompt,
            domain=self._current_domain,
        )

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

    @abstractmethod
    def _build_system_prompt_with_examples(
        self,
        examples: List[Any]
    ) -> str:
        """Build system prompt with examples."""
        raise NotImplementedError

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: ParametricCLAgentState
    ) -> Tuple[AssistantMessage, ParametricCLAgentState]:
        """Generate next message with parametric tool selection."""
        # Update state with new message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Extract current state embedding
        state_embedding = self._extract_state_embedding(state.messages)
        state.state_embeddings.append(state_embedding)

        # Get tool scores if enabled
        scorer_probs = None
        if self.use_tool_scorer_for_selection:
            scorer_probs = self._select_tool_with_scorer(state_embedding)
            logger.debug(f"Tool scorer probabilities: {scorer_probs}")

        # Generate response from LLM
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **self.llm_args,
        )

        # Optional: Blend scorer with LLM's tool selection
        if self.use_tool_scorer_for_selection and scorer_probs and assistant_message.tool_calls:
            blend_weight = self.cl_config.get("scorer_blend_weight", 0.3)
            selected_tool = self._blend_scorer_with_llm(
                scorer_probs,
                assistant_message,
                blend_weight=blend_weight,
            )
            if selected_tool and selected_tool != assistant_message.tool_calls[0].name:
                # Update the tool call to use scorer's selection
                logger.info(
                    f"Overriding LLM tool selection: "
                    f"{assistant_message.tool_calls[0].name} -> {selected_tool}"
                )
                # Note: This would require modifying the tool call
                # For simplicity, we'll track both but use LLM's actual call

        # Track selected tools
        if assistant_message.tool_calls:
            selected_tool = assistant_message.tool_calls[0].name
            state.selected_tools.append(selected_tool)

        state.messages.append(assistant_message)
        self._total_steps += 1

        return assistant_message, state

    def learn_from_trajectory(
        self,
        task_id: str,
        domain: str,
        trajectory: List[Message],
        reward: float,
        success: bool = False,
    ) -> Dict[str, Any]:
        """
        Learn from trajectory by updating parameters.

        This is the key difference from non-parametric agents:
        we actually UPDATE PARAMETERS here, not just store examples.
        """
        logger.info(
            f"Learning from trajectory: task={task_id}, "
            f"reward={reward:.3f}, success={success}"
        )

        # Extract state-action-reward tuples
        state_embeddings = []
        selected_tools = []
        step_rewards = []

        # Parse trajectory to extract (state, action) pairs
        message_buffer = []
        for msg in trajectory:
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                # Extract state before this action
                state_emb = self._extract_state_embedding(message_buffer)
                tool_name = msg.tool_calls[0].name

                state_embeddings.append(state_emb)
                selected_tools.append(tool_name)
                # Assign reward to this step (could be sparse or dense)
                step_rewards.append(reward if success else 0.0)

            message_buffer.append(msg)

        if not state_embeddings:
            logger.warning("No tool calls found in trajectory")
            return {
                "parameter_updates": 0,
                "experiences_added": 0,
                "reason": "no_tool_calls"
            }

        # Update parameters for each (state, action, reward) tuple
        update_stats = []
        for state_emb, tool, r in zip(state_embeddings, selected_tools, step_rewards):
            stats = self._update_parameters(state_emb, tool, r, success)
            update_stats.append(stats)

        # Store in parametric memory
        added_count = self.parametric_memory.add_from_trajectory(
            task_id=task_id,
            domain=domain,
            messages=trajectory,
            reward=reward,
            success=success,
        )

        # Reinforce importance of successful patterns
        if success:
            # Get the experience IDs that were just added
            recent_ids = [
                exp.experience_id for exp in self.parametric_memory._experiences[-added_count:]
            ]
            self.parametric_memory.reinforce_successful_retrievals(recent_ids, reward)

        self._tasks_completed += 1

        logger.info(
            f"Parameter updates: {len(update_stats)}, "
            f"Experiences added: {added_count}"
        )

        return {
            "parameter_updates": len(update_stats),
            "update_details": update_stats,
            "experiences_added": added_count,
            "task_id": task_id,
            "domain": domain,
            "reward": reward,
            "success": success,
        }

    def get_parameters(self) -> Dict[str, Any]:
        """Get all learnable parameters."""
        return {
            "tool_scorer": self.tool_scorer.get_parameters(),
            "parametric_memory": self.parametric_memory.get_parameters(),
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set all learnable parameters."""
        if "tool_scorer" in params:
            self.tool_scorer.set_parameters(params["tool_scorer"])
        if "parametric_memory" in params:
            self.parametric_memory.set_parameters(params["parametric_memory"])

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics including parametric components."""
        base_stats = super().get_statistics()
        base_stats.update({
            "tool_scorer_stats": self.tool_scorer.get_statistics(),
            "parametric_memory_stats": self.parametric_memory.get_statistics(),
        })
        return base_stats

    def save_state(self, path: str) -> None:
        """Save agent state including parameters."""
        super().save_state(path)

        # Save tool scorer
        scorer_path = path.replace('.json', '_scorer.pkl')
        self.tool_scorer.save(scorer_path)

        # Parametric memory is saved by base class
        logger.info(f"Saved parametric agent state to {path}")

    def load_state(self, path: str) -> None:
        """Load agent state including parameters."""
        super().load_state(path)

        # Load tool scorer
        scorer_path = path.replace('.json', '_scorer.pkl')
        try:
            self.tool_scorer.load(scorer_path)
        except FileNotFoundError:
            logger.warning(f"Tool scorer file not found: {scorer_path}")

        logger.info(f"Loaded parametric agent state from {path}")
