# Copyright Sierra
"""
Base Continual Learning Agent for Parameter-Free API Agents

This module provides the base class for continual learning agents
that work with API-based LLMs without parameter updates.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Generic, TypeVar

from pydantic import BaseModel, Field
from loguru import logger

from tau2.agent.base import BaseAgent, LocalAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

from tau2.continual_learning.memory.buffer import Experience, MemoryBuffer


class CLAgentState(BaseModel):
    """State for continual learning agents"""

    system_messages: List[SystemMessage] = Field(default_factory=list)
    messages: List[Message] = Field(default_factory=list)
    few_shot_examples: List[Dict[str, Any]] = Field(default_factory=list)
    current_task_id: Optional[str] = None
    current_domain: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class ContinualLearningAgent(LocalAgent, ABC):
    """
    Base class for continual learning agents.

    This is designed for parameter-free API agents where "learning"
    happens through updating the example bank and prompt construction
    rather than model parameter updates.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[Dict[str, Any]] = None,
        memory_buffer: Optional[MemoryBuffer] = None,
        max_examples_in_prompt: int = 5,
        cl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the continual learning agent.

        Args:
            tools: List of available tools
            domain_policy: The domain policy string
            llm: LLM model name
            llm_args: Additional LLM arguments
            memory_buffer: Shared memory buffer for experiences
            max_examples_in_prompt: Maximum few-shot examples in prompt
            cl_config: Additional continual learning configuration
        """
        super().__init__(tools=tools, domain_policy=domain_policy)

        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args else {}
        self.memory_buffer = memory_buffer or MemoryBuffer()
        self.max_examples_in_prompt = max_examples_in_prompt
        self.cl_config = cl_config or {}

        # Track current task context
        self._current_task_id: Optional[str] = None
        self._current_domain: Optional[str] = None
        self._current_trajectory: List[Dict[str, Any]] = []

        # Statistics
        self._total_steps = 0
        self._tasks_completed = 0

        logger.info(
            f"Initialized {self.__class__.__name__} with LLM={llm}, "
            f"max_examples={max_examples_in_prompt}"
        )

    @abstractmethod
    def _retrieve_examples(
        self,
        observation: str,
        k: int = 5
    ) -> List[Experience]:
        """
        Retrieve relevant examples from the memory buffer.

        Args:
            observation: Current observation/context
            k: Number of examples to retrieve

        Returns:
            List of relevant experiences
        """
        raise NotImplementedError

    @abstractmethod
    def _build_prompt(
        self,
        observation: str,
        examples: List[Experience]
    ) -> str:
        """
        Build the prompt with few-shot examples.

        Args:
            observation: Current observation/context
            examples: Retrieved examples

        Returns:
            The constructed prompt string
        """
        raise NotImplementedError

    @abstractmethod
    def learn_from_trajectory(
        self,
        task_id: str,
        domain: str,
        trajectory: List[Message],
        reward: float,
        success: bool = False,
    ) -> Dict[str, Any]:
        """
        Learn from a completed trajectory.

        Args:
            task_id: The task ID
            domain: The domain name
            trajectory: List of messages in the trajectory
            reward: Final reward for the trajectory
            success: Whether the task was successful

        Returns:
            Dictionary with learning statistics
        """
        raise NotImplementedError

    @abstractmethod
    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None
    ) -> CLAgentState:
        """
        Get the initial state of the agent.

        Args:
            message_history: Optional existing message history

        Returns:
            Initial CLAgentState
        """
        raise NotImplementedError

    @abstractmethod
    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: CLAgentState
    ) -> Tuple[AssistantMessage, CLAgentState]:
        """
        Generate the next message.

        Args:
            message: Input message
            state: Current state

        Returns:
            Tuple of (response message, updated state)
        """
        raise NotImplementedError

    def set_task_context(
        self,
        task_id: str,
        domain: str
    ) -> None:
        """
        Set the current task context.

        Args:
            task_id: The task ID
            domain: The domain name
        """
        self._current_task_id = task_id
        self._current_domain = domain
        self._current_trajectory = []
        logger.debug(f"Set task context: task_id={task_id}, domain={domain}")

    def clear_task_context(self) -> None:
        """Clear the current task context."""
        self._current_task_id = None
        self._current_domain = None
        self._current_trajectory = []

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_steps": self._total_steps,
            "tasks_completed": self._tasks_completed,
            "memory_buffer_size": len(self.memory_buffer),
            "memory_buffer_stats": self.memory_buffer.get_statistics(),
        }

    def consolidate(self) -> None:
        """
        Consolidate learned knowledge.

        This can be called periodically to prune/reorganize
        the memory buffer or update internal structures.
        """
        pass  # Default: no consolidation

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message."""
        if message.content and "###STOP###" in message.content:
            return True
        return False

    def save_state(self, path: str) -> None:
        """Save agent state to file."""
        import json
        state = {
            "total_steps": self._total_steps,
            "tasks_completed": self._tasks_completed,
            "cl_config": self.cl_config,
        }
        with open(path, 'w') as f:
            json.dump(state, f)

        # Save memory buffer separately
        buffer_path = path.replace('.json', '_buffer.json')
        self.memory_buffer.save(buffer_path)

        logger.info(f"Saved agent state to {path}")

    def load_state(self, path: str) -> None:
        """Load agent state from file."""
        import json
        with open(path, 'r') as f:
            state = json.load(f)

        self._total_steps = state.get("total_steps", 0)
        self._tasks_completed = state.get("tasks_completed", 0)
        self.cl_config = state.get("cl_config", {})

        # Load memory buffer
        buffer_path = path.replace('.json', '_buffer.json')
        try:
            self.memory_buffer.load(buffer_path)
        except FileNotFoundError:
            logger.warning(f"Memory buffer file not found: {buffer_path}")

        logger.info(f"Loaded agent state from {path}")
