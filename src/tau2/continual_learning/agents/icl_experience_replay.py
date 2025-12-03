# Copyright Sierra
"""
ICL Experience Replay Agent

This is the main continual learning agent that uses In-Context Learning
with Experience Replay for parameter-free continual learning.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel

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
from tau2.continual_learning.memory.buffer import (
    Experience,
    MemoryBuffer,
    SamplingStrategy,
)


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


class ICLExperienceReplayAgent(ContinualLearningAgent):
    """
    In-Context Learning with Experience Replay Agent.

    This agent implements continual learning for parameter-free API agents
    by maintaining an experience buffer and using retrieved experiences
    as few-shot examples in the prompt.

    Key features:
    - Stores successful experiences in a memory buffer
    - Retrieves relevant experiences based on similarity
    - Constructs few-shot prompts with retrieved examples
    - Supports diversity-aware example selection
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[Dict[str, Any]] = None,
        memory_buffer: Optional[MemoryBuffer] = None,
        max_examples_in_prompt: int = 5,
        retrieval_strategy: SamplingStrategy = SamplingStrategy.DIVERSITY,
        min_reward_for_storage: float = 0.5,
        enable_similarity_retrieval: bool = True,
        diversity_weight: float = 0.3,
        cl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ICL-ER agent.

        Args:
            tools: Available tools
            domain_policy: Domain policy string
            llm: LLM model name
            llm_args: Additional LLM arguments
            memory_buffer: Shared memory buffer
            max_examples_in_prompt: Maximum examples in prompt
            retrieval_strategy: Strategy for retrieving examples
            min_reward_for_storage: Minimum reward to store experience
            enable_similarity_retrieval: Enable embedding-based retrieval
            diversity_weight: Weight for diversity in selection
            cl_config: Additional configuration
        """
        # Create memory buffer with appropriate settings
        if memory_buffer is None:
            memory_buffer = MemoryBuffer(
                max_size=1000,
                sampling_strategy=retrieval_strategy,
                min_reward_threshold=min_reward_for_storage,
                diversity_weight=diversity_weight,
                enable_embeddings=enable_similarity_retrieval,
            )

        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            memory_buffer=memory_buffer,
            max_examples_in_prompt=max_examples_in_prompt,
            cl_config=cl_config,
        )

        self.retrieval_strategy = retrieval_strategy
        self.min_reward_for_storage = min_reward_for_storage
        self.enable_similarity_retrieval = enable_similarity_retrieval
        self.diversity_weight = diversity_weight

    @property
    def system_prompt(self) -> str:
        """Generate system prompt without few-shot examples."""
        return SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION,
            few_shot_section="",
            domain_policy=self.domain_policy,
        )

    def _build_system_prompt_with_examples(
        self,
        examples: List[Experience]
    ) -> str:
        """
        Build system prompt with few-shot examples.

        Args:
            examples: List of experiences to include

        Returns:
            System prompt string with examples
        """
        if not examples:
            few_shot_section = ""
        else:
            example_strs = []
            for i, exp in enumerate(examples, 1):
                # Truncate long observations/actions
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

    def _retrieve_examples(
        self,
        observation: str,
        k: int = None
    ) -> List[Experience]:
        """
        Retrieve relevant examples from memory buffer.

        Args:
            observation: Current observation/context
            k: Number of examples to retrieve

        Returns:
            List of relevant experiences
        """
        if k is None:
            k = self.max_examples_in_prompt

        if len(self.memory_buffer) == 0:
            return []

        # Use similarity-based retrieval if enabled
        if self.enable_similarity_retrieval:
            examples = self.memory_buffer.retrieve_similar(
                query=observation,
                k=k,
                domain=self._current_domain,
            )
        else:
            # Fall back to strategy-based sampling
            examples = self.memory_buffer.sample(
                n=k,
                strategy=self.retrieval_strategy,
                domain=self._current_domain,
                exclude_task_id=self._current_task_id,  # Avoid self-reference
            )

        # Apply diversity filtering
        if len(examples) > 1:
            examples = self._diversify_examples(examples, k)

        logger.debug(f"Retrieved {len(examples)} examples for prompt")
        return examples

    def _diversify_examples(
        self,
        examples: List[Experience],
        k: int
    ) -> List[Experience]:
        """
        Ensure diversity in selected examples.

        Args:
            examples: Candidate examples
            k: Target number of examples

        Returns:
            Diversified list of examples
        """
        if len(examples) <= k:
            return examples

        selected = []
        seen_domains = set()
        seen_tasks = set()
        seen_tools = set()

        # First pass: prioritize diversity
        for exp in examples:
            if len(selected) >= k:
                break

            is_diverse = (
                exp.domain not in seen_domains or
                exp.task_id not in seen_tasks or
                not any(t in seen_tools for t in exp.required_tools)
            )

            if is_diverse:
                selected.append(exp)
                seen_domains.add(exp.domain)
                seen_tasks.add(exp.task_id)
                seen_tools.update(exp.required_tools)

        # Second pass: fill remaining slots
        for exp in examples:
            if len(selected) >= k:
                break
            if exp not in selected:
                selected.append(exp)

        return selected[:k]

    def _build_prompt(
        self,
        observation: str,
        examples: List[Experience]
    ) -> str:
        """
        Build the prompt with retrieved examples.

        Note: For this implementation, examples are included in the
        system prompt rather than this method.

        Args:
            observation: Current observation
            examples: Retrieved examples

        Returns:
            The observation (examples are in system prompt)
        """
        return observation

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None
    ) -> CLAgentState:
        """
        Get initial state for the agent.

        Args:
            message_history: Optional existing message history

        Returns:
            Initial CLAgentState
        """
        if message_history is None:
            message_history = []

        # Filter for valid agent messages
        valid_messages = [
            msg for msg in message_history
            if is_valid_agent_history_message(msg)
        ]

        # Retrieve examples for the current context
        context = " ".join(
            getattr(msg, 'content', '') or ''
            for msg in valid_messages[-5:]  # Last 5 messages for context
        )
        examples = self._retrieve_examples(context)

        # Build system prompt with examples
        system_prompt = self._build_system_prompt_with_examples(examples)

        return CLAgentState(
            system_messages=[SystemMessage(role="system", content=system_prompt)],
            messages=valid_messages,
            few_shot_examples=[exp.model_dump() for exp in examples],
            current_task_id=self._current_task_id,
            current_domain=self._current_domain,
        )

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
        # Update state with new message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Optionally refresh examples based on current context
        if len(state.messages) % 3 == 0:  # Refresh every 3 turns
            context = " ".join(
                getattr(msg, 'content', '') or ''
                for msg in state.messages[-5:]
            )
            examples = self._retrieve_examples(context)
            system_prompt = self._build_system_prompt_with_examples(examples)
            state.system_messages = [SystemMessage(role="system", content=system_prompt)]

        # Generate response
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **self.llm_args,
        )

        state.messages.append(assistant_message)
        self._total_steps += 1

        # Track in current trajectory
        self._current_trajectory.append({
            "type": "assistant",
            "message": assistant_message,
        })

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
        Learn from a completed trajectory.

        This method extracts successful experiences from the trajectory
        and adds them to the memory buffer.

        Args:
            task_id: The task ID
            domain: The domain name
            trajectory: List of messages in the trajectory
            reward: Final reward
            success: Whether task succeeded

        Returns:
            Learning statistics
        """
        # Only learn from successful trajectories
        if reward < self.min_reward_for_storage:
            logger.debug(
                f"Skipping learning from trajectory with reward={reward} "
                f"(< {self.min_reward_for_storage})"
            )
            return {
                "experiences_added": 0,
                "reason": "below_reward_threshold"
            }

        # Add experiences from trajectory
        added_count = self.memory_buffer.add_from_trajectory(
            task_id=task_id,
            domain=domain,
            messages=trajectory,
            reward=reward,
            success=success,
        )

        self._tasks_completed += 1

        logger.info(
            f"Learned from trajectory: task={task_id}, domain={domain}, "
            f"reward={reward}, added={added_count} experiences"
        )

        return {
            "experiences_added": added_count,
            "total_experiences": len(self.memory_buffer),
            "task_id": task_id,
            "domain": domain,
            "reward": reward,
        }

    def consolidate(self) -> None:
        """
        Consolidate the memory buffer.

        This can be called periodically to clean up the buffer,
        remove low-quality experiences, or reorganize storage.
        """
        stats_before = self.memory_buffer.get_statistics()

        # Currently just logs stats, could add pruning logic
        logger.info(
            f"Memory buffer consolidation: "
            f"{stats_before['total_experiences']} experiences, "
            f"avg_reward={stats_before['avg_reward']:.3f}"
        )

    def set_seed(self, seed: int) -> None:
        """Set random seed."""
        if "seed" in self.llm_args:
            logger.warning(f"Overwriting existing seed with {seed}")
        self.llm_args["seed"] = seed


class ICLERAgentConfig(BaseModel):
    """Configuration for ICL-ER Agent"""

    llm: str = "gpt-4"
    max_examples_in_prompt: int = 5
    retrieval_strategy: str = "diversity"
    min_reward_for_storage: float = 0.5
    enable_similarity_retrieval: bool = True
    diversity_weight: float = 0.3

    # Memory buffer config
    memory_buffer_size: int = 1000
    embedding_model: Optional[str] = None


def create_icl_er_agent(
    tools: List[Tool],
    domain_policy: str,
    config: ICLERAgentConfig,
    memory_buffer: Optional[MemoryBuffer] = None,
) -> ICLExperienceReplayAgent:
    """
    Factory function to create an ICL-ER agent.

    Args:
        tools: Available tools
        domain_policy: Domain policy string
        config: Agent configuration
        memory_buffer: Optional shared memory buffer

    Returns:
        Configured ICLExperienceReplayAgent
    """
    return ICLExperienceReplayAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=config.llm,
        memory_buffer=memory_buffer,
        max_examples_in_prompt=config.max_examples_in_prompt,
        retrieval_strategy=SamplingStrategy(config.retrieval_strategy),
        min_reward_for_storage=config.min_reward_for_storage,
        enable_similarity_retrieval=config.enable_similarity_retrieval,
        diversity_weight=config.diversity_weight,
    )
