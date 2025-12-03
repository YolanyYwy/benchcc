# Copyright Sierra
"""
Prompt Strategy Agent

This agent implements continual learning through prompt strategy evolution,
maintaining multiple prompt variants and adaptively selecting the best one.
"""

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import ValidAgentInputMessage, is_valid_agent_history_message
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

from tau2.continual_learning.agents.base import ContinualLearningAgent, CLAgentState
from tau2.continual_learning.memory.buffer import (
    Experience,
    MemoryBuffer,
    SamplingStrategy,
)


class PromptVariant(BaseModel):
    """A prompt variant with its template and configuration"""

    name: str
    template: str
    example_selection: str = "similarity"  # similarity, diversity, recent, random
    description: str = ""


class PromptStrategyAgent(ContinualLearningAgent):
    """
    Prompt Strategy Evolution Agent.

    This agent maintains multiple prompt variants and learns which
    works best for different domains/tasks through online evaluation.
    """

    # Default prompt variants
    DEFAULT_VARIANTS = [
        PromptVariant(
            name="standard",
            template="""You are a customer service agent. Follow the policy below.

{examples_section}

<policy>
{domain_policy}
</policy>

Help the user with their request. Either send a message or make a tool call.""",
            example_selection="similarity",
            description="Standard few-shot prompt"
        ),
        PromptVariant(
            name="chain_of_thought",
            template="""You are a customer service agent. Think step by step before acting.

{examples_section}

<policy>
{domain_policy}
</policy>

First think about what you need to do, then either send a message or make a tool call.""",
            example_selection="similarity",
            description="Chain of thought prompting"
        ),
        PromptVariant(
            name="task_specific",
            template="""You are a customer service agent specialized in handling requests.

{examples_section}

<policy>
{domain_policy}
</policy>

Focus on completing the user's request efficiently.""",
            example_selection="diversity",
            description="Task-focused prompt"
        ),
        PromptVariant(
            name="minimal",
            template="""Customer service agent. Policy:

{domain_policy}

Help the user. Either message or tool call, not both.""",
            example_selection="recent",
            description="Minimal prompt for efficiency"
        ),
    ]

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[Dict[str, Any]] = None,
        memory_buffer: Optional[MemoryBuffer] = None,
        max_examples_in_prompt: int = 5,
        prompt_variants: Optional[List[PromptVariant]] = None,
        exploration_rate: float = 0.1,
        cl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the prompt strategy agent.

        Args:
            tools: Available tools
            domain_policy: Domain policy string
            llm: LLM model name
            llm_args: Additional LLM arguments
            memory_buffer: Shared memory buffer
            max_examples_in_prompt: Max examples in prompt
            prompt_variants: Custom prompt variants
            exploration_rate: Probability of exploring non-best variant
            cl_config: Additional configuration
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            memory_buffer=memory_buffer,
            max_examples_in_prompt=max_examples_in_prompt,
            cl_config=cl_config,
        )

        self.variants = prompt_variants or self.DEFAULT_VARIANTS
        self.exploration_rate = exploration_rate

        # Track performance per variant
        self._variant_performance: Dict[str, List[float]] = {
            v.name: [] for v in self.variants
        }
        self._domain_variant_performance: Dict[str, Dict[str, List[float]]] = {}

        # Current selection
        self._current_variant_idx = 0
        self._current_variant: PromptVariant = self.variants[0]

    def _retrieve_examples(
        self,
        observation: str,
        k: int = None
    ) -> List[Experience]:
        """Retrieve examples based on current variant's strategy."""
        if k is None:
            k = self.max_examples_in_prompt

        if len(self.memory_buffer) == 0:
            return []

        # Use variant's example selection strategy
        strategy_map = {
            "similarity": SamplingStrategy.SIMILARITY,
            "diversity": SamplingStrategy.DIVERSITY,
            "recent": SamplingStrategy.RECENCY_WEIGHTED,
            "random": SamplingStrategy.UNIFORM,
        }
        strategy = strategy_map.get(
            self._current_variant.example_selection,
            SamplingStrategy.DIVERSITY
        )

        return self.memory_buffer.sample(
            n=k,
            strategy=strategy,
            query=observation if strategy == SamplingStrategy.SIMILARITY else None,
            domain=self._current_domain,
        )

    def _build_prompt(
        self,
        observation: str,
        examples: List[Experience]
    ) -> str:
        """Build prompt using current variant's template."""
        # Build examples section
        if examples:
            example_strs = []
            for i, exp in enumerate(examples, 1):
                obs_truncated = exp.observation[:300] + "..." if len(exp.observation) > 300 else exp.observation
                action_truncated = exp.action[:200] if len(exp.action) > 200 else exp.action
                example_strs.append(
                    f"Example {i}:\n"
                    f"Context: {obs_truncated}\n"
                    f"Action: {action_truncated}"
                )
            examples_section = "<examples>\n" + "\n\n".join(example_strs) + "\n</examples>"
        else:
            examples_section = ""

        return self._current_variant.template.format(
            examples_section=examples_section,
            domain_policy=self.domain_policy,
        )

    def _select_variant(self) -> PromptVariant:
        """Select which variant to use (exploration vs exploitation)."""
        if random.random() < self.exploration_rate:
            # Explore: random variant
            return random.choice(self.variants)

        # Exploit: best performing variant
        # Consider domain-specific performance if available
        if self._current_domain and self._current_domain in self._domain_variant_performance:
            domain_perf = self._domain_variant_performance[self._current_domain]
            best_variant = max(
                self.variants,
                key=lambda v: (
                    np.mean(domain_perf.get(v.name, [0.5])) if domain_perf.get(v.name) else 0.5
                )
            )
        else:
            # Use global performance
            best_variant = max(
                self.variants,
                key=lambda v: (
                    np.mean(self._variant_performance.get(v.name, [0.5])) if self._variant_performance.get(v.name) else 0.5
                )
            )

        return best_variant

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None
    ) -> CLAgentState:
        """Get initial state for the agent."""
        if message_history is None:
            message_history = []

        valid_messages = [
            msg for msg in message_history
            if is_valid_agent_history_message(msg)
        ]

        # Select variant for this task
        self._current_variant = self._select_variant()
        self._current_variant_idx = self.variants.index(self._current_variant)

        # Retrieve examples and build prompt
        context = " ".join(
            getattr(msg, 'content', '') or ''
            for msg in valid_messages[-5:]
        )
        examples = self._retrieve_examples(context)
        system_prompt = self._build_prompt("", examples)

        logger.debug(f"Selected variant: {self._current_variant.name}")

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
        """Generate the next message."""
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **self.llm_args,
        )

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
        """Learn from trajectory - update variant performance."""
        # Record performance for current variant
        variant_name = self._current_variant.name
        self._variant_performance[variant_name].append(reward)

        # Record domain-specific performance
        if domain not in self._domain_variant_performance:
            self._domain_variant_performance[domain] = {
                v.name: [] for v in self.variants
            }
        self._domain_variant_performance[domain][variant_name].append(reward)

        # Also store experiences in buffer
        added_count = 0
        min_reward = self.cl_config.get("min_reward_threshold", 0.5)
        if reward >= min_reward:
            added_count = self.memory_buffer.add_from_trajectory(
                task_id=task_id,
                domain=domain,
                messages=trajectory,
                reward=reward,
                success=success,
            )

        self._tasks_completed += 1

        # Log variant performance update
        avg_perf = np.mean(self._variant_performance[variant_name])
        logger.info(
            f"Updated variant '{variant_name}' performance: "
            f"reward={reward}, avg={avg_perf:.3f}"
        )

        return {
            "variant_used": variant_name,
            "reward": reward,
            "experiences_added": added_count,
            "variant_avg_performance": float(avg_perf),
        }

    def get_variant_statistics(self) -> Dict[str, Any]:
        """Get statistics about variant performance."""
        stats = {}
        for v in self.variants:
            perfs = self._variant_performance.get(v.name, [])
            stats[v.name] = {
                "num_uses": len(perfs),
                "avg_reward": float(np.mean(perfs)) if perfs else 0.0,
                "total_reward": float(np.sum(perfs)) if perfs else 0.0,
            }
        return stats

    def set_seed(self, seed: int) -> None:
        """Set random seed."""
        random.seed(seed)
        np.random.seed(seed)
        if "seed" in self.llm_args:
            logger.warning(f"Overwriting existing seed with {seed}")
        self.llm_args["seed"] = seed
