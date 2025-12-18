# Copyright Sierra
"""
Progressive / Modular Continual Learning Agent

This agent implements progressive learning by:
1. Freezing old modules when learning new tasks
2. Adding new modules/skills for new task families
3. Learning a routing policy to select modules
4. Preventing forgetting through module freezing
"""

from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from loguru import logger
from copy import deepcopy

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


class Module:
    """
    A skill module that can be frozen/unfrozen.

    Each module is essentially a tool scorer for a specific skill.
    """

    def __init__(
        self,
        module_id: int,
        name: str,
        tool_scorer: ToolScorer,
        frozen: bool = False,
    ):
        """
        Initialize module.

        Args:
            module_id: Unique module ID
            name: Module name
            tool_scorer: Tool scorer for this module
            frozen: Whether module is frozen
        """
        self.module_id = module_id
        self.name = name
        self.tool_scorer = tool_scorer
        self.frozen = frozen
        self.creation_step = 0
        self.last_used_step = 0
        self.total_uses = 0

    def freeze(self) -> None:
        """Freeze this module (prevent updates)."""
        self.frozen = True
        logger.info(f"Froze module {self.module_id}: {self.name}")

    def unfreeze(self) -> None:
        """Unfreeze this module (allow updates)."""
        self.frozen = False
        logger.info(f"Unfroze module {self.module_id}: {self.name}")

    def get_scores(
        self,
        state_embedding: np.ndarray,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Get tool scores from this module."""
        return self.tool_scorer.get_tool_probabilities(
            state_embedding,
            temperature=temperature,
        )

    def update(
        self,
        state_embedding: np.ndarray,
        selected_tool: str,
        reward: float,
        success: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Update module parameters if not frozen.

        Args:
            state_embedding: State embedding
            selected_tool: Selected tool
            reward: Reward signal
            success: Whether succeeded

        Returns:
            Update statistics, or None if frozen
        """
        if self.frozen:
            logger.debug(f"Module {self.module_id} is frozen, skipping update")
            return None

        stats = self.tool_scorer.update_weights(
            state_embedding=state_embedding,
            selected_tool=selected_tool,
            reward=reward,
            success=success,
        )

        self.total_uses += 1
        return stats


class ModuleRouter:
    """
    Router that selects which module(s) to use.

    Learns: π_route(module | state)
    """

    def __init__(
        self,
        embedding_dim: int,
        learning_rate: float = 0.01,
        use_attention: bool = True,
    ):
        """
        Initialize module router.

        Args:
            embedding_dim: Embedding dimension
            learning_rate: Learning rate
            use_attention: Whether to use attention-based routing
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.use_attention = use_attention

        # Router parameters: one weight vector per potential module
        self.module_weights: Dict[int, np.ndarray] = {}

        # Attention weights (if using attention)
        if use_attention:
            self.attention_query = np.random.randn(embedding_dim) * 0.01

        logger.info(
            f"Initialized ModuleRouter with embedding_dim={embedding_dim}, "
            f"attention={use_attention}"
        )

    def add_module(self, module_id: int) -> None:
        """Add a new module to the router."""
        if module_id not in self.module_weights:
            self.module_weights[module_id] = np.random.randn(self.embedding_dim) * 0.01
            logger.info(f"Added routing weights for module {module_id}")

    def route(
        self,
        state_embedding: np.ndarray,
        available_modules: List[int],
        temperature: float = 1.0,
    ) -> Dict[int, float]:
        """
        Compute routing probabilities over modules.

        Args:
            state_embedding: Current state embedding
            available_modules: List of available module IDs
            temperature: Temperature for softmax

        Returns:
            Dictionary mapping module_id to probability
        """
        if not available_modules:
            return {}

        # Compute scores for each module
        scores = []
        for module_id in available_modules:
            if module_id not in self.module_weights:
                self.add_module(module_id)

            if self.use_attention:
                # Attention-based routing
                score = np.dot(self.attention_query, state_embedding)
                score += np.dot(self.module_weights[module_id], state_embedding)
            else:
                # Simple dot product
                score = np.dot(self.module_weights[module_id], state_embedding)

            scores.append(score)

        scores = np.array(scores) / temperature

        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        return {module_id: float(p) for module_id, p in zip(available_modules, probs)}

    def update(
        self,
        state_embedding: np.ndarray,
        selected_module: int,
        reward: float,
    ) -> Dict[str, Any]:
        """
        Update routing weights.

        Uses REINFORCE:
            ∇W_k = α * reward * ∇log π(k | state)

        Args:
            state_embedding: State embedding
            selected_module: Selected module ID
            reward: Reward signal

        Returns:
            Update statistics
        """
        if selected_module not in self.module_weights:
            return {"updated": False, "reason": "module_not_found"}

        # Get current probabilities
        probs = self.route(state_embedding, list(self.module_weights.keys()))
        prob = probs.get(selected_module, 0.0)

        # Gradient: (1 - prob) * state_embedding
        gradient = reward * (1 - prob) * state_embedding

        # Update module weights
        self.module_weights[selected_module] += self.learning_rate * gradient

        # Update attention if using
        if self.use_attention:
            self.attention_query += self.learning_rate * gradient * 0.1

        return {
            "updated": True,
            "selected_module": selected_module,
            "probability": prob,
            "reward": reward,
        }


class ProgressiveModularAgent(ParametricCLAgent):
    """
    Progressive / Modular Continual Learning Agent.

    Key features:
    1. Modules are added progressively for new tasks
    2. Old modules are frozen to prevent forgetting
    3. Router learns to select appropriate modules
    4. No backward updates to old modules

    Process for new task:
        1. Freeze all existing modules
        2. Create new module for the task
        3. Learn routing to new module
        4. Learn new module's parameters
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[Dict[str, Any]] = None,
        parametric_memory: Optional[ParametricMemory] = None,
        embedding_dim: int = 768,
        learning_rate: float = 0.01,
        max_examples_in_prompt: int = 5,
        # Progressive learning specific
        freeze_on_task_change: bool = True,
        max_modules: int = 10,
        module_reuse_threshold: float = 0.7,
        use_attention_routing: bool = True,
        allow_module_composition: bool = True,
        cl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Progressive Modular agent.

        Args:
            tools: Available tools
            domain_policy: Domain policy
            llm: LLM model name
            llm_args: LLM arguments
            parametric_memory: Memory system
            embedding_dim: Embedding dimension
            learning_rate: Learning rate
            max_examples_in_prompt: Max examples in prompt
            freeze_on_task_change: Freeze modules when task changes
            max_modules: Maximum number of modules
            module_reuse_threshold: Threshold for reusing existing modules
            use_attention_routing: Use attention-based routing
            allow_module_composition: Allow combining multiple modules
            cl_config: Additional config
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            tool_scorer=None,  # We'll use modules instead
            parametric_memory=parametric_memory,
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            max_examples_in_prompt=max_examples_in_prompt,
            use_tool_scorer_for_selection=True,
            cl_config=cl_config,
        )

        self.freeze_on_task_change = freeze_on_task_change
        self.max_modules = max_modules
        self.module_reuse_threshold = module_reuse_threshold
        self.allow_module_composition = allow_module_composition

        # Module management
        self.modules: Dict[int, Module] = {}
        self.next_module_id = 0
        self.active_modules: Set[int] = set()
        self.current_module_id: Optional[int] = None

        # Create initial module
        self._create_module("initial_module")

        # Module router
        self.router = ModuleRouter(
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            use_attention=use_attention_routing,
        )

        # Track module usage per domain
        self.domain_to_modules: Dict[str, List[int]] = {}

        logger.info(
            f"Initialized ProgressiveModularAgent with "
            f"max_modules={max_modules}, "
            f"freeze_on_task_change={freeze_on_task_change}"
        )

    @property
    def tool_scorer(self) -> ToolScorer:
        """Get current active module's tool scorer."""
        if self.current_module_id is not None:
            return self.modules[self.current_module_id].tool_scorer
        else:
            return self.modules[0].tool_scorer

    @tool_scorer.setter
    def tool_scorer(self, value):
        """Ignore setter."""
        pass

    def _create_module(self, name: str) -> int:
        """
        Create a new module.

        Args:
            name: Module name

        Returns:
            Module ID
        """
        if len(self.modules) >= self.max_modules:
            logger.warning(f"Reached max modules ({self.max_modules}), cannot create new module")
            return list(self.modules.keys())[0]

        module_id = self.next_module_id
        self.next_module_id += 1

        # Create new tool scorer for this module
        scorer = ToolScorer(
            tools=self.tools,
            embedding_dim=self.embedding_dim,
            learning_rate=self.learning_rate,
        )

        module = Module(
            module_id=module_id,
            name=name,
            tool_scorer=scorer,
            frozen=False,
        )

        self.modules[module_id] = module
        self.active_modules.add(module_id)

        # Add to router
        self.router.add_module(module_id)

        logger.info(f"Created module {module_id}: {name}")
        return module_id

    def _freeze_all_modules_except(self, keep_active: Optional[int] = None) -> None:
        """
        Freeze all modules except the specified one.

        Args:
            keep_active: Module ID to keep active (None = freeze all)
        """
        for module_id, module in self.modules.items():
            if module_id == keep_active:
                module.unfreeze()
            else:
                module.freeze()

    def _select_module(
        self,
        state_embedding: np.ndarray,
        domain: str,
    ) -> int:
        """
        Select which module to use.

        Args:
            state_embedding: State embedding
            domain: Current domain

        Returns:
            Selected module ID
        """
        # Check if domain has associated modules
        if domain in self.domain_to_modules and self.domain_to_modules[domain]:
            # Use domain-specific modules
            available = self.domain_to_modules[domain]
        else:
            # Use all modules
            available = list(self.modules.keys())

        # Route to module
        probs = self.router.route(state_embedding, available)

        # Sample module
        module_ids = list(probs.keys())
        prob_values = list(probs.values())

        selected = int(np.random.choice(module_ids, p=prob_values))

        logger.debug(f"Selected module {selected} for domain '{domain}' (prob={probs[selected]:.3f})")

        return selected

    def _handle_new_task(self, domain: str) -> int:
        """
        Handle a new task by creating/selecting a module.

        Args:
            domain: Task domain

        Returns:
            Module ID to use
        """
        # Freeze all existing modules if configured
        if self.freeze_on_task_change:
            self._freeze_all_modules_except(None)

        # Create new module for this domain
        module_id = self._create_module(f"module_{domain}")

        # Associate domain with module
        if domain not in self.domain_to_modules:
            self.domain_to_modules[domain] = []
        self.domain_to_modules[domain].append(module_id)

        # Unfreeze the new module
        self.modules[module_id].unfreeze()

        logger.info(f"Created new module {module_id} for domain '{domain}'")

        return module_id

    def _select_tool_with_scorer(
        self,
        state_embedding: np.ndarray,
        available_tools: Optional[List[str]] = None,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Get tool scores using modular architecture.

        Args:
            state_embedding: State embedding
            available_tools: Available tools
            temperature: Temperature

        Returns:
            Tool probabilities
        """
        domain = self._current_domain or "default"

        # Select or create module
        if domain not in self.domain_to_modules or not self.domain_to_modules[domain]:
            module_id = self._handle_new_task(domain)
        else:
            module_id = self._select_module(state_embedding, domain)

        self.current_module_id = module_id

        # Get scores from selected module
        module = self.modules[module_id]
        scores = module.get_scores(state_embedding, temperature)

        # Optional: Compose with other modules
        if self.allow_module_composition and len(self.modules) > 1:
            # Get top-2 modules
            probs = self.router.route(state_embedding, list(self.modules.keys()))
            top_modules = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]

            if len(top_modules) > 1:
                # Blend scores
                other_id = top_modules[1][0]
                other_weight = top_modules[1][1]

                if other_weight > 0.2:  # Only if significant
                    other_scores = self.modules[other_id].get_scores(state_embedding, temperature)

                    # Weighted blend
                    for tool_name in scores.keys():
                        scores[tool_name] = (
                            0.7 * scores[tool_name] +
                            0.3 * other_scores.get(tool_name, 0.0)
                        )

                    logger.debug(f"Composed module {module_id} with module {other_id}")

        return scores

    def _update_parameters(
        self,
        state_embedding: np.ndarray,
        selected_tool: str,
        reward: float,
        success: bool,
    ) -> Dict[str, Any]:
        """
        Update parameters (only active module).

        Args:
            state_embedding: State embedding
            selected_tool: Selected tool
            reward: Reward
            success: Success flag

        Returns:
            Update statistics
        """
        if self.current_module_id is None:
            return {"updated": False, "reason": "no_active_module"}

        module = self.modules[self.current_module_id]

        # Update module (only if not frozen)
        module_stats = module.update(state_embedding, selected_tool, reward, success)

        # Update router
        router_stats = self.router.update(state_embedding, self.current_module_id, reward)

        return {
            "updated": module_stats is not None,
            "active_module": self.current_module_id,
            "module_frozen": module.frozen,
            "module_update": module_stats,
            "router_update": router_stats,
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

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics including module info."""
        stats = super().get_statistics()

        module_stats = {}
        for module_id, module in self.modules.items():
            module_stats[f"module_{module_id}"] = {
                "name": module.name,
                "frozen": module.frozen,
                "total_uses": module.total_uses,
                "tool_scorer_stats": module.tool_scorer.get_statistics(),
            }

        # Router统计信息
        router_stats = {}
        if hasattr(self, 'router') and self.router is not None:
            router_stats = {
                "num_module_weights": len(self.router.module_weights),
                "use_attention": self.router.use_attention,
                "learning_rate": self.router.learning_rate,
            }

        stats.update({
            "num_modules": len(self.modules),
            "num_frozen_modules": sum(1 for m in self.modules.values() if m.frozen),
            "active_module": self.current_module_id,
            "domain_to_modules": {k: len(v) for k, v in self.domain_to_modules.items()},
            "module_stats": module_stats,
            "router_stats": router_stats,
        })

        return stats

    def consolidate(self) -> None:
        """Consolidate modules by pruning unused ones."""
        super().consolidate()

        # Find unused modules
        unused = [
            module_id for module_id, module in self.modules.items()
            if module.total_uses == 0 and len(self.modules) > 1
        ]

        # Remove unused modules
        for module_id in unused:
            del self.modules[module_id]
            self.active_modules.discard(module_id)
            logger.info(f"Pruned unused module {module_id}")

    def save_state(self, path: str) -> None:
        """Save progressive agent state."""
        super().save_state(path)

        import pickle
        modules_path = path.replace('.json', '_modules.pkl')
        with open(modules_path, 'wb') as f:
            pickle.dump({
                "modules": {
                    k: {
                        "name": m.name,
                        "frozen": m.frozen,
                        "parameters": m.tool_scorer.get_parameters(),
                    }
                    for k, m in self.modules.items()
                },
                "router_weights": self.router.module_weights,
                "domain_to_modules": dict(self.domain_to_modules),
            }, f)

        logger.info(f"Saved progressive agent state to {modules_path}")


def create_progressive_agent(
    tools: List[Tool],
    domain_policy: str,
    llm: str = "gpt-4",
    embedding_dim: int = 768,
    learning_rate: float = 0.01,
    freeze_on_task_change: bool = True,
    max_modules: int = 10,
    **kwargs
) -> ProgressiveModularAgent:
    """
    Factory function to create a Progressive Modular agent.

    Args:
        tools: Available tools
        domain_policy: Domain policy
        llm: LLM model name
        embedding_dim: Embedding dimension
        learning_rate: Learning rate
        freeze_on_task_change: Freeze modules on task change
        max_modules: Maximum number of modules
        **kwargs: Additional arguments

    Returns:
        Configured ProgressiveModularAgent
    """
    return ProgressiveModularAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        freeze_on_task_change=freeze_on_task_change,
        max_modules=max_modules,
        **kwargs
    )
