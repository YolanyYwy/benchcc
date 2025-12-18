# Copyright Sierra
"""
Parameter Isolation (Agent Adapter) Continual Learning Agent

This agent implements parameter isolation by:
1. Allocating separate parameter subsets for different task families
2. Sharing state embeddings φ(s) but isolating tool scorer weights
3. Learning a router to select which parameter subset to use
4. Preventing interference between different task domains
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from collections import defaultdict

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


class TaskRouter:
    """
    Learnable router that selects which parameter subset to use.

    For a given task, computes:
        p(k | task_embedding) = softmax(W_route @ task_embedding)
    """

    def __init__(
        self,
        num_tasks: int,
        embedding_dim: int,
        learning_rate: float = 0.01,
    ):
        """
        Initialize task router.

        Args:
            num_tasks: Number of task families
            embedding_dim: Dimension of task embeddings
            learning_rate: Learning rate
        """
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # Router weights: (num_tasks, embedding_dim)
        self.weights = np.random.randn(num_tasks, embedding_dim) * 0.01

        # Track routing decisions
        self.routing_history: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized TaskRouter with {num_tasks} tasks, "
            f"embedding_dim={embedding_dim}"
        )

    def route(
        self,
        task_embedding: np.ndarray,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute routing probabilities.

        Args:
            task_embedding: Task embedding vector
            temperature: Temperature for softmax

        Returns:
            Dictionary mapping task_id to probability
        """
        # Compute scores
        scores = self.weights @ task_embedding
        scores = scores / temperature

        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        return {f"task_{i}": float(p) for i, p in enumerate(probs)}

    def select_task(
        self,
        task_embedding: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select a task based on routing probabilities.

        Args:
            task_embedding: Task embedding
            deterministic: If True, select argmax; else sample

        Returns:
            Selected task index
        """
        probs = self.route(task_embedding)
        prob_values = np.array(list(probs.values()))

        if deterministic:
            return int(np.argmax(prob_values))
        else:
            return int(np.random.choice(len(prob_values), p=prob_values))

    def update(
        self,
        task_embedding: np.ndarray,
        correct_task: int,
        reward: float,
    ) -> Dict[str, Any]:
        """
        Update router weights.

        Uses cross-entropy loss for supervised routing:
            L = -log p(correct_task | task_embedding)

        Or REINFORCE for reward-based routing:
            ∇W = α * reward * ∇log p(selected_task | task_embedding)

        Args:
            task_embedding: Task embedding
            correct_task: Ground truth task index
            reward: Reward signal

        Returns:
            Update statistics
        """
        probs = self.route(task_embedding)
        prob_values = np.array(list(probs.values()))

        # Compute gradient
        # For cross-entropy: ∇W_k = (1[k=correct] - p(k)) * task_embedding
        grad = np.zeros_like(self.weights)
        for k in range(self.num_tasks):
            if k == correct_task:
                grad[k] = (1 - prob_values[k]) * task_embedding
            else:
                grad[k] = -prob_values[k] * task_embedding

        # Weight by reward
        grad = reward * grad

        # Update
        self.weights += self.learning_rate * grad

        return {
            "updated": True,
            "correct_task": correct_task,
            "routing_prob": float(prob_values[correct_task]),
            "gradient_norm": float(np.linalg.norm(grad)),
        }


class ParameterIsolationAgent(ParametricCLAgent):
    """
    Parameter Isolation Continual Learning Agent.

    Key features:
    1. Maintains separate tool scorer weights for each task family
    2. Shares state embeddings but isolates parameters
    3. Uses learnable router to select parameter subset
    4. Prevents catastrophic forgetting through isolation

    Architecture:
        w = [w_shared, w_task1, w_task2, ..., w_taskN]

    When processing:
        1. Compute task embedding
        2. Route to appropriate parameter subset
        3. Use selected parameters for tool scoring
        4. Update only the selected parameters
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
        # Parameter isolation specific
        num_task_families: int = 5,
        use_shared_parameters: bool = True,
        shared_weight: float = 0.3,
        router_learning_rate: float = 0.01,
        auto_create_tasks: bool = True,
        task_similarity_threshold: float = 0.7,
        cl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Parameter Isolation agent.

        Args:
            tools: Available tools
            domain_policy: Domain policy
            llm: LLM model name
            llm_args: LLM arguments
            parametric_memory: Memory system
            embedding_dim: Embedding dimension
            learning_rate: Learning rate
            max_examples_in_prompt: Max examples in prompt
            num_task_families: Number of task families to allocate
            use_shared_parameters: Whether to use shared parameters
            shared_weight: Weight for shared vs task-specific parameters
            router_learning_rate: Learning rate for router
            auto_create_tasks: Automatically create new task families
            task_similarity_threshold: Threshold for task assignment
            cl_config: Additional config
        """
        # Don't pass tool_scorer to parent - we'll manage multiple scorers
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            tool_scorer=None,  # We'll create multiple
            parametric_memory=parametric_memory,
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            max_examples_in_prompt=max_examples_in_prompt,
            use_tool_scorer_for_selection=True,
            cl_config=cl_config,
        )

        self.num_task_families = num_task_families
        self.use_shared_parameters = use_shared_parameters
        self.shared_weight = shared_weight
        self.auto_create_tasks = auto_create_tasks
        self.task_similarity_threshold = task_similarity_threshold

        # Create shared tool scorer
        if use_shared_parameters:
            self.shared_scorer = ToolScorer(
                tools=tools,
                embedding_dim=embedding_dim,
                learning_rate=learning_rate,
            )
        else:
            self.shared_scorer = None

        # Create task-specific tool scorers
        self.task_scorers: Dict[int, ToolScorer] = {}
        for i in range(num_task_families):
            self.task_scorers[i] = ToolScorer(
                tools=tools,
                embedding_dim=embedding_dim,
                learning_rate=learning_rate,
            )

        # Create task router
        self.router = TaskRouter(
            num_tasks=num_task_families,
            embedding_dim=embedding_dim,
            learning_rate=router_learning_rate,
        )

        # Track domain to task family mapping
        self.domain_to_task: Dict[str, int] = {}
        self.task_to_domains: Dict[int, List[str]] = defaultdict(list)
        self.next_task_id = 0

        # Statistics
        self.routing_correct = 0
        self.routing_total = 0

        # Override parent's tool_scorer with a property
        self._current_task_id_for_routing: Optional[int] = None

        logger.info(
            f"Initialized ParameterIsolationAgent with "
            f"{num_task_families} task families, "
            f"shared={use_shared_parameters}, "
            f"auto_create={auto_create_tasks}"
        )

    @property
    def tool_scorer(self) -> ToolScorer:
        """Get current active tool scorer based on routing."""
        if self._current_task_id_for_routing is not None:
            return self.task_scorers[self._current_task_id_for_routing]
        else:
            # Default to shared or first task
            return self.shared_scorer if self.shared_scorer else self.task_scorers[0]

    @tool_scorer.setter
    def tool_scorer(self, value):
        """Ignore setter - we manage scorers internally."""
        pass

    def _compute_task_embedding(
        self,
        domain: str,
        state_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Compute task embedding from domain and state.

        Args:
            domain: Domain name
            state_embedding: Current state embedding

        Returns:
            Task embedding
        """
        # Simple approach: use state embedding + domain hash
        domain_hash = hash(domain) % 1000 / 1000.0

        # Create task embedding
        task_emb = state_embedding.copy()
        task_emb[0] = domain_hash  # Inject domain info

        return task_emb

    def _assign_task_family(
        self,
        domain: str,
        task_embedding: np.ndarray,
    ) -> int:
        """
        Assign a task family for the given domain.

        Args:
            domain: Domain name
            task_embedding: Task embedding

        Returns:
            Task family ID
        """
        # Check if domain already assigned
        if domain in self.domain_to_task:
            return self.domain_to_task[domain]

        # Auto-create new task family or use router
        if self.auto_create_tasks and self.next_task_id < self.num_task_families:
            # Assign new task family
            task_id = self.next_task_id
            self.next_task_id += 1

            self.domain_to_task[domain] = task_id
            self.task_to_domains[task_id].append(domain)

            logger.info(f"Assigned domain '{domain}' to new task family {task_id}")
            return task_id

        else:
            # Use router to select most similar existing task
            task_id = self.router.select_task(task_embedding, deterministic=True)

            self.domain_to_task[domain] = task_id
            self.task_to_domains[task_id].append(domain)

            logger.info(f"Assigned domain '{domain}' to existing task family {task_id}")
            return task_id

    def _blend_scores(
        self,
        shared_scores: Optional[Dict[str, float]],
        task_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Blend shared and task-specific scores.

        Args:
            shared_scores: Scores from shared parameters
            task_scores: Scores from task-specific parameters

        Returns:
            Blended scores
        """
        if shared_scores is None:
            return task_scores

        blended = {}
        for tool_name in task_scores.keys():
            shared_score = shared_scores.get(tool_name, 0.0)
            task_score = task_scores[tool_name]

            # Weighted combination
            blended[tool_name] = (
                self.shared_weight * shared_score +
                (1 - self.shared_weight) * task_score
            )

        return blended

    def _select_tool_with_scorer(
        self,
        state_embedding: np.ndarray,
        available_tools: Optional[List[str]] = None,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Get tool scores using parameter isolation.

        Args:
            state_embedding: Current state embedding
            available_tools: Optional list of available tools
            temperature: Sampling temperature

        Returns:
            Dictionary mapping tool names to probabilities
        """
        # Compute task embedding
        domain = self._current_domain or "default"
        task_embedding = self._compute_task_embedding(domain, state_embedding)

        # Route to task family
        task_id = self._assign_task_family(domain, task_embedding)
        self._current_task_id_for_routing = task_id

        # Get task-specific scores
        task_scorer = self.task_scorers[task_id]
        task_scores = task_scorer.get_tool_probabilities(
            state_embedding,
            tool_names=available_tools,
            temperature=temperature,
        )

        # Get shared scores if enabled
        if self.use_shared_parameters:
            shared_scores = self.shared_scorer.get_tool_probabilities(
                state_embedding,
                tool_names=available_tools,
                temperature=temperature,
            )

            # Blend scores
            final_scores = self._blend_scores(shared_scores, task_scores)
        else:
            final_scores = task_scores

        logger.debug(
            f"Routed to task family {task_id} for domain '{domain}', "
            f"using {'blended' if self.use_shared_parameters else 'isolated'} parameters"
        )

        return final_scores

    def _update_parameters(
        self,
        state_embedding: np.ndarray,
        selected_tool: str,
        reward: float,
        success: bool,
    ) -> Dict[str, Any]:
        """
        Update parameters with isolation.

        Only updates:
        1. The selected task-specific parameters
        2. Optionally the shared parameters

        Does NOT update other task parameters (isolation).

        Args:
            state_embedding: State embedding
            selected_tool: Selected tool
            reward: Reward signal
            success: Whether succeeded

        Returns:
            Update statistics
        """
        domain = self._current_domain or "default"
        task_embedding = self._compute_task_embedding(domain, state_embedding)
        task_id = self._assign_task_family(domain, task_embedding)

        # Update task-specific scorer
        task_scorer = self.task_scorers[task_id]
        task_stats = task_scorer.update_weights(
            state_embedding=state_embedding,
            selected_tool=selected_tool,
            reward=reward,
            success=success,
        )

        # Update shared scorer if enabled
        shared_stats = None
        if self.use_shared_parameters:
            shared_stats = self.shared_scorer.update_weights(
                state_embedding=state_embedding,
                selected_tool=selected_tool,
                reward=reward * self.shared_weight,  # Weighted update
                success=success,
            )

        # Update router
        router_stats = self.router.update(
            task_embedding=task_embedding,
            correct_task=task_id,
            reward=reward,
        )

        if router_stats["routing_prob"] > 0.5:
            self.routing_correct += 1
        self.routing_total += 1

        return {
            "updated": True,
            "task_id": task_id,
            "domain": domain,
            "task_specific_update": task_stats,
            "shared_update": shared_stats,
            "router_update": router_stats,
            "routing_accuracy": self.routing_correct / max(1, self.routing_total),
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
        """Get statistics including routing info."""
        stats = super().get_statistics()

        # Task family statistics
        task_stats = {}
        for task_id, scorer in self.task_scorers.items():
            task_stats[f"task_{task_id}"] = scorer.get_statistics()

        stats.update({
            "num_task_families": self.num_task_families,
            "num_assigned_tasks": len(self.domain_to_task),
            "routing_accuracy": self.routing_correct / max(1, self.routing_total),
            "domain_to_task_mapping": dict(self.domain_to_task),
            "task_family_stats": task_stats,
        })

        if self.shared_scorer:
            stats["shared_scorer_stats"] = self.shared_scorer.get_statistics()

        return stats

    def save_state(self, path: str) -> None:
        """Save parameter isolation agent state."""
        super().save_state(path)

        # Save all task scorers
        import pickle
        scorers_path = path.replace('.json', '_task_scorers.pkl')
        with open(scorers_path, 'wb') as f:
            pickle.dump({
                "task_scorers": {k: v.get_parameters() for k, v in self.task_scorers.items()},
                "shared_scorer": self.shared_scorer.get_parameters() if self.shared_scorer else None,
                "router_weights": self.router.weights,
                "domain_to_task": dict(self.domain_to_task),
            }, f)

        logger.info(f"Saved parameter isolation state to {scorers_path}")

    def load_state(self, path: str) -> None:
        """Load parameter isolation agent state."""
        super().load_state(path)

        # Load all task scorers
        import pickle
        scorers_path = path.replace('.json', '_task_scorers.pkl')
        try:
            with open(scorers_path, 'rb') as f:
                data = pickle.load(f)

            for task_id, params in data["task_scorers"].items():
                self.task_scorers[task_id].set_parameters(params)

            if data["shared_scorer"] and self.shared_scorer:
                self.shared_scorer.set_parameters(data["shared_scorer"])

            self.router.weights = data["router_weights"]
            self.domain_to_task = data["domain_to_task"]

            logger.info(f"Loaded parameter isolation state from {scorers_path}")
        except FileNotFoundError:
            logger.warning(f"Task scorers file not found: {scorers_path}")


def create_parameter_isolation_agent(
    tools: List[Tool],
    domain_policy: str,
    llm: str = "gpt-4",
    embedding_dim: int = 768,
    learning_rate: float = 0.01,
    num_task_families: int = 5,
    use_shared_parameters: bool = True,
    **kwargs
) -> ParameterIsolationAgent:
    """
    Factory function to create a Parameter Isolation agent.

    Args:
        tools: Available tools
        domain_policy: Domain policy
        llm: LLM model name
        embedding_dim: Embedding dimension
        learning_rate: Learning rate
        num_task_families: Number of task families
        use_shared_parameters: Whether to use shared parameters
        **kwargs: Additional arguments

    Returns:
        Configured ParameterIsolationAgent
    """
    return ParameterIsolationAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        num_task_families=num_task_families,
        use_shared_parameters=use_shared_parameters,
        **kwargs
    )
