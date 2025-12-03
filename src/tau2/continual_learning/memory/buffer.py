# Copyright Sierra
"""
Memory Buffer for Continual Learning

This module provides the experience buffer system for storing and retrieving
past experiences for in-context learning based continual learning.
"""

import random
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import numpy as np
from pydantic import BaseModel, Field
from loguru import logger

from tau2.data_model.message import Message, AssistantMessage, ToolCall


class SamplingStrategy(str, Enum):
    """Sampling strategy for experience replay"""
    UNIFORM = "uniform"           # Random uniform sampling
    REWARD_WEIGHTED = "reward"    # Sample proportional to reward
    RECENCY_WEIGHTED = "recency"  # Prefer recent experiences
    DIVERSITY = "diversity"       # Maximize diversity in samples
    TASK_BALANCED = "balanced"    # Balance across tasks
    SIMILARITY = "similarity"     # Sample similar experiences


class Experience(BaseModel):
    """
    A single experience/trajectory step stored in the memory buffer.

    Experiences capture successful tool use interactions that can be
    retrieved and used as few-shot examples for in-context learning.
    """

    experience_id: str = Field(description="Unique experience identifier")
    task_id: str = Field(description="Source task ID")
    domain: str = Field(description="Domain name")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Interaction data
    observation: str = Field(description="The observation/context")
    action: str = Field(description="The action taken (assistant message)")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tool calls made in this step"
    )
    tool_results: Optional[List[str]] = Field(
        default=None,
        description="Results from tool calls"
    )

    # Evaluation
    reward: float = Field(default=0.0, description="Reward for this step")
    success: bool = Field(default=False, description="Whether the task succeeded")

    # Metadata for retrieval
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Embedding vector for similarity search"
    )
    required_tools: List[str] = Field(
        default_factory=list,
        description="Tools used in this experience"
    )

    class Config:
        arbitrary_types_allowed = True


class MemoryBuffer:
    """
    Experience buffer for storing and retrieving past experiences.

    This buffer is the core component of ICL-based continual learning,
    providing efficient storage and retrieval of successful experiences
    that can be used as few-shot examples.
    """

    def __init__(
        self,
        max_size: int = 1000,
        sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        min_reward_threshold: float = 0.5,
        diversity_weight: float = 0.3,
        enable_embeddings: bool = False,
        embedding_model: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the memory buffer.

        Args:
            max_size: Maximum number of experiences to store
            sampling_strategy: Strategy for sampling experiences
            min_reward_threshold: Minimum reward to store an experience
            diversity_weight: Weight for diversity in sampling
            enable_embeddings: Whether to compute embeddings for retrieval
            embedding_model: Model to use for embeddings
            seed: Random seed
        """
        self.max_size = max_size
        self.sampling_strategy = sampling_strategy
        self.min_reward_threshold = min_reward_threshold
        self.diversity_weight = diversity_weight
        self.enable_embeddings = enable_embeddings
        self.embedding_model = embedding_model
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Storage
        self._experiences: List[Experience] = []
        self._experience_lookup: Dict[str, Experience] = {}

        # Index structures
        self._task_index: Dict[str, List[str]] = {}  # task_id -> experience_ids
        self._domain_index: Dict[str, List[str]] = {}  # domain -> experience_ids
        self._tool_index: Dict[str, List[str]] = {}  # tool_name -> experience_ids

        # Embedding index (for similarity search)
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_client = None

        logger.info(
            f"Initialized MemoryBuffer with max_size={max_size}, "
            f"strategy={sampling_strategy.value}"
        )

    def add(
        self,
        experience: Experience,
        compute_embedding: bool = True
    ) -> bool:
        """
        Add an experience to the buffer.

        Args:
            experience: The experience to add
            compute_embedding: Whether to compute embedding

        Returns:
            True if experience was added, False if filtered out
        """
        # Filter by reward threshold
        if experience.reward < self.min_reward_threshold:
            logger.debug(
                f"Experience {experience.experience_id} filtered out "
                f"(reward={experience.reward} < {self.min_reward_threshold})"
            )
            return False

        # Compute embedding if enabled
        if self.enable_embeddings and compute_embedding and experience.embedding is None:
            experience.embedding = self._compute_embedding(experience.observation)

        # Check capacity
        if len(self._experiences) >= self.max_size:
            self._evict_experience()

        # Add to storage
        self._experiences.append(experience)
        self._experience_lookup[experience.experience_id] = experience

        # Update indices
        self._update_indices(experience)

        logger.debug(f"Added experience {experience.experience_id}")
        return True

    def add_from_trajectory(
        self,
        task_id: str,
        domain: str,
        messages: List[Message],
        reward: float,
        success: bool = False,
    ) -> int:
        """
        Add experiences from a complete trajectory.

        Args:
            task_id: The task ID
            domain: The domain name
            messages: List of messages in the trajectory
            reward: Final reward for the trajectory
            success: Whether the task succeeded

        Returns:
            Number of experiences added
        """
        added_count = 0

        # Build observation-action pairs
        observation = ""
        for i, msg in enumerate(messages):
            if isinstance(msg, AssistantMessage):
                # Create experience from this assistant message
                experience = Experience(
                    experience_id=f"{task_id}_{i}_{datetime.now().timestamp()}",
                    task_id=task_id,
                    domain=domain,
                    observation=observation,
                    action=msg.content or "",
                    tool_calls=[tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else None,
                    reward=reward,
                    success=success,
                    required_tools=[tc.name for tc in msg.tool_calls] if msg.tool_calls else [],
                )

                if self.add(experience):
                    added_count += 1

                # Update observation with assistant response
                if msg.content:
                    observation += f"\nAssistant: {msg.content}"
            else:
                # Add to observation
                content = getattr(msg, 'content', str(msg))
                role = getattr(msg, 'role', 'unknown')
                observation += f"\n{role.capitalize()}: {content}"

        logger.info(
            f"Added {added_count} experiences from trajectory "
            f"(task={task_id}, reward={reward})"
        )
        return added_count

    def sample(
        self,
        n: int = 5,
        strategy: Optional[SamplingStrategy] = None,
        query: Optional[str] = None,
        domain: Optional[str] = None,
        task_id: Optional[str] = None,
        exclude_task_id: Optional[str] = None,
        required_tools: Optional[List[str]] = None,
    ) -> List[Experience]:
        """
        Sample experiences from the buffer.

        Args:
            n: Number of experiences to sample
            strategy: Sampling strategy (default: buffer's strategy)
            query: Query string for similarity sampling
            domain: Filter by domain
            task_id: Filter by task ID
            exclude_task_id: Exclude experiences from this task
            required_tools: Filter by required tools

        Returns:
            List of sampled experiences
        """
        if len(self._experiences) == 0:
            return []

        strategy = strategy or self.sampling_strategy

        # Get candidate experiences
        candidates = self._get_candidates(
            domain=domain,
            task_id=task_id,
            exclude_task_id=exclude_task_id,
            required_tools=required_tools,
        )

        if len(candidates) == 0:
            return []

        n = min(n, len(candidates))

        # Sample based on strategy
        if strategy == SamplingStrategy.UNIFORM:
            return self._uniform_sample(candidates, n)
        elif strategy == SamplingStrategy.REWARD_WEIGHTED:
            return self._reward_weighted_sample(candidates, n)
        elif strategy == SamplingStrategy.RECENCY_WEIGHTED:
            return self._recency_weighted_sample(candidates, n)
        elif strategy == SamplingStrategy.DIVERSITY:
            return self._diversity_sample(candidates, n)
        elif strategy == SamplingStrategy.TASK_BALANCED:
            return self._task_balanced_sample(candidates, n)
        elif strategy == SamplingStrategy.SIMILARITY:
            return self._similarity_sample(candidates, n, query)
        else:
            return self._uniform_sample(candidates, n)

    def retrieve_similar(
        self,
        query: str,
        k: int = 5,
        domain: Optional[str] = None,
        similarity_threshold: float = 0.0,
    ) -> List[Experience]:
        """
        Retrieve experiences similar to the query.

        Args:
            query: Query string
            k: Number of experiences to retrieve
            domain: Filter by domain
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar experiences, sorted by similarity
        """
        if not self.enable_embeddings or len(self._experiences) == 0:
            # Fall back to uniform sampling
            return self.sample(n=k, domain=domain)

        # Compute query embedding
        query_embedding = self._compute_embedding(query)
        if query_embedding is None:
            return self.sample(n=k, domain=domain)

        # Get candidates
        candidates = self._get_candidates(domain=domain)
        if len(candidates) == 0:
            return []

        # Compute similarities
        similarities = []
        for exp in candidates:
            if exp.embedding is not None:
                sim = self._cosine_similarity(query_embedding, exp.embedding)
                if sim >= similarity_threshold:
                    similarities.append((exp, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return [exp for exp, _ in similarities[:k]]

    def _get_candidates(
        self,
        domain: Optional[str] = None,
        task_id: Optional[str] = None,
        exclude_task_id: Optional[str] = None,
        required_tools: Optional[List[str]] = None,
    ) -> List[Experience]:
        """Get candidate experiences based on filters."""
        candidates = self._experiences.copy()

        if domain is not None:
            exp_ids = self._domain_index.get(domain, [])
            candidates = [e for e in candidates if e.experience_id in exp_ids]

        if task_id is not None:
            exp_ids = self._task_index.get(task_id, [])
            candidates = [e for e in candidates if e.experience_id in exp_ids]

        if exclude_task_id is not None:
            candidates = [e for e in candidates if e.task_id != exclude_task_id]

        if required_tools is not None:
            candidates = [
                e for e in candidates
                if any(t in e.required_tools for t in required_tools)
            ]

        return candidates

    def _uniform_sample(
        self,
        candidates: List[Experience],
        n: int
    ) -> List[Experience]:
        """Uniform random sampling."""
        return random.sample(candidates, min(n, len(candidates)))

    def _reward_weighted_sample(
        self,
        candidates: List[Experience],
        n: int
    ) -> List[Experience]:
        """Sample proportional to reward."""
        rewards = np.array([e.reward for e in candidates])
        rewards = rewards - rewards.min() + 0.1  # Ensure positive
        probs = rewards / rewards.sum()

        indices = np.random.choice(
            len(candidates),
            size=min(n, len(candidates)),
            replace=False,
            p=probs
        )
        return [candidates[i] for i in indices]

    def _recency_weighted_sample(
        self,
        candidates: List[Experience],
        n: int
    ) -> List[Experience]:
        """Sample with preference for recent experiences."""
        # Compute recency weights (more recent = higher weight)
        now = datetime.now()
        ages = [(now - e.timestamp).total_seconds() for e in candidates]
        max_age = max(ages) + 1
        weights = np.array([max_age - age for age in ages])
        probs = weights / weights.sum()

        indices = np.random.choice(
            len(candidates),
            size=min(n, len(candidates)),
            replace=False,
            p=probs
        )
        return [candidates[i] for i in indices]

    def _diversity_sample(
        self,
        candidates: List[Experience],
        n: int
    ) -> List[Experience]:
        """Sample to maximize diversity (different domains/tasks)."""
        selected = []
        seen_domains = set()
        seen_tasks = set()

        # Shuffle candidates
        shuffled = candidates.copy()
        random.shuffle(shuffled)

        # First pass: prioritize unseen domains/tasks
        for exp in shuffled:
            if len(selected) >= n:
                break
            if exp.domain not in seen_domains or exp.task_id not in seen_tasks:
                selected.append(exp)
                seen_domains.add(exp.domain)
                seen_tasks.add(exp.task_id)

        # Second pass: fill remaining slots
        for exp in shuffled:
            if len(selected) >= n:
                break
            if exp not in selected:
                selected.append(exp)

        return selected[:n]

    def _task_balanced_sample(
        self,
        candidates: List[Experience],
        n: int
    ) -> List[Experience]:
        """Sample balanced across tasks."""
        # Group by task
        task_groups: Dict[str, List[Experience]] = {}
        for exp in candidates:
            if exp.task_id not in task_groups:
                task_groups[exp.task_id] = []
            task_groups[exp.task_id].append(exp)

        # Sample from each task in round-robin
        selected = []
        tasks = list(task_groups.keys())
        random.shuffle(tasks)
        task_indices = {t: 0 for t in tasks}

        while len(selected) < n:
            added = False
            for task in tasks:
                if len(selected) >= n:
                    break
                if task_indices[task] < len(task_groups[task]):
                    selected.append(task_groups[task][task_indices[task]])
                    task_indices[task] += 1
                    added = True
            if not added:
                break

        return selected

    def _similarity_sample(
        self,
        candidates: List[Experience],
        n: int,
        query: Optional[str] = None
    ) -> List[Experience]:
        """Sample based on similarity to query."""
        if query is None or not self.enable_embeddings:
            return self._uniform_sample(candidates, n)

        return self.retrieve_similar(query, k=n)

    def _evict_experience(self) -> None:
        """Evict an experience to make room for new ones."""
        if len(self._experiences) == 0:
            return

        # Eviction strategy: remove lowest reward experience
        min_idx = 0
        min_reward = self._experiences[0].reward
        for i, exp in enumerate(self._experiences):
            if exp.reward < min_reward:
                min_reward = exp.reward
                min_idx = i

        evicted = self._experiences.pop(min_idx)
        del self._experience_lookup[evicted.experience_id]
        self._remove_from_indices(evicted)

        logger.debug(f"Evicted experience {evicted.experience_id}")

    def _update_indices(self, experience: Experience) -> None:
        """Update index structures when adding an experience."""
        exp_id = experience.experience_id

        # Task index
        if experience.task_id not in self._task_index:
            self._task_index[experience.task_id] = []
        self._task_index[experience.task_id].append(exp_id)

        # Domain index
        if experience.domain not in self._domain_index:
            self._domain_index[experience.domain] = []
        self._domain_index[experience.domain].append(exp_id)

        # Tool index
        for tool in experience.required_tools:
            if tool not in self._tool_index:
                self._tool_index[tool] = []
            self._tool_index[tool].append(exp_id)

    def _remove_from_indices(self, experience: Experience) -> None:
        """Remove an experience from index structures."""
        exp_id = experience.experience_id

        # Task index
        if experience.task_id in self._task_index:
            self._task_index[experience.task_id] = [
                e for e in self._task_index[experience.task_id]
                if e != exp_id
            ]

        # Domain index
        if experience.domain in self._domain_index:
            self._domain_index[experience.domain] = [
                e for e in self._domain_index[experience.domain]
                if e != exp_id
            ]

        # Tool index
        for tool in experience.required_tools:
            if tool in self._tool_index:
                self._tool_index[tool] = [
                    e for e in self._tool_index[tool]
                    if e != exp_id
                ]

    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for text."""
        if not self.enable_embeddings:
            return None

        try:
            # Try to use OpenAI embeddings
            if self._embedding_client is None:
                try:
                    import openai
                    self._embedding_client = openai.OpenAI()
                except Exception:
                    logger.warning("Could not initialize OpenAI client for embeddings")
                    return None

            response = self._embedding_client.embeddings.create(
                model=self.embedding_model or "text-embedding-3-small",
                input=text[:8000],  # Truncate if too long
            )
            return response.data[0].embedding

        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return None

    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "total_experiences": len(self._experiences),
            "max_size": self.max_size,
            "fill_ratio": len(self._experiences) / self.max_size,
            "num_tasks": len(self._task_index),
            "num_domains": len(self._domain_index),
            "num_tools": len(self._tool_index),
            "avg_reward": np.mean([e.reward for e in self._experiences]) if self._experiences else 0,
            "domains": list(self._domain_index.keys()),
            "experiences_per_domain": {
                d: len(ids) for d, ids in self._domain_index.items()
            },
        }

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self._experiences.clear()
        self._experience_lookup.clear()
        self._task_index.clear()
        self._domain_index.clear()
        self._tool_index.clear()
        self._embeddings = None
        logger.info("Cleared memory buffer")

    def save(self, path: str) -> None:
        """Save buffer to file."""
        import json
        data = {
            "experiences": [e.model_dump() for e in self._experiences],
            "max_size": self.max_size,
            "sampling_strategy": self.sampling_strategy.value,
            "min_reward_threshold": self.min_reward_threshold,
        }
        with open(path, 'w') as f:
            json.dump(data, f, default=str)
        logger.info(f"Saved {len(self._experiences)} experiences to {path}")

    def load(self, path: str) -> None:
        """Load buffer from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)

        self.clear()
        for exp_data in data["experiences"]:
            exp = Experience(**exp_data)
            self.add(exp, compute_embedding=False)

        logger.info(f"Loaded {len(self._experiences)} experiences from {path}")

    def __len__(self) -> int:
        """Return number of experiences in buffer."""
        return len(self._experiences)

    def __iter__(self):
        """Iterate over experiences."""
        return iter(self._experiences)
