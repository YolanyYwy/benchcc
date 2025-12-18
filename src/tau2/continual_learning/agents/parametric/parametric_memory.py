# Copyright Sierra
"""
Parametric Memory System - Learnable Memory with Importance Weights

This module implements a memory system with learnable importance weights
for each experience, enabling the agent to learn which memories are most useful.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
from loguru import logger

from tau2.continual_learning.memory.buffer import Experience, MemoryBuffer


class ParametricMemory(MemoryBuffer):
    """
    Memory system with learnable importance weights.

    Each experience m_i has:
        - z_i: trajectory embedding (fixed)
        - τ_i: timestamp (fixed)
        - α_i: importance weight (learnable)

    The importance weight α_i determines how much this memory
    contributes to the agent's decision-making.
    """

    def __init__(
        self,
        max_size: int = 1000,
        embedding_dim: int = 768,
        learning_rate: float = 0.01,
        initial_importance: float = 1.0,
        importance_decay: float = 0.0,
        **kwargs,
    ):
        """
        Initialize parametric memory.

        Args:
            max_size: Maximum number of experiences
            embedding_dim: Dimension of embeddings
            learning_rate: Learning rate for importance weights
            initial_importance: Initial value for importance weights
            importance_decay: Decay rate for importance over time
            **kwargs: Additional arguments for MemoryBuffer
        """
        super().__init__(max_size=max_size, **kwargs)

        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.initial_importance = initial_importance
        self.importance_decay = importance_decay

        # Learnable importance weights: α_i for each experience
        # Maps experience_id -> importance weight
        self._importance_weights: Dict[str, float] = {}

        # Track gradient statistics
        self.total_weight_updates = 0
        self.weight_update_history: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized ParametricMemory with learnable importance weights, "
            f"lr={learning_rate}, initial_α={initial_importance}"
        )

    def add(
        self,
        experience: Experience,
        compute_embedding: bool = True,
        initial_importance: Optional[float] = None,
    ) -> bool:
        """
        Add an experience with an importance weight.

        Args:
            experience: The experience to add
            compute_embedding: Whether to compute embedding
            initial_importance: Initial importance weight (default: self.initial_importance)

        Returns:
            True if added successfully
        """
        # Add to base buffer
        added = super().add(experience, compute_embedding)

        if added:
            # Initialize importance weight
            importance = initial_importance if initial_importance is not None else self.initial_importance
            self._importance_weights[experience.experience_id] = importance

        return added

    def get_importance(self, experience_id: str) -> float:
        """Get importance weight for an experience."""
        return self._importance_weights.get(experience_id, self.initial_importance)

    def set_importance(self, experience_id: str, importance: float) -> None:
        """Set importance weight for an experience."""
        if experience_id in self._experience_lookup:
            self._importance_weights[experience_id] = importance

    def update_importance(
        self,
        experience_id: str,
        gradient: float,
        clip_value: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Update importance weight using gradient.

        Args:
            experience_id: ID of the experience
            gradient: Gradient signal (e.g., from reward)
            clip_value: Maximum gradient magnitude

        Returns:
            Update statistics
        """
        if experience_id not in self._importance_weights:
            return {"updated": False, "reason": "not_found"}

        # Clip gradient
        gradient = np.clip(gradient, -clip_value, clip_value)

        # Get current importance
        old_importance = self._importance_weights[experience_id]

        # Update: α_i ← α_i + lr * gradient
        new_importance = old_importance + self.learning_rate * gradient

        # Ensure importance stays positive
        new_importance = max(0.01, new_importance)

        self._importance_weights[experience_id] = new_importance

        self.total_weight_updates += 1

        # Track statistics
        update_info = {
            "updated": True,
            "experience_id": experience_id,
            "old_importance": old_importance,
            "new_importance": new_importance,
            "gradient": gradient,
            "delta": new_importance - old_importance,
        }

        self.weight_update_history.append(update_info)

        # Keep history bounded
        if len(self.weight_update_history) > 1000:
            self.weight_update_history = self.weight_update_history[-1000:]

        logger.debug(
            f"Updated importance for {experience_id}: "
            f"{old_importance:.3f} -> {new_importance:.3f} (grad={gradient:.3f})"
        )

        return update_info

    def sample_by_importance(
        self,
        n: int = 5,
        domain: Optional[str] = None,
        task_id: Optional[str] = None,
        temperature: float = 1.0,
    ) -> List[Experience]:
        """
        Sample experiences weighted by importance.

        Args:
            n: Number of experiences to sample
            domain: Filter by domain
            task_id: Filter by task ID
            temperature: Temperature for softmax (higher = more uniform)

        Returns:
            List of sampled experiences
        """
        # Get candidates
        candidates = self._get_candidates(domain=domain, task_id=task_id)

        if len(candidates) == 0:
            return []

        n = min(n, len(candidates))

        # Get importance weights
        importances = np.array([
            self.get_importance(exp.experience_id)
            for exp in candidates
        ])

        # Apply temperature and softmax
        importances = importances / temperature
        exp_importances = np.exp(importances - importances.max())
        probs = exp_importances / exp_importances.sum()

        # Sample without replacement
        indices = np.random.choice(
            len(candidates),
            size=n,
            replace=False,
            p=probs
        )

        return [candidates[i] for i in indices]

    def retrieve_by_importance(
        self,
        k: int = 5,
        domain: Optional[str] = None,
        similarity_query: Optional[str] = None,
        similarity_weight: float = 0.5,
    ) -> List[Experience]:
        """
        Retrieve top-k experiences by importance, optionally combined with similarity.

        Args:
            k: Number of experiences to retrieve
            domain: Filter by domain
            similarity_query: Optional query for similarity-based retrieval
            similarity_weight: Weight for similarity vs importance (0 = only importance, 1 = only similarity)

        Returns:
            List of top-k experiences
        """
        # Get candidates
        candidates = self._get_candidates(domain=domain)

        if len(candidates) == 0:
            return []

        k = min(k, len(candidates))

        # Compute scores
        scores = []
        for exp in candidates:
            # Importance component
            importance = self.get_importance(exp.experience_id)
            score = importance

            # Optional similarity component
            if similarity_query and self.enable_embeddings and exp.embedding:
                query_emb = self._compute_embedding(similarity_query)
                if query_emb:
                    similarity = self._cosine_similarity(query_emb, exp.embedding)
                    # Combine importance and similarity
                    score = (
                        (1 - similarity_weight) * importance +
                        similarity_weight * similarity
                    )

            scores.append((exp, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return [exp for exp, _ in scores[:k]]

    def apply_importance_decay(self, decay_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Apply time-based decay to importance weights.

        Older memories gradually lose importance:
            α_i ← α_i * exp(-decay_rate * age_i)

        Args:
            decay_rate: Decay rate (default: self.importance_decay)

        Returns:
            Decay statistics
        """
        if decay_rate is None:
            decay_rate = self.importance_decay

        if decay_rate == 0:
            return {"decayed": False, "reason": "decay_disabled"}

        now = datetime.now()
        updated_count = 0

        for exp in self._experiences:
            age_seconds = (now - exp.timestamp).total_seconds()
            age_hours = age_seconds / 3600

            # Decay factor
            decay_factor = np.exp(-decay_rate * age_hours)

            # Update importance
            old_importance = self._importance_weights[exp.experience_id]
            new_importance = old_importance * decay_factor

            if new_importance < 0.01:
                new_importance = 0.01  # Minimum importance

            self._importance_weights[exp.experience_id] = new_importance

            if abs(new_importance - old_importance) > 0.001:
                updated_count += 1

        logger.info(
            f"Applied importance decay to {updated_count} experiences "
            f"(rate={decay_rate})"
        )

        return {
            "decayed": True,
            "updated_count": updated_count,
            "decay_rate": decay_rate,
        }

    def reinforce_successful_retrievals(
        self,
        retrieved_exp_ids: List[str],
        reward: float,
    ) -> Dict[str, Any]:
        """
        Reinforce importance weights of retrieved experiences based on outcome.

        If a retrieved memory led to success, increase its importance.
        If it led to failure, decrease its importance.

        Args:
            retrieved_exp_ids: List of experience IDs that were retrieved
            reward: Reward signal (0-1 typically)

        Returns:
            Update statistics
        """
        if not retrieved_exp_ids:
            return {"updated": False, "reason": "no_experiences"}

        # Gradient is proportional to reward
        gradient = reward - 0.5  # Center around 0.5

        updates = []
        for exp_id in retrieved_exp_ids:
            update_info = self.update_importance(exp_id, gradient)
            if update_info["updated"]:
                updates.append(update_info)

        logger.info(
            f"Reinforced {len(updates)} memories with reward={reward:.3f}"
        )

        return {
            "updated": True,
            "num_updates": len(updates),
            "reward": reward,
            "gradient": gradient,
        }

    def _evict_experience(self) -> None:
        """
        Evict an experience to make room.

        Eviction prioritizes:
        1. Low importance
        2. Low reward
        3. Older timestamp
        """
        if len(self._experiences) == 0:
            return

        # Compute eviction scores (lower = more likely to evict)
        scores = []
        for exp in self._experiences:
            importance = self.get_importance(exp.experience_id)
            age_hours = (datetime.now() - exp.timestamp).total_seconds() / 3600

            # Combined score: importance * reward - age_penalty
            score = importance * exp.reward - 0.01 * age_hours
            scores.append((exp, score))

        # Evict the experience with lowest score
        scores.sort(key=lambda x: x[1])
        evicted_exp = scores[0][0]

        # Remove from buffer
        self._experiences.remove(evicted_exp)
        del self._experience_lookup[evicted_exp.experience_id]
        del self._importance_weights[evicted_exp.experience_id]
        self._remove_from_indices(evicted_exp)

        logger.debug(
            f"Evicted experience {evicted_exp.experience_id} "
            f"(importance={self.get_importance(evicted_exp.experience_id):.3f}, "
            f"reward={evicted_exp.reward:.3f})"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics including importance weights."""
        base_stats = super().get_statistics()

        # Compute importance statistics
        importances = list(self._importance_weights.values())

        base_stats.update({
            "total_weight_updates": self.total_weight_updates,
            "importance_mean": np.mean(importances) if importances else 0,
            "importance_std": np.std(importances) if importances else 0,
            "importance_min": np.min(importances) if importances else 0,
            "importance_max": np.max(importances) if importances else 0,
        })

        return base_stats

    def get_parameters(self) -> Dict[str, Any]:
        """Get all learnable parameters."""
        return {
            "importance_weights": self._importance_weights.copy(),
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set all learnable parameters."""
        if "importance_weights" in params:
            self._importance_weights = params["importance_weights"].copy()

    def save(self, path: str) -> None:
        """Save memory with importance weights."""
        # Save base buffer
        super().save(path)

        # Save importance weights separately
        import json
        import_path = path.replace('.json', '_importance.json')
        with open(import_path, 'w') as f:
            json.dump({
                "importance_weights": self._importance_weights,
                "total_weight_updates": self.total_weight_updates,
                "learning_rate": self.learning_rate,
                "initial_importance": self.initial_importance,
            }, f)

        logger.info(f"Saved importance weights to {import_path}")

    def load(self, path: str) -> None:
        """Load memory with importance weights."""
        # Load base buffer
        super().load(path)

        # Load importance weights
        import json
        import_path = path.replace('.json', '_importance.json')
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)

            self._importance_weights = data["importance_weights"]
            self.total_weight_updates = data.get("total_weight_updates", 0)
            self.learning_rate = data.get("learning_rate", self.learning_rate)
            self.initial_importance = data.get("initial_importance", self.initial_importance)

            logger.info(f"Loaded importance weights from {import_path}")
        except FileNotFoundError:
            logger.warning(f"Importance weights file not found: {import_path}")
            # Initialize with default importance
            self._importance_weights = {
                exp.experience_id: self.initial_importance
                for exp in self._experiences
            }
