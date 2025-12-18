# Copyright Sierra
"""
Tool Scorer - Learnable Tool Selection Module

This module implements a parametric tool scorer that learns which tools
to use in which contexts through gradient-based updates.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger

from tau2.environment.tool import Tool


class ToolScorer:
    """
    Learnable tool scorer with explicit parameters.

    For each tool, maintains a weight vector w_i that scores
    how relevant the tool is given the current state φ(s):

        score(s, tool_i) = w_i^T φ(s)

    where φ(s) is a frozen LLM-generated state embedding.
    """

    def __init__(
        self,
        tools: List[Tool],
        embedding_dim: int = 768,
        learning_rate: float = 0.01,
        init_method: str = "uniform",
        seed: Optional[int] = None,
    ):
        """
        Initialize the tool scorer.

        Args:
            tools: List of available tools
            embedding_dim: Dimension of state embeddings
            learning_rate: Learning rate for parameter updates
            init_method: Weight initialization method ("uniform", "normal", "zeros")
            seed: Random seed for initialization
        """
        self.tools = tools
        self.tool_names = [tool.name for tool in tools]
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        if seed is not None:
            np.random.seed(seed)

        # Initialize learnable parameters: w_i for each tool
        # Shape: (num_tools, embedding_dim)
        self.weights = self._initialize_weights(init_method)

        # For EWC: store Fisher information matrix and optimal weights
        self.fisher_information = None
        self.optimal_weights = None

        # Track statistics
        self.total_updates = 0
        self.tool_selection_counts = {name: 0 for name in self.tool_names}

        logger.info(
            f"Initialized ToolScorer with {len(tools)} tools, "
            f"embedding_dim={embedding_dim}, lr={learning_rate}"
        )

    def _initialize_weights(self, method: str) -> np.ndarray:
        """Initialize weight vectors."""
        num_tools = len(self.tools)

        if method == "uniform":
            # Uniform initialization [-0.1, 0.1]
            return np.random.uniform(-0.1, 0.1, size=(num_tools, self.embedding_dim))
        elif method == "normal":
            # Normal initialization with small std
            return np.random.normal(0, 0.01, size=(num_tools, self.embedding_dim))
        elif method == "zeros":
            # Zero initialization
            return np.zeros((num_tools, self.embedding_dim))
        else:
            raise ValueError(f"Unknown init_method: {method}")

    def score_tools(
        self,
        state_embedding: np.ndarray,
        tool_names: Optional[List[str]] = None,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Score all tools given the current state.

        Args:
            state_embedding: State embedding φ(s) of shape (embedding_dim,)
            tool_names: Optional list of tools to score (default: all tools)
            temperature: Temperature for softmax scaling

        Returns:
            Dictionary mapping tool names to scores
        """
        if state_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"State embedding dimension {state_embedding.shape[0]} "
                f"does not match expected {self.embedding_dim}"
            )

        # Compute scores: w_i^T φ(s)
        scores = self.weights @ state_embedding  # Shape: (num_tools,)

        # Apply temperature
        scores = scores / temperature

        # Create score dictionary
        if tool_names is None:
            tool_names = self.tool_names

        score_dict = {}
        for i, name in enumerate(self.tool_names):
            if name in tool_names:
                score_dict[name] = float(scores[i])

        return score_dict

    def get_tool_probabilities(
        self,
        state_embedding: np.ndarray,
        tool_names: Optional[List[str]] = None,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Get softmax probabilities for tool selection.

        Args:
            state_embedding: State embedding φ(s)
            tool_names: Optional list of tools to consider
            temperature: Temperature for softmax

        Returns:
            Dictionary mapping tool names to probabilities
        """
        scores = self.score_tools(state_embedding, tool_names, temperature)

        # Apply softmax
        score_values = np.array(list(scores.values()))
        exp_scores = np.exp(score_values - np.max(score_values))  # Numerical stability
        probs = exp_scores / exp_scores.sum()

        return {name: float(prob) for name, prob in zip(scores.keys(), probs)}

    def update_weights(
        self,
        state_embedding: np.ndarray,
        selected_tool: str,
        reward: float,
        success: bool = False,
        regularization_weight: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Update tool scorer weights using policy gradient.

        Uses REINFORCE-style update:
            ∇w_i = α * (reward - baseline) * ∇log π(tool_i | s)

        Args:
            state_embedding: State embedding φ(s)
            selected_tool: The tool that was selected
            reward: Reward signal (typically 0 or 1)
            success: Whether the action succeeded
            regularization_weight: L2 regularization strength

        Returns:
            Dictionary with update statistics
        """
        if selected_tool not in self.tool_names:
            logger.warning(f"Unknown tool: {selected_tool}")
            return {"updated": False, "reason": "unknown_tool"}

        # Get tool index
        tool_idx = self.tool_names.index(selected_tool)

        # Compute current probabilities
        probs = self.get_tool_probabilities(state_embedding)
        current_prob = probs[selected_tool]

        # Compute gradient of log probability
        # For softmax: ∇log π(a|s) = φ(s) - E[φ(s)]
        # Simplified: ∇log π(a|s) ≈ φ(s) * (1 - π(a|s))

        # Advantage: use reward directly (could use baseline in future)
        advantage = reward

        # Gradient for selected tool
        grad = advantage * state_embedding * (1 - current_prob)

        # Update weights
        self.weights[tool_idx] += self.learning_rate * grad

        # Apply L2 regularization (weight decay)
        if regularization_weight > 0:
            self.weights[tool_idx] -= self.learning_rate * regularization_weight * self.weights[tool_idx]

        self.total_updates += 1
        self.tool_selection_counts[selected_tool] += 1

        logger.debug(
            f"Updated weights for {selected_tool}: "
            f"reward={reward:.3f}, prob={current_prob:.3f}"
        )

        return {
            "updated": True,
            "tool": selected_tool,
            "reward": reward,
            "probability": current_prob,
            "gradient_norm": float(np.linalg.norm(grad)),
        }

    def compute_fisher_information(
        self,
        state_embeddings: List[np.ndarray],
        selected_tools: List[str],
    ) -> np.ndarray:
        """
        Compute Fisher information matrix for EWC.

        F_i = E[(∂log π(a|s) / ∂θ_i)^2]

        Args:
            state_embeddings: List of state embeddings
            selected_tools: List of selected tools

        Returns:
            Fisher information matrix of shape (num_tools, embedding_dim)
        """
        fisher = np.zeros_like(self.weights)

        for state_emb, tool in zip(state_embeddings, selected_tools):
            if tool not in self.tool_names:
                continue

            tool_idx = self.tool_names.index(tool)
            probs = self.get_tool_probabilities(state_emb)
            prob = probs[tool]

            # Gradient of log probability
            grad_log_prob = state_emb * (1 - prob)

            # Fisher: E[grad^2]
            fisher[tool_idx] += grad_log_prob ** 2

        # Average over samples
        fisher /= len(state_embeddings)

        self.fisher_information = fisher
        self.optimal_weights = self.weights.copy()

        logger.info(
            f"Computed Fisher information from {len(state_embeddings)} samples, "
            f"mean Fisher value: {fisher.mean():.6f}"
        )

        return fisher

    def get_ewc_regularization_loss(self, ewc_lambda: float = 1.0) -> float:
        """
        Compute EWC regularization loss.

        L_ewc = (λ/2) * Σ_i F_i * (θ_i - θ_i*)^2

        Args:
            ewc_lambda: EWC regularization strength

        Returns:
            EWC loss value
        """
        if self.fisher_information is None or self.optimal_weights is None:
            return 0.0

        diff = self.weights - self.optimal_weights
        loss = 0.5 * ewc_lambda * np.sum(self.fisher_information * (diff ** 2))

        return float(loss)

    def apply_ewc_penalty(
        self,
        tool_idx: int,
        gradient: np.ndarray,
        ewc_lambda: float = 1.0,
    ) -> np.ndarray:
        """
        Apply EWC penalty to gradient.

        Args:
            tool_idx: Index of the tool being updated
            gradient: Original gradient
            ewc_lambda: EWC strength

        Returns:
            Modified gradient with EWC penalty
        """
        if self.fisher_information is None or self.optimal_weights is None:
            return gradient

        # EWC penalty: F_i * (θ_i - θ_i*)
        penalty = (
            ewc_lambda *
            self.fisher_information[tool_idx] *
            (self.weights[tool_idx] - self.optimal_weights[tool_idx])
        )

        return gradient - penalty

    def get_parameters(self) -> Dict[str, Any]:
        """Get scorer parameters."""
        return {
            "weights": self.weights.copy(),
            "fisher_information": self.fisher_information.copy() if self.fisher_information is not None else None,
            "optimal_weights": self.optimal_weights.copy() if self.optimal_weights is not None else None,
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set scorer parameters."""
        if "weights" in params:
            self.weights = params["weights"].copy()
        if "fisher_information" in params and params["fisher_information"] is not None:
            self.fisher_information = params["fisher_information"].copy()
        if "optimal_weights" in params and params["optimal_weights"] is not None:
            self.optimal_weights = params["optimal_weights"].copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get scorer statistics."""
        return {
            "total_updates": self.total_updates,
            "tool_selection_counts": self.tool_selection_counts.copy(),
            "weights_norm": float(np.linalg.norm(self.weights)),
            "weights_mean": float(self.weights.mean()),
            "weights_std": float(self.weights.std()),
            "has_fisher": self.fisher_information is not None,
        }

    def save(self, path: str) -> None:
        """Save scorer state to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                "weights": self.weights,
                "fisher_information": self.fisher_information,
                "optimal_weights": self.optimal_weights,
                "tool_names": self.tool_names,
                "embedding_dim": self.embedding_dim,
                "learning_rate": self.learning_rate,
                "total_updates": self.total_updates,
                "tool_selection_counts": self.tool_selection_counts,
            }, f)
        logger.info(f"Saved ToolScorer to {path}")

    def load(self, path: str) -> None:
        """Load scorer state from file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.weights = data["weights"]
        self.fisher_information = data.get("fisher_information")
        self.optimal_weights = data.get("optimal_weights")
        self.tool_names = data["tool_names"]
        self.embedding_dim = data["embedding_dim"]
        self.learning_rate = data["learning_rate"]
        self.total_updates = data.get("total_updates", 0)
        self.tool_selection_counts = data.get("tool_selection_counts", {})

        logger.info(f"Loaded ToolScorer from {path}")
