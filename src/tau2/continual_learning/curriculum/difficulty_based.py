# Copyright Sierra
"""
Difficulty-Based Curriculum Strategy

Orders tasks by difficulty, typically from easy to hard (curriculum learning)
or hard to easy (anti-curriculum).
"""

import random
from typing import Any, Dict, List, Optional

from tau2.data_model.tasks import Task

from tau2.continual_learning.curriculum.base import (
    CurriculumManager,
    CurriculumStrategy,
)


class DifficultyBasedCurriculum(CurriculumManager):
    """
    Difficulty-based curriculum: order tasks by difficulty.

    This strategy implements curriculum learning where tasks are presented
    in order of increasing difficulty, which can help learning efficiency.
    """

    def __init__(
        self,
        tasks: List[Task],
        progression: str = "easy_to_hard",
        difficulty_metric: str = "action_count",
        group_by_domain: bool = False,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize difficulty-based curriculum.

        Args:
            tasks: List of tasks
            progression: Difficulty progression order
                - "easy_to_hard": Standard curriculum learning
                - "hard_to_easy": Anti-curriculum
                - "mixed": Alternating difficulty
            difficulty_metric: How to measure difficulty
                - "action_count": Number of required actions
                - "tool_count": Number of unique tools needed
                - "estimated": Use pre-computed difficulty score
            group_by_domain: Whether to apply difficulty ordering within
                           each domain separately
            config: Additional configuration
            seed: Random seed
        """
        self.progression = progression
        self.difficulty_metric = difficulty_metric
        self.group_by_domain = group_by_domain
        super().__init__(
            tasks=tasks,
            strategy=CurriculumStrategy.DIFFICULTY_BASED,
            config=config,
            seed=seed,
        )

    def _compute_difficulty(self, task_id: str) -> float:
        """
        Compute difficulty score for a task.

        Args:
            task_id: The task ID

        Returns:
            Difficulty score (higher = harder)
        """
        metadata = self.task_metadata[task_id]

        if self.difficulty_metric == "action_count":
            return metadata.num_agent_actions + metadata.num_user_actions
        elif self.difficulty_metric == "tool_count":
            return len(metadata.required_tools)
        elif self.difficulty_metric == "estimated":
            return metadata.difficulty or 0.5
        else:
            return metadata.difficulty or 0.5

    def _generate_curriculum(self) -> List[str]:
        """Generate difficulty-ordered curriculum."""
        if self.seed is not None:
            random.seed(self.seed)

        if self.group_by_domain:
            return self._generate_by_domain()
        else:
            return self._generate_global()

    def _generate_global(self) -> List[str]:
        """Generate global difficulty ordering."""
        # Compute difficulties
        task_difficulties = [
            (task.id, self._compute_difficulty(task.id))
            for task in self.tasks
        ]

        # Sort by difficulty
        reverse = self.progression == "hard_to_easy"
        task_difficulties.sort(key=lambda x: x[1], reverse=reverse)

        if self.progression == "mixed":
            return self._mixed_ordering(task_difficulties)

        return [tid for tid, _ in task_difficulties]

    def _generate_by_domain(self) -> List[str]:
        """Generate difficulty ordering within each domain."""
        # Group by domain
        domain_tasks: Dict[str, List[tuple]] = {}
        for task in self.tasks:
            domain = self.task_metadata[task.id].domain
            if domain not in domain_tasks:
                domain_tasks[domain] = []
            difficulty = self._compute_difficulty(task.id)
            domain_tasks[domain].append((task.id, difficulty))

        # Sort within each domain
        reverse = self.progression == "hard_to_easy"
        for domain in domain_tasks:
            domain_tasks[domain].sort(key=lambda x: x[1], reverse=reverse)
            if self.progression == "mixed":
                domain_tasks[domain] = self._mixed_ordering(domain_tasks[domain])
            else:
                domain_tasks[domain] = [tid for tid, _ in domain_tasks[domain]]

        # Concatenate domains (alphabetical order)
        curriculum = []
        for domain in sorted(domain_tasks.keys()):
            curriculum.extend(domain_tasks[domain])

        return curriculum

    def _mixed_ordering(
        self,
        task_difficulties: List[tuple]
    ) -> List[str]:
        """Create mixed difficulty ordering (alternating easy/hard)."""
        sorted_tasks = sorted(task_difficulties, key=lambda x: x[1])
        result = []
        left, right = 0, len(sorted_tasks) - 1

        while left <= right:
            if left == right:
                result.append(sorted_tasks[left][0])
            else:
                # Alternate between easy and hard
                if len(result) % 2 == 0:
                    result.append(sorted_tasks[left][0])
                    left += 1
                else:
                    result.append(sorted_tasks[right][0])
                    right -= 1

        return result
