# Copyright Sierra
"""
Interleaved Curriculum Strategy

Alternates between domains to reduce forgetting through natural
replay of previously seen domain types.
"""

import random
from typing import Any, Dict, List, Optional

from tau2.data_model.tasks import Task

from tau2.continual_learning.curriculum.base import (
    CurriculumManager,
    CurriculumStrategy,
)


class InterleavedCurriculum(CurriculumManager):
    """
    Interleaved curriculum: mix tasks from different domains.

    This strategy alternates between domains, which can help reduce
    catastrophic forgetting by providing natural replay opportunities.
    """

    def __init__(
        self,
        tasks: List[Task],
        interleave_pattern: Optional[str] = "round_robin",
        replay_ratio: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize interleaved curriculum.

        Args:
            tasks: List of tasks
            interleave_pattern: Pattern for interleaving
                - "round_robin": Cycle through domains
                - "random": Random domain selection
                - "weighted": Weight by domain size
            replay_ratio: Fraction of tasks to repeat from earlier domains
            config: Additional configuration
            seed: Random seed
        """
        self.interleave_pattern = interleave_pattern
        self.replay_ratio = replay_ratio
        super().__init__(
            tasks=tasks,
            strategy=CurriculumStrategy.INTERLEAVED,
            config=config,
            seed=seed,
        )

    def _generate_curriculum(self) -> List[str]:
        """Generate interleaved curriculum."""
        if self.seed is not None:
            random.seed(self.seed)

        # Group tasks by domain
        domain_tasks: Dict[str, List[str]] = {}
        for task in self.tasks:
            domain = self.task_metadata[task.id].domain
            if domain not in domain_tasks:
                domain_tasks[domain] = []
            domain_tasks[domain].append(task.id)

        # Shuffle within domains
        for domain in domain_tasks:
            random.shuffle(domain_tasks[domain])

        # Build interleaved curriculum
        if self.interleave_pattern == "round_robin":
            return self._round_robin_interleave(domain_tasks)
        elif self.interleave_pattern == "random":
            return self._random_interleave(domain_tasks)
        elif self.interleave_pattern == "weighted":
            return self._weighted_interleave(domain_tasks)
        else:
            return self._round_robin_interleave(domain_tasks)

    def _round_robin_interleave(
        self,
        domain_tasks: Dict[str, List[str]]
    ) -> List[str]:
        """Round-robin interleaving across domains."""
        curriculum = []
        domains = sorted(domain_tasks.keys())
        indices = {d: 0 for d in domains}

        while True:
            added = False
            for domain in domains:
                if indices[domain] < len(domain_tasks[domain]):
                    curriculum.append(domain_tasks[domain][indices[domain]])
                    indices[domain] += 1
                    added = True
            if not added:
                break

        return curriculum

    def _random_interleave(
        self,
        domain_tasks: Dict[str, List[str]]
    ) -> List[str]:
        """Random interleaving - select domain randomly for each task."""
        curriculum = []
        indices = {d: 0 for d in domain_tasks}
        available_domains = list(domain_tasks.keys())

        while available_domains:
            domain = random.choice(available_domains)
            curriculum.append(domain_tasks[domain][indices[domain]])
            indices[domain] += 1

            if indices[domain] >= len(domain_tasks[domain]):
                available_domains.remove(domain)

        return curriculum

    def _weighted_interleave(
        self,
        domain_tasks: Dict[str, List[str]]
    ) -> List[str]:
        """Weighted interleaving - probability proportional to remaining tasks."""
        curriculum = []
        indices = {d: 0 for d in domain_tasks}

        while True:
            # Calculate weights
            weights = []
            domains = []
            for d in domain_tasks:
                remaining = len(domain_tasks[d]) - indices[d]
                if remaining > 0:
                    weights.append(remaining)
                    domains.append(d)

            if not domains:
                break

            # Select domain
            domain = random.choices(domains, weights=weights, k=1)[0]
            curriculum.append(domain_tasks[domain][indices[domain]])
            indices[domain] += 1

        return curriculum
