# Copyright Sierra
"""
Sequential Curriculum Strategy

Presents tasks in domain-sequential order: all tasks from domain A,
then all tasks from domain B, etc.
"""

import random
from typing import Any, Dict, List, Optional

from tau2.data_model.tasks import Task

from tau2.continual_learning.curriculum.base import (
    CurriculumManager,
    CurriculumStrategy,
)


class SequentialCurriculum(CurriculumManager):
    """
    Sequential curriculum: learn domains one at a time.

    This is the standard continual learning setup where the agent learns
    tasks from domain A, then moves to domain B, etc. This setup is most
    prone to catastrophic forgetting.
    """

    def __init__(
        self,
        tasks: List[Task],
        domain_order: Optional[List[str]] = None,
        shuffle_within_domain: bool = False,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize sequential curriculum.

        Args:
            tasks: List of tasks
            domain_order: Optional explicit domain order. If None, uses
                         alphabetical order.
            shuffle_within_domain: Whether to shuffle tasks within each domain
            config: Additional configuration
            seed: Random seed
        """
        self.domain_order = domain_order
        self.shuffle_within_domain = shuffle_within_domain
        super().__init__(
            tasks=tasks,
            strategy=CurriculumStrategy.SEQUENTIAL,
            config=config,
            seed=seed,
        )

    def _generate_curriculum(self) -> List[str]:
        """Generate sequential curriculum ordered by domain."""
        # Set random seed if provided
        if self.seed is not None:
            random.seed(self.seed)

        # Group tasks by domain
        domain_tasks: Dict[str, List[str]] = {}
        for task in self.tasks:
            domain = self.task_metadata[task.id].domain
            if domain not in domain_tasks:
                domain_tasks[domain] = []
            domain_tasks[domain].append(task.id)

        # Determine domain order
        if self.domain_order is not None:
            domains = [d for d in self.domain_order if d in domain_tasks]
            # Add any domains not in the explicit order
            for d in domain_tasks:
                if d not in domains:
                    domains.append(d)
        else:
            domains = sorted(domain_tasks.keys())

        # Build curriculum
        curriculum = []
        for domain in domains:
            task_ids = domain_tasks[domain]
            if self.shuffle_within_domain:
                random.shuffle(task_ids)
            curriculum.extend(task_ids)

        return curriculum
