# Copyright Sierra
"""
Curriculum Manager Base Classes for Continual Learning

This module provides the base classes and interfaces for managing task sequences
in continual learning experiments for tool use agents.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from loguru import logger

from tau2.data_model.tasks import Task


class CurriculumStrategy(str, Enum):
    """Curriculum strategy enumeration"""
    SEQUENTIAL = "sequential"          # Learn domains sequentially
    INTERLEAVED = "interleaved"        # Interleave tasks from different domains
    DIFFICULTY_BASED = "difficulty"    # Progress by difficulty
    SIMILARITY_BASED = "similarity"    # Cluster similar tasks together
    RANDOM = "random"                  # Random order
    ANTI_CURRICULUM = "anti"           # Hard to easy (anti-curriculum)


class TaskMetadata(BaseModel):
    """Metadata for a task used in curriculum management"""

    task_id: str = Field(description="Unique task identifier")
    domain: str = Field(description="Domain name (e.g., airline, retail, telecom)")
    difficulty: Optional[float] = Field(
        default=None,
        description="Task difficulty score (0.0-1.0)"
    )
    required_tools: List[str] = Field(
        default_factory=list,
        description="List of tools required for this task"
    )
    semantic_tags: List[str] = Field(
        default_factory=list,
        description="Semantic tags for clustering/similarity"
    )
    estimated_steps: Optional[int] = Field(
        default=None,
        description="Estimated number of steps to complete"
    )
    num_agent_actions: int = Field(
        default=0,
        description="Number of agent actions in evaluation criteria"
    )
    num_user_actions: int = Field(
        default=0,
        description="Number of user actions in evaluation criteria"
    )


class CurriculumState(BaseModel):
    """State of the curriculum manager"""

    current_index: int = Field(default=0, description="Current position in curriculum")
    tasks_completed: List[str] = Field(
        default_factory=list,
        description="List of completed task IDs"
    )
    performance_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Performance history for adaptive curriculum"
    )
    created_at: datetime = Field(default_factory=datetime.now)


class CurriculumManager(ABC):
    """
    Base class for curriculum managers.

    Curriculum managers control the order in which tasks are presented
    to the agent during continual learning experiments.
    """

    def __init__(
        self,
        tasks: List[Task],
        strategy: CurriculumStrategy,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the curriculum manager.

        Args:
            tasks: List of Task objects to include in the curriculum
            strategy: Curriculum strategy to use
            config: Additional configuration options
            seed: Random seed for reproducibility
        """
        self.tasks = tasks
        self.strategy = strategy
        self.config = config or {}
        self.seed = seed

        # Build task lookup
        self._task_lookup: Dict[str, Task] = {task.id: task for task in tasks}

        # Compute metadata for each task
        self.task_metadata: Dict[str, TaskMetadata] = {}
        for task in tasks:
            self.task_metadata[task.id] = self._compute_task_metadata(task)

        # Generate the curriculum (ordered list of task IDs)
        self.curriculum: List[str] = self._generate_curriculum()

        # Initialize state
        self.state = CurriculumState()

        logger.info(
            f"Initialized {self.__class__.__name__} with {len(tasks)} tasks, "
            f"strategy={strategy.value}"
        )

    def _compute_task_metadata(self, task: Task) -> TaskMetadata:
        """
        Compute metadata for a task.

        Args:
            task: The task to compute metadata for

        Returns:
            TaskMetadata object with computed values
        """
        # Extract domain from task user_scenario.instructions if available
        # Fall back to task ID parsing (format: domain_xxx_xxx)
        domain = "unknown"
        if (hasattr(task, 'user_scenario') and hasattr(task.user_scenario, 'instructions')):
            if isinstance(task.user_scenario.instructions, dict):
                domain = task.user_scenario.instructions.get('domain', 'unknown')
            elif hasattr(task.user_scenario.instructions, 'domain'):
                domain = task.user_scenario.instructions.domain

        # If still unknown, try to extract from task ID
        if domain == "unknown" and "_" in task.id:
            domain = task.id.split("_")[0]

        # Get action counts from evaluation criteria
        num_agent_actions = 0
        num_user_actions = 0
        required_tools = []

        if task.evaluation_criteria and task.evaluation_criteria.actions:
            for action in task.evaluation_criteria.actions:
                if action.requestor == "assistant":
                    num_agent_actions += 1
                    if action.name not in required_tools:
                        required_tools.append(action.name)
                else:
                    num_user_actions += 1

        # Compute difficulty based on action counts
        total_actions = num_agent_actions + num_user_actions
        difficulty = min(total_actions / 10.0, 1.0) if total_actions > 0 else 0.5

        return TaskMetadata(
            task_id=task.id,
            domain=domain,
            difficulty=difficulty,
            required_tools=required_tools,
            num_agent_actions=num_agent_actions,
            num_user_actions=num_user_actions,
            estimated_steps=total_actions * 2 if total_actions > 0 else 10,
        )

    @abstractmethod
    def _generate_curriculum(self) -> List[str]:
        """
        Generate the curriculum sequence.

        Returns:
            Ordered list of task IDs representing the curriculum
        """
        raise NotImplementedError

    def get_next_task(self) -> Optional[Task]:
        """
        Get the next task in the curriculum.

        Returns:
            The next Task object, or None if curriculum is complete
        """
        if self.is_complete():
            return None

        task_id = self.curriculum[self.state.current_index]
        self.state.current_index += 1

        return self._task_lookup[task_id]

    def get_next_task_batch(self, batch_size: int = 1) -> List[Task]:
        """
        Get the next batch of tasks.

        Args:
            batch_size: Number of tasks to return

        Returns:
            List of Task objects
        """
        tasks = []
        for _ in range(batch_size):
            task = self.get_next_task()
            if task is None:
                break
            tasks.append(task)
        return tasks

    def peek_next_task(self) -> Optional[Task]:
        """
        Peek at the next task without advancing the curriculum.

        Returns:
            The next Task object, or None if curriculum is complete
        """
        if self.is_complete():
            return None

        task_id = self.curriculum[self.state.current_index]
        return self._task_lookup[task_id]

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """
        Get a task by its ID.

        Args:
            task_id: The task ID to look up

        Returns:
            The Task object, or None if not found
        """
        return self._task_lookup.get(task_id)

    def is_complete(self) -> bool:
        """
        Check if the curriculum is complete.

        Returns:
            True if all tasks have been presented
        """
        return self.state.current_index >= len(self.curriculum)

    def get_progress(self) -> float:
        """
        Get the progress through the curriculum.

        Returns:
            Progress as a fraction (0.0 to 1.0)
        """
        if len(self.curriculum) == 0:
            return 1.0
        return self.state.current_index / len(self.curriculum)

    def record_performance(
        self,
        task_id: str,
        reward: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record performance on a task for adaptive curriculum.

        Args:
            task_id: The task ID
            reward: The reward obtained
            metrics: Additional metrics to record
        """
        self.state.tasks_completed.append(task_id)
        self.state.performance_history.append({
            "task_id": task_id,
            "reward": reward,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        })

    def adapt_curriculum(self) -> None:
        """
        Adapt the curriculum based on performance history.

        This method can be overridden by subclasses to implement
        adaptive curriculum strategies.
        """
        pass  # Default: no adaptation

    def get_tasks_by_domain(self, domain: str) -> List[Task]:
        """
        Get all tasks for a specific domain.

        Args:
            domain: The domain name

        Returns:
            List of tasks for the domain
        """
        return [
            task for task in self.tasks
            if self.task_metadata[task.id].domain == domain
        ]

    def get_domains(self) -> List[str]:
        """
        Get all unique domains in the curriculum.

        Returns:
            List of domain names
        """
        domains = set()
        for metadata in self.task_metadata.values():
            domains.add(metadata.domain)
        return sorted(list(domains))

    def reset(self) -> None:
        """Reset the curriculum to the beginning."""
        self.state = CurriculumState()
        logger.info(f"Reset curriculum to beginning")

    def get_remaining_tasks(self) -> List[Task]:
        """
        Get all remaining tasks in the curriculum.

        Returns:
            List of remaining tasks
        """
        remaining_ids = self.curriculum[self.state.current_index:]
        return [self._task_lookup[tid] for tid in remaining_ids]

    def get_completed_tasks(self) -> List[Task]:
        """
        Get all completed tasks.

        Returns:
            List of completed tasks
        """
        return [
            self._task_lookup[tid]
            for tid in self.state.tasks_completed
            if tid in self._task_lookup
        ]

    def __len__(self) -> int:
        """Return the total number of tasks in the curriculum."""
        return len(self.curriculum)

    def __iter__(self):
        """Iterate over all tasks in curriculum order."""
        for task_id in self.curriculum:
            yield self._task_lookup[task_id]
