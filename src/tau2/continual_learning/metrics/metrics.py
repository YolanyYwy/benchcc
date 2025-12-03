# Copyright Sierra
"""
Continual Learning Metrics

This module provides metrics specific to continual learning evaluation,
including forgetting rate, forward/backward transfer, and performance matrices.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from loguru import logger


class MetricType(str, Enum):
    """Types of CL metrics"""
    ACCURACY = "accuracy"           # Task completion accuracy
    FORGETTING = "forgetting"       # Forgetting rate
    FORWARD_TRANSFER = "fwt"        # Forward transfer
    BACKWARD_TRANSFER = "bwt"       # Backward transfer
    AVERAGE_ACCURACY = "avg_acc"    # Average accuracy across all tasks
    LEARNING_CURVE = "learning"     # Learning curve data


class TaskResult(BaseModel):
    """Result for a single task evaluation"""

    task_id: str
    domain: str
    reward: float
    success: bool = False
    num_steps: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResults(BaseModel):
    """Results from a CL evaluation"""

    # Basic metrics
    average_accuracy: float = 0.0
    forgetting_rate: float = 0.0
    forward_transfer: float = 0.0
    backward_transfer: float = 0.0

    # Per-domain metrics
    domain_accuracies: Dict[str, float] = Field(default_factory=dict)
    domain_forgetting: Dict[str, float] = Field(default_factory=dict)

    # Detailed results
    task_results: List[TaskResult] = Field(default_factory=list)

    # Performance matrix (task x time)
    performance_matrix: Optional[List[List[float]]] = None

    # Learning curves
    learning_curves: Dict[str, List[float]] = Field(default_factory=dict)

    # Metadata
    num_tasks: int = 0
    num_domains: int = 0
    evaluation_time: float = 0.0


class PerformanceMatrix:
    """
    Performance matrix for tracking task performance over time.

    Matrix[i, j] = performance on task i after learning task j
    """

    def __init__(self, num_tasks: int):
        """
        Initialize the performance matrix.

        Args:
            num_tasks: Number of tasks in the experiment
        """
        self.num_tasks = num_tasks
        self.matrix = np.zeros((num_tasks, num_tasks))
        self.task_ids: List[str] = []
        self.domains: List[str] = []

    def record(
        self,
        task_idx: int,
        time_idx: int,
        performance: float
    ) -> None:
        """
        Record performance for a task at a time point.

        Args:
            task_idx: Index of the task being evaluated
            time_idx: Time point (after learning task time_idx)
            performance: Performance score
        """
        if task_idx < self.num_tasks and time_idx < self.num_tasks:
            self.matrix[task_idx, time_idx] = performance

    def get_forgetting(self, task_idx: int) -> float:
        """
        Calculate forgetting for a specific task.

        Forgetting = max performance - final performance

        Args:
            task_idx: Task index

        Returns:
            Forgetting rate for the task
        """
        if task_idx >= self.num_tasks:
            return 0.0

        row = self.matrix[task_idx, :]
        max_perf = np.max(row)
        final_perf = row[-1] if row[-1] > 0 else row[np.nonzero(row)[0][-1]] if np.any(row > 0) else 0
        return max(0, max_perf - final_perf)

    def get_average_forgetting(self) -> float:
        """Calculate average forgetting across all tasks."""
        forgetting_values = [
            self.get_forgetting(i)
            for i in range(self.num_tasks)
        ]
        return np.mean(forgetting_values)

    def get_forward_transfer(self) -> float:
        """
        Calculate forward transfer.

        FWT = average performance on task i immediately after learning task i-1
              compared to random baseline
        """
        if self.num_tasks < 2:
            return 0.0

        # Performance on task i right after task i-1 (before learning task i)
        # vs baseline (first task's initial performance)
        fwt_values = []
        for i in range(1, self.num_tasks):
            # Performance on task i before learning it
            if self.matrix[i, i-1] > 0:
                fwt_values.append(self.matrix[i, i-1])

        if not fwt_values:
            return 0.0
        return np.mean(fwt_values)

    def get_backward_transfer(self) -> float:
        """
        Calculate backward transfer.

        BWT = average improvement on previous tasks after learning new tasks
        """
        if self.num_tasks < 2:
            return 0.0

        bwt_values = []
        for i in range(self.num_tasks - 1):
            # Performance on task i at the end vs right after learning it
            perf_after_learning = self.matrix[i, i]
            perf_final = self.matrix[i, -1]
            if perf_after_learning > 0:
                bwt_values.append(perf_final - perf_after_learning)

        if not bwt_values:
            return 0.0
        return np.mean(bwt_values)

    def get_average_accuracy(self) -> float:
        """Get average accuracy across all tasks at final time."""
        final_perfs = self.matrix[:, -1]
        valid_perfs = final_perfs[final_perfs > 0]
        return np.mean(valid_perfs) if len(valid_perfs) > 0 else 0.0

    def to_list(self) -> List[List[float]]:
        """Convert matrix to nested list."""
        return self.matrix.tolist()


class ContinualLearningMetrics:
    """
    Metrics calculator for continual learning experiments.
    """

    def __init__(self):
        """Initialize the metrics calculator."""
        self._task_results: List[TaskResult] = []
        self._performance_matrix: Optional[PerformanceMatrix] = None
        self._task_order: List[str] = []
        self._domain_order: List[str] = []
        self._baseline_performances: Dict[str, float] = {}

    def initialize(
        self,
        task_order: List[str],
        domains: List[str]
    ) -> None:
        """
        Initialize metrics tracking for an experiment.

        Args:
            task_order: Order of tasks in the curriculum
            domains: List of domains
        """
        self._task_order = task_order
        self._domain_order = domains
        self._performance_matrix = PerformanceMatrix(len(task_order))
        self._performance_matrix.task_ids = task_order
        self._task_results = []

        logger.info(
            f"Initialized CL metrics for {len(task_order)} tasks "
            f"across {len(domains)} domains"
        )

    def record_task_result(
        self,
        task_id: str,
        domain: str,
        reward: float,
        success: bool = False,
        num_steps: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """
        Record the result of a task execution.

        Args:
            task_id: Task identifier
            domain: Domain name
            reward: Reward obtained
            success: Whether task succeeded
            num_steps: Number of steps taken
            metadata: Additional metadata

        Returns:
            The recorded TaskResult
        """
        result = TaskResult(
            task_id=task_id,
            domain=domain,
            reward=reward,
            success=success,
            num_steps=num_steps,
            metadata=metadata or {},
        )
        self._task_results.append(result)
        return result

    def record_evaluation(
        self,
        task_id: str,
        time_idx: int,
        reward: float
    ) -> None:
        """
        Record an evaluation result in the performance matrix.

        Args:
            task_id: Task being evaluated
            time_idx: Time point (task index in curriculum)
            reward: Performance/reward
        """
        if self._performance_matrix is None:
            logger.warning("Performance matrix not initialized")
            return

        if task_id in self._task_order:
            task_idx = self._task_order.index(task_id)
            self._performance_matrix.record(task_idx, time_idx, reward)

    def set_baseline(self, task_id: str, performance: float) -> None:
        """
        Set baseline performance for a task (e.g., zero-shot performance).

        Args:
            task_id: Task identifier
            performance: Baseline performance
        """
        self._baseline_performances[task_id] = performance

    def compute_metrics(self) -> EvaluationResults:
        """
        Compute all CL metrics.

        Returns:
            EvaluationResults with computed metrics
        """
        results = EvaluationResults()

        if self._performance_matrix is None:
            logger.warning("Cannot compute metrics: not initialized")
            return results

        # Basic metrics from performance matrix
        results.average_accuracy = self._performance_matrix.get_average_accuracy()
        results.forgetting_rate = self._performance_matrix.get_average_forgetting()
        results.forward_transfer = self._performance_matrix.get_forward_transfer()
        results.backward_transfer = self._performance_matrix.get_backward_transfer()

        # Per-domain metrics
        results.domain_accuracies = self._compute_domain_accuracies()
        results.domain_forgetting = self._compute_domain_forgetting()

        # Store raw data
        results.task_results = self._task_results
        results.performance_matrix = self._performance_matrix.to_list()

        # Learning curves
        results.learning_curves = self._compute_learning_curves()

        # Metadata
        results.num_tasks = len(self._task_order)
        results.num_domains = len(set(self._domain_order))

        return results

    def _compute_domain_accuracies(self) -> Dict[str, float]:
        """Compute accuracy per domain."""
        domain_rewards: Dict[str, List[float]] = {}

        for result in self._task_results:
            if result.domain not in domain_rewards:
                domain_rewards[result.domain] = []
            domain_rewards[result.domain].append(result.reward)

        return {
            domain: np.mean(rewards)
            for domain, rewards in domain_rewards.items()
        }

    def _compute_domain_forgetting(self) -> Dict[str, float]:
        """Compute forgetting per domain."""
        if self._performance_matrix is None:
            return {}

        domain_forgetting: Dict[str, List[float]] = {}

        for i, task_id in enumerate(self._task_order):
            if i < len(self._domain_order):
                domain = self._domain_order[i]
            else:
                continue

            if domain not in domain_forgetting:
                domain_forgetting[domain] = []

            forgetting = self._performance_matrix.get_forgetting(i)
            domain_forgetting[domain].append(forgetting)

        return {
            domain: np.mean(values)
            for domain, values in domain_forgetting.items()
        }

    def _compute_learning_curves(self) -> Dict[str, List[float]]:
        """Compute learning curves over time."""
        curves = {}

        # Overall accuracy over time
        if self._performance_matrix is not None:
            overall_curve = []
            for t in range(self._performance_matrix.num_tasks):
                # Average accuracy on all tasks seen so far
                seen_perfs = []
                for i in range(t + 1):
                    if self._performance_matrix.matrix[i, t] > 0:
                        seen_perfs.append(self._performance_matrix.matrix[i, t])
                if seen_perfs:
                    overall_curve.append(np.mean(seen_perfs))
            curves["overall"] = overall_curve

        # Per-domain curves
        domain_results: Dict[str, List[float]] = {}
        for result in self._task_results:
            if result.domain not in domain_results:
                domain_results[result.domain] = []
            domain_results[result.domain].append(result.reward)

        for domain, rewards in domain_results.items():
            curves[f"domain_{domain}"] = rewards

        return curves

    def get_summary(self) -> str:
        """Get a text summary of metrics."""
        results = self.compute_metrics()

        lines = [
            "=" * 50,
            "Continual Learning Metrics Summary",
            "=" * 50,
            f"Average Accuracy: {results.average_accuracy:.3f}",
            f"Forgetting Rate:  {results.forgetting_rate:.3f}",
            f"Forward Transfer: {results.forward_transfer:.3f}",
            f"Backward Transfer: {results.backward_transfer:.3f}",
            "",
            "Per-Domain Accuracy:",
        ]

        for domain, acc in results.domain_accuracies.items():
            lines.append(f"  {domain}: {acc:.3f}")

        lines.extend([
            "",
            "Per-Domain Forgetting:",
        ])

        for domain, fgt in results.domain_forgetting.items():
            lines.append(f"  {domain}: {fgt:.3f}")

        lines.append("=" * 50)

        return "\n".join(lines)


class CLEvaluator:
    """
    Evaluator for running continual learning evaluations.

    This class handles the evaluation protocol for CL experiments,
    including periodic evaluation on all seen tasks.
    """

    def __init__(
        self,
        metrics: ContinualLearningMetrics,
        eval_frequency: int = 10,
        eval_on_all_seen: bool = True,
        num_eval_trials: int = 1,
    ):
        """
        Initialize the evaluator.

        Args:
            metrics: Metrics calculator instance
            eval_frequency: How often to evaluate (every N tasks)
            eval_on_all_seen: Whether to evaluate on all seen tasks
            num_eval_trials: Number of evaluation trials per task
        """
        self.metrics = metrics
        self.eval_frequency = eval_frequency
        self.eval_on_all_seen = eval_on_all_seen
        self.num_eval_trials = num_eval_trials

        self._tasks_since_eval = 0
        self._seen_tasks: List[str] = []

    def should_evaluate(self) -> bool:
        """Check if evaluation should be performed."""
        return self._tasks_since_eval >= self.eval_frequency

    def record_task_learned(self, task_id: str) -> None:
        """Record that a task has been learned."""
        if task_id not in self._seen_tasks:
            self._seen_tasks.append(task_id)
        self._tasks_since_eval += 1

    def get_tasks_to_evaluate(self) -> List[str]:
        """Get list of tasks to evaluate."""
        if self.eval_on_all_seen:
            return self._seen_tasks.copy()
        else:
            return [self._seen_tasks[-1]] if self._seen_tasks else []

    def record_evaluation_done(self) -> None:
        """Record that evaluation has been completed."""
        self._tasks_since_eval = 0
