# Copyright Sierra
"""
Task Generator for Continual Learning Data.

This module provides utilities for generating CL-specific data splits
and augmenting existing task data for continual learning experiments.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger


@dataclass
class CLSplitConfig:
    """Configuration for continual learning data splits."""

    # Split ratios
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # CL-specific settings
    num_phases: int = 3  # Number of learning phases
    tasks_per_phase: Optional[int] = None  # If None, divide evenly

    # Difficulty-based settings
    use_difficulty_ordering: bool = False
    difficulty_metric: str = "num_actions"  # num_actions, nl_assertions, etc.

    # Randomization
    seed: int = 42


class TaskGenerator:
    """
    Generator for CL-specific task data and splits.
    """

    def __init__(
        self,
        tasks: List[Dict[str, Any]],
        domain: str,
        config: Optional[CLSplitConfig] = None,
    ):
        """
        Initialize the task generator.

        Args:
            tasks: List of task dictionaries
            domain: Domain name
            config: Split configuration
        """
        self.tasks = tasks
        self.domain = domain
        self.config = config or CLSplitConfig()

        # Set random seed
        random.seed(self.config.seed)

    def compute_task_difficulty(self, task: Dict[str, Any]) -> float:
        """
        Compute difficulty score for a task.

        Args:
            task: Task dictionary

        Returns:
            Difficulty score (higher = more difficult)
        """
        eval_criteria = task.get("evaluation_criteria", {})
        if not eval_criteria:
            return 0.0

        score = 0.0

        # Number of required actions
        actions = eval_criteria.get("actions", [])
        score += len(actions) * 1.0

        # Number of NL assertions (harder to satisfy)
        nl_assertions = eval_criteria.get("nl_assertions", [])
        score += len(nl_assertions) * 1.5

        # Number of env assertions
        env_assertions = eval_criteria.get("env_assertions", [])
        score += len(env_assertions) * 1.2

        # Complexity from task instructions
        instructions = task.get("user_scenario", {}).get("instructions", {})
        if isinstance(instructions, dict):
            task_inst = instructions.get("task_instructions", "")
            # More complex instructions = higher difficulty
            score += len(task_inst) / 200.0

        return score

    def get_difficulty_ordered_tasks(self) -> List[Dict[str, Any]]:
        """
        Get tasks ordered by difficulty.

        Returns:
            List of tasks sorted by difficulty (easy to hard)
        """
        tasks_with_difficulty = [
            (task, self.compute_task_difficulty(task))
            for task in self.tasks
        ]
        tasks_with_difficulty.sort(key=lambda x: x[1])
        return [task for task, _ in tasks_with_difficulty]

    def generate_train_val_test_split(
        self
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Generate train/val/test split.

        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        task_ids = [task["id"] for task in self.tasks]

        if self.config.use_difficulty_ordering:
            ordered_tasks = self.get_difficulty_ordered_tasks()
            task_ids = [task["id"] for task in ordered_tasks]
        else:
            random.shuffle(task_ids)

        n = len(task_ids)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        train_ids = task_ids[:train_end]
        val_ids = task_ids[train_end:val_end]
        test_ids = task_ids[val_end:]

        return train_ids, val_ids, test_ids

    def generate_phase_splits(self) -> List[List[str]]:
        """
        Generate phase-based splits for continual learning.

        Returns:
            List of task ID lists, one per phase
        """
        if self.config.use_difficulty_ordering:
            ordered_tasks = self.get_difficulty_ordered_tasks()
            task_ids = [task["id"] for task in ordered_tasks]
        else:
            task_ids = [task["id"] for task in self.tasks]
            random.shuffle(task_ids)

        n = len(task_ids)
        num_phases = self.config.num_phases

        if self.config.tasks_per_phase:
            # Fixed number of tasks per phase
            phases = []
            for i in range(num_phases):
                start = i * self.config.tasks_per_phase
                end = min(start + self.config.tasks_per_phase, n)
                if start < n:
                    phases.append(task_ids[start:end])
            return phases
        else:
            # Divide evenly
            phase_size = n // num_phases
            phases = []
            for i in range(num_phases):
                start = i * phase_size
                end = start + phase_size if i < num_phases - 1 else n
                phases.append(task_ids[start:end])
            return phases

    def generate_curriculum_splits(
        self,
        strategy: str = "sequential"
    ) -> Dict[str, Any]:
        """
        Generate curriculum-based splits for CL experiments.

        Args:
            strategy: "sequential", "interleaved", or "difficulty"

        Returns:
            Dictionary with split information
        """
        train_ids, val_ids, test_ids = self.generate_train_val_test_split()

        result = {
            "domain": self.domain,
            "strategy": strategy,
            "config": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
                "num_phases": self.config.num_phases,
                "seed": self.config.seed,
            },
            "splits": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            },
        }

        # Add phase information for CL
        if strategy == "sequential":
            # Train in phases, test on all
            phases = self.generate_phase_splits()
            # Only use train set for phases
            train_task_ids = set(train_ids)
            result["phases"] = []
            phase_idx = 0
            for phase in phases:
                phase_train = [tid for tid in phase if tid in train_task_ids]
                if phase_train:
                    result["phases"].append({
                        "phase_id": phase_idx,
                        "task_ids": phase_train,
                        "size": len(phase_train),
                    })
                    phase_idx += 1

        elif strategy == "difficulty":
            # Order by difficulty
            ordered = self.get_difficulty_ordered_tasks()
            ordered_ids = [t["id"] for t in ordered]
            train_ordered = [tid for tid in ordered_ids if tid in set(train_ids)]

            # Create difficulty-based phases
            phase_size = len(train_ordered) // self.config.num_phases
            result["phases"] = []
            for i in range(self.config.num_phases):
                start = i * phase_size
                end = start + phase_size if i < self.config.num_phases - 1 else len(train_ordered)
                phase_ids = train_ordered[start:end]
                result["phases"].append({
                    "phase_id": i,
                    "task_ids": phase_ids,
                    "size": len(phase_ids),
                    "difficulty_level": ["easy", "medium", "hard"][min(i, 2)],
                })

        elif strategy == "interleaved":
            # Interleave tasks from different "types"
            # Group by action count as a proxy for task type
            task_by_complexity = {}
            for task in self.tasks:
                if task["id"] not in set(train_ids):
                    continue
                difficulty = self.compute_task_difficulty(task)
                bucket = "easy" if difficulty < 3 else "medium" if difficulty < 6 else "hard"
                if bucket not in task_by_complexity:
                    task_by_complexity[bucket] = []
                task_by_complexity[bucket].append(task["id"])

            # Interleave
            interleaved = []
            max_len = max(len(v) for v in task_by_complexity.values()) if task_by_complexity else 0
            for i in range(max_len):
                for bucket in ["easy", "medium", "hard"]:
                    if bucket in task_by_complexity and i < len(task_by_complexity[bucket]):
                        interleaved.append(task_by_complexity[bucket][i])

            result["phases"] = [{
                "phase_id": 0,
                "task_ids": interleaved,
                "size": len(interleaved),
                "strategy": "interleaved",
            }]

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the tasks.

        Returns:
            Dictionary of statistics
        """
        difficulties = [self.compute_task_difficulty(t) for t in self.tasks]

        return {
            "total_tasks": len(self.tasks),
            "domain": self.domain,
            "difficulty": {
                "min": min(difficulties) if difficulties else 0,
                "max": max(difficulties) if difficulties else 0,
                "mean": sum(difficulties) / len(difficulties) if difficulties else 0,
            },
            "config": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
                "num_phases": self.config.num_phases,
            },
        }


def generate_cl_split(
    tasks_file: Path,
    output_file: Path,
    strategy: str = "sequential",
    config: Optional[CLSplitConfig] = None,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate CL splits from a tasks file.

    Args:
        tasks_file: Path to tasks.json
        output_file: Path to save split file
        strategy: Curriculum strategy
        config: Split configuration
        domain: Domain name (auto-detected if None)

    Returns:
        Split dictionary
    """
    # Load tasks
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    # Auto-detect domain
    if domain is None:
        parts = tasks_file.parts
        if "domains" in parts:
            idx = parts.index("domains")
            if idx + 1 < len(parts):
                domain = parts[idx + 1]
        else:
            domain = "unknown"

    # Generate splits
    generator = TaskGenerator(tasks, domain, config)
    splits = generator.generate_curriculum_splits(strategy)

    # Add statistics
    splits["statistics"] = generator.get_statistics()

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Generated CL splits: {output_file}")
    logger.info(f"  Strategy: {strategy}")
    logger.info(f"  Train: {len(splits['splits']['train'])} tasks")
    logger.info(f"  Val: {len(splits['splits']['val'])} tasks")
    logger.info(f"  Test: {len(splits['splits']['test'])} tasks")
    logger.info(f"  Phases: {len(splits.get('phases', []))}")

    return splits


def generate_multi_domain_cl_split(
    domain_tasks_files: Dict[str, Path],
    output_file: Path,
    strategy: str = "sequential",
    config: Optional[CLSplitConfig] = None,
) -> Dict[str, Any]:
    """
    Generate CL splits across multiple domains.

    Args:
        domain_tasks_files: Dict mapping domain name to tasks.json path
        output_file: Path to save combined split file
        strategy: Curriculum strategy
        config: Split configuration

    Returns:
        Combined split dictionary
    """
    config = config or CLSplitConfig()

    result = {
        "strategy": strategy,
        "domains": {},
        "domain_order": list(domain_tasks_files.keys()),
        "combined_phases": [],
    }

    phase_id = 0

    for domain, tasks_file in domain_tasks_files.items():
        # Load tasks
        with open(tasks_file, 'r', encoding='utf-8') as f:
            tasks = json.load(f)

        # Generate splits for this domain
        generator = TaskGenerator(tasks, domain, config)
        splits = generator.generate_curriculum_splits(strategy)

        result["domains"][domain] = splits

        # Add to combined phases
        for phase in splits.get("phases", []):
            combined_phase = {
                "phase_id": phase_id,
                "domain": domain,
                "task_ids": phase["task_ids"],
                "size": phase["size"],
            }
            result["combined_phases"].append(combined_phase)
            phase_id += 1

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Generated multi-domain CL splits: {output_file}")
    logger.info(f"  Domains: {list(domain_tasks_files.keys())}")
    logger.info(f"  Total phases: {len(result['combined_phases'])}")

    return result


# Utility functions for data requirements analysis

def analyze_data_requirements(
    domain_tasks: Dict[str, List[Dict[str, Any]]],
    target_buffer_size: int = 1000,
    experiences_per_task: int = 4,
    success_rate: float = 0.7,
) -> Dict[str, Any]:
    """
    Analyze data requirements for CL experiments.

    Args:
        domain_tasks: Dict mapping domain to task list
        target_buffer_size: Target memory buffer size
        experiences_per_task: Average experiences per successful task
        success_rate: Expected task success rate

    Returns:
        Dictionary with recommendations
    """
    recommendations = {
        "current_data": {},
        "requirements": {},
        "recommendations": {},
    }

    total_current = 0
    for domain, tasks in domain_tasks.items():
        current = len(tasks)
        total_current += current

        # Calculate how many tasks needed to fill buffer
        needed_for_buffer = target_buffer_size / (experiences_per_task * success_rate)

        # For statistical significance, need at least 100 train + 30 test
        min_for_stats = 130

        # Recommendation
        recommended = max(needed_for_buffer, min_for_stats)

        recommendations["current_data"][domain] = current
        recommendations["requirements"][domain] = {
            "for_buffer": int(needed_for_buffer),
            "for_statistics": min_for_stats,
        }
        recommendations["recommendations"][domain] = {
            "recommended_total": int(recommended),
            "additional_needed": max(0, int(recommended - current)),
            "status": "sufficient" if current >= recommended else "needs_more",
        }

    recommendations["summary"] = {
        "total_current_tasks": total_current,
        "target_buffer_size": target_buffer_size,
        "experiences_per_task": experiences_per_task,
        "success_rate": success_rate,
    }

    return recommendations


def print_data_requirements(
    domain_tasks: Dict[str, List[Dict[str, Any]]],
    target_buffer_size: int = 1000,
) -> None:
    """Print formatted data requirements analysis."""
    analysis = analyze_data_requirements(domain_tasks, target_buffer_size)

    print("\n" + "=" * 60)
    print("Continual Learning Data Requirements Analysis")
    print("=" * 60)

    print(f"\nTarget buffer size: {target_buffer_size}")
    print(f"Assumed experiences per task: 4")
    print(f"Assumed success rate: 70%")

    print("\n" + "-" * 60)
    print(f"{'Domain':<15} {'Current':<10} {'Needed':<10} {'Status':<15}")
    print("-" * 60)

    for domain in analysis["current_data"]:
        current = analysis["current_data"][domain]
        needed = analysis["recommendations"][domain]["recommended_total"]
        status = analysis["recommendations"][domain]["status"]
        additional = analysis["recommendations"][domain]["additional_needed"]

        status_str = f"{status} (+{additional})" if additional > 0 else status
        print(f"{domain:<15} {current:<10} {needed:<10} {status_str:<15}")

    print("-" * 60)
    print(f"{'Total':<15} {analysis['summary']['total_current_tasks']:<10}")
    print("=" * 60 + "\n")
