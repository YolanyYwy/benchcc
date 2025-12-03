# Copyright Sierra
"""
Task Validator for Continual Learning Data Generation.

This module provides validation utilities to ensure generated tasks
are valid and consistent with the tau2-bench data format.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ValidationResult:
    """Result of task validation."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    task_id: Optional[str] = None

    def add_error(self, error: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning (does not affect validity)."""
        self.warnings.append(warning)

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"Task {self.task_id}: {status}"]
        for error in self.errors:
            lines.append(f"  [ERROR] {error}")
        for warning in self.warnings:
            lines.append(f"  [WARN] {warning}")
        return "\n".join(lines)


class TaskValidator:
    """
    Validator for tau2-bench task data.

    Validates task structure, field types, and consistency with
    domain tools and database.
    """

    # Required fields at each level
    REQUIRED_TASK_FIELDS = {"id", "user_scenario"}
    REQUIRED_USER_SCENARIO_FIELDS = {"instructions"}
    REQUIRED_INSTRUCTIONS_FIELDS = {"domain", "reason_for_call", "task_instructions"}

    # Valid domains
    VALID_DOMAINS = {"airline", "retail", "telecom", "mock"}

    # Valid reward types
    VALID_REWARD_TYPES = {"DB", "COMMUNICATE", "ENV_ASSERTION", "NL_ASSERTION", "ACTION"}

    def __init__(
        self,
        domain: Optional[str] = None,
        db_path: Optional[Path] = None,
        tool_names: Optional[Set[str]] = None,
    ):
        """
        Initialize the validator.

        Args:
            domain: Expected domain for tasks (optional)
            db_path: Path to domain database for consistency checks
            tool_names: Set of valid tool names for the domain
        """
        self.domain = domain
        self.db_data = None
        self.tool_names = tool_names or set()

        if db_path and db_path.exists():
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    self.db_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load database: {e}")

    def validate_task(self, task: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single task.

        Args:
            task: Task dictionary to validate

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()

        # Extract task ID
        task_id = task.get("id")
        result.task_id = task_id

        # Check required fields
        self._validate_required_fields(task, result)
        if not result.is_valid:
            return result

        # Validate user scenario
        self._validate_user_scenario(task.get("user_scenario", {}), result)

        # Validate evaluation criteria if present
        if "evaluation_criteria" in task and task["evaluation_criteria"]:
            self._validate_evaluation_criteria(task["evaluation_criteria"], task_id, result)

        # Validate initial state if present
        if "initial_state" in task and task["initial_state"]:
            self._validate_initial_state(task["initial_state"], result)

        return result

    def _validate_required_fields(
        self,
        task: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Check that all required fields are present."""
        # Task level
        for field in self.REQUIRED_TASK_FIELDS:
            if field not in task or task[field] is None:
                result.add_error(f"Missing required field: {field}")

        # User scenario level
        user_scenario = task.get("user_scenario", {})
        if user_scenario:
            for field in self.REQUIRED_USER_SCENARIO_FIELDS:
                if field not in user_scenario or user_scenario[field] is None:
                    result.add_error(f"Missing required field: user_scenario.{field}")

            # Instructions level
            instructions = user_scenario.get("instructions", {})
            if isinstance(instructions, dict):
                for field in self.REQUIRED_INSTRUCTIONS_FIELDS:
                    if field not in instructions or not instructions[field]:
                        result.add_error(
                            f"Missing required field: user_scenario.instructions.{field}"
                        )

    def _validate_user_scenario(
        self,
        user_scenario: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate user scenario structure and content."""
        instructions = user_scenario.get("instructions", {})

        if not isinstance(instructions, dict):
            # String instructions are allowed but less structured
            result.add_warning("Instructions should be structured (StructuredUserInstructions)")
            return

        # Check domain
        domain = instructions.get("domain")
        if domain:
            if domain not in self.VALID_DOMAINS:
                result.add_error(f"Invalid domain: {domain}")
            if self.domain and domain != self.domain:
                result.add_error(
                    f"Domain mismatch: task has '{domain}' but expected '{self.domain}'"
                )

        # Check that reason_for_call is meaningful
        reason = instructions.get("reason_for_call", "")
        if len(reason) < 10:
            result.add_warning("reason_for_call is too short (< 10 chars)")

        # Check that task_instructions is meaningful
        task_inst = instructions.get("task_instructions", "")
        if len(task_inst) < 10:
            result.add_warning("task_instructions is too short (< 10 chars)")

        # Check known_info format
        known_info = instructions.get("known_info")
        if known_info:
            # Should typically contain user name and/or user_id
            if "user" not in known_info.lower() and "name" not in known_info.lower():
                result.add_warning("known_info should typically contain user identity")

    def _validate_evaluation_criteria(
        self,
        eval_criteria: Dict[str, Any],
        task_id: str,
        result: ValidationResult
    ) -> None:
        """Validate evaluation criteria."""
        # Validate actions
        actions = eval_criteria.get("actions", [])
        if actions:
            seen_action_ids = set()
            for i, action in enumerate(actions):
                self._validate_action(action, task_id, i, seen_action_ids, result)

        # Validate reward_basis
        reward_basis = eval_criteria.get("reward_basis", [])
        if reward_basis:
            for rb in reward_basis:
                if rb not in self.VALID_REWARD_TYPES:
                    result.add_error(f"Invalid reward_basis type: {rb}")

        # Check nl_assertions format
        nl_assertions = eval_criteria.get("nl_assertions", [])
        if nl_assertions:
            for i, assertion in enumerate(nl_assertions):
                if not isinstance(assertion, str) or len(assertion) < 5:
                    result.add_warning(f"nl_assertion[{i}] is too short or not a string")

    def _validate_action(
        self,
        action: Dict[str, Any],
        task_id: str,
        index: int,
        seen_action_ids: Set[str],
        result: ValidationResult
    ) -> None:
        """Validate a single action."""
        # Check required action fields
        if "action_id" not in action:
            result.add_error(f"Action[{index}] missing action_id")
            return

        action_id = action["action_id"]

        # Check action_id uniqueness
        if action_id in seen_action_ids:
            result.add_error(f"Duplicate action_id: {action_id}")
        seen_action_ids.add(action_id)

        # Check action_id format (should be task_id_number)
        expected_prefix = f"{task_id}_"
        if not action_id.startswith(expected_prefix):
            result.add_warning(
                f"Action ID '{action_id}' should start with '{expected_prefix}'"
            )

        # Check name
        name = action.get("name")
        if not name:
            result.add_error(f"Action[{index}] missing name")
        elif self.tool_names and name not in self.tool_names:
            result.add_warning(f"Unknown tool name: {name}")

        # Check arguments
        arguments = action.get("arguments")
        if arguments is not None and not isinstance(arguments, dict):
            result.add_error(f"Action[{index}] arguments must be a dict")

    def _validate_initial_state(
        self,
        initial_state: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate initial state if present."""
        # Check message_history format
        message_history = initial_state.get("message_history", [])
        if message_history:
            for i, msg in enumerate(message_history):
                if not isinstance(msg, dict):
                    result.add_error(f"message_history[{i}] must be a dict")
                    continue
                if "role" not in msg:
                    result.add_error(f"message_history[{i}] missing 'role'")

    def validate_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Tuple[List[ValidationResult], bool]:
        """
        Validate a list of tasks.

        Args:
            tasks: List of task dictionaries

        Returns:
            Tuple of (list of results, overall validity)
        """
        results = []
        all_valid = True
        seen_ids = set()

        for task in tasks:
            result = self.validate_task(task)
            results.append(result)

            if not result.is_valid:
                all_valid = False

            # Check global ID uniqueness
            task_id = task.get("id")
            if task_id in seen_ids:
                result.add_error(f"Duplicate task ID across tasks: {task_id}")
                all_valid = False
            seen_ids.add(task_id)

        return results, all_valid


def validate_task_file(
    file_path: Path,
    domain: Optional[str] = None,
    verbose: bool = True
) -> Tuple[List[ValidationResult], bool]:
    """
    Validate all tasks in a JSON file.

    Args:
        file_path: Path to tasks.json file
        domain: Expected domain (optional)
        verbose: Print results

    Returns:
        Tuple of (list of results, overall validity)
    """
    # Load tasks
    with open(file_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError("Tasks file must contain a JSON array")

    # Auto-detect domain from path if not provided
    if domain is None:
        parts = file_path.parts
        if "domains" in parts:
            idx = parts.index("domains")
            if idx + 1 < len(parts):
                domain = parts[idx + 1]

    # Try to load tool names
    tool_names = set()
    if domain:
        try:
            from tau2.environment.loader import get_tools
            tools = get_tools(domain)
            tool_names = {t.name for t in tools}
        except Exception:
            pass

    # Find db.json if it exists
    db_path = file_path.parent / "db.json"

    # Create validator
    validator = TaskValidator(
        domain=domain,
        db_path=db_path if db_path.exists() else None,
        tool_names=tool_names,
    )

    # Validate
    results, all_valid = validator.validate_tasks(tasks)

    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print(f"Validation Results for: {file_path}")
        print(f"Domain: {domain or 'unknown'}")
        print(f"Total tasks: {len(tasks)}")
        print(f"{'='*60}")

        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count
        warning_count = sum(len(r.warnings) for r in results)

        print(f"\nSummary:")
        print(f"  Valid tasks: {valid_count}")
        print(f"  Invalid tasks: {invalid_count}")
        print(f"  Total warnings: {warning_count}")

        # Show details for invalid tasks
        if invalid_count > 0:
            print(f"\nInvalid Tasks:")
            for result in results:
                if not result.is_valid:
                    print(f"  - {result}")

        # Show some warnings
        if warning_count > 0 and verbose:
            print(f"\nFirst few warnings:")
            warn_shown = 0
            for result in results:
                if warn_shown >= 5:
                    break
                for warning in result.warnings:
                    if warn_shown >= 5:
                        break
                    print(f"  - Task {result.task_id}: {warning}")
                    warn_shown += 1

        print(f"\n{'='*60}")
        print(f"Overall: {'PASSED' if all_valid else 'FAILED'}")
        print(f"{'='*60}\n")

    return results, all_valid


def get_task_statistics(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about a list of tasks.

    Args:
        tasks: List of task dictionaries

    Returns:
        Dictionary of statistics
    """
    stats = {
        "total_tasks": len(tasks),
        "tasks_with_actions": 0,
        "tasks_with_nl_assertions": 0,
        "tasks_with_env_assertions": 0,
        "total_actions": 0,
        "total_nl_assertions": 0,
        "unique_tools": set(),
        "actions_per_task": [],
    }

    for task in tasks:
        eval_criteria = task.get("evaluation_criteria", {})
        if not eval_criteria:
            continue

        actions = eval_criteria.get("actions", [])
        nl_assertions = eval_criteria.get("nl_assertions", [])
        env_assertions = eval_criteria.get("env_assertions", [])

        if actions:
            stats["tasks_with_actions"] += 1
            stats["total_actions"] += len(actions)
            stats["actions_per_task"].append(len(actions))
            for action in actions:
                stats["unique_tools"].add(action.get("name", "unknown"))

        if nl_assertions:
            stats["tasks_with_nl_assertions"] += 1
            stats["total_nl_assertions"] += len(nl_assertions)

        if env_assertions:
            stats["tasks_with_env_assertions"] += 1

    # Convert set to list for JSON serialization
    stats["unique_tools"] = list(stats["unique_tools"])

    # Calculate averages
    if stats["actions_per_task"]:
        stats["avg_actions_per_task"] = sum(stats["actions_per_task"]) / len(stats["actions_per_task"])
    else:
        stats["avg_actions_per_task"] = 0

    return stats
