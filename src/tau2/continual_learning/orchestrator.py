# Copyright Sierra
"""
Continual Learning Orchestrator

This module provides the main orchestrator for running continual learning
experiments with tau2-bench.
"""

from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type
import json
import os
import time

from pydantic import BaseModel, Field
from loguru import logger

from tau2.data_model.tasks import Task
from tau2.data_model.simulation import SimulationRun
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.user.user_simulator import UserSimulator
from tau2.registry import registry

from tau2.continual_learning.curriculum import (
    CurriculumManager,
    CurriculumStrategy,
    SequentialCurriculum,
    InterleavedCurriculum,
    DifficultyBasedCurriculum,
)
from tau2.continual_learning.agents import (
    ContinualLearningAgent,
    ICLExperienceReplayAgent,
    PromptStrategyAgent,
)
from tau2.continual_learning.memory import MemoryBuffer, SamplingStrategy
from tau2.continual_learning.metrics import (
    ContinualLearningMetrics,
    CLEvaluator,
    EvaluationResults,
)


class CLPhase(str, Enum):
    """Phases of continual learning experiment"""
    TRAINING = "training"
    EVALUATION = "evaluation"
    CONSOLIDATION = "consolidation"


class AgentType(str, Enum):
    """Types of continual learning agents"""
    ICL_ER = "icl_er"               # ICL + Experience Replay
    PROMPT_STRATEGY = "prompt_strategy"  # Prompt Strategy Evolution
    BASELINE = "baseline"           # No CL (baseline)


class CLExperimentConfig(BaseModel):
    """Configuration for a continual learning experiment"""

    # Experiment identification
    name: str = "cl_experiment"
    seed: int = 42
    output_dir: str = "./experiments"

    # Curriculum configuration
    curriculum_strategy: CurriculumStrategy = CurriculumStrategy.SEQUENTIAL
    domains: List[str] = Field(default_factory=lambda: ["airline", "retail"])
    task_split: str = "train"
    num_tasks_per_domain: Optional[int] = None
    domain_order: Optional[List[str]] = None
    shuffle_within_domain: bool = False

    # Agent configuration
    agent_type: AgentType = AgentType.ICL_ER
    agent_llm: str = "gpt-4"
    agent_llm_args: Dict[str, Any] = Field(default_factory=dict)

    # ICL-ER specific config
    max_examples_in_prompt: int = 5
    memory_buffer_size: int = 1000
    min_reward_for_storage: float = 0.5
    retrieval_strategy: SamplingStrategy = SamplingStrategy.DIVERSITY
    enable_similarity_retrieval: bool = False

    # Training configuration
    episodes_per_task: int = 1
    max_steps_per_episode: int = 100

    # Evaluation configuration
    eval_frequency: int = 10
    eval_on_all_seen_tasks: bool = True
    num_eval_trials: int = 1

    # User simulator configuration
    user_llm: str = "gpt-4"
    user_llm_args: Dict[str, Any] = Field(default_factory=dict)

    # Logging
    save_trajectories: bool = True
    save_checkpoints: bool = True
    checkpoint_frequency: int = 20
    log_level: str = "INFO"


class CLExperimentState(BaseModel):
    """State of a continual learning experiment"""

    current_task_idx: int = 0
    current_phase: CLPhase = CLPhase.TRAINING
    tasks_completed: List[str] = Field(default_factory=list)
    total_episodes: int = 0
    total_steps: int = 0
    start_time: Optional[str] = None
    last_checkpoint_time: Optional[str] = None


class CLExperimentResult(BaseModel):
    """Result of a continual learning experiment"""

    config: CLExperimentConfig
    metrics: EvaluationResults
    state: CLExperimentState
    training_history: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_history: List[Dict[str, Any]] = Field(default_factory=list)
    duration_seconds: float = 0.0


class CLOrchestrator:
    """
    Main orchestrator for continual learning experiments.

    This class coordinates:
    - Curriculum management (task ordering)
    - Agent training and learning
    - Periodic evaluation
    - Metrics computation
    - Checkpointing and logging
    """

    def __init__(
        self,
        config: CLExperimentConfig,
    ):
        """
        Initialize the CL orchestrator.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.state = CLExperimentState()

        # Initialize components
        self._initialize_output_dir()
        self._initialize_tasks()
        self._initialize_curriculum()
        self._initialize_memory_buffer()
        self._initialize_metrics()

        # Agent will be created per-domain as tools differ
        self._agent: Optional[ContinualLearningAgent] = None
        self._current_domain: Optional[str] = None

        logger.info(
            f"Initialized CLOrchestrator: {config.name}, "
            f"{len(self._all_tasks)} tasks across {len(config.domains)} domains"
        )

    def _initialize_output_dir(self) -> None:
        """Create output directory structure."""
        self._output_dir = os.path.join(
            self.config.output_dir,
            self.config.name,
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self._output_dir, exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "trajectories"), exist_ok=True)

        # Save config
        config_path = os.path.join(self._output_dir, "config.json")
        with open(config_path, 'w') as f:
            f.write(self.config.model_dump_json(indent=2))

        logger.info(f"Output directory: {self._output_dir}")

    def _initialize_tasks(self) -> None:
        """Load tasks from all domains."""
        self._all_tasks: List[Task] = []
        self._domain_tasks: Dict[str, List[Task]] = {}
        self._environments: Dict[str, Callable] = {}

        for domain in self.config.domains:
            # Load tasks
            tasks = registry.get_tasks_loader(domain)()

            # Filter by num_tasks_per_domain if specified
            if self.config.num_tasks_per_domain:
                tasks = tasks[:self.config.num_tasks_per_domain]

            self._domain_tasks[domain] = tasks
            self._all_tasks.extend(tasks)

            # Store environment constructor
            self._environments[domain] = registry.get_env_constructor(domain)

            logger.info(f"Loaded {len(tasks)} tasks from domain '{domain}'")

    def _initialize_curriculum(self) -> None:
        """Initialize the curriculum manager."""
        strategy = self.config.curriculum_strategy

        if strategy == CurriculumStrategy.SEQUENTIAL:
            self._curriculum = SequentialCurriculum(
                tasks=self._all_tasks,
                domain_order=self.config.domain_order,
                shuffle_within_domain=self.config.shuffle_within_domain,
                seed=self.config.seed,
            )
        elif strategy == CurriculumStrategy.INTERLEAVED:
            self._curriculum = InterleavedCurriculum(
                tasks=self._all_tasks,
                interleave_pattern="round_robin",
                seed=self.config.seed,
            )
        elif strategy == CurriculumStrategy.DIFFICULTY_BASED:
            self._curriculum = DifficultyBasedCurriculum(
                tasks=self._all_tasks,
                progression="easy_to_hard",
                seed=self.config.seed,
            )
        else:
            # Default to sequential
            self._curriculum = SequentialCurriculum(
                tasks=self._all_tasks,
                seed=self.config.seed,
            )

        logger.info(
            f"Initialized curriculum: {strategy.value}, "
            f"{len(self._curriculum)} tasks"
        )

    def _initialize_memory_buffer(self) -> None:
        """Initialize the shared memory buffer."""
        self._memory_buffer = MemoryBuffer(
            max_size=self.config.memory_buffer_size,
            sampling_strategy=self.config.retrieval_strategy,
            min_reward_threshold=self.config.min_reward_for_storage,
            enable_embeddings=self.config.enable_similarity_retrieval,
            seed=self.config.seed,
        )

    def _initialize_metrics(self) -> None:
        """Initialize metrics tracking."""
        self._metrics = ContinualLearningMetrics()

        # Get task order and domains from curriculum
        task_order = [t.id for t in self._curriculum]
        domains = [
            self._curriculum.task_metadata[t.id].domain
            for t in self._curriculum
        ]

        self._metrics.initialize(task_order, domains)

        self._evaluator = CLEvaluator(
            metrics=self._metrics,
            eval_frequency=self.config.eval_frequency,
            eval_on_all_seen=self.config.eval_on_all_seen_tasks,
            num_eval_trials=self.config.num_eval_trials,
        )

    def _create_agent_for_domain(self, domain: str) -> ContinualLearningAgent:
        """
        Create an agent configured for a specific domain.

        Args:
            domain: Domain name

        Returns:
            Configured CL agent
        """
        # Get environment for tools and policy
        env = self._environments[domain]()
        tools = env.get_tools()
        policy = env.get_policy()

        if self.config.agent_type == AgentType.ICL_ER:
            agent = ICLExperienceReplayAgent(
                tools=tools,
                domain_policy=policy,
                llm=self.config.agent_llm,
                llm_args=self.config.agent_llm_args,
                memory_buffer=self._memory_buffer,
                max_examples_in_prompt=self.config.max_examples_in_prompt,
                retrieval_strategy=self.config.retrieval_strategy,
                min_reward_for_storage=self.config.min_reward_for_storage,
                enable_similarity_retrieval=self.config.enable_similarity_retrieval,
            )
        elif self.config.agent_type == AgentType.PROMPT_STRATEGY:
            agent = PromptStrategyAgent(
                tools=tools,
                domain_policy=policy,
                llm=self.config.agent_llm,
                llm_args=self.config.agent_llm_args,
                memory_buffer=self._memory_buffer,
                max_examples_in_prompt=self.config.max_examples_in_prompt,
            )
        else:
            # Baseline: ICL-ER without storing experiences
            agent = ICLExperienceReplayAgent(
                tools=tools,
                domain_policy=policy,
                llm=self.config.agent_llm,
                llm_args=self.config.agent_llm_args,
                memory_buffer=self._memory_buffer,
                max_examples_in_prompt=0,  # No few-shot
                min_reward_for_storage=2.0,  # Never store (impossible reward)
            )

        agent.set_seed(self.config.seed)
        return agent

    def _get_agent_for_task(self, task: Task) -> ContinualLearningAgent:
        """Get or create agent for the task's domain."""
        domain = self._curriculum.task_metadata[task.id].domain

        if self._current_domain != domain:
            logger.info(f"Switching to domain: {domain}")
            self._agent = self._create_agent_for_domain(domain)
            self._current_domain = domain

        return self._agent

    def _run_task_episode(
        self,
        task: Task,
        agent: ContinualLearningAgent,
        is_evaluation: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a single episode on a task.

        Args:
            task: The task to run
            agent: The agent to use
            is_evaluation: Whether this is an evaluation episode

        Returns:
            Episode result dictionary
        """
        domain = self._curriculum.task_metadata[task.id].domain

        # Create environment
        env = self._environments[domain]()

        # Create user simulator
        user = UserSimulator(
            tools=env.get_user_tools() if env.user_tools else None,
            instructions=task.user_scenario,
            llm=self.config.user_llm,
            llm_args=self.config.user_llm_args,
        )

        # Set task context on agent
        agent.set_task_context(task.id, domain)

        # Create tau2 orchestrator
        tau2_orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=self.config.max_steps_per_episode,
            seed=self.config.seed,
        )

        # Run simulation
        start_time = time.perf_counter()
        simulation: SimulationRun = tau2_orchestrator.run()
        duration = time.perf_counter() - start_time

        # Evaluate
        eval_result = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=domain,
        )

        reward = eval_result.reward
        success = reward >= 0.9  # Consider 0.9+ as success

        # Record metrics
        self._metrics.record_task_result(
            task_id=task.id,
            domain=domain,
            reward=reward,
            success=success,
            num_steps=len(simulation.messages),
            metadata={"duration": duration}
        )

        # Learn from trajectory (if not evaluation)
        learning_stats = {}
        if not is_evaluation and self.config.agent_type != AgentType.BASELINE:
            learning_stats = agent.learn_from_trajectory(
                task_id=task.id,
                domain=domain,
                trajectory=simulation.messages,
                reward=reward,
                success=success,
            )

        # Save trajectory if configured
        if self.config.save_trajectories:
            self._save_trajectory(task.id, simulation, is_evaluation)

        result = {
            "task_id": task.id,
            "domain": domain,
            "reward": reward,
            "success": success,
            "num_steps": len(simulation.messages),
            "duration": duration,
            "is_evaluation": is_evaluation,
            "learning_stats": learning_stats,
            "termination_reason": simulation.termination_reason,
        }

        logger.info(
            f"{'[EVAL]' if is_evaluation else '[TRAIN]'} "
            f"Task {task.id}: reward={reward:.3f}, "
            f"steps={len(simulation.messages)}, "
            f"duration={duration:.2f}s"
        )

        return result

    def _save_trajectory(
        self,
        task_id: str,
        simulation: SimulationRun,
        is_evaluation: bool
    ) -> None:
        """Save trajectory to file."""
        prefix = "eval" if is_evaluation else "train"
        filename = f"{prefix}_{task_id}_{datetime.now().strftime('%H%M%S')}.json"
        path = os.path.join(self._output_dir, "trajectories", filename)

        with open(path, 'w') as f:
            f.write(simulation.model_dump_json(indent=2))

    def _save_checkpoint(self) -> None:
        """Save experiment checkpoint."""
        checkpoint_dir = os.path.join(self._output_dir, "checkpoints")

        # Save state
        state_path = os.path.join(checkpoint_dir, "state.json")
        with open(state_path, 'w') as f:
            f.write(self.state.model_dump_json(indent=2))

        # Save memory buffer
        buffer_path = os.path.join(checkpoint_dir, "memory_buffer.json")
        self._memory_buffer.save(buffer_path)

        # Save metrics
        metrics_path = os.path.join(checkpoint_dir, "metrics.json")
        results = self._metrics.compute_metrics()
        with open(metrics_path, 'w') as f:
            f.write(results.model_dump_json(indent=2))

        self.state.last_checkpoint_time = datetime.now().isoformat()
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def run(self) -> CLExperimentResult:
        """
        Run the complete continual learning experiment.

        Returns:
            CLExperimentResult with all metrics and history
        """
        self.state.start_time = datetime.now().isoformat()
        start_time = time.perf_counter()

        training_history = []
        evaluation_history = []

        logger.info("=" * 60)
        logger.info(f"Starting CL experiment: {self.config.name}")
        logger.info("=" * 60)

        # Main training loop
        task_idx = 0
        for task in self._curriculum:
            self.state.current_task_idx = task_idx
            domain = self._curriculum.task_metadata[task.id].domain

            logger.info(f"\n--- Task {task_idx + 1}/{len(self._curriculum)}: {task.id} ---")

            # Get agent
            agent = self._get_agent_for_task(task)

            # Run training episode(s)
            self.state.current_phase = CLPhase.TRAINING
            for episode in range(self.config.episodes_per_task):
                result = self._run_task_episode(task, agent, is_evaluation=False)
                training_history.append(result)
                self.state.total_episodes += 1
                self.state.total_steps += result["num_steps"]

            self.state.tasks_completed.append(task.id)
            self._evaluator.record_task_learned(task.id)

            # Periodic evaluation
            if self._evaluator.should_evaluate():
                self.state.current_phase = CLPhase.EVALUATION
                tasks_to_eval = self._evaluator.get_tasks_to_evaluate()

                logger.info(f"\n--- Evaluation on {len(tasks_to_eval)} tasks ---")

                for eval_task_id in tasks_to_eval:
                    eval_task = self._curriculum.get_task_by_id(eval_task_id)
                    if eval_task:
                        eval_agent = self._get_agent_for_task(eval_task)
                        eval_result = self._run_task_episode(
                            eval_task, eval_agent, is_evaluation=True
                        )
                        evaluation_history.append(eval_result)

                        # Record in performance matrix
                        self._metrics.record_evaluation(
                            eval_task_id, task_idx, eval_result["reward"]
                        )

                self._evaluator.record_evaluation_done()

            # Checkpoint
            if self.config.save_checkpoints:
                if (task_idx + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint()

            task_idx += 1

        # Final evaluation on all tasks
        logger.info("\n" + "=" * 60)
        logger.info("Final Evaluation")
        logger.info("=" * 60)

        self.state.current_phase = CLPhase.EVALUATION
        for task in self._curriculum:
            agent = self._get_agent_for_task(task)
            eval_result = self._run_task_episode(task, agent, is_evaluation=True)
            evaluation_history.append(eval_result)
            self._metrics.record_evaluation(
                task.id, len(self._curriculum) - 1, eval_result["reward"]
            )

        # Compute final metrics
        final_metrics = self._metrics.compute_metrics()
        duration = time.perf_counter() - start_time

        # Log summary
        logger.info("\n" + self._metrics.get_summary())

        # Save final results
        result = CLExperimentResult(
            config=self.config,
            metrics=final_metrics,
            state=self.state,
            training_history=training_history,
            evaluation_history=evaluation_history,
            duration_seconds=duration,
        )

        result_path = os.path.join(self._output_dir, "results.json")
        with open(result_path, 'w') as f:
            f.write(result.model_dump_json(indent=2))

        logger.info(f"\nExperiment complete. Results saved to {self._output_dir}")

        return result


def run_cl_experiment(config: CLExperimentConfig) -> CLExperimentResult:
    """
    Convenience function to run a CL experiment.

    Args:
        config: Experiment configuration

    Returns:
        Experiment results
    """
    orchestrator = CLOrchestrator(config)
    return orchestrator.run()
