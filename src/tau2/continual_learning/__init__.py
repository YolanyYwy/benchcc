# Copyright Sierra
# Continual Learning Module for Tau2-Bench
# Designed for parameter-free API Agents

"""
Tau2-CL: Continual Learning Framework for Tool Use Agents

This module provides a framework for evaluating continual learning algorithms
specifically designed for parameter-free API agents in tool use scenarios.

Main components:
- curriculum: Task sequencing and curriculum management
- memory: Experience buffer and retrieval systems
- agents: ICL-based continual learning agents
- metrics: CL-specific evaluation metrics
- orchestrator: Experiment coordination

Usage:
    from tau2.continual_learning import (
        CLOrchestrator,
        CLExperimentConfig,
        ICLExperienceReplayAgent,
        MemoryBuffer,
    )

    # Create config
    config = CLExperimentConfig(
        name="my_experiment",
        domains=["airline", "retail"],
        curriculum_strategy="sequential",
        agent_type="icl_er",
    )

    # Run experiment
    orchestrator = CLOrchestrator(config)
    results = orchestrator.run()
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    # Curriculum
    "CurriculumManager",
    "CurriculumStrategy",
    "TaskMetadata",
    "SequentialCurriculum",
    "InterleavedCurriculum",
    "DifficultyBasedCurriculum",
    # Memory
    "Experience",
    "MemoryBuffer",
    "SamplingStrategy",
    # Agents
    "ContinualLearningAgent",
    "CLAgentState",
    "ICLExperienceReplayAgent",
    "PromptStrategyAgent",
    # Metrics
    "ContinualLearningMetrics",
    "CLEvaluator",
    "EvaluationResults",
    # Orchestrator
    "CLOrchestrator",
    "CLExperimentConfig",
    "CLPhase",
    "AgentType",
]


def __getattr__(name):
    """Lazy loading of submodules to avoid circular imports."""

    # Curriculum imports
    if name in ("CurriculumManager", "CurriculumStrategy", "TaskMetadata"):
        from tau2.continual_learning.curriculum.base import (
            CurriculumManager,
            CurriculumStrategy,
            TaskMetadata,
        )
        return locals()[name]

    if name == "SequentialCurriculum":
        from tau2.continual_learning.curriculum.sequential import SequentialCurriculum
        return SequentialCurriculum

    if name == "InterleavedCurriculum":
        from tau2.continual_learning.curriculum.interleaved import InterleavedCurriculum
        return InterleavedCurriculum

    if name == "DifficultyBasedCurriculum":
        from tau2.continual_learning.curriculum.difficulty_based import DifficultyBasedCurriculum
        return DifficultyBasedCurriculum

    # Memory imports
    if name in ("Experience", "MemoryBuffer", "SamplingStrategy"):
        from tau2.continual_learning.memory.buffer import (
            Experience,
            MemoryBuffer,
            SamplingStrategy,
        )
        return locals()[name]

    # Agent imports
    if name in ("ContinualLearningAgent", "CLAgentState"):
        from tau2.continual_learning.agents.base import (
            ContinualLearningAgent,
            CLAgentState,
        )
        return locals()[name]

    if name == "ICLExperienceReplayAgent":
        from tau2.continual_learning.agents.icl_experience_replay import ICLExperienceReplayAgent
        return ICLExperienceReplayAgent

    if name == "PromptStrategyAgent":
        from tau2.continual_learning.agents.prompt_strategy import PromptStrategyAgent
        return PromptStrategyAgent

    # Metrics imports
    if name in ("ContinualLearningMetrics", "CLEvaluator", "EvaluationResults"):
        from tau2.continual_learning.metrics.metrics import (
            ContinualLearningMetrics,
            CLEvaluator,
            EvaluationResults,
        )
        return locals()[name]

    # Orchestrator imports
    if name in ("CLOrchestrator", "CLExperimentConfig", "CLPhase", "AgentType"):
        from tau2.continual_learning.orchestrator import (
            CLOrchestrator,
            CLExperimentConfig,
            CLPhase,
            AgentType,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
