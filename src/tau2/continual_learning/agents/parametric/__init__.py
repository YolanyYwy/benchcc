# Copyright Sierra
"""
Parametric Continual Learning Agents

This module implements continual learning agents with learnable parameters
at the Agent and Memory layers, while keeping the LLM backbone frozen.

Available methods:
1. EWC (Elastic Weight Consolidation) - Fisher Information-based forgetting prevention
2. Replay - Gradient-level experience replay
3. Parameter Isolation - Task-specific parameter subsets with routing
4. Progressive/Modular - Module expansion with freezing
5. Meta-CL - Meta-learning for continual learning
"""

from .tool_scorer import ToolScorer
from .parametric_memory import ParametricMemory
from .base import ParametricCLAgent, ParametricCLAgentState
from .ewc_agent import EWCContinualLearningAgent, create_ewc_agent
from .replay_agent import ReplayContinualLearningAgent, create_replay_agent
from .parameter_isolation_agent import ParameterIsolationAgent, create_parameter_isolation_agent
from .progressive_agent import ProgressiveModularAgent, create_progressive_agent
from .meta_cl_agent import MetaContinualLearningAgent, create_meta_cl_agent

__all__ = [
    # Core components
    "ToolScorer",
    "ParametricMemory",
    "ParametricCLAgent",
    "ParametricCLAgentState",

    # Method 1: EWC
    "EWCContinualLearningAgent",
    "create_ewc_agent",

    # Method 2: Replay
    "ReplayContinualLearningAgent",
    "create_replay_agent",

    # Method 3: Parameter Isolation
    "ParameterIsolationAgent",
    "create_parameter_isolation_agent",

    # Method 4: Progressive/Modular
    "ProgressiveModularAgent",
    "create_progressive_agent",

    # Method 5: Meta-CL
    "MetaContinualLearningAgent",
    "create_meta_cl_agent",
]
