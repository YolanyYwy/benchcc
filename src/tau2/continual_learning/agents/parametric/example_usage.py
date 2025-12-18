#!/usr/bin/env python3
# Copyright Sierra
"""
Example Usage of Parametric Continual Learning Agents

This script demonstrates how to use the parametric CL agents
(EWC and Replay) for true continual learning with learnable parameters.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from tau2.environment.tool import Tool
from tau2.continual_learning.agents.parametric import (
    EWCContinualLearningAgent,
    ReplayContinualLearningAgent,
)
from tau2.continual_learning.agents.icl_experience_replay import ICLExperienceReplayAgent


# ============================================================================
# Example Tool Definitions
# ============================================================================

def search_database(query: str) -> str:
    """Search the customer database."""
    logger.info(f"Searching database for: {query}")
    return f"Found 3 results for '{query}'"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer."""
    logger.info(f"Sending email to {to}: {subject}")
    return f"Email sent successfully to {to}"


def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """Create a support ticket."""
    logger.info(f"Creating ticket: {title} (priority: {priority})")
    return f"Ticket #{hash(title) % 10000} created"


# ============================================================================
# Helper Functions
# ============================================================================

def create_tools() -> List[Tool]:
    """Create example tools."""
    return [
        Tool(search_database),
        Tool(send_email),
        Tool(create_ticket),
    ]


def create_domain_policy() -> str:
    """Create example domain policy."""
    return """
You are a helpful customer service agent.

Your responsibilities:
1. Answer customer questions using the database
2. Send emails to customers when needed
3. Create support tickets for complex issues

Always be polite and professional.
    """.strip()


def create_mock_trajectory() -> Dict[str, Any]:
    """Create a mock trajectory for testing."""
    from tau2.data_model.message import (
        UserMessage,
        AssistantMessage,
        ToolCall,
        ToolMessage,
    )

    trajectory = [
        UserMessage(role="user", content="I need help with my order #12345"),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_database",
                    arguments={"query": "order #12345"}
                )
            ]
        ),
        ToolMessage(
            role="tool",
            tool_call_id="call_1",
            content="Found order #12345: Status is pending"
        ),
        AssistantMessage(
            role="assistant",
            content="I found your order #12345. It's currently pending."
        ),
    ]

    return {
        "task_id": "task_001",
        "domain": "customer_service",
        "trajectory": trajectory,
        "reward": 1.0,
        "success": True,
    }


# ============================================================================
# Example 1: Basic EWC Agent Usage
# ============================================================================

def example_1_basic_ewc():
    """Example 1: Basic EWC agent usage."""
    print("\n" + "=" * 80)
    print("Example 1: Basic EWC Agent Usage")
    print("=" * 80 + "\n")

    # Create tools and policy
    tools = create_tools()
    policy = create_domain_policy()

    # Create EWC agent
    agent = EWCContinualLearningAgent(
        tools=tools,
        domain_policy=policy,
        llm="gpt-4",
        embedding_dim=768,
        learning_rate=0.01,
        ewc_lambda=1.0,
        online_ewc=True,
        ewc_lambda_growth="adaptive",
        fisher_sample_size=100,
    )

    logger.info(f"Created EWC agent: {agent.__class__.__name__}")

    # Simulate learning from multiple trajectories
    for i in range(5):
        mock_data = create_mock_trajectory()
        mock_data["task_id"] = f"task_{i:03d}"

        logger.info(f"\n--- Learning from task {mock_data['task_id']} ---")

        stats = agent.learn_from_trajectory(
            task_id=mock_data["task_id"],
            domain=mock_data["domain"],
            trajectory=mock_data["trajectory"],
            reward=mock_data["reward"],
            success=mock_data["success"],
        )

        logger.info(f"Learning stats: {stats}")

    # Get agent statistics
    agent_stats = agent.get_statistics()
    logger.info(f"\nAgent statistics:")
    logger.info(f"  - Tasks completed: {agent_stats['tasks_completed']}")
    logger.info(f"  - Total steps: {agent_stats['total_steps']}")
    logger.info(f"  - Num tasks learned: {agent_stats['num_tasks_learned']}")
    logger.info(f"  - Current λ: {agent_stats['current_lambda']:.3f}")

    if "cumulative_fisher_stats" in agent_stats:
        fisher_stats = agent_stats["cumulative_fisher_stats"]
        logger.info(f"  - Fisher mean: {fisher_stats['mean']:.6f}")
        logger.info(f"  - Important params: {fisher_stats['num_important']}")

    return agent


# ============================================================================
# Example 2: Basic Replay Agent Usage
# ============================================================================

def example_2_basic_replay():
    """Example 2: Basic Replay agent usage."""
    print("\n" + "=" * 80)
    print("Example 2: Basic Replay Agent Usage")
    print("=" * 80 + "\n")

    # Create tools and policy
    tools = create_tools()
    policy = create_domain_policy()

    # Create Replay agent
    agent = ReplayContinualLearningAgent(
        tools=tools,
        domain_policy=policy,
        llm="gpt-4",
        embedding_dim=768,
        learning_rate=0.01,
        replay_ratio=0.5,
        replay_batch_size=5,
        replay_strategy="importance",
        update_memory_importance=True,
        replay_frequency=1,
    )

    logger.info(f"Created Replay agent: {agent.__class__.__name__}")

    # Simulate learning from multiple trajectories
    for i in range(5):
        mock_data = create_mock_trajectory()
        mock_data["task_id"] = f"task_{i:03d}"

        logger.info(f"\n--- Learning from task {mock_data['task_id']} ---")

        stats = agent.learn_from_trajectory(
            task_id=mock_data["task_id"],
            domain=mock_data["domain"],
            trajectory=mock_data["trajectory"],
            reward=mock_data["reward"],
            success=mock_data["success"],
        )

        logger.info(f"Learning stats: {stats}")

    # Get agent statistics
    agent_stats = agent.get_statistics()
    logger.info(f"\nAgent statistics:")
    logger.info(f"  - Tasks completed: {agent_stats['tasks_completed']}")
    logger.info(f"  - Total steps: {agent_stats['total_steps']}")
    logger.info(f"  - Total replay updates: {agent_stats['total_replay_updates']}")
    logger.info(f"  - Replay ratio: {agent_stats['replay_ratio']}")

    # Memory statistics
    if "parametric_memory_stats" in agent_stats:
        mem_stats = agent_stats["parametric_memory_stats"]
        logger.info(f"  - Memory size: {mem_stats['total_experiences']}")
        logger.info(f"  - Importance mean: {mem_stats['importance_mean']:.3f}")
        logger.info(f"  - Importance range: [{mem_stats['importance_min']:.3f}, {mem_stats['importance_max']:.3f}]")

    return agent


# ============================================================================
# Example 3: Comparing with ICL-ER (Non-parametric)
# ============================================================================

def example_3_compare_with_icl_er():
    """Example 3: Compare parametric agents with ICL-ER."""
    print("\n" + "=" * 80)
    print("Example 3: Comparison with ICL-ER (Non-parametric)")
    print("=" * 80 + "\n")

    tools = create_tools()
    policy = create_domain_policy()

    # Create ICL-ER agent (non-parametric)
    icl_agent = ICLExperienceReplayAgent(
        tools=tools,
        domain_policy=policy,
        llm="gpt-4",
        max_examples_in_prompt=5,
    )

    # Create EWC agent (parametric)
    ewc_agent = EWCContinualLearningAgent(
        tools=tools,
        domain_policy=policy,
        llm="gpt-4",
        embedding_dim=768,
        learning_rate=0.01,
        ewc_lambda=1.0,
    )

    logger.info("Created both agents for comparison:")
    logger.info(f"  - ICL-ER: {icl_agent.__class__.__name__} (non-parametric)")
    logger.info(f"  - EWC: {ewc_agent.__class__.__name__} (parametric)")

    # Simulate learning
    mock_data = create_mock_trajectory()

    logger.info("\n--- Learning with ICL-ER ---")
    icl_stats = icl_agent.learn_from_trajectory(
        task_id=mock_data["task_id"],
        domain=mock_data["domain"],
        trajectory=mock_data["trajectory"],
        reward=mock_data["reward"],
        success=mock_data["success"],
    )
    logger.info(f"ICL-ER stats: {icl_stats}")

    logger.info("\n--- Learning with EWC ---")
    ewc_stats = ewc_agent.learn_from_trajectory(
        task_id=mock_data["task_id"],
        domain=mock_data["domain"],
        trajectory=mock_data["trajectory"],
        reward=mock_data["reward"],
        success=mock_data["success"],
    )
    logger.info(f"EWC stats: {ewc_stats}")

    logger.info("\n" + "-" * 80)
    logger.info("Key Differences:")
    logger.info("-" * 80)
    logger.info("ICL-ER:")
    logger.info("  ❌ No learnable parameters")
    logger.info("  ❌ No parameter updates")
    logger.info("  ❌ Only stores experiences in memory")
    logger.info("  ❌ Only uses examples in prompt")

    logger.info("\nEWC Agent:")
    logger.info("  ✅ Learnable Tool Scorer weights w_i")
    logger.info("  ✅ Learnable Memory importance weights α_i")
    logger.info(f"  ✅ Parameter updates: {ewc_stats.get('parameter_updates', 0)}")
    logger.info("  ✅ Fisher Information for forgetting prevention")


# ============================================================================
# Example 4: Saving and Loading Agent State
# ============================================================================

def example_4_save_and_load():
    """Example 4: Saving and loading agent state."""
    print("\n" + "=" * 80)
    print("Example 4: Saving and Loading Agent State")
    print("=" * 80 + "\n")

    tools = create_tools()
    policy = create_domain_policy()

    # Create and train agent
    agent = EWCContinualLearningAgent(
        tools=tools,
        domain_policy=policy,
        llm="gpt-4",
        embedding_dim=768,
    )

    # Train on a task
    mock_data = create_mock_trajectory()
    agent.learn_from_trajectory(
        task_id=mock_data["task_id"],
        domain=mock_data["domain"],
        trajectory=mock_data["trajectory"],
        reward=mock_data["reward"],
        success=mock_data["success"],
    )

    # Save state
    save_path = "/tmp/test_agent.json"
    agent.save_state(save_path)
    logger.info(f"Saved agent state to {save_path}")

    # Create new agent and load state
    new_agent = EWCContinualLearningAgent(
        tools=tools,
        domain_policy=policy,
        llm="gpt-4",
        embedding_dim=768,
    )

    new_agent.load_state(save_path)
    logger.info(f"Loaded agent state from {save_path}")

    # Compare statistics
    old_stats = agent.get_statistics()
    new_stats = new_agent.get_statistics()

    logger.info("\nStatistics match:")
    logger.info(f"  - Tasks completed: {old_stats['tasks_completed']} == {new_stats['tasks_completed']}")
    logger.info(f"  - Total steps: {old_stats['total_steps']} == {new_stats['total_steps']}")


# ============================================================================
# Example 5: Analyzing Tool Scorer Parameters
# ============================================================================

def example_5_analyze_tool_scorer():
    """Example 5: Analyzing Tool Scorer parameters."""
    print("\n" + "=" * 80)
    print("Example 5: Analyzing Tool Scorer Parameters")
    print("=" * 80 + "\n")

    tools = create_tools()
    policy = create_domain_policy()

    agent = EWCContinualLearningAgent(
        tools=tools,
        domain_policy=policy,
        llm="gpt-4",
        embedding_dim=768,
    )

    # Get tool scorer
    scorer = agent.tool_scorer

    logger.info(f"Tool Scorer initialized with {len(scorer.tools)} tools:")
    for i, tool_name in enumerate(scorer.tool_names):
        weight_norm = float(np.linalg.norm(scorer.weights[i]))
        logger.info(f"  - {tool_name}: weight_norm = {weight_norm:.4f}")

    # Train on several tasks
    logger.info("\nTraining on 3 tasks...")
    for i in range(3):
        mock_data = create_mock_trajectory()
        mock_data["task_id"] = f"task_{i:03d}"
        agent.learn_from_trajectory(**mock_data)

    # Analyze after training
    logger.info("\nTool Scorer statistics after training:")
    scorer_stats = scorer.get_statistics()
    logger.info(f"  - Total updates: {scorer_stats['total_updates']}")
    logger.info(f"  - Weights norm: {scorer_stats['weights_norm']:.4f}")
    logger.info(f"  - Weights mean: {scorer_stats['weights_mean']:.4f}")
    logger.info(f"  - Weights std: {scorer_stats['weights_std']:.4f}")

    logger.info("\nTool selection counts:")
    for tool_name, count in scorer_stats['tool_selection_counts'].items():
        logger.info(f"  - {tool_name}: {count} times")

    # Get parameters
    params = scorer.get_parameters()
    logger.info(f"\nParameter shapes:")
    logger.info(f"  - Weights: {params['weights'].shape}")
    if params['fisher_information'] is not None:
        logger.info(f"  - Fisher: {params['fisher_information'].shape}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    import numpy as np

    logger.info("Starting Parametric Continual Learning Examples")
    logger.info("=" * 80)

    # Run examples
    try:
        example_1_basic_ewc()
        example_2_basic_replay()
        example_3_compare_with_icl_er()
        example_4_save_and_load()

        # This example needs numpy
        import numpy as np  # noqa: F401
        example_5_analyze_tool_scorer()

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "=" * 80)
    logger.info("All examples completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
