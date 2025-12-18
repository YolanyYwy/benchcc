#!/usr/bin/env python3
"""
快速开始脚本 - 演示如何使用参数化持续学习Agent

这是最简单的例子，展示基本用法。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tau2.environment.tool import Tool
from tau2.continual_learning.agents.parametric import (
    EWCContinualLearningAgent,
)
from tau2.data_model.message import (
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolMessage,
)


# ============================================================================
# 1. 定义简单的工具
# ============================================================================

def search_database(query: str) -> str:
    """搜索客户数据库"""
    return f"找到了关于 '{query}' 的3条结果"


def send_email(to: str, subject: str) -> str:
    """发送邮件给客户"""
    return f"邮件已发送给 {to}: {subject}"


# ============================================================================
# 2. 创建Agent
# ============================================================================

def main():
    print("=" * 80)
    print("参数化持续学习 - 快速开始")
    print("=" * 80)

    # 创建工具
    tools = [
        Tool(search_database),
        Tool(send_email),
    ]

    # 创建domain policy
    policy = """
你是一个客服助手。
可以搜索数据库和发送邮件。
要友好和专业。
    """.strip()

    # 创建EWC Agent
    print("\n1. 创建EWC Agent...")
    agent = EWCContinualLearningAgent(
        tools=tools,
        domain_policy=policy,
        llm="gpt-4",
        embedding_dim=768,
        learning_rate=0.01,
        ewc_lambda=1.0,
        online_ewc=True,
    )
    print(f"   ✓ Agent创建成功: {agent.__class__.__name__}")

    # ============================================================================
    # 3. 模拟一个简单的任务
    # ============================================================================

    print("\n2. 模拟任务执行...")

    # 创建一个模拟轨迹
    trajectory = [
        UserMessage(role="user", content="我想查询订单 #12345"),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_database",
                    arguments={"query": "订单 #12345"}
                )
            ]
        ),
        ToolMessage(
            role="tool",
            tool_call_id="call_1",
            content="找到了关于 '订单 #12345' 的3条结果"
        ),
        AssistantMessage(
            role="assistant",
            content="我找到了您的订单 #12345，状态是待处理。"
        ),
    ]

    print(f"   ✓ 创建了包含 {len(trajectory)} 条消息的轨迹")

    # ============================================================================
    # 4. Agent学习（关键步骤！）
    # ============================================================================

    print("\n3. Agent从轨迹中学习...")

    learning_stats = agent.learn_from_trajectory(
        task_id="task_001",
        domain="customer_service",
        trajectory=trajectory,
        reward=1.0,  # 任务成功
        success=True,
    )

    print(f"   ✓ 学习完成!")
    print(f"   - 参数更新次数: {learning_stats.get('parameter_updates', 0)}")
    print(f"   - 经验添加数: {learning_stats.get('experiences_added', 0)}")

    if 'fisher_computation' in learning_stats:
        fisher_stats = learning_stats['fisher_computation']
        if fisher_stats.get('computed'):
            print(f"   - Fisher Information已计算 (样本数: {fisher_stats.get('num_samples', 0)})")

    # ============================================================================
    # 5. 查看Agent统计
    # ============================================================================

    print("\n4. Agent统计信息:")
    stats = agent.get_statistics()

    print(f"   - 完成任务数: {stats['tasks_completed']}")
    print(f"   - 总步骤数: {stats['total_steps']}")
    print(f"   - 学习的任务数: {stats.get('num_tasks_learned', 0)}")
    print(f"   - 当前λ值: {stats.get('current_lambda', 0):.3f}")

    # Tool Scorer统计
    if 'tool_scorer_stats' in stats:
        ts_stats = stats['tool_scorer_stats']
        print(f"\n   Tool Scorer:")
        print(f"   - 总更新次数: {ts_stats['total_updates']}")
        print(f"   - 权重范数: {ts_stats['weights_norm']:.4f}")
        print(f"   - 工具选择次数: {ts_stats['tool_selection_counts']}")

    # Memory统计
    if 'memory_buffer_stats' in stats:
        mem_stats = stats['memory_buffer_stats']
        print(f"\n   Memory Buffer:")
        print(f"   - 经验总数: {mem_stats['total_experiences']}")
        print(f"   - 平均奖励: {mem_stats['avg_reward']:.3f}")

    # ============================================================================
    # 6. 再学习几个任务
    # ============================================================================

    print("\n5. 继续学习更多任务...")

    for i in range(2, 4):
        # 创建新轨迹
        new_trajectory = [
            UserMessage(role="user", content=f"我需要帮助处理问题 #{i}"),
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"call_{i}",
                        name="search_database",
                        arguments={"query": f"问题 #{i}"}
                    )
                ]
            ),
            ToolMessage(
                role="tool",
                tool_call_id=f"call_{i}",
                content=f"找到了关于 '问题 #{i}' 的3条结果"
            ),
            AssistantMessage(
                role="assistant",
                content=f"我已经找到了问题 #{i} 的信息。"
            ),
        ]

        stats = agent.learn_from_trajectory(
            task_id=f"task_{i:03d}",
            domain="customer_service",
            trajectory=new_trajectory,
            reward=1.0,
            success=True,
        )

        print(f"   ✓ Task {i}: 参数更新 {stats.get('parameter_updates', 0)} 次")

    # ============================================================================
    # 7. 最终统计
    # ============================================================================

    print("\n6. 最终统计:")
    final_stats = agent.get_statistics()

    print(f"   - 总完成任务: {final_stats['tasks_completed']}")
    print(f"   - 学习任务数: {final_stats.get('num_tasks_learned', 0)}")

    if 'cumulative_fisher_stats' in final_stats:
        fisher_stats = final_stats['cumulative_fisher_stats']
        print(f"\n   Fisher Information (累积):")
        print(f"   - 平均值: {fisher_stats['mean']:.6f}")
        print(f"   - 重要参数数: {fisher_stats['num_important']}")

    print("\n" + "=" * 80)
    print("✓ 演示完成！")
    print("=" * 80)

    print("\n💡 关键要点:")
    print("  1. Agent有可学习的参数（Tool Scorer权重 w_i）")
    print("  2. 每次learn_from_trajectory都会更新参数")
    print("  3. Fisher Information保护重要参数防止遗忘")
    print("  4. 这是真正的持续学习，不只是prompt engineering！")

    print("\n📖 下一步:")
    print("  - 查看 README.md 了解完整功能")
    print("  - 运行 example_usage.py 查看更多示例")
    print("  - 尝试其他方法: Replay, Parameter Isolation, Progressive, Meta-CL")


if __name__ == "__main__":
    main()
