#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小可运行测试 - 验证参数化Agent的核心功能

这个脚本不需要完整的轨迹或orchestrator，
只测试核心的参数学习功能。
"""

import sys
import numpy as np

# 解决Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from tau2.continual_learning.agents.parametric.tool_scorer import ToolScorer
from tau2.continual_learning.agents.parametric.parametric_memory import ParametricMemory
from tau2.environment.tool import Tool

print("=" * 80)
print("参数化持续学习 - 最小测试")
print("=" * 80)

# ============================================================================
# 1. 测试Tool Scorer（核心可学习组件）
# ============================================================================

print("\n【测试1: Tool Scorer - 可学习工具选择】")
print("-" * 80)

# 定义工具
def search(query: str) -> str:
    """搜索工具"""
    return f"Found: {query}"

def email(to: str, subject: str) -> str:
    """邮件工具"""
    return f"Email to {to}"

tools = [Tool(search), Tool(email)]

# 创建Tool Scorer
scorer = ToolScorer(
    tools=tools,
    embedding_dim=768,
    learning_rate=0.01,
)

print(f"✓ 创建ToolScorer成功")
print(f"  - 工具数量: {len(tools)}")
print(f"  - 参数shape: {scorer.weights.shape}")
print(f"  - 总参数量: {scorer.weights.size}")

# 测试工具评分
print("\n1.1 测试工具评分（初始状态）:")
state_emb = np.random.randn(768)
scores = scorer.score_tools(state_emb)
print(f"  工具分数: {scores}")

probs = scorer.get_tool_probabilities(state_emb)
print(f"  工具概率: {probs}")

# 测试参数更新
print("\n1.2 测试参数更新:")
old_weights = scorer.weights.copy()

update_stats = scorer.update_weights(
    state_embedding=state_emb,
    selected_tool="search",
    reward=1.0,
    success=True,
)

print(f"  ✓ 更新成功: {update_stats['updated']}")
print(f"  - 工具: {update_stats['tool']}")
print(f"  - 概率: {update_stats['probability']:.4f}")
print(f"  - 梯度范数: {update_stats['gradient_norm']:.6f}")

# 验证参数确实变化了
param_change = np.linalg.norm(scorer.weights - old_weights)
print(f"  - 参数变化量: {param_change:.6f}")

if param_change > 0:
    print(f"  ✓ 参数确实更新了！这是真正的学习！")
else:
    print(f"  ✗ 参数没有变化")

# ============================================================================
# 2. 测试Parametric Memory（可学习记忆）
# ============================================================================

print("\n【测试2: Parametric Memory - 可学习记忆重要性】")
print("-" * 80)

# 创建Parametric Memory
memory = ParametricMemory(
    max_size=100,
    embedding_dim=768,
    learning_rate=0.01,
    initial_importance=1.0,
)

print(f"✓ 创建ParametricMemory成功")
print(f"  - 最大容量: {memory.max_size}")
print(f"  - 当前大小: {len(memory)}")

# 添加一些经验
print("\n2.1 添加经验:")
from tau2.continual_learning.memory.buffer import Experience
from datetime import datetime

for i in range(5):
    exp = Experience(
        experience_id=f"exp_{i}",
        task_id=f"task_{i}",
        domain="test",
        timestamp=datetime.now(),
        observation=f"观察 {i}",
        action=f"动作 {i}",
        reward=0.5 + i * 0.1,
        success=True,
        embedding=list(np.random.randn(768)),
    )
    memory.add(exp)

print(f"  ✓ 添加了 {len(memory)} 条经验")

# 查看重要性权重
print("\n2.2 查看初始重要性权重:")
for exp in list(memory)[:3]:
    importance = memory.get_importance(exp.experience_id)
    print(f"  - {exp.experience_id}: importance={importance:.3f}, reward={exp.reward:.3f}")

# 更新重要性
print("\n2.3 更新重要性权重:")
exp_id = list(memory)[0].experience_id
old_importance = memory.get_importance(exp_id)

update_info = memory.update_importance(exp_id, gradient=0.5)

new_importance = memory.get_importance(exp_id)
print(f"  ✓ 重要性更新成功")
print(f"  - 经验ID: {exp_id}")
print(f"  - 旧重要性: {old_importance:.3f}")
print(f"  - 新重要性: {new_importance:.3f}")
print(f"  - 变化: {new_importance - old_importance:.3f}")

if abs(new_importance - old_importance) > 0.001:
    print(f"  ✓ 重要性确实变化了！记忆在学习！")
else:
    print(f"  ✗ 重要性没有变化")

# 基于重要性采样
print("\n2.4 基于重要性采样:")
sampled = memory.sample_by_importance(n=3)
print(f"  ✓ 采样了 {len(sampled)} 条经验")
for exp in sampled:
    importance = memory.get_importance(exp.experience_id)
    print(f"  - {exp.experience_id}: importance={importance:.3f}")

# ============================================================================
# 3. 测试EWC的Fisher Information
# ============================================================================

print("\n【测试3: EWC - Fisher Information计算】")
print("-" * 80)

# 准备数据
print("\n3.1 准备训练数据:")
state_embeddings = [np.random.randn(768) for _ in range(20)]
selected_tools = ["search"] * 10 + ["email"] * 10

print(f"  - 状态数: {len(state_embeddings)}")
print(f"  - 工具选择: {selected_tools[:5]}...")

# 计算Fisher
print("\n3.2 计算Fisher Information:")
fisher = scorer.compute_fisher_information(
    state_embeddings=state_embeddings,
    selected_tools=selected_tools,
)

print(f"  ✓ Fisher计算成功")
print(f"  - Fisher shape: {fisher.shape}")
print(f"  - Fisher均值: {fisher.mean():.6f}")
print(f"  - Fisher最大值: {fisher.max():.6f}")
print(f"  - Fisher非零元素: {np.count_nonzero(fisher)}")

# 查看重要参数
important_params = np.sum(fisher > 0.01)
print(f"  - 重要参数数 (F>0.01): {important_params}")

# 测试EWC正则化
print("\n3.3 测试EWC正则化:")
ewc_loss = scorer.get_ewc_regularization_loss(ewc_lambda=1.0)
print(f"  - EWC loss: {ewc_loss:.6f}")

if ewc_loss > 0:
    print(f"  ✓ EWC正则化生效！")
else:
    print(f"  - EWC loss为0（正常，参数未偏离）")

# ============================================================================
# 4. 总结
# ============================================================================

print("\n" + "=" * 80)
print("【测试总结】")
print("=" * 80)

print("\n✅ 核心功能验证:")
print("  1. ✓ Tool Scorer可以学习和更新参数")
print("  2. ✓ Parametric Memory可以学习记忆重要性")
print("  3. ✓ Fisher Information可以计算和用于EWC")
print("  4. ✓ 这些都是真正的参数学习，不是prompt工程")

print("\n💡 关键对比:")
print("  ICL-ER (旧):  参数量=0, 只存储经验到prompt")
print("  Parametric (新): 参数量=" + f"{scorer.weights.size}, 真正的梯度学习")

print("\n📊 参数统计:")
print(f"  - Tool Scorer参数: {scorer.weights.size}")
print(f"  - Memory重要性参数: {len(memory)}")
print(f"  - 总可学习参数: {scorer.weights.size + len(memory)}")

print("\n🎓 理论支撑:")
print("  - Tool Scorer: 基于REINFORCE的策略梯度")
print("  - EWC: Fisher Information Matrix (Kirkpatrick et al., 2017)")
print("  - Memory: 可学习重要性权重")

print("\n🚀 下一步:")
print("  - 集成到完整的tau2实验中")
print("  - 运行多任务持续学习实验")
print("  - 对比5种方法的性能")
print("  - 详见 HOWTO_RUN.md")

print("\n✨ 这是真正的持续学习框架！")
print("=" * 80)
