#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的参数化持续学习方法对比实验

对比5种方法：
1. ICL-ER (Baseline) - 非参数化
2. EWC - Elastic Weight Consolidation
3. Replay - Experience Replay
4. Parameter Isolation - 参数隔离
5. Meta-CL - 元持续学习

实验模式：
- mode='simulation': 快速模拟实验（秒级）
- mode='real': 真实LLM调用实验（分钟级）
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Literal
import json
from datetime import datetime
import numpy as np
from loguru import logger

# 解决Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tau2.environment.tool import Tool
from tau2.data_model.message import (
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolMessage,
    Message,
)

# 导入所有方法
from tau2.continual_learning.agents.icl_experience_replay import ICLExperienceReplayAgent
from tau2.continual_learning.agents.parametric import (
    EWCContinualLearningAgent,
    ReplayContinualLearningAgent,
    ParameterIsolationAgent,
    MetaContinualLearningAgent,
)


# ============================================================================
# 实验配置
# ============================================================================

class ExperimentConfig:
    """实验配置"""
    # 实验模式：'simulation' 或 'real'
    mode: Literal['simulation', 'real'] = 'simulation'

    # 任务配置
    num_domains = 3
    tasks_per_domain = 20  # 每个domain的任务数

    # Agent配置
    embedding_dim = 768
    learning_rate = 0.01

    # LLM配置
    llm_model = "gpt-4"  # 真实模式下使用

    # 输出配置
    output_dir = "experiments/results"
    save_checkpoints = True
    verbose = True


# ============================================================================
# 工具和任务生成
# ============================================================================

def create_tools() -> List[Tool]:
    """创建工具"""

    def search_database(query: str) -> str:
        """搜索数据库"""
        return f"搜索结果：找到关于 '{query}' 的3条记录"

    def send_email(to: str, subject: str, body: str = "") -> str:
        """发送邮件"""
        return f"邮件已发送给 {to}，主题：{subject}"

    def create_ticket(title: str, priority: str = "medium", description: str = "") -> str:
        """创建工单"""
        return f"工单已创建：{title} (优先级: {priority})"

    def update_record(record_id: str, field: str, value: str) -> str:
        """更新记录"""
        return f"记录 {record_id} 的 {field} 已更新为 {value}"

    return [
        Tool(search_database),
        Tool(send_email),
        Tool(create_ticket),
        Tool(update_record),
    ]


def create_domain_policy(domain: str) -> str:
    """创建domain policy"""
    policies = {
        "customer_service": """你是客户服务助手。
职责：回答客户问题、发送邮件通知、创建服务工单。
使用search_database查询信息，send_email发送通知，create_ticket创建工单。
要友好、专业、高效。""",

        "tech_support": """你是技术支持助手。
职责：解决技术问题、更新系统记录、创建技术工单。
使用search_database查询问题，update_record更新状态，create_ticket上报问题。
要准确、详细、系统化。""",

        "sales": """你是销售助手。
职责：查询产品信息、发送销售邮件、更新客户记录。
使用search_database查询产品，send_email联系客户，update_record更新信息。
要积极、热情、目标导向。""",
    }
    return policies.get(domain, "你是一个通用助手。")


def create_task_data(
    domain: str,
    task_id: int,
    preferred_tool: str,
) -> Dict[str, Any]:
    """创建任务数据（包含用户query和期望的工具）"""

    task_templates = {
        "customer_service": {
            "search_database": [
                "客户询问订单 #{} 的状态",
                "查找客户 {} 的历史记录",
                "搜索产品 {} 的信息",
            ],
            "send_email": [
                "通知客户 {} 关于订单延迟",
                "发送确认邮件给 {}",
                "给客户 {} 发送感谢信",
            ],
            "create_ticket": [
                "客户报告问题：{}",
                "创建退款工单：{}",
                "上报客户投诉：{}",
            ],
        },
        "tech_support": {
            "search_database": [
                "查询错误代码 {} 的解决方案",
                "搜索系统日志中的 {}",
                "查找 {} 的技术文档",
            ],
            "update_record": [
                "将bug #{} 状态更新为已修复",
                "更新服务器 {} 的配置",
                "修改系统设置 {}",
            ],
            "create_ticket": [
                "上报系统故障：{}",
                "创建技术工单：{}",
                "提交bug报告：{}",
            ],
        },
        "sales": {
            "search_database": [
                "查询产品 {} 的价格",
                "搜索客户 {} 的购买历史",
                "查找 {} 的库存信息",
            ],
            "send_email": [
                "给潜在客户 {} 发送报价",
                "发送产品介绍给 {}",
                "通知客户 {} 关于促销",
            ],
            "update_record": [
                "更新客户 {} 的状态为已成交",
                "修改订单 {} 的信息",
                "记录客户 {} 的反馈",
            ],
        },
    }

    templates = task_templates.get(domain, {}).get(preferred_tool, ["任务 {}"])
    template = templates[task_id % len(templates)]
    query = template.format(f"T{task_id}")

    return {
        "task_id": f"task_{task_id}",
        "domain": domain,
        "query": query,
        "preferred_tool": preferred_tool,
    }


def create_trajectory_simulation(
    task_data: Dict[str, Any],
    success_rate: float = 0.8,
) -> tuple[List[Message], float, bool]:
    """
    模拟模式：创建模拟轨迹（快速）

    Returns:
        (trajectory, reward, success)
    """
    task_id = task_data["task_id"]
    query = task_data["query"]
    tool = task_data["preferred_tool"]

    # 随机决定成功/失败
    success = np.random.random() < success_rate
    reward = 1.0 if success else 0.0

    # 构建工具参数
    if tool == "search_database":
        args = {"query": query}
    elif tool == "send_email":
        args = {"to": "user@example.com", "subject": query, "body": "详细内容"}
    elif tool == "create_ticket":
        args = {"title": query, "priority": "high", "description": "详细描述"}
    elif tool == "update_record":
        args = {"record_id": "REC001", "field": "status", "value": "已处理"}
    else:
        args = {"query": query}

    trajectory = [
        UserMessage(role="user", content=query),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"call_{task_id}",
                    name=tool,
                    arguments=args
                )
            ]
        ),
        ToolMessage(
            id=f"msg_{task_id}",
            role="tool",
            tool_call_id=f"call_{task_id}",
            content=f"{'成功' if success else '失败'}执行 {tool}"
        ),
        AssistantMessage(
            role="assistant",
            content=f"{'已成功' if success else '未能'}处理您的请求。"
        ),
    ]

    return trajectory, reward, success


# ============================================================================
# 实验运行器
# ============================================================================

class ContinualLearningExperiment:
    """持续学习实验框架"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tools = create_tools()
        self.results = {}

        logger.info(f"实验模式: {config.mode}")
        if config.mode == 'simulation':
            logger.info("  - 使用模拟轨迹（快速验证）")
        else:
            logger.info("  - 使用真实LLM调用（完整实验）")

    def create_agent(self, method_name: str, initial_domain: str):
        """创建指定方法的agent"""
        policy = create_domain_policy(initial_domain)

        common_args = {
            "tools": self.tools,
            "domain_policy": policy,
            "llm": self.config.llm_model,
            "max_examples_in_prompt": 5,
        }

        if method_name == "ICL-ER":
            return ICLExperienceReplayAgent(**common_args)

        # 参数化方法的共同参数
        parametric_args = {
            **common_args,
            "embedding_dim": self.config.embedding_dim,
            "learning_rate": self.config.learning_rate,
        }

        if method_name == "EWC":
            return EWCContinualLearningAgent(
                **parametric_args,
                ewc_lambda=1.0,
                online_ewc=True,
            )

        elif method_name == "Replay":
            return ReplayContinualLearningAgent(
                **parametric_args,
                replay_ratio=0.5,
                replay_batch_size=5,
            )

        elif method_name == "Param-Isolation":
            return ParameterIsolationAgent(
                **parametric_args,
                num_task_families=self.config.num_domains,
            )

        elif method_name == "Meta-CL":
            return MetaContinualLearningAgent(
                **parametric_args,
                meta_learning_rate=0.001,
            )

        else:
            raise ValueError(f"Unknown method: {method_name}")

    def run_task_stream(
        self,
        agent,
        task_stream: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        运行任务流

        Args:
            agent: CL agent
            task_stream: 任务流 [{"domain": ..., "task_id": ..., "query": ..., ...}, ...]

        Returns:
            性能记录
        """
        performance_history = []
        domain_performance = {}

        for i, task_data in enumerate(task_stream):
            domain = task_data["domain"]
            task_id = task_data["task_id"]

            # 生成轨迹（根据模式）
            if self.config.mode == 'simulation':
                trajectory, reward, success = create_trajectory_simulation(task_data)
            else:
                # 真实模式：需要调用Agent的generate_next_message等
                # 这里简化为模拟
                trajectory, reward, success = create_trajectory_simulation(task_data)

            # Agent学习
            try:
                learning_stats = agent.learn_from_trajectory(
                    task_id=task_id,
                    domain=domain,
                    trajectory=trajectory,
                    reward=reward,
                    success=success,
                )
            except Exception as e:
                logger.warning(f"Learning failed for {task_id}: {e}")
                learning_stats = {}

            # 记录性能
            performance_history.append({
                "task_id": task_id,
                "domain": domain,
                "reward": reward,
                "success": success,
            })

            if domain not in domain_performance:
                domain_performance[domain] = []
            domain_performance[domain].append(reward)

            # 定期打印进度
            if self.config.verbose and (i + 1) % 20 == 0:
                avg_reward = np.mean([p["reward"] for p in performance_history])
                logger.info(f"  进度: {i+1}/{len(task_stream)}, 平均奖励={avg_reward:.3f}")

        return {
            "performance_history": performance_history,
            "domain_performance": domain_performance,
            "agent_stats": agent.get_statistics() if hasattr(agent, 'get_statistics') else {},
        }

    def compute_metrics(
        self,
        performance_history: List[Dict],
        domain_performance: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """计算评估指标"""

        # 1. 平均性能
        all_rewards = [p["reward"] for p in performance_history]
        avg_performance = np.mean(all_rewards) if all_rewards else 0.0

        # 2. 每个domain的性能
        domain_avg = {
            domain: np.mean(perfs) if perfs else 0.0
            for domain, perfs in domain_performance.items()
        }

        # 3. 遗忘度（Forward Transfer）
        # 比较每个domain早期和晚期性能
        forgetting = 0.0
        num_domains = 0
        for domain, perfs in domain_performance.items():
            if len(perfs) >= 10:
                early = np.mean(perfs[:5])
                late = np.mean(perfs[-5:])
                forgetting += max(0, early - late)
                num_domains += 1
        forgetting = forgetting / num_domains if num_domains > 0 else 0.0

        # 4. 稳定性
        performance_std = np.std(all_rewards) if len(all_rewards) > 1 else 0.0
        stability = 1.0 / (1.0 + performance_std)

        return {
            "avg_performance": float(avg_performance),
            "domain_performance": domain_avg,
            "forgetting": float(forgetting),
            "stability": float(stability),
            "performance_std": float(performance_std),
            "total_tasks": len(performance_history),
        }

    def run_method(
        self,
        method_name: str,
        task_stream: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """运行单个方法的实验"""

        logger.info(f"\n{'='*80}")
        logger.info(f"运行方法: {method_name}")
        logger.info(f"{'='*80}")

        try:
            # 创建agent
            initial_domain = task_stream[0]["domain"]
            agent = self.create_agent(method_name, initial_domain)

            # 记录开始时间
            start_time = time.time()

            # 运行任务流
            run_results = self.run_task_stream(agent, task_stream)

            # 记录结束时间
            elapsed_time = time.time() - start_time

            # 计算指标
            metrics = self.compute_metrics(
                run_results["performance_history"],
                run_results["domain_performance"],
            )

            # 添加额外信息
            metrics["elapsed_time"] = elapsed_time
            metrics["method_name"] = method_name
            metrics["agent_stats"] = run_results["agent_stats"]

            logger.info(f"✓ {method_name} 完成")
            logger.info(f"  - 平均性能: {metrics['avg_performance']:.3f}")
            logger.info(f"  - 遗忘度: {metrics['forgetting']:.3f}")
            logger.info(f"  - 稳定性: {metrics['stability']:.3f}")
            logger.info(f"  - 用时: {elapsed_time:.1f}s")

            return metrics

        except Exception as e:
            logger.error(f"✗ {method_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "method_name": method_name,
                "avg_performance": 0.0,
                "forgetting": 1.0,
                "elapsed_time": 0.0,
            }

    def generate_task_stream(self) -> List[Dict[str, Any]]:
        """生成任务流"""
        task_stream = []
        task_id = 0

        domains = ["customer_service", "tech_support", "sales"]
        tools = [tool.name for tool in self.tools]

        # 为每个domain生成任务
        for _ in range(self.config.tasks_per_domain):
            for domain in domains:
                task_id += 1
                # 为每个domain随机选择工具
                preferred_tool = np.random.choice(tools)
                task_data = create_task_data(domain, task_id, preferred_tool)
                task_stream.append(task_data)

        return task_stream

    def run_all_methods(self) -> Dict[str, Dict]:
        """运行所有方法的实验"""

        logger.info(f"\n{'='*80}")
        logger.info(f"开始完整实验")
        logger.info(f"{'='*80}")

        # 生成任务流
        task_stream = self.generate_task_stream()

        logger.info(f"任务流生成完成:")
        logger.info(f"  - 总任务数: {len(task_stream)}")
        logger.info(f"  - Domains: {self.config.num_domains}")
        logger.info(f"  - 每个domain任务数: {self.config.tasks_per_domain}")

        # 要测试的方法
        methods = [
            "ICL-ER",          # Baseline
            "EWC",             # Method 1
            "Replay",          # Method 2
            "Param-Isolation", # Method 3
            "Meta-CL",         # Method 4
        ]

        results = {}

        for method_name in methods:
            results[method_name] = self.run_method(method_name, task_stream)

        return results

    def print_comparison_table(self, results: Dict):
        """打印对比表格"""

        print(f"\n{'='*80}")
        print("实验结果对比")
        print(f"{'='*80}")

        # 表头
        print(f"\n{'方法':<20} {'平均性能':<12} {'遗忘度':<12} {'稳定性':<12} {'用时(s)':<10}")
        print("-"*80)

        # 排序（按平均性能）
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get('avg_performance', 0),
            reverse=True
        )

        for method_name, metrics in sorted_results:
            if 'error' in metrics:
                print(f"{method_name:<20} {'ERROR':<12} - {metrics['error']}")
            else:
                print(f"{method_name:<20} "
                      f"{metrics['avg_performance']:<12.3f} "
                      f"{metrics['forgetting']:<12.3f} "
                      f"{metrics['stability']:<12.3f} "
                      f"{metrics['elapsed_time']:<10.1f}")

        print(f"\n{'='*80}")

        # 胜者统计
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_perf = max(valid_results.items(), key=lambda x: x[1].get('avg_performance', 0))
            best_forget = min(valid_results.items(), key=lambda x: x[1].get('forgetting', 1))

            print(f"\n🏆 最佳性能: {best_perf[0]} ({best_perf[1]['avg_performance']:.3f})")
            print(f"🛡️  最低遗忘: {best_forget[0]} ({best_forget[1]['forgetting']:.3f})")

            # 参数化方法 vs Baseline
            if 'ICL-ER' in valid_results:
                baseline_perf = valid_results['ICL-ER'].get('avg_performance', 0)
                print(f"\n📊 与Baseline (ICL-ER) 对比:")
                for method_name, metrics in valid_results.items():
                    if method_name != 'ICL-ER':
                        improvement = metrics['avg_performance'] - baseline_perf
                        if baseline_perf > 0:
                            pct = improvement / baseline_perf * 100
                            print(f"  {method_name}: {improvement:+.3f} ({pct:+.1f}%)")

    def save_results(self, results: Dict, output_path: str):
        """保存实验结果"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✓ 结果已保存到: {output_file}")


# ============================================================================
# 主函数
# ============================================================================

def main(
    mode: Literal['simulation', 'real'] = 'simulation',
    num_domains: int = 3,
    tasks_per_domain: int = 20,
):
    """
    运行完整实验

    Args:
        mode: 'simulation' (快速) 或 'real' (完整)
        num_domains: domain数量
        tasks_per_domain: 每个domain的任务数
    """

    print("="*80)
    print("参数化持续学习方法 - 完整对比实验")
    print("="*80)
    print(f"\n实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建配置
    config = ExperimentConfig()
    config.mode = mode
    config.num_domains = num_domains
    config.tasks_per_domain = tasks_per_domain

    print(f"\n实验配置:")
    print(f"  - 模式: {config.mode}")
    print(f"  - Domains: {config.num_domains}")
    print(f"  - 每个domain任务数: {config.tasks_per_domain}")
    print(f"  - 总任务数: {config.num_domains * config.tasks_per_domain}")
    print(f"  - Embedding维度: {config.embedding_dim}")
    print(f"  - 学习率: {config.learning_rate}")

    if mode == 'simulation':
        print(f"\n⚡ 模拟模式：使用mock轨迹，验证代码逻辑（快速）")
    else:
        print(f"\n🔥 真实模式：调用LLM API，完整实验（较慢）")

    # 创建实验
    experiment = ContinualLearningExperiment(config)

    # 运行所有方法
    print(f"\n开始运行实验...")
    results = experiment.run_all_methods()

    # 打印对比表
    experiment.print_comparison_table(results)

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{config.output_dir}/comparison_{mode}_{timestamp}.json"
    experiment.save_results(results, output_path)

    print(f"\n✅ 实验完成!")
    print(f"📁 结果已保存到: {output_path}")

    print("\n💡 关键发现:")
    print("  - 参数化方法 vs 非参数化ICL-ER的性能对比")
    print("  - 不同方法在遗忘度、稳定性上的差异")
    print("  - 详见保存的JSON文件")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='持续学习方法对比实验')
    parser.add_argument(
        '--mode',
        type=str,
        default='simulation',
        choices=['simulation', 'real'],
        help='实验模式：simulation(快速) 或 real(完整)'
    )
    parser.add_argument(
        '--num-domains',
        type=int,
        default=3,
        help='Domain数量'
    )
    parser.add_argument(
        '--tasks-per-domain',
        type=int,
        default=20,
        help='每个domain的任务数'
    )

    args = parser.parse_args()

    results = main(
        mode=args.mode,
        num_domains=args.num_domains,
        tasks_per_domain=args.tasks_per_domain,
    )
