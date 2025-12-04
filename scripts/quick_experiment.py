#!/usr/bin/env python3
"""
快速实验脚本 - 用于快速运行和测试CL实验

Usage:
    python scripts/quick_experiment.py --help
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

def run_command(cmd: str, description: str = ""):
    """运行shell命令"""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}\n")

    print(f"Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        return False
    return True


def quick_test(args):
    """运行快速测试实验"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         Tau2-CL Quick Test Experiment                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # 1. 验证数据
    if not run_command(
        f"tau2 cl-validate-data data/tau2/domains/{args.domain}/tasks.json",
        "Step 1: Validating data"
    ):
        return

    # 2. 运行实验
    cmd = f"""tau2 cl-run \
        --name quick_test_{args.domain} \
        --domains {args.domain} \
        --curriculum {args.curriculum} \
        --agent-type {args.agent} \
        --agent-llm {args.model} \
        --user-llm {args.model} \
        --num-tasks {args.num_tasks} \
        --eval-frequency {args.eval_freq} \
        --output-dir ./experiments/quick_test_{args.domain} \
        --seed {args.seed}"""

    if not run_command(cmd, "Step 2: Running experiment"):
        return

    # 3. 分析结果
    if not run_command(
        f"tau2 cl-analyze ./experiments/quick_test_{args.domain}/results.json",
        "Step 3: Analyzing results"
    ):
        return

    print("\n✓ Quick test completed successfully!")
    print(f"  Results saved to: ./experiments/quick_test_{args.domain}/")


def full_experiment(args):
    """运行完整的对比实验"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         Tau2-CL Full Comparison Experiment               ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    domains = args.domains.split(',')

    # 1. 验证所有domain数据
    run_command("tau2 cl-validate-data data/tau2/domains/", "Step 1: Validating all data")

    # 2. 查看数据需求
    run_command(
        f"tau2 cl-data-requirements --domains {' '.join(domains)}",
        "Step 2: Checking data requirements"
    )

    # 3. 生成数据划分
    run_command(
        f"""tau2 cl-generate-splits \
            --domains {' '.join(domains)} \
            --strategy {args.curriculum} \
            --train-ratio {args.train_ratio} \
            --num-phases {args.num_phases} \
            --output-dir data/tau2/cl_splits""",
        "Step 3: Generating CL splits"
    )

    # 4. 运行多个agent进行对比
    agents = ["icl_er", "prompt_strategy", "baseline"]

    for agent in agents:
        print(f"\n{'='*60}")
        print(f"  Running {agent.upper()} agent")
        print(f"{'='*60}\n")

        cmd = f"""tau2 cl-run \
            --name {agent}_{args.curriculum} \
            --domains {' '.join(domains)} \
            --curriculum {args.curriculum} \
            --agent-type {agent} \
            --agent-llm {args.model} \
            --user-llm {args.model} \
            --num-tasks {args.num_tasks} \
            --eval-frequency {args.eval_freq} \
            --max-examples {args.max_examples} \
            --buffer-size {args.buffer_size} \
            --output-dir ./experiments/{agent}_{args.curriculum} \
            --seed {args.seed}"""

        run_command(cmd)

    # 5. 分析所有结果
    print(f"\n{'='*60}")
    print(f"  Analyzing all results")
    print(f"{'='*60}\n")

    for agent in agents:
        print(f"\n--- {agent.upper()} Results ---")
        run_command(f"tau2 cl-analyze ./experiments/{agent}_{args.curriculum}/results.json")

    print("\n✓ Full experiment completed!")
    print("  Check ./experiments/ for all results")


def generate_data(args):
    """辅助生成任务数据"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         Task Data Generation Helper                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    domain = args.domain
    num_tasks = args.num_tasks

    print(f"Domain: {domain}")
    print(f"Number of tasks to generate: {num_tasks}")
    print("\nGenerating task templates...")

    # 加载现有任务作为模板
    tasks_file = Path(f"data/tau2/domains/{domain}/tasks.json")
    if not tasks_file.exists():
        print(f"[ERROR] Tasks file not found: {tasks_file}")
        return

    with open(tasks_file, 'r', encoding='utf-8') as f:
        existing_tasks = json.load(f)

    if not existing_tasks:
        print("[ERROR] No existing tasks to use as template")
        return

    print(f"Loaded {len(existing_tasks)} existing tasks as templates")

    # 创建任务模板
    template = existing_tasks[0].copy()
    template['id'] = "TASK_ID_HERE"
    template['user_scenario']['instructions']['reason_for_call'] = "FILL IN REASON FOR CALL"
    template['user_scenario']['instructions']['task_instructions'] = "FILL IN TASK INSTRUCTIONS"

    # 保存模板
    output_file = Path(f"data/tau2/domains/{domain}/new_tasks_template.json")

    new_tasks = [template.copy() for _ in range(num_tasks)]
    for i, task in enumerate(new_tasks):
        task['id'] = f"new_task_{i:03d}"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_tasks, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Generated {num_tasks} task templates")
    print(f"  Saved to: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Edit {output_file} to fill in task details")
    print(f"  2. Validate: tau2 cl-validate-data {output_file}")
    print(f"  3. Merge with existing tasks if valid")


def main():
    parser = argparse.ArgumentParser(
        description="Quick experiment runner for Tau2-CL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test on airline domain
  python scripts/quick_experiment.py quick-test --domain airline --num-tasks 20

  # Full comparison experiment
  python scripts/quick_experiment.py full --domains airline,retail --num-tasks 100

  # Generate task templates
  python scripts/quick_experiment.py generate --domain airline --num-tasks 50
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Quick test command
    quick_parser = subparsers.add_parser('quick-test', help='Run quick test experiment')
    quick_parser.add_argument('--domain', default='airline', help='Domain to test')
    quick_parser.add_argument('--curriculum', default='sequential', choices=['sequential', 'interleaved', 'difficulty'])
    quick_parser.add_argument('--agent', default='icl_er', choices=['icl_er', 'prompt_strategy', 'baseline'])
    quick_parser.add_argument('--model', default='gpt-4o-mini', help='LLM model to use')
    quick_parser.add_argument('--num-tasks', type=int, default=20, help='Number of tasks')
    quick_parser.add_argument('--eval-freq', type=int, default=5, help='Evaluation frequency')
    quick_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Full experiment command
    full_parser = subparsers.add_parser('full', help='Run full comparison experiment')
    full_parser.add_argument('--domains', default='airline,retail', help='Comma-separated domains')
    full_parser.add_argument('--curriculum', default='sequential', choices=['sequential', 'interleaved', 'difficulty'])
    full_parser.add_argument('--model', default='gpt-4o-mini', help='LLM model to use')
    full_parser.add_argument('--num-tasks', type=int, default=100, help='Number of tasks per domain')
    full_parser.add_argument('--eval-freq', type=int, default=10, help='Evaluation frequency')
    full_parser.add_argument('--max-examples', type=int, default=5, help='Max examples in prompt')
    full_parser.add_argument('--buffer-size', type=int, default=1000, help='Memory buffer size')
    full_parser.add_argument('--train-ratio', type=float, default=0.7, help='Training ratio')
    full_parser.add_argument('--num-phases', type=int, default=3, help='Number of phases')
    full_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Generate data command
    gen_parser = subparsers.add_parser('generate', help='Generate task data templates')
    gen_parser.add_argument('--domain', required=True, help='Domain for tasks')
    gen_parser.add_argument('--num-tasks', type=int, default=50, help='Number of task templates to generate')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'quick-test':
        quick_test(args)
    elif args.command == 'full':
        full_experiment(args)
    elif args.command == 'generate':
        generate_data(args)


if __name__ == '__main__':
    main()
