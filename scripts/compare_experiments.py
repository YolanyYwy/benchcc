#!/usr/bin/env python3
"""
实验对比脚本 - 用于对比多个CL实验的结果

Usage:
    python scripts/compare_experiments.py exp1_dir exp2_dir exp3_dir
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def load_experiment(exp_dir: Path) -> Dict[str, Any]:
    """加载实验结果"""
    results_file = exp_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_comparison_table(experiments: List[Dict[str, Any]]):
    """打印对比表格"""
    print("\n" + "="*80)
    print("Experiment Comparison Table")
    print("="*80)

    # 表头
    headers = ["Metric"] + [exp['config']['name'] for exp in experiments]
    col_width = 20

    print(f"{'Metric':<{col_width}}", end="")
    for name in headers[1:]:
        print(f"{name:<{col_width}}", end="")
    print()
    print("-"*80)

    # 关键指标
    metrics = [
        ("Average Accuracy", lambda e: e['metrics']['average_accuracy']),
        ("Forgetting Rate", lambda e: e['metrics']['forgetting_rate']),
        ("Forward Transfer", lambda e: e['metrics']['forward_transfer']),
        ("Backward Transfer", lambda e: e['metrics']['backward_transfer']),
    ]

    for metric_name, metric_fn in metrics:
        print(f"{metric_name:<{col_width}}", end="")
        for exp in experiments:
            value = metric_fn(exp)
            print(f"{value:<{col_width}.3f}", end="")
        print()

    # Per-domain metrics
    print("\nPer-Domain Accuracy:")
    print("-"*80)

    all_domains = set()
    for exp in experiments:
        all_domains.update(exp['metrics']['domain_accuracies'].keys())

    for domain in sorted(all_domains):
        print(f"  {domain:<{col_width-2}}", end="")
        for exp in experiments:
            value = exp['metrics']['domain_accuracies'].get(domain, 0.0)
            print(f"{value:<{col_width}.3f}", end="")
        print()

    print("="*80 + "\n")


def plot_comparison(experiments: List[Dict[str, Any]], output_dir: Path):
    """绘制对比图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置样式
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(experiments))

    # 1. 关键指标对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Continual Learning Metrics Comparison', fontsize=16)

    names = [exp['config']['name'] for exp in experiments]

    # Average Accuracy
    accs = [exp['metrics']['average_accuracy'] for exp in experiments]
    axes[0, 0].bar(range(len(names)), accs, color=colors)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Average Accuracy')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Forgetting Rate
    forgetting = [exp['metrics']['forgetting_rate'] for exp in experiments]
    axes[0, 1].bar(range(len(names)), forgetting, color=colors)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Forgetting Rate')
    axes[0, 1].set_title('Forgetting Rate (lower is better)')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Forward Transfer
    forward = [exp['metrics']['forward_transfer'] for exp in experiments]
    axes[1, 0].bar(range(len(names)), forward, color=colors)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Forward Transfer')
    axes[1, 0].set_title('Forward Transfer (higher is better)')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Backward Transfer
    backward = [exp['metrics']['backward_transfer'] for exp in experiments]
    axes[1, 1].bar(range(len(names)), backward, color=colors)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Backward Transfer')
    axes[1, 1].set_title('Backward Transfer (higher is better)')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics comparison to {output_dir / 'metrics_comparison.png'}")

    # 2. Per-domain accuracy comparison
    all_domains = set()
    for exp in experiments:
        all_domains.update(exp['metrics']['domain_accuracies'].keys())

    if len(all_domains) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(all_domains))
        width = 0.8 / len(experiments)

        for i, (exp, color) in enumerate(zip(experiments, colors)):
            domain_accs = [exp['metrics']['domain_accuracies'].get(d, 0.0) for d in sorted(all_domains)]
            ax.bar(x + i * width, domain_accs, width, label=exp['config']['name'], color=color)

        ax.set_xlabel('Domain')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Domain Accuracy Comparison')
        ax.set_xticks(x + width * (len(experiments) - 1) / 2)
        ax.set_xticklabels(sorted(all_domains))
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'domain_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved domain accuracy comparison to {output_dir / 'domain_accuracy_comparison.png'}")

    # 3. Learning curves (if available)
    fig, ax = plt.subplots(figsize=(12, 6))

    for exp, color in zip(experiments, colors):
        if 'checkpoints' in exp and exp['checkpoints']:
            checkpoints = exp['checkpoints']
            steps = [cp['step'] for cp in checkpoints]
            accuracies = [cp['metrics']['accuracy'] for cp in checkpoints]
            ax.plot(steps, accuracies, marker='o', label=exp['config']['name'], color=color, linewidth=2)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved learning curves to {output_dir / 'learning_curves.png'}")

    plt.close('all')


def export_to_csv(experiments: List[Dict[str, Any]], output_dir: Path):
    """导出为CSV格式"""
    # 主要指标
    metrics_data = []
    for exp in experiments:
        metrics_data.append({
            'experiment': exp['config']['name'],
            'average_accuracy': exp['metrics']['average_accuracy'],
            'forgetting_rate': exp['metrics']['forgetting_rate'],
            'forward_transfer': exp['metrics']['forward_transfer'],
            'backward_transfer': exp['metrics']['backward_transfer'],
        })

    df = pd.DataFrame(metrics_data)
    csv_path = output_dir / 'metrics_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved metrics to {csv_path}")

    # Per-domain accuracy
    domain_data = []
    for exp in experiments:
        for domain, acc in exp['metrics']['domain_accuracies'].items():
            domain_data.append({
                'experiment': exp['config']['name'],
                'domain': domain,
                'accuracy': acc,
            })

    df_domain = pd.DataFrame(domain_data)
    csv_domain_path = output_dir / 'domain_accuracy_comparison.csv'
    df_domain.to_csv(csv_domain_path, index=False)
    print(f"✓ Saved domain accuracies to {csv_domain_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_experiments.py <exp_dir1> <exp_dir2> [exp_dir3 ...]")
        print("\nExample:")
        print("  python scripts/compare_experiments.py \\")
        print("      experiments/icl_er_sequential \\")
        print("      experiments/prompt_strategy_sequential \\")
        print("      experiments/baseline")
        sys.exit(1)

    exp_dirs = [Path(d) for d in sys.argv[1:]]

    # 验证目录
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            print(f"[ERROR] Experiment directory not found: {exp_dir}")
            sys.exit(1)
        if not (exp_dir / "results.json").exists():
            print(f"[ERROR] Results file not found in: {exp_dir}")
            sys.exit(1)

    print(f"\nComparing {len(exp_dirs)} experiments...")

    # 加载实验结果
    experiments = []
    for exp_dir in exp_dirs:
        try:
            exp = load_experiment(exp_dir)
            experiments.append(exp)
            print(f"  ✓ Loaded: {exp['config']['name']}")
        except Exception as e:
            print(f"  ✗ Failed to load {exp_dir}: {e}")

    if not experiments:
        print("[ERROR] No valid experiments loaded")
        sys.exit(1)

    # 打印对比表格
    print_comparison_table(experiments)

    # 生成对比图表
    output_dir = Path("experiments/comparison")
    plot_comparison(experiments, output_dir)

    # 导出CSV
    export_to_csv(experiments, output_dir)

    print(f"\n✓ Comparison complete!")
    print(f"  Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
