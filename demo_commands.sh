#!/bin/bash
# Tau2-CL 演示脚本
# 这个脚本展示了如何一步步运行持续学习实验

set -e  # 遇到错误就停止

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                 Tau2-CL 实验演示脚本                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ========================================
# 第一部分：数据验证
# ========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "第1步：验证现有数据"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 验证airline domain
echo "→ 验证 airline domain 数据..."
tau2 cl-validate-data data/tau2/domains/airline/tasks.json
echo ""

# 查看数据统计
echo "→ 查看数据统计..."
tau2 cl-data-requirements --domains airline retail telecom
echo ""

read -p "按 Enter 继续到下一步..."

# ========================================
# 第二部分：生成数据划分
# ========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "第2步：生成CL数据划分"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "→ 生成sequential策略的数据划分..."
tau2 cl-generate-splits \
    --domains airline retail \
    --strategy sequential \
    --train-ratio 0.7 \
    --num-phases 3 \
    --output-dir data/tau2/cl_splits

echo ""
echo "✓ 生成的文件："
ls -lh data/tau2/cl_splits/*.json
echo ""

read -p "按 Enter 继续到下一步..."

# ========================================
# 第三部分：快速测试实验
# ========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "第3步：运行快速测试实验（10个任务）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "→ 使用ICL-ER方法在airline domain上运行10个任务..."
tau2 cl-run \
    --name demo_quick_test \
    --domains airline \
    --curriculum sequential \
    --agent-type icl_er \
    --agent-llm gpt-4o-mini \
    --user-llm gpt-4o-mini \
    --num-tasks 10 \
    --eval-frequency 3 \
    --output-dir ./experiments/demo_quick_test \
    --seed 42

echo ""
echo "✓ 实验完成！结果保存在: ./experiments/demo_quick_test/"
echo ""

read -p "按 Enter 查看结果..."

# ========================================
# 第四部分：分析结果
# ========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "第4步：分析实验结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

tau2 cl-analyze ./experiments/demo_quick_test/results.json

echo ""
echo "✓ 生成的文件："
ls -lh experiments/demo_quick_test/
echo ""
echo "  可视化图表："
ls -lh experiments/demo_quick_test/metrics/ 2>/dev/null || echo "  （图表将在实验完成后生成）"
echo ""

read -p "按 Enter 运行对比实验..."

# ========================================
# 第五部分：对比实验
# ========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "第5步：运行对比实验"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "→ 运行Baseline实验（无CL）..."
tau2 cl-run \
    --name demo_baseline \
    --domains airline \
    --curriculum sequential \
    --agent-type baseline \
    --agent-llm gpt-4o-mini \
    --user-llm gpt-4o-mini \
    --num-tasks 10 \
    --eval-frequency 3 \
    --output-dir ./experiments/demo_baseline \
    --seed 42

echo ""
echo "✓ Baseline实验完成！"
echo ""

read -p "按 Enter 查看对比结果..."

# ========================================
# 第六部分：对比分析
# ========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "第6步：对比两个实验"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "→ ICL-ER 结果:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tau2 cl-analyze ./experiments/demo_quick_test/results.json

echo ""
echo "→ Baseline 结果:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tau2 cl-analyze ./experiments/demo_baseline/results.json

echo ""
echo "→ 生成对比图表..."
python scripts/compare_experiments.py \
    experiments/demo_quick_test \
    experiments/demo_baseline

echo ""
echo "✓ 对比结果保存在: experiments/comparison/"
echo ""

# ========================================
# 完成
# ========================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     演示完成！                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "你可以查看以下结果："
echo ""
echo "  1. 实验配置:"
echo "     - experiments/demo_quick_test/config.json"
echo "     - experiments/demo_baseline/config.json"
echo ""
echo "  2. 实验结果:"
echo "     - experiments/demo_quick_test/results.json"
echo "     - experiments/demo_baseline/results.json"
echo ""
echo "  3. 可视化图表:"
echo "     - experiments/demo_quick_test/metrics/"
echo "     - experiments/demo_baseline/metrics/"
echo "     - experiments/comparison/"
echo ""
echo "  4. 对比数据:"
echo "     - experiments/comparison/metrics_comparison.csv"
echo "     - experiments/comparison/domain_accuracy_comparison.csv"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "下一步："
echo "  - 阅读 快速开始指南.md 了解更多"
echo "  - 阅读 docs/continual_learning_workflow.md 了解完整流程"
echo "  - 运行 python scripts/quick_experiment.py --help 查看更多选项"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
