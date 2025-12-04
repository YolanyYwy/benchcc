# 🚀 Tau2-CL 持续学习实验快速开始

这是一套完整的持续学习（Continual Learning）实验工具和教程，用于在 Tau2-Bench 项目中训练和评估 Agent 的 tool use 能力。

## 📁 项目结构

```
tau2-bench/
├── 快速开始指南.md          # 详细的新手教程
├── demo_commands.sh         # Linux/Mac 演示脚本
├── demo_commands.bat        # Windows 演示脚本
├── scripts/
│   ├── quick_experiment.py  # 快速实验脚本
│   └── compare_experiments.py  # 实验对比工具
├── docs/
│   └── continual_learning_workflow.md  # 完整工作流程文档
└── src/tau2/continual_learning/  # 核心实现代码
```

## ⚡ 快速开始（3种方式）

### 方式 1️⃣ ：最快上手（5分钟）

```bash
# 一键运行快速测试
python scripts/quick_experiment.py quick-test --domain airline --num-tasks 10
```

### 方式 2️⃣ ：使用演示脚本（10分钟）

**Windows 用户：**
```bash
demo_commands.bat
```

**Linux/Mac 用户：**
```bash
bash demo_commands.sh
```

### 方式 3️⃣ ：手动运行（完全控制）

```bash
# 1. 验证数据
tau2 cl-validate-data data/tau2/domains/airline/tasks.json

# 2. 运行实验
tau2 cl-run \
    --name my_experiment \
    --domains airline \
    --curriculum sequential \
    --agent-type icl_er \
    --num-tasks 20 \
    --output-dir ./experiments/my_experiment

# 3. 分析结果
tau2 cl-analyze experiments/my_experiment/results.json
```

## 📚 完整文档

- **[快速开始指南.md](./快速开始指南.md)** - 新手完全指南，包含详细步骤和解释
- **[continual_learning_workflow.md](./docs/continual_learning_workflow.md)** - 完整的工作流程文档

## 🔧 主要功能

### 1. 数据管理
```bash
# 验证数据
tau2 cl-validate-data data/tau2/domains/

# 查看数据统计
tau2 cl-data-requirements --domains airline retail telecom

# 生成训练/测试划分
tau2 cl-generate-splits --domains airline retail --strategy sequential
```

### 2. 运行实验
```bash
# 基础实验
tau2 cl-run --name my_exp --domains airline --agent-type icl_er

# 使用配置文件
tau2 cl-run --config configs/cl_experiments/my_config.yaml

# 完整对比实验
python scripts/quick_experiment.py full --domains airline,retail
```

### 3. 结果分析
```bash
# 分析单个实验
tau2 cl-analyze experiments/my_exp/results.json

# 对比多个实验
python scripts/compare_experiments.py exp1/ exp2/ exp3/
```

## 🎯 支持的方法

- **ICL-ER**: 带经验回放的上下文学习（In-Context Learning with Experience Replay）
- **Prompt Strategy**: 提示策略方法
- **Baseline**: 无持续学习的基线

## 📊 支持的 Curriculum 策略

- **Sequential**: 顺序学习（先学完A再学B）
- **Interleaved**: 交错学习（A和B交替学习）
- **Difficulty-based**: 基于难度的学习（从易到难）

## 🎓 学习路径

### 初学者（1小时）
1. ✅ 运行 `quick-test`（10分钟）
2. ✅ 理解输出结果（10分钟）
3. ✅ 阅读快速开始指南（40分钟）

### 进阶用户（2-3小时）
1. ✅ 在多个 domain 上运行实验
2. ✅ 对比不同的方法
3. ✅ 尝试不同的 curriculum 策略

### 高级用户（1天）
1. ✅ 生成新的训练数据
2. ✅ 实现自定义的 Agent 策略
3. ✅ 运行完整的对比实验

## 📈 实验输出

每个实验会生成：

```
experiments/my_experiment/
├── config.json              # 实验配置
├── results.json             # 完整结果
├── metrics/
│   ├── accuracy_curve.png   # 准确率曲线
│   ├── forgetting_matrix.png  # 遗忘矩阵
│   └── performance_matrix.png  # 性能矩阵
├── agent_state/
│   ├── final_state.json     # 最终 agent 状态
│   └── memory_buffer.json   # 最终 memory buffer
└── logs/
    └── experiment.log       # 详细日志
```

## 🔍 关键指标解释

- **Average Accuracy**: 平均准确率（越高越好）
- **Forgetting Rate**: 遗忘率（越低越好）
- **Forward Transfer**: 正向迁移（新知识帮助旧任务，越高越好）
- **Backward Transfer**: 负向迁移（学习新任务对旧任务的影响，越接近0越好）

## 💡 常见用例

### 快速验证想法
```bash
python scripts/quick_experiment.py quick-test --num-tasks 10
```

### 对比不同方法
```bash
python scripts/quick_experiment.py full --domains airline,retail
```

### 生成任务模板
```bash
python scripts/quick_experiment.py generate --domain airline --num-tasks 50
```

### 自定义实验
```bash
tau2 cl-run \
    --name custom_exp \
    --domains airline retail telecom \
    --curriculum interleaved \
    --agent-type icl_er \
    --max-examples 10 \
    --buffer-size 2000
```

## 🐛 常见问题

### Q: 命令找不到？
```bash
# 确保正确安装
pip install -e .
```

### Q: 实验太慢？
```bash
# 使用更快的模型和更少的任务
--agent-llm gpt-4o-mini --num-tasks 20
```

### Q: 数据不足？
```bash
# 生成新的任务模板
python scripts/quick_experiment.py generate --domain airline
```

更多问题请查看 [快速开始指南.md](./快速开始指南.md)

## 🤝 贡献

欢迎贡献新的：
- Agent 策略（在 `src/tau2/continual_learning/agents/`）
- Curriculum 策略（在 `src/tau2/continual_learning/curriculum/`）
- 评估指标（在 `src/tau2/continual_learning/metrics/`）

## 📞 获取帮助

- **查看所有 CL 命令**: `tau2 cl-info`
- **查看命令帮助**: `tau2 cl-run --help`
- **报告问题**: GitHub Issues
- **阅读文档**: `docs/` 目录

## 🎉 开始实验吧！

选择一种方式开始你的第一个实验：

```bash
# 最简单的方式
python scripts/quick_experiment.py quick-test --domain airline --num-tasks 10

# 或者使用演示脚本
demo_commands.bat  # Windows
bash demo_commands.sh  # Linux/Mac
```

祝实验顺利！🚀

---

**提示**: 建议先阅读 [快速开始指南.md](./快速开始指南.md) 了解详细的步骤和解释。
