# 配置文件使用说明

## 📁 可用配置

| 配置文件 | 用途 | 预计时间 | 推荐场景 |
|---------|------|---------|---------|
| `quick_test.yaml` | 快速测试 | 5-10分钟 | 验证环境和代码 |
| `single_domain.yaml` | 单domain实验 | 20-30分钟 | 测试单个领域 |
| `multi_domain.yaml` | 多domain实验 | 1-2小时 | 标准CL实验（推荐） |
| `interleaved.yaml` | 交错学习 | 1-2小时 | 测试交错策略 |
| `baseline.yaml` | 基线对比 | 1-2小时 | 对比无CL性能 |
| `high_performance.yaml` | 高性能模型 | 2-4小时 | 追求最佳性能 |
| `prompt_strategy.yaml` | 提示策略 | 1-2小时 | 测试PSE方法 |

## 🚀 使用方法

### 1. 直接使用预设配置

```bash
# 快速测试
tau2 cl-run --config configs/cl_experiments/quick_test.yaml

# 多domain实验
tau2 cl-run --config configs/cl_experiments/multi_domain.yaml

# Baseline对比
tau2 cl-run --config configs/cl_experiments/baseline.yaml
```

### 2. 修改现有配置

复制一个配置文件并修改：

```bash
# 复制模板
cp configs/cl_experiments/multi_domain.yaml configs/cl_experiments/my_experiment.yaml

# 编辑配置
# 修改 name, domains, num_tasks_per_domain 等参数

# 运行
tau2 cl-run --config configs/cl_experiments/my_experiment.yaml
```

### 3. 创建自定义配置

创建新的YAML文件：

```yaml
name: "my_custom_experiment"
seed: 42
output_dir: "./experiments/my_custom"

curriculum_strategy: "SEQUENTIAL"
domains: ["airline", "retail"]
num_tasks_per_domain: 50

agent_type: "ICL_ER"
agent_llm: "gpt-4o-mini"
max_examples_in_prompt: 5

memory_buffer_size: 1000
sampling_strategy: "DIVERSITY"

user_llm: "gpt-4o-mini"
eval_frequency: 10
```

## 📋 配置参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `name` | 实验名称 | `"my_experiment"` |
| `domains` | 训练的domain列表 | `["airline", "retail"]` |
| `curriculum_strategy` | Curriculum策略 | `"SEQUENTIAL"` |
| `agent_type` | Agent类型 | `"ICL_ER"` |
| `agent_llm` | Agent使用的LLM | `"gpt-4o-mini"` |
| `user_llm` | User使用的LLM | `"gpt-4o-mini"` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `seed` | 随机种子 | `42` |
| `output_dir` | 输出目录 | `"./experiments/{name}"` |
| `num_tasks_per_domain` | 每个domain的任务数 | `null`（使用全部） |
| `max_examples_in_prompt` | prompt中最多几个示例 | `5` |
| `memory_buffer_size` | Memory buffer大小 | `1000` |
| `sampling_strategy` | 采样策略 | `"DIVERSITY"` |
| `eval_frequency` | 评估频率 | `10` |

## 🎯 常见实验组合

### 实验1：对比不同Agent方法

```bash
# 运行3个配置
tau2 cl-run --config configs/cl_experiments/multi_domain.yaml      # ICL-ER
tau2 cl-run --config configs/cl_experiments/prompt_strategy.yaml   # Prompt Strategy
tau2 cl-run --config configs/cl_experiments/baseline.yaml          # Baseline

# 对比结果
python scripts/compare_experiments.py \
    experiments/multi_domain_sequential \
    experiments/prompt_strategy_experiment \
    experiments/baseline_no_cl
```

### 实验2：对比不同Curriculum策略

```bash
# Sequential
tau2 cl-run --config configs/cl_experiments/multi_domain.yaml

# Interleaved
tau2 cl-run --config configs/cl_experiments/interleaved.yaml

# 对比
python scripts/compare_experiments.py \
    experiments/multi_domain_sequential \
    experiments/interleaved_experiment
```

### 实验3：对比不同模型

```bash
# GPT-4o-mini
tau2 cl-run --config configs/cl_experiments/multi_domain.yaml

# GPT-4o
tau2 cl-run --config configs/cl_experiments/high_performance.yaml

# 对比
python scripts/compare_experiments.py \
    experiments/multi_domain_sequential \
    experiments/high_performance
```

## 💡 配置优化建议

### 快速实验（开发/调试）
```yaml
num_tasks_per_domain: 10
agent_llm: "gpt-4o-mini"
user_llm: "gpt-4o-mini"
eval_frequency: 3
memory_buffer_size: 100
```

### 标准实验（论文结果）
```yaml
num_tasks_per_domain: 50
agent_llm: "gpt-4o-mini"
user_llm: "gpt-4o-mini"
eval_frequency: 10
memory_buffer_size: 1000
```

### 高质量实验（最终性能）
```yaml
num_tasks_per_domain: 100
agent_llm: "gpt-4o"
user_llm: "gpt-4o-mini"
eval_frequency: 10
memory_buffer_size: 2000
max_examples_in_prompt: 8
```

## 🔧 故障排除

### 问题1：配置文件找不到
```bash
# 确保在项目根目录
pwd  # 应该显示 tau2-bench/

# 使用绝对路径
tau2 cl-run --config /absolute/path/to/config.yaml
```

### 问题2：参数无效
```bash
# 检查YAML语法
python -c "import yaml; yaml.safe_load(open('configs/cl_experiments/my_config.yaml'))"
```

### 问题3：想覆盖配置参数
```bash
# 配置文件 + 命令行参数（命令行优先）
tau2 cl-run --config my_config.yaml --num-tasks 20 --seed 123
```

## 📚 更多资源

- **完整参数列表**: 查看 `src/tau2/continual_learning/__init__.py` 中的 `CLExperimentConfig`
- **示例配置**: 所有 `configs/cl_experiments/*.yaml` 文件
- **使用教程**: 查看 `快速开始指南.md`

---

**推荐开始实验：**

```bash
# 第1步：快速测试（5分钟）
tau2 cl-run --config configs/cl_experiments/quick_test.yaml

# 第2步：完整实验（1小时）
tau2 cl-run --config configs/cl_experiments/multi_domain.yaml
```
