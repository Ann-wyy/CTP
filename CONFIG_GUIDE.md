# YAML 配置文件使用指南

## 📖 概述

从现在开始，**推荐使用YAML配置文件**来配置训练参数，而不是冗长的命令行参数。这样可以：

- ✅ 配置更清晰易读
- ✅ 易于版本控制和分享
- ✅ 方便复用和修改
- ✅ 减少命令行输入错误

---

## 🚀 快速开始

### 1. 选择合适的配置模板

我们提供了多个预设配置文件：

| 配置文件 | 适用场景 |
|----------|----------|
| **config_basic.yaml** | 基础训练，默认参数 |
| **config_advanced.yaml** | 高级训练，包含早停、类别权重 |
| **config_small_gpu.yaml** | GPU显存≤8GB |
| **config_multiclass.yaml** | 多分类任务（3类或更多） |
| **config_template.yaml** | 完整模板，包含所有选项和注释 |

### 2. 复制并修改配置

```bash
# 复制模板
cp config_basic.yaml my_config.yaml

# 编辑配置文件
nano my_config.yaml  # 或使用您喜欢的编辑器
```

### 3. 使用配置文件训练

```bash
# 使用YAML配置
python train.py --config my_config.yaml
```

---

## 📝 配置文件结构

### 完整配置示例

```yaml
# ==================== 数据配置 ====================
csv_file: data_mixed_example.csv
val_split: 0.2
num_classes: 2

# ==================== 模型配置 ====================
resnet_type: resnet50
pretrained: true

# ==================== 训练配置 ====================
epochs: 50
batch_size: 4
lr: 0.001
weight_decay: 0.00001
class_weights: null

# ==================== 早停配置 ====================
early_stopping: false
patience: 10

# ==================== 其他配置 ====================
output_dir: ./output
num_workers: 4
seed: 42
```

### 参数说明

#### 数据参数

```yaml
# CSV数据文件路径（必需）
csv_file: /path/to/your/data.csv

# 验证集比例（0-1之间，推荐0.2）
val_split: 0.2

# 分类类别数
num_classes: 2
```

#### 模型参数

```yaml
# ResNet类型
# 选项: resnet18, resnet34, resnet50, resnet101, resnet152
resnet_type: resnet50

# 是否使用ImageNet预训练权重（推荐true）
pretrained: true
```

#### 训练参数

```yaml
# 训练轮数
epochs: 50

# 批次大小（根据GPU显存调整）
batch_size: 4

# 学习率（推荐范围：0.0001-0.001）
lr: 0.001

# 权重衰减/L2正则化
weight_decay: 0.00001

# 类别权重（处理类别不平衡）
# null = 不使用
# [1.0, 3.0] = 二分类，给第2类3倍权重
# [1.0, 2.0, 3.0] = 三分类
class_weights: null
```

#### 早停参数

```yaml
# 是否启用早停（防止过拟合）
early_stopping: true

# 连续N个epoch验证F1不提升则停止
patience: 10
```

#### 其他参数

```yaml
# 输出目录
output_dir: ./output/exp_01

# DataLoader工作线程数
num_workers: 4

# 随机种子（确保可复现）
seed: 42
```

---

## 💡 使用技巧

### 技巧1: 命令行覆盖配置

YAML配置 + 命令行参数可以灵活组合：

```bash
# 使用配置文件，但覆盖某些参数
python train.py --config config_basic.yaml --epochs 100 --lr 0.0001

# 覆盖输出目录
python train.py --config config_basic.yaml --output_dir ./output/exp_02
```

**优先级**: 命令行参数 > YAML配置

### 技巧2: 多个实验配置

为不同实验创建不同配置文件：

```bash
experiments/
├── exp01_baseline.yaml       # 基线实验
├── exp02_pretrained.yaml     # 使用预训练
├── exp03_large_model.yaml    # 大模型
└── exp04_class_weights.yaml  # 类别权重
```

运行实验：
```bash
python train.py --config experiments/exp01_baseline.yaml
python train.py --config experiments/exp02_pretrained.yaml
...
```

### 技巧3: 版本控制

将配置文件加入Git版本控制：

```bash
git add my_config.yaml
git commit -m "Add training config for experiment 01"
```

这样可以追踪每次实验的配置变化。

---

## 📊 常见场景配置

### 场景1: 首次训练

```yaml
csv_file: my_data.csv
val_split: 0.2
num_classes: 2
resnet_type: resnet50
pretrained: true
epochs: 50
batch_size: 4
lr: 0.001
weight_decay: 0.00001
class_weights: null
early_stopping: false
patience: 10
output_dir: ./output/first_run
num_workers: 4
seed: 42
```

```bash
python train.py --config first_run.yaml
```

### 场景2: 类别不平衡数据

假设数据分布：正常150个，中风50个（3:1）

```yaml
csv_file: imbalanced_data.csv
num_classes: 2
class_weights: [1.0, 3.0]  # 给中风类3倍权重
early_stopping: true
patience: 15
# ... 其他参数同上
```

### 场景3: 小显存GPU (≤8GB)

```yaml
csv_file: my_data.csv
resnet_type: resnet18      # 使用小模型
batch_size: 1              # 小批次
num_workers: 2             # 少线程
# ... 其他参数
```

### 场景4: 大数据集优化训练

```yaml
csv_file: large_dataset.csv
resnet_type: resnet101     # 大模型
pretrained: true
epochs: 100
batch_size: 8              # 大批次
lr: 0.0005                 # 小学习率
early_stopping: true
patience: 20
num_workers: 8             # 多线程
```

### 场景5: 多分类（3类）

```yaml
csv_file: multiclass_data.csv
num_classes: 3
class_weights: [1.0, 2.0, 3.0]  # 三个类别的权重
# ... 其他参数
```

---

## 🔧 计算类别权重

### 方法1: 使用Python脚本

```python
import pandas as pd

# 读取CSV
df = pd.read_csv('my_data.csv')

# 统计标签分布
label_counts = df['label'].value_counts().sort_index()
print("标签分布:")
print(label_counts)

# 计算权重
total = len(df)
num_classes = len(label_counts)
weights = [total / (num_classes * count) for count in label_counts]

print(f"\n推荐class_weights: {weights}")
```

### 方法2: 使用验证脚本

```bash
python validate_csv.py my_data.csv
```

输出会显示类别分布，根据比例设置权重。

### 权重设置示例

| 数据分布 | class_weights |
|----------|---------------|
| 正常:中风 = 1:1 | `[1.0, 1.0]` 或 `null` |
| 正常:中风 = 2:1 | `[1.0, 2.0]` |
| 正常:中风 = 3:1 | `[1.0, 3.0]` |
| 正常:缺血:出血 = 3:2:1 | `[1.0, 1.5, 3.0]` |

---

## ⚙️ 参数调优指南

### 学习率 (lr)

| 情况 | 推荐值 |
|------|--------|
| 小数据集 (<500) | 0.0001 - 0.0005 |
| 中等数据集 | 0.0005 - 0.001 |
| 大数据集 (>2000) | 0.001 - 0.002 |
| 使用预训练 | 0.0001 - 0.0005 |

### 批次大小 (batch_size)

| GPU显存 | 推荐batch_size |
|---------|----------------|
| ≤ 8GB | 1-2 |
| 12-16GB | 2-4 |
| ≥ 24GB | 4-8 |

### 权重衰减 (weight_decay)

| 情况 | 推荐值 |
|------|--------|
| 小数据集（容易过拟合） | 1e-4 - 1e-3 |
| 中等数据集 | 1e-5 - 1e-4 |
| 大数据集 | 1e-6 - 1e-5 |

### 训练轮数 (epochs)

| 情况 | 推荐值 |
|------|--------|
| 快速实验 | 10-20 |
| 正常训练 | 50-100 |
| 启用早停 | 100-200（实际可能提前停止） |

---

## 🐛 常见问题

### Q1: YAML语法错误

**错误**: `yaml.scanner.ScannerError`

**解决**:
- 检查缩进（使用空格，不要用Tab）
- 检查冒号后有空格
- 字符串包含特殊字符时加引号

```yaml
# 正确
csv_file: data.csv
class_weights: [1.0, 2.0]

# 错误
csv_file:data.csv           # 冒号后缺少空格
class_weights:[1.0, 2.0]    # 冒号后缺少空格
```

### Q2: 配置文件路径找不到

**错误**: `FileNotFoundError: config.yaml`

**解决**:
- 使用绝对路径或相对于当前目录的路径
- 检查文件名拼写

```bash
# 推荐：使用相对路径
python train.py --config ./configs/my_config.yaml

# 或绝对路径
python train.py --config /home/user/project/config.yaml
```

### Q3: 参数类型错误

**错误**: `TypeError: 'NoneType' object is not iterable`

**解决**: 确保所有必需参数都在YAML中设置

必需参数：
- `csv_file`
- `num_classes`

### Q4: 命令行参数不生效

**记住**: 命令行参数优先级高于YAML配置

如果命令行参数似乎不生效，检查：
- 参数名是否正确
- YAML中是否设置了相同参数

---

## 📚 更多示例

查看项目中的示例配置文件：

```bash
ls -1 config_*.yaml
```

输出：
```
config_advanced.yaml
config_basic.yaml
config_multiclass.yaml
config_small_gpu.yaml
config_template.yaml
```

每个文件都包含详细注释和使用说明。

---

## 🎓 最佳实践

1. **使用模板**: 从`config_template.yaml`开始
2. **添加注释**: 在配置中记录为什么选择某个参数值
3. **命名规范**: 使用描述性的配置文件名（如`exp01_baseline.yaml`）
4. **版本控制**: 将配置文件加入Git
5. **记录结果**: 在配置文件中添加注释记录训练结果

示例：
```yaml
# Experiment 01 - Baseline
# Date: 2024-01-16
# Result: Val F1=0.85, Acc=0.87
# Note: Good baseline, try increasing epochs next

csv_file: data.csv
epochs: 50
# ...
```

---

## 💻 完整工作流

```bash
# 1. 复制模板
cp config_template.yaml my_experiment.yaml

# 2. 编辑配置
nano my_experiment.yaml

# 3. 验证数据
python validate_csv.py my_data.csv

# 4. 训练模型
python train.py --config my_experiment.yaml

# 5. 如需调整，覆盖部分参数
python train.py --config my_experiment.yaml --epochs 100

# 6. 保存结果和配置
git add my_experiment.yaml
git commit -m "Experiment: baseline with config"
```

---

需要帮助？查看：
- `TRAINING_GUIDE.md` - 完整训练指南
- `CSV_FORMAT.md` - CSV格式说明
- `USAGE_GUIDE.md` - 模型使用指南
