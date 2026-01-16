# CTP 分类模型训练指南

## 📋 数据准备

### 1. CSV文件格式

创建CSV文件，包含以下三列：

| 列名 | 说明 | 示例 |
|------|------|------|
| `ctp_path` | CTP数据的.nii.gz文件路径 | `/data/patient_001/ctp.nii.gz` |
| `features_dir` | 5个特征图所在的目录路径 | `/data/patient_001/features/` |
| `label` | 分类标签（0或1，多分类可以是0,1,2...） | `0` |

### 2. 目录结构示例

```
your_data/
├── patient_001/
│   ├── ctp.nii.gz                        # CTP 4D数据 (512×512×32×21)
│   └── features/
│       ├── generated_cbf.nii.gz          # 脑血流量
│       ├── generated_cbv.nii.gz          # 脑血容量
│       ├── generated_mtt.nii.gz          # 平均通过时间
│       ├── generated_tmax.nii.gz         # 达峰时间
│       └── generated_ttp.nii.gz          # 峰值时间
├── patient_002/
│   ├── ctp.nii.gz
│   └── features/
│       ├── generated_cbf.nii.gz
│       ├── generated_cbv.nii.gz
│       ├── generated_mtt.nii.gz
│       ├── generated_tmax.nii.gz
│       └── generated_ttp.nii.gz
└── ...
```

### 3. CSV示例文件

#### T=21 时间点 (data_t21.csv)

```csv
ctp_path,features_dir,label
/data/patient_001/ctp.nii.gz,/data/patient_001/features/,0
/data/patient_002/ctp.nii.gz,/data/patient_002/features/,1
/data/patient_003/ctp.nii.gz,/data/patient_003/features/,0
/data/patient_004/ctp.nii.gz,/data/patient_004/features/,1
/data/patient_005/ctp.nii.gz,/data/patient_005/features/,0
```

#### T=20 时间点 (data_t20.csv)

```csv
ctp_path,features_dir,label
/data/stroke_t20/case_001/ctp.nii.gz,/data/stroke_t20/case_001/features/,0
/data/stroke_t20/case_002/ctp.nii.gz,/data/stroke_t20/case_002/features/,1
/data/stroke_t20/case_003/ctp.nii.gz,/data/stroke_t20/case_003/features/,0
/data/stroke_t20/case_004/ctp.nii.gz,/data/stroke_t20/case_004/features/,1
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据CSV

根据上面的格式创建您的CSV文件，例如 `my_data.csv`

### 3. 开始训练（基础用法）

```bash
# T=21时间点，二分类
python train.py --csv_file data_t21.csv --num_time_points 21 --num_classes 2 --pretrained

# T=20时间点，二分类
python train.py --csv_file data_t20.csv --num_time_points 20 --num_classes 2 --pretrained
```

---

## 🎯 完整训练命令示例

### 示例1: 基础训练（推荐新手）

```bash
python train.py \
  --csv_file data_t21.csv \
  --num_time_points 21 \
  --num_classes 2 \
  --resnet_type resnet50 \
  --pretrained \
  --epochs 50 \
  --batch_size 2 \
  --lr 0.001 \
  --output_dir ./output/exp_001
```

### 示例2: 类别不平衡处理

如果您的数据中正常样本:中风样本 = 3:1，使用加权损失：

```bash
python train.py \
  --csv_file data_t21.csv \
  --num_time_points 21 \
  --num_classes 2 \
  --pretrained \
  --class_weights 1.0 3.0 \
  --epochs 50 \
  --batch_size 2 \
  --output_dir ./output/weighted
```

### 示例3: 大数据集训练

```bash
python train.py \
  --csv_file large_dataset.csv \
  --num_time_points 21 \
  --num_classes 2 \
  --resnet_type resnet101 \
  --pretrained \
  --epochs 100 \
  --batch_size 4 \
  --lr 0.0001 \
  --weight_decay 1e-4 \
  --early_stopping \
  --patience 15 \
  --num_workers 8 \
  --output_dir ./output/large_exp
```

### 示例4: 多分类（3类）

例如：正常 vs 缺血 vs 出血

```bash
python train.py \
  --csv_file multiclass_data.csv \
  --num_time_points 21 \
  --num_classes 3 \
  --resnet_type resnet50 \
  --pretrained \
  --class_weights 1.0 2.0 2.5 \
  --epochs 50 \
  --batch_size 2 \
  --output_dir ./output/multiclass
```

### 示例5: 从头训练（不用预训练）

```bash
python train.py \
  --csv_file data_t21.csv \
  --num_time_points 21 \
  --num_classes 2 \
  --resnet_type resnet34 \
  --epochs 100 \
  --batch_size 2 \
  --lr 0.001 \
  --output_dir ./output/from_scratch
```

---

## 🔧 命令行参数详解

### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--csv_file` | str | **必需** | CSV文件路径 |
| `--val_split` | float | 0.2 | 验证集比例 |
| `--num_time_points` | int | 21 | CTP时间点数量 (20或21) |
| `--num_classes` | int | 2 | 分类类别数 |

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--resnet_type` | str | resnet50 | ResNet类型 (18/34/50/101/152) |
| `--pretrained` | flag | False | 是否使用预训练权重 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 50 | 训练轮数 |
| `--batch_size` | int | 2 | 批次大小 |
| `--lr` | float | 0.001 | 学习率 |
| `--weight_decay` | float | 1e-5 | 权重衰减 |
| `--class_weights` | float[] | None | 类别权重，例如 `1.0 3.0` |

### 早停参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--early_stopping` | flag | False | 是否启用早停 |
| `--patience` | int | 10 | 早停patience |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_dir` | str | ./output | 输出目录 |
| `--num_workers` | int | 4 | DataLoader工作线程数 |
| `--seed` | int | 42 | 随机种子 |

---

## 📊 训练输出

训练完成后，`output_dir` 中会包含：

```
output/
├── best_model.pth              # 最佳模型（按F1分数）
├── final_model.pth             # 最终模型
└── training_history.csv        # 训练历史（loss, acc, f1等）
```

### 加载训练好的模型

```python
import torch
from ctp_classification_model import create_ctp_classifier

# 创建模型（参数需与训练时一致）
model = create_ctp_classifier(
    num_time_points=21,
    num_classes=2,
    resnet_type='resnet50'
)

# 加载权重
checkpoint = torch.load('output/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"加载epoch {checkpoint['epoch']}的模型")
print(f"验证准确率: {checkpoint['val_acc']:.4f}")
print(f"验证F1分数: {checkpoint['val_f1']:.4f}")
```

---

## 💡 实用技巧

### 1. 计算类别权重

如果数据不平衡，建议使用类别权重：

```python
import pandas as pd

df = pd.read_csv('data_t21.csv')
label_counts = df['label'].value_counts().sort_index()
print(label_counts)

# 计算权重（使用总数/类别数）
total = len(df)
num_classes = len(label_counts)
weights = [total / (num_classes * count) for count in label_counts]
print(f"建议权重: {' '.join(map(str, weights))}")
```

然后在训练时使用：
```bash
python train.py --csv_file data_t21.csv --class_weights 1.2 2.8 ...
```

### 2. 批次大小选择

- GPU内存 <= 8GB: `--batch_size 1` 或 `2`
- GPU内存 12-16GB: `--batch_size 2` 或 `4`
- GPU内存 >= 24GB: `--batch_size 4` 或 `8`

### 3. 学习率调整

- 小数据集 (<1000样本): `--lr 0.0001`
- 中等数据集: `--lr 0.001` (默认)
- 大数据集 (>5000样本): `--lr 0.001` 或 `0.0005`

### 4. ResNet类型选择

| 模型 | 参数量 | 推荐场景 |
|------|--------|----------|
| resnet18 | ~11M | 小数据集 (<500) 或快速实验 |
| resnet34 | ~21M | 中小数据集 (500-2000) |
| resnet50 | ~23M | 中等数据集 (1000-5000) ✅ 推荐 |
| resnet101 | ~42M | 大数据集 (>5000) |
| resnet152 | ~58M | 超大数据集 (>10000) |

---

## ⚠️ 常见问题

### Q1: CUDA out of memory 错误
**解决方案:**
- 减小 batch_size: `--batch_size 1`
- 使用更小的模型: `--resnet_type resnet18`
- 减少 num_workers: `--num_workers 2`

### Q2: 训练速度很慢
**解决方案:**
- 增加 num_workers: `--num_workers 8`
- 使用GPU: 确保 `torch.cuda.is_available()` 返回 True
- 启用 pin_memory（代码已自动处理）

### Q3: 验证准确率不提升
**解决方案:**
- 检查数据是否平衡，使用 `--class_weights`
- 降低学习率: `--lr 0.0001`
- 增加训练轮数: `--epochs 100`
- 使用预训练: `--pretrained`

### Q4: FileNotFoundError: 找不到特征图
**检查：**
- features_dir 路径是否正确（结尾有无 `/` 都可以）
- 特征图文件名是否完全匹配：
  - `generated_cbf.nii.gz`
  - `generated_cbv.nii.gz`
  - `generated_mtt.nii.gz`
  - `generated_tmax.nii.gz`
  - `generated_ttp.nii.gz`

### Q5: 图像尺寸不对
**当前要求:**
- CTP数据: `(512, 512, 32, T)` 其中T=20或21
- 特征图: `(512, 512)` 或 `(512, 512, Z)` 会自动取中间层

如果尺寸不匹配，代码会自动resize（使用scipy.ndimage.zoom）

---

## 📈 监控训练进度

### 实时查看日志

```bash
# 训练时重定向输出
python train.py --csv_file data.csv ... 2>&1 | tee train.log
```

### 查看训练历史

```python
import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv('output/training_history.csv')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## 🎓 推荐训练流程

### 第一步：小规模实验
```bash
# 使用小模型快速验证数据加载是否正确
python train.py \
  --csv_file data_t21.csv \
  --num_time_points 21 \
  --resnet_type resnet18 \
  --epochs 5 \
  --batch_size 2 \
  --output_dir ./output/test
```

### 第二步：完整训练
```bash
# 使用完整配置训练
python train.py \
  --csv_file data_t21.csv \
  --num_time_points 21 \
  --num_classes 2 \
  --resnet_type resnet50 \
  --pretrained \
  --epochs 50 \
  --batch_size 2 \
  --lr 0.001 \
  --early_stopping \
  --patience 10 \
  --output_dir ./output/full_training
```

### 第三步：评估和调优
- 查看混淆矩阵
- 调整类别权重
- 尝试不同的学习率
- 使用更大的模型或更多epoch

---

## 📞 技术支持

遇到问题？检查：
1. requirements.txt 中的依赖是否全部安装
2. CSV文件格式是否正确
3. 数据路径是否存在且可访问
4. 特征图文件名是否完全匹配

更多帮助请查看：
- `USAGE_GUIDE.md` - 模型使用指南
- `example_usage.py` - 示例代码
- `test_model.py` - 模型测试
