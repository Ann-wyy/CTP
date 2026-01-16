# 灵活时间点支持

## 🎯 概述

训练脚本现在**支持任意时间点数**（20, 21, 22, 23, ...），而不仅限于T=20和T=21。

您可以：
- ✅ 使用单一时间点的数据集（如全部T=22）
- ✅ 混合不同时间点的数据集（如同时包含T=20, T=21, T=22）
- ✅ 自动为每个时间点创建独立的模型

---

## 📋 CSV格式

CSV文件的`time_points`列可以是**任意正整数**：

```csv
ctp_path,features_dir,time_points,label
/data/patient_001/ctp.nii.gz,/data/patient_001/features/,21,0
/data/patient_002/ctp.nii.gz,/data/patient_002/features/,22,1
/data/patient_003/ctp.nii.gz,/data/patient_003/features/,20,0
```

---

## 🚀 使用示例

### 示例1: 仅使用T=22的数据

```csv
ctp_path,features_dir,time_points,label
/data/case_001/ctp.nii.gz,/data/case_001/features/,22,0
/data/case_002/ctp.nii.gz,/data/case_002/features/,22,1
/data/case_003/ctp.nii.gz,/data/case_003/features/,22,0
```

训练：
```bash
python train.py --config config_basic.yaml
```

训练时会自动：
1. 检测数据集包含T=22
2. 创建T=22的模型
3. 保存为`best_model_t22.pth`

### 示例2: 混合T=21和T=22的数据

```csv
ctp_path,features_dir,time_points,label
/data/old_scanner/patient_001.nii.gz,/data/old_scanner/patient_001/,21,0
/data/old_scanner/patient_002.nii.gz,/data/old_scanner/patient_002/,21,1
/data/new_scanner/patient_003.nii.gz,/data/new_scanner/patient_003/,22,0
/data/new_scanner/patient_004.nii.gz,/data/new_scanner/patient_004/,22,1
```

训练时会自动：
1. 检测数据集包含T=21和T=22
2. 创建两个独立模型（T=21模型和T=22模型）
3. 训练时根据time_points自动选择对应模型
4. 保存两个模型文件：`best_model_t21.pth`和`best_model_t22.pth`

### 示例3: 混合T=20, T=21, T=22三种时间点

```csv
ctp_path,features_dir,time_points,label
/data/site_a/patient_001.nii.gz,/data/site_a/patient_001/,20,0
/data/site_b/patient_002.nii.gz,/data/site_b/patient_002/,21,1
/data/site_c/patient_003.nii.gz,/data/site_c/patient_003/,22,0
```

训练时会自动创建三个模型，保存为：
- `best_model_t20.pth`
- `best_model_t21.pth`
- `best_model_t22.pth`

---

## 📝 配置文件示例

YAML配置文件无需特殊修改，只需确保CSV路径正确：

```yaml
# config_t22.yaml
csv_file: data_with_t22.csv
val_split: 0.2
num_classes: 2

resnet_type: resnet50
pretrained: true

epochs: 50
batch_size: 4
lr: 0.001
weight_decay: 0.00001

early_stopping: true
patience: 10

output_dir: ./output/t22_exp
num_workers: 4
seed: 42
```

---

## 🔍 训练输出说明

### 加载数据时的输出

```
加载数据...
======================================================================
数据集包含的时间点: [20, 21, 22]
加载数据集: 150 个样本
时间点分布:
20    50
21    50
22    50
类别分布:
0    75
1    75
```

### 创建模型时的输出

```
创建模型...
======================================================================
创建 T=20 模型...
  T=20 模型参数: 23,761,986
创建 T=21 模型...
  T=21 模型参数: 23,762,050
创建 T=22 模型...
  T=22 模型参数: 23,762,114
```

可以看到每个时间点都创建了独立的模型。

### 训练完成后的输出

```
output/
├── best_model_t20.pth          # T=20的最佳模型
├── best_model_t21.pth          # T=21的最佳模型
├── best_model_t22.pth          # T=22的最佳模型
└── training_history.csv        # 训练历史
```

---

## 🎓 工作原理

### 1. 自动检测时间点

训练脚本会扫描CSV文件中的`time_points`列，自动识别所有唯一的时间点值。

### 2. 动态创建模型

为每个检测到的时间点创建一个独立的CTPClassificationNet模型：

```python
# 伪代码
unique_time_points = [20, 21, 22]  # 从CSV中检测

models = {}
for T in unique_time_points:
    models[T] = CTPClassificationNet(num_time_points=T, ...)
```

### 3. 批次分组

在批次处理时，数据会按时间点自动分组：
- T=20的样本发送到T=20模型
- T=21的样本发送到T=21模型
- T=22的样本发送到T=22模型

### 4. 统一训练

尽管有多个模型，但它们共享：
- 相同的损失函数
- 相同的训练目标
- 统一的评估指标

---

## 💡 最佳实践

### 1. 数据平衡

如果混合不同时间点，建议保持各时间点样本数量相近：

```python
# 检查数据分布
import pandas as pd

df = pd.read_csv('your_data.csv')
print(df['time_points'].value_counts())

# 输出示例:
# 21    100
# 22     95
# 20     90
```

### 2. 批次大小调整

如果使用混合时间点且某个时间点样本很少，考虑调整batch_size：

```yaml
# 如果T=22只有10个样本，而T=21有100个
batch_size: 2  # 使用较小的batch_size
```

### 3. 模型选择

对于不同时间点，模型架构本质相同，只是前端编码器输入通道数不同：
- T=20: 32×20=640通道
- T=21: 32×21=672通道
- T=22: 32×22=704通道

### 4. 推理时的模型选择

推理时需要根据输入数据的时间点选择对应的模型：

```python
# 加载对应时间点的模型
if time_points == 20:
    checkpoint = torch.load('best_model_t20.pth')
elif time_points == 21:
    checkpoint = torch.load('best_model_t21.pth')
elif time_points == 22:
    checkpoint = torch.load('best_model_t22.pth')

model = create_ctp_classifier(num_time_points=time_points, ...)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## ⚠️ 注意事项

### 1. CTP数据维度必须匹配

CSV中的`time_points`值必须与实际.nii.gz文件的时间维度一致：

```python
# 如果CSV中写的是 time_points=22
# 那么对应的.nii.gz文件shape必须是 (512, 512, 32, 22)
```

否则会报错：
```
ValueError: CTP数据时间点不匹配。CSV中为22，但文件中为21
```

### 2. 时间点值必须为正整数

```yaml
# ✅ 正确
time_points: 22

# ❌ 错误
time_points: 22.5  # 不能是小数
time_points: -1    # 不能是负数
time_points: 0     # 必须大于0
```

### 3. 显存占用

如果数据集包含多个时间点，会同时加载多个模型到GPU，注意显存占用：

```
1个模型: ~1GB显存
2个模型: ~2GB显存
3个模型: ~3GB显存
```

如果显存不足，可以：
- 使用更小的模型（resnet18）
- 减小batch_size
- 使用单一时间点的数据集

---

## 📚 相关文档

- **CSV_FORMAT.md** - CSV格式详细说明
- **TRAINING_GUIDE.md** - 训练指南
- **CONFIG_GUIDE.md** - YAML配置指南

---

## 🎉 总结

现在您可以灵活使用任意时间点的CTP数据：

1. **准备CSV**：在`time_points`列填写实际时间点数
2. **运行训练**：`python train.py --config config.yaml`
3. **自动处理**：脚本会自动检测并创建所需的模型
4. **获取结果**：每个时间点都会有对应的模型文件

**无需修改代码，开箱即用！**
