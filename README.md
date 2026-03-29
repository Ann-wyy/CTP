# CTP分类模型

基于ResNet的CT灌注(CTP)图像分类模型，用于脑卒中分类任务。

## 项目结构

```
.
├── ctp_classification_model.py    # 核心模型代码
├── train.py                       # 训练脚本
├── validate_csv.py                # CSV数据验证工具
├── config.yaml                    # 配置文件模板
└── requirements.txt               # Python依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

创建CSV文件，格式如下（**必须包含表头**）：

```csv
label,nii_path,time_points,mask_path
0,/path/to/patient001/ctp.nii.gz,20,/path/to/patient001/features
1,/path/to/patient002/ctp.nii.gz,21,/path/to/patient002/features
0,/path/to/patient003/ctp.nii.gz,22,/path/to/patient003/features
```

**列说明：**
- **label**: 标签（0=正常，1=异常，支持多分类）
- **nii_path**: CTP图像路径（.nii.gz格式，形状为512×512×32×T）
- **time_points**: 时间点数量（T，可以是20、21、22等任意正整数）
- **mask_path**: 特征图目录（包含5个.nii.gz文件：generated_cbf.nii.gz, generated_cbv.nii.gz, generated_mtt.nii.gz, generated_tmax.nii.gz, generated_ttp.nii.gz）

**验证数据格式：**

```bash
python validate_csv.py data.csv
```

### 3. 配置训练参数

编辑 `config.yaml`：

```yaml
csv_file: /path/to/your/data.csv    # 数据CSV路径
val_split: 0.2                       # 验证集比例
num_classes: 2                       # 分类类别数
resnet_type: resnet50                # 模型类型
pretrained: true                     # 使用ImageNet预训练
epochs: 50                           # 训练轮数
batch_size: 4                        # 批次大小
lr: 0.001                            # 学习率
output_dir: ./output/exp_01          # 输出目录
```

**关键参数说明：**

- **resnet_type**: `resnet18`（快速）| `resnet34` | `resnet50`（推荐）| `resnet101` | `resnet152`（大模型）
- **batch_size**: 根据GPU显存调整（8GB显存建议1-2，16GB建议2-4）
- **class_weights**: 类别不平衡时使用，例如 `[1.0, 3.0]` 给第2类3倍权重
- **use_amp**: 混合精度训练（推荐在A100/V100/RTX 3090上启用，可减少50%显存，加速30-50%）

### 4. 开始训练

```bash
python train.py --config config.yaml
```

训练过程会自动：
- 按时间点数量分组训练（每个时间点T创建一个独立模型）
- 保存最佳模型到 `output_dir/best_model_tXX.pth`
- 保存训练日志到 `output_dir/training.log`
- 在验证F1不提升时早停（如启用early_stopping）

## 模型使用

### 推理示例

```python
import torch
from ctp_classification_model import create_ctp_classifier

# 1. 创建模型
model = create_ctp_classifier(
    num_time_points=21,      # 根据数据时间点数
    num_classes=2,
    resnet_type='resnet50',
    pretrained=False
)

# 2. 加载训练好的权重
checkpoint = torch.load('output/exp_01/best_model_t21.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. 推理
ctp_data = torch.randn(1, 512, 512, 32, 21)     # 原始CTP数据
prior_maps = torch.randn(1, 5, 512, 512)        # 灌注特征图
with torch.no_grad():
    output = model(ctp_data, prior_maps)
    pred = torch.argmax(output, dim=1)
    print(f"预测类别: {pred.item()}")
```

## 常见问题

**Q: 支持哪些时间点数量？**
A: 支持任意正整数（20、21、22、25等），模型会自动适配。

**Q: 显存不足怎么办？**
A:
1. 启用混合精度训练（设置`use_amp: true`，可减少50%显存）
2. 减小`batch_size`（最小为1）
3. 使用更小的`resnet_type`（如resnet18）

**Q: 如何加速训练？**
A:
1. 启用混合精度训练（设置`use_amp: true`，在A100/V100/RTX 3090上可提速30-50%）
2. 增大`batch_size`（如果显存允许）
3. 增加`num_workers`（数据加载线程数）

**Q: 类别不平衡怎么办？**
A: 在config.yaml中设置`class_weights`，例如`[1.0, 3.0]`给少数类更高权重。

**Q: 训练后在哪找模型？**
A: 在`output_dir/best_model_tXX.pth`，其中XX是时间点数量。

## 引用

如果使用本代码，请引用相关论文。

## 许可

[添加您的许可信息]
