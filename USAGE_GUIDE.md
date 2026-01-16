# CTP 分类网络使用指南

## 快速开始

### 1. 基础使用 - 创建模型并进行预测

```python
import torch
from ctp_classification_model import create_ctp_classifier

# 创建模型（二分类：中风 vs 正常）
model = create_ctp_classifier(
    num_time_points=21,      # CTP时间点数量 (20 或 21)
    num_classes=2,           # 分类类别数
    resnet_type='resnet50',  # ResNet类型
    pretrained=True          # 使用预训练权重
)

# 移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 准备输入数据
# CTP数据: (batch_size, 512, 512, 32, 21) - 4D灰度医学图像
# 先验图: (batch_size, 5, 512, 512) - CBF, CBV, MTT, Tmax, TTP
ctp_data = torch.randn(2, 512, 512, 32, 21).to(device)
prior_maps = torch.randn(2, 5, 512, 512).to(device)

# 推理
model.eval()
with torch.no_grad():
    logits = model(ctp_data, prior_maps)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

print(f"预测类别: {predictions}")
print(f"预测概率: {probabilities}")
```

---

## 2. 训练模型

```python
import torch.nn as nn
import torch.optim as optim

# 创建模型
model = create_ctp_classifier(
    num_time_points=21,
    num_classes=2,
    resnet_type='resnet34',  # 使用较小的模型训练更快
    pretrained=True
).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
model.train()
for epoch in range(num_epochs):
    for ctp_data, prior_maps, labels in train_loader:
        ctp_data = ctp_data.to(device)
        prior_maps = prior_maps.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(ctp_data, prior_maps)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## 3. 准备您的数据

### 数据格式要求

```python
# CTP数据格式
ctp_shape = (batch_size, H, W, Z, T)
# - H, W = 512 (图像高度和宽度)
# - Z = 32 (层数/切片数)
# - T = 20 或 21 (时间点数量)

# 先验灌注图格式
prior_shape = (batch_size, 5, H, W)
# 5个通道依次为:
# 0: CBF (脑血流量)
# 1: CBV (脑血容量)
# 2: MTT (平均通过时间)
# 3: Tmax (达峰时间)
# 4: TTP (峰值时间)

# 标签格式（二分类）
labels_shape = (batch_size,)  # 值为 0 或 1
```

### 数据加载示例

```python
from torch.utils.data import Dataset, DataLoader

class CTPDataset(Dataset):
    def __init__(self, ctp_paths, prior_paths, labels):
        self.ctp_paths = ctp_paths
        self.prior_paths = prior_paths
        self.labels = labels

    def __len__(self):
        return len(self.ctp_paths)

    def __getitem__(self, idx):
        # 从文件加载您的数据
        ctp_data = load_ctp_data(self.ctp_paths[idx])      # (512, 512, 32, 21)
        prior_maps = load_prior_maps(self.prior_paths[idx]) # (5, 512, 512)
        label = self.labels[idx]

        # 转换为tensor
        ctp_data = torch.from_numpy(ctp_data).float()
        prior_maps = torch.from_numpy(prior_maps).float()
        label = torch.tensor(label, dtype=torch.long)

        return ctp_data, prior_maps, label

# 创建DataLoader
dataset = CTPDataset(ctp_paths, prior_paths, labels)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

---

## 4. 模型配置选项

### ResNet类型选择

```python
# 从小到大的模型选择（根据数据量和计算资源选择）
resnet_type='resnet18'   # ~11M 参数，推荐用于小数据集或快速实验
resnet_type='resnet34'   # ~21M 参数
resnet_type='resnet50'   # ~23M 参数，推荐用于一般情况
resnet_type='resnet101'  # ~42M 参数，用于大数据集
resnet_type='resnet152'  # ~58M 参数，用于大数据集
```

### 多分类

```python
# 三分类示例：正常、缺血、出血
model = create_ctp_classifier(
    num_time_points=21,
    num_classes=3,          # 改为3类
    resnet_type='resnet50',
    pretrained=True
)
```

### 不同时间点

```python
# T=20时间点
model = create_ctp_classifier(
    num_time_points=20,     # 改为20
    num_classes=2,
    resnet_type='resnet50',
    pretrained=True
)

# 输入数据相应改变
ctp_data = torch.randn(2, 512, 512, 32, 20).to(device)  # T=20
```

---

## 5. 特征提取

```python
# 提取中间特征用于可视化或分析
model.eval()
with torch.no_grad():
    learned_features, prior_features, combined_features = \
        model.get_feature_maps(ctp_data, prior_maps)

print(f"学习特征: {learned_features.shape}")    # (B, 155, 512, 512)
print(f"先验特征: {prior_features.shape}")      # (B, 5, 512, 512)
print(f"组合特征: {combined_features.shape}")   # (B, 160, 512, 512)
```

---

## 6. 保存和加载模型

```python
# 保存模型
torch.save(model.state_dict(), 'ctp_model.pth')

# 加载模型
model = create_ctp_classifier(
    num_time_points=21,
    num_classes=2,
    resnet_type='resnet50',
    pretrained=False  # 不需要预训练
)
model.load_state_dict(torch.load('ctp_model.pth'))
model.to(device)
model.eval()
```

---

## 7. 完整示例代码

查看项目中的详细示例：

```bash
# 运行所有示例
python example_usage.py

# 运行测试
python test_model.py

# 快速测试模型
python ctp_classification_model.py
```

---

## 常见问题

### Q1: 我的图像尺寸不是512x512怎么办？
**A:** 需要预处理将图像resize到512x512，或修改模型支持任意尺寸（需要调整代码）

### Q2: 我没有先验灌注图怎么办？
**A:** 如果没有CBF/CBV等先验图，可以：
- 传入全零tensor: `torch.zeros(batch_size, 5, 512, 512)`
- 或修改模型去掉PriorFusionModule

### Q3: 如何选择pretrained=True还是False？
**A:**
- `pretrained=True`: 使用ImageNet预训练权重，推荐优先尝试（通常收敛更快）
- `pretrained=False`: 完全从头训练，适合数据量很大或预训练效果不好的情况

### Q4: 训练需要多少数据？
**A:** 建议至少：
- 使用pretrained=True: 每类至少100-500个样本
- 使用pretrained=False: 每类至少1000+个样本

### Q5: GPU内存不够怎么办？
**A:**
- 减小batch_size
- 使用更小的ResNet (resnet18/34)
- 使用梯度累积技术

---

## 性能建议

1. **数据增强**: 使用旋转、翻转、亮度调整等增强医学图像
2. **学习率调度**: 使用学习率衰减策略
3. **早停**: 监控验证集性能，避免过拟合
4. **交叉验证**: 使用K折交叉验证评估模型泛化性能
5. **类别不平衡**: 如果数据不平衡，使用加权损失函数

```python
# 加权损失函数示例
weights = torch.tensor([1.0, 3.0]).to(device)  # 给少数类更大权重
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## 技术支持

- 查看 `README.md` 了解架构细节
- 查看 `example_usage.py` 获取更多示例
- 运行 `test_model.py` 验证环境配置
