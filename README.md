# CTP Classification Network

A PyTorch implementation of a CT Perfusion (CTP) classification network that combines raw 4D CTP data with perfusion prior maps for stroke classification.

## Architecture Overview

The network consists of three main components:

1. **Frontend Encoder**: Processes raw 4D CTP data
   - Input: `(B, 512, 512, 32, T)` where T ∈ {20, 21}
   - Reshapes to: `(B, 32*T, 512, 512)`
   - Output: `(B, 155, 512, 512)` learned features

2. **Prior Fusion Module**: Adapts perfusion prior maps
   - Input: `(B, 5, 512, 512)` - 5 perfusion maps (CBF, CBV, MTT, Tmax, TTP)
   - Output: `(B, 5, 512, 512)` adapted features

3. **Modified ResNet Backbone**: Classification
   - Input: `(B, 160, 512, 512)` - concatenated learned + prior features
   - Output: `(B, num_classes)` classification logits

## Features

- ✅ Flexible time points: supports both 20 and 21 time points
- ✅ Multi-class classification support
- ✅ Multiple ResNet backbones (ResNet18/34/50/101/152)
- ✅ Optional pretrained weights
- ✅ Feature extraction capabilities
- ✅ Clean, well-documented code

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0

## Quick Start

### Basic Usage

```python
import torch
from ctp_classification_model import create_ctp_classifier

# Create model
model = create_ctp_classifier(
    num_time_points=21,
    num_classes=2,
    resnet_type='resnet50',
    pretrained=False
)

# Prepare input data
ctp_data = torch.randn(2, 512, 512, 32, 21)      # Raw CTP 4D data
prior_maps = torch.randn(2, 5, 512, 512)         # Perfusion maps

# Forward pass
output = model(ctp_data, prior_maps)
print(output.shape)  # (2, 2)
```

### Training Example

```python
import torch.nn as nn
import torch.optim as optim

# Setup
model = create_ctp_classifier(num_time_points=21, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(num_epochs):
    outputs = model(ctp_data, prior_maps)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Feature Extraction

```python
# Extract intermediate features
learned, priors, combined = model.get_feature_maps(ctp_data, prior_maps)

print(learned.shape)   # (B, 155, 512, 512)
print(priors.shape)    # (B, 5, 512, 512)
print(combined.shape)  # (B, 160, 512, 512)
```

## Model Configurations

### ResNet Variants

```python
# Lightweight model
model = create_ctp_classifier(resnet_type='resnet18')

# Balanced model (recommended)
model = create_ctp_classifier(resnet_type='resnet50')

# Heavy model
model = create_ctp_classifier(resnet_type='resnet101')
```

### Time Points

```python
# For 20 time points
model = create_ctp_classifier(num_time_points=20)  # Input: 32*20=640 channels

# For 21 time points
model = create_ctp_classifier(num_time_points=21)  # Input: 32*21=672 channels
```

### Multi-class Classification

```python
# Binary classification (stroke vs no stroke)
model = create_ctp_classifier(num_classes=2)

# Multi-class (e.g., no stroke, ischemic, hemorrhagic)
model = create_ctp_classifier(num_classes=3)
```

## Input Data Format

### CTP Data
- Shape: `(B, 512, 512, 32, T)`
- B: Batch size
- 512 × 512: Spatial dimensions (H × W)
- 32: Number of slices (Z)
- T: Number of time points (20 or 21)

### Prior Maps
- Shape: `(B, 5, 512, 512)`
- Channel 0: CBF (Cerebral Blood Flow)
- Channel 1: CBV (Cerebral Blood Volume)
- Channel 2: MTT (Mean Transit Time)
- Channel 3: Tmax (Time to Maximum)
- Channel 4: TTP (Time to Peak)

## Model Parameters

| ResNet Type | Parameters | Memory (approx) |
|-------------|------------|-----------------|
| ResNet18    | ~11M       | ~45 MB          |
| ResNet34    | ~21M       | ~85 MB          |
| ResNet50    | ~23M       | ~95 MB          |
| ResNet101   | ~42M       | ~170 MB         |
| ResNet152   | ~58M       | ~235 MB         |

*Note: Actual memory usage depends on batch size and input resolution.*

## Architecture Details

### Frontend Encoder
```
Input (B, 512, 512, 32, T)
  ↓ Reshape
(B, 32*T, 512, 512)
  ↓ 1×1 Conv + BN + ReLU
(B, 256, 512, 512)
  ↓ 3×3 Conv + BN + ReLU
(B, 155, 512, 512)
```

### Prior Fusion Module
```
Input (B, 5, 512, 512)
  ↓ 1×1 Conv + BN + ReLU
Output (B, 5, 512, 512)
```

### Modified ResNet
```
Input (B, 160, 512, 512)
  ↓ Modified Conv1 (160 channels)
  ↓ BN + ReLU + MaxPool
  ↓ Layer1-4 (ResNet blocks)
  ↓ AvgPool + Flatten
  ↓ FC
Output (B, num_classes)
```

## Advanced Usage

### Custom Configuration

```python
from ctp_classification_model import CTPClassificationNet

model = CTPClassificationNet(
    num_time_points=21,
    num_classes=2,
    resnet_type='resnet50',
    pretrained=True,
    learned_features=155,  # Custom number of learned features
    prior_channels=5        # Number of prior maps
)
```

### Pretrained Weights

```python
# Use pretrained ResNet weights (from ImageNet)
# Useful for transfer learning
model = create_ctp_classifier(pretrained=True)
```

### Model Saving and Loading

```python
# Save model
torch.save(model.state_dict(), 'ctp_model.pth')

# Load model
model = create_ctp_classifier(num_time_points=21, num_classes=2)
model.load_state_dict(torch.load('ctp_model.pth'))
model.eval()
```

## Examples

Run the example script to see various usage patterns:

```bash
python example_usage.py
```

This will demonstrate:
1. Basic usage
2. Training loop
3. Feature extraction
4. Different configurations
5. Multi-class classification
6. Model saving/loading

## Testing

Run the built-in test:

```bash
python ctp_classification_model.py
```

This will test the model with dummy data and print:
- Input/output shapes
- Feature map shapes
- Model parameter counts

## Performance Considerations

### Memory Optimization
- Use smaller ResNet variants (ResNet18/34) for limited GPU memory
- Reduce batch size if encountering OOM errors
- Consider gradient checkpointing for very deep models

### Speed Optimization
- Use mixed precision training (AMP)
- Enable cuDNN benchmarking: `torch.backends.cudnn.benchmark = True`
- Use DataParallel or DistributedDataParallel for multi-GPU

### Example: Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for ctp_data, prior_maps, labels in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(ctp_data, prior_maps)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{ctp_classification_network,
  title={CTP Classification Network},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/CTP}
}
```

## License

MIT License - feel free to use this code for research and commercial purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
