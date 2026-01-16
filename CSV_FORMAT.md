# CSV 数据格式说明

## 📋 CSV 文件格式

训练脚本需要一个CSV文件，包含**4列**：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `ctp_path` | 字符串 | CTP数据文件的完整路径 | `/data/patient_001/ctp.nii.gz` |
| `features_dir` | 字符串 | 特征图所在目录路径 | `/data/patient_001/features/` |
| `time_points` | 整数 | 时间点数量（20或21） | `21` |
| `label` | 整数 | 分类标签（0, 1, 2, ...） | `0` |

---

## 📁 目录结构要求

### CTP数据文件
- 格式: `.nii.gz`
- 维度: `(H, W, Z, T)` 其中 `T=20` 或 `T=21`
- 推荐尺寸: `(512, 512, 32, T)`

### 特征图目录
`features_dir` 目录中必须包含以下**5个文件**：

```
features_dir/
├── generated_cbf.nii.gz      # 脑血流量 (CBF)
├── generated_cbv.nii.gz      # 脑血容量 (CBV)
├── generated_mtt.nii.gz      # 平均通过时间 (MTT)
├── generated_tmax.nii.gz     # 达峰时间 (Tmax)
└── generated_ttp.nii.gz      # 峰值时间 (TTP)
```

**注意**: 文件名必须完全匹配，包括前缀 `generated_`

---

## 📝 CSV 示例

### 示例1: 混合T=20和T=21的数据

```csv
ctp_path,features_dir,time_points,label
/data/patient_001/ctp.nii.gz,/data/patient_001/features/,21,0
/data/patient_002/ctp.nii.gz,/data/patient_002/features/,21,1
/data/patient_003/ctp.nii.gz,/data/patient_003/features/,20,0
/data/patient_004/ctp.nii.gz,/data/patient_004/features/,21,1
/data/patient_005/ctp.nii.gz,/data/patient_005/features/,20,1
```

### 示例2: 仅T=21数据

```csv
ctp_path,features_dir,time_points,label
/data/stroke/case_001/ctp_scan.nii.gz,/data/stroke/case_001/maps/,21,0
/data/stroke/case_002/ctp_scan.nii.gz,/data/stroke/case_002/maps/,21,1
/data/stroke/case_003/ctp_scan.nii.gz,/data/stroke/case_003/maps/,21,0
```

### 示例3: 仅T=20数据

```csv
ctp_path,features_dir,time_points,label
/mnt/data/patient_a/ctp.nii.gz,/mnt/data/patient_a/perfusion/,20,0
/mnt/data/patient_b/ctp.nii.gz,/mnt/data/patient_b/perfusion/,20,1
/mnt/data/patient_c/ctp.nii.gz,/mnt/data/patient_c/perfusion/,20,0
```

---

## ✅ 创建CSV文件的Python脚本

```python
import pandas as pd
from pathlib import Path

# 方法1: 手动创建列表
data = {
    'ctp_path': [
        '/data/patient_001/ctp.nii.gz',
        '/data/patient_002/ctp.nii.gz',
        '/data/patient_003/ctp.nii.gz',
    ],
    'features_dir': [
        '/data/patient_001/features/',
        '/data/patient_002/features/',
        '/data/patient_003/features/',
    ],
    'time_points': [21, 21, 20],
    'label': [0, 1, 0]
}

df = pd.DataFrame(data)
df.to_csv('my_dataset.csv', index=False)
print(f"创建CSV文件: {len(df)} 个样本")
```

```python
# 方法2: 从目录自动扫描
import pandas as pd
from pathlib import Path

data_root = Path('/data/stroke_dataset')
rows = []

for patient_dir in data_root.iterdir():
    if not patient_dir.is_dir():
        continue

    ctp_file = patient_dir / 'ctp.nii.gz'
    features_dir = patient_dir / 'features'

    if ctp_file.exists() and features_dir.exists():
        # 从文件名或其他元数据推断时间点和标签
        time_points = 21  # 或从文件读取
        label = 0  # 或从标签文件读取

        rows.append({
            'ctp_path': str(ctp_file),
            'features_dir': str(features_dir) + '/',
            'time_points': time_points,
            'label': label
        })

df = pd.DataFrame(rows)
df.to_csv('auto_generated.csv', index=False)
print(f"自动生成CSV: {len(df)} 个样本")
```

---

## ⚠️ 常见错误

### 错误1: 缺少列
```
ValueError: CSV文件缺少必需列: time_points
```
**解决**: 确保CSV包含所有4列：`ctp_path, features_dir, time_points, label`

### 错误2: time_points值无效
```
ValueError: time_points列包含无效值: [19]. 只允许20或21
```
**解决**: `time_points` 列只能包含 `20` 或 `21`

### 错误3: 找不到特征图
```
FileNotFoundError: 特征图不存在: /data/patient_001/features/generated_cbf.nii.gz
```
**解决**:
- 检查 `features_dir` 路径是否正确
- 确保所有5个特征图文件存在
- 文件名必须完全匹配（包括`generated_`前缀）

### 错误4: CTP时间点不匹配
```
ValueError: CTP数据时间点不匹配。CSV中为21，但文件中为20
```
**解决**: 确保CSV中的 `time_points` 值与实际CTP文件的时间点数一致

---

## 📊 验证CSV文件

使用以下脚本验证您的CSV文件：

```python
import pandas as pd
from pathlib import Path

def validate_csv(csv_file):
    """验证CSV文件格式"""
    print(f"验证CSV文件: {csv_file}")
    print("=" * 60)

    # 读取CSV
    df = pd.read_csv(csv_file)

    # 检查列
    required_cols = ['ctp_path', 'features_dir', 'time_points', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ 缺少列: {missing_cols}")
        return False
    else:
        print(f"✓ 所有必需列存在")

    # 检查time_points
    invalid_tp = df[~df['time_points'].isin([20, 21])]
    if len(invalid_tp) > 0:
        print(f"❌ 发现无效time_points值:")
        print(invalid_tp[['ctp_path', 'time_points']])
        return False
    else:
        print(f"✓ time_points值有效 (20或21)")

    # 统计
    print(f"\n数据统计:")
    print(f"  总样本数: {len(df)}")
    print(f"  T=20样本: {(df['time_points'] == 20).sum()}")
    print(f"  T=21样本: {(df['time_points'] == 21).sum()}")
    print(f"  类别分布:\n{df['label'].value_counts()}")

    # 检查文件存在性（可选，较慢）
    print(f"\n检查文件存在性...")
    missing_files = []
    for idx, row in df.iterrows():
        if not Path(row['ctp_path']).exists():
            missing_files.append(row['ctp_path'])

    if missing_files:
        print(f"⚠️  发现 {len(missing_files)} 个CTP文件不存在")
        for f in missing_files[:5]:  # 只显示前5个
            print(f"  - {f}")
    else:
        print(f"✓ 所有CTP文件存在")

    print("=" * 60)
    print("验证完成!")
    return True

# 使用
validate_csv('my_dataset.csv')
```

---

## 🚀 使用CSV训练

创建好CSV文件后，使用以下命令训练：

```bash
# 基础训练
python train.py --csv_file my_dataset.csv --num_classes 2 --pretrained

# 完整配置
python train.py \
  --csv_file my_dataset.csv \
  --num_classes 2 \
  --resnet_type resnet50 \
  --pretrained \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.001 \
  --early_stopping \
  --output_dir ./output/experiment_01
```

---

## 📚 相关文档

- **训练指南**: 查看 `TRAINING_GUIDE.md`
- **使用指南**: 查看 `USAGE_GUIDE.md`
- **示例CSV**: 查看 `data_mixed_example.csv`
