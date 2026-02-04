#!/usr/bin/env python3
"""
修复CSV文件中的time_points值
从实际的.nii.gz文件中读取正确的时间点数量
"""
import pandas as pd
import nibabel as nib
from pathlib import Path
import sys
from tqdm import tqdm

if len(sys.argv) < 3:
    print("用法: python fix_timepoints.py <input_csv> <output_csv>")
    print("例如: python fix_timepoints.py data.csv data_fixed.csv")
    sys.exit(1)

input_csv = sys.argv[1]
output_csv = sys.argv[2]

print(f"读取CSV: {input_csv}")
df = pd.read_csv(input_csv)

print(f"总样本数: {len(df)}")
print(f"\n修复前time_points分布:")
print(df['time_points'].value_counts().sort_index())

# 修复每一行
fixed_count = 0
error_count = 0
errors = []

print(f"\n开始修复...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    nii_path = row['nii_path']
    csv_time_points = row['time_points']

    try:
        if Path(nii_path).exists():
            # 读取实际时间点
            img = nib.load(nii_path)
            data = img.get_fdata()

            if data.ndim == 4:
                actual_time_points = data.shape[-1]

                if actual_time_points != csv_time_points:
                    df.at[idx, 'time_points'] = actual_time_points
                    fixed_count += 1
                    print(f"\n修复行 {idx}: {nii_path}")
                    print(f"  CSV值: {csv_time_points} → 实际值: {actual_time_points}")
            else:
                error_count += 1
                errors.append((idx, nii_path, f"数据不是4D: {data.shape}"))
        else:
            error_count += 1
            errors.append((idx, nii_path, "文件不存在"))

    except Exception as e:
        error_count += 1
        errors.append((idx, nii_path, str(e)))

print(f"\n{'='*70}")
print(f"修复完成！")
print(f"  - 修复了 {fixed_count} 个错误的time_points值")
print(f"  - {error_count} 个文件处理失败")

if errors:
    print(f"\n错误详情:")
    for idx, path, error in errors[:10]:
        print(f"  行{idx}: {path}")
        print(f"    错误: {error}")
    if len(errors) > 10:
        print(f"  ... 还有 {len(errors)-10} 个错误")

print(f"\n修复后time_points分布:")
print(df['time_points'].value_counts().sort_index())

# 保存修复后的CSV
df.to_csv(output_csv, index=False)
print(f"\n修复后的CSV已保存到: {output_csv}")
print(f"\n请使用修复后的CSV文件重新训练:")
print(f"  1. 在config.yaml中设置: csv_file: {output_csv}")
print(f"  2. 运行: python train.py --config config.yaml")
