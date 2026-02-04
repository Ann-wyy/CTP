#!/usr/bin/env python3
"""
快速检查CSV中time_points列的脚本
"""
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("用法: python check_timepoints.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

print(f"检查文件: {csv_file}")
print("=" * 70)

# 读取CSV
df = pd.read_csv(csv_file)

print(f"\n总样本数: {len(df)}")
print(f"CSV列名: {list(df.columns)}")

# 检查time_points列
if 'time_points' in df.columns:
    print("\n【time_points列统计】")
    print(f"数据类型: {df['time_points'].dtype}")
    print(f"\n唯一值: {sorted(df['time_points'].unique())}")
    print(f"\n值分布:")
    print(df['time_points'].value_counts().sort_index())

    # 检查异常值
    print("\n【异常值检查】")

    # 非整数值
    non_int = df[df['time_points'] != df['time_points'].astype(int)]
    if len(non_int) > 0:
        print(f"⚠️ 发现 {len(non_int)} 个非整数值:")
        print(non_int[['nii_path', 'time_points']].head(10))

    # 小于15的异常值（CTP通常>=20）
    abnormal = df[df['time_points'] < 15]
    if len(abnormal) > 0:
        print(f"\n⚠️ 发现 {len(abnormal)} 个异常小的time_points值 (<15):")
        print(abnormal[['nii_path', 'time_points']])

    # 显示前20行的time_points
    print("\n【前20行time_points值】")
    print(df[['nii_path', 'time_points']].head(20))

else:
    print("❌ 没有找到time_points列！")
    print(f"可用列: {list(df.columns)}")
