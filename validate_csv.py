#!/usr/bin/env python3
"""
CSV文件验证脚本
用于验证训练数据CSV文件的格式和完整性
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

def validate_csv(csv_file, check_files=False):
    """
    验证CSV文件格式和内容

    Args:
        csv_file: CSV文件路径
        check_files: 是否检查文件存在性（可能较慢）
    """
    print("=" * 70)
    print(f"验证CSV文件: {csv_file}")
    print("=" * 70)

    # 检查CSV文件是否存在
    if not Path(csv_file).exists():
        print(f"❌ CSV文件不存在: {csv_file}")
        return False

    try:
        # 读取CSV
        df = pd.read_csv(csv_file)
        print(f"✓ 成功读取CSV文件 ({len(df)} 行)")

    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        return False

    # 检查必需列
    print("\n【检查列】")
    required_cols = ['label', 'nii_path', 'time_points', 'mask_path']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ 缺少必需列: {missing_cols}")
        print(f"   当前列: {list(df.columns)}")
        return False
    else:
        print(f"✓ 所有必需列存在: {required_cols}")

    # 检查time_points值
    print("\n【检查time_points】")
    # 检查是否为正整数
    valid_tp = (df['time_points'] > 0) & (df['time_points'] == df['time_points'].astype(int))
    invalid_tp = df[~valid_tp]
    if len(invalid_tp) > 0:
        print(f"❌ 发现 {len(invalid_tp)} 个无效time_points值 (必须为正整数):")
        print(invalid_tp[['nii_path', 'time_points']].head())
        return False
    else:
        unique_tps = sorted(df['time_points'].unique())
        print(f"✓ time_points值有效，包含时间点: {unique_tps}")

    # 检查空值
    print("\n【检查空值】")
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"⚠️  发现空值:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"   {col}: {count} 个空值")
        return False
    else:
        print(f"✓ 无空值")

    # 数据统计
    print("\n【数据统计】")
    print(f"总样本数: {len(df)}")
    print(f"\n时间点分布:")
    tp_counts = df['time_points'].value_counts().sort_index()
    for tp, count in tp_counts.items():
        print(f"  T={tp}: {count} 个样本 ({count/len(df)*100:.1f}%)")

    print(f"\n类别分布:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  Label={label}: {count} 个样本 ({count/len(df)*100:.1f}%)")

    # 检查类别平衡
    if len(label_counts) > 1:
        max_count = label_counts.max()
        min_count = label_counts.min()
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 3:
            print(f"\n⚠️  类别不平衡 (比例 {imbalance_ratio:.1f}:1)")
            print(f"   建议使用 --class_weights 参数")

    # 检查文件存在性
    if check_files:
        print("\n【检查文件存在性】")
        print("检查CTP文件...")

        missing_ctp = []
        for idx, row in df.iterrows():
            if not Path(row['nii_path']).exists():
                missing_ctp.append(row['nii_path'])

        if missing_ctp:
            print(f"❌ 发现 {len(missing_ctp)} 个CTP文件不存在:")
            for f in missing_ctp[:5]:  # 只显示前5个
                print(f"   - {f}")
            if len(missing_ctp) > 5:
                print(f"   ... 还有 {len(missing_ctp)-5} 个")
        else:
            print(f"✓ 所有CTP文件存在")

        print("\n检查特征图目录...")
        missing_features = []
        incomplete_features = []

        feature_names = [
            'generated_cbf.nii.gz',
            'generated_cbv.nii.gz',
            'generated_mtt.nii.gz',
            'generated_tmax.nii.gz',
            'generated_ttp.nii.gz'
        ]

        for idx, row in df.iterrows():
            features_dir = Path(row['mask_path'])

            if not features_dir.exists():
                missing_features.append(str(features_dir))
            else:
                # 检查是否包含所有5个特征图
                missing_files = []
                for fname in feature_names:
                    if not (features_dir / fname).exists():
                        missing_files.append(fname)

                if missing_files:
                    incomplete_features.append({
                        'dir': str(features_dir),
                        'missing': missing_files
                    })

        if missing_features:
            print(f"❌ 发现 {len(missing_features)} 个特征图目录不存在:")
            for d in missing_features[:5]:
                print(f"   - {d}")
            if len(missing_features) > 5:
                print(f"   ... 还有 {len(missing_features)-5} 个")

        if incomplete_features:
            print(f"❌ 发现 {len(incomplete_features)} 个目录缺少特征图文件:")
            for item in incomplete_features[:3]:
                print(f"   - {item['dir']}")
                print(f"     缺少: {', '.join(item['missing'])}")
            if len(incomplete_features) > 3:
                print(f"   ... 还有 {len(incomplete_features)-3} 个")

        if not missing_features and not incomplete_features:
            print(f"✓ 所有特征图目录完整")

        if missing_ctp or missing_features or incomplete_features:
            return False

    # 最终结果
    print("\n" + "=" * 70)
    print("✅ CSV文件验证通过！")
    print("=" * 70)
    print("\n可以使用以下命令开始训练:")
    print(f"\npython train.py --config config.yaml\n")
    print("（请先在config.yaml中设置csv_file和num_classes等参数）")

    return True


def main():
    parser = argparse.ArgumentParser(description='验证训练数据CSV文件')
    parser.add_argument('csv_file', type=str, help='CSV文件路径')
    parser.add_argument('--check-files', action='store_true',
                        help='检查文件是否存在（可能较慢）')

    args = parser.parse_args()

    success = validate_csv(args.csv_file, check_files=args.check_files)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
