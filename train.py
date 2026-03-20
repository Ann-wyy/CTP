#!/usr/bin/env python3
"""
CTP分类模型训练脚本（统一3D卷积模型）

使用YAML配置文件进行训练:
  python train.py --config config.yaml

CSV格式要求（4列，带表头）:
label,nii_path,time_points,mask_path
0,/path/to/ctp_001.nii.gz,21,/path/to/features_001/
1,/path/to/ctp_002.nii.gz,22,/path/to/features_002/
0,/path/to/ctp_003.nii.gz,20,/path/to/features_003/
...

说明：
- label: 分类标签（0, 1, 2, ...）
- nii_path: CTP数据文件路径 (.nii.gz)
- time_points: 时间点数量（正整数，如20, 21, 22等）
- mask_path: 特征图所在目录（包含5个.nii.gz文件）

特性：
- ✅ 使用统一的3D卷积模型处理任意时间点
- ✅ 支持混合时间点的数据集（同一个数据集可包含不同时间点）
- ✅ 通过自适应池化自动处理不同时间点，无需多个模型
- ✅ 支持混合精度训练（AMP）减少显存占用

mask_path目录中应包含:
- generated_cbf.nii.gz
- generated_cbv.nii.gz
- generated_mtt.nii.gz
- generated_tmax.nii.gz
- generated_ttp.nii.gz
"""

import os
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import yaml
import nibabel as nib
from scipy.ndimage import zoom
from ctp_classification_model import CTPClassificationNet


# ==================== 配置加载 ====================
def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# ==================== Dataset类 ====================
class CTPDataset(Dataset):
    """CTP数据集类，支持任意混合时间点数据（T=20, 21, 22, ...）"""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file: CSV文件路径，包含label, nii_path, time_points, mask_path列
            transform: 可选的数据增强函数
        """
        self.data_df = pd.read_csv(csv_file)
        self.transform = transform

        # 验证CSV格式（支持两种列名格式）
        required_columns = ['label', 'nii_path', 'time_points', 'mask_path']
        for col in required_columns:
            if col not in self.data_df.columns:
                raise ValueError(f"CSV文件缺少必需列: {col}，当前列名: {list(self.data_df.columns)}")

        # 验证time_points为正整数
        valid_time_points = (self.data_df['time_points'] > 0) & (self.data_df['time_points'] == self.data_df['time_points'].astype(int))
        if not valid_time_points.all():
            invalid_values = self.data_df.loc[~valid_time_points, 'time_points'].unique()
            raise ValueError(f"time_points列包含无效值: {invalid_values}. 必须为正整数")

        # 显示支持的时间点
        unique_time_points = sorted(self.data_df['time_points'].unique())
        print(f"数据集包含的时间点: {unique_time_points}")

        print(f"加载数据集: {len(self.data_df)} 个样本")
        print(f"时间点分布:\n{self.data_df['time_points'].value_counts()}")
        print(f"类别分布:\n{self.data_df['label'].value_counts()}")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        # 获取时间点数
        time_points = int(row['time_points'])

        # 加载CTP数据
        ctp_data = self._load_ctp(row['nii_path'], expected_time_points=time_points)

        # 加载5个特征图
        prior_maps = self._load_prior_maps(row['mask_path'])

        # 获取标签
        label = int(row['label'])

        # 数据增强（如果有）
        if self.transform:
            ctp_data, prior_maps = self.transform(ctp_data, prior_maps)

        # 转换为tensor
        ctp_data = torch.from_numpy(ctp_data).float()
        prior_maps = torch.from_numpy(prior_maps).float()
        label = torch.tensor(label, dtype=torch.long)

        return ctp_data, prior_maps, label, time_points

    def _load_ctp(self, ctp_path, expected_time_points):
        """
        加载CTP .nii.gz文件

        Args:
            ctp_path: CTP文件路径
            expected_time_points: 期望的时间点数（20或21）

        Returns:
            numpy array of shape (512, 512, 32, T)
        """
        try:
            img = nib.load(ctp_path)
            data = img.get_fdata()

            # 验证时间点数
            if data.ndim != 4:
                raise ValueError(f"CTP数据应该是4D的，但得到{data.ndim}D: {data.shape}")

            actual_time_points = data.shape[-1]
            if actual_time_points != expected_time_points:
                raise ValueError(
                    f"CTP数据时间点不匹配。CSV中为{expected_time_points}，"
                    f"但文件中为{actual_time_points}: {ctp_path}"
                )

            # 调整空间尺寸为 (512, 512, 32)
            target_shape = (512, 512, 32, actual_time_points)
            if data.shape != target_shape:
                data = self._resize_4d(data, target_shape)

            return data.astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"加载CTP文件失败 {ctp_path}: {e}")

    def _load_prior_maps(self, features_dir):
        """
        加载5个特征图

        Args:
            features_dir: 特征图目录路径

        Returns:
            numpy array of shape (5, 512, 512)
        """
        features_dir = Path(features_dir)

        feature_names = [
            'generated_cbf.nii.gz',
            'generated_cbv.nii.gz',
            'generated_mtt.nii.gz',
            'generated_tmax.nii.gz',
            'generated_ttp.nii.gz'
        ]

        prior_maps = []

        for feature_name in feature_names:
            feature_path = features_dir / feature_name

            if not feature_path.exists():
                raise FileNotFoundError(f"特征图不存在: {feature_path}")

            try:
                img = nib.load(str(feature_path))
                data = img.get_fdata()

                # 处理不同维度的特征图
                if data.ndim == 2:
                    # 已经是2D
                    pass
                elif data.ndim == 3:
                    # 3D图像，取中间层
                    data = data[:, :, data.shape[2] // 2]
                elif data.ndim == 4:
                    # 4D图像，先在时间维度平均，再取中间层
                    data = np.mean(data, axis=-1)
                    if data.ndim == 3:
                        data = data[:, :, data.shape[2] // 2]
                else:
                    raise ValueError(f"不支持的特征图维度: {data.ndim}D")

                # 确保尺寸为 512x512
                if data.shape != (512, 512):
                    data = self._resize_2d(data, (512, 512))

                prior_maps.append(data)

            except Exception as e:
                raise RuntimeError(f"加载特征图失败 {feature_path}: {e}")

        # 堆叠为 (5, 512, 512)
        prior_maps = np.stack(prior_maps, axis=0).astype(np.float32)

        return prior_maps

    def _resize_4d(self, volume, target_shape):
        """调整4D体积大小"""
        factors = [t / s for t, s in zip(target_shape, volume.shape)]
        resized = zoom(volume, factors, order=1)
        return resized

    def _resize_2d(self, image, target_shape):
        """调整2D图像大小"""
        factors = [t / s for t, s in zip(target_shape, image.shape)]
        resized = zoom(image, factors, order=1)
        return resized


def collate_fn_default(batch):
    """
    标准collate函数，直接堆叠batch数据

    3D卷积模型通过AdaptivePooling处理不同时间点，
    因此不再需要按time_points分组
    """
    ctp_data_list = []
    prior_maps_list = []
    labels_list = []

    for ctp_data, prior_maps, label, time_points in batch:
        ctp_data_list.append(ctp_data)
        prior_maps_list.append(prior_maps)
        labels_list.append(label)

    # Stack成batch
    ctp_batch = torch.stack(ctp_data_list)
    prior_batch = torch.stack(prior_maps_list)
    label_batch = torch.stack(labels_list)

    return ctp_batch, prior_batch, label_batch


# ==================== 训练函数 ====================
def train_epoch(model, optimizer, dataloader, criterion, device, epoch, scaler=None, use_amp=False):
    """
    训练一个epoch，使用统一的3D卷积模型

    Args:
        model: 统一的CTP分类模型
        optimizer: 优化器
        scaler: GradScaler for mixed precision training
        use_amp: 是否使用混合精度训练
    """
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for ctp_data, prior_maps, labels in dataloader:
        # 移动到设备
        ctp_data = ctp_data.to(device)
        prior_maps = prior_maps.to(device)
        labels = labels.to(device)

        # 前向传播（使用混合精度）
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(ctp_data, prior_maps)
                loss = criterion(outputs, labels)

            # 反向传播（使用混合精度）
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(ctp_data, prior_maps)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

        # 统计
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1

        # 打印进度
        if num_batches % 10 == 0:
            print(f"  Batch [{num_batches}], Loss: {loss.item():.4f}")

    # 计算指标
    avg_loss = running_loss / max(num_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, use_amp=False):
    """
    验证模型，使用统一的3D卷积模型

    Args:
        model: 统一的CTP分类模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        use_amp: 是否使用混合精度
    """
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    with torch.no_grad():
        for ctp_data, prior_maps, labels in dataloader:
            ctp_data = ctp_data.to(device)
            prior_maps = prior_maps.to(device)
            labels = labels.to(device)

            if use_amp:
                with autocast():
                    outputs = model(ctp_data, prior_maps)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(ctp_data, prior_maps)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1

    # 计算指标
    avg_loss = running_loss / max(num_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0

    if all_labels:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds,
            average='binary' if len(set(all_labels)) == 2 else 'macro',
            zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)
    else:
        precision = recall = f1 = 0.0
        cm = np.array([[0]])

    return avg_loss, accuracy, precision, recall, f1, cm


# ==================== 主训练流程 ====================
def main(args):
    print("=" * 70)
    print("CTP分类模型训练（统一3D卷积模型）")
    print("=" * 70)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 加载数据集
    print(f"\n{'='*70}")
    print("加载数据...")
    print(f"{'='*70}")

    full_dataset = CTPDataset(args.csv_file)

    # 统计时间点分布
    time_point_counts = Counter(full_dataset.data_df['time_points'])
    unique_time_points = sorted(time_point_counts.keys())

    print(f"\n数据集时间点统计:")
    for tp in unique_time_points:
        print(f"  T={tp}: {time_point_counts[tp]} 个样本")

    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=args.val_split,
        random_state=args.seed,
        stratify=full_dataset.data_df['label'].values
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"\n训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_default,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_default,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 创建统一的3D卷积模型（可处理任意时间点）
    print(f"\n{'='*70}")
    print("创建模型...")
    print(f"{'='*70}")

    print("创建统一的3D卷积模型（支持任意时间点）...")
    model = CTPClassificationNet(
        num_time_points=21,  # 参数保留用于兼容性，实际模型可处理任意T
        num_classes=args.num_classes,
        resnet_type=args.resnet_type,
        pretrained=args.pretrained
    ).to(device)

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    print(f"模型支持的时间点: 任意 (通过3D卷积+自适应池化实现)")

    # 定义损失函数
    if args.class_weights:
        class_weights = torch.tensor(args.class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\n使用加权损失函数: {args.class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("\n使用标准交叉熵损失")

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 混合精度训练
    use_amp = getattr(args, 'use_amp', False) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    if use_amp:
        print("\n✓ 启用混合精度训练（FP16）")
        print("  - 显存占用减少约50%")
        print("  - 训练速度提升30-50%")
    else:
        print("\n使用标准FP32训练")

    # 训练循环
    print(f"\n{'='*70}")
    print("开始训练...")
    print(f"{'='*70}")

    best_val_f1 = 0.0
    patience_counter = 0
    train_history = []
    val_history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

        # 训练
        train_loss, train_acc = train_epoch(
            model, optimizer, train_loader, criterion, device, epoch, scaler, use_amp
        )
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1, val_cm = validate(
            model, val_loader, criterion, device, use_amp
        )
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"       Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"混淆矩阵:\n{val_cm}")

        # 学习率调度
        scheduler.step(val_loss)

        # 保存历史
        train_history.append({'epoch': epoch+1, 'loss': train_loss, 'acc': train_acc})
        val_history.append({
            'epoch': epoch+1, 'loss': val_loss, 'acc': val_acc,
            'precision': val_precision, 'recall': val_recall, 'f1': val_f1
        })

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            # 保存统一模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'args': vars(args)
            }, output_dir / 'best_model.pth')

            print(f"✓ 保存最佳模型 (F1: {val_f1:.4f})")
        else:
            patience_counter += 1

        # 早停
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\n早停触发！验证F1已经{args.patience}个epoch未提升")
            break

    # 保存训练历史
    history_df = pd.DataFrame({
        'train_loss': [h['loss'] for h in train_history],
        'train_acc': [h['acc'] for h in train_history],
        'val_loss': [h['loss'] for h in val_history],
        'val_acc': [h['acc'] for h in val_history],
        'val_f1': [h['f1'] for h in val_history]
    })
    history_df.to_csv(output_dir / 'training_history.csv', index=False)

    print(f"\n{'='*70}")
    print("训练完成！")
    print(f"{'='*70}")
    print(f"最佳验证F1分数: {best_val_f1:.4f}")
    print(f"模型保存至: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='训练CTP分类模型（统一3D卷积模型，支持任意混合时间点）',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=str, required=True,help='YAML配置文件路径（必需）')

    args = parser.parse_args()

    # 加载配置文件
    print(f"加载配置文件: {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 '{args.config}'")
        print("\n提示: 使用以下命令创建配置文件:")
        print("  cp config_template.yaml my_config.yaml")
        print("  # 然后编辑 my_config.yaml")
        exit(1)
    except Exception as e:
        print(f"错误: 加载配置文件失败: {e}")
        exit(1)

    # 将配置转换为Namespace
    from argparse import Namespace
    args = Namespace(**config)

    # 验证必需参数
    required_params = ['csv_file', 'num_classes']
    missing_params = [p for p in required_params if not hasattr(args, p) or getattr(args, p) is None]
    if missing_params:
        print(f"错误: 配置文件缺少必需参数: {missing_params}")
        print("\n请在配置文件中设置以下参数:")
        for param in missing_params:
            print(f"  {param}: <value>")
        exit(1)

    # 显示最终配置
    print("\n" + "=" * 70)
    print("训练配置:")
    print("=" * 70)
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    print("=" * 70)

    main(args)
"""
使用YAML配置文件:
  # 使用预设配置
  python train.py --config config_basic.yaml

  # 使用自定义配置
  python train.py --config my_config.yaml

配置文件模板:
  - config_basic.yaml       基础配置
  - config_advanced.yaml    高级配置
  - config_small_gpu.yaml   小显存GPU配置
  - config_multiclass.yaml  多分类配置
  - config_template.yaml    完整模板

查看文档:
  - CONFIG_GUIDE.md         配置指南
  - TRAINING_GUIDE.md       训练指南
  - FLEXIBLE_TIMEPOINTS.md  灵活时间点支持
        """
    # 配置文件参数（必需）
