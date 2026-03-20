#!/usr/bin/env python3
"""
CTP分类模型训练脚本（统一3D卷积模型）

使用YAML配置文件进行训练:
  python train.py --config config.yaml

CSV格式要求（3列，带表头）:
label,image_path,mask_path
0,/data/.../A110034307/images/Head Volume Perfusion_5.0 x 5.0_301.pth,/data/.../A110034307/mask/Head Volume Perfusion_5.0 x 5.0_301_mask.pth
1,/data/.../A110034308/images/Head Volume Perfusion_5.0 x 5.0_301.pth,/data/.../A110034308/mask/Head Volume Perfusion_5.0 x 5.0_301_mask.pth

说明：
- label:      分类标签（0, 1, 2, ...）
- image_path: CTP图像 .pth 文件路径，tensor shape (H, W, Z, T)
- mask_path:  Prior图 .pth 文件路径，tensor shape (C, H, W)

特性：
- ✅ 使用统一的3D卷积模型处理任意时间点
- ✅ 直接加载 .pth 格式预处理数据，无需额外转换
- ✅ 通过自适应池化自动处理不同时间点，无需多个模型
- ✅ 支持混合精度训练（AMP）减少显存占用
"""

import os
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import yaml
from ctp_classification_model import CTPClassificationNet


# ==================== 配置加载 ====================
def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# ==================== Dataset类 ====================
class CTPDataset(Dataset):
    """
    CTP数据集类，直接加载预处理好的张量文件，支持 .pth 和 .npy 两种格式。

    CSV格式:
        label,image_path,mask_path
        0,/data/.../images/xxx.pth,/data/.../mask/xxx_mask.pth
        0,/data/.../images/xxx.npy,/data/.../mask/xxx_mask.npy

    期望的张量维度:
        image_path -> (H, W, Z, T)  4D CTP体积，最后一维为时间轴
        mask_path  -> (C, H, W)     C通道的先验图（C通常为5）
    """

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file:  CSV文件路径，包含 label, image_path, mask_path 三列
            transform: 可选的数据增强函数，接收 (ctp, mask) 返回 (ctp, mask)
        """
        self.data_df = pd.read_csv(csv_file)
        self.transform = transform

        required_columns = ['label', 'image_path', 'mask_path']
        for col in required_columns:
            if col not in self.data_df.columns:
                raise ValueError(
                    f"CSV文件缺少必需列: '{col}'，当前列名: {list(self.data_df.columns)}"
                )

        print(f"加载数据集: {len(self.data_df)} 个样本")
        print(f"类别分布:\n{self.data_df['label'].value_counts()}")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        # 支持 .pth / .npy 两种格式，自动识别
        ctp_data  = self._load_tensor(row['image_path'])   # (H, W, Z, T)
        mask_data = self._load_tensor(row['mask_path'])    # (C, H, W)
        label     = torch.tensor(int(row['label']), dtype=torch.long)

        if self.transform:
            ctp_data, mask_data = self.transform(ctp_data, mask_data)

        return ctp_data, mask_data, label

    def _load_tensor(self, path: str) -> torch.Tensor:
        """
        加载 .pth 或 .npy 文件并返回 float32 张量。

        Args:
            path: .pth 或 .npy 文件路径

        Returns:
            torch.Tensor (float32)
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        suffix = p.suffix.lower()

        if suffix == '.npy':
            data = torch.from_numpy(np.load(str(p)))
        elif suffix == '.pth':
            data = torch.load(str(p), map_location='cpu', weights_only=False)
            if not isinstance(data, torch.Tensor):
                raise TypeError(
                    f"期望 torch.Tensor，得到 {type(data).__name__}。\n"
                    f"文件: {path}\n"
                    f"请确认 .pth 文件由 torch.save(tensor, path) 直接保存。"
                )
        else:
            raise ValueError(f"不支持的文件格式: '{suffix}'，仅支持 .pth / .npy。路径: {path}")

        return data.float()


def collate_fn_default(batch):
    """
    标准collate函数，直接堆叠batch数据。
    3D卷积 + AdaptivePooling 天然支持不同时间点，无需按T分组。
    """
    ctp_batch   = torch.stack([item[0] for item in batch])
    prior_batch = torch.stack([item[1] for item in batch])
    label_batch = torch.stack([item[2] for item in batch])
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
            with autocast(device_type='cuda'):
                outputs = model(ctp_data, prior_maps)
                loss = criterion(outputs, labels)

            # 反向传播（使用混合精度）
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(ctp_data, prior_maps)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                with autocast(device_type='cuda'):
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

    # 设置随机种子（确保完全可复现）
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True  # 固定输入尺寸时加速卷积

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

    # 创建优化器（AdamW正确实现权重衰减解耦，比Adam更优）
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    scaler = GradScaler(device='cuda') if use_amp else None

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
