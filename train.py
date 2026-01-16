#!/usr/bin/env python3
"""
CTP分类模型训练脚本

CSV格式要求（4列）:
ctp_path,features_dir,time_points,label
/path/to/ctp_001.nii.gz,/path/to/features_001/,21,0
/path/to/ctp_002.nii.gz,/path/to/features_002/,20,1
...

说明：
- ctp_path: CTP数据文件路径 (.nii.gz)
- features_dir: 特征图所在目录（包含5个.nii.gz文件）
- time_points: 时间点数量（20或21）
- label: 分类标签（0, 1, 2, ...）

features_dir中应包含:
- generated_cbf.nii.gz
- generated_cbv.nii.gz
- generated_mtt.nii.gz
- generated_tmax.nii.gz
- generated_ttp.nii.gz
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    import nibabel as nib
except ImportError:
    print("错误: 请安装 nibabel 库")
    print("运行: pip install nibabel")
    exit(1)

try:
    from scipy.ndimage import zoom
except ImportError:
    print("错误: 请安装 scipy 库")
    print("运行: pip install scipy")
    exit(1)

from ctp_classification_model import CTPClassificationNet


# ==================== Dataset类 ====================
class CTPDataset(Dataset):
    """CTP数据集类，支持混合T=20和T=21的数据"""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file: CSV文件路径，包含ctp_path, features_dir, time_points, label列
            transform: 可选的数据增强函数
        """
        self.data_df = pd.read_csv(csv_file)
        self.transform = transform

        # 验证CSV格式
        required_columns = ['ctp_path', 'features_dir', 'time_points', 'label']
        for col in required_columns:
            if col not in self.data_df.columns:
                raise ValueError(f"CSV文件缺少必需列: {col}")

        # 验证time_points只包含20或21
        valid_time_points = self.data_df['time_points'].isin([20, 21])
        if not valid_time_points.all():
            invalid_values = self.data_df.loc[~valid_time_points, 'time_points'].unique()
            raise ValueError(f"time_points列包含无效值: {invalid_values}. 只允许20或21")

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
        ctp_data = self._load_ctp(row['ctp_path'], expected_time_points=time_points)

        # 加载5个特征图
        prior_maps = self._load_prior_maps(row['features_dir'])

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


def collate_fn_mixed_timepoints(batch):
    """
    自定义collate函数，处理混合时间点的batch
    将batch拆分为T=20和T=21的子批次
    """
    # 按时间点分组
    t20_items = []
    t21_items = []

    for ctp_data, prior_maps, label, time_points in batch:
        if time_points == 20:
            t20_items.append((ctp_data, prior_maps, label))
        else:
            t21_items.append((ctp_data, prior_maps, label))

    # 构建批次列表
    batches = []

    if t20_items:
        ctp_batch = torch.stack([item[0] for item in t20_items])
        prior_batch = torch.stack([item[1] for item in t20_items])
        label_batch = torch.stack([item[2] for item in t20_items])
        batches.append((ctp_batch, prior_batch, label_batch, 20))

    if t21_items:
        ctp_batch = torch.stack([item[0] for item in t21_items])
        prior_batch = torch.stack([item[1] for item in t21_items])
        label_batch = torch.stack([item[2] for item in t21_items])
        batches.append((ctp_batch, prior_batch, label_batch, 21))

    return batches


# ==================== 训练函数 ====================
def train_epoch(model_t20, model_t21, dataloader, criterion, optimizer_t20, optimizer_t21, device, epoch):
    """训练一个epoch，支持混合时间点"""
    if model_t20 is not None:
        model_t20.train()
    if model_t21 is not None:
        model_t21.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch_list in dataloader:
        for ctp_data, prior_maps, labels, time_points in batch_list:
            # 选择对应的模型和优化器
            if time_points == 20:
                model = model_t20
                optimizer = optimizer_t20
            else:
                model = model_t21
                optimizer = optimizer_t21

            if model is None:
                continue

            # 移动到设备
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


def validate(model_t20, model_t21, dataloader, criterion, device):
    """验证模型，支持混合时间点"""
    if model_t20 is not None:
        model_t20.eval()
    if model_t21 is not None:
        model_t21.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    with torch.no_grad():
        for batch_list in dataloader:
            for ctp_data, prior_maps, labels, time_points in batch_list:
                # 选择对应的模型
                model = model_t20 if time_points == 20 else model_t21

                if model is None:
                    continue

                ctp_data = ctp_data.to(device)
                prior_maps = prior_maps.to(device)
                labels = labels.to(device)

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
    print("CTP分类模型训练（支持混合T=20/21）")
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
    has_t20 = 20 in time_point_counts
    has_t21 = 21 in time_point_counts

    print(f"\n数据集时间点统计:")
    print(f"  T=20: {time_point_counts.get(20, 0)} 个样本")
    print(f"  T=21: {time_point_counts.get(21, 0)} 个样本")

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
        collate_fn=collate_fn_mixed_timepoints,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_mixed_timepoints,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 创建模型（根据数据创建需要的模型）
    print(f"\n{'='*70}")
    print("创建模型...")
    print(f"{'='*70}")

    model_t20 = None
    model_t21 = None
    optimizer_t20 = None
    optimizer_t21 = None

    if has_t20:
        print("创建 T=20 模型...")
        model_t20 = CTPClassificationNet(
            num_time_points=20,
            num_classes=args.num_classes,
            resnet_type=args.resnet_type,
            pretrained=args.pretrained
        ).to(device)
        optimizer_t20 = optim.Adam(model_t20.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"  T=20 模型参数: {sum(p.numel() for p in model_t20.parameters()):,}")

    if has_t21:
        print("创建 T=21 模型...")
        model_t21 = CTPClassificationNet(
            num_time_points=21,
            num_classes=args.num_classes,
            resnet_type=args.resnet_type,
            pretrained=args.pretrained
        ).to(device)
        optimizer_t21 = optim.Adam(model_t21.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"  T=21 模型参数: {sum(p.numel() for p in model_t21.parameters()):,}")

    # 定义损失函数
    if args.class_weights:
        class_weights = torch.tensor(args.class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\n使用加权损失函数: {args.class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("\n使用标准交叉熵损失")

    # 学习率调度器
    schedulers = []
    if optimizer_t20:
        schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_t20, mode='min', factor=0.5, patience=5, verbose=True
        ))
    if optimizer_t21:
        schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_t21, mode='min', factor=0.5, patience=5, verbose=True
        ))

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
            model_t20, model_t21, train_loader, criterion,
            optimizer_t20, optimizer_t21, device, epoch
        )
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1, val_cm = validate(
            model_t20, model_t21, val_loader, criterion, device
        )
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"       Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"混淆矩阵:\n{val_cm}")

        # 学习率调度
        for scheduler in schedulers:
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

            # 保存模型
            if model_t20:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_t20.state_dict(),
                    'optimizer_state_dict': optimizer_t20.state_dict(),
                    'val_f1': val_f1,
                    'args': vars(args)
                }, output_dir / 'best_model_t20.pth')

            if model_t21:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_t21.state_dict(),
                    'optimizer_state_dict': optimizer_t21.state_dict(),
                    'val_f1': val_f1,
                    'args': vars(args)
                }, output_dir / 'best_model_t21.pth')

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
    parser = argparse.ArgumentParser(description='训练CTP分类模型（支持混合T=20/21）')

    # 数据参数
    parser.add_argument('--csv_file', type=str, required=True,
                        help='CSV文件路径，包含ctp_path, features_dir, time_points, label列')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例 (默认: 0.2)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='分类类别数 (默认: 2)')

    # 模型参数
    parser.add_argument('--resnet_type', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='ResNet类型 (默认: resnet50)')
    parser.add_argument('--pretrained', action='store_true',
                        help='使用预训练权重')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小 (默认: 4)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减 (默认: 1e-5)')
    parser.add_argument('--class_weights', type=float, nargs='+', default=None,
                        help='类别权重，例如: --class_weights 1.0 3.0')

    # 早停参数
    parser.add_argument('--early_stopping', action='store_true',
                        help='启用早停')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停patience (默认: 10)')

    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录 (默认: ./output)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader工作线程数 (默认: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')

    args = parser.parse_args()

    main(args)
