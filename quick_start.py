#!/usr/bin/env python3
"""
快速开始 - CTP分类网络最简使用示例
运行: python quick_start.py
"""

import torch
from ctp_classification_model import create_ctp_classifier

def main():
    print("=" * 60)
    print("CTP 分类网络 - 快速开始示例")
    print("=" * 60)

    # ========== 步骤1: 创建模型 ==========
    print("\n[步骤1] 创建模型...")
    model = create_ctp_classifier(
        num_time_points=21,      # CTP时间点数量 (20或21)
        num_classes=2,           # 二分类: 中风 vs 正常
        resnet_type='resnet50',  # ResNet-50 (平衡性能和速度)
        pretrained=True          # 使用ImageNet预训练权重
    )
    print("✓ 模型创建成功")

    # ========== 步骤2: 移动到设备 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✓ 模型已移动到: {device}")

    # ========== 步骤3: 准备数据 ==========
    print("\n[步骤2] 准备输入数据...")
    batch_size = 2  # 一次处理2个样本

    # CTP 4D灰度医学图像数据
    # 形状: (batch_size, height, width, slices, time_points)
    ctp_data = torch.randn(batch_size, 512, 512, 32, 21).to(device)

    # 灌注先验图 (5个通道)
    # 通道顺序: [CBF, CBV, MTT, Tmax, TTP]
    prior_maps = torch.randn(batch_size, 5, 512, 512).to(device)

    print(f"✓ CTP数据形状: {ctp_data.shape}")
    print(f"✓ 先验图形状: {prior_maps.shape}")

    # ========== 步骤4: 模型推理 ==========
    print("\n[步骤3] 进行预测...")
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 不计算梯度
        # 前向传播
        logits = model(ctp_data, prior_maps)

        # 计算概率
        probabilities = torch.softmax(logits, dim=1)

        # 获取预测类别
        predictions = torch.argmax(probabilities, dim=1)

    print("✓ 预测完成")

    # ========== 步骤5: 查看结果 ==========
    print("\n[步骤4] 预测结果:")
    print("-" * 60)
    for i in range(batch_size):
        pred_class = predictions[i].item()
        confidence = probabilities[i, pred_class].item()
        class_name = "中风" if pred_class == 1 else "正常"

        print(f"样本 {i+1}:")
        print(f"  预测类别: {pred_class} ({class_name})")
        print(f"  置信度: {confidence:.2%}")
        print(f"  完整概率: 正常={probabilities[i, 0]:.2%}, "
              f"中风={probabilities[i, 1]:.2%}")
        print()

    # ========== 额外信息 ==========
    print("=" * 60)
    print("模型信息:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print("=" * 60)

    print("\n✅ 完成! 您已成功运行CTP分类网络")
    print("\n下一步:")
    print("  1. 准备您自己的CTP数据和先验图")
    print("  2. 查看 USAGE_GUIDE.md 了解如何训练模型")
    print("  3. 运行 python example_usage.py 查看更多示例")
    print("  4. 运行 python test_model.py 进行完整测试")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保已安装依赖: pip install -r requirements.txt")
