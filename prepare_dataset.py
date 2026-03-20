#!/usr/bin/env python3
"""
将 NIfTI 格式数据转换为 .pth 格式，供 train.py 直接加载。

输出 CSV 格式（与 train.py CTPDataset 对应）:
    label, image_path, mask_path

mask .pth shape: (5, H, W) — 5 个灌注参数图，每个用 Z 轴投影降维
image .pth shape: (H, W, Z, T) — 4D CTP 体积，float16 存储
"""

import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- 固定根目录配置 ---
SOURCE_CSV  = "/data/truenas_B2/yyi/datapath/CTP_2_2/CTP_第一轮数据_truenas_36T_21_22.csv"
OUTPUT_CSV  = "/data/truenas_B2/yyi/datapath/CTP_2_2/CTP_第一轮数据_truenas_36T_21_22_pth.csv"
IMG_SRC_ROOT = Path("/data/truenas_36T/HeadStroke_Yan")
MSK_SRC_ROOT = Path("/data/truenas_B2/yyi/CTP_mask")
SAVE_ROOT    = Path("/data/truenas_B2/yyi/data/CTP_pth")

# 进程数：CTP 4D 数据非常耗内存，建议 cpu_count // 4
NUM_WORKERS = max(1, cpu_count() // 4)

# --- 灌注参数图：Z 轴投影方向 ---
# max: 高值代表异常（延迟/缺血），捕获最严重的层
# min: 低值代表异常（缺血核心），捕获最严重的层
FEATURE_PROJECTION = {
    'generated_cbf.nii.gz':  'min',   # CBF 越低 → 梗死核心
    'generated_cbv.nii.gz':  'min',   # CBV 越低 → 梗死核心
    'generated_mtt.nii.gz':  'max',   # MTT 越高 → 灌注延迟
    'generated_tmax.nii.gz': 'max',   # Tmax 越高 → 低灌注半暗带
    'generated_ttp.nii.gz':  'max',   # TTP 越高 → 灌注延迟
}
FEATURE_NAMES = list(FEATURE_PROJECTION.keys())


def torch_resize_4d(volume: np.ndarray, target_spatial_shape=(512, 512, 32)) -> torch.Tensor:
    """
    使用 PyTorch Trilinear 插值缩放 4D CTP 数据 (H, W, D, T)。
    返回 tensor shape: (H, W, D, T)
    """
    # (H, W, D, T) -> (T, 1, H, W, D) — 对所有时间点并行做 3D 空间缩放
    v = torch.from_numpy(volume).float().permute(3, 0, 1, 2).unsqueeze(1)
    v = torch.nn.functional.interpolate(v, size=target_spatial_shape,
                                        mode='trilinear', align_corners=False)
    return v.squeeze(1).permute(1, 2, 3, 0)   # -> (H, W, D, T)


def torch_resize_2d(image: np.ndarray, target_shape=(512, 512)) -> torch.Tensor:
    """
    使用 PyTorch Bilinear 插值缩放 2D 图像 (H, W)。
    返回 tensor shape: (H, W)
    """
    v = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    v = torch.nn.functional.interpolate(v, size=target_shape,
                                        mode='bilinear', align_corners=False)
    return v.squeeze(0).squeeze(0)


def project_to_2d(arr: np.ndarray, proj: str) -> np.ndarray:
    """
    将 3D 灌注图 (H, W, Z) 沿 Z 轴投影为 2D (H, W)。

    proj='max': 捕获每个 XY 位置上 Z 轴最大值（适合 Tmax/MTT/TTP）
    proj='min': 捕获每个 XY 位置上 Z 轴最小值（适合 CBF/CBV）
    """
    if proj == 'max':
        return arr.max(axis=2)
    else:
        return arr.min(axis=2)


def process_single_case(row: dict):
    try:
        orig_nii_path = row['nii_path']
        nii_path = Path(orig_nii_path)

        patient_id  = nii_path.parent.parent.name       # e.g. A110034307
        series_name = nii_path.name.replace('.nii.gz', '')  # e.g. Head Volume Perfusion_5.0 x 5.0_301

        p_dir          = SAVE_ROOT / patient_id
        img_target_dir = p_dir / "images"
        msk_target_dir = p_dir / "mask"
        img_target_dir.mkdir(parents=True, exist_ok=True)
        msk_target_dir.mkdir(parents=True, exist_ok=True)

        img_pth_path = img_target_dir / f"{series_name}.pth"
        msk_pth_path = msk_target_dir / f"{series_name}_mask.pth"

        # ── 1. 处理 CTP Image ──────────────────────────────────────────────
        if not img_pth_path.exists():
            img_data = nib.load(str(nii_path)).get_fdata(dtype=np.float32)

            if img_data.ndim != 4:
                raise ValueError(f"CTP 应为 4D，得到 {img_data.ndim}D: {nii_path}")

            if img_data.shape[:3] != (512, 512, 32):
                final_img = torch_resize_4d(img_data)   # (H, W, D, T)
            else:
                final_img = torch.from_numpy(img_data)  # (H, W, D, T)

            torch.save(final_img.to(torch.float16), img_pth_path)
            del img_data, final_img

        # ── 2. 处理 Mask（5 个灌注参数图 → 逐通道 Z 轴投影 → (5, H, W)）──
        if not msk_pth_path.exists():
            mask_src_dir = MSK_SRC_ROOT / patient_id / series_name
            if not mask_src_dir.exists():
                raise FileNotFoundError(f"Mask 目录不存在: {mask_src_dir}")

            prior_maps = []
            for f_name in FEATURE_NAMES:
                f_p = mask_src_dir / f_name
                if not f_p.exists():
                    # [Bug Fix 3] 严格要求 5 个文件都存在，否则整个样本跳过
                    raise FileNotFoundError(f"灌注图缺失: {f_p}")

                m_data = nib.load(str(f_p)).get_fdata(dtype=np.float32)

                # 降维到 2D
                if m_data.ndim == 4:
                    # [Bug Fix 2a] 4D: 先时间轴均值 → 3D
                    m_data = m_data.mean(axis=-1)
                if m_data.ndim == 3:
                    # [Bug Fix 2b] 3D: 用 max/min 投影代替取中间切片
                    m_data = project_to_2d(m_data, FEATURE_PROJECTION[f_name])
                if m_data.ndim != 2:
                    raise ValueError(f"投影后维度异常: {m_data.shape} ({f_name})")

                if m_data.shape != (512, 512):
                    m_tensor = torch_resize_2d(m_data)
                else:
                    m_tensor = torch.from_numpy(m_data)

                prior_maps.append(m_tensor)

            # [Bug Fix 3] 验证确实有 5 个通道再保存
            if len(prior_maps) != len(FEATURE_NAMES):
                raise RuntimeError(f"只处理了 {len(prior_maps)}/{len(FEATURE_NAMES)} 个灌注图")

            stacked = torch.stack(prior_maps, dim=0).to(torch.float16)  # (5, H, W)
            torch.save(stacked, msk_pth_path)
            del prior_maps, stacked

        # ── 3. 构建输出行 ──────────────────────────────────────────────────
        updated_row = row.copy()
        updated_row['original_nii_path'] = str(orig_nii_path)
        # [Bug Fix 1] 列名改为 image_path，与 train.py CTPDataset 对齐
        updated_row['image_path'] = str(img_pth_path.absolute())
        updated_row['mask_path']  = str(msk_pth_path.absolute())
        # 删除原 nii_path（避免与 image_path 混淆，训练代码不需要它）
        updated_row.pop('nii_path', None)

        return updated_row

    except Exception as e:
        # [Bug Fix 4] 打印错误信息，方便 debug
        print(f"\n[SKIP] {row.get('nii_path', '?')} — {type(e).__name__}: {e}")
        return None


def main():
    print(f"检测到核心数: {cpu_count()}，当前配置进程数: {NUM_WORKERS}")

    df = pd.read_csv(SOURCE_CSV)

    # 断点续跑：跳过已处理的样本
    if Path(OUTPUT_CSV).exists():
        existing_df    = pd.read_csv(OUTPUT_CSV)
        processed_set  = set(existing_df['original_nii_path'].astype(str))
        pending_df     = df[~df['nii_path'].astype(str).isin(processed_set)]
        final_results  = existing_df.to_dict('records')
    else:
        pending_df    = df
        final_results = []

    if len(pending_df) == 0:
        print("所有数据已处理完毕！")
        return

    print(f"剩余待处理任务: {len(pending_df)}")
    tasks = pending_df.to_dict('records')

    with Pool(processes=NUM_WORKERS) as pool:
        pbar = tqdm(pool.imap_unordered(process_single_case, tasks), total=len(tasks))
        for result in pbar:
            if result is not None:
                final_results.append(result)

            # 每处理 50 个就保存一次（断点续跑保障）
            if len(final_results) % 50 == 0 and final_results:
                pd.DataFrame(final_results).to_csv(OUTPUT_CSV, index=False)

    pd.DataFrame(final_results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n转换完成！结果存至: {OUTPUT_CSV}")
    print(f"成功: {len(final_results)} / {len(df)} 个样本")


if __name__ == "__main__":
    main()
