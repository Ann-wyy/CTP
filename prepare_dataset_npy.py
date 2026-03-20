#!/usr/bin/env python3
"""
将 NIfTI 格式数据转换为 .npy 格式，供 train.py 直接加载。

输出 CSV 格式（与 train.py CTPDataset 对应）:
    label, image_path, mask_path          ← 列名必须与此一致

image .npy shape : (H, W, Z, T)  float16
mask  .npy shape : (5, H, W)     float16  — 5 个灌注参数图 Z 轴投影
"""

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed

# ======================
# 配置
# ======================
SOURCE_CSV   = "/data/truenas_B2/yyi/datapath/CTP_2_2/CTP_第一轮数据_truenas_36T_21_22.csv"
OUTPUT_CSV   = "/data/truenas_B2/yyi/datapath/CTP_2_2/CTP_第一轮数据_truenas_36T_21_22_npy.csv"
MSK_SRC_ROOT = Path("/data/truenas_B2/yyi/CTP_mask")
SAVE_ROOT    = Path("/data/truenas_B2/yyi/data/CTP_npy")
NUM_WORKERS  = min(8, max(1, cpu_count() // 4))

# 灌注参数图 Z 轴投影方向（同 prepare_dataset.py）
FEATURE_CONFIG = {
    'generated_cbf.nii.gz':  'min',
    'generated_cbv.nii.gz':  'min',
    'generated_mtt.nii.gz':  'max',
    'generated_tmax.nii.gz': 'max',
    'generated_ttp.nii.gz':  'max',
}

# ======================
# Resize 函数
# ======================
def torch_resize_4d(volume: np.ndarray, target_shape=(512, 512, 32)) -> np.ndarray:
    if volume.shape[:3] == target_shape:
        return volume
    v = torch.from_numpy(volume).float().permute(3, 0, 1, 2).unsqueeze(1)
    v = torch.nn.functional.interpolate(v, size=target_shape,
                                        mode='trilinear', align_corners=False)
    return v.squeeze(1).permute(1, 2, 3, 0).numpy()


def torch_resize_2d(image: np.ndarray, target_shape=(512, 512)) -> np.ndarray:
    if image.shape == target_shape:
        return image
    v = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    v = torch.nn.functional.interpolate(v, size=target_shape,
                                        mode='bilinear', align_corners=False)
    return v.squeeze(0).squeeze(0).numpy()

# ======================
# 读取 NIfTI（mmap 减少内存复制）
# ======================
def load_nii(path) -> np.ndarray:
    return np.asarray(nib.load(str(path), mmap=True).dataobj, dtype=np.float32)

# ======================
# Mask 单通道投影
# ======================
def process_mask_channel(f_path: Path, proj_type: str) -> np.ndarray:
    data = load_nii(f_path)
    if data.ndim == 4:
        data = data.mean(axis=-1)           # 4D → 3D (时间均值)
    if data.ndim == 3:                      # 3D → 2D (Z 轴投影)
        data = data.min(axis=2) if proj_type == 'min' else data.max(axis=2)
    if data.shape != (512, 512):
        data = torch_resize_2d(data)
    return np.ascontiguousarray(data, dtype=np.float16)

# ======================
# 断点续跑检测
# ======================
def _series_name(nii_path: Path) -> str:
    # [Bug Fix 2] .stem 对 .nii.gz 只去掉 .gz，须用 replace
    return nii_path.name.replace('.nii.gz', '')


def is_done(row) -> bool:
    p = Path(row['nii_path'])
    if not p.exists():
        return True
    patient_id  = p.parent.parent.name
    series_name = _series_name(p)
    img_npy = SAVE_ROOT / patient_id / "images" / f"{series_name}.npy"
    msk_npy = SAVE_ROOT / patient_id / "mask"   / f"{series_name}_mask.npy"
    return img_npy.exists() and msk_npy.exists()

# ======================
# 单样本处理
# ======================
def process_case(row: dict):
    try:
        orig_nii = Path(row['nii_path'])
        if not orig_nii.exists():
            return None

        patient_id  = orig_nii.parent.parent.name
        # [Bug Fix 2] .stem → replace，避免 "xxx.nii.npy" 的错误文件名
        series_name = _series_name(orig_nii)

        img_dir = SAVE_ROOT / patient_id / "images"
        msk_dir = SAVE_ROOT / patient_id / "mask"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        img_npy_path = img_dir / f"{series_name}.npy"
        msk_npy_path = msk_dir / f"{series_name}_mask.npy"

        # ── CTP Image ──────────────────────────────────────────────────────
        if not img_npy_path.exists():
            img_data = load_nii(orig_nii)
            if img_data.ndim != 4:
                raise ValueError(f"CTP 应为 4D，得到 {img_data.ndim}D")
            img_resized = torch_resize_4d(img_data)   # (H, W, Z, T)
            np.save(img_npy_path, np.ascontiguousarray(img_resized, dtype=np.float16))
            del img_data, img_resized

        # ── Mask ───────────────────────────────────────────────────────────
        if not msk_npy_path.exists():
            mask_src = MSK_SRC_ROOT / patient_id / series_name
            if not mask_src.exists():
                raise FileNotFoundError(f"Mask 目录不存在: {mask_src}")

            # 5 通道并行处理
            prior_maps = Parallel(n_jobs=5)(
                delayed(process_mask_channel)(mask_src / fname, proj)
                for fname, proj in FEATURE_CONFIG.items()
                if (mask_src / fname).exists()
            )

            if len(prior_maps) != 5:
                raise FileNotFoundError(
                    f"灌注图不完整：只找到 {len(prior_maps)}/5 个文件，目录: {mask_src}"
                )

            stacked = np.stack(prior_maps, axis=0)   # (5, H, W) float16
            np.save(msk_npy_path, stacked)
            del prior_maps, stacked

        # ── 构建输出行 ─────────────────────────────────────────────────────
        updated_row = row.copy()
        updated_row['original_nii_path'] = str(orig_nii)
        # [Bug Fix 1] 列名改为 image_path，与 train.py CTPDataset 对齐
        updated_row['image_path'] = str(img_npy_path.absolute())
        updated_row['mask_path']  = str(msk_npy_path.absolute())
        updated_row.pop('nii_path', None)   # 避免与 image_path 混淆
        return updated_row

    except Exception as e:
        print(f"\n[SKIP] {row.get('nii_path', '?')} — {type(e).__name__}: {e}")
        return None

# ======================
# 主函数
# ======================
def main():
    print(f"启动 CTP 预处理 | Workers={NUM_WORKERS}")

    df       = pd.read_csv(SOURCE_CSV)
    mask_done = df.apply(is_done, axis=1)
    pending_df = df[~mask_done]
    print(f"总数: {len(df)} | 已完成: {mask_done.sum()} | 待处理: {len(pending_df)}")

    if len(pending_df) == 0:
        print("所有数据已完成")
        return

    tasks = pending_df.to_dict('records')
    final_results = []

    with Pool(NUM_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(process_case, tasks),
                        total=len(tasks), desc="Processing"):
            if res is not None:
                final_results.append(res)

    pd.DataFrame(final_results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n完成 | 有效样本: {len(final_results)} / {len(df)}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
