#!/usr/bin/env python3
"""
将 CTP NIfTI 数据转换为模型可用的 .npy 格式。

路径结构（从 CSV nii_path 自动解析）:
  image : {IMG_ROOT}/{PATIENT_ID}/CTP/{SERIES}.nii.gz
  mask  : {MSK_ROOT}/{patient_id}/{SERIES}/generated_cbf.nii.gz  (共5个)

输出 .npy shape:
  image -> (H, W, 32, T)  float16，空间目标尺寸 512×512×32
  mask  -> (5, 512, 512)  float16，每个灌注图 Z 轴投影后叠加

输出 CSV 列（与 train.py CTPDataset 严格对应）:
  label, image_path, mask_path
"""

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ============================================================
# 配置（按需修改）
# ============================================================
SOURCE_CSV   = "/data/truenas_B2/yyi/datapath/CTP_2_2/CTP_第一轮数据_truenas_36T_21_22.csv"
OUTPUT_CSV   = "/data/truenas_B2/yyi/datapath/CTP_2_2/CTP_第一轮数据_truenas_36T_21_22_npy.csv"
MSK_SRC_ROOT = Path("/data/truenas_B2/yyi/CTP_mask")
SAVE_ROOT    = Path("/data/truenas_B2/yyi/data/CTP_npy")

# CTP 4D 非常耗内存，建议不超过 cpu // 4
NUM_WORKERS = min(8, max(1, cpu_count() // 4))

# 5 个灌注参数图及 Z 轴投影方向
# min: 低值代表梗死核心 (CBF/CBV)
# max: 高值代表低灌注半暗带 (MTT/Tmax/TTP)
FEATURE_CONFIG = {
    'generated_cbf.nii.gz':  'min',
    'generated_cbv.nii.gz':  'min',
    'generated_mtt.nii.gz':  'max',
    'generated_tmax.nii.gz': 'max',
    'generated_ttp.nii.gz':  'max',
}

# ============================================================
# 工具函数
# ============================================================

def load_nii(path: Path) -> np.ndarray:
    """内存映射加载 NIfTI，减少内存复制。"""
    return np.asarray(nib.load(str(path), mmap=True).dataobj, dtype=np.float32)


def resize_4d(volume: np.ndarray, target=(512, 512, 32)) -> np.ndarray:
    """
    Trilinear 插值缩放 4D CTP 体积。
    输入/输出 shape: (H, W, Z, T)
    """
    if volume.shape[:3] == target:
        return volume
    # (H,W,Z,T) -> (T,1,H,W,Z)
    v = torch.from_numpy(volume).float().permute(3, 0, 1, 2).unsqueeze(1)
    v = torch.nn.functional.interpolate(v, size=target, mode='trilinear', align_corners=False)
    return v.squeeze(1).permute(1, 2, 3, 0).numpy()  # (H,W,Z,T)


def resize_2d(img: np.ndarray, target=(512, 512)) -> np.ndarray:
    """Bilinear 插值缩放 2D 图像，输入/输出 shape: (H, W)"""
    if img.shape == target:
        return img
    v = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    v = torch.nn.functional.interpolate(v, size=target, mode='bilinear', align_corners=False)
    return v.squeeze().numpy()


def find_mask_dir(patient_id: str, series_name: str) -> Path | None:
    """
    查找 mask 目录，自动处理 patient_id 大小写差异。
    优先精确匹配，失败则遍历大小写。
    """
    exact = MSK_SRC_ROOT / patient_id / series_name
    if exact.exists():
        return exact

    # 大小写不敏感搜索（mask 目录有时全小写）
    pid_lower = patient_id.lower()
    for candidate in MSK_SRC_ROOT.iterdir():
        if candidate.name.lower() == pid_lower:
            p = candidate / series_name
            if p.exists():
                return p

    return None


def parse_paths(nii_path: Path):
    """
    从 nii_path 解析 patient_id 和 series_name。

    nii_path 结构: .../{PATIENT_ID}/CTP/{SERIES}.nii.gz
    """
    patient_id  = nii_path.parent.parent.name           # A110034307
    series_name = nii_path.name.replace('.nii.gz', '')  # Head Volume Perfusion_5.0 x 5.0_301
    return patient_id, series_name

# ============================================================
# 单通道 mask 处理（Z 轴投影）
# ============================================================

def process_mask_channel(f_path: Path, proj: str) -> np.ndarray:
    """
    读取单个灌注参数图，Z 轴投影到 2D，缩放到 (512, 512)。

    Returns: shape (512, 512) float16
    """
    data = load_nii(f_path)               # (H, W, Z) 或 (H, W, Z, T)

    if data.ndim == 4:
        data = data.mean(axis=-1)         # 时间轴均值 → (H, W, Z)

    if data.ndim != 3:
        raise ValueError(f"灌注图维度异常: {data.shape} → {f_path.name}")

    # Z 轴投影：(H, W, Z) → (H, W)
    projected = data.max(axis=2) if proj == 'max' else data.min(axis=2)

    # 空间缩放到 (512, 512)
    projected = resize_2d(projected, target=(512, 512))

    return np.ascontiguousarray(projected, dtype=np.float16)

# ============================================================
# 断点续跑检测
# ============================================================

def is_done(row) -> bool:
    p = Path(row['nii_path'])
    patient_id, series_name = parse_paths(p)
    img_npy = SAVE_ROOT / patient_id / "images" / f"{series_name}.npy"
    msk_npy = SAVE_ROOT / patient_id / "mask"   / f"{series_name}_mask.npy"
    return img_npy.exists() and msk_npy.exists()

# ============================================================
# 单样本处理（多进程 worker）
# ============================================================

def process_case(row: dict):
    try:
        orig_nii = Path(row['nii_path'])
        if not orig_nii.exists():
            print(f"\n[SKIP] 原始文件不存在: {orig_nii}")
            return None

        patient_id, series_name = parse_paths(orig_nii)

        # 输出目录
        img_dir = SAVE_ROOT / patient_id / "images"
        msk_dir = SAVE_ROOT / patient_id / "mask"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        img_npy_path = img_dir / f"{series_name}.npy"
        msk_npy_path = msk_dir / f"{series_name}_mask.npy"

        # ── 1. CTP Image → (H, W, 32, T) float16 ──────────────────────
        if not img_npy_path.exists():
            img = load_nii(orig_nii)
            if img.ndim != 4:
                raise ValueError(f"CTP 应为 4D，实际 {img.ndim}D: {orig_nii}")

            img = resize_4d(img, target=(512, 512, 32))   # (512, 512, 32, T)
            np.save(img_npy_path,
                    np.ascontiguousarray(img, dtype=np.float16))
            del img

        # ── 2. Mask → (5, 512, 512) float16 ───────────────────────────
        if not msk_npy_path.exists():
            mask_src = find_mask_dir(patient_id, series_name)
            if mask_src is None:
                raise FileNotFoundError(
                    f"找不到 mask 目录: {MSK_SRC_ROOT}/{patient_id}/{series_name}"
                )

            channels = []
            for fname, proj in FEATURE_CONFIG.items():
                f_path = mask_src / fname
                if not f_path.exists():
                    raise FileNotFoundError(f"灌注图缺失: {f_path}")
                channels.append(process_mask_channel(f_path, proj))

            stacked = np.stack(channels, axis=0)          # (5, 512, 512)
            np.save(msk_npy_path, stacked)
            del channels, stacked

        # ── 3. 构建输出行 ───────────────────────────────────────────────
        out = {
            'label':               row['label'],
            'image_path':          str(img_npy_path.absolute()),
            'mask_path':           str(msk_npy_path.absolute()),
            'original_nii_path':   str(orig_nii),
        }
        return out

    except Exception as e:
        print(f"\n[SKIP] {row.get('nii_path', '?')} — {type(e).__name__}: {e}")
        return None

# ============================================================
# 主函数
# ============================================================

def main():
    print(f"Workers = {NUM_WORKERS}  (total cores = {cpu_count()})")

    df = pd.read_csv(SOURCE_CSV)
    print(f"CSV 共 {len(df)} 行，列: {list(df.columns)}")

    # 断点续跑：跳过已完成样本
    done_mask  = df.apply(is_done, axis=1)
    pending_df = df[~done_mask]
    print(f"已完成: {done_mask.sum()} | 待处理: {len(pending_df)}")

    if len(pending_df) == 0:
        print("所有样本已处理完毕。")
        _write_output(df)
        return

    tasks = pending_df.to_dict('records')
    results = []

    with Pool(NUM_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(process_case, tasks),
                        total=len(tasks), desc="Converting"):
            if res is not None:
                results.append(res)

    _write_output(results)
    print(f"\n完成 | 有效样本: {len(results)} / {len(df)}")
    print(f"输出 CSV: {OUTPUT_CSV}")


def _write_output(results):
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
