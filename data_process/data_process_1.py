"""Prepare raw robot demonstrations for ACT training.

This script mutates task folders after creating `task_copy/`. Its job is to
keep CSV rows, RGB frames, depth frames, and normalized depth frames aligned by
frame id before producing `states_filtered.csv`.
"""

import os
import shutil
import glob
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from PIL import Image
from tqdm import tqdm
import math

# ====================== 配置参数 ======================
DISTANCE_THRESHOLD = 2.5       # 轨迹距离阈值，小于此值删除帧
DEPTH_CLIP_MIN = 0             # 深度截取最小值
DEPTH_CLIP_MAX = 800           # 深度截取最大值
WINDOW_SIZE = 5                # 轨迹平滑窗口（必须奇数）
POLY_ORDER = 2                 # 轨迹平滑阶数
MAX_FRAME_GAP = 5              # 最大允许连续静止帧数
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ======================================================

def natural_sort(lst):
    """Sort filenames by embedded numbers instead of lexicographic order."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key=alphanum_key)

def backup_task_folder(task_dir):
    """Create a one-time backup before destructive preprocessing."""
    backup_dir = os.path.join(task_dir, "task_copy")
    if os.path.exists(backup_dir):
        return
    shutil.copytree(task_dir, backup_dir, ignore=shutil.ignore_patterns("task_copy"))
    print(f"[OK] 备份完成：{backup_dir}")

def process_depth_images(task_dir):
    """Clip raw depth and save an 8-bit normalized copy per frame."""
    depth_in_dir = os.path.join(task_dir, "depth")
    depth_out_dir = os.path.join(task_dir, "depth_normalized")
    os.makedirs(depth_out_dir, exist_ok=True)

    depth_paths = natural_sort(glob.glob(os.path.join(depth_in_dir, "*.png")))
    if not depth_paths:
        return

    print(f"[INFO] 处理深度图：{len(depth_paths)} 张")
    for path in tqdm(depth_paths):
        img = np.array(Image.open(path), dtype=np.float32)
        img = np.clip(img, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
        img = (img - DEPTH_CLIP_MIN) / (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN) * 255
        img = img.astype(np.uint8)
        out_path = os.path.join(depth_out_dir, os.path.basename(path).replace(".png", ".jpg"))
        Image.fromarray(img).save(out_path, "JPEG")

def frame_number_from_path(path):
    """Extract the numeric frame id from a sensor image filename."""
    matches = re.findall(r"\d+", os.path.basename(path))
    if not matches:
        raise ValueError(f"文件名没有帧号: {path}")
    return int(matches[0])

def index_frame_files(img_dir, pattern):
    """Map frame id to file path and reject duplicated frame ids."""
    frame_map = {}
    duplicates = []
    for path in natural_sort(glob.glob(os.path.join(img_dir, pattern))):
        frame_id = frame_number_from_path(path)
        if frame_id in frame_map:
            duplicates.append(frame_id)
        frame_map[frame_id] = path
    if duplicates:
        raise ValueError(f"{img_dir} 存在重复帧号: {duplicates[:10]}")
    return frame_map

def sync_raw_images_with_csv(task_dir):
    """
    以 states_clean.csv 的 frame 列、rgb 文件名、depth 文件名三方交集为准。
    - CSV 缺行：删除对应 rgb/depth/depth_normalized 图片
    - 图片缺失：删除对应 CSV 行
    这样后续不会因为简单截断导致图像和关节错位。
    """
    rgb_dir = os.path.join(task_dir, "rgb")
    depth_dir = os.path.join(task_dir, "depth")
    csv_path = os.path.join(task_dir, "states_clean.csv")

    if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir) or not os.path.exists(csv_path):
        print(f"[WARN] {task_dir} 缺少关键文件夹/文件，跳过！")
        return False

    try:
        rgb_map = index_frame_files(rgb_dir, "*.jpg")
        depth_map = index_frame_files(depth_dir, "*.png")
        csv_df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[WARN] {task_dir} 一致性检查失败：{exc}")
        return False

    if "frame" not in csv_df.columns:
        print("[WARN] states_clean.csv 没有 frame 列，按行号补 frame。")
        csv_df.insert(0, "frame", list(range(len(csv_df))))

    frame_values = pd.to_numeric(csv_df["frame"], errors="coerce")
    if frame_values.isnull().any():
        bad_rows = frame_values[frame_values.isnull()].index.tolist()
        print(f"[WARN] CSV frame 列有非数字值，行号：{bad_rows[:10]}，跳过！")
        return False

    csv_df["frame"] = frame_values.astype(int)
    csv_frames = set(csv_df["frame"].tolist())
    rgb_frames = set(rgb_map.keys())
    depth_frames = set(depth_map.keys())
    common_frames = csv_frames & rgb_frames & depth_frames

    if not common_frames:
        print("[WARN] CSV/RGB/Depth 没有任何共同帧，跳过！")
        return False

    extra_image_frames = (rgb_frames | depth_frames) - common_frames
    dropped_csv_frames = csv_frames - common_frames
    changed = False

    if extra_image_frames:
        delete_unused_images(task_dir, common_frames, delete_mode=False)
        changed = True
        print(f"[FIX] CSV 缺行或图像不成对，已删除多余图片帧：{sorted(extra_image_frames)[:20]}")

    if dropped_csv_frames:
        csv_df = csv_df[csv_df["frame"].isin(common_frames)].reset_index(drop=True)
        csv_df.to_csv(csv_path, index=False)
        changed = True
        print(f"[FIX] 图片缺失，已删除 CSV 行帧：{sorted(dropped_csv_frames)[:20]}")

    if changed:
        rgb_num = len(index_frame_files(rgb_dir, "*.jpg"))
        depth_num = len(index_frame_files(depth_dir, "*.png"))
        csv_num = len(pd.read_csv(csv_path))
        print(f"[OK] 已修正数据对齐：RGB={rgb_num} | Depth={depth_num} | 轨迹={csv_num}")
    else:
        print(f"[OK] 数据匹配：总帧数 {len(common_frames)}")

    return True

def clean_bad_rows_in_trajectory(df, task_dir):
    """Remove invalid joint rows and delete their corresponding images."""
    joint_cols = [c for c in df.columns if c != "frame"]
    original_len = len(df)

    bad_mask = df[joint_cols].isnull().any(axis=1) | \
               np.isinf(df[joint_cols]).any(axis=1) | \
               (df[joint_cols] == "").any(axis=1)

    bad_indices = df.index[bad_mask].tolist()
    if len(bad_indices) == 0:
        return df

    print(f"[WARN] 发现 {len(bad_indices)} 行坏数据（空/NaN/inf），帧号：{bad_indices}")
    df_clean = df[~bad_mask].reset_index(drop=True)
    deleted_frames = df.loc[bad_mask, "frame"].tolist()
    delete_unused_images(task_dir, deleted_frames, delete_mode=True)
    print(f"[OK] 已删除坏帧图片：{deleted_frames}")
    return df_clean

def smooth_trajectory(df):
    """Apply Savitzky-Golay smoothing to joint columns."""
    joint_cols = [col for col in df.columns if col != "frame"]
    for col in joint_cols:
        df[col] = savgol_filter(df[col], window_length=WINDOW_SIZE, polyorder=POLY_ORDER, mode="nearest")
        df[col] = df[col].round().astype(int)
    return df

def filter_trajectory(df):
    """Drop near-static frames while preserving occasional samples across pauses."""
    if len(df) <= 1:
        return df

    joint_cols = [c for c in df.columns if c != "frame"]
    data = df[joint_cols].values
    keep_mask = [True]
    flag = 0

    for i in range(1, len(data)):
        prev = data[flag]
        curr = data[i]
        dist = math.sqrt(np.sum((curr - prev) ** 2))
        if dist >= DISTANCE_THRESHOLD or (i - flag) >= MAX_FRAME_GAP:
            keep_mask.append(True)
            flag = i
        else:
            keep_mask.append(False)

    filtered_df = df[keep_mask].reset_index(drop=True)
    filtered_df["original_frame"] = df.loc[keep_mask, "frame"].values
    deleted = len(df) - len(filtered_df)
    print(f"[INFO] 轨迹过滤：保留 {len(filtered_df)} 帧，删除 {deleted} 帧")
    return filtered_df

def delete_unused_images(task_dir, frame_list, delete_mode=False):
    """Delete images either in `frame_list` or outside it, depending on mode."""
    frames = set(frame_list)
    targets = [
        (os.path.join(task_dir, "rgb"), ".jpg"),
        (os.path.join(task_dir, "depth"), ".png"),
        (os.path.join(task_dir, "depth_normalized"), ".jpg"),
    ]

    for img_dir, ext in targets:
        if not os.path.exists(img_dir):
            continue
        files = natural_sort(glob.glob(os.path.join(img_dir, f"*{ext}")))
        for f in files:
            # 修复点1
            num = int(re.findall(r"\d+", os.path.basename(f))[0])
            if delete_mode:
                if num in frames:
                    os.remove(f)
            else:
                if num not in frames:
                    os.remove(f)

def rename_images_continuous(task_dir, total_frames):
    """Rename remaining images to 000000... after filtering."""
    targets = [
        (os.path.join(task_dir, "rgb"), ".jpg"),
        (os.path.join(task_dir, "depth"), ".png"),
        (os.path.join(task_dir, "depth_normalized"), ".jpg"),
    ]

    for img_dir, ext in targets:
        if not os.path.exists(img_dir):
            continue
        imgs = natural_sort(glob.glob(os.path.join(img_dir, f"*{ext}")))
        for idx, path in enumerate(imgs):
            new_name = f"{idx:06d}{ext}"
            new_path = os.path.join(img_dir, new_name)
            if path != new_path:
                os.rename(path, new_path)
    print(f"[OK] 图片已重新连续编号：000000 ~ {total_frames-1:06d}")

def process_single_task(task_dir):
    """Run the full preprocessing pipeline for one task folder."""
    print(f"\n======= 开始处理：{task_dir} =======")

    backup_task_folder(task_dir)

    if not sync_raw_images_with_csv(task_dir):
        return

    process_depth_images(task_dir)

    csv_path = os.path.join(task_dir, "states_clean.csv")
    df = pd.read_csv(csv_path)
    df = clean_bad_rows_in_trajectory(df, task_dir)
    df = smooth_trajectory(df)
    df_filtered = filter_trajectory(df)
    
    original_frames = df_filtered["original_frame"].tolist()
    delete_unused_images(task_dir, original_frames, delete_mode=False)

    df_filtered["frame"] = list(range(len(df_filtered)))
    rename_images_continuous(task_dir, len(df_filtered))

    out_csv = os.path.join(task_dir, "states_filtered.csv")
    df_filtered.to_csv(out_csv, index=False)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print(f"[OK] 任务完成：{task_dir}")

def batch_process_all_tasks():
    """Process every task under data_process/data."""
    task_dirs = natural_sort(glob.glob(os.path.join(SCRIPT_DIR, "./data/task_*")))
    task_dirs = [d for d in task_dirs if os.path.isdir(d) and "task_copy" not in d]

    if not task_dirs:
        print("[WARN] 未找到 task_* 文件夹")
        return

    print(f"[INFO] 找到 {len(task_dirs)} 个任务，开始批量处理...")
    for td in task_dirs:
        process_single_task(td)

    print("\n[OK] 全部任务处理完成！")

if __name__ == "__main__":
    batch_process_all_tasks()
