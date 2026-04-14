"""Prepare raw robot demonstrations for ACT training.

This module is called by prepare_act_data.py after raw CSV cleaning. Its job is
to keep CSV rows, RGB frames, depth frames, and normalized depth frames aligned
by frame id before producing `states_filtered.csv`.
"""

import glob
import math
import os
import re
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import savgol_filter
from tqdm import tqdm


DISTANCE_THRESHOLD = 2.5
DEPTH_CLIP_MIN = 0
DEPTH_CLIP_MAX = 800
WINDOW_SIZE = 5
POLY_ORDER = 2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def natural_sort(lst):
    """Sort filenames by embedded numbers instead of lexicographic order."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split(r"([0-9]+)", key)]
    return sorted(lst, key=alphanum_key)


def process_depth_images(task_dir):
    """Clip raw depth and save an 8-bit normalized copy per frame."""
    depth_in_dir = os.path.join(task_dir, "depth")
    depth_out_dir = os.path.join(task_dir, "depth_normalized")
    os.makedirs(depth_out_dir, exist_ok=True)

    depth_paths = natural_sort(glob.glob(os.path.join(depth_in_dir, "*.png")))
    if not depth_paths:
        return

    print(f"[INFO] Processing depth images: {len(depth_paths)}")
    for path in tqdm(depth_paths):
        img = np.array(Image.open(path), dtype=np.float32)
        img = np.clip(img, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
        img = (img - DEPTH_CLIP_MIN) / (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN) * 255
        img = img.astype(np.uint8)
        out_path = os.path.join(depth_out_dir, os.path.basename(path))
        Image.fromarray(img).save(out_path, "PNG")


def frame_number_from_path(path):
    """Extract the numeric frame id from a sensor image filename."""
    matches = re.findall(r"\d+", os.path.basename(path))
    if not matches:
        raise ValueError(f"Filename has no frame number: {path}")
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
        raise ValueError(f"{img_dir} has duplicate frame ids: {duplicates[:10]}")
    return frame_map


def sync_raw_images_with_csv(task_dir):
    """Strictly align raw RGB/depth frames with states_clean.csv by frame id."""
    rgb_dir = os.path.join(task_dir, "rgb")
    depth_dir = os.path.join(task_dir, "depth")
    csv_path = os.path.join(task_dir, "states_clean.csv")

    if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir) or not os.path.exists(csv_path):
        print(f"[WARN] {task_dir} is missing required rgb/depth/states_clean inputs")
        return False

    try:
        rgb_map = index_frame_files(rgb_dir, "*.jpg")
        depth_map = index_frame_files(depth_dir, "*.png")
        csv_df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[WARN] {task_dir} alignment check failed: {exc}")
        return False

    if "frame" not in csv_df.columns:
        print("[WARN] states_clean.csv has no frame column; using row indices")
        csv_df.insert(0, "frame", list(range(len(csv_df))))

    frame_values = pd.to_numeric(csv_df["frame"], errors="coerce")
    if frame_values.isnull().any():
        bad_rows = frame_values[frame_values.isnull()].index.tolist()
        print(f"[WARN] CSV frame column has non-numeric rows: {bad_rows[:10]}")
        return False

    csv_df["frame"] = frame_values.astype(int)
    csv_frames = set(csv_df["frame"].tolist())
    rgb_frames = set(rgb_map.keys())
    depth_frames = set(depth_map.keys())
    common_frames = csv_frames & rgb_frames & depth_frames

    if not common_frames:
        print("[WARN] CSV/RGB/Depth have no shared frames")
        return False

    extra_image_frames = (rgb_frames | depth_frames) - common_frames
    dropped_csv_frames = csv_frames - common_frames
    changed = False

    if extra_image_frames:
        delete_unused_images(task_dir, common_frames, delete_mode=False)
        changed = True
        print(f"[FIX] Removed extra image frames: {sorted(extra_image_frames)[:20]}")

    if dropped_csv_frames:
        csv_df = csv_df[csv_df["frame"].isin(common_frames)].reset_index(drop=True)
        csv_df.to_csv(csv_path, index=False)
        changed = True
        print(f"[FIX] Removed CSV rows without matching images: {sorted(dropped_csv_frames)[:20]}")

    if changed:
        rgb_num = len(index_frame_files(rgb_dir, "*.jpg"))
        depth_num = len(index_frame_files(depth_dir, "*.png"))
        csv_num = len(pd.read_csv(csv_path))
        print(f"[OK] Alignment repaired: RGB={rgb_num} | Depth={depth_num} | CSV={csv_num}")
    else:
        print(f"[OK] Raw alignment already consistent: total_frames={len(common_frames)}")

    return True


def clean_bad_rows_in_trajectory(df, task_dir):
    """Remove invalid joint rows and delete their corresponding images."""
    joint_cols = [c for c in df.columns if c != "frame"]
    bad_mask = (
        df[joint_cols].isnull().any(axis=1)
        | np.isinf(df[joint_cols]).any(axis=1)
        | (df[joint_cols] == "").any(axis=1)
    )

    bad_indices = df.index[bad_mask].tolist()
    if not bad_indices:
        return df

    print(f"[WARN] Found {len(bad_indices)} bad trajectory rows: {bad_indices[:20]}")
    df_clean = df[~bad_mask].reset_index(drop=True)
    deleted_frames = df.loc[bad_mask, "frame"].tolist()
    delete_unused_images(task_dir, deleted_frames, delete_mode=True)
    print(f"[OK] Deleted images for bad frames: {deleted_frames[:20]}")
    return df_clean


def smooth_trajectory(df):
    """Apply Savitzky-Golay smoothing to joint columns."""
    joint_cols = [col for col in df.columns if col != "frame"]
    for col in joint_cols:
        df[col] = savgol_filter(df[col], window_length=WINDOW_SIZE, polyorder=POLY_ORDER, mode="nearest")
        df[col] = df[col].round().astype(int)
    return df


def filter_trajectory(df):
    """Drop near-static frames and keep only motion-above-threshold samples."""
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
        if dist >= DISTANCE_THRESHOLD:
            keep_mask.append(True)
            flag = i
        else:
            keep_mask.append(False)

    filtered_df = df[keep_mask].reset_index(drop=True)
    filtered_df["original_frame"] = df.loc[keep_mask, "frame"].values
    deleted = len(df) - len(filtered_df)
    print(f"[INFO] Trajectory filtered: kept={len(filtered_df)} removed={deleted}")
    return filtered_df


def delete_unused_images(task_dir, frame_list, delete_mode=False):
    """Delete images either in `frame_list` or outside it, depending on mode."""
    frames = set(frame_list)
    targets = [
        (os.path.join(task_dir, "rgb"), ".jpg"),
        (os.path.join(task_dir, "depth"), ".png"),
        (os.path.join(task_dir, "depth_normalized"), ".png"),
    ]

    for img_dir, ext in targets:
        if not os.path.exists(img_dir):
            continue
        files = natural_sort(glob.glob(os.path.join(img_dir, f"*{ext}")))
        for path in files:
            num = int(re.findall(r"\d+", os.path.basename(path))[0])
            if delete_mode:
                if num in frames:
                    os.remove(path)
            else:
                if num not in frames:
                    os.remove(path)


def rename_images_continuous(task_dir, total_frames):
    """Rename remaining images to 000000... after filtering."""
    targets = [
        (os.path.join(task_dir, "rgb"), ".jpg"),
        (os.path.join(task_dir, "depth"), ".png"),
        (os.path.join(task_dir, "depth_normalized"), ".png"),
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
    print(f"[OK] Renamed images to continuous ids: 000000 ~ {total_frames - 1:06d}")


def process_single_task(task_dir):
    """Run the full preprocessing pipeline for one task folder."""
    print(f"\n======= Processing task: {task_dir} =======")

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

    print(f"[OK] Task complete: {task_dir}")


if __name__ == "__main__":
    data_root = os.path.join(SCRIPT_DIR, "data")
    if not os.path.exists(data_root):
        print("No data directory found.")
    else:
        task_dirs = [
            os.path.join(data_root, name)
            for name in os.listdir(data_root)
            if name.startswith("task") and os.path.isdir(os.path.join(data_root, name))
        ]
        for task_dir in task_dirs:
            process_single_task(task_dir)
