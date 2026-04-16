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


DISTANCE_THRESHOLD = 20
DEPTH_CLIP_MIN = 0
DEPTH_CLIP_MAX = 800
WINDOW_SIZE = 10
POLY_ORDER = 2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def summarize_frame_ids(frame_ids, limit=8):
    """Compactly format a frame-id list for logs."""
    ids = list(frame_ids)
    if not ids:
        return "[]"
    if len(ids) <= limit:
        return str(ids)
    head = ", ".join(str(x) for x in ids[: limit // 2])
    tail = ", ".join(str(x) for x in ids[-(limit // 2) :])
    return f"[{head}, ..., {tail}] (total={len(ids)})"


def summarize_removed_runs(removed_ids, total_frames):
    """Describe contiguous removal blocks for high-level task logs."""
    ids = sorted(int(x) for x in removed_ids)
    if not ids:
        return {
            "count": 0,
            "ratio": 0.0,
            "runs": 0,
            "largest_run": 0,
            "largest_start": None,
            "largest_end": None,
            "largest_zone": "none",
        }

    runs = []
    start = prev = ids[0]
    for frame_id in ids[1:]:
        if frame_id == prev + 1:
            prev = frame_id
        else:
            runs.append((start, prev, prev - start + 1))
            start = prev = frame_id
    runs.append((start, prev, prev - start + 1))

    largest = max(runs, key=lambda x: x[2])
    center = ((largest[0] + largest[1]) / 2) / max(total_frames, 1)
    if center < 0.2:
        zone = "head"
    elif center < 0.8:
        zone = "mid"
    else:
        zone = "tail"

    return {
        "count": len(ids),
        "ratio": len(ids) / max(total_frames, 1),
        "runs": len(runs),
        "largest_run": largest[2],
        "largest_start": largest[0],
        "largest_end": largest[1],
        "largest_zone": zone,
    }


def count_files(img_dir, ext):
    """Count files for one image directory."""
    if not os.path.exists(img_dir):
        return 0
    return len(glob.glob(os.path.join(img_dir, f"*{ext}")))


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

    extra_image_frames = sorted((rgb_frames | depth_frames) - common_frames)
    dropped_csv_frames = sorted(csv_frames - common_frames)
    changed = False

    print(
        "[ALIGN] "
        f"csv={len(csv_frames)} rgb={len(rgb_frames)} depth={len(depth_frames)} "
        f"shared={len(common_frames)}"
    )

    if extra_image_frames:
        delete_unused_images(task_dir, common_frames, delete_mode=False)
        changed = True
        print(f"[FIX] Removed extra image frames: {summarize_frame_ids(extra_image_frames)}")

    if dropped_csv_frames:
        csv_df = csv_df[csv_df["frame"].isin(common_frames)].reset_index(drop=True)
        csv_df.to_csv(csv_path, index=False)
        changed = True
        print(f"[FIX] Removed CSV rows without matching images: {summarize_frame_ids(dropped_csv_frames)}")

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

    print(
        f"[WARN] Found {len(bad_indices)} bad trajectory rows: "
        f"{summarize_frame_ids(df.loc[bad_mask, 'frame'].tolist())}"
    )
    df_clean = df[~bad_mask].reset_index(drop=True)
    deleted_frames = df.loc[bad_mask, "frame"].tolist()
    delete_unused_images(task_dir, deleted_frames, delete_mode=True)
    print(f"[OK] Deleted images for bad frames: {summarize_frame_ids(deleted_frames)}")
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
    removed_ids = df.loc[~pd.Series(keep_mask, index=df.index), "frame"].tolist()
    stats = summarize_removed_runs(removed_ids, len(df))
    print(
        "[FILTER] "
        f"kept={len(filtered_df)} removed={stats['count']} "
        f"ratio={stats['ratio']:.2%} runs={stats['runs']} "
        f"largest_run={stats['largest_run']} "
        f"largest_span={stats['largest_start']}->{stats['largest_end']} "
        f"zone={stats['largest_zone']}"
    )
    if removed_ids:
        print(f"[FILTER] removed original frames: {summarize_frame_ids(removed_ids)}")
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
    raw_rows = len(df)
    print(f"[TASK] {os.path.basename(task_dir)} raw rows after CSV clean/alignment: {raw_rows}")
    df = clean_bad_rows_in_trajectory(df, task_dir)
    after_bad_rows = len(df)
    if raw_rows != after_bad_rows:
        print(
            "[TASK] "
            f"after bad-row cleanup: kept={after_bad_rows} removed={raw_rows - after_bad_rows} "
            f"ratio={(raw_rows - after_bad_rows) / max(raw_rows, 1):.2%}"
        )
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

    rgb_count = count_files(os.path.join(task_dir, "rgb"), ".jpg")
    depth_count = count_files(os.path.join(task_dir, "depth"), ".png")
    depth_norm_count = count_files(os.path.join(task_dir, "depth_normalized"), ".png")
    print(
        "[TASK] final counts "
        f"csv={len(df_filtered)} rgb={rgb_count} depth={depth_count} depth_norm={depth_norm_count}"
    )
    print(
        "[TASK] overall retention "
        f"kept={len(df_filtered)}/{raw_rows} ratio={len(df_filtered) / max(raw_rows, 1):.2%}"
    )

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
