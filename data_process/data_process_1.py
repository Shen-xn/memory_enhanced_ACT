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
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key=alphanum_key)

def backup_task_folder(task_dir):
    backup_dir = os.path.join(task_dir, "task_copy")
    if os.path.exists(backup_dir):
        return
    shutil.copytree(task_dir, backup_dir, ignore=shutil.ignore_patterns("task_copy"))
    print(f"✅ 备份完成：{backup_dir}")

def process_depth_images(task_dir):
    depth_in_dir = os.path.join(task_dir, "depth")
    depth_out_dir = os.path.join(task_dir, "depth_normalized")
    os.makedirs(depth_out_dir, exist_ok=True)

    depth_paths = natural_sort(glob.glob(os.path.join(depth_in_dir, "*.png")))
    if not depth_paths:
        return

    print(f"📏 处理深度图：{len(depth_paths)} 张")
    for path in tqdm(depth_paths):
        img = np.array(Image.open(path), dtype=np.float32)
        img = np.clip(img, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
        img = (img - DEPTH_CLIP_MIN) / (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN) * 255
        img = img.astype(np.uint8)
        out_path = os.path.join(depth_out_dir, os.path.basename(path).replace(".png", ".jpg"))
        Image.fromarray(img).save(out_path, "JPEG")

def check_dataset_consistency(task_dir):
    rgb_dir = os.path.join(task_dir, "rgb")
    depth_dir = os.path.join(task_dir, "depth")
    csv_path = os.path.join(task_dir, "states_clean.csv")

    if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir) or not os.path.exists(csv_path):
        print(f"❌ 【警告】{task_dir} 缺少关键文件夹/文件，跳过！")
        return False

    rgb_num = len(natural_sort(glob.glob(os.path.join(rgb_dir, "*.jpg"))))
    depth_num = len(natural_sort(glob.glob(os.path.join(depth_dir, "*.png"))))
    csv_df = pd.read_csv(csv_path)
    csv_num = len(csv_df)

    if rgb_num == depth_num == csv_num:
        print(f"✅ 数据匹配：总帧数 {rgb_num}")
        return True
    else:
        print(f"❌ 【警告】{task_dir} 数据不匹配！")
        print(f"   - RGB: {rgb_num} | Depth: {depth_num} | 轨迹: {csv_num}")
        print(f"   跳过此任务！\n")
        return False

def clean_bad_rows_in_trajectory(df, task_dir):
    joint_cols = [c for c in df.columns if c != "frame"]
    original_len = len(df)

    bad_mask = df[joint_cols].isnull().any(axis=1) | \
               np.isinf(df[joint_cols]).any(axis=1) | \
               (df[joint_cols] == "").any(axis=1)

    bad_indices = df.index[bad_mask].tolist()
    if len(bad_indices) == 0:
        return df

    print(f"⚠️  发现 {len(bad_indices)} 行坏数据（空/NaN/inf），帧号：{bad_indices}")
    df_clean = df[~bad_mask].reset_index(drop=True)
    deleted_frames = df.loc[bad_mask, "frame"].tolist()
    delete_unused_images(task_dir, deleted_frames, delete_mode=True)
    print(f"✅ 已删除坏帧图片：{deleted_frames}")
    return df_clean

def smooth_trajectory(df):
    joint_cols = [col for col in df.columns if col != "frame"]
    for col in joint_cols:
        df[col] = savgol_filter(df[col], window_length=WINDOW_SIZE, polyorder=POLY_ORDER, mode="nearest")
        df[col] = df[col].round().astype(int)
    return df

def filter_trajectory(df):
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
    print(f"✂️ 轨迹过滤：保留 {len(filtered_df)} 帧，删除 {deleted} 帧")
    return filtered_df

def delete_unused_images(task_dir, frame_list, delete_mode=False):
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
    print(f"✅ 图片已重新连续编号：000000 ~ {total_frames-1:06d}")

def process_single_task(task_dir):
    print(f"\n======= 开始处理：{task_dir} =======")

    if not check_dataset_consistency(task_dir):
        return

    backup_task_folder(task_dir)
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

    print(f"🎉 任务完成：{task_dir}")

def batch_process_all_tasks():
    task_dirs = natural_sort(glob.glob(os.path.join(SCRIPT_DIR, "./data/task_*")))
    task_dirs = [d for d in task_dirs if os.path.isdir(d) and "task_copy" not in d]

    if not task_dirs:
        print("❌ 未找到 task_* 文件夹")
        return

    print(f"🚀 找到 {len(task_dirs)} 个任务，开始批量处理...")
    for td in task_dirs:
        process_single_task(td)

    print("\n🎉🎉🎉 全部任务处理完成！")

if __name__ == "__main__":
    batch_process_all_tasks()
