#!/usr/bin/env python3
"""Build four-channel BGRA training images from aligned RGB/depth folders.

This module is called by prepare_act_data.py. It assumes `rgb/` and
`depth_normalized/` already have the same frame ids and refuses to pair by list
position when they do not.
"""

import os
import glob
import cv2
import numpy as np
import re

# ====================== 【全局Padding参数】======================
# 深度图 左/上 填充像素（可自由修改）
PAD_LEFT = 0      # 左边填充像素
PAD_TOP = 40    # 上边填充像素

# 目标最终尺寸（固定）
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ==============================================================

def natural_sort(l):
    """Sort filenames by embedded numbers instead of lexicographic order."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split(r'(\d+)', key)]
    return sorted(l, key=alphanum_key)

def frame_number_from_path(path):
    """Extract the numeric frame id from a filename."""
    matches = re.findall(r'\d+', os.path.basename(path))
    if not matches:
        raise ValueError(f"文件名没有帧号: {path}")
    return int(matches[0])

def index_frame_files(paths):
    """Map frame id to path and reject duplicate frame ids."""
    frame_map = {}
    duplicates = []
    for path in natural_sort(paths):
        frame_id = frame_number_from_path(path)
        if frame_id in frame_map:
            duplicates.append(frame_id)
        frame_map[frame_id] = path
    if duplicates:
        raise ValueError(f"存在重复帧号: {duplicates[:10]}")
    return frame_map

def pad_depth_to_target_size(depth_img, target_w, target_h, pad_left, pad_top):
    """Pad/crop depth into the final camera-aligned target size."""
    h, w = depth_img.shape[:2]

    # 1. 先按要求 padding left + top
    padded = np.pad(
        depth_img,
        ((pad_top, 0), (pad_left, 0)),  # 上、左 填充
        mode='constant',
        constant_values=0
    )

    # 2. 获取当前新尺寸
    curr_h, curr_w = padded.shape[:2]

    # 3. 自动补齐到目标大小（不够就补，多了就从右下角裁）
    final = np.zeros((target_h, target_w), dtype=np.uint8)
    
    # 复制有效区域
    copy_h = min(curr_h, target_h)
    copy_w = min(curr_w, target_w)
    final[:copy_h, :copy_w] = padded[:copy_h, :copy_w]

    return final

def create_four_channel_image(rgb_path, depth_path, output_path):
    """Create one OpenCV BGRA-style image: BGR color channels plus depth."""
    try:
        # 读取RGB
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            return False
        
        # 读取深度图
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            return False

        # ============== 新版深度图对齐 ==============
        depth_aligned = pad_depth_to_target_size(
            depth_img,
            target_w=TARGET_WIDTH,
            target_h=TARGET_HEIGHT,
            pad_left=PAD_LEFT,
            pad_top=PAD_TOP
        )

        # RGB强制缩放到目标尺寸（保证完全对齐）
        rgb_aligned = cv2.resize(rgb_img, (TARGET_WIDTH, TARGET_HEIGHT))

        # ============== 合成四通道 ==============
        four_channel = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 4), dtype=np.uint8)
        four_channel[..., :3] = rgb_aligned
        four_channel[..., 3] = depth_aligned

        cv2.imwrite(output_path, four_channel)
        return True

    except Exception as e:
        print(f"  错误: {e}")
        return False

def process_single_task(task_dir):
    """Generate four_channel PNGs for one task."""
    task_name = os.path.basename(task_dir)
    print(f"\n======= 处理任务: {task_name} =======")

    rgb_dir = os.path.join(task_dir, "rgb")
    depth_dir = os.path.join(task_dir, "depth_normalized")
    output_dir = os.path.join(task_dir, "four_channel")

    if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
        print("[WARN] 缺少目录")
        return

    os.makedirs(output_dir, exist_ok=True)

    rgb_files = natural_sort(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    depth_files = natural_sort(glob.glob(os.path.join(depth_dir, "*.png")))

    if not rgb_files or not depth_files:
        print("[WARN] 无图像文件")
        return

    try:
        rgb_map = index_frame_files(rgb_files)
        depth_map = index_frame_files(depth_files)
    except ValueError as exc:
        print(f"[WARN] 帧号检查失败: {exc}")
        return

    rgb_frames = set(rgb_map.keys())
    depth_frames = set(depth_map.keys())
    if rgb_frames != depth_frames:
        print("[WARN] RGB 与 depth_normalized 帧号不一致，请先运行 python prepare_act_data.py 修正。")
        print(f"   RGB 多出: {sorted(rgb_frames - depth_frames)[:20]}")
        print(f"   Depth 多出: {sorted(depth_frames - rgb_frames)[:20]}")
        return

    frame_ids = sorted(rgb_frames)
    existing_outputs = {
        frame_number_from_path(path): path
        for path in glob.glob(os.path.join(output_dir, "*.png"))
    }
    for frame_id, path in existing_outputs.items():
        if frame_id not in rgb_frames:
            os.remove(path)

    n_pairs = len(frame_ids)
    success = 0

    for i, frame_id in enumerate(frame_ids):
        rgb_p = rgb_map[frame_id]
        depth_p = depth_map[frame_id]
        out_p = os.path.join(output_dir, f"{frame_id:06d}.png")

        if create_four_channel_image(rgb_p, depth_p, out_p):
            success += 1

        if (i+1) % 50 == 0 or (i+1) == n_pairs:
            print(f"  进度: {i+1}/{n_pairs}  成功:{success}")

    print(f"[OK] 完成: {success}/{n_pairs}")

