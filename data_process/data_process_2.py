#!/usr/bin/env python3
"""
data_process_2_final.py - 四通道图像生成（新版padding）

功能：
1. 深度图 TOP+LEFT 方向 padding（XY双向）
2. 自动对齐到 640×480
3. RGB与深度图合并为RGBA四通道PNG
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image
import shutil
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
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split(r'(\d+)', key)]
    return sorted(l, key=alphanum_key)

def pad_depth_to_target_size(depth_img, target_w, target_h, pad_left, pad_top):
    """
    【新版核心】深度图：TOP+LEFT padding → 强制对齐到 target_w × target_h
    """
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
    """
    创建四通道图像 (RGB + Depth)
    """
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
    task_name = os.path.basename(task_dir)
    print(f"\n======= 处理任务: {task_name} =======")

    rgb_dir = os.path.join(task_dir, "rgb")
    depth_dir = os.path.join(task_dir, "depth_normalized")
    output_dir = os.path.join(task_dir, "four_channel")

    if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
        print(f"❌ 缺少目录")
        return

    os.makedirs(output_dir, exist_ok=True)

    rgb_files = natural_sort(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    depth_files = natural_sort(glob.glob(os.path.join(depth_dir, "*.jpg")))

    if not rgb_files or not depth_files:
        print(f"❌ 无图像文件")
        return

    n_pairs = min(len(rgb_files), len(depth_files))
    success = 0

    for i in range(n_pairs):
        rgb_p = rgb_files[i]
        depth_p = depth_files[i]

        # 帧号匹配
        f_num = re.findall(r'\d+', os.path.basename(rgb_p))[0]
        out_p = os.path.join(output_dir, f"{int(f_num):06d}.png")

        if create_four_channel_image(rgb_p, depth_p, out_p):
            success += 1

        if (i+1) % 50 == 0 or (i+1) == n_pairs:
            print(f"  进度: {i+1}0{n_pairs}  成功:{success}")

    print(f"✅ 完成: {success}/{n_pairs}")

def batch_process_all_tasks(data_root):
    task_dirs = natural_sort(glob.glob(os.path.join(data_root, "./data/task_*")))
    task_dirs = [d for d in task_dirs if os.path.isdir(d) and "task_copy" not in d]

    if not task_dirs:
        print("❌ 未找到 task_*")
        return

    print(f"🚀 找到 {len(task_dirs)} 个任务")
    print(f"📏 目标尺寸: {TARGET_WIDTH}×{TARGET_HEIGHT}")
    print(f"🧩 Padding: 左{PAD_LEFT}px, 上{PAD_TOP}px")
    print("="*60)

    for td in task_dirs:
        process_single_task(td)

    print("\n🎉🎉🎉 全部生成完成！")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    batch_process_all_tasks(SCRIPT_DIR)
