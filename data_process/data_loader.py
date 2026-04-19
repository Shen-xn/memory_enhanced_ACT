"""
ACT data loading.

This loader now serves one single visual contract:
- disk images are OpenCV BGRA four-channel PNGs in `four_channel/`;
- training consumes one current image plus one current qpos;
- phase-prototype supervision comes from per-task offline files.
"""

from __future__ import annotations

import glob
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data_process.exclusions import EXCLUSION_FILENAME, is_task_excluded, load_excluded_tasks


FIXED_JOINT_MIN = np.array([0, 100, 50, 50, 50, 150], dtype=np.float32)
FIXED_JOINT_MAX = np.array([1000, 800, 650, 900, 950, 700], dtype=np.float32)
FIXED_JOINT_RNG = FIXED_JOINT_MAX - FIXED_JOINT_MIN
JOINT_COLS = ["j1", "j2", "j3", "j4", "j5", "j10"]


def get_fixed_joint_stats():
    return {
        "min": FIXED_JOINT_MIN.copy(),
        "max": FIXED_JOINT_MAX.copy(),
        "rng": FIXED_JOINT_RNG.copy(),
    }


def read_bgra_image(path):
    image = None
    buffer = np.fromfile(path, dtype=np.uint8)
    if buffer.size > 0:
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected BGRA image, got {image.shape} from {path}")
    return image.astype(np.float32) / 255.0


def select_image_channels(image, image_channels):
    if image_channels not in (3, 4):
        raise ValueError(f"image_channels must be 3 or 4, got {image_channels}")
    return image[:, :, :image_channels]


def _preview_values(values, limit=8):
    values = sorted(values)
    shown = values[:limit]
    suffix = "..." if len(values) > limit else ""
    return f"{shown}{suffix}"


def _frame_id_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return int(stem)
    match = re.search(r"\d+", stem)
    if match:
        return int(match.group(0))
    raise ValueError(f"Image filename has no numeric frame id: {path}")


def _index_image_paths_by_frame(paths):
    frame_to_path = {}
    duplicates = []
    for path in paths:
        frame_id = _frame_id_from_path(path)
        if frame_id in frame_to_path:
            duplicates.append(frame_id)
        frame_to_path[frame_id] = path
    if duplicates:
        raise ValueError(f"Duplicate image frame ids found: {_preview_values(duplicates)}")
    return frame_to_path


def _frame_ids_from_dataframe(df):
    if "frame" not in df.columns:
        return np.arange(len(df), dtype=np.int64)
    frame_values = pd.to_numeric(df["frame"], errors="coerce")
    if frame_values.isnull().any():
        bad_rows = frame_values[frame_values.isnull()].index.tolist()
        raise ValueError(f"CSV frame column contains non-numeric values at rows {_preview_values(bad_rows)}")
    return frame_values.astype(np.int64).to_numpy()


def _sample_shared_affine():
    return (
        np.random.uniform(-2.0, 2.0),
        np.random.uniform(0.98, 1.02),
        np.random.uniform(-8.0, 8.0),
        np.random.uniform(-8.0, 8.0),
    )


def _warp_bgra(image, angle, scale, tx, ty):
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _augment_rgb_channels(image):
    rgb = image[:, :, :3]
    depth = image[:, :, 3:]
    gamma = np.random.uniform(0.6, 1.4)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), gamma)
    noise_std = np.random.uniform(0.02, 0.05)
    noise = np.random.normal(0.0, noise_std, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb + noise, 0.0, 1.0)
    return np.concatenate([rgb, depth], axis=2)


def _source_group_key(task_path):
    task_name = os.path.basename(task_path)
    if task_name.startswith("task_obst_"):
        return task_name[len("task_obst_") :]
    if task_name.startswith("task_"):
        return task_name[len("task_") :]
    return task_name


def _split_group_keys(group_keys, train_ratio, label):
    if len(group_keys) <= 1:
        raise ValueError(f"Need at least 2 {label} source groups, got {len(group_keys)}.")
    split_idx = int(len(group_keys) * train_ratio)
    split_idx = max(1, min(len(group_keys) - 1, split_idx))
    return set(group_keys[:split_idx]), set(group_keys[split_idx:])


def _format_joint_range_errors(samples, joint_min, joint_max, limit=8):
    bad = []
    for sample in samples:
        values = np.vstack([sample["curr_raw"].reshape(1, -1), sample["future_raw"]])
        mask = (values < joint_min.reshape(1, -1)) | (values > joint_max.reshape(1, -1))
        if not mask.any():
            continue
        cols = sorted({JOINT_COLS[idx] for idx in np.where(mask)[1]})
        bad.append(f"{os.path.basename(sample['task'])}:frame={sample['frame_index']} cols={cols}")
        if len(bad) >= limit:
            break
    suffix = " ..." if len(bad) >= limit else ""
    return "; ".join(bad) + suffix


class ImitationDataset(Dataset):
    def __init__(
        self,
        data_root,
        future_steps=10,
        mode="train",
        train_ratio=0.8,
        seed=42,
        normalize_joints=True,
        strict_alignment=True,
        joint_min_max=None,
        image_channels=4,
        target_mode="absolute",
        delta_qpos_scale=10.0,
        phase_targets_filename="phase_pca16_targets.npz",
        require_phase_targets=True,
    ):
        self.data_root = data_root
        self.future_steps = future_steps
        self.mode = mode
        self.normalize_joints = normalize_joints
        self.strict_alignment = strict_alignment
        self.joint_min_max = joint_min_max
        self.seed = seed
        self.image_channels = int(image_channels)
        self.target_mode = str(target_mode).lower()
        self.delta_qpos_scale = float(delta_qpos_scale)
        self.phase_targets_filename = phase_targets_filename
        self.require_phase_targets = bool(require_phase_targets)

        if self.image_channels not in (3, 4):
            raise ValueError(f"image_channels must be 3 or 4, got {self.image_channels}")
        if self.target_mode not in ("absolute", "delta"):
            raise ValueError(f"target_mode must be 'absolute' or 'delta', got {target_mode}")
        if self.target_mode == "delta" and self.delta_qpos_scale <= 0:
            raise ValueError("delta_qpos_scale must be > 0 when target_mode='delta'.")

        self.all_samples = self._load_all_samples()
        self._split_by_task_strict(train_ratio, seed)
        if normalize_joints and self.joint_min_max is None:
            self.joint_min_max = get_fixed_joint_stats()
        if normalize_joints:
            self._validate_joint_ranges()
        self._prepare_model_inputs_and_targets()

    def _align_dataframe_and_images(self, task_name, img_paths, df):
        if not all(col in df.columns for col in JOINT_COLS):
            print(f"{task_name} 缺少关节列，跳过")
            return [], pd.DataFrame()

        bad_mask = df[JOINT_COLS].isnull().any(axis=1)
        if bad_mask.any():
            bad_frame_ids = _frame_ids_from_dataframe(df.loc[bad_mask])
            raise ValueError(
                f"{task_name} 的 states_filtered.csv 存在空关节行，帧号 {_preview_values(bad_frame_ids.tolist())}。"
            )

        frame_to_image = _index_image_paths_by_frame(img_paths)
        frame_ids = _frame_ids_from_dataframe(df)
        image_frames = set(frame_to_image.keys())
        csv_frames = set(int(x) for x in frame_ids)
        missing_images = csv_frames - image_frames
        extra_images = image_frames - csv_frames
        if self.strict_alignment and (missing_images or extra_images):
            raise ValueError(
                f"{task_name} 的 four_channel 与 states_filtered.csv 不一致。"
                f" CSV有但图片缺失: {_preview_values(missing_images)};"
                f" 图片有但CSV缺失: {_preview_values(extra_images)}"
            )

        keep_mask = [int(frame_id) in frame_to_image for frame_id in frame_ids]
        aligned_df = df.loc[keep_mask].reset_index(drop=True)
        aligned_frame_ids = frame_ids[keep_mask]
        aligned_img_paths = [frame_to_image[int(frame_id)] for frame_id in aligned_frame_ids]
        return aligned_img_paths, aligned_df

    def _load_phase_targets(self, task_dir, expected_samples):
        phase_path = os.path.join(task_dir, self.phase_targets_filename)
        if not os.path.exists(phase_path):
            if self.require_phase_targets:
                raise FileNotFoundError(f"Missing phase targets: {phase_path}")
            return None

        payload = np.load(phase_path)
        frame_index = payload["frame_index"].astype(np.int64)
        pca_coord_tgt = payload["pca_coord_tgt"].astype(np.float32)
        residual_tgt = payload["residual_tgt"].astype(np.float32)
        pca_recon_tgt = payload["pca_recon_tgt"].astype(np.float32)
        if len(frame_index) != expected_samples:
            raise ValueError(
                f"{phase_path} sample count mismatch: expected {expected_samples}, got {len(frame_index)}"
            )
        return {
            "frame_index": frame_index,
            "pca_coord_tgt": pca_coord_tgt,
            "residual_tgt": residual_tgt,
            "pca_recon_tgt": pca_recon_tgt,
        }

    def _load_all_samples(self):
        samples = []
        task_dirs = sorted(glob.glob(os.path.join(self.data_root, "task*")))
        task_dirs = [d for d in task_dirs if os.path.isdir(d) and "task_copy" not in d]
        excluded_tasks = load_excluded_tasks(self.data_root)
        if excluded_tasks:
            before_count = len(task_dirs)
            task_dirs = [d for d in task_dirs if not is_task_excluded(os.path.basename(d), excluded_tasks)]
            print(
                f"[WARN] 根据 {EXCLUSION_FILENAME} 跳过 {before_count - len(task_dirs)} 个任务 "
                f"{_preview_values(excluded_tasks.keys())}"
            )

        print(f"找到 {len(task_dirs)} 个任务文件夹")
        for task_dir in task_dirs:
            task_name = os.path.basename(task_dir)
            img_dir = os.path.join(task_dir, "four_channel")
            csv_path = os.path.join(task_dir, "states_filtered.csv")
            if not os.path.exists(img_dir) or not os.path.exists(csv_path):
                print(f"{task_name} 缺少文件，跳过")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                print(f"{task_name} CSV读取失败，跳过")
                continue

            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            if not img_paths:
                print(f"{task_name} 无 four_channel 图像，跳过")
                continue

            try:
                img_paths, df = self._align_dataframe_and_images(task_name, img_paths, df)
            except ValueError:
                if self.strict_alignment:
                    raise
                print(f"{task_name} 对齐失败，跳过")
                continue

            n_frames = min(len(img_paths), len(df))
            max_idx = n_frames - self.future_steps
            if max_idx <= 0:
                print(f"{task_name} 帧数不足，跳过")
                continue

            phase_targets = self._load_phase_targets(task_dir, max_idx)
            is_obstacle = "obst" in task_name.lower()
            for i in range(max_idx):
                sample = {
                    "img_path": img_paths[i],
                    "curr_raw": df.iloc[i][JOINT_COLS].values.astype(np.float32),
                    "future_raw": df.iloc[i + 1 : i + 1 + self.future_steps][JOINT_COLS].values.astype(np.float32),
                    "pca_coord_tgt": None if phase_targets is None else phase_targets["pca_coord_tgt"][i],
                    "residual_tgt": None if phase_targets is None else phase_targets["residual_tgt"][i],
                    "pca_recon_tgt": None if phase_targets is None else phase_targets["pca_recon_tgt"][i],
                    "task": task_dir,
                    "frame_index": i,
                    "obst": is_obstacle and i > 0,
                }
                samples.append(sample)

        print(f"总有效样本数: {len(samples)}")
        return samples

    def _interleave_samples(self, samples, seed):
        grouped = {}
        for sample in samples:
            grouped.setdefault(sample["task"], []).append(sample)

        rng = random.Random(seed)
        task_keys = list(grouped.keys())
        rng.shuffle(task_keys)
        for task in task_keys:
            rng.shuffle(grouped[task])

        mixed = []
        while task_keys:
            next_task_keys = []
            for task in task_keys:
                task_samples = grouped[task]
                if task_samples:
                    mixed.append(task_samples.pop())
                if task_samples:
                    next_task_keys.append(task)
            rng.shuffle(next_task_keys)
            task_keys = next_task_keys
        return mixed

    def _balance_obstacle_ratio(self, samples, seed):
        obst_samples = [sample for sample in samples if sample["obst"]]
        normal_samples = [sample for sample in samples if not sample["obst"]]
        if not obst_samples or not normal_samples:
            return samples
        keep_count = min(len(obst_samples), len(normal_samples))
        rng = random.Random(seed)
        rng.shuffle(obst_samples)
        rng.shuffle(normal_samples)
        return obst_samples[:keep_count] + normal_samples[:keep_count]

    def _split_by_task_strict(self, train_ratio, seed):
        task_to_obst = {}
        task_to_group = {}
        for sample in self.all_samples:
            task_to_obst[sample["task"]] = bool(sample["obst"])
            task_to_group[sample["task"]] = _source_group_key(sample["task"])

        grouped_tasks = {}
        for task, group in task_to_group.items():
            grouped_tasks.setdefault(group, set()).add(task)

        normal_group_keys = sorted(
            group for group, tasks in grouped_tasks.items() if any(not task_to_obst[task] for task in tasks)
        )
        obstacle_group_keys = sorted(
            group for group, tasks in grouped_tasks.items() if any(task_to_obst[task] for task in tasks)
        )

        rng = random.Random(seed)
        rng.shuffle(normal_group_keys)
        train_groups, val_groups = _split_group_keys(normal_group_keys, train_ratio, "normal")

        rng.shuffle(obstacle_group_keys)
        if obstacle_group_keys:
            obstacle_train_groups, obstacle_val_groups = _split_group_keys(
                obstacle_group_keys, train_ratio, "obstacle"
            )
            train_groups |= obstacle_train_groups
            val_groups |= obstacle_val_groups

        train_tasks = {task for group in train_groups for task in grouped_tasks[group]}
        val_tasks = {task for group in val_groups for task in grouped_tasks[group]}
        self.samples = [s for s in self.all_samples if s["task"] in (train_tasks if self.mode == "train" else val_tasks)]
        self.samples = self._balance_obstacle_ratio(
            self.samples, seed + (0 if self.mode == "train" else 1000)
        )
        self.samples = self._interleave_samples(
            self.samples, seed + (0 if self.mode == "train" else 1000)
        )

        sample_obst_count = sum(1 for sample in self.samples if sample["obst"])
        self.split_has_obstacle_samples = sample_obst_count > 0
        print(
            f"{self.mode.upper()} samples: {len(self.samples)} | "
            f"obst={sample_obst_count}, normal={len(self.samples) - sample_obst_count}"
        )
        if not self.samples:
            raise ValueError(f"{self.mode.upper()} split is empty.")

    def _prepare_model_inputs_and_targets(self):
        if self.normalize_joints:
            jmin = self.joint_min_max["min"]
            jrng = self.joint_min_max["rng"]
        for sample in self.samples:
            curr_raw = sample["curr_raw"]
            future_raw = sample["future_raw"]
            if self.normalize_joints:
                sample["curr"] = (curr_raw - jmin) / jrng
            else:
                sample["curr"] = curr_raw.copy()

            if self.target_mode == "delta":
                step_delta = np.empty_like(future_raw)
                step_delta[0] = future_raw[0] - curr_raw
                if len(future_raw) > 1:
                    step_delta[1:] = future_raw[1:] - future_raw[:-1]
                sample["future"] = step_delta / self.delta_qpos_scale
            elif self.normalize_joints:
                sample["future"] = (future_raw - jmin.reshape(1, -1)) / jrng.reshape(1, -1)
            else:
                sample["future"] = future_raw.copy()

    def _validate_joint_ranges(self):
        jmin = self.joint_min_max["min"]
        jmax = self.joint_min_max["max"]
        bad_message = _format_joint_range_errors(self.samples, jmin, jmax)
        if bad_message:
            raise ValueError(
                "states_filtered.csv contains joint values outside the fixed physical limits. "
                f"Limits min={jmin.tolist()} max={jmax.tolist()}. Examples: {bad_message}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_np = read_bgra_image(sample["img_path"])
        if self.mode == "train":
            angle, scale, tx, ty = _sample_shared_affine()
            img_np = _warp_bgra(img_np, angle, scale, tx, ty)
            img_np = _augment_rgb_channels(img_np)
        img_np = select_image_channels(img_np, self.image_channels)
        img = torch.from_numpy(img_np.copy()).permute(2, 0, 1)

        curr = torch.from_numpy(sample["curr"]).float()
        future = torch.from_numpy(sample["future"]).float()
        pca_coord_tgt = torch.from_numpy(sample["pca_coord_tgt"]).float()
        residual_tgt = torch.from_numpy(sample["residual_tgt"]).float()
        pca_recon_tgt = torch.from_numpy(sample["pca_recon_tgt"]).float()
        obst = torch.tensor([sample["obst"]], dtype=torch.bool)
        return img, curr, future, pca_coord_tgt, residual_tgt, pca_recon_tgt, obst


def get_data_loaders(
    data_root,
    future_steps=10,
    batch_size=8,
    num_workers=0,
    image_channels=4,
    target_mode="absolute",
    delta_qpos_scale=10.0,
    phase_targets_filename="phase_pca16_targets.npz",
):
    train_dataset = ImitationDataset(
        data_root,
        future_steps=future_steps,
        mode="train",
        image_channels=image_channels,
        target_mode=target_mode,
        delta_qpos_scale=delta_qpos_scale,
        phase_targets_filename=phase_targets_filename,
        require_phase_targets=True,
    )
    val_dataset = ImitationDataset(
        data_root,
        future_steps=future_steps,
        mode="val",
        joint_min_max=train_dataset.joint_min_max,
        image_channels=image_channels,
        target_mode=target_mode,
        delta_qpos_scale=delta_qpos_scale,
        phase_targets_filename=phase_targets_filename,
        require_phase_targets=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("\n数据加载器创建完成")
    return train_loader, val_loader
