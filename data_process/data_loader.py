"""
ACT data loading.

This module is the contract between processed demonstration data and ACT
training. Keep it conservative:
- images on disk are OpenCV BGRA four-channel PNGs;
- `states_filtered.csv` is aligned by its `frame` column, not by truncation;
- optional memory images may be missing, in which case the sample receives a
  zero memory image and the loader prints a warning.
"""

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


def get_fixed_joint_stats():
    """Return fixed physical joint limits used by training and deployment."""
    return {
        "min": FIXED_JOINT_MIN.copy(),
        "max": FIXED_JOINT_MAX.copy(),
        "rng": FIXED_JOINT_RNG.copy(),
    }


def read_bgra_image(path):
    """Read one normalized OpenCV BGRA image from disk."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected a 4-channel image, got shape {image.shape} from {path}")
    return image.astype(np.float32) / 255.0


def select_image_channels(image, image_channels):
    """Return the visual channels requested by the ACT model.

    Disk data stays in one stable BGRA format. RGB baselines simply ignore the
    depth channel here, while RGBD and memory models keep all four channels.
    """
    if image_channels not in (3, 4):
        raise ValueError(f"image_channels must be 3 or 4, got {image_channels}")
    return image[:, :, :image_channels]


def _preview_values(values, limit=8):
    values = sorted(values)
    shown = values[:limit]
    suffix = "..." if len(values) > limit else ""
    return f"{shown}{suffix}"


def _frame_id_from_path(path):
    """Extract the numeric frame id used to align images with CSV rows."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return int(stem)
    match = re.search(r"\d+", stem)
    if match:
        return int(match.group(0))
    raise ValueError(f"Image filename has no numeric frame id: {path}")


def _index_image_paths_by_frame(paths):
    """Build a frame-id -> file-path map and fail on duplicate ids."""
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
    """Return frame ids from `frame`, falling back to row numbers for old CSVs."""
    if "frame" not in df.columns:
        return np.arange(len(df), dtype=np.int64)

    frame_values = pd.to_numeric(df["frame"], errors="coerce")
    if frame_values.isnull().any():
        bad_rows = frame_values[frame_values.isnull()].index.tolist()
        raise ValueError(f"CSV frame column contains non-numeric values at rows {_preview_values(bad_rows)}")
    return frame_values.astype(np.int64).to_numpy()


def _sample_shared_affine():
    """Sample one affine transform shared by image and memory image.

    Using the same transform keeps the current frame and its memory image in
    register; otherwise we would teach the model mismatched geometry.
    """
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
    """Apply photometric augmentation to color channels while preserving depth."""
    rgb = image[:, :, :3]
    depth = image[:, :, 3:]
    gamma = np.random.uniform(0.6, 1.4)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), gamma)
    noise_std = np.random.uniform(0.02, 0.05)
    noise = np.random.normal(0.0, noise_std, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb + noise, 0.0, 1.0)
    return np.concatenate([rgb, depth], axis=2)


def _source_group_key(task_path):
    """Group normal and obstacle versions of the same source trajectory together."""
    task_name = os.path.basename(task_path)
    if task_name.startswith("task_obst_"):
        return task_name[len("task_obst_") :]
    if task_name.startswith("task_"):
        return task_name[len("task_") :]
    return task_name


def _split_group_keys(group_keys, train_ratio, label):
    if len(group_keys) <= 1:
        raise ValueError(f"Need at least 2 {label} source groups to split train/val, but got {len(group_keys)}.")

    split_idx = int(len(group_keys) * train_ratio)
    split_idx = max(1, min(len(group_keys) - 1, split_idx))
    return set(group_keys[:split_idx]), set(group_keys[split_idx:])


def _format_joint_range_errors(samples, joint_min, joint_max, limit=8):
    """Build a compact error message for samples outside fixed joint limits."""
    bad = []
    for sample in samples:
        curr = sample["curr_raw"] if "curr_raw" in sample else sample["curr"]
        future = sample["future_raw"] if "future_raw" in sample else sample["future"]
        values = np.vstack([curr.reshape(1, -1), future])
        mask = (values < joint_min.reshape(1, -1)) | (values > joint_max.reshape(1, -1))
        if not mask.any():
            continue
        cols = sorted({["j1", "j2", "j3", "j4", "j5", "j10"][idx] for idx in np.where(mask)[1]})
        bad.append(f"{os.path.basename(sample['task'])}:frame={sample['frame_index']} cols={cols}")
        if len(bad) >= limit:
            break
    suffix = " ..." if len(bad) >= limit else ""
    return "; ".join(bad) + suffix


class ImitationDataset(Dataset):
    """Dataset for ACT action-chunk training.

    Each item contains:
    - current image: [C, H, W]
    - current normalized qpos: [state_dim]
    - future action targets: [future_steps, state_dim]
    - memory image: [C, H, W], zero when unavailable
    - obstacle flag: [1]
    """
    def __init__(
        self,
        data_root,
        future_steps=10,
        use_memory_image_input=False,
        mode="train",
        train_ratio=0.8,
        seed=42,
        normalize_joints=True,
        strict_alignment=True,
        joint_min_max=None,
        image_channels=4,
        target_mode="absolute",
        delta_qpos_scale=10.0,
    ):
        self.data_root = data_root
        self.future_steps = future_steps
        self.use_memory_image_input = use_memory_image_input
        self.image_channels = int(image_channels)
        if self.image_channels not in (3, 4):
            raise ValueError(f"image_channels must be 3 or 4, got {self.image_channels}")
        if self.use_memory_image_input and self.image_channels != 4:
            raise ValueError("Memory-image ACT currently requires image_channels=4.")
        self.mode = mode
        self.normalize_joints = normalize_joints
        self.strict_alignment = strict_alignment
        self.joint_min_max = joint_min_max
        self.seed = seed
        self.target_mode = str(target_mode).lower()
        self.delta_qpos_scale = float(delta_qpos_scale)
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

    def _align_dataframe_and_images(self, task_name, img_paths, df, joint_cols):
        """Strictly align `states_filtered.csv` with image files by frame id."""
        if not all(c in df.columns for c in joint_cols):
            print(f"{task_name} 缺少关节列，跳过")
            return [], pd.DataFrame()

        bad_mask = df[joint_cols].isnull().any(axis=1)
        if bad_mask.any():
            bad_frame_ids = _frame_ids_from_dataframe(df.loc[bad_mask])
            raise ValueError(
                f"{task_name} 的 states_filtered.csv 存在空/NaN 关节行，帧号 {_preview_values(bad_frame_ids.tolist())}。"
                "请先运行数据预处理同步删除对应图片，再训练。"
            )

        try:
            frame_to_image = _index_image_paths_by_frame(img_paths)
            frame_ids = _frame_ids_from_dataframe(df)
        except ValueError as exc:
            raise ValueError(f"{task_name} 帧编号解析失败: {exc}") from exc

        if len(set(frame_ids.tolist())) != len(frame_ids):
            duplicated = pd.Series(frame_ids).value_counts()
            duplicated = duplicated[duplicated > 1].index.tolist()
            raise ValueError(f"{task_name} 的 CSV frame 有重复值: {_preview_values(duplicated)}")

        image_frames = set(frame_to_image.keys())
        csv_frames = set(int(x) for x in frame_ids)
        missing_images = csv_frames - image_frames
        extra_images = image_frames - csv_frames

        if self.strict_alignment and (missing_images or extra_images):
            raise ValueError(
                f"{task_name} 的 four_channel 与 states_filtered.csv 不一致。"
                f"CSV 有但图片缺失: {_preview_values(missing_images)}；"
                f"图片有但 CSV 缺失: {_preview_values(extra_images)}。"
                "请先运行 python prepare_act_data.py 重新生成并验证数据。"
            )

        # In non-strict mode we keep only the intersection. In strict mode we
        # should already have raised if anything was missing on either side.
        keep_mask = [int(frame_id) in frame_to_image for frame_id in frame_ids]
        aligned_df = df.loc[keep_mask].reset_index(drop=True)
        aligned_frame_ids = frame_ids[keep_mask]
        aligned_img_paths = [frame_to_image[int(frame_id)] for frame_id in aligned_frame_ids]
        return aligned_img_paths, aligned_df

    def _load_all_samples(self):
        """Scan all task folders and flatten them into frame-level samples."""
        samples = []
        task_dirs = sorted(glob.glob(os.path.join(self.data_root, "task*")))
        task_dirs = [d for d in task_dirs if os.path.isdir(d) and "task_copy" not in d]
        excluded_tasks = load_excluded_tasks(self.data_root)
        if excluded_tasks:
            before_count = len(task_dirs)
            task_dirs = [
                d for d in task_dirs
                if not is_task_excluded(os.path.basename(d), excluded_tasks)
            ]
            print(
                f"[WARN] 根据 {EXCLUSION_FILENAME} 跳过 {before_count - len(task_dirs)} 个任务: "
                f"{_preview_values(excluded_tasks.keys())}"
            )

        print(f"找到 {len(task_dirs)} 个任务文件夹")

        for task_dir in task_dirs:
            task_name = os.path.basename(task_dir)
            img_dir = os.path.join(task_dir, "four_channel")
            csv_path = os.path.join(task_dir, "states_filtered.csv")

            mem_img_dir = os.path.join(task_dir, "memory_image_four_channel")
            # Memory images are optional even in memory mode. Missing entries
            # fall back to all-zero tensors so partially processed datasets can
            # still be inspected or trained with explicit warnings.
            has_memory_folder = self.use_memory_image_input and os.path.exists(mem_img_dir)

            if not os.path.exists(img_dir) or not os.path.exists(csv_path):
                print(f"{task_name} 缺失文件，跳过")
                continue

            try:
                df = pd.read_csv(csv_path)
            except:
                print(f"{task_name} CSV读取失败，跳过")
                continue

            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            if len(img_paths) == 0:
                print(f"{task_name} 无四通道图像，跳过")
                continue

            joint_cols = ["j1", "j2", "j3", "j4", "j5", "j10"]
            try:
                img_paths, df = self._align_dataframe_and_images(task_name, img_paths, df, joint_cols)
            except ValueError as exc:
                if self.strict_alignment:
                    raise
                print(f"{exc}，跳过")
                continue

            n_frames = min(len(img_paths), len(df))
            max_idx = n_frames - self.future_steps
            if max_idx <= 0:
                print(f"{task_name} 帧数不足，跳过")
                continue

            is_obstacle = "obst" in task_name.lower()
            missing_memory_count = 0

            for i in range(max_idx):
                mem_img_path = ""
                # Memory image filenames are aligned by the same frame id as
                # `four_channel`, so we reuse the current image basename.
                if has_memory_folder:
                    frame_filename = os.path.basename(img_paths[i])
                    mem_img_path = os.path.join(mem_img_dir, frame_filename)
                has_mem_img = os.path.exists(mem_img_path)
                if self.use_memory_image_input and not has_mem_img:
                    missing_memory_count += 1

                sample = {
                    "img_path": img_paths[i],
                    "mem_img_path": mem_img_path,
                    "has_mem_img": has_mem_img,
                    "curr_raw": df.iloc[i][joint_cols].values.astype(np.float32),
                    "future_raw": df.iloc[i+1 : i+1+self.future_steps][joint_cols].values.astype(np.float32),
                    "task": task_dir,
                    "frame_index": i,
                    # The first frame of a trajectory is the reset/context
                    # frame, so do not let obstacle mode start there.
                    "obst": is_obstacle and i > 0
                }
                samples.append(sample)

            if self.use_memory_image_input and missing_memory_count:
                if has_memory_folder:
                    print(
                        f"[WARN] {task_name}: {missing_memory_count} 个样本缺少 memory_image_four_channel，"
                        "训练时会用全零 memory 图补齐。"
                    )
                else:
                    print(
                        f"[WARN] {task_name}: 未找到 memory_image_four_channel，"
                        "该任务训练时会用全零 memory 图补齐。"
                    )

        print(f"总有效样本数：{len(samples)}")
        return samples

    def _interleave_samples(self, samples, seed):
        """Mix tasks without letting one long trajectory dominate each epoch block."""
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
        """Downsample the larger normal/obstacle bucket when both are present.

        This keeps training batches from being overwhelmed by whichever bucket
        happens to have many more samples after task-level splitting.
        """
        obst_samples = [sample for sample in samples if sample["obst"]]
        normal_samples = [sample for sample in samples if not sample["obst"]]
        if not obst_samples or not normal_samples:
            return samples

        keep_count = min(len(obst_samples), len(normal_samples))
        rng = random.Random(seed)
        rng.shuffle(obst_samples)
        rng.shuffle(normal_samples)
        return obst_samples[:keep_count] + normal_samples[:keep_count]

    def _split_by_task(self, train_ratio, seed):
        """Split by source task group to avoid frame-level leakage into validation."""
        task_to_obst = {}
        task_to_group = {}
        for sample in self.all_samples:
            task_to_obst[sample["task"]] = bool(sample["obst"])
            task_to_group[sample["task"]] = _source_group_key(sample["task"])

        grouped_tasks = {}
        for task, group in task_to_group.items():
            grouped_tasks.setdefault(group, set()).add(task)

        group_keys = sorted(grouped_tasks)
        rng = random.Random(seed)
        rng.shuffle(group_keys)
        if len(group_keys) <= 1:
            raise ValueError(
                f"需要至少 2 个 source group 才能严格划分 train/val；当前只有 {len(group_keys)} 个。"
            )

        split_idx = int(len(group_keys) * train_ratio)
        split_idx = max(1, min(len(group_keys) - 1, split_idx))
        train_groups = set(group_keys[:split_idx])
        val_groups = set(group_keys[split_idx:])

        train_tasks = {task for group in train_groups for task in grouped_tasks[group]}
        val_tasks = {task for group in val_groups for task in grouped_tasks[group]}

        if self.mode == "train":
            self.samples = [s for s in self.all_samples if s["task"] in train_tasks]
        else:
            self.samples = [s for s in self.all_samples if s["task"] in val_tasks]

        self.samples = self._balance_obstacle_ratio(
            self.samples,
            seed + (0 if self.mode == "train" else 1000),
        )
        self.samples = self._interleave_samples(self.samples, seed + (0 if self.mode == "train" else 1000))

        train_task_count = len(train_tasks)
        val_task_count = len(val_tasks)
        train_obst_count = sum(1 for task in train_tasks if task_to_obst[task])
        val_obst_count = sum(1 for task in val_tasks if task_to_obst[task])
        train_group_count = len(train_groups)
        val_group_count = len(val_groups)
        sample_obst_count = sum(1 for sample in self.samples if sample["obst"])
        print(
            f"Task split | train={train_task_count} (obst={train_obst_count}, normal={train_task_count - train_obst_count}) "
            f"| val={val_task_count} (obst={val_obst_count}, normal={val_task_count - val_obst_count})"
        )
        print(
            f"Source groups | train={train_group_count} | val={val_group_count} | "
            f"{self.mode} samples obst={sample_obst_count}, normal={len(self.samples) - sample_obst_count}"
        )

        if not self.samples:
            raise ValueError(f"{self.mode.upper()} split 为空，请检查任务数量、train_ratio 和障碍/普通轨迹分布。")

        print(f"{self.mode.upper()} 集样本数：{len(self.samples)}")

    def _split_by_task_strict(self, train_ratio, seed):
        """Split normal/obstacle source groups separately.

        A single source trajectory may have both `task_xxx` and `task_obst_xxx`.
        Grouping by source key prevents train/val leakage across those paired
        variants, and splitting normal/obstacle groups separately reduces the
        chance that one split ends up with only one category.
        """
        task_to_obst = {}
        task_to_group = {}
        for sample in self.all_samples:
            task_to_obst[sample["task"]] = bool(sample["obst"])
            task_to_group[sample["task"]] = _source_group_key(sample["task"])

        grouped_tasks = {}
        for task, group in task_to_group.items():
            grouped_tasks.setdefault(group, set()).add(task)

        normal_group_keys = sorted(
            group for group, tasks in grouped_tasks.items()
            if any(not task_to_obst[task] for task in tasks)
        )
        obstacle_group_keys = sorted(
            group for group, tasks in grouped_tasks.items()
            if any(task_to_obst[task] for task in tasks)
        )

        rng = random.Random(seed)
        rng.shuffle(normal_group_keys)
        train_groups, val_groups = _split_group_keys(normal_group_keys, train_ratio, "normal")

        rng.shuffle(obstacle_group_keys)
        if obstacle_group_keys:
            obstacle_train_groups, obstacle_val_groups = _split_group_keys(
                obstacle_group_keys,
                train_ratio,
                "obstacle",
            )
            train_groups |= obstacle_train_groups
            val_groups |= obstacle_val_groups

        train_tasks = {task for group in train_groups for task in grouped_tasks[group]}
        val_tasks = {task for group in val_groups for task in grouped_tasks[group]}

        if self.mode == "train":
            self.samples = [s for s in self.all_samples if s["task"] in train_tasks]
        else:
            self.samples = [s for s in self.all_samples if s["task"] in val_tasks]

        self.samples = self._balance_obstacle_ratio(
            self.samples,
            seed + (0 if self.mode == "train" else 1000),
        )
        self.samples = self._interleave_samples(self.samples, seed + (0 if self.mode == "train" else 1000))

        train_task_count = len(train_tasks)
        val_task_count = len(val_tasks)
        train_obst_count = sum(1 for task in train_tasks if task_to_obst[task])
        val_obst_count = sum(1 for task in val_tasks if task_to_obst[task])
        sample_obst_count = sum(1 for sample in self.samples if sample["obst"])
        self.has_obstacle_groups = bool(obstacle_group_keys)
        self.split_has_obstacle_samples = sample_obst_count > 0

        print(
            f"Task split | train={train_task_count} (obst={train_obst_count}, normal={train_task_count - train_obst_count}) "
            f"| val={val_task_count} (obst={val_obst_count}, normal={val_task_count - val_obst_count})"
        )
        print(
            f"Source groups | train={len(train_groups)} | val={len(val_groups)} | "
            f"{self.mode} samples obst={sample_obst_count}, normal={len(self.samples) - sample_obst_count}"
        )

        if not self.samples:
            raise ValueError(f"{self.mode.upper()} split is empty. Check task count, train_ratio, and obstacle distribution.")

        print(f"{self.mode.upper()} samples: {len(self.samples)}")

    def _prepare_model_inputs_and_targets(self):
        """Populate model-facing qpos/action tensors from raw trajectory values."""
        if self.normalize_joints:
            jmin = self.joint_min_max["min"]
            jrng = self.joint_min_max["rng"]
        for s in self.samples:
            curr_raw = s["curr_raw"]
            future_raw = s["future_raw"]
            if self.normalize_joints:
                s["curr"] = (curr_raw - jmin) / jrng
            else:
                s["curr"] = curr_raw.copy()

            if self.target_mode == "delta":
                step_delta = np.empty_like(future_raw)
                step_delta[0] = future_raw[0] - curr_raw
                if len(future_raw) > 1:
                    step_delta[1:] = future_raw[1:] - future_raw[:-1]
                s["future"] = step_delta / self.delta_qpos_scale
            elif self.normalize_joints:
                s["future"] = (future_raw - jmin.reshape(1, -1)) / jrng.reshape(1, -1)
            else:
                s["future"] = future_raw.copy()

    def _validate_joint_ranges(self):
        """Fail fast if data falls outside the fixed physical joint limits."""
        jmin = self.joint_min_max["min"]
        jmax = self.joint_min_max["max"]
        bad_message = _format_joint_range_errors(self.samples, jmin, jmax)
        if bad_message:
            raise ValueError(
                "states_filtered.csv contains joint values outside the fixed physical limits. "
                f"Limits min={jmin.tolist()} max={jmax.tolist()}. "
                f"Examples: {bad_message}. "
                "Fix/remove those task folders or adjust the agreed physical limits before training."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img_np = read_bgra_image(s["img_path"])
        if self.use_memory_image_input and s["has_mem_img"]:
            m_img_np = read_bgra_image(s["mem_img_path"])
        else:
            m_img_np = np.zeros_like(img_np)

        if self.mode == "train":
            # Geometric augmentation must be shared between the main image and
            # memory image so their spatial correspondence stays valid.
            angle, scale, tx, ty = _sample_shared_affine()
            img_np = _warp_bgra(img_np, angle, scale, tx, ty)
            m_img_np = _warp_bgra(m_img_np, angle, scale, tx, ty)
            img_np = _augment_rgb_channels(img_np)

        img_np = select_image_channels(img_np, self.image_channels)
        m_img_np = select_image_channels(m_img_np, self.image_channels)

        img = torch.from_numpy(img_np.copy()).permute(2, 0, 1)
        m_img = torch.from_numpy(m_img_np.copy()).permute(2, 0, 1)

        curr = torch.from_numpy(s["curr"]).float()
        future = torch.from_numpy(s["future"]).float()
        obst = torch.tensor([s["obst"]], dtype=torch.bool)

        return img, curr, future, m_img, obst

# Convenience wrapper used by training.py.
def get_data_loaders(
    data_root,
    future_steps=10,
    use_memory_image_input=False,
    batch_size=8,
    num_workers=0,
    image_channels=4,
    target_mode="absolute",
    delta_qpos_scale=10.0,
):
    """Create train/val dataloaders with the same alignment and normalization rules."""
    train_dataset = ImitationDataset(
        data_root,
        future_steps,
        use_memory_image_input=use_memory_image_input,
        mode="train",
        image_channels=image_channels,
        target_mode=target_mode,
        delta_qpos_scale=delta_qpos_scale,
    )
    val_dataset = ImitationDataset(
        data_root,
        future_steps,
        use_memory_image_input=use_memory_image_input,
        mode="val",
        joint_min_max=train_dataset.joint_min_max,
        image_channels=image_channels,
        target_mode=target_mode,
        delta_qpos_scale=delta_qpos_scale,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("\n数据加载器创建完成！")
    return train_loader, val_loader

