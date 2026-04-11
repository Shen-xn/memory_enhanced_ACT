"""ACT data loading with fixed joint limits and OpenCV BGRA images."""

import glob
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


FIXED_JOINT_MIN = np.array([0, 0, 0, 0, 0, 100], dtype=np.float32)
FIXED_JOINT_MAX = np.array([1000, 1000, 1000, 1000, 1000, 700], dtype=np.float32)
FIXED_JOINT_RNG = FIXED_JOINT_MAX - FIXED_JOINT_MIN


def get_fixed_joint_stats():
    return {
        "min": FIXED_JOINT_MIN.copy(),
        "max": FIXED_JOINT_MAX.copy(),
        "rng": FIXED_JOINT_RNG.copy(),
    }


def read_bgra_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected a 4-channel image, got shape {image.shape} from {path}")
    return image.astype(np.float32) / 255.0


def _preview_values(values, limit=8):
    values = sorted(values)
    shown = values[:limit]
    suffix = "..." if len(values) > limit else ""
    return f"{shown}{suffix}"


def _frame_id_from_path(path, fallback_index):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return int(stem)
    match = re.search(r"\d+", stem)
    if match:
        return int(match.group(0))
    return int(fallback_index)


def _index_image_paths_by_frame(paths):
    frame_to_path = {}
    duplicates = []
    for fallback_index, path in enumerate(paths):
        frame_id = _frame_id_from_path(path, fallback_index)
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


class ImitationDataset(Dataset):
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
        joint_min_max=None
    ):
        self.data_root = data_root
        self.future_steps = future_steps
        self.use_memory_image_input = use_memory_image_input
        self.mode = mode
        self.normalize_joints = normalize_joints
        self.strict_alignment = strict_alignment
        self.joint_min_max = joint_min_max
        self.seed = seed
        
        self.all_samples = self._load_all_samples()
        self._split_by_task(train_ratio, seed)

        if normalize_joints and self.joint_min_max is None:
            self.joint_min_max = get_fixed_joint_stats()
        
        if normalize_joints:
            self._normalize_joints()

    def _align_dataframe_and_images(self, task_name, img_paths, df, joint_cols):
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
                "请先运行 data_process_1.py 修正 rgb/depth/csv，再运行 data_process_2.py 重建 four_channel。"
            )

        keep_mask = [int(frame_id) in frame_to_image for frame_id in frame_ids]
        aligned_df = df.loc[keep_mask].reset_index(drop=True)
        aligned_frame_ids = frame_ids[keep_mask]
        aligned_img_paths = [frame_to_image[int(frame_id)] for frame_id in aligned_frame_ids]
        return aligned_img_paths, aligned_df

    def _load_all_samples(self):
        samples = []
        task_dirs = sorted(glob.glob(os.path.join(self.data_root, "task*")))
        task_dirs = [d for d in task_dirs if os.path.isdir(d) and "task_copy" not in d]

        print(f"找到 {len(task_dirs)} 个任务文件夹")

        for task_dir in task_dirs:
            task_name = os.path.basename(task_dir)
            img_dir = os.path.join(task_dir, "four_channel")
            csv_path = os.path.join(task_dir, "states_filtered.csv")

            # --------------------- 检查 memory 图像文件夹 ---------------------
            mem_img_dir = os.path.join(task_dir, "memory_image_four_channel")
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
                # --------------------- 拼接 memory 图像路径（严格同编号） ---------------------
                mem_img_path = ""
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
                    "curr": df.iloc[i][joint_cols].values.astype(np.float32),
                    "future": df.iloc[i+1 : i+1+self.future_steps][joint_cols].values.astype(np.float32),
                    "task": task_dir,
                    "obst": is_obstacle
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

    def _split_by_task(self, train_ratio, seed):
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

    def _normalize_joints(self):
        jmin = self.joint_min_max["min"]
        jrng = self.joint_min_max["rng"]
        for s in self.samples:
            s["curr"] = (s["curr"] - jmin) / jrng
            s["future"] = (s["future"] - jmin) / jrng

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
            angle, scale, tx, ty = _sample_shared_affine()
            img_np = _warp_bgra(img_np, angle, scale, tx, ty)
            m_img_np = _warp_bgra(m_img_np, angle, scale, tx, ty)
            img_np = _augment_rgb_channels(img_np)

        img = torch.from_numpy(img_np.copy()).permute(2, 0, 1)
        m_img = torch.from_numpy(m_img_np.copy()).permute(2, 0, 1)

        curr = torch.from_numpy(s["curr"]).float()
        future = torch.from_numpy(s["future"]).float()
        obst = torch.tensor([s["obst"]], dtype=torch.bool)

        return img, curr, future, m_img, obst

# ====================== 快速创建加载器 ======================
def get_data_loaders(
    data_root,
    future_steps=10,
    use_memory_image_input=False,
    batch_size=8,
    num_workers=0
):
    train_dataset = ImitationDataset(
        data_root,
        future_steps,
        use_memory_image_input=use_memory_image_input,
        mode="train",
    )
    val_dataset = ImitationDataset(
        data_root,
        future_steps,
        use_memory_image_input=use_memory_image_input,
        mode="val",
        joint_min_max=train_dataset.joint_min_max
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("\n数据加载器创建完成！")
    return train_loader, val_loader


# ====================== 测试 ======================
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    train_loader, val_loader = get_data_loaders(
        data_root=os.path.join(SCRIPT_DIR, "./data"),
        future_steps=10,
        batch_size=16,
        num_workers=0
    )

    print("\n======= 训练集测试 =======")
    for imgs, currs, futures, m_imgs, obsts in train_loader:
        print("主图像:", imgs.shape)
        print("当前关节:", currs.shape)
        print("未来动作:", futures.shape)
        print("Memory图像:", m_imgs.shape)
        print("障碍标签:", obsts)
        print("Memory 是否全零:", (torch.sum(m_imgs, (1, 2, 3)) == 0).unsqueeze(0).unsqueeze(2))
        break
        

    print("\n======= 验证集测试 =======")
    for imgs, currs, futures, m_imgs, obsts in val_loader:
        print("主图像:", imgs.shape)
        print("当前关节:", currs.shape)
        print("未来动作:", futures.shape)
        print("Memory图像:", m_imgs.shape)
        print("障碍标签:", obsts.shape)
        print("Memory 是否全零:", torch.all(m_imgs == 0))
        break
