"""ACT data loading with fixed joint limits and OpenCV BGRA images."""

import glob
import os
import random

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

            if self.strict_alignment and len(img_paths) != len(df):
                print(f"{task_name} 图像与轨迹帧数不匹配，跳过")
                continue

            n_frames = min(len(img_paths), len(df))
            img_paths = img_paths[:n_frames]
            df = df.head(n_frames).reset_index(drop=True)

            if df.isnull().any().any():
                df = df.dropna().reset_index(drop=True)
                n_frames = len(df)
                img_paths = img_paths[:n_frames]

            joint_cols = ["j1", "j2", "j3", "j4", "j5", "j10"]
            if not all(c in df.columns for c in joint_cols):
                print(f"{task_name} 缺少关节列，跳过")
                continue

            max_idx = n_frames - self.future_steps
            if max_idx <= 0:
                print(f"{task_name} 帧数不足，跳过")
                continue

            is_obstacle = "obst" in task_name.lower()

            for i in range(max_idx):
                # --------------------- 拼接 memory 图像路径（严格同编号） ---------------------
                mem_img_path = ""
                if has_memory_folder:
                    frame_filename = f"{i:06d}.png"  # 按你的命名规则：000000.png
                    mem_img_path = os.path.join(mem_img_dir, frame_filename)

                sample = {
                    "img_path": img_paths[i],
                    "mem_img_path": mem_img_path,
                    "has_mem_img": os.path.exists(mem_img_path),
                    "curr": df.iloc[i][joint_cols].values.astype(np.float32),
                    "future": df.iloc[i+1 : i+1+self.future_steps][joint_cols].values.astype(np.float32),
                    "task": task_dir,
                    "obst": is_obstacle
                }
                samples.append(sample)

        print(f"总有效样本数：{len(samples)}")
        return samples

    def _split_grouped_tasks(self, task_paths, train_ratio, seed):
        task_paths = sorted(task_paths)
        if len(task_paths) <= 1:
            return set(task_paths), set()

        shuffled = task_paths[:]
        random.Random(seed).shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        split_idx = max(1, min(len(shuffled) - 1, split_idx))
        return set(shuffled[:split_idx]), set(shuffled[split_idx:])

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

    def _split_by_task(self, train_ratio, seed):
        task_to_obst = {}
        for sample in self.all_samples:
            task_to_obst[sample["task"]] = bool(sample["obst"])

        obst_tasks = [task for task, is_obst in task_to_obst.items() if is_obst]
        normal_tasks = [task for task, is_obst in task_to_obst.items() if not is_obst]

        train_obst, val_obst = self._split_grouped_tasks(obst_tasks, train_ratio, seed)
        train_normal, val_normal = self._split_grouped_tasks(normal_tasks, train_ratio, seed + 1)
        train_tasks = train_obst | train_normal
        val_tasks = val_obst | val_normal

        if self.mode == "train":
            self.samples = [s for s in self.all_samples if s["task"] in train_tasks]
        else:
            self.samples = [s for s in self.all_samples if s["task"] in val_tasks]

        self.samples = self._interleave_samples(self.samples, seed + (0 if self.mode == "train" else 1000))

        train_task_count = len(train_tasks)
        val_task_count = len(val_tasks)
        train_obst_count = len(train_obst)
        val_obst_count = len(val_obst)
        print(
            f"Task split | train={train_task_count} (obst={train_obst_count}, normal={train_task_count - train_obst_count}) "
            f"| val={val_task_count} (obst={val_obst_count}, normal={val_task_count - val_obst_count})"
        )

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
