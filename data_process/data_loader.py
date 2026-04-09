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


def load_bgra_tensor(path, apply_augmentation=False):
    """
    加载BGRA图像并转换为tensor
    如果apply_augmentation=True，对RGB通道应用随机gamma矫正和噪声
    """
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected a 4-channel image, got shape {image.shape} from {path}")
    
    # 转换为float32并归一化到[0, 1]
    image_float = image.astype(np.float32) / 255.0
    
    # 只在训练时应用数据增强
    if apply_augmentation:
        # 分离RGB和深度通道
        rgb = image_float[:, :, :3]  # BGR顺序
        depth = image_float[:, :, 3:]  # 深度通道
        
        # 1. 随机gamma矫正 (0.6 ~ 1.4)
        gamma = np.random.uniform(0.6, 1.4)
        rgb = np.power(rgb, gamma)
        
        # 2. 添加少量随机噪声 (高斯噪声，标准差0.01~0.03)
        noise_std = np.random.uniform(0.01, 0.03)
        noise = np.random.normal(0, noise_std, rgb.shape).astype(np.float32)
        rgb = rgb + noise
        
        # 裁剪到有效范围 [0, 1]
        rgb = np.clip(rgb, 0.0, 1.0)
        
        # 重新组合图像
        image_float = np.concatenate([rgb, depth], axis=2)
    
    # 转换为tensor并调整维度顺序
    tensor = torch.from_numpy(image_float.copy()).permute(2, 0, 1)
    return tensor


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

    def _split_by_task(self, train_ratio, seed):
        task_set = sorted(list({s["task"] for s in self.all_samples}))
        random.Random(seed).shuffle(task_set)
        split_idx = int(len(task_set) * train_ratio)
        train_tasks = set(task_set[:split_idx])

        if self.mode == "train":
            self.samples = [s for s in self.all_samples if s["task"] in train_tasks]
        else:
            self.samples = [s for s in self.all_samples if s["task"] not in train_tasks]

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

        # --------------------- 加载主图像 ---------------------
        # 只在训练时应用数据增强
        apply_aug = (self.mode == "train")
        img = load_bgra_tensor(s["img_path"], apply_augmentation=apply_aug)

        # --------------------- 加载 memory 图像（不存在则返回全零） ---------------------
        if self.use_memory_image_input and s["has_mem_img"]:
            # memory图像不应用数据增强，保持原始状态
            m_img = load_bgra_tensor(s["mem_img_path"], apply_augmentation=False)
        else:
            # 创建全零tensor，需要与img相同的形状
            m_img = torch.zeros_like(img)

        # --------------------- 关节 & 标签 ---------------------
        curr = torch.from_numpy(s["curr"]).float()
        future = torch.from_numpy(s["future"]).float()
        obst = torch.tensor([s["obst"]], dtype=torch.bool)

        # ====================== 现在返回 5 个值！======================
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
