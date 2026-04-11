from __future__ import annotations

import glob
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .me_block_config import ImportanceModelConfig, ImportanceTrainingConfig


SPECIAL_TASK_NAME = "special_data"


def resolve_task_image_dir(task_dir: str, image_dirname: str = "auto") -> str:
    task_name = os.path.basename(os.path.normpath(task_dir))
    four_channel_dir = os.path.join(task_dir, "four_channel")
    rgb_dir = os.path.join(task_dir, "rgb")

    if image_dirname == "auto":
        if task_name == SPECIAL_TASK_NAME and os.path.isdir(rgb_dir):
            return "rgb"
        if os.path.isdir(four_channel_dir):
            return "four_channel"
        if os.path.isdir(rgb_dir):
            return "rgb"
        return ""

    if image_dirname == "four_channel":
        if os.path.isdir(four_channel_dir):
            return "four_channel"
        if task_name == SPECIAL_TASK_NAME and os.path.isdir(rgb_dir):
            return "rgb"
        return ""

    if image_dirname == "rgb" and os.path.isdir(rgb_dir):
        return "rgb"
    return ""


def list_task_dirs(data_root: str, image_dirname: str = "auto") -> List[str]:
    candidates = []
    candidates.extend(sorted(glob.glob(os.path.join(data_root, "task*"))))
    special_dir = os.path.join(data_root, "special_data")
    if os.path.isdir(special_dir):
        candidates.append(special_dir)
    return [
        path
        for path in candidates
        if os.path.isdir(path)
        and "task_copy" not in path
        and resolve_task_image_dir(path, image_dirname)
    ]


def list_image_files(image_dir: str) -> List[str]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")
    return sorted({path for pattern in patterns for path in glob.glob(os.path.join(image_dir, pattern))})


def read_four_channel(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError(f"Expected a 4-channel PNG, got shape {image.shape} from {path}")
    return image


def read_rgb_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Expected a 3-channel image, got shape {image.shape} from {path}")
    return image


def read_model_input(path: str, image_dirname: str, input_channels: int) -> np.ndarray:
    if image_dirname == "four_channel":
        image = read_four_channel(path)
    else:
        image = read_rgb_image(path)

    if image.shape[2] == input_channels:
        return image
    if image.shape[2] > input_channels:
        return image[:, :, :input_channels]
    if image.shape[2] == 3 and input_channels == 4:
        depth = np.zeros((image.shape[0], image.shape[1], 1), dtype=image.dtype)
        return np.concatenate([image, depth], axis=2)
    raise RuntimeError(
        f"Cannot adapt image with {image.shape[2]} channels to {input_channels} channels for {path}."
    )


def read_label(path: str) -> np.ndarray:
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Failed to read label: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


def _sample_affine(config: ImportanceTrainingConfig, width: int, height: int) -> np.ndarray:
    center = (width / 2.0, height / 2.0)
    angle = random.uniform(-config.rotation_deg, config.rotation_deg)
    scale = random.uniform(config.scale_min, config.scale_max)
    tx = random.uniform(-config.translation_px, config.translation_px)
    ty = random.uniform(-config.translation_px, config.translation_px)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return matrix


def _warp_image(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    border_value = tuple([0] * image.shape[2]) if image.ndim == 3 else 0
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _warp_label(label: np.ndarray, matrix: np.ndarray, unlabeled_index: int) -> np.ndarray:
    height, width = label.shape[:2]
    return cv2.warpAffine(
        label,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(unlabeled_index),
    )


def augment_sample(
    image: np.ndarray,
    label: np.ndarray,
    config: ImportanceTrainingConfig,
    unlabeled_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not config.use_augmentation:
        return image, label

    if random.random() < config.horizontal_flip_prob:
        image = np.ascontiguousarray(image[:, ::-1])
        label = np.ascontiguousarray(label[:, ::-1])

    matrix = _sample_affine(config, image.shape[1], image.shape[0])
    image = _warp_image(image, matrix)
    label = _warp_label(label, matrix, unlabeled_index)

    rgb = np.clip(image[..., :3], 0.0, 1.0)
    gamma = random.uniform(config.gamma_min, config.gamma_max)
    rgb = np.power(rgb, gamma).astype(np.float32)

    if config.noise_std > 0:
        noise = np.random.normal(0.0, config.noise_std, size=rgb.shape).astype(np.float32)
        rgb = rgb + noise

    augmented = image.copy()
    augmented[..., :3] = np.clip(rgb, 0.0, 1.0)
    return augmented, label


class ImportanceFrameDataset(Dataset):
    def __init__(
        self,
        config: ImportanceTrainingConfig,
        model_config: ImportanceModelConfig,
        split: str,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        self.config = config
        self.model_config = model_config
        self.split = split
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        task_dirs = list_task_dirs(self.config.data_root, image_dirname=self.config.image_dirname)
        if not task_dirs:
            raise FileNotFoundError(f"No task folders found under {self.config.data_root}")

        eligible_tasks: List[str] = []
        task_matches: Dict[str, List[Tuple[str, str]]] = {}
        for task_dir in task_dirs:
            resolved_image_dirname = resolve_task_image_dir(task_dir, self.config.image_dirname)
            if not resolved_image_dirname:
                continue
            label_dir = os.path.join(task_dir, self.config.label_dirname)
            image_dir = os.path.join(task_dir, resolved_image_dirname)
            if not os.path.isdir(label_dir):
                continue
            if not os.path.isdir(image_dir):
                continue
            image_paths = list_image_files(image_dir)
            label_paths = sorted(glob.glob(os.path.join(label_dir, "*.png")))
            image_map = {os.path.splitext(os.path.basename(path))[0]: path for path in image_paths}
            label_map = {os.path.splitext(os.path.basename(path))[0]: path for path in label_paths}
            shared_names = sorted(set(image_map.keys()) & set(label_map.keys()))
            if shared_names:
                eligible_tasks.append(task_dir)
                task_matches[task_dir] = [
                    (image_map[name], label_map[name], resolved_image_dirname)
                    for name in shared_names
                ]

        if not eligible_tasks:
            raise FileNotFoundError(
                f"No tasks with both image inputs and `{self.config.label_dirname}` were found."
            )

        primary_tasks = [task_dir for task_dir in eligible_tasks if os.path.basename(task_dir).startswith("task")]
        auxiliary_tasks = [task_dir for task_dir in eligible_tasks if task_dir not in primary_tasks]

        rng = random.Random(self.config.seed)
        rng.shuffle(primary_tasks)
        split_index = max(1, int(len(primary_tasks) * self.config.train_ratio)) if primary_tasks else 0
        train_tasks = set(primary_tasks[:split_index])
        val_tasks = set(primary_tasks[split_index:])

        if self.split == "train":
            selected = train_tasks | set(auxiliary_tasks)
        else:
            selected = val_tasks
        if not selected:
            selected = train_tasks | set(auxiliary_tasks)

        samples = []
        for task_dir in eligible_tasks:
            if task_dir not in selected:
                continue
            for image_path, label_path, image_dirname in task_matches[task_dir]:
                samples.append(
                    {
                        "task": os.path.basename(task_dir),
                        "image_path": image_path,
                        "label_path": label_path,
                        "image_dirname": image_dirname,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = read_model_input(
            sample["image_path"],
            sample["image_dirname"],
            self.model_config.input_channels,
        ).astype(np.float32) / 255.0
        label = read_label(sample["label_path"])
        if self.split == "train":
            image, label = augment_sample(
                image,
                label,
                self.config,
                self.model_config.unlabeled_index,
            )

        allowed_values = {self.model_config.background_index, self.model_config.unlabeled_index}
        allowed_values.update(range(1, self.model_config.num_foreground_classes + 1))
        unique_values = set(int(v) for v in np.unique(label))
        invalid_values = sorted(unique_values - allowed_values)
        if invalid_values:
            raise ValueError(f"Unexpected label values {invalid_values} in {sample['label_path']}.")

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        label_tensor = torch.from_numpy(label).long()
        return image_tensor, label_tensor
