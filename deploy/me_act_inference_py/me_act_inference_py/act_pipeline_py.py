#!/usr/bin/env python3
"""Python deployment pipeline for baseline ACT inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch


@dataclass
class DeployConfig:
    target_width: int = 640
    target_height: int = 480
    pad_left: int = 0
    pad_top: int = 40
    depth_clip_min: float = 0.0
    depth_clip_max: float = 800.0
    state_dim: int = 6
    num_queries: int = 10
    image_channels: int = 4


class ActPipelinePy:
    def __init__(self, deploy_dir: str, device: str = "cpu") -> None:
        self.deploy_dir = Path(deploy_dir)
        self.config = self._load_config(self.deploy_dir / "deploy_config.yml")
        self.device = self._parse_device(device)
        self.act_module = torch.jit.load(str(self.deploy_dir / "act_inference.pt"), map_location="cpu").eval()
        self.act_module.to(self.device)

    def reset_memory(self) -> None:
        return None

    def predict(self, bgr: np.ndarray, depth: np.ndarray, qpos: List[float]) -> List[List[float]]:
        four_channel = self.build_four_channel_image(bgr, depth, self.config)
        return self.predict_from_four_channel(four_channel, qpos)

    def predict_from_four_channel(self, four_channel_bgra: np.ndarray, qpos: List[float]) -> List[List[float]]:
        if four_channel_bgra is None or four_channel_bgra.size == 0 or four_channel_bgra.ndim != 3 or four_channel_bgra.shape[2] != 4:
            raise ValueError("PredictFromFourChannel expects a non-empty BGRA image.")
        image_tensor = self._mat_to_tensor(four_channel_bgra)
        qpos_tensor = self._qpos_to_tensor(qpos)
        with torch.no_grad():
            actions = self.act_module(qpos_tensor, image_tensor)
        return self._tensor_to_trajectory(actions.squeeze(0).to("cpu"))

    def build_debug_four_channel_image(self, bgr: np.ndarray, depth: np.ndarray) -> np.ndarray:
        return self.build_four_channel_image(bgr, depth, self.config)

    @staticmethod
    def _load_config(path: Path) -> DeployConfig:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Failed to open deploy config: {path}")
        cfg = DeployConfig()

        def read_value(key: str, default):
            node = fs.getNode(key)
            if node.empty():
                return default
            if isinstance(default, int):
                return int(node.real())
            return float(node.real())

        cfg.target_width = read_value("target_width", cfg.target_width)
        cfg.target_height = read_value("target_height", cfg.target_height)
        cfg.pad_left = read_value("pad_left", cfg.pad_left)
        cfg.pad_top = read_value("pad_top", cfg.pad_top)
        cfg.depth_clip_min = read_value("depth_clip_min", cfg.depth_clip_min)
        cfg.depth_clip_max = read_value("depth_clip_max", cfg.depth_clip_max)
        cfg.state_dim = read_value("state_dim", cfg.state_dim)
        cfg.num_queries = read_value("num_queries", cfg.num_queries)
        cfg.image_channels = read_value("image_channels", cfg.image_channels)
        fs.release()
        return cfg

    @staticmethod
    def _parse_device(device: str) -> torch.device:
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device='cuda' was requested, but CUDA is not available.")
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def normalize_depth(depth: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
        if depth is None or depth.size == 0 or depth.ndim != 2:
            raise ValueError("Depth image must be non-empty and single-channel.")
        depth_f32 = depth.astype(np.float32, copy=False)
        depth_f32 = np.clip(depth_f32, clip_min, clip_max)
        depth_f32 = (depth_f32 - clip_min) * (255.0 / (clip_max - clip_min))
        return depth_f32.astype(np.uint8)

    @staticmethod
    def build_four_channel_image(bgr: np.ndarray, depth: np.ndarray, config: DeployConfig) -> np.ndarray:
        if bgr is None or bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("BGR image must be non-empty and 3-channel.")
        if depth is None or depth.size == 0:
            raise ValueError("Depth image must be non-empty.")
        if depth.ndim == 3:
            if depth.shape[2] != 1:
                raise ValueError("Depth image must be single-channel.")
            depth = depth[:, :, 0]
        bgr_resized = cv2.resize(bgr, (config.target_width, config.target_height), interpolation=cv2.INTER_LINEAR)
        depth_u8 = ActPipelinePy.normalize_depth(depth, config.depth_clip_min, config.depth_clip_max)
        depth_aligned = np.zeros((config.target_height, config.target_width), dtype=np.uint8)
        copy_w = min(depth_u8.shape[1], config.target_width - config.pad_left)
        copy_h = min(depth_u8.shape[0], config.target_height - config.pad_top)
        if copy_w > 0 and copy_h > 0:
            depth_aligned[
                config.pad_top : config.pad_top + copy_h,
                config.pad_left : config.pad_left + copy_w,
            ] = depth_u8[:copy_h, :copy_w]
        channels = cv2.split(bgr_resized)
        return cv2.merge([channels[0], channels[1], channels[2], depth_aligned])

    def _mat_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).to(torch.float32).div_(255.0).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _qpos_to_tensor(self, qpos: List[float]) -> torch.Tensor:
        if len(qpos) != self.config.state_dim:
            raise ValueError("Unexpected qpos dimension.")
        tensor = torch.tensor(qpos, dtype=torch.float32).view(1, self.config.state_dim)
        return tensor.to(self.device)

    @staticmethod
    def _tensor_to_trajectory(tensor: torch.Tensor) -> List[List[float]]:
        return tensor.contiguous().tolist()
