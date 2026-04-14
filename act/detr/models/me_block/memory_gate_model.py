"""Segmentation and recurrent memory-image logic for me_block.

The segmentation model predicts background + foreground classes for each
frame. MemoryImageUpdater then keeps a separate score/image state per
foreground class and renders one four-channel memory image for ACT.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .me_block_config import ImportanceModelConfig, MEBlockConfig, MemoryUpdateConfig, default_me_block_config


def _channel_tensor(values, device, dtype):
    return torch.tensor(values, device=device, dtype=dtype).view(1, -1, 1, 1)


def normalize_image_channels(images: torch.Tensor, config: ImportanceModelConfig) -> torch.Tensor:
    """Normalize the selected model input channels with checkpoint config values."""
    mean = _channel_tensor(config.image_mean[: images.shape[1]], images.device, images.dtype)
    std = _channel_tensor(config.image_std[: images.shape[1]], images.device, images.dtype)
    return (images - mean) / std


def select_segmentation_input(images: torch.Tensor, config: ImportanceModelConfig) -> torch.Tensor:
    """Select the channels consumed by segmentation while preserving full input for memory."""
    expected_channels = int(config.segmentation_input_channels)
    if images.shape[1] < expected_channels:
        raise ValueError(
            f"Expected at least {expected_channels} channels for segmentation input, got {images.shape[1]}."
        )
    return images[:, :expected_channels]


class ConvBNAct(nn.Module):
    """Reusable 3x3 conv block for the scratch compact FPN."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Small residual block used by the non-pretrained FPN encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class CompactFPNV1Segmentation(nn.Module):
    """Scratch FPN tuned for small, contiguous target/goal/arm masks.

    The encoder reaches stride 8 like the old truncated ResNet18 path, while
    the decoder fuses stride-4 and stride-2 features before the final logits.
    This keeps the parameter count close to the old model but gives small
    objects and boundaries a higher-resolution path.
    """

    def __init__(self, config: ImportanceModelConfig):
        super().__init__()
        in_channels = int(config.segmentation_input_channels)
        fpn_channels = 64
        self.stem = ConvBNAct(in_channels, 32, stride=2)
        self.stage1 = ResidualBlock(32, 32, stride=1)
        self.stage2 = ResidualBlock(32, 64, stride=2)
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )

        self.lateral1 = nn.Conv2d(32, fpn_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(64, fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(128, fpn_channels, kernel_size=1)
        self.refine2 = ConvBNAct(fpn_channels, fpn_channels)
        self.refine1 = ConvBNAct(fpn_channels, fpn_channels)
        self.head = nn.Sequential(
            ConvBNAct(fpn_channels, 32),
            nn.Conv2d(32, config.num_output_classes, kernel_size=1),
        )

    def forward(self, images: torch.Tensor, config: ImportanceModelConfig) -> torch.Tensor:
        normalized = normalize_image_channels(images, config)
        feat1 = self.stage1(self.stem(normalized))
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)

        pyramid = self.lateral3(feat3)
        pyramid = F.interpolate(pyramid, size=feat2.shape[-2:], mode="bilinear", align_corners=False)
        pyramid = self.refine2(pyramid + self.lateral2(feat2))
        pyramid = F.interpolate(pyramid, size=feat1.shape[-2:], mode="bilinear", align_corners=False)
        pyramid = self.refine1(pyramid + self.lateral1(feat1))

        logits = self.head(pyramid)
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


class ImportanceSegmentationModel(nn.Module):
    """Thin wrapper that owns the architecture choice and input-channel selection."""

    def __init__(self, config: ImportanceModelConfig):
        super().__init__()
        self.config = config
        if config.model_name != "compact_fpn_v1":
            raise ValueError(
                f"Unsupported importance model: {config.model_name}. Supported model: compact_fpn_v1."
            )
        self.model = CompactFPNV1Segmentation(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        segmentation_images = select_segmentation_input(images, self.config)
        return self.model(segmentation_images, self.config)


@dataclass
class MemoryStepResult:
    """All tensors produced by one recurrent memory update step."""

    memory_image: torch.Tensor
    memory_state: torch.Tensor
    score_state: torch.Tensor
    importance_score: torch.Tensor
    write_mask: torch.Tensor
    output_mask: torch.Tensor
    class_probs: torch.Tensor


class MemoryImageUpdater(nn.Module):
    """Deterministic class-wise memory update used offline and in deployment."""

    def __init__(self, importance_config: ImportanceModelConfig, memory_config: MemoryUpdateConfig):
        super().__init__()
        self.importance_config = importance_config
        self.memory_config = memory_config

    @property
    def num_classes(self) -> int:
        return len(self.importance_config.class_names)

    def _top_fraction_mask(self, scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """Keep the highest-scoring fraction of pixels for one foreground class."""
        if keep_ratio <= 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        if keep_ratio >= 1:
            return torch.ones_like(scores, dtype=torch.bool)

        batch_size, _, height, width = scores.shape
        flat_scores = scores.view(batch_size, -1)
        keep_count = max(1, math.ceil(keep_ratio * flat_scores.shape[1]))
        topk_indices = flat_scores.topk(keep_count, dim=1).indices
        flat_mask = torch.zeros_like(flat_scores, dtype=torch.float32).scatter(1, topk_indices, 1.0).to(torch.bool)
        return flat_mask.view(batch_size, 1, height, width)

    def _blur_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Suppress isolated hot pixels before memory writing and top-ratio masks."""
        return F.avg_pool2d(scores, kernel_size=5, stride=1, padding=2)

    def step(
        self,
        current_image: torch.Tensor,
        class_probs: torch.Tensor,
        background_prob: Optional[torch.Tensor] = None,
        prev_memory: Optional[torch.Tensor] = None,
        prev_scores: Optional[torch.Tensor] = None,
    ) -> MemoryStepResult:
        """Update memory state from current image and foreground probabilities."""
        if current_image.dim() != 4:
            raise ValueError(f"Expected current_image shape [B, C, H, W], got {tuple(current_image.shape)}.")

        batch_size, channels, height, width = current_image.shape
        if channels != self.importance_config.input_channels:
            raise ValueError(
                f"Expected {self.importance_config.input_channels} channels, got {channels}."
            )

        if prev_memory is None:
            prev_memory = torch.zeros(
                batch_size,
                self.num_classes,
                channels,
                height,
                width,
                device=current_image.device,
                dtype=current_image.dtype,
            )
        elif prev_memory.dim() != 5 or prev_memory.size(1) != self.num_classes or prev_memory.size(2) != channels:
            raise ValueError(
                f"Expected prev_memory shape [B, {self.num_classes}, {channels}, H, W], got {tuple(prev_memory.shape)}."
            )
        else:
            prev_memory = prev_memory.to(device=current_image.device, dtype=current_image.dtype)

        if prev_scores is None:
            prev_scores = torch.zeros(
                batch_size,
                self.num_classes,
                height,
                width,
                device=current_image.device,
                dtype=current_image.dtype,
            )
        elif prev_scores.dim() != 4 or prev_scores.size(1) != self.num_classes:
            raise ValueError(
                f"Expected prev_scores shape [B, {self.num_classes}, H, W], got {tuple(prev_scores.shape)}."
            )
        else:
            prev_scores = prev_scores.to(device=current_image.device, dtype=current_image.dtype)

        candidate_scores = class_probs.to(current_image.dtype)
        if background_prob is not None:
            candidate_scores = candidate_scores
        candidate_scores = self._blur_scores(candidate_scores)
        decayed_scores = prev_scores * float(self.memory_config.score_decay)
        write_mask_per_class = candidate_scores > (decayed_scores + float(self.memory_config.tau_up))

        expanded_current = current_image.unsqueeze(1).expand(-1, self.num_classes, -1, -1, -1)
        expanded_write = write_mask_per_class.unsqueeze(2).expand_as(expanded_current)
        memory_state = torch.where(expanded_write, expanded_current, prev_memory)
        score_state = torch.where(write_mask_per_class, candidate_scores, decayed_scores)

        occupied = torch.zeros((batch_size, 1, height, width), device=current_image.device, dtype=torch.bool)
        memory_image = torch.zeros_like(current_image)

        keep_ratios = [
            float(self.memory_config.keep_top_ratio_target),
            float(self.memory_config.keep_top_ratio_goal),
            float(self.memory_config.keep_top_ratio_arm),
        ]
        # Render classes in config order. `occupied` prevents later classes from
        # overwriting pixels already claimed by earlier, higher-priority classes.
        for class_idx in range(self.num_classes):
            keep_ratio = keep_ratios[class_idx] if class_idx < len(keep_ratios) else keep_ratios[-1]
            class_mask = self._top_fraction_mask(score_state[:, class_idx : class_idx + 1], keep_ratio) & (~occupied)
            occupied |= class_mask
            memory_image = torch.where(
                class_mask.expand_as(current_image),
                memory_state[:, class_idx],
                memory_image,
            )

        output_mask = occupied
        write_mask = torch.any(write_mask_per_class, dim=1, keepdim=True)
        importance_score = torch.max(candidate_scores, dim=1, keepdim=True).values

        return MemoryStepResult(
            memory_image=memory_image,
            memory_state=memory_state,
            score_state=score_state,
            importance_score=importance_score,
            write_mask=write_mask,
            output_mask=output_mask,
            class_probs=class_probs,
        )


class ImportanceMemoryModel(nn.Module):
    """
    Offline memory-image model.

    The intended usage is:
    1. train the segmentation head on `rgb -> importance_labels`
    2. freeze the model
    3. iterate through each task frame sequence to generate `memory_image_four_channel`
    """

    def __init__(self, config: Optional[MEBlockConfig] = None):
        super().__init__()
        self.config = config or default_me_block_config()
        self.segmenter = ImportanceSegmentationModel(self.config.importance)
        self.updater = MemoryImageUpdater(self.config.importance, self.config.memory)

    def forward(
        self,
        current_image: torch.Tensor,
        prev_memory: Optional[torch.Tensor] = None,
        prev_scores: Optional[torch.Tensor] = None,
    ) -> MemoryStepResult:
        """Predict segmentation probabilities and apply one memory update."""
        logits = self.segmenter(current_image)
        probs = F.softmax(logits, dim=1)
        background_prob = probs[:, :1]
        class_probs = probs[:, 1:]
        return self.updater.step(
            current_image=current_image,
            class_probs=class_probs,
            background_prob=background_prob,
            prev_memory=prev_memory,
            prev_scores=prev_scores,
        )


def build_importance_memory_model(config: Optional[MEBlockConfig] = None) -> ImportanceMemoryModel:
    return ImportanceMemoryModel(config=config)


def checkpoint_payload(model: ImportanceMemoryModel) -> Dict:
    return {
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
    }
