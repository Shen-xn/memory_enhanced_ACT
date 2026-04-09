from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)

from .me_block_config import ImportanceModelConfig, MEBlockConfig, MemoryUpdateConfig, default_me_block_config


def _channel_tensor(values, device, dtype):
    return torch.tensor(values, device=device, dtype=dtype).view(1, -1, 1, 1)


def normalize_image_channels(images: torch.Tensor, config: ImportanceModelConfig) -> torch.Tensor:
    mean = _channel_tensor(config.image_mean[: images.shape[1]], images.device, images.dtype)
    std = _channel_tensor(config.image_std[: images.shape[1]], images.device, images.dtype)
    return (images - mean) / std

def select_segmentation_input(images: torch.Tensor, config: ImportanceModelConfig) -> torch.Tensor:
    expected_channels = int(config.segmentation_input_channels)
    if images.shape[1] < expected_channels:
        raise ValueError(
            f"Expected at least {expected_channels} channels for segmentation input, got {images.shape[1]}."
        )
    return images[:, :expected_channels]


def _adapt_conv_to_input_channels(old_conv: nn.Conv2d, input_channels: int) -> nn.Conv2d:
    if old_conv.in_channels == input_channels:
        return old_conv

    new_conv = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        if input_channels <= old_conv.in_channels:
            new_conv.weight.copy_(old_conv.weight[:, :input_channels])
        else:
            new_conv.weight[:, : old_conv.in_channels].copy_(old_conv.weight)
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            for channel_idx in range(old_conv.in_channels, input_channels):
                new_conv.weight[:, channel_idx : channel_idx + 1].copy_(mean_weight)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    return new_conv


def _resolve_weights(weight_enum, pretrained_backbone: bool, name: str):
    if not pretrained_backbone:
        return None
    try:
        return weight_enum.DEFAULT
    except Exception as exc:
        warnings.warn(f"Failed to resolve {name} pretrained weights: {exc}. Falling back to random init.")
        return None


def _load_backbone(builder, weights, name: str):
    try:
        return builder(weights=weights)
    except Exception as exc:
        if weights is None:
            raise
        warnings.warn(f"Failed to load {name} pretrained weights: {exc}. Falling back to random init.")
        return builder(weights=None)


class TruncatedResNet18Layer2Segmentation(nn.Module):
    def __init__(self, config: ImportanceModelConfig):
        super().__init__()
        weights = _resolve_weights(ResNet18_Weights, config.pretrained_backbone, "ResNet18")
        backbone = _load_backbone(resnet18, weights, "ResNet18")

        backbone.conv1 = _adapt_conv_to_input_channels(
            backbone.conv1,
            config.segmentation_input_channels,
        )
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, config.num_output_classes, kernel_size=1),
        )

    def forward(self, images: torch.Tensor, config: ImportanceModelConfig) -> torch.Tensor:
        normalized = normalize_image_channels(images, config)
        features = self.stem(normalized)
        features = self.layer1(features)
        features = self.layer2(features)
        logits = self.head(features)
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


class ImportanceSegmentationModel(nn.Module):
    def __init__(self, config: ImportanceModelConfig):
        super().__init__()
        self.config = config
        if config.model_name != "truncated_resnet18_layer2":
            raise ValueError(
                f"Unsupported importance model: {config.model_name}. "
                "Current code only keeps truncated_resnet18_layer2."
            )
        self.model = TruncatedResNet18Layer2Segmentation(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        segmentation_images = select_segmentation_input(images, self.config)
        return self.model(segmentation_images, self.config)


@dataclass
class MemoryStepResult:
    memory_image: torch.Tensor
    memory_state: torch.Tensor
    score_state: torch.Tensor
    importance_score: torch.Tensor
    write_mask: torch.Tensor
    output_mask: torch.Tensor
    class_probs: torch.Tensor


class MemoryImageUpdater(nn.Module):
    def __init__(self, importance_config: ImportanceModelConfig, memory_config: MemoryUpdateConfig):
        super().__init__()
        self.importance_config = importance_config
        self.memory_config = memory_config

        weights = importance_config.normalized_class_weights()
        ordered = [weights[name] for name in importance_config.class_names]
        self.register_buffer("class_weights", torch.tensor(ordered, dtype=torch.float32).view(1, -1, 1, 1))

    def compute_importance_score(
        self,
        class_probs: torch.Tensor,
        background_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if class_probs.size(1) != self.class_weights.size(1):
            raise ValueError(
                f"Expected {self.class_weights.size(1)} class probability maps, got {class_probs.size(1)}."
            )
        importance_score = torch.sum(
            class_probs * self.class_weights.to(class_probs.device, class_probs.dtype),
            dim=1,
            keepdim=True,
        )
        if background_prob is not None:
            all_probs = torch.cat([background_prob.to(class_probs.dtype), class_probs], dim=1)
            max_prob_class = torch.argmax(all_probs, dim=1, keepdim=True)
            foreground_mask = (max_prob_class != 0).to(class_probs.dtype)
            importance_score = importance_score * foreground_mask
        return importance_score

    def _top_fraction_mask(self, scores: torch.Tensor) -> torch.Tensor:
        keep_ratio = float(self.memory_config.keep_top_ratio)
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

    def step(
        self,
        current_image: torch.Tensor,
        class_probs: torch.Tensor,
        background_prob: Optional[torch.Tensor] = None,
        prev_memory: Optional[torch.Tensor] = None,
        prev_scores: Optional[torch.Tensor] = None,
    ) -> MemoryStepResult:
        if current_image.dim() != 4:
            raise ValueError(f"Expected current_image shape [B, C, H, W], got {tuple(current_image.shape)}.")

        batch_size, channels, height, width = current_image.shape
        if channels != self.importance_config.input_channels:
            raise ValueError(
                f"Expected {self.importance_config.input_channels} channels, got {channels}."
            )

        if prev_memory is None:
            prev_memory = torch.zeros_like(current_image)
        if prev_scores is None:
            prev_scores = torch.zeros((batch_size, 1, height, width), device=current_image.device, dtype=current_image.dtype)

        importance_score = self.compute_importance_score(class_probs, background_prob=background_prob).to(current_image.dtype)
        decayed_scores = prev_scores * float(self.memory_config.score_decay)
        write_mask = importance_score > (decayed_scores + float(self.memory_config.tau_up))

        memory_state = torch.where(write_mask.expand_as(current_image), current_image, prev_memory)
        score_state = torch.where(write_mask, importance_score, decayed_scores)

        output_mask = self._top_fraction_mask(score_state)
        memory_image = memory_state * output_mask.to(current_image.dtype)

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
