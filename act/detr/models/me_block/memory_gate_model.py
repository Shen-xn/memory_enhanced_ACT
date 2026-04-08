from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

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


def _adapt_first_conv_to_input_channels(model: nn.Module, input_channels: int) -> None:
    first_block = model.backbone["0"]
    first_block[0] = _adapt_conv_to_input_channels(first_block[0], input_channels)


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


class LiteMobileNetV3SmallSegmentation(nn.Module):
    def __init__(self, config: ImportanceModelConfig):
        super().__init__()
        weights = _resolve_weights(MobileNet_V3_Small_Weights, config.pretrained_backbone, "MobileNetV3-Small")
        try:
            backbone = _load_backbone(mobilenet_v3_small, weights, "MobileNetV3-Small")
        except TypeError:
            backbone = mobilenet_v3_small(weights=None)

        backbone.features[0][0] = _adapt_conv_to_input_channels(
            backbone.features[0][0],
            config.segmentation_input_channels,
        )
        self.backbone = backbone.features
        self.classifier = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.Conv2d(128, config.num_output_classes, kernel_size=1),
        )

    def forward(self, images: torch.Tensor, config: ImportanceModelConfig) -> torch.Tensor:
        normalized = normalize_image_channels(images, config)
        features = self.backbone(normalized)
        logits = self.classifier(features)
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class UltraLiteSegmentationNet(nn.Module):
    def __init__(self, config: ImportanceModelConfig):
        super().__init__()
        if config.pretrained_backbone:
            warnings.warn(
                "Pretrained backbone is not available for ultralite_custom_v1. Using random initialization."
            )

        self.stem = nn.Sequential(
            nn.Conv2d(config.segmentation_input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(16, 24, stride=2),
            DepthwiseSeparableBlock(24, 24, stride=1),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(24, 32, stride=2),
            DepthwiseSeparableBlock(32, 32, stride=1),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(32, 48, stride=2),
            DepthwiseSeparableBlock(48, 48, stride=1),
        )
        self.context = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.reduce2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.lateral2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.lateral1 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
        )
        self.refine2 = DepthwiseSeparableBlock(32, 32, stride=1)
        self.reduce1 = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.refine1 = DepthwiseSeparableBlock(24, 24, stride=1)
        self.classifier = nn.Conv2d(24, config.num_output_classes, kernel_size=1)

    def forward(self, images: torch.Tensor, config: ImportanceModelConfig) -> torch.Tensor:
        normalized = normalize_image_channels(images, config)
        x = self.stem(normalized)
        feat_4 = self.stage1(x)
        feat_8 = self.stage2(feat_4)
        feat_16 = self.stage3(feat_8)

        context = self.context(feat_16)
        up_8 = F.interpolate(context, size=feat_8.shape[-2:], mode="bilinear", align_corners=False)
        up_8 = self.reduce2(up_8) + self.lateral2(feat_8)
        up_8 = self.refine2(up_8)

        up_4 = F.interpolate(up_8, size=feat_4.shape[-2:], mode="bilinear", align_corners=False)
        up_4 = self.reduce1(up_4) + self.lateral1(feat_4)
        up_4 = self.refine1(up_4)

        logits = self.classifier(up_4)
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


class TruncatedResNet18Segmentation(nn.Module):
    def __init__(self, config: ImportanceModelConfig, stop_stage: str):
        super().__init__()
        if stop_stage not in {"layer1", "layer2", "layer3"}:
            raise ValueError(f"Unsupported ResNet18 truncation stage: {stop_stage}")
        weights = _resolve_weights(ResNet18_Weights, config.pretrained_backbone, "ResNet18")
        backbone = _load_backbone(resnet18, weights, "ResNet18")

        backbone.conv1 = _adapt_conv_to_input_channels(
            backbone.conv1,
            config.segmentation_input_channels,
        )
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.stop_stage = stop_stage

        out_channels = {
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
        }[stop_stage]
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, config.num_output_classes, kernel_size=1),
        )

    def forward(self, images: torch.Tensor, config: ImportanceModelConfig) -> torch.Tensor:
        normalized = normalize_image_channels(images, config)
        features = self.stem(normalized)
        features = self.layer1(features)
        if self.stop_stage in {"layer2", "layer3"}:
            features = self.layer2(features)
        if self.stop_stage == "layer3":
            features = self.layer3(features)

        logits = self.head(features)
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


class ImportanceSegmentationModel(nn.Module):
    def __init__(self, config: ImportanceModelConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_name

        if self.model_name == "lraspp_mobilenet_v3_large":
            backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if config.pretrained_backbone else None
            self.model = lraspp_mobilenet_v3_large(
                weights=None,
                weights_backbone=backbone_weights,
                num_classes=config.num_output_classes,
            )
            _adapt_first_conv_to_input_channels(self.model, config.segmentation_input_channels)
        elif self.model_name == "lite_mobilenet_v3_small":
            self.model = LiteMobileNetV3SmallSegmentation(config)
        elif self.model_name == "truncated_resnet18_layer1":
            self.model = TruncatedResNet18Segmentation(config, stop_stage="layer1")
        elif self.model_name == "truncated_resnet18_layer2":
            self.model = TruncatedResNet18Segmentation(config, stop_stage="layer2")
        elif self.model_name == "truncated_resnet18_layer3":
            self.model = TruncatedResNet18Segmentation(config, stop_stage="layer3")
        elif self.model_name == "ultralite_custom_v1":
            self.model = UltraLiteSegmentationNet(config)
        else:
            raise ValueError(f"Unsupported importance model: {self.model_name}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        segmentation_images = select_segmentation_input(images, self.config)
        if self.model_name == "lraspp_mobilenet_v3_large":
            normalized = normalize_image_channels(segmentation_images, self.config)
            return self.model(normalized)["out"]
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
            importance_score = importance_score * (1.0 - background_prob.to(class_probs.dtype))
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


class MemoryImageIdentity(nn.Module):
    """
    Compatibility stub for the current ACT training path.

    We do not use an online me_block in the main training loop right now.
    The recommended path is to precompute `memory_image_four_channel` offline
    and feed it directly into ACT as an additional visual input.
    """

    def forward(self, memory_image: torch.Tensor, _current_image: torch.Tensor) -> torch.Tensor:
        return memory_image


def build_importance_memory_model(config: Optional[MEBlockConfig] = None) -> ImportanceMemoryModel:
    return ImportanceMemoryModel(config=config)


def build_me_block(*_args, **_kwargs) -> nn.Module:
    return MemoryImageIdentity()


def checkpoint_payload(model: ImportanceMemoryModel) -> Dict:
    return {
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
    }
