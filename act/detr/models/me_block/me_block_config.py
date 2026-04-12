from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List


DEFAULT_CLASS_NAMES = ["target", "goal", "arm"]


@dataclass
class ImportanceModelConfig:
    """Architecture and normalization settings saved inside me_block checkpoints."""

    # Changes in this block affect the segmentation model itself.
    # If you change them, train a new checkpoint.
    model_name: str = "compact_fpn_v1"
    input_channels: int = 4
    segmentation_input_channels: int = 3
    background_index: int = 0
    unlabeled_index: int = 255
    class_names: List[str] = field(default_factory=lambda: list(DEFAULT_CLASS_NAMES))
    image_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    image_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    pretrained_backbone: bool = False

    @property
    def num_foreground_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_output_classes(self) -> int:
        return self.num_foreground_classes + 1


@dataclass
class MemoryUpdateConfig:
    """State-update parameters for converting segmentation probabilities to memory."""

    # keep_top_ratio_* means "keep the top fraction of pixels by score_state for each class".
    # Example: keep_top_ratio_target=0.05 keeps the top 5% target pixels in each frame.
    score_decay: float = 0.92
    tau_up: float = 0.0
    keep_top_ratio_target: float = 0.02
    keep_top_ratio_goal: float = 0.04
    keep_top_ratio_arm: float = 0.2


@dataclass
class ImportanceTrainingConfig:
    """Training-only settings for the importance segmentation model."""

    # Data/training settings below are used when you start a new training run.
    # Changing them does not modify an existing checkpoint.
    # If you want the new settings to take effect, train again.
    data_root: str = ""
    image_dirname: str = "auto"
    label_dirname: str = "importance_labels"
    use_augmentation: bool = True
    gamma_min: float = 0.6
    gamma_max: float = 1.8
    noise_std: float = 0.02
    horizontal_flip_prob: float = 0.5
    translation_px: int = 32
    rotation_deg: float = 10.0
    scale_min: float = 0.8
    scale_max: float = 1.2
    batch_size: int = 8
    num_workers: int = 0
    num_epochs: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 1e-6
    lr_scheduler: str = "warmup_cosine"
    # Warmup prevents early unstable updates; cosine decay keeps late training gentle.
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.1
    train_ratio: float = 0.8
    seed: int = 42
    save_root: str = ""
    log_every: int = 10


@dataclass
class MemoryGenerationConfig:
    """Path settings used when generating memory_image_four_channel."""

    # Output/path settings below are used only when generating memory images.
    # They do not require retraining.
    # generate_memory_images.py still restores the saved generation config from checkpoint first.
    data_root: str = ""
    image_dirname: str = "four_channel"
    output_dirname: str = "memory_image_four_channel"
    save_score_dirname: str = "memory_scores"
    save_mask_dirname: str = "memory_binary_masks"


@dataclass
class MEBlockConfig:
    """Top-level me_block config written to config.json and checkpoint payloads."""

    # Quick guide:
    # - importance / training changes -> retrain
    # - memory changes -> no retrain, but regenerate memory images
    # - generation changes -> no retrain
    importance: ImportanceModelConfig = field(default_factory=ImportanceModelConfig)
    memory: MemoryUpdateConfig = field(default_factory=MemoryUpdateConfig)
    training: ImportanceTrainingConfig = field(default_factory=ImportanceTrainingConfig)
    generation: MemoryGenerationConfig = field(default_factory=MemoryGenerationConfig)

    def to_dict(self) -> Dict:
        return asdict(self)


def default_me_block_config() -> MEBlockConfig:
    return MEBlockConfig()


def _filter_dataclass_payload(cls, payload: Dict | None) -> Dict:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in dict(payload or {}).items() if key in allowed}


def importance_model_config_from_dict(payload: Dict | None) -> ImportanceModelConfig:
    data = _filter_dataclass_payload(ImportanceModelConfig, payload)
    # Older checkpoints were trained before `model_name` was saved. Treat those
    # configs as the original ResNet18-layer2 model instead of inheriting today's
    # default architecture.
    if payload is not None and "model_name" not in payload:
        data.setdefault("model_name", "truncated_resnet18_layer2")
        data.setdefault("segmentation_input_channels", 3)
        data.setdefault("image_mean", [0.485, 0.456, 0.406, 0.5])
        data.setdefault("image_std", [0.229, 0.224, 0.225, 0.5])
        data.setdefault("pretrained_backbone", True)
    return ImportanceModelConfig(**data)


def memory_update_config_from_dict(payload: Dict | None) -> MemoryUpdateConfig:
    data = dict(payload or {})
    data = _filter_dataclass_payload(MemoryUpdateConfig, data)
    return MemoryUpdateConfig(**data)


def training_config_from_dict(payload: Dict | None) -> ImportanceTrainingConfig:
    return ImportanceTrainingConfig(**_filter_dataclass_payload(ImportanceTrainingConfig, payload))


def generation_config_from_dict(payload: Dict | None) -> MemoryGenerationConfig:
    return MemoryGenerationConfig(**_filter_dataclass_payload(MemoryGenerationConfig, payload))


def save_config(config: MEBlockConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def load_config(path: str) -> MEBlockConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return MEBlockConfig(
        importance=importance_model_config_from_dict(payload.get("importance", {})),
        memory=memory_update_config_from_dict(payload.get("memory", {})),
        training=training_config_from_dict(payload.get("training", {})),
        generation=generation_config_from_dict(payload.get("generation", {})),
    )
