from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List


DEFAULT_CLASS_NAMES = ["target", "goal", "arm"]
DEFAULT_CLASS_WEIGHTS = {
    "target": 0.6,
    "goal": 0.3,
    "arm": 0.1,
}


def normalize_class_weights(class_names: List[str], weights: Dict[str, float]) -> Dict[str, float]:
    normalized = {name: max(float(weights.get(name, 0.0)), 0.0) for name in class_names}
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("Class weights must sum to a positive value.")
    return {name: value / total for name, value in normalized.items()}


@dataclass
class ImportanceModelConfig:
    # Changes in this block affect the segmentation model itself.
    # If you change them, train a new checkpoint.
    model_name: str = "truncated_resnet18_layer2"
    input_channels: int = 4
    segmentation_input_channels: int = 3
    background_index: int = 0
    unlabeled_index: int = 255
    class_names: List[str] = field(default_factory=lambda: list(DEFAULT_CLASS_NAMES))
    class_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CLASS_WEIGHTS))
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406, 0.5])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225, 0.5])
    pretrained_backbone: bool = True

    def normalized_class_weights(self) -> Dict[str, float]:
        return normalize_class_weights(self.class_names, self.class_weights)

    @property
    def num_foreground_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_output_classes(self) -> int:
        return self.num_foreground_classes + 1

    @property
    def class_to_index(self) -> Dict[str, int]:
        return {name: idx + 1 for idx, name in enumerate(self.class_names)}

    @property
    def index_to_class(self) -> Dict[int, str]:
        mapping = {self.background_index: "background"}
        mapping.update({idx: name for name, idx in self.class_to_index.items()})
        return mapping


@dataclass
class MemoryUpdateConfig:
    # Changes in this block do not require retraining the segmenter.
    # They only affect memory-image generation, so regenerate memory images.
    # Note: generate_memory_images.py loads these values from the checkpoint config.
    # Changing only the defaults here will not modify an old checkpoint automatically.
    # keep_top_ratio means "keep the top keep_top_ratio fraction of pixels by score_state".
    # Example: keep_top_ratio=0.05 keeps the top 5% pixels in each frame.
    score_decay: float = 0.9
    tau_up: float = 0.1
    keep_top_ratio: float = 0.5


@dataclass
class ImportanceTrainingConfig:
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
    batch_size: int = 4
    num_workers: int = 0
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    train_ratio: float = 0.8
    seed: int = 42
    save_root: str = ""
    log_every: int = 10


@dataclass
class MemoryGenerationConfig:
    # Output/path settings below are used only when generating memory images.
    # They do not require retraining.
    # generate_memory_images.py still restores the saved generation config from checkpoint first.
    data_root: str = ""
    image_dirname: str = "four_channel"
    output_dirname: str = "memory_image_four_channel"
    save_score_dirname: str = "memory_scores"
    save_mask_dirname: str = "memory_binary_masks"
    task_filter: str = ""
    force: bool = False


@dataclass
class MEBlockConfig:
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


def memory_update_config_from_dict(payload: Dict | None) -> MemoryUpdateConfig:
    data = dict(payload or {})
    if "keep_top_ratio" not in data and "tau_out" in data:
        data["keep_top_ratio"] = data.pop("tau_out")
    data.pop("output_dilation_radius", None)
    return MemoryUpdateConfig(**data)


def save_config(config: MEBlockConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def load_config(path: str) -> MEBlockConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return MEBlockConfig(
        importance=ImportanceModelConfig(**payload.get("importance", {})),
        memory=memory_update_config_from_dict(payload.get("memory", {})),
        training=ImportanceTrainingConfig(**payload.get("training", {})),
        generation=MemoryGenerationConfig(**payload.get("generation", {})),
    )
