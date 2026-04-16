"""TorchScript-friendly wrappers used by deployment export.

Training code normalizes images/qpos before calling ACT. Deployment receives
raw robot qpos and OpenCV images, so these wrappers reproduce the same
preprocessing inside the exported TorchScript modules.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from act.detr.models.me_block.memory_gate_model import ImportanceMemoryModel


def _tensor_1d(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


class _BaseActInferenceWrapper(nn.Module):
    """Shared preprocessing for ACT inference wrappers."""

    def __init__(
        self,
        model: nn.Module,
        joint_min,
        joint_rng,
        image_channels: int = 4,
        predict_delta_qpos: bool = False,
        delta_qpos_scale: float = 10.0,
    ):
        super().__init__()
        if image_channels not in (3, 4):
            raise ValueError(f"image_channels must be 3 or 4, got {image_channels}")
        if delta_qpos_scale <= 0:
            raise ValueError("delta_qpos_scale must be > 0.")
        self.model = model.eval()
        self.image_channels = int(image_channels)
        self.predict_delta_qpos = bool(predict_delta_qpos)
        self.register_buffer("joint_min", _tensor_1d(joint_min).view(1, -1))
        self.register_buffer("joint_rng", _tensor_1d(joint_rng).view(1, -1))
        self.register_buffer("delta_qpos_scale", torch.tensor(float(delta_qpos_scale), dtype=torch.float32))
        self.register_buffer("image_mean", _tensor_1d([0.485, 0.456, 0.406, 0.5][: self.image_channels]).view(1, self.image_channels, 1, 1))
        self.register_buffer("image_std", _tensor_1d([0.229, 0.224, 0.225, 0.5][: self.image_channels]).view(1, self.image_channels, 1, 1))

    def _normalize_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        return (qpos - self.joint_min) / self.joint_rng

    def _decode_actions(self, qpos_raw: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.predict_delta_qpos:
            return qpos_raw.unsqueeze(1) + actions * self.delta_qpos_scale.view(1, 1, 1)
        return actions * self.joint_rng.view(1, 1, -1) + self.joint_min.view(1, 1, -1)

    def _normalize_bgra_image(self, image: torch.Tensor) -> torch.Tensor:
        """Convert deployment BGRA into the RGB/RGBD tensor expected by ACT."""
        image = image[:, [2, 1, 0, 3]]
        image = image[:, : self.image_channels]
        return (image - self.image_mean) / self.image_std


class ACTSingleImageInferenceWrapper(_BaseActInferenceWrapper):
    """Export wrapper for baseline ACT: qpos + current image -> action chunk."""

    def forward(self, qpos: torch.Tensor, image_bgra: torch.Tensor) -> torch.Tensor:
        qpos_raw = qpos
        qpos = self._normalize_qpos(qpos)
        image = self._normalize_bgra_image(image_bgra).unsqueeze(1)
        actions, _, _ = self.model(qpos, image, None, None, None, None)
        return self._decode_actions(qpos_raw, actions)


class ACTDualImageInferenceWrapper(_BaseActInferenceWrapper):
    """Export wrapper for memory ACT: qpos + image + memory image -> actions."""

    def forward(self, qpos: torch.Tensor, image_bgra: torch.Tensor, memory_image_bgra: torch.Tensor) -> torch.Tensor:
        qpos_raw = qpos
        qpos = self._normalize_qpos(qpos)
        image = self._normalize_bgra_image(image_bgra).unsqueeze(1)
        memory_image = self._normalize_bgra_image(memory_image_bgra)
        actions, _, _ = self.model(qpos, image, None, memory_image, None, None)
        return self._decode_actions(qpos_raw, actions)


class MEBlockInferenceWrapper(nn.Module):
    """Export wrapper for online recurrent memory-image generation."""

    def __init__(self, model: ImportanceMemoryModel):
        super().__init__()
        self.model = model.eval()

    def forward(
        self,
        current_image_bgra: torch.Tensor,
        prev_memory_bgra: torch.Tensor,
        prev_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step = self.model(current_image_bgra, prev_memory=prev_memory_bgra, prev_scores=prev_scores)
        return step.memory_image, step.memory_state, step.score_state
