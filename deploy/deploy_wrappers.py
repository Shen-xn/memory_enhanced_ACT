"""TorchScript-friendly wrappers used by deployment export."""

from __future__ import annotations

import torch
import torch.nn as nn


def _tensor_1d(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


class ACTSingleImageInferenceWrapper(nn.Module):
    """Export wrapper for baseline RGB/RGBD ACT."""

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
        self.model = model.eval()
        self.image_channels = int(image_channels)
        self.predict_delta_qpos = bool(predict_delta_qpos)
        self.register_buffer("joint_min", _tensor_1d(joint_min).view(1, -1))
        self.register_buffer("joint_rng", _tensor_1d(joint_rng).view(1, -1))
        self.register_buffer("delta_qpos_scale", torch.tensor(float(delta_qpos_scale), dtype=torch.float32))
        self.register_buffer(
            "image_mean",
            _tensor_1d([0.485, 0.456, 0.406, 0.5][: self.image_channels]).view(1, self.image_channels, 1, 1),
        )
        self.register_buffer(
            "image_std",
            _tensor_1d([0.229, 0.224, 0.225, 0.5][: self.image_channels]).view(1, self.image_channels, 1, 1),
        )

    def _normalize_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        return (qpos - self.joint_min) / self.joint_rng

    def _decode_actions(self, qpos_raw: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.predict_delta_qpos:
            step_delta = actions * self.delta_qpos_scale.view(1, 1, 1)
            cumulative_delta = torch.cumsum(step_delta, dim=1)
            return qpos_raw.unsqueeze(1) + cumulative_delta
        return actions * self.joint_rng.view(1, 1, -1) + self.joint_min.view(1, 1, -1)

    def _normalize_bgra_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image[:, [2, 1, 0, 3]]
        image = image[:, : self.image_channels]
        return (image - self.image_mean) / self.image_std

    def forward(self, qpos: torch.Tensor, image_bgra: torch.Tensor) -> torch.Tensor:
        qpos_raw = qpos
        qpos = self._normalize_qpos(qpos)
        image = self._normalize_bgra_image(image_bgra).unsqueeze(1)
        actions, _, _, _ = self.model(qpos, image, None)
        return self._decode_actions(qpos_raw, actions)
