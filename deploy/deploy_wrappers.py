from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from act.detr.models.me_block.memory_gate_model import ImportanceMemoryModel


def _tensor_1d(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


class _BaseActInferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module, joint_min, joint_rng):
        super().__init__()
        self.model = model.eval()
        self.register_buffer("joint_min", _tensor_1d(joint_min).view(1, -1))
        self.register_buffer("joint_rng", _tensor_1d(joint_rng).view(1, -1))
        self.register_buffer("image_mean", _tensor_1d([0.485, 0.456, 0.406, 0.5]).view(1, 4, 1, 1))
        self.register_buffer("image_std", _tensor_1d([0.229, 0.224, 0.225, 0.5]).view(1, 4, 1, 1))

    def _normalize_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        return (qpos - self.joint_min) / self.joint_rng

    def _denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.joint_rng.view(1, 1, -1) + self.joint_min.view(1, 1, -1)

    def _normalize_bgra_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image[:, [2, 1, 0, 3]]
        return (image - self.image_mean) / self.image_std


class ACTSingleImageInferenceWrapper(_BaseActInferenceWrapper):
    def forward(self, qpos: torch.Tensor, image_bgra: torch.Tensor) -> torch.Tensor:
        qpos = self._normalize_qpos(qpos)
        image = self._normalize_bgra_image(image_bgra).unsqueeze(1)
        actions, _, _ = self.model(qpos, image, None, None, None, None)
        return self._denormalize_actions(actions)


class ACTDualImageInferenceWrapper(_BaseActInferenceWrapper):
    def forward(self, qpos: torch.Tensor, image_bgra: torch.Tensor, memory_image_bgra: torch.Tensor) -> torch.Tensor:
        qpos = self._normalize_qpos(qpos)
        image = self._normalize_bgra_image(image_bgra).unsqueeze(1)
        memory_image = self._normalize_bgra_image(memory_image_bgra)
        actions, _, _ = self.model(qpos, image, None, memory_image, None, None)
        return self._denormalize_actions(actions)


class MEBlockInferenceWrapper(nn.Module):
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
