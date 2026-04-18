from __future__ import annotations

import numpy as np
import torch


class PhasePrototypeBank:
    """Offline phase-prototype bank used by the phase-token ACT variant."""

    def __init__(self, prototype_actions_flat, pca_components, pca_mean):
        self.prototype_actions_flat = torch.as_tensor(prototype_actions_flat, dtype=torch.float32)
        self.pca_components = torch.as_tensor(pca_components, dtype=torch.float32)
        self.pca_mean = torch.as_tensor(pca_mean, dtype=torch.float32)

    @classmethod
    def from_npz(cls, path: str) -> "PhasePrototypeBank":
        payload = np.load(path)
        return cls(
            prototype_actions_flat=payload["prototype_actions_flat"],
            pca_components=payload["pca_components"],
            pca_mean=payload["pca_mean"],
        )

    @property
    def num_prototypes(self) -> int:
        return int(self.prototype_actions_flat.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.prototype_actions_flat.shape[1])

