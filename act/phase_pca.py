from __future__ import annotations

import numpy as np
import torch


class PhasePCABank:
    """Offline PCA bank used by the phase-token ACT variant."""

    def __init__(
        self,
        pca_mean,
        pca_components,
        pca_coord_mean,
        pca_coord_std,
        residual_mean,
        residual_std,
    ):
        self.pca_mean = torch.as_tensor(pca_mean, dtype=torch.float32)
        self.pca_components = torch.as_tensor(pca_components, dtype=torch.float32)
        self.pca_coord_mean = torch.as_tensor(pca_coord_mean, dtype=torch.float32)
        self.pca_coord_std = torch.as_tensor(pca_coord_std, dtype=torch.float32)
        self.residual_mean = torch.as_tensor(residual_mean, dtype=torch.float32)
        self.residual_std = torch.as_tensor(residual_std, dtype=torch.float32)

    @classmethod
    def from_npz(cls, path: str) -> "PhasePCABank":
        payload = np.load(path)

        pca_mean = payload["pca_mean"]
        pca_components = payload["pca_components"]

        pca_dim = int(payload["pca_dim"][0]) if "pca_dim" in payload else int(pca_components.shape[1])
        state_dim = int(payload["state_dim"][0]) if "state_dim" in payload else 6
        future_steps = int(payload["future_steps"][0]) if "future_steps" in payload else int(pca_mean.shape[0] // state_dim)

        pca_coord_mean = payload["pca_coord_mean"] if "pca_coord_mean" in payload else np.zeros((pca_dim,), dtype=np.float32)
        pca_coord_std = payload["pca_coord_std"] if "pca_coord_std" in payload else np.ones((pca_dim,), dtype=np.float32)
        residual_mean = payload["residual_mean"] if "residual_mean" in payload else np.zeros((future_steps, state_dim), dtype=np.float32)
        residual_std = payload["residual_std"] if "residual_std" in payload else np.ones((future_steps, state_dim), dtype=np.float32)

        return cls(
            pca_mean=pca_mean,
            pca_components=pca_components,
            pca_coord_mean=pca_coord_mean,
            pca_coord_std=pca_coord_std,
            residual_mean=residual_mean,
            residual_std=residual_std,
        )

    @property
    def pca_dim(self) -> int:
        return int(self.pca_components.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.pca_components.shape[0])
