"""Prototype-based auxiliary loss for structured action supervision.

This module keeps the core ACT regression path unchanged. It adds a frozen
action prototype bank built offline from training targets and exposes a small
helper that turns predicted action chunks into prototype logits.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionPrototypeBank(nn.Module):
    """Frozen motion prototype bank operating on flattened action chunks."""

    def __init__(self, centers, mean, std, pca_components=None, temperature=1.0):
        super().__init__()
        centers = torch.as_tensor(centers, dtype=torch.float32)
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if pca_components is None:
            pca_components = torch.zeros((mean.numel(), 0), dtype=torch.float32)
        else:
            pca_components = torch.as_tensor(pca_components, dtype=torch.float32)

        if centers.ndim != 2:
            raise ValueError(f"Prototype centers must be [K, D], got {tuple(centers.shape)}")
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError("Prototype mean/std must be 1D arrays.")
        if pca_components.ndim != 2:
            raise ValueError("pca_components must be a 2D matrix.")
        if pca_components.shape[0] != mean.shape[0]:
            raise ValueError("pca_components input dimension must match mean/std dimension.")
        projected_dim = pca_components.shape[1] if pca_components.shape[1] > 0 else mean.shape[0]
        if centers.shape[1] != projected_dim:
            raise ValueError(
                f"Prototype center dimension mismatch: expected {projected_dim}, got {centers.shape[1]}"
            )
        if temperature <= 0:
            raise ValueError("Prototype temperature must be > 0.")

        self.register_buffer("centers", centers)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-6))
        self.register_buffer("pca_components", pca_components)
        self.temperature = float(temperature)

    @classmethod
    def from_npz(cls, npz_path, temperature=1.0):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Prototype file not found: {npz_path}")
        payload = np.load(npz_path)
        pca_components = payload["pca_components"] if "pca_components" in payload else None
        return cls(
            centers=payload["centers"],
            mean=payload["mean"],
            std=payload["std"],
            pca_components=pca_components,
            temperature=temperature,
        )

    @property
    def input_dim(self):
        return int(self.mean.numel())

    @property
    def num_prototypes(self):
        return int(self.centers.shape[0])

    def _flatten_actions(self, actions):
        if actions.ndim != 3:
            raise ValueError(f"Expected action chunk shape [B, T, D], got {tuple(actions.shape)}")
        flat = actions.reshape(actions.shape[0], -1)
        if flat.shape[1] != self.input_dim:
            raise ValueError(
                f"Prototype input dim mismatch: expected {self.input_dim}, got {flat.shape[1]}"
            )
        return flat

    def encode_actions(self, actions):
        flat = self._flatten_actions(actions)
        normalized = (flat - self.mean.unsqueeze(0)) / self.std.unsqueeze(0)
        if self.pca_components.shape[1] > 0:
            normalized = normalized @ self.pca_components
        return normalized

    def logits(self, actions):
        encoded = self.encode_actions(actions)
        distances = torch.cdist(encoded, self.centers, p=2).pow(2)
        return -distances / self.temperature

    def assign(self, actions):
        with torch.no_grad():
            return self.logits(actions).argmax(dim=1)

    def classification_loss(self, pred_actions, target_actions):
        target_ids = self.assign(target_actions)
        logits = self.logits(pred_actions)
        loss = F.cross_entropy(logits, target_ids)
        acc = (logits.argmax(dim=1) == target_ids).float().mean()
        return loss, acc
