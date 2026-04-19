#!/usr/bin/env python3
"""Build a PCA bank and aligned per-task PCA supervision targets from data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_process.data_loader import ImitationDataset
from tools.build_phase_pca_targets import (
    _stack_targets,
    _validate_written_targets,
    _write_per_task_targets,
)


def build_argparser():
    parser = argparse.ArgumentParser(description="Build PCA bank plus per-task PCA supervision targets.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--pca-dim", type=int, required=True)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--image-channels", type=int, default=4)
    parser.add_argument("--target-mode", choices=["absolute", "delta"], default="delta")
    parser.add_argument("--delta-qpos-scale", type=float, default=10.0)
    parser.add_argument("--target-filename", required=True)
    return parser


def _load_all_samples(args):
    dataset = ImitationDataset(
        args.data_root,
        future_steps=args.future_steps,
        mode="train",
        image_channels=args.image_channels,
        target_mode=args.target_mode,
        delta_qpos_scale=args.delta_qpos_scale,
        require_phase_targets=False,
    )
    return dataset.all_samples


def _fit_pca(vectors, pca_dim):
    if pca_dim <= 0:
        raise ValueError(f"pca_dim must be positive, got {pca_dim}")
    if pca_dim > vectors.shape[1]:
        raise ValueError(f"pca_dim={pca_dim} exceeds action_dim={vectors.shape[1]}")

    mean = vectors.mean(axis=0).astype(np.float32)
    centered = vectors - mean.reshape(1, -1)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:pca_dim].T.astype(np.float32)

    eigenvalues = (singular_values.astype(np.float64) ** 2) / max(vectors.shape[0] - 1, 1)
    explained_ratio = float(eigenvalues[:pca_dim].sum() / eigenvalues.sum()) if eigenvalues.sum() > 0 else 0.0
    return mean, components, explained_ratio


def _project(vectors, mean, components):
    return (vectors - mean.reshape(1, -1)) @ components


def _reconstruct(coords, mean, components):
    return coords @ components.T + mean.reshape(1, -1)


def main():
    args = build_argparser().parse_args()
    all_samples = _load_all_samples(args)
    vectors, task_names, frame_indices = _stack_targets(
        all_samples,
        target_mode=args.target_mode,
        delta_qpos_scale=args.delta_qpos_scale,
    )

    mean, components, explained_ratio = _fit_pca(vectors, args.pca_dim)
    pca_coord_tgt = _project(vectors, mean, components).astype(np.float32)
    pca_recon_flat = _reconstruct(pca_coord_tgt, mean, components).astype(np.float32)
    residual_flat = (vectors - pca_recon_flat).astype(np.float32)

    pca_coord_mean = pca_coord_tgt.mean(axis=0).astype(np.float32)
    pca_coord_std = np.clip(pca_coord_tgt.std(axis=0).astype(np.float32), 1e-6, None)
    state_dim = vectors.shape[1] // args.future_steps
    pca_recon_tgt = pca_recon_flat.reshape(-1, args.future_steps, state_dim).astype(np.float32)
    residual_tgt = residual_flat.reshape(-1, args.future_steps, state_dim).astype(np.float32)
    residual_mean = residual_tgt.mean(axis=0).astype(np.float32)
    residual_std = np.clip(residual_tgt.std(axis=0).astype(np.float32), 1e-6, None)

    output_bank = Path(args.output_bank)
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_bank,
        pca_mean=mean.astype(np.float32),
        pca_components=components.astype(np.float32),
        pca_coord_mean=pca_coord_mean,
        pca_coord_std=pca_coord_std,
        residual_mean=residual_mean,
        residual_std=residual_std,
        explained_ratio=np.array([explained_ratio], dtype=np.float32),
        pca_dim=np.array([args.pca_dim], dtype=np.int64),
        future_steps=np.array([args.future_steps], dtype=np.int64),
        state_dim=np.array([state_dim], dtype=np.int64),
        target_mode=np.array([args.target_mode]),
        delta_qpos_scale=np.array([args.delta_qpos_scale], dtype=np.float32),
    )

    _write_per_task_targets(
        task_names=task_names,
        frame_indices=frame_indices,
        pca_coord_tgt=pca_coord_tgt,
        pca_recon_tgt=pca_recon_tgt,
        residual_tgt=residual_tgt,
        filename=args.target_filename,
    )
    _validate_written_targets(
        task_names=task_names,
        frame_indices=frame_indices,
        vectors=vectors,
        filename=args.target_filename,
    )

    residual_l1_per_joint = np.mean(np.abs(residual_tgt), axis=(0, 1))
    print(f"Saved PCA bank to: {output_bank}")
    print(f"PCA dim: {args.pca_dim} | explained_ratio={explained_ratio:.6f}")
    print(f"Wrote per-task targets as: {args.target_filename}")
    print(f"Residual L1 per joint: {residual_l1_per_joint.tolist()}")


if __name__ == "__main__":
    main()
