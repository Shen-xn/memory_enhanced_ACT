#!/usr/bin/env python3
"""Build direct PCA-coordinate supervision and PCA-reconstruction residuals."""

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


def build_argparser():
    parser = argparse.ArgumentParser(description="Build PCA-coordinate supervision targets.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pca-bank", required=True, help="Existing PCA bank npz containing pca_mean and pca_components.")
    parser.add_argument("--output-bank", required=True, help="Output bank npz for direct PCA-coordinate training.")
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--image-channels", type=int, default=4)
    parser.add_argument("--target-mode", choices=["absolute", "delta"], default="delta")
    parser.add_argument("--delta-qpos-scale", type=float, default=10.0)
    parser.add_argument("--target-filename", default="phase_pca16_targets.npz")
    return parser


def _load_dataset(args):
    return ImitationDataset(
        args.data_root,
        future_steps=args.future_steps,
        mode="train",
        image_channels=args.image_channels,
        target_mode=args.target_mode,
        delta_qpos_scale=args.delta_qpos_scale,
        require_phase_targets=False,
    )


def _target_from_raw(sample, target_mode, delta_qpos_scale):
    future_raw = sample["future_raw"].astype(np.float32)
    if target_mode == "absolute":
        return future_raw

    step_delta = np.empty_like(future_raw)
    step_delta[0] = future_raw[0] - sample["curr_raw"].astype(np.float32)
    if len(future_raw) > 1:
        step_delta[1:] = future_raw[1:] - future_raw[:-1]
    return step_delta / float(delta_qpos_scale)


def _stack_targets(samples, target_mode, delta_qpos_scale):
    vectors = np.stack(
        [
            _target_from_raw(sample, target_mode, delta_qpos_scale).reshape(-1)
            for sample in samples
        ],
        axis=0,
    ).astype(np.float32)
    task_names = [sample["task"] for sample in samples]
    frame_indices = np.array([sample["frame_index"] for sample in samples], dtype=np.int64)
    return vectors, task_names, frame_indices


def _project(vectors, mean, components):
    return (vectors - mean.reshape(1, -1)) @ components


def _reconstruct(coords, mean, components):
    return coords @ components.T + mean.reshape(1, -1)


def _write_per_task_targets(task_names, frame_indices, pca_coord_tgt, pca_recon_tgt, residual_tgt, filename):
    grouped = {}
    for idx, task_name in enumerate(task_names):
        grouped.setdefault(task_name, []).append(idx)

    for task_name, indices in grouped.items():
        indices = np.asarray(indices, dtype=np.int64)
        indices = indices[np.argsort(frame_indices[indices])]
        output_path = Path(task_name) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            frame_index=frame_indices[indices],
            pca_coord_tgt=pca_coord_tgt[indices],
            pca_recon_tgt=pca_recon_tgt[indices],
            residual_tgt=residual_tgt[indices],
        )


def _validate_written_targets(task_names, frame_indices, vectors, filename, atol=1e-5):
    grouped = {}
    for idx, task_name in enumerate(task_names):
        grouped.setdefault(task_name, []).append(idx)

    for task_name, indices in grouped.items():
        indices = np.asarray(indices, dtype=np.int64)
        order = np.argsort(frame_indices[indices])
        sorted_indices = indices[order]
        sorted_frames = frame_indices[sorted_indices]
        output_path = Path(task_name) / filename
        payload = np.load(output_path)

        if not np.array_equal(payload["frame_index"].astype(np.int64), sorted_frames):
            raise ValueError(f"{output_path} frame_index is not sorted/aligned after writing.")

        expected = vectors[sorted_indices]
        reconstructed = (
            payload["pca_recon_tgt"].reshape(len(sorted_indices), -1)
            + payload["residual_tgt"].reshape(len(sorted_indices), -1)
        )
        max_abs = float(np.max(np.abs(reconstructed - expected)))
        if max_abs > atol:
            raise ValueError(
                f"{output_path} pca_recon_tgt + residual_tgt mismatch: max_abs={max_abs:.6g}"
            )


def main():
    args = build_argparser().parse_args()

    pca_bank = np.load(args.pca_bank, allow_pickle=True)
    if "pca_mean" not in pca_bank or "pca_components" not in pca_bank:
        raise ValueError(f"{args.pca_bank} must contain pca_mean and pca_components")

    mean = pca_bank["pca_mean"].astype(np.float32)
    components = pca_bank["pca_components"].astype(np.float32)
    explained_ratio = float(pca_bank["explained_ratio"][0]) if "explained_ratio" in pca_bank else float("nan")
    pca_dim = int(pca_bank["pca_dim"][0]) if "pca_dim" in pca_bank else int(components.shape[1])

    dataset = _load_dataset(args)
    all_samples = dataset.all_samples
    vectors, task_names, frame_indices = _stack_targets(
        all_samples,
        target_mode=args.target_mode,
        delta_qpos_scale=args.delta_qpos_scale,
    )

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
        pca_dim=np.array([pca_dim], dtype=np.int64),
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
    print(f"Saved PCA coord bank to: {output_bank}")
    print(f"PCA dim: {pca_dim} | explained_ratio={explained_ratio:.4f}")
    print(f"Wrote per-task targets as: {args.target_filename}")
    print(f"Residual L1 per joint: {residual_l1_per_joint.tolist()}")


if __name__ == "__main__":
    main()
