"""Build phase-prototype bank and per-task alpha/residual targets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_process.data_loader import ImitationDataset


def build_argparser():
    parser = argparse.ArgumentParser(description="Build phase-prototype bank and per-task targets.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--bank-output", required=True)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--image-channels", type=int, default=4)
    parser.add_argument("--target-mode", choices=["absolute", "delta"], default="delta")
    parser.add_argument("--delta-qpos-scale", type=float, default=10.0)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--kmeans-epochs", type=int, default=20)
    parser.add_argument("--pca-var-ratio", type=float, default=0.85)
    parser.add_argument("--target-filename", default="phase_proto_targets.npz")
    parser.add_argument("--simplex-steps", type=int, default=60)
    parser.add_argument("--simplex-lr", type=float, default=0.2)
    return parser


def _load_dataset(args, mode, joint_min_max=None):
    return ImitationDataset(
        args.data_root,
        future_steps=args.future_steps,
        mode=mode,
        joint_min_max=joint_min_max,
        image_channels=args.image_channels,
        target_mode=args.target_mode,
        delta_qpos_scale=args.delta_qpos_scale,
        require_phase_targets=False,
    )


def _stack_targets(samples):
    vectors = np.stack([sample["future"].reshape(-1) for sample in samples], axis=0).astype(np.float32)
    task_names = [sample["task"] for sample in samples]
    frame_indices = np.array([sample["frame_index"] for sample in samples], dtype=np.int64)
    return vectors, task_names, frame_indices


def _fit_pca(vectors, var_ratio):
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained = (singular_values ** 2) / max(np.sum(singular_values ** 2), 1e-8)
    cumulative = np.cumsum(explained)
    pca_dim = int(np.searchsorted(cumulative, var_ratio) + 1)
    components = vt[:pca_dim].T.astype(np.float32)
    return mean.squeeze(0).astype(np.float32), components, float(cumulative[pca_dim - 1]), pca_dim


def _project(vectors, mean, components):
    return (vectors - mean.reshape(1, -1)) @ components


def _run_kmeans(z, clusters, device, batch_size, epochs, seed):
    x = torch.from_numpy(z).to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=generator, device=device)[:clusters]
    centers = x[perm].clone()
    counts = torch.zeros(clusters, device=device, dtype=torch.float32)

    for epoch in range(epochs):
        order = torch.randperm(x.shape[0], generator=generator, device=device)
        epoch_dist = 0.0
        for start in range(0, x.shape[0], batch_size):
            batch = x[order[start : start + batch_size]]
            dists = torch.cdist(batch, centers).pow(2)
            assign = dists.argmin(dim=1)
            epoch_dist += dists.gather(1, assign.unsqueeze(1)).sum().item()

            batch_counts = torch.bincount(assign, minlength=clusters).float()
            sums = torch.zeros_like(centers)
            sums.index_add_(0, assign, batch)
            valid = batch_counts > 0
            if valid.any():
                batch_means = sums[valid] / batch_counts[valid].unsqueeze(1)
                eta = batch_counts[valid] / (counts[valid] + batch_counts[valid]).clamp_min(1e-6)
                centers[valid] = centers[valid] * (1.0 - eta.unsqueeze(1)) + batch_means * eta.unsqueeze(1)
                counts[valid] += batch_counts[valid]
        print(f"[kmeans {epoch + 1}/{epochs}] mean sq dist={epoch_dist / max(x.shape[0], 1):.6f}")
    return centers.cpu().numpy().astype(np.float32)


def _project_to_simplex(alpha):
    sorted_alpha, _ = torch.sort(alpha, dim=-1, descending=True)
    cssv = torch.cumsum(sorted_alpha, dim=-1) - 1.0
    arange = torch.arange(1, alpha.shape[-1] + 1, device=alpha.device, dtype=alpha.dtype).view(1, -1)
    cond = sorted_alpha - cssv / arange > 0
    rho = cond.sum(dim=-1).clamp_min(1) - 1
    theta = cssv.gather(1, rho.unsqueeze(1)) / (rho.to(alpha.dtype).unsqueeze(1) + 1.0)
    return torch.clamp(alpha - theta, min=0.0)


def _solve_alpha_targets(z, centers, steps, lr, device, batch_size):
    z_t = torch.from_numpy(z).to(device=device, dtype=torch.float32)
    c_t = torch.from_numpy(centers).to(device=device, dtype=torch.float32)
    gram = c_t @ c_t.T
    zc = z_t @ c_t.T
    results = []

    for start in range(0, z_t.shape[0], batch_size):
        zc_batch = zc[start : start + batch_size]
        alpha = torch.full(
            (zc_batch.shape[0], c_t.shape[0]),
            1.0 / c_t.shape[0],
            device=device,
            dtype=torch.float32,
        )
        for _ in range(steps):
            grad = alpha @ gram - zc_batch
            alpha = _project_to_simplex(alpha - lr * grad)
        results.append(alpha)
    return torch.cat(results, dim=0).cpu().numpy().astype(np.float32)


def _write_per_task_targets(task_names, frame_indices, alpha_tgt, prototype_tgt, residual_tgt, filename):
    grouped = {}
    for idx, task_name in enumerate(task_names):
        grouped.setdefault(task_name, []).append(idx)

    for task_name, indices in grouped.items():
        indices = np.asarray(indices, dtype=np.int64)
        output_path = Path(task_name) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            frame_index=frame_indices[indices],
            alpha_tgt=alpha_tgt[indices],
            prototype_tgt=prototype_tgt[indices],
            residual_tgt=residual_tgt[indices],
        )


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")

    train_dataset = _load_dataset(args, mode="train")
    val_dataset = _load_dataset(args, mode="val", joint_min_max=train_dataset.joint_min_max)
    train_vectors, _, _ = _stack_targets(train_dataset.samples)
    all_samples = train_dataset.samples + val_dataset.samples
    vectors, task_names, frame_indices = _stack_targets(all_samples)

    mean, components, explained_ratio, pca_dim = _fit_pca(train_vectors, args.pca_var_ratio)
    train_z = _project(train_vectors, mean, components)
    z = _project(vectors, mean, components)
    centers = _run_kmeans(train_z, args.clusters, device, args.batch_size, args.kmeans_epochs, args.seed)
    alpha_tgt = _solve_alpha_targets(z, centers, args.simplex_steps, args.simplex_lr, device, args.batch_size)

    z_proto = alpha_tgt @ centers
    prototype_flat = z_proto @ components.T + mean.reshape(1, -1)
    residual_flat = vectors - prototype_flat

    prototype_actions = centers @ components.T + mean.reshape(1, -1)

    bank_output = Path(args.bank_output)
    bank_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        bank_output,
        pca_mean=mean.astype(np.float32),
        pca_components=components.astype(np.float32),
        prototype_centers=centers.astype(np.float32),
        prototype_actions_flat=prototype_actions.astype(np.float32),
        explained_ratio=np.array([explained_ratio], dtype=np.float32),
        pca_dim=np.array([pca_dim], dtype=np.int64),
        num_prototypes=np.array([args.clusters], dtype=np.int64),
        future_steps=np.array([args.future_steps], dtype=np.int64),
        state_dim=np.array([vectors.shape[1] // args.future_steps], dtype=np.int64),
        target_mode=np.array([args.target_mode]),
        delta_qpos_scale=np.array([args.delta_qpos_scale], dtype=np.float32),
    )

    _write_per_task_targets(
        task_names=task_names,
        frame_indices=frame_indices,
        alpha_tgt=alpha_tgt,
        prototype_tgt=prototype_flat.reshape(-1, args.future_steps, vectors.shape[1] // args.future_steps).astype(np.float32),
        residual_tgt=residual_flat.reshape(-1, args.future_steps, vectors.shape[1] // args.future_steps).astype(np.float32),
        filename=args.target_filename,
    )

    print(f"Saved phase bank to: {bank_output}")
    print(f"PCA dim: {pca_dim} | explained_ratio={explained_ratio:.4f} | prototypes={args.clusters}")
    print(f"Wrote per-task targets as: {args.target_filename}")


if __name__ == "__main__":
    main()
