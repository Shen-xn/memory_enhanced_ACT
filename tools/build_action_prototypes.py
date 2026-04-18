"""Build offline action prototypes with GPU-friendly mini-batch k-means.

This version is designed for very large ACT datasets:
- it never stacks the whole action set into one giant matrix;
- it computes per-dimension mean/std in a streaming pass;
- it runs k-means updates on GPU in mini-batches;
- it makes a final streaming pass to estimate cluster counts and inertia.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_process.data_loader import ImitationDataset


def build_argparser():
    parser = argparse.ArgumentParser(description="Build motion prototypes for ACT.")
    parser.add_argument("--data-root", required=True, help="Dataset root used by training.py")
    parser.add_argument("--output", required=True, help="Where to save the .npz prototype bank")
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--image-channels", type=int, default=4)
    parser.add_argument("--target-mode", choices=["absolute", "delta", "raw"], default="delta")
    parser.add_argument("--delta-qpos-scale", type=float, default=10.0)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--kmeans-epochs", type=int, default=20)
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="If > 0, cluster in PCA space with this many principal components.",
    )
    return parser


def _load_training_dataset(args):
    return ImitationDataset(
        args.data_root,
        future_steps=args.future_steps,
        use_memory_image_input=False,
        mode="train",
        image_channels=args.image_channels,
        target_mode=args.target_mode,
        delta_qpos_scale=args.delta_qpos_scale,
    )


def _iter_action_batches(dataset, batch_size, shuffle, seed):
    indices = np.arange(len(dataset.samples))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch = np.stack(
            [dataset.samples[int(i)]["future"].reshape(-1) for i in batch_idx],
            axis=0,
        ).astype(np.float32)
        yield batch


def _compute_streaming_stats(dataset, batch_size):
    feature_dim = dataset.samples[0]["future"].size
    total_count = 0
    feature_sum = np.zeros(feature_dim, dtype=np.float64)
    feature_sumsq = np.zeros(feature_dim, dtype=np.float64)

    for batch in _iter_action_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
        total_count += batch.shape[0]
        feature_sum += batch.sum(axis=0, dtype=np.float64)
        feature_sumsq += np.square(batch, dtype=np.float64).sum(axis=0, dtype=np.float64)

    mean = (feature_sum / max(total_count, 1)).astype(np.float32)
    var = (feature_sumsq / max(total_count, 1) - np.square(mean, dtype=np.float64)).astype(np.float32)
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var, dtype=np.float32)
    return mean, std, total_count, feature_dim


def _normalize_batch(batch, mean, std, device):
    x = torch.from_numpy(batch).to(device=device, dtype=torch.float32, non_blocking=True)
    return (x - mean.unsqueeze(0)) / std.unsqueeze(0)


def _fit_pca(dataset, batch_size, mean, std, device, pca_dim):
    mean_t = torch.from_numpy(mean).to(device=device, dtype=torch.float32)
    std_t = torch.from_numpy(std).to(device=device, dtype=torch.float32)
    feature_dim = mean.shape[0]
    cov = torch.zeros((feature_dim, feature_dim), device=device, dtype=torch.float32)
    total = 0

    for batch in _iter_action_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
        x = _normalize_batch(batch, mean_t, std_t, device)
        cov += x.T @ x
        total += x.shape[0]

    cov = cov / max(total, 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    components = evecs[:, :pca_dim].contiguous()
    explained = evals[:pca_dim].sum() / evals.clamp_min(1e-8).sum()
    return components, float(explained.item())


def _project_normalized(x, components):
    if components is None:
        return x
    return x @ components


def _reconstruct_center_space(centers, components):
    if components is None:
        return centers
    return centers @ components.T


def _init_centers(dataset, args, mean, std, device, components=None):
    if args.clusters > len(dataset.samples):
        raise ValueError(
            f"clusters ({args.clusters}) cannot exceed number of training samples ({len(dataset.samples)})."
        )

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(dataset.samples), size=args.clusters, replace=False)
    init = np.stack(
        [dataset.samples[int(i)]["future"].reshape(-1) for i in chosen],
        axis=0,
    ).astype(np.float32)
    init = _normalize_batch(init, mean, std, device)
    return _project_normalized(init, components)


def _mini_batch_kmeans(dataset, args, mean, std, device, components=None):
    mean_t = torch.from_numpy(mean).to(device=device, dtype=torch.float32)
    std_t = torch.from_numpy(std).to(device=device, dtype=torch.float32)
    centers = _init_centers(dataset, args, mean_t, std_t, device, components=components)
    running_counts = torch.zeros(args.clusters, device=device, dtype=torch.float32)

    for epoch in range(args.kmeans_epochs):
        epoch_distance = 0.0
        epoch_count = 0

        for batch in _iter_action_batches(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            x = _normalize_batch(batch, mean_t, std_t, device)
            x = _project_normalized(x, components)
            distances = torch.cdist(x, centers, p=2).pow(2)
            assignment = distances.argmin(dim=1)
            epoch_distance += distances.gather(1, assignment.unsqueeze(1)).sum().item()
            epoch_count += x.shape[0]

            counts = torch.bincount(assignment, minlength=args.clusters).to(dtype=torch.float32)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, assignment, x)

            valid = counts > 0
            if valid.any():
                batch_means = sums[valid] / counts[valid].unsqueeze(1)
                eta = counts[valid] / (running_counts[valid] + counts[valid]).clamp_min(1e-6)
                centers[valid] = centers[valid] * (1.0 - eta.unsqueeze(1)) + batch_means * eta.unsqueeze(1)
                running_counts[valid] += counts[valid]

        mean_sq_dist = epoch_distance / max(epoch_count, 1)
        print(f"[epoch {epoch + 1}/{args.kmeans_epochs}] mean squared distance: {mean_sq_dist:.6f}")

    return centers


def _final_assignment_stats(dataset, args, mean, std, centers, device, components=None):
    mean_t = torch.from_numpy(mean).to(device=device, dtype=torch.float32)
    std_t = torch.from_numpy(std).to(device=device, dtype=torch.float32)

    counts = np.zeros(args.clusters, dtype=np.int64)
    inertia = 0.0

    for batch in _iter_action_batches(dataset, batch_size=args.batch_size, shuffle=False, seed=0):
        x = _normalize_batch(batch, mean_t, std_t, device)
        x = _project_normalized(x, components)
        distances = torch.cdist(x, centers, p=2).pow(2)
        assignment = distances.argmin(dim=1)
        nearest = distances.gather(1, assignment.unsqueeze(1)).squeeze(1)

        inertia += nearest.sum().item()
        bincount = torch.bincount(assignment, minlength=args.clusters).cpu().numpy().astype(np.int64)
        counts += bincount

    return counts, float(inertia)


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")

    dataset = _load_training_dataset(args)
    mean, std, total_count, feature_dim = _compute_streaming_stats(dataset, batch_size=args.batch_size)
    components = None
    explained = None
    if args.pca_dim > 0:
        if args.pca_dim >= feature_dim:
            raise ValueError(f"pca-dim must be < feature_dim ({feature_dim}), got {args.pca_dim}.")
        components, explained = _fit_pca(dataset, args.batch_size, mean, std, device, args.pca_dim)
        print(f"PCA enabled | dim={args.pca_dim} | explained_variance_ratio={explained:.4f}")

    centers = _mini_batch_kmeans(dataset, args, mean, std, device, components=components)
    counts, inertia = _final_assignment_stats(dataset, args, mean, std, centers, device, components=components)

    centers_np = centers.detach().cpu().numpy().astype(np.float32)
    centers_feature = _reconstruct_center_space(centers, components)
    centers_feature_np = centers_feature.detach().cpu().numpy().astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        centers=centers_np,
        centers_feature=centers_feature_np,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_dim=np.array([feature_dim], dtype=np.int64),
        num_samples=np.array([total_count], dtype=np.int64),
        num_clusters=np.array([args.clusters], dtype=np.int64),
        future_steps=np.array([args.future_steps], dtype=np.int64),
        state_dim=np.array([feature_dim // args.future_steps], dtype=np.int64),
        target_mode=np.array([args.target_mode]),
        delta_qpos_scale=np.array([args.delta_qpos_scale], dtype=np.float32),
        seed=np.array([args.seed], dtype=np.int64),
        batch_size=np.array([args.batch_size], dtype=np.int64),
        kmeans_epochs=np.array([args.kmeans_epochs], dtype=np.int64),
        device=np.array([str(device)]),
        pca_dim=np.array([args.pca_dim], dtype=np.int64),
        pca_explained_variance_ratio=np.array([explained if explained is not None else 1.0], dtype=np.float32),
        counts=counts,
        inertia=np.array([inertia], dtype=np.float32),
        pca_components=(
            components.detach().cpu().numpy().astype(np.float32)
            if components is not None
            else np.zeros((feature_dim, 0), dtype=np.float32)
        ),
    )

    largest_idx = np.argsort(counts)[::-1][:10]
    largest = [(int(i), int(counts[i])) for i in largest_idx if counts[i] > 0]
    print(f"Saved prototype bank to: {args.output}")
    print(f"Samples: {total_count} | feature_dim: {feature_dim} | clusters: {args.clusters}")
    print(f"Device: {device} | batch_size: {args.batch_size} | kmeans_epochs: {args.kmeans_epochs}")
    if explained is not None:
        print(f"PCA dim: {args.pca_dim} | explained_variance_ratio: {explained:.4f}")
    print(f"Final inertia: {inertia:.4f}")
    print(f"Largest clusters: {largest}")


if __name__ == "__main__":
    main()
