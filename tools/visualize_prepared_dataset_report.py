#!/usr/bin/env python3
"""Build paper-friendly sanity plots for a prepared ACT dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.plot_strict_four_line_phase import (
    build_curves,
    load_phase_payload,
    load_task_states,
    plot_four_lines,
)


JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j10"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize prepared ACT data and Phase-PCA targets.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pca-dims", default="8,16,32")
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--delta-scale", type=float, default=10.0)
    parser.add_argument("--example-task-count", type=int, default=4)
    parser.add_argument("--example-k", default="0,2,5,9")
    return parser


def natural_task_dirs(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.glob("task_*") if path.is_dir())


def load_all_states(task_dirs: list[Path]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    per_task = {}
    all_qpos = []
    for task_dir in task_dirs:
        csv_path = task_dir / "states_filtered.csv"
        df = pd.read_csv(csv_path)
        qpos = df[JOINT_NAMES].to_numpy(dtype=np.float32)
        per_task[task_dir.name] = qpos
        all_qpos.append(qpos)
    return per_task, np.concatenate(all_qpos, axis=0)


def save_frame_count_plot(task_dirs: list[Path], per_task: dict[str, np.ndarray], output_dir: Path) -> dict:
    lengths = np.asarray([len(per_task[task.name]) for task in task_dirs], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=40, color="#3b6f8f", alpha=0.85)
    ax.set_title("Filtered trajectory length distribution")
    ax.set_xlabel("frames per task")
    ax.set_ylabel("task count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_frame_count_hist.png", dpi=220)
    plt.close(fig)
    return {
        "task_count": int(len(lengths)),
        "total_frames": int(lengths.sum()),
        "min_frames": int(lengths.min()),
        "max_frames": int(lengths.max()),
        "mean_frames": float(lengths.mean()),
        "median_frames": float(np.median(lengths)),
    }


def save_joint_plots(all_qpos: np.ndarray, output_dir: Path) -> dict:
    summary = {}
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot([all_qpos[:, i] for i in range(len(JOINT_NAMES))], labels=JOINT_NAMES, showfliers=False)
    ax.set_title("Joint pulse distribution after preprocessing")
    ax.set_ylabel("pulse")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "joint_pulse_boxplot.png", dpi=220)
    plt.close(fig)

    deltas = np.diff(all_qpos, axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for idx, ax in enumerate(axes.ravel()):
        ax.hist(deltas[:, idx], bins=80, color="#8f5b3b", alpha=0.85)
        ax.set_title(f"{JOINT_NAMES[idx]} frame delta")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "joint_frame_delta_hist.png", dpi=220)
    plt.close(fig)

    for idx, name in enumerate(JOINT_NAMES):
        summary[name] = {
            "min": float(all_qpos[:, idx].min()),
            "max": float(all_qpos[:, idx].max()),
            "mean": float(all_qpos[:, idx].mean()),
            "std": float(all_qpos[:, idx].std()),
            "mean_abs_frame_delta": float(np.abs(deltas[:, idx]).mean()),
        }
    return summary


def load_pca_bank(data_root: Path, dim: int) -> tuple[Path, dict]:
    bank_path = data_root / f"_phase_pca{dim}" / f"phase_pca{dim}_bank.npz"
    if not bank_path.exists():
        raise FileNotFoundError(f"Missing PCA bank: {bank_path}")
    bank = np.load(bank_path, allow_pickle=True)
    summary = {
        "bank_path": str(bank_path),
        "pca_dim": int(bank["pca_dim"][0]),
        "explained_ratio": float(bank["explained_ratio"][0]),
        "coord_mean_abs_max": float(np.abs(bank["pca_coord_mean"]).max()),
        "coord_std_min": float(bank["pca_coord_std"].min()),
        "coord_std_max": float(bank["pca_coord_std"].max()),
        "residual_std_mean": float(bank["residual_std"].mean()),
    }
    return bank_path, summary


def collect_phase_residual_stats(task_dirs: list[Path], target_name: str) -> tuple[np.ndarray, np.ndarray]:
    residuals = []
    coords = []
    for task_dir in task_dirs:
        payload = np.load(task_dir / target_name)
        residuals.append(payload["residual_tgt"].astype(np.float32))
        coords.append(payload["pca_coord_tgt"].astype(np.float32))
    return np.concatenate(residuals, axis=0), np.concatenate(coords, axis=0)


def save_pca_plots(task_dirs: list[Path], data_root: Path, dims: list[int], output_dir: Path) -> dict:
    summaries = {}
    explained = []
    for dim in dims:
        _, bank_summary = load_pca_bank(data_root, dim)
        target_name = f"phase_pca{dim}_targets.npz"
        residuals, coords = collect_phase_residual_stats(task_dirs, target_name)
        residual_l1 = np.mean(np.abs(residuals), axis=(0, 1))
        coord_std = coords.std(axis=0)
        bank_summary["residual_l1_per_joint"] = {
            name: float(residual_l1[idx]) for idx, name in enumerate(JOINT_NAMES)
        }
        summaries[f"pca{dim}"] = bank_summary
        explained.append((dim, bank_summary["explained_ratio"]))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(JOINT_NAMES, residual_l1, color="#4c7c59")
        ax.set_title(f"PCA{dim} residual target L1 per joint")
        ax.set_ylabel("L1 in scaled-delta units")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"pca{dim}_residual_l1_per_joint.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(dim), coord_std, marker="o", linewidth=1.8)
        ax.set_title(f"PCA{dim} coordinate std")
        ax.set_xlabel("component")
        ax.set_ylabel("std")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"pca{dim}_coord_std.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(dim) for dim, _ in explained], [ratio for _, ratio in explained], color="#6f4c8f")
    ax.set_ylim(0, 1.0)
    ax.set_title("PCA retained variance ratio")
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("explained variance ratio")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, (_, ratio) in enumerate(explained):
        ax.text(idx, ratio + 0.015, f"{ratio:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_explained_ratio.png", dpi=220)
    plt.close(fig)
    return summaries


def choose_example_tasks(task_dirs: list[Path], per_task: dict[str, np.ndarray], count: int) -> list[Path]:
    ranked = sorted(task_dirs, key=lambda task: len(per_task[task.name]), reverse=True)
    if count <= 0:
        return []
    if len(ranked) <= count:
        return ranked
    indices = np.linspace(0, len(ranked) - 1, count, dtype=int)
    return [ranked[int(i)] for i in indices]


def save_decomposition_examples(
    example_tasks: list[Path],
    dims: list[int],
    ks: list[int],
    output_dir: Path,
    delta_scale: float,
) -> dict:
    result = {}
    for dim in dims:
        dim_dir = output_dir / f"strict_four_line_pca{dim}"
        dim_dir.mkdir(parents=True, exist_ok=True)
        target_name = f"phase_pca{dim}_targets.npz"
        result[f"pca{dim}"] = []
        for task_dir in example_tasks:
            qpos = load_task_states(task_dir)
            frame_index, pca_recon_tgt, residual_tgt = load_phase_payload(task_dir, target_name)
            for k in ks:
                x_tgt, tgt_t, x_pred, tgt_plus_res, tgt_plus_proto, x_tgt_shift, tgt_tk1 = build_curves(
                    qpos=qpos,
                    frame_index=frame_index,
                    pca_recon_tgt=pca_recon_tgt,
                    residual_tgt=residual_tgt,
                    k=k,
                    command_mode="delta",
                    delta_scale=delta_scale,
                )
                save_path = dim_dir / f"{task_dir.name}_k{k}_four_lines.png"
                plot_four_lines(
                    task_name=task_dir.name,
                    k=k,
                    x_tgt=x_tgt,
                    tgt_t=tgt_t,
                    x_pred=x_pred,
                    tgt_plus_res=tgt_plus_res,
                    tgt_plus_proto=tgt_plus_proto,
                    x_tgt_shift=x_tgt_shift,
                    tgt_tk1=tgt_tk1,
                    save_path=save_path,
                )
                result[f"pca{dim}"].append(str(save_path))
    return result


def main() -> None:
    args = build_argparser().parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_root / "_paper_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    dims = [int(item) for item in args.pca_dims.split(",") if item.strip()]
    ks = [int(item) for item in args.example_k.split(",") if item.strip()]

    task_dirs = natural_task_dirs(data_root)
    per_task, all_qpos = load_all_states(task_dirs)
    summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "future_steps": int(args.future_steps),
        "delta_scale": float(args.delta_scale),
    }
    summary["frames"] = save_frame_count_plot(task_dirs, per_task, output_dir)
    summary["joints"] = save_joint_plots(all_qpos, output_dir)
    summary["pca"] = save_pca_plots(task_dirs, data_root, dims, output_dir)
    example_tasks = choose_example_tasks(task_dirs, per_task, args.example_task_count)
    summary["example_tasks"] = [task.name for task in example_tasks]
    summary["decomposition_plots"] = save_decomposition_examples(
        example_tasks=example_tasks,
        dims=dims,
        ks=ks,
        output_dir=output_dir,
        delta_scale=args.delta_scale,
    )
    summary_path = output_dir / "prepared_dataset_report_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
