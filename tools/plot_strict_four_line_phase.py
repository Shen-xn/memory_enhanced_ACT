#!/usr/bin/env python3
"""Plot a strict 4-line single-task single-k figure.

Spec implemented exactly as requested:
- one task
- one k
- six joint subplots
- exactly four lines in each subplot:
  1) tgt(t)
  2) tgt(t) + sum(rescmd_0..k)
  3) tgt(t) + sum(protcmd_0..k)
  4) tgt(t+k+1)

No extra background trajectory, markers, or helper curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j10"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot strict four-line phase/prototype figure.")
    parser.add_argument("--task-dir", required=True, help="Single task directory containing states_filtered.csv and phase_pca16_targets.npz")
    parser.add_argument("--k", type=int, required=True, help="Largest predicted step index to accumulate")
    parser.add_argument(
        "--command-mode",
        choices=["delta", "absolute"],
        default="delta",
        help="Interpret protcmd/rescmd tensors as delta or absolute values",
    )
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=10.0,
        help="Multiply targets by this factor before accumulation when command-mode=delta",
    )
    parser.add_argument(
        "--phase-targets-name",
        default="phase_pca16_targets.npz",
        help="Per-task phase supervision filename",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to <task-dir>/strict_four_line_plots",
    )
    return parser


def load_task_states(task_dir: Path) -> np.ndarray:
    csv_path = task_dir / "states_filtered.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing states_filtered.csv: {csv_path}")
    df = pd.read_csv(csv_path)
    missing_cols = [name for name in JOINT_NAMES if name not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path} missing joint columns: {missing_cols}")
    qpos = df[JOINT_NAMES].to_numpy(dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"Unexpected qpos shape from {csv_path}: {qpos.shape}")
    return qpos


def load_phase_payload(task_dir: Path, filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload_path = task_dir / filename
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing phase targets file: {payload_path}")
    payload = np.load(payload_path)
    if "frame_index" not in payload or "residual_tgt" not in payload:
        raise ValueError(f"{payload_path} must contain frame_index and residual_tgt")

    if "pca_recon_tgt" in payload:
        base_tgt = payload["pca_recon_tgt"].astype(np.float32)
    else:
        raise ValueError(f"{payload_path} must contain pca_recon_tgt")

    frame_index = payload["frame_index"].astype(np.int64)
    residual_tgt = payload["residual_tgt"].astype(np.float32)

    if base_tgt.shape != residual_tgt.shape:
        raise ValueError(
            f"base target and residual_tgt shape mismatch: {base_tgt.shape} vs {residual_tgt.shape}"
        )
    if base_tgt.ndim != 3 or base_tgt.shape[2] != len(JOINT_NAMES):
        raise ValueError(f"Unexpected command tensor shape: {base_tgt.shape}")
    if len(frame_index) != base_tgt.shape[0]:
        raise ValueError(
            f"frame_index length {len(frame_index)} != command sample count {base_tgt.shape[0]}"
        )
    return frame_index, base_tgt, residual_tgt


def build_curves(
    qpos: np.ndarray,
    frame_index: np.ndarray,
    pca_recon_tgt: np.ndarray,
    residual_tgt: np.ndarray,
    k: int,
    command_mode: str,
    delta_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    future_steps = pca_recon_tgt.shape[1]
    if k < 0 or k >= future_steps:
        raise ValueError(f"k={k} is out of range for future_steps={future_steps}")

    # Full state trajectory is always defined on the whole task.
    x_tgt = np.arange(len(qpos), dtype=np.int32)
    tgt_t = qpos

    # Ground-truth shifted line follows the user's locked definition: tgt(t+k+1).
    gt_limit = len(qpos) - (k + 1)
    if gt_limit <= 0:
        raise ValueError(f"Task length {len(qpos)} is too short for k={k}")
    x_tgt_shift = np.arange(gt_limit, dtype=np.int32)
    tgt_tk1 = qpos[x_tgt_shift + k + 1]

    # Prediction-derived curves can only exist where the full future chunk was available.
    valid_mask = (frame_index >= 0) & (frame_index < len(qpos))
    x_pred = frame_index[valid_mask].astype(np.int32)
    if len(x_pred) == 0:
        raise ValueError(f"No prediction-aligned t remains for k={k} and task length={len(qpos)}")

    order = np.argsort(x_pred)
    x_pred = x_pred[order]
    base_t = qpos[x_pred]

    prototype_slice = pca_recon_tgt[valid_mask, : k + 1, :][order]
    residual_slice = residual_tgt[valid_mask, : k + 1, :][order]

    if command_mode == "delta":
        prototype_term = np.cumsum(prototype_slice * float(delta_scale), axis=1)[:, -1, :]
        residual_term = np.cumsum(residual_slice * float(delta_scale), axis=1)[:, -1, :]
    else:
        prototype_term = prototype_slice[:, -1, :]
        residual_term = residual_slice[:, -1, :]

    tgt_plus_proto = base_t + prototype_term
    tgt_plus_res = base_t + residual_term
    return x_tgt, tgt_t, x_pred, tgt_plus_res, tgt_plus_proto, x_tgt_shift, tgt_tk1


def plot_four_lines(
    task_name: str,
    k: int,
    x_tgt: np.ndarray,
    tgt_t: np.ndarray,
    x_pred: np.ndarray,
    tgt_plus_res: np.ndarray,
    tgt_plus_proto: np.ndarray,
    x_tgt_shift: np.ndarray,
    tgt_tk1: np.ndarray,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(len(JOINT_NAMES), 1, figsize=(15, 2.8 * len(JOINT_NAMES)), sharex=True)
    if len(JOINT_NAMES) == 1:
        axes = [axes]

    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        ax = axes[joint_idx]
        ax.plot(x_tgt, tgt_t[:, joint_idx], color="black", linewidth=1.8, label="tgt(t)")
        ax.plot(
            x_pred,
            tgt_plus_res[:, joint_idx],
            color="#ff7f0e",
            linewidth=1.8,
            label="tgt(t)+sum(rescmd_0..k)",
        )
        ax.plot(
            x_pred,
            tgt_plus_proto[:, joint_idx],
            color="#1f77b4",
            linewidth=1.8,
            label="tgt(t)+sum(protcmd_0..k)",
        )
        ax.plot(
            x_tgt_shift,
            tgt_tk1[:, joint_idx],
            color="#2ca02c",
            linewidth=1.8,
            linestyle="--",
            label=f"tgt(t+{k + 1})",
        )
        ax.set_ylabel(joint_name)
        ax.grid(True, alpha=0.25)
        if joint_idx == 0:
            ax.legend(loc="best", fontsize=8, ncol=2)

    axes[-1].set_xlabel("t")
    fig.suptitle(f"{task_name} | k={k} | strict four-line plot (four functions of t)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()
    task_dir = Path(args.task_dir).resolve()
    if not task_dir.is_dir():
        raise NotADirectoryError(f"Task directory does not exist: {task_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else task_dir / "strict_four_line_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    qpos = load_task_states(task_dir)
    frame_index, pca_recon_tgt, residual_tgt = load_phase_payload(task_dir, args.phase_targets_name)
    x_tgt, tgt_t, x_pred, tgt_plus_res, tgt_plus_proto, x_tgt_shift, tgt_tk1 = build_curves(
        qpos=qpos,
        frame_index=frame_index,
        pca_recon_tgt=pca_recon_tgt,
        residual_tgt=residual_tgt,
        k=args.k,
        command_mode=args.command_mode,
        delta_scale=args.delta_scale,
    )

    save_path = output_dir / f"{task_dir.name}_k{args.k}_four_lines.png"
    plot_four_lines(
        task_name=task_dir.name,
        k=args.k,
        x_tgt=x_tgt,
        tgt_t=tgt_t,
        x_pred=x_pred,
        tgt_plus_res=tgt_plus_res,
        tgt_plus_proto=tgt_plus_proto,
        x_tgt_shift=x_tgt_shift,
        tgt_tk1=tgt_tk1,
        save_path=save_path,
    )

    summary = {
        "task_dir": str(task_dir),
        "k": int(args.k),
        "command_mode": args.command_mode,
        "delta_scale": float(args.delta_scale),
        "tgt_t_start": int(x_tgt[0]),
        "tgt_t_end": int(x_tgt[-1]),
        "tgt_t_num_points": int(len(x_tgt)),
        "pred_start": int(x_pred[0]),
        "pred_end": int(x_pred[-1]),
        "pred_num_points": int(len(x_pred)),
        "tgt_shift_start": int(x_tgt_shift[0]),
        "tgt_shift_end": int(x_tgt_shift[-1]),
        "tgt_shift_num_points": int(len(x_tgt_shift)),
        "output_path": str(save_path),
    }
    summary_path = output_dir / f"{task_dir.name}_k{args.k}_four_lines_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
