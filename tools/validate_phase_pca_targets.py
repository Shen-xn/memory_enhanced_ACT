#!/usr/bin/env python3
"""Validate per-task PCA supervision alignment against states_filtered.csv."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_process.data_loader import JOINT_COLS


def build_argparser():
    parser = argparse.ArgumentParser(description="Validate phase PCA per-task supervision files.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target-filename", default="phase_pca16_targets.npz")
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--target-mode", choices=["absolute", "delta"], default="delta")
    parser.add_argument("--delta-qpos-scale", type=float, default=10.0)
    parser.add_argument("--output-json", default="")
    return parser


def _expected_future_targets(states, future_steps, target_mode, delta_qpos_scale):
    sample_count = len(states) - future_steps
    if sample_count <= 0:
        return np.zeros((0, future_steps, states.shape[1]), dtype=np.float32)

    targets = np.zeros((sample_count, future_steps, states.shape[1]), dtype=np.float32)
    for frame_idx in range(sample_count):
        future = states[frame_idx + 1 : frame_idx + 1 + future_steps]
        if target_mode == "absolute":
            targets[frame_idx] = future
            continue

        prev = states[frame_idx]
        for step_idx in range(future_steps):
            targets[frame_idx, step_idx] = (future[step_idx] - prev) / delta_qpos_scale
            prev = future[step_idx]
    return targets


def _task_dirs(data_root):
    for path in sorted(Path(data_root).glob("task*")):
        if path.is_dir() and "task_copy" not in path.name:
            yield path


def validate_task(task_dir, args):
    csv_path = task_dir / "states_filtered.csv"
    target_path = task_dir / args.target_filename
    if not csv_path.exists() or not target_path.exists():
        return {
            "task": task_dir.name,
            "ok": False,
            "error": f"missing {'states_filtered.csv' if not csv_path.exists() else args.target_filename}",
        }

    states = pd.read_csv(csv_path)[JOINT_COLS].to_numpy(np.float32)
    expected = _expected_future_targets(
        states=states,
        future_steps=args.future_steps,
        target_mode=args.target_mode,
        delta_qpos_scale=args.delta_qpos_scale,
    )
    payload = np.load(target_path)
    required_keys = {"frame_index", "pca_coord_tgt", "pca_recon_tgt", "residual_tgt"}
    missing_keys = sorted(required_keys - set(payload.files))
    if missing_keys:
        return {"task": task_dir.name, "ok": False, "error": f"missing keys: {missing_keys}"}

    frame_index = payload["frame_index"].astype(np.int64)
    pca_recon = payload["pca_recon_tgt"].astype(np.float32)
    residual = payload["residual_tgt"].astype(np.float32)
    sample_count = expected.shape[0]

    errors = []
    if len(frame_index) != sample_count:
        errors.append(f"sample_count expected={sample_count} got={len(frame_index)}")
    if len(np.unique(frame_index)) != len(frame_index):
        errors.append("duplicate frame_index values")
    if set(frame_index.tolist()) != set(range(sample_count)):
        missing = sorted(set(range(sample_count)) - set(frame_index.tolist()))[:8]
        extra = sorted(set(frame_index.tolist()) - set(range(sample_count)))[:8]
        errors.append(f"frame_index mismatch missing={missing} extra={extra}")
    if pca_recon.shape != expected.shape or residual.shape != expected.shape:
        errors.append(f"target shape mismatch expected={expected.shape} pca={pca_recon.shape} residual={residual.shape}")

    recon_max_abs = None
    recon_mean_abs = None
    sorted_by_frame = bool(np.array_equal(frame_index, np.arange(len(frame_index), dtype=np.int64)))
    if not errors:
        ordered = np.argsort(frame_index)
        reconstructed = pca_recon[ordered] + residual[ordered]
        delta = reconstructed - expected
        recon_max_abs = float(np.max(np.abs(delta)))
        recon_mean_abs = float(np.mean(np.abs(delta)))

    ok = not errors and recon_max_abs is not None and recon_max_abs < 1e-4
    return {
        "task": task_dir.name,
        "ok": ok,
        "sorted_by_frame": sorted_by_frame,
        "sample_count": int(sample_count),
        "reconstruction_max_abs": recon_max_abs,
        "reconstruction_mean_abs": recon_mean_abs,
        "error": "; ".join(errors),
    }


def main():
    args = build_argparser().parse_args()
    results = [validate_task(task_dir, args) for task_dir in _task_dirs(args.data_root)]
    failed = [item for item in results if not item["ok"]]
    unsorted = [item for item in results if item.get("ok") and not item.get("sorted_by_frame")]
    report = {
        "data_root": os.path.abspath(args.data_root),
        "target_filename": args.target_filename,
        "task_count": len(results),
        "failed_count": len(failed),
        "unsorted_count": len(unsorted),
        "failed": failed[:20],
        "unsorted_examples": [item["task"] for item in unsorted[:20]],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"summary": report, "tasks": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
