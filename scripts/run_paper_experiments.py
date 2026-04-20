#!/usr/bin/env python3
"""Run all paper training variants serially with a dedicated run directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


EXPERIMENTS = [
    {"name": "paper_baseline_e{epochs}", "method": "baseline", "dim": None},
    {"name": "paper_pca8res_e{epochs}", "method": "pca-residual", "dim": 8},
    {"name": "paper_pca16res_e{epochs}", "method": "pca-residual", "dim": 16},
    {"name": "paper_pca32res_e{epochs}", "method": "pca-residual", "dim": 32},
    {"name": "paper_pca8only_e{epochs}", "method": "pca-only", "dim": 8},
    {"name": "paper_pca16only_e{epochs}", "method": "pca-only", "dim": 16},
    {"name": "paper_pca32only_e{epochs}", "method": "pca-only", "dim": 32},
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serial runner for paper ACT experiments.")
    parser.add_argument("--data-root", default="", help="Dataset root. Defaults to DATA_ROOT env or data_process/data.")
    parser.add_argument("--run-root", default="", help="Dedicated output root. Defaults to <repo>/paper_runs/run_<timestamp>.")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "25")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "16")))
    parser.add_argument("--log-print-freq", type=int, default=int(os.environ.get("LOG_PRINT_FREQ", "608")))
    parser.add_argument("--lr", default=os.environ.get("LR", "1e-5"))
    parser.add_argument("--lr-backbone", default=os.environ.get("LR_BACKBONE", "1e-6"))
    parser.add_argument("--kl-weight", default=os.environ.get("KL_WEIGHT", "1.0"))
    parser.add_argument("--recon-loss-weight", default=os.environ.get("RECON_LOSS_WEIGHT", "1.0"))
    parser.add_argument("--pca-coord-loss-weight", default=os.environ.get("PCA_COORD_LOSS_WEIGHT", "1.0"))
    parser.add_argument("--residual-loss-weight", default=os.environ.get("RESIDUAL_LOSS_WEIGHT", "1.0"))
    parser.add_argument("--qpos-noise-std", default=os.environ.get("QPOS_NOISE_STD", "2.0"))
    parser.add_argument("--qpos-noise-clip", default=os.environ.get("QPOS_NOISE_CLIP", "4.0"))
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch training.py.")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show tqdm progress bars. Disabled by default to keep runner logs readable.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Do not resume or skip existing experiment directories. Existing outputs may be overwritten/appended.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_root(root: Path, value: str) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("DATA_ROOT", "")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return root / "data_process" / "data"


def make_run_root(root: Path, value: str) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "paper_runs" / f"run_{timestamp}"


def validate_inputs(data_root: Path) -> None:
    if not data_root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")
    task_count = len([path for path in data_root.glob("task_*") if path.is_dir()])
    if task_count == 0:
        raise FileNotFoundError(f"No task_* directories found under DATA_ROOT: {data_root}")
    for dim in (8, 16, 32):
        bank = data_root / f"_phase_pca{dim}" / f"phase_pca{dim}_bank.npz"
        if not bank.exists():
            raise FileNotFoundError(f"Missing PCA bank for pca{dim}: {bank}")
    print(f"[runner] DATA_ROOT={data_root}")
    print(f"[runner] task_count={task_count}")


def command_for_experiment(args, root: Path, data_root: Path, log_root: Path, exp: dict) -> list[str]:
    exp_name = exp["name"].format(epochs=args.epochs)
    cmd = [
        args.python,
        str(root / "training.py"),
        "--method",
        exp["method"],
        "--data-root",
        str(data_root),
        "--log-root",
        str(log_root),
        "--exp-name",
        exp_name,
        "--num-epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--log-print-freq",
        str(args.log_print_freq),
        "--lr",
        str(args.lr),
        "--lr-backbone",
        str(args.lr_backbone),
        "--kl-weight",
        str(args.kl_weight),
        "--recon-loss-weight",
        str(args.recon_loss_weight),
        "--pca-coord-loss-weight",
        str(args.pca_coord_loss_weight),
        "--residual-loss-weight",
        str(args.residual_loss_weight),
        "--qpos-input-noise-std-pulse",
        str(args.qpos_noise_std),
        "--qpos-input-noise-clip-std",
        str(args.qpos_noise_clip),
    ]
    if not args.show_progress:
        cmd.append("--disable-progress")
    dim = exp["dim"]
    if dim is not None:
        cmd.extend(
            [
                "--phase-pca-dim",
                str(dim),
                "--phase-bank-path",
                str(data_root / f"_phase_pca{dim}" / f"phase_pca{dim}_bank.npz"),
                "--phase-targets-filename",
                f"phase_pca{dim}_targets.npz",
            ]
        )
    return cmd


def checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r"ckpt_epoch_(\d+)\.pth", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(exp_dir: Path) -> tuple[Path | None, int]:
    checkpoints = sorted(
        [path for path in exp_dir.glob("ckpt_epoch_*.pth") if checkpoint_epoch(path) >= 0],
        key=checkpoint_epoch,
    )
    if not checkpoints:
        return None, 0
    latest = checkpoints[-1]
    return latest, checkpoint_epoch(latest)


def is_experiment_complete(exp_dir: Path, target_epochs: int) -> bool:
    latest, epoch = find_latest_checkpoint(exp_dir)
    return latest is not None and epoch >= target_epochs and (exp_dir / "best_model.pth").exists()


def stream_process(cmd: list[str], log_path: Path, cwd: Path) -> int:
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def main() -> int:
    args = build_argparser().parse_args()
    root = repo_root()
    data_root = resolve_data_root(root, args.data_root)
    run_root = make_run_root(root, args.run_root)
    exp_log_root = run_root / "experiments"
    runner_log_root = run_root / "runner_logs"
    exp_log_root.mkdir(parents=True, exist_ok=True)
    runner_log_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"[runner] dry-run DATA_ROOT={data_root}")
    else:
        validate_inputs(data_root)

    run_config = {
        "data_root": str(data_root),
        "run_root": str(run_root),
        "exp_log_root": str(exp_log_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "log_print_freq": args.log_print_freq,
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "kl_weight": args.kl_weight,
        "recon_loss_weight": args.recon_loss_weight,
        "pca_coord_loss_weight": args.pca_coord_loss_weight,
        "residual_loss_weight": args.residual_loss_weight,
        "qpos_noise_std": args.qpos_noise_std,
        "qpos_noise_clip": args.qpos_noise_clip,
        "experiments": EXPERIMENTS,
    }
    (run_root / "runner_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = []
    for index, exp in enumerate(EXPERIMENTS, start=1):
        exp_name = exp["name"].format(epochs=args.epochs)
        exp_dir = exp_log_root / exp_name
        cmd = command_for_experiment(args, root, data_root, exp_log_root, exp)
        resume_ckpt = None
        resume_epoch = 0
        if not args.fresh:
            if is_experiment_complete(exp_dir, args.epochs):
                latest_ckpt, latest_epoch = find_latest_checkpoint(exp_dir)
                log_path = runner_log_root / f"{index:02d}_{exp_name}_skipped.log"
                message = {
                    "event": "skip_complete",
                    "experiment": exp_name,
                    "latest_checkpoint": str(latest_ckpt),
                    "latest_epoch": latest_epoch,
                    "target_epochs": args.epochs,
                }
                log_path.write_text(json.dumps(message, ensure_ascii=False, indent=2), encoding="utf-8")
                item = {
                    "index": index,
                    "name": exp_name,
                    "method": exp["method"],
                    "dim": exp["dim"],
                    "return_code": 0,
                    "status": "skipped_complete",
                    "resume_from_epoch": latest_epoch,
                    "stdout_log": str(log_path),
                    "experiment_dir": str(exp_dir),
                }
                summary.append(item)
                (run_root / "runner_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print("=" * 96)
                print(f"[runner] {index}/{len(EXPERIMENTS)} skip complete: {exp_name} epoch={latest_epoch}")
                continue

            resume_ckpt, resume_epoch = find_latest_checkpoint(exp_dir)
            if resume_ckpt is not None:
                cmd.extend(["--resume-ckpt-path", str(resume_ckpt)])
        log_path = runner_log_root / f"{index:02d}_{exp_name}_stdout.log"
        print("=" * 96)
        print(f"[runner] {index}/{len(EXPERIMENTS)} start: {exp_name}")
        if resume_ckpt is not None:
            print(f"[runner] resume: {resume_ckpt} epoch={resume_epoch}")
        print("[runner] command:", " ".join(cmd))
        if args.dry_run:
            return_code = 0
        else:
            return_code = stream_process(cmd, log_path, root)
        item = {
            "index": index,
            "name": exp_name,
            "method": exp["method"],
            "dim": exp["dim"],
            "return_code": return_code,
            "status": "finished" if return_code == 0 else "failed",
            "resume_from_checkpoint": str(resume_ckpt) if resume_ckpt is not None else "",
            "resume_from_epoch": resume_epoch,
            "stdout_log": str(log_path),
            "experiment_dir": str(exp_dir),
        }
        summary.append(item)
        (run_root / "runner_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if return_code != 0:
            print(f"[runner] FAILED: {exp_name} return_code={return_code}")
            print(f"[runner] summary written to: {run_root / 'runner_summary.json'}")
            return return_code
        print(f"[runner] finished: {exp_name}")

    print("=" * 96)
    print(f"[runner] all experiments finished. run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
