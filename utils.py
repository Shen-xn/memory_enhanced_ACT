import os
import json
import torch
import numpy as np
import logging
import random
import matplotlib.pyplot as plt
from datetime import datetime

# ===================== 日志配置 =====================
def setup_logger(log_dir, exp_name):
    """Create a per-experiment logger with both console and file handlers."""
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # Console output is intentionally lightweight for interactive monitoring.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(ch_formatter)
    
    # File output keeps level names because it becomes the long-term record.
    log_file = os.path.join(log_dir, f"train_{exp_name}.log")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fh_formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def _to_serializable(obj):
    """Convert numpy-heavy objects into plain JSON-serializable Python values."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_config_snapshot(config, save_dir, filename="config.json"):
    """Write a readable config snapshot for the current run."""
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, filename)
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("__")}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(config_dict), f, indent=2, ensure_ascii=False)
    return config_path


def append_metrics_record(save_dir, stage, epoch, metrics, **extra_fields):
    """Append one structured metrics record to metrics.jsonl."""
    os.makedirs(save_dir, exist_ok=True)
    record = {
        "epoch": epoch,
        "stage": stage,
        "metrics": _to_serializable(metrics),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    record.update(_to_serializable(extra_fields))
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics_path


def capture_rng_state():
    """Capture global RNG states for reproducible resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Restore global RNG states if they are present in a checkpoint."""
    if not state:
        return False

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    return True

# ===================== 可视化工具 =====================
def _plot_metric_series(ax, xs, ys, label, color, linewidth=2, linestyle="-", marker=None, alpha=1.0):
    """Plot one metric series while quietly skipping all-NaN inputs."""
    if not xs or not ys:
        return
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    valid = ~np.isnan(ys)
    if not np.any(valid):
        return
    ax.plot(
        xs[valid],
        ys[valid],
        label=label,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        alpha=alpha,
    )


def plot_training_curves(train_metrics, val_metrics, val_obst_metrics, save_path):
    """Render the standard training/validation curves used by this repo.

    `train_metrics` contains sparse intra-epoch points sampled every
    `LOG_PRINT_FREQ` batches. Validation metrics are epoch-level points.
    """
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    train_x = train_metrics.get("x", [])
    val_x = val_metrics.get("x", [])
    val_obst_x = val_obst_metrics.get("x", [])

    # Top panel: overall loss for train / val-normal / val-obstacle.
    ax1 = axes[0]
    _plot_metric_series(ax1, train_x, train_metrics.get("loss", []), label="Train Loss", color="blue", linewidth=1.8)
    _plot_metric_series(ax1, val_x, val_metrics.get("loss", []), label="Val (Normal) Loss", color="orange", linewidth=2.2, marker="o")
    _plot_metric_series(ax1, val_obst_x, val_obst_metrics.get("loss", []), label="Val (Obstacle) Loss", color="red", linewidth=2.2, marker="o")
    ax1.set_xlabel("Epoch Progress")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training/Validation Loss Curve")
    ax1.legend()
    ax1.grid(True)

    # Bottom panel: policy-specific detail metrics.
    ax2 = axes[1]
    if "l1" in train_metrics:
        # ACTPolicy
        _plot_metric_series(ax2, train_x, train_metrics.get("l1", []), label="Train L1 Loss", color="green", linewidth=1.8)
        _plot_metric_series(ax2, val_x, val_metrics.get("l1", []), label="Val (Normal) L1 Loss", color="purple", linewidth=2.2, marker="o")
        _plot_metric_series(ax2, val_obst_x, val_obst_metrics.get("l1", []), label="Val (Obstacle) L1 Loss", color="brown", linewidth=2.2, marker="o")

        ax2_twin = ax2.twinx()
        _plot_metric_series(ax2_twin, train_x, train_metrics.get("kl", []), label="Train KL Loss", color="cyan", linewidth=1.8, linestyle="--")
        _plot_metric_series(ax2_twin, val_x, val_metrics.get("kl", []), label="Val (Normal) KL Loss", color="magenta", linewidth=2.0, linestyle="--", marker="o")
        _plot_metric_series(ax2_twin, val_obst_x, val_obst_metrics.get("kl", []), label="Val (Obstacle) KL Loss", color="gray", linewidth=2.0, linestyle="--", marker="o")

        ax2.set_xlabel("Epoch Progress")
        ax2.set_ylabel("L1 Loss")
        ax2_twin.set_ylabel("KL Loss")
        # Merge legends from the left and right y-axes.
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax2.grid(True)
    else:
        # CNNMLPPolicy
        _plot_metric_series(ax2, train_x, train_metrics.get("mse", []), label="Train MSE Loss", color="green", linewidth=1.8)
        _plot_metric_series(ax2, val_x, val_metrics.get("mse", []), label="Val (Normal) MSE Loss", color="purple", linewidth=2.2, marker="o")
        _plot_metric_series(ax2, val_obst_x, val_obst_metrics.get("mse", []), label="Val (Obstacle) MSE Loss", color="brown", linewidth=2.2, marker="o")
        ax2.set_xlabel("Epoch Progress")
        ax2.set_ylabel("MSE Loss")
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ===================== 模型保存/加载 =====================
def save_checkpoint(epoch, model, optimizer, config, metrics, is_best, save_dir):
    """Save a full training checkpoint plus `best_model.pth` when applicable."""
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("__")},
        "metrics": metrics,
        "best_loss": metrics["best_loss"] if "best_loss" in metrics else metrics["loss"],
        "rng_state": capture_rng_state(),
    }
    
    # Always keep the epoch checkpoint.
    ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")
    torch.save(ckpt, ckpt_path)
    
    # Mirror the current checkpoint to a stable path for deployment/export scripts.
    if is_best:
        best_ckpt_path = os.path.join(save_dir, "best_model.pth")
        torch.save(ckpt, best_ckpt_path)
    
    return ckpt_path

def load_checkpoint(ckpt_path, model, optimizer):
    """Load model/optimizer state from a checkpoint and return the payload."""
    ckpt = torch.load(
        ckpt_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt

# ===================== 指标计算 =====================
def compute_metrics(loss_dict):
    """Convert a loss dict into plain scalar metrics for logging/serialization."""
    avg_metrics = {}
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            avg_metrics[k] = v.item()
        else:
            avg_metrics[k] = v
    return avg_metrics

def aggregate_metrics(metrics_list):
    """Average the same metric keys across a list of per-batch metric dicts."""
    agg_metrics = {}
    for k in metrics_list[0].keys():
        agg_metrics[k] = np.mean([m[k] for m in metrics_list])
    return agg_metrics
