import os
import json
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# ===================== 日志配置 =====================
def setup_logger(log_dir, exp_name):
    """设置日志记录器"""
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # 控制台输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(ch_formatter)
    
    # 文件输出
    log_file = os.path.join(log_dir, f"train_{exp_name}.log")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fh_formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def _to_serializable(obj):
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


def append_metrics_record(save_dir, stage, epoch, metrics):
    """Append one structured metrics record to metrics.jsonl."""
    os.makedirs(save_dir, exist_ok=True)
    record = {
        "epoch": epoch,
        "stage": stage,
        "metrics": _to_serializable(metrics),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics_path

# ===================== 可视化工具 =====================
def plot_training_curves(train_metrics, val_metrics, val_obst_metrics, save_path):
    """
    绘制训练/验证损失曲线
    Args:
        train_metrics: 训练指标字典，key为指标名，value为列表
        val_metrics: 普通轨迹验证指标
        val_obst_metrics: 障碍轨迹验证指标
        save_path: 保存路径
    """
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. 总损失曲线
    ax1 = axes[0]
    epochs = range(1, len(train_metrics["loss"]) + 1)
    ax1.plot(epochs, train_metrics["loss"], label="Train Loss", color="blue", linewidth=2)
    ax1.plot(epochs, val_metrics["loss"], label="Val (Normal) Loss", color="orange", linewidth=2)
    ax1.plot(epochs, val_obst_metrics["loss"], label="Val (Obstacle) Loss", color="red", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training/Validation Loss Curve")
    ax1.legend()
    ax1.grid(True)
    
    # 2. 细分指标（ACTPolicy显示L1+KL，CNNMLPPolicy显示MSE）
    ax2 = axes[1]
    if "l1" in train_metrics:
        # ACTPolicy
        ax2.plot(epochs, train_metrics["l1"], label="Train L1 Loss", color="green", linewidth=2)
        ax2.plot(epochs, val_metrics["l1"], label="Val (Normal) L1 Loss", color="purple", linewidth=2)
        ax2.plot(epochs, val_obst_metrics["l1"], label="Val (Obstacle) L1 Loss", color="brown", linewidth=2)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs, train_metrics["kl"], label="Train KL Loss", color="cyan", linewidth=2, linestyle="--")
        ax2_twin.plot(epochs, val_metrics["kl"], label="Val (Normal) KL Loss", color="magenta", linewidth=2, linestyle="--")
        ax2_twin.plot(epochs, val_obst_metrics["kl"], label="Val (Obstacle) KL Loss", color="gray", linewidth=2, linestyle="--")
        
        ax2.set_ylabel("L1 Loss")
        ax2_twin.set_ylabel("KL Loss")
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        # CNNMLPPolicy
        ax2.plot(epochs, train_metrics["mse"], label="Train MSE Loss", color="green", linewidth=2)
        ax2.plot(epochs, val_metrics["mse"], label="Val (Normal) MSE Loss", color="purple", linewidth=2)
        ax2.plot(epochs, val_obst_metrics["mse"], label="Val (Obstacle) MSE Loss", color="brown", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE Loss")
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ===================== 模型保存/加载 =====================
def save_checkpoint(epoch, model, optimizer, config, metrics, is_best, save_dir):
    """
    保存模型检查点
    Args:
        epoch: 当前轮数
        model: 模型实例
        optimizer: 优化器
        config: 配置实例
        metrics: 当前指标
        is_best: 是否为最优模型
        save_dir: 保存目录
    """
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("__")},
        "metrics": metrics,
        "best_loss": metrics["best_loss"] if "best_loss" in metrics else metrics["loss"]
    }
    
    # 保存普通ckpt
    ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")
    torch.save(ckpt, ckpt_path)
    
    # 保存最优模型
    if is_best:
        best_ckpt_path = os.path.join(save_dir, "best_model.pth")
        torch.save(ckpt, best_ckpt_path)
    
    return ckpt_path

def load_checkpoint(ckpt_path, model, optimizer):
    """
    加载模型检查点
    Args:
        ckpt_path: 检查点路径
        model: 模型实例
        optimizer: 优化器实例
    Returns:
        ckpt: 检查点字典
    """
    ckpt = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt

# ===================== 指标计算 =====================
def compute_metrics(loss_dict):
    """
    计算批次指标（平均）
    Args:
        loss_dict: 损失字典
    Returns:
        avg_metrics: 平均指标字典
    """
    avg_metrics = {}
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            avg_metrics[k] = v.item()
        else:
            avg_metrics[k] = v
    return avg_metrics

def aggregate_metrics(metrics_list):
    """
    聚合多个批次的指标
    Args:
        metrics_list: 指标列表（每个元素是批次指标字典）
    Returns:
        agg_metrics: 聚合后的指标字典（平均值）
    """
    agg_metrics = {}
    for k in metrics_list[0].keys():
        agg_metrics[k] = np.mean([m[k] for m in metrics_list])
    return agg_metrics
