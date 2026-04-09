from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .me_block_config import MEBlockConfig, default_me_block_config, save_config
from .importance_dataset import ImportanceFrameDataset
from .memory_gate_model import ImportanceMemoryModel, checkpoint_payload


def parse_args(default_data_root: str = "", default_save_root: str = "") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the importance segmentation model used by the memory-image pipeline.")
    parser.add_argument("--data-root", type=str, default=default_data_root, help="Root containing task_* folders.")
    parser.add_argument("--save-root", type=str, default=default_save_root, help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gamma-min", type=float, default=None)
    parser.add_argument("--gamma-max", type=float, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--no-augmentation", action="store_true", help="Disable train-time image augmentation.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> MEBlockConfig:
    config = default_me_block_config()

    if args.data_root:
        config.training.data_root = args.data_root
        config.generation.data_root = args.data_root
    if args.save_root:
        config.training.save_root = args.save_root
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.seed is not None:
        config.training.seed = args.seed
    if args.gamma_min is not None:
        config.training.gamma_min = args.gamma_min
    if args.gamma_max is not None:
        config.training.gamma_max = args.gamma_max
    if args.noise_std is not None:
        config.training.noise_std = args.noise_std
    if args.no_augmentation:
        config.training.use_augmentation = False

    return config


def compute_mean_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, unlabeled_index: int) -> float:
    preds = logits.argmax(dim=1)
    valid_mask = targets != unlabeled_index
    if valid_mask.sum().item() == 0:
        return 0.0

    ious = []
    for class_index in range(num_classes):
        pred_mask = (preds == class_index) & valid_mask
        target_mask = (targets == class_index) & valid_mask
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue
        inter = (pred_mask & target_mask).sum().item()
        ious.append(inter / union)
    return float(sum(ious) / len(ious)) if ious else 0.0


def accumulate_foreground_recall_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_foreground_classes: int,
    unlabeled_index: int,
) -> tuple[list[int], list[int]]:
    preds = logits.argmax(dim=1)
    true_positive = []
    ground_truth = []
    for class_index in range(1, num_foreground_classes + 1):
        valid_gt = targets == class_index
        tp = ((preds == class_index) & valid_gt).sum().item()
        gt = valid_gt.sum().item()
        true_positive.append(int(tp))
        ground_truth.append(int(gt))
    return true_positive, ground_truth


def run_epoch(
    model: ImportanceMemoryModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    class_names: list[str],
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    stage_name = "train" if is_train else "val"
    losses = []
    ious = []
    tp_counts = [0 for _ in class_names]
    gt_counts = [0 for _ in class_names]

    progress = tqdm(loader, leave=False, desc=stage_name)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model.segmenter(images)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        ious.append(
            compute_mean_iou(
                logits.detach(),
                labels,
                model.config.importance.num_output_classes,
                model.config.importance.unlabeled_index,
            )
        )
        batch_tp, batch_gt = accumulate_foreground_recall_counts(
            logits.detach(),
            labels,
            model.config.importance.num_foreground_classes,
            model.config.importance.unlabeled_index,
        )
        tp_counts = [a + b for a, b in zip(tp_counts, batch_tp)]
        gt_counts = [a + b for a, b in zip(gt_counts, batch_gt)]
        progress.set_postfix(loss=f"{losses[-1]:.4f}", miou=f"{ious[-1]:.4f}")

    per_class = {}
    for name, tp, gt in zip(class_names, tp_counts, gt_counts):
        recall = float(tp / gt) if gt > 0 else 0.0
        per_class[name] = {"tp": int(tp), "gt": int(gt), "recall": recall}

    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "miou": float(sum(ious) / max(len(ious), 1)),
        "per_class": per_class,
    }


def append_metrics(log_path: str, epoch: int, stage: str, metrics: dict) -> None:
    record = {
        "epoch": epoch,
        "stage": stage,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_per_class_metrics(metrics: dict) -> str:
    per_class = metrics.get("per_class", {})
    parts = []
    for name, values in per_class.items():
        parts.append(f"{name}:{values['tp']}/{values['gt']}({values['recall']:.3f})")
    return " ".join(parts)


def compute_class_weights(train_loader: DataLoader, num_classes: int, unlabeled_index: int) -> torch.Tensor:
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for _images, labels in train_loader:
        labels = labels.numpy()
        valid = labels != unlabeled_index
        valid_labels = labels[valid]
        if valid_labels.size == 0:
            continue
        bincount = np.bincount(valid_labels.reshape(-1), minlength=num_classes)
        class_counts += bincount[:num_classes]

    class_counts = np.maximum(class_counts, 1.0)
    class_freq = class_counts / class_counts.sum()
    class_weights = 1.0 / np.sqrt(class_freq)
    class_weights = class_weights / class_weights.mean()
    return torch.tensor(class_weights, dtype=torch.float32)


def main(default_data_root: str = "", default_save_root: str = "") -> None:
    args = parse_args(default_data_root=default_data_root, default_save_root=default_save_root)
    config = resolve_config(args)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    run_name = datetime.now().strftime("importance_%Y%m%d_%H%M%S")
    save_dir = os.path.join(config.training.save_root, run_name)
    os.makedirs(save_dir, exist_ok=True)
    save_config(config, os.path.join(save_dir, "config.json"))

    torch.manual_seed(config.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.training.seed)

    train_dataset = ImportanceFrameDataset(config.training, config.importance, split="train")
    val_dataset = ImportanceFrameDataset(config.training, config.importance, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ImportanceMemoryModel(config=config).to(device)
    class_weights = compute_class_weights(
        train_loader,
        config.importance.num_output_classes,
        config.importance.unlabeled_index,
    ).to(device)
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")
    with open(os.path.join(save_dir, "class_weights.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "background": float(class_weights[0].item()),
                **{
                    name: float(class_weights[idx].item())
                    for idx, name in enumerate(config.importance.class_names, start=1)
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=config.importance.unlabeled_index,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    best_miou = -1.0
    metrics_path = os.path.join(save_dir, "metrics.jsonl")

    for epoch in range(1, config.training.num_epochs + 1):
        print(f"Epoch {epoch}/{config.training.num_epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, config.importance.class_names)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, config.importance.class_names)

        print(f"  train loss={train_metrics['loss']:.4f} miou={train_metrics['miou']:.4f}")
        print(f"  val   loss={val_metrics['loss']:.4f} miou={val_metrics['miou']:.4f}")
        print(f"  train recall {format_per_class_metrics(train_metrics)}")
        print(f"  val   recall {format_per_class_metrics(val_metrics)}")

        append_metrics(metrics_path, epoch, "train", train_metrics)
        append_metrics(metrics_path, epoch, "val", val_metrics)

        latest_payload = checkpoint_payload(model)
        latest_payload.update({"epoch": epoch, "train_metrics": train_metrics, "val_metrics": val_metrics})
        torch.save(latest_payload, os.path.join(save_dir, "latest_model.pth"))

        if val_metrics["miou"] >= best_miou:
            best_miou = val_metrics["miou"]
            torch.save(latest_payload, os.path.join(save_dir, "best_model.pth"))

    print(f"Training finished. Outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
