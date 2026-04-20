"""ACT training entry point."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from act.policy import ACTPolicy, CNNMLPPolicy
from config import cfg
from data_process.data_loader import get_data_loaders
from utils import (
    aggregate_metrics,
    append_metrics_record,
    compute_metrics,
    plot_training_curves,
    restore_rng_state,
    save_checkpoint,
    save_config_snapshot,
    setup_logger,
)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Train ACT / Phase-PCA ACT variants.")
    parser.add_argument(
        "--method",
        choices=("config", "baseline", "pca-residual", "pca-only"),
        default="config",
        help=(
            "Training variant. 'config' uses config.py as-is; baseline disables "
            "Phase-PCA; pca-residual enables PCA+residual; pca-only removes residual head."
        ),
    )
    parser.add_argument("--data-root", default="", help="Override cfg.DATA_ROOT.")
    parser.add_argument("--exp-name", default="", help="Override cfg.EXP_NAME.")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--recon-loss-weight", type=float, default=None)
    parser.add_argument("--pca-coord-loss-weight", type=float, default=None)
    parser.add_argument("--residual-loss-weight", type=float, default=None)
    parser.add_argument("--phase-pca-dim", type=int, default=None)
    parser.add_argument("--phase-bank-path", default="")
    parser.add_argument("--phase-targets-filename", default="")
    parser.add_argument("--use-phase-token", type=str_to_bool, default=None)
    parser.add_argument("--qpos-input-noise-std-pulse", type=float, default=None)
    parser.add_argument("--qpos-input-noise-clip-std", type=float, default=None)
    return parser


def apply_cli_overrides(config, args):
    if args.data_root:
        config.DATA_ROOT = args.data_root
    if args.exp_name:
        config.EXP_NAME = args.exp_name

    if args.method == "baseline":
        config.USE_PHASE_PCA_SUPERVISION = False
        config.USE_PHASE_TOKEN = False
        config.USE_RESIDUAL_ACTION = True
    elif args.method == "pca-residual":
        config.USE_PHASE_PCA_SUPERVISION = True
        config.USE_RESIDUAL_ACTION = True
    elif args.method == "pca-only":
        config.USE_PHASE_PCA_SUPERVISION = True
        config.USE_RESIDUAL_ACTION = False

    scalar_overrides = {
        "NUM_EPOCHS": args.num_epochs,
        "BATCH_SIZE": args.batch_size,
        "NUM_WORKERS": args.num_workers,
        "LR": args.lr,
        "LR_BACKBONE": args.lr_backbone,
        "KL_WEIGHT": args.kl_weight,
        "RECON_LOSS_WEIGHT": args.recon_loss_weight,
        "PCA_COORD_LOSS_WEIGHT": args.pca_coord_loss_weight,
        "RESIDUAL_LOSS_WEIGHT": args.residual_loss_weight,
        "PHASE_PCA_DIM": args.phase_pca_dim,
        "QPOS_INPUT_NOISE_STD_PULSE": args.qpos_input_noise_std_pulse,
        "QPOS_INPUT_NOISE_CLIP_STD": args.qpos_input_noise_clip_std,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            setattr(config, key, value)

    if args.use_phase_token is not None:
        config.USE_PHASE_TOKEN = bool(args.use_phase_token)

    if args.phase_targets_filename:
        config.PHASE_TARGETS_FILENAME = args.phase_targets_filename
    elif config.USE_PHASE_PCA_SUPERVISION:
        config.PHASE_TARGETS_FILENAME = f"phase_pca{int(config.PHASE_PCA_DIM)}_targets.npz"

    if args.phase_bank_path:
        config.PHASE_BANK_PATH = args.phase_bank_path
    elif config.USE_PHASE_PCA_SUPERVISION:
        config.PHASE_BANK_PATH = os.path.join(
            config.DATA_ROOT,
            f"_phase_pca{int(config.PHASE_PCA_DIM)}",
            f"phase_pca{int(config.PHASE_PCA_DIM)}_bank.npz",
        )

    config.refresh_model_params()
    return config


def get_device(config):
    return torch.device("cuda" if config.USE_CUDA else "cpu")


def set_global_seed(seed, use_cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def move_tensor_batch_to_device(batch_tensors, device):
    return [tensor.to(device, non_blocking=True) for tensor in batch_tensors]


def prepare_run_config(config):
    if config.TRAIN_MODE == "resume":
        if not os.path.exists(config.RESUME_CKPT_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {config.RESUME_CKPT_PATH}")
        ckpt = torch.load(config.RESUME_CKPT_PATH, map_location="cpu", weights_only=False)
        config.update_from_ckpt(ckpt["config"])
        return ckpt
    config.start_new_experiment()
    return None


def init_model_and_optimizer(config):
    device = get_device(config)
    args_override = {
        "lr": config.LR,
        "lr_backbone": config.LR_BACKBONE,
        "weight_decay": config.WEIGHT_DECAY,
        "kl_weight": config.KL_WEIGHT,
        "pca_coord_loss_weight": config.PCA_COORD_LOSS_WEIGHT,
        "residual_loss_weight": config.RESIDUAL_LOSS_WEIGHT,
        "recon_loss_weight": config.RECON_LOSS_WEIGHT,
        **config.MODEL_PARAMS,
    }
    if config.POLICY_CLASS == "ACTPolicy":
        policy = ACTPolicy(args_override, device=device)
    elif config.POLICY_CLASS == "CNNMLPPolicy":
        policy = CNNMLPPolicy(args_override, device=device)
    else:
        raise ValueError(f"Unsupported policy class: {config.POLICY_CLASS}")
    optimizer = policy.configure_optimizers()
    return policy, optimizer


def train_one_epoch(model, train_loader, optimizer, epoch, config, logger):
    device = get_device(config)
    model.train()
    train_metrics_list = []
    train_curve_points = []
    total_batches = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=total_batches, desc=f"Train Epoch {epoch}")
    for batch_idx, batch in pbar:
        imgs, currs, futures, pca_coord_tgts, residual_tgts, pca_recon_tgts, obsts = batch
        imgs, currs, futures, pca_coord_tgts, residual_tgts, pca_recon_tgts = move_tensor_batch_to_device(
            (imgs, currs, futures, pca_coord_tgts, residual_tgts, pca_recon_tgts), device
        )

        optimizer.zero_grad()
        if config.POLICY_CLASS == "ACTPolicy":
            is_pad = torch.zeros((futures.shape[0], futures.shape[1]), dtype=torch.bool, device=futures.device)
            loss_dict = model(
                qpos=currs,
                image=imgs,
                pca_coord_targets=pca_coord_tgts,
                residual_targets=residual_tgts,
                actions=futures,
                is_pad=is_pad,
            )
        else:
            loss_dict = model(qpos=currs, image=imgs, actions=futures[:, 0])

        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        batch_metrics = compute_metrics(loss_dict)
        train_metrics_list.append(batch_metrics)

        if (batch_idx + 1) % config.LOG_PRINT_FREQ == 0 or batch_idx == total_batches - 1:
            log_str = f"Epoch {epoch} | Batch {batch_idx + 1}/{total_batches} | "
            log_str += " | ".join([f"{k}: {v:.4f}" for k, v in batch_metrics.items()])
            logger.info(log_str)
            pbar.set_postfix(**batch_metrics)
            progress_x = (epoch - 1) + (batch_idx + 1) / max(total_batches, 1)
            train_curve_points.append({"x": progress_x, **batch_metrics})
            append_metrics_record(
                config.EXP_LOG_DIR,
                "train_log",
                epoch,
                batch_metrics,
                batch=batch_idx + 1,
                total_batches=total_batches,
                x=progress_x,
            )

    train_metrics = aggregate_metrics(train_metrics_list)
    logger.info(f"===== Train Epoch {epoch} Summary =====")
    logger.info(" | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
    append_metrics_record(config.EXP_LOG_DIR, "train", epoch, train_metrics)
    return train_metrics, train_curve_points


def validate(model, val_loader, config, logger, epoch, is_obst=False):
    device = get_device(config)
    model.eval()
    val_metrics_list = []
    total_batches = len(val_loader)
    pbar = tqdm(enumerate(val_loader), total=total_batches, desc=f"Val (Obst={is_obst}) Epoch")
    with torch.no_grad():
        for _, batch in pbar:
            imgs, currs, futures, pca_coord_tgts, residual_tgts, pca_recon_tgts, obsts = batch
            mask = obsts.squeeze(1).bool() if is_obst else ~obsts.squeeze(1).bool()
            if mask.sum() == 0:
                continue
            imgs = imgs[mask]
            currs = currs[mask]
            futures = futures[mask]
            pca_coord_tgts = pca_coord_tgts[mask]
            residual_tgts = residual_tgts[mask]

            imgs, currs, futures, pca_coord_tgts, residual_tgts = move_tensor_batch_to_device(
                (imgs, currs, futures, pca_coord_tgts, residual_tgts), device
            )

            if config.POLICY_CLASS == "ACTPolicy":
                is_pad = torch.zeros((futures.shape[0], futures.shape[1]), dtype=torch.bool, device=futures.device)
                loss_dict = model(
                    qpos=currs,
                    image=imgs,
                    pca_coord_targets=pca_coord_tgts,
                    residual_targets=residual_tgts,
                    actions=futures,
                    is_pad=is_pad,
                )
            else:
                loss_dict = model(qpos=currs, image=imgs, actions=futures[:, 0])

            batch_metrics = compute_metrics(loss_dict)
            val_metrics_list.append(batch_metrics)
            pbar.set_postfix(**batch_metrics)

    if not val_metrics_list:
        stage = "val_obst" if is_obst else "val"
        raise ValueError(f"{stage} has no available samples.")

    val_metrics = aggregate_metrics(val_metrics_list)
    stage = "val_obst" if is_obst else "val"
    logger.info(f"===== {stage} Summary =====")
    logger.info(" | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
    append_metrics_record(config.EXP_LOG_DIR, stage, epoch, val_metrics)
    return val_metrics


def nan_metrics_like(metrics):
    return {key: float("nan") for key in metrics.keys()}


def main():
    args = build_argparser().parse_args()
    apply_cli_overrides(cfg, args)
    resume_ckpt = prepare_run_config(cfg)
    set_global_seed(cfg.SEED, cfg.USE_CUDA)

    train_loader, val_loader = get_data_loaders(
        data_root=cfg.DATA_ROOT,
        future_steps=cfg.FUTURE_STEPS,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        image_channels=cfg.IMAGE_CHANNELS,
        target_mode="delta" if cfg.PREDICT_DELTA_QPOS else "absolute",
        delta_qpos_scale=cfg.DELTA_QPOS_SCALE,
        phase_targets_filename=cfg.PHASE_TARGETS_FILENAME,
        require_phase_targets=cfg.USE_PHASE_PCA_SUPERVISION,
        phase_pca_dim=cfg.PHASE_PCA_DIM,
        qpos_input_noise_std_pulse=cfg.QPOS_INPUT_NOISE_STD_PULSE,
        qpos_input_noise_clip_std=cfg.QPOS_INPUT_NOISE_CLIP_STD,
    )

    model, optimizer = init_model_and_optimizer(cfg)
    start_epoch = 1
    best_val_loss = float("inf")
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val_loss = resume_ckpt["best_loss"]
        restore_rng_state(resume_ckpt.get("rng_state"))

    os.makedirs(cfg.EXP_LOG_DIR, exist_ok=True)
    logger = setup_logger(cfg.EXP_LOG_DIR, cfg.EXP_NAME)
    save_config_snapshot(cfg, cfg.EXP_LOG_DIR)

    logger.info(f"===== 实验目录: {cfg.EXP_LOG_DIR} =====")
    val_has_obstacle = bool(getattr(val_loader.dataset, "split_has_obstacle_samples", False))
    if cfg.POLICY_CLASS == "ACTPolicy":
        tracked_metric_keys = ["loss", "recon_l1", "kl"]
        if cfg.USE_PHASE_PCA_SUPERVISION:
            tracked_metric_keys = ["loss", "recon_l1", "residual_l1", "pca_coord_mse", "kl"]
    else:
        tracked_metric_keys = ["loss", "mse"]
    train_metrics_history = {"x": [], **{k: [] for k in tracked_metric_keys}}
    val_metrics_history = {"x": [], **{k: [] for k in tracked_metric_keys}}
    val_obst_metrics_history = {"x": [], **{k: [] for k in tracked_metric_keys}}

    logger.info("===== 开始训练 =====")
    original_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        is_best = False
        val_metrics = None
        val_obst_metrics = None

        if epoch == 1 and resume_ckpt is None:
            warmup_factor = 1.0
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = original_lrs[i] * warmup_factor
        elif epoch == 2 and resume_ckpt is None:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = original_lrs[i]

        train_metrics, train_curve_points = train_one_epoch(model, train_loader, optimizer, epoch, cfg, logger)
        for point in train_curve_points:
            train_metrics_history["x"].append(point["x"])
            for key in tracked_metric_keys:
                train_metrics_history[key].append(point.get(key, float("nan")))

        if epoch % cfg.VAL_FREQ == 0 or epoch == cfg.NUM_EPOCHS:
            val_metrics = validate(model, val_loader, cfg, logger, epoch, is_obst=False)
            if val_has_obstacle:
                val_obst_metrics = validate(model, val_loader, cfg, logger, epoch, is_obst=True)
            else:
                val_obst_metrics = nan_metrics_like(val_metrics)

            val_metrics_history["x"].append(float(epoch))
            val_obst_metrics_history["x"].append(float(epoch))
            for key in tracked_metric_keys:
                val_metrics_history[key].append(val_metrics.get(key, float("nan")))
                val_obst_metrics_history[key].append(val_obst_metrics.get(key, float("nan")))

            if cfg.SAVE_PLOT:
                plot_training_curves(train_metrics_history, val_metrics_history, val_obst_metrics_history, cfg.EXP_LOG_DIR)

            current_val_loss = val_metrics["loss"]
            is_best = current_val_loss < best_val_loss
            if is_best:
                best_val_loss = current_val_loss
                logger.info(f"===== 最优模型更新 (Epoch {epoch}) | Best Loss: {best_val_loss:.4f} =====")

        if epoch % cfg.SAVE_FREQ == 0 or epoch == cfg.NUM_EPOCHS or is_best:
            metrics = {"train": train_metrics, "val": val_metrics, "val_obst": val_obst_metrics, "best_loss": best_val_loss}
            ckpt_path = save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=cfg,
                metrics=metrics,
                is_best=is_best,
                save_dir=cfg.EXP_LOG_DIR,
            )
            logger.info(f"===== 保存检查点: {ckpt_path} =====")

    logger.info("===== 训练完成 =====")


if __name__ == "__main__":
    main()
