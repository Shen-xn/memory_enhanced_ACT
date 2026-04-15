"""ACT training entry point.

The training loop intentionally stays thin: config owns hyperparameters,
dataloader owns data alignment/normalization, and policy owns model-specific
loss computation. That separation is important when adding new visual modes.
"""

import os
import torch
import numpy as np
from tqdm import tqdm

# 导入自定义模块
from config import cfg
from utils import (
    setup_logger,
    plot_training_curves,
    save_checkpoint,
    compute_metrics,
    aggregate_metrics,
    save_config_snapshot,
    append_metrics_record,
    restore_rng_state,
)
from data_process.data_loader import get_data_loaders
from act.policy import ACTPolicy, CNNMLPPolicy

def get_device(config):
    """Return the target device for the current run."""
    return torch.device("cuda" if config.USE_CUDA else "cpu")


def set_global_seed(seed, use_cuda):
    """Initialize reproducible RNG states for a fresh process."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def move_tensor_batch_to_device(batch_tensors, device):
    """Move a flat tuple/list of tensors onto the same device."""
    return [tensor.to(device, non_blocking=True) for tensor in batch_tensors]


def prepare_run_config(config):
    """
    Finalize run metadata before training starts.
    For resume, restore the original run config from checkpoint.
    For fresh training, create a new experiment name/path.
    """
    if config.TRAIN_MODE == "resume":
        if not os.path.exists(config.RESUME_CKPT_PATH):
            raise FileNotFoundError(f"断点续训文件不存在: {config.RESUME_CKPT_PATH}")
        ckpt = torch.load(config.RESUME_CKPT_PATH, map_location="cpu", weights_only=False)
        config.update_from_ckpt(ckpt["config"])
        return ckpt

    config.start_new_experiment()
    return None


def init_model_and_optimizer(config):
    """Build the selected policy and return its optimizer."""
    device = get_device(config)
    args_override = {
        "lr": config.LR,
        "lr_backbone": config.LR_BACKBONE,
        "weight_decay": config.WEIGHT_DECAY,
        "kl_weight": config.KL_WEIGHT,
        **config.MODEL_PARAMS
    }
    
    if config.POLICY_CLASS == "ACTPolicy":
        policy = ACTPolicy(args_override, device=device)
    elif config.POLICY_CLASS == "CNNMLPPolicy":
        policy = CNNMLPPolicy(args_override, device=device)
    else:
        raise ValueError(f"不支持的策略类型: {config.POLICY_CLASS}")
    
    optimizer = policy.configure_optimizers()
    return policy, optimizer

def train_one_epoch(model, train_loader, optimizer, epoch, config, logger):
    """Train ACT for one epoch and persist aggregate metrics."""
    device = get_device(config)
    model.train()
    train_metrics_list = []
    train_curve_points = []
    total_batches = len(train_loader)
    
    pbar = tqdm(enumerate(train_loader), total=total_batches, desc=f"Train Epoch {epoch}")
    for batch_idx, (imgs, currs, futures, m_imgs, obsts) in pbar:
        tensors = (imgs, currs, futures, m_imgs) if config.USE_MEMORY_IMAGE_INPUT else (imgs, currs, futures)
        moved = move_tensor_batch_to_device(tensors, device)
        imgs, currs, futures = moved[:3]
        if config.USE_MEMORY_IMAGE_INPUT:
            m_imgs = moved[3]
        
        # 前向传播
        optimizer.zero_grad()
        if config.POLICY_CLASS == "ACTPolicy":
            # ACTPolicy需要memory_image和is_pad（这里is_pad设为全False）
            is_pad = torch.zeros((futures.shape[0], futures.shape[1]), dtype=torch.bool, device=futures.device)
            model_kwargs = {
                "qpos": currs,
                "image": imgs,
                "actions": futures,
                "is_pad": is_pad,
            }
            if config.USE_MEMORY_IMAGE_INPUT:
                model_kwargs["memory_image"] = m_imgs
            loss_dict = model(**model_kwargs)
        else:
            # CNNMLPPolicy
            loss_dict = model(
                qpos=currs,
                image=imgs,
                actions=futures[:, 0]  # 只取第一个未来步
            )
        
        # 反向传播
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        
        # 记录指标
        batch_metrics = compute_metrics(loss_dict)
        train_metrics_list.append(batch_metrics)
        
        # 打印日志
        if (batch_idx + 1) % config.LOG_PRINT_FREQ == 0 or batch_idx == total_batches - 1:
            log_str = f"Epoch {epoch} | Batch {batch_idx+1}/{total_batches} | "
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
    
    # 聚合训练指标
    train_metrics = aggregate_metrics(train_metrics_list)
    logger.info(f"===== Train Epoch {epoch} Summary =====")
    logger.info(" | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
    append_metrics_record(config.EXP_LOG_DIR, "train", epoch, train_metrics)
    return train_metrics, train_curve_points

def validate(model, val_loader, config, logger, epoch, is_obst=False):
    """Validate on normal or obstacle trajectories only."""
    device = get_device(config)
    model.eval()
    val_metrics_list = []
    total_batches = len(val_loader)
    
    pbar = tqdm(enumerate(val_loader), total=total_batches, desc=f"Val (Obst={is_obst}) Epoch")
    with torch.no_grad():
        for batch_idx, (imgs, currs, futures, m_imgs, obsts) in pbar:
            # 筛选障碍/普通轨迹
            if is_obst:
                mask = obsts.squeeze(1).bool()
                if mask.sum() == 0:
                    continue
                imgs = imgs[mask]
                currs = currs[mask]
                futures = futures[mask]
                m_imgs = m_imgs[mask]
            else:
                mask = ~obsts.squeeze(1).bool()
                if mask.sum() == 0:
                    continue
                imgs = imgs[mask]
                currs = currs[mask]
                futures = futures[mask]
                m_imgs = m_imgs[mask]
            
            if len(imgs) == 0:
                continue
            
            tensors = (imgs, currs, futures, m_imgs) if config.USE_MEMORY_IMAGE_INPUT else (imgs, currs, futures)
            moved = move_tensor_batch_to_device(tensors, device)
            imgs, currs, futures = moved[:3]
            if config.USE_MEMORY_IMAGE_INPUT:
                m_imgs = moved[3]
            
            # 前向传播
            if config.POLICY_CLASS == "ACTPolicy":
                is_pad = torch.zeros((futures.shape[0], futures.shape[1]), dtype=torch.bool, device=futures.device)
                model_kwargs = {
                    "qpos": currs,
                    "image": imgs,
                    "actions": futures,
                    "is_pad": is_pad,
                }
                if config.USE_MEMORY_IMAGE_INPUT:
                    model_kwargs["memory_image"] = m_imgs
                loss_dict = model(**model_kwargs)
            else:
                loss_dict = model(
                    qpos=currs,
                    image=imgs,
                    actions=futures[:, 0]
                )
            
            # 记录指标
            batch_metrics = compute_metrics(loss_dict)
            val_metrics_list.append(batch_metrics)
            pbar.set_postfix(**batch_metrics)
    
    if len(val_metrics_list) == 0:
        stage = "val_obst" if is_obst else "val"
        raise ValueError(
            f"{stage} 没有可验证样本。请检查 train/val 任务划分，以及验证集中是否包含对应类型轨迹。"
        )

    # 聚合验证指标
    val_metrics = aggregate_metrics(val_metrics_list)
    
    # 打印日志
    stage = "val_obst" if is_obst else "val"
    logger.info(f"===== Val (Obst={is_obst}) Summary =====")
    logger.info(" | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
    append_metrics_record(config.EXP_LOG_DIR, stage, epoch, val_metrics)
    return val_metrics


def nan_metrics_like(metrics):
    return {key: float("nan") for key in metrics.keys()}

def main():
    """Run a fresh or resumed ACT experiment."""
    resume_ckpt = prepare_run_config(cfg)
    set_global_seed(cfg.SEED, cfg.USE_CUDA)

    # 1. 初始化数据加载器
    train_loader, val_loader = get_data_loaders(
        data_root=cfg.DATA_ROOT,
        future_steps=cfg.FUTURE_STEPS,
        use_memory_image_input=cfg.USE_MEMORY_IMAGE_INPUT,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        image_channels=cfg.IMAGE_CHANNELS,
    )
    
    # 2. 初始化模型和优化器
    model, optimizer = init_model_and_optimizer(cfg)
    start_epoch = 1
    best_val_loss = float("inf")
    
    # 3. 断点续训处理
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val_loss = resume_ckpt["best_loss"]
        rng_restored = restore_rng_state(resume_ckpt.get("rng_state"))
    else:
        rng_restored = False

    os.makedirs(cfg.EXP_LOG_DIR, exist_ok=True)
    logger = setup_logger(cfg.EXP_LOG_DIR, cfg.EXP_NAME)
    save_config_snapshot(cfg, cfg.EXP_LOG_DIR)

    logger.info(f"===== 实验目录: {cfg.EXP_LOG_DIR} =====")
    logger.info("===== 加载数据集完成 =====")
    logger.info("===== 初始化模型完成 =====")
    if resume_ckpt is not None:
        logger.info(f"===== 加载断点: {cfg.RESUME_CKPT_PATH} =====")
        logger.info(f"断点续训，从Epoch {start_epoch} 开始")
        if rng_restored:
            logger.info("已恢复 RNG 状态，后续 batch 打乱顺序会更接近未中断训练。")
        else:
            logger.info("该断点不含 RNG 状态，续训后 batch 顺序不会严格接续上次训练。")
    
    # 4. 训练过程记录
    tracked_metric_keys = ["loss"] + (["l1", "kl"] if cfg.POLICY_CLASS == "ACTPolicy" else ["mse"])
    train_metrics_history = {"x": [], **{k: [] for k in tracked_metric_keys}}
    val_metrics_history = {"x": [], **{k: [] for k in tracked_metric_keys}}
    val_obst_metrics_history = {"x": [], **{k: [] for k in tracked_metric_keys}}
    
    # 5. 开始训练
    logger.info("===== 开始训练 =====")
    val_has_obstacle = bool(getattr(val_loader.dataset, "split_has_obstacle_samples", False))
    if not val_has_obstacle:
        logger.info("===== validation split has no obstacle samples; val_obst will be skipped =====")
    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        is_best = False
        val_metrics = None
        val_obst_metrics = None

        # 训练
        train_metrics, train_curve_points = train_one_epoch(model, train_loader, optimizer, epoch, cfg, logger)

        # 记录训练曲线采样点（按 LOG_PRINT_FREQ）
        for point in train_curve_points:
            train_metrics_history["x"].append(point["x"])
            for k in tracked_metric_keys:
                train_metrics_history[k].append(point.get(k, float("nan")))
        
        # 验证
        if epoch % cfg.VAL_FREQ == 0 or epoch == cfg.NUM_EPOCHS:
            # 普通轨迹验证
            val_metrics = validate(model, val_loader, cfg, logger, epoch, is_obst=False)
            # 障碍轨迹验证
            if val_has_obstacle:
                val_obst_metrics = validate(model, val_loader, cfg, logger, epoch, is_obst=True)
            else:
                val_obst_metrics = nan_metrics_like(val_metrics)
            
            # 记录验证指标
            val_metrics_history["x"].append(float(epoch))
            val_obst_metrics_history["x"].append(float(epoch))
            for k in tracked_metric_keys:
                val_metrics_history[k].append(val_metrics.get(k, float("nan")))
                val_obst_metrics_history[k].append(val_obst_metrics.get(k, float("nan")))
            
            # 保存可视化
            if cfg.SAVE_PLOT:
                plot_training_curves(
                    train_metrics_history,
                    val_metrics_history,
                    val_obst_metrics_history,
                    cfg.EXP_LOG_DIR
                )
            
            # 检查是否为最优模型
            current_val_loss = val_metrics["loss"]
            is_best = current_val_loss < best_val_loss
            if is_best:
                best_val_loss = current_val_loss
                logger.info(f"===== 最优模型更新 (Epoch {epoch}) | Best Loss: {best_val_loss:.4f} =====")
        
        # 保存检查点
        if epoch % cfg.SAVE_FREQ == 0 or epoch == cfg.NUM_EPOCHS or is_best:
            metrics = {
                "train": train_metrics,
                "val": val_metrics,
                "val_obst": val_obst_metrics,
                "best_loss": best_val_loss
            }
            ckpt_path = save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=cfg,
                metrics=metrics,
                is_best=is_best,
                save_dir=cfg.EXP_LOG_DIR
            )
            logger.info(f"===== 保存检查点: {ckpt_path} =====")
    
    logger.info("===== 训练完成 =====")

if __name__ == "__main__":
    main()
