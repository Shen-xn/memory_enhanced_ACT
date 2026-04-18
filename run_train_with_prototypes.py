"""Convenience entry point for ACT training with prototype supervision."""

from __future__ import annotations

import argparse
import os

from config import cfg
import training


def build_argparser():
    parser = argparse.ArgumentParser(description="Train ACT with prototype loss.")
    parser.add_argument("--prototype-file", required=True, help="Path to the .npz prototype bank.")
    parser.add_argument("--data-root", default=None, help="Dataset root. Defaults to config.DATA_ROOT.")
    parser.add_argument("--prototype-loss-weight", type=float, default=0.1)
    parser.add_argument("--prototype-temperature", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--action-l1-weight", type=float, default=None)
    parser.add_argument("--exp-name", default=None, help="Optional explicit experiment name.")
    parser.add_argument("--resume-ckpt", default=None, help="Resume from a checkpoint path.")
    return parser


def main():
    args = build_argparser().parse_args()

    cfg.ENABLE_PROTOTYPE_LOSS = True
    cfg.PROTOTYPE_FILE = os.path.abspath(args.prototype_file)
    cfg.PROTOTYPE_LOSS_WEIGHT = float(args.prototype_loss_weight)
    cfg.PROTOTYPE_TEMPERATURE = float(args.prototype_temperature)

    if args.data_root:
        cfg.DATA_ROOT = os.path.abspath(args.data_root)
    if args.num_epochs is not None:
        cfg.NUM_EPOCHS = int(args.num_epochs)
    if args.batch_size is not None:
        cfg.BATCH_SIZE = int(args.batch_size)
    if args.lr is not None:
        cfg.LR = float(args.lr)
    if args.lr_backbone is not None:
        cfg.LR_BACKBONE = float(args.lr_backbone)
    if args.action_l1_weight is not None:
        cfg.ACTION_L1_WEIGHT = float(args.action_l1_weight)

    if args.resume_ckpt:
        cfg.TRAIN_MODE = "resume"
        cfg.RESUME_CKPT_PATH = os.path.abspath(args.resume_ckpt)
    else:
        cfg.TRAIN_MODE = ""
        cfg.RESUME_CKPT_PATH = ""

    if args.exp_name:
        cfg.EXP_NAME = args.exp_name

    cfg.refresh_model_params()
    training.main()


if __name__ == "__main__":
    main()
