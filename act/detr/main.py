# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse

import torch

from .models import build_ACT_model, build_CNNMLP_model


class ArgsObject:
    def __init__(self, data_dict):
        self.__dict__ = data_dict


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument("--clip_max_norm", default=0.1, type=float)

    parser.add_argument("--depth_channel", default=False, type=bool)
    parser.add_argument("--image_channels", default=None, type=int)
    parser.add_argument("--backbone", default="resnet18", type=str)
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned"))
    parser.add_argument("--camera_names", default=[], type=list)

    parser.add_argument("--enc_layers", default=4, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--num_queries", default=400, type=int)
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--state_dim", default=6, type=int)

    parser.add_argument("--use_phase_token", default=True, type=bool)
    parser.add_argument("--phase_bank_path", default="", type=str)
    parser.add_argument("--phase_pca_dim", default=0, type=int)
    parser.add_argument("--pca_head_hidden_dim", default=1024, type=int)
    parser.add_argument("--pca_head_depth", default=3, type=int)

    parser.add_argument("--masks", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, required=True)
    parser.add_argument("--policy_class", action="store", type=str, required=True)
    parser.add_argument("--task_name", action="store", type=str, required=True)
    parser.add_argument("--seed", action="store", type=int, required=True)
    parser.add_argument("--num_epochs", action="store", type=int, required=True)
    parser.add_argument("--kl_weight", action="store", type=float, required=False)
    parser.add_argument("--chunk_size", action="store", type=int, required=False)
    parser.add_argument("--temporal_agg", action="store_true")
    return parser


def _build_args_object(args_override):
    parser = get_args_parser()
    defaults = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        defaults[action.dest] = action.default
    defaults.update(args_override)
    if defaults.get("image_channels") is not None:
        defaults["depth_channel"] = int(defaults["image_channels"]) == 4
    return ArgsObject(defaults)


def _move_model_to_device(model, device=None):
    return model.to(device) if device is not None else model


def _freeze_module_if_present(model, module_name):
    module = getattr(model, module_name, None)
    if module is not None:
        module.requires_grad_(False)
        return True
    return False


def build_ACT_model_and_optimizer(args_override, device=None):
    args = _build_args_object(args_override)
    model = build_ACT_model(args)
    model = _move_model_to_device(model, device)

    if args.lr_backbone <= 0:
        if not _freeze_module_if_present(model, "backbone"):
            _freeze_module_if_present(model, "backbones")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override, device=None):
    args = _build_args_object(args_override)
    model = build_CNNMLP_model(args)
    model = _move_model_to_device(model, device)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer
