# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse

import torch

from .models import build_CNNMLP_model, build_me_ACT_model


class ArgsObject:
    """Lightweight namespace used by the model builders."""

    def __init__(self, data_dict):
        self.__dict__ = data_dict


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--lr_me", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")

    # Model parameters
    parser.add_argument("--me_block", default=False, type=bool, help="If true, model will include me_block")
    parser.add_argument(
        "--depth_channel",
        default=False,
        type=bool,
        help="If true, backbone is adapted to 4-channel input",
    )
    parser.add_argument("--backbone", default="resnet18", type=str, help="Name of the convolutional backbone to use")
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument("--camera_names", default=[], type=list, help="A list of camera names")

    # Transformer
    parser.add_argument("--enc_layers", default=4, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument("--hidden_dim", default=256, type=int, help="Size of the embeddings")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument("--nheads", default=8, type=int, help="Number of attention heads")
    parser.add_argument("--num_queries", default=400, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--state_dim", default=6, type=int, help="Dimension of the output state")

    # Segmentation
    parser.add_argument("--masks", action="store_true", help="Train segmentation head if the flag is provided")

    # Compatibility args kept for the original ACT script path.
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument("--policy_class", action="store", type=str, help="policy_class, capitalize", required=True)
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--temporal_agg", action="store_true")

    return parser


def _build_args_object(args_override):
    """
    Build an args-like object from parser defaults plus explicit overrides,
    without parsing CLI flags from the current process.
    """
    parser = get_args_parser()
    defaults = {}

    for action in parser._actions:
        if action.dest == "help":
            continue
        defaults[action.dest] = action.default

    defaults.update(args_override)
    return ArgsObject(defaults)


def _move_model_to_device(model, device=None):
    """Move a model to the requested device when one is provided."""
    if device is not None:
        return model.to(device)
    return model


def _freeze_module_if_present(model, module_name):
    """
    Freeze a submodule only when it exists and is not None.
    This avoids crashes for optional modules like me_block.
    """
    module = getattr(model, module_name, None)
    if module is not None:
        module.requires_grad_(False)
        return True
    return False


def build_me_ACT_model_and_optimizer(args_override, device=None):
    args = _build_args_object(args_override)

    model = build_me_ACT_model(args)
    model = _move_model_to_device(model, device)

    for param in model.parameters():
        param.requires_grad = True

    if args.lr_backbone <= 0:
        if not _freeze_module_if_present(model, "backbone"):
            _freeze_module_if_present(model, "backbones")

    if args.lr_me <= 0:
        _freeze_module_if_present(model, "me_block")

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and "me_block" not in n and p.requires_grad
            ]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if "me_block" in n and p.requires_grad],
            "lr": args.lr_me,
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
