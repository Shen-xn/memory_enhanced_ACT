"""Export trained ACT/me_block checkpoints to TorchScript deployment artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from act.policy import ACTPolicy
from act.detr.models.me_block.generate_memory_images import load_model_from_checkpoint as load_me_block_model
from data_process.data_loader import get_fixed_joint_stats
from deploy.deploy_wrappers import (
    ACTDualImageInferenceWrapper,
    ACTSingleImageInferenceWrapper,
    MEBlockInferenceWrapper,
)


DEPTH_CLIP_MIN = 0.0
DEPTH_CLIP_MAX = 800.0
PAD_LEFT = 0
PAD_TOP = 40
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ACT and optional me_block to TorchScript for C++ deployment.")
    parser.add_argument("--act-checkpoint", type=str, required=True, help="Path to ACT checkpoint (.pth).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for exported deployment artifacts.")
    parser.add_argument("--me-block-checkpoint", type=str, default="", help="Optional me_block checkpoint (.pth).")
    parser.add_argument(
        "--data-root",
        type=str,
        default="",
        help="Compatibility argument. Export now uses fixed physical joint limits instead of dataset statistics.",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device used during export.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one forward pass through exported TorchScript files.")
    return parser.parse_args()


def _pick(payload: Dict, key: str, default=None):
    """Read config values from either old top-level keys or MODEL_PARAMS."""
    model_params = payload.get("MODEL_PARAMS") or {}
    if key in payload:
        return payload[key]
    return model_params.get(key, default)


def normalize_act_config(payload: Dict) -> Dict:
    """Convert historical checkpoint config shapes into current ACT args."""
    use_memory = payload.get("USE_MEMORY_IMAGE_INPUT")
    if use_memory is None:
        use_memory = _pick(payload, "use_memory_image_input")
    if use_memory is None:
        use_memory = bool(payload.get("ME_BLOCK") or _pick(payload, "me_block", False))

    return {
        "lr": payload.get("LR", 1e-4),
        "lr_backbone": payload.get("LR_BACKBONE", 1e-5),
        "weight_decay": payload.get("WEIGHT_DECAY", 1e-4),
        "kl_weight": payload.get("KL_WEIGHT", _pick(payload, "kl_weight", 1.0)),
        "camera_names": payload.get("CAMERA_NAMES", _pick(payload, "camera_names", ["gemini"])),
        "use_memory_image_input": bool(use_memory),
        "depth_channel": payload.get("DEPTH_CHANNEL", _pick(payload, "depth_channel", True)),
        "backbone": payload.get("BACKBONE", _pick(payload, "backbone", "resnet18")),
        "position_embedding": payload.get("POSITION_EMBEDDING", _pick(payload, "position_embedding", "sine")),
        "dilation": payload.get("DILATION", _pick(payload, "dilation", False)),
        "pre_norm": payload.get("PRE_NORM", _pick(payload, "pre_norm", True)),
        "enc_layers_enc": payload.get("ENC_LAYERS_ENC", _pick(payload, "enc_layers_enc", 4)),
        "enc_layers": payload.get("ENC_LAYERS", _pick(payload, "enc_layers", 4)),
        "dec_layers": payload.get("DEC_LAYERS", _pick(payload, "dec_layers", 6)),
        "dropout": payload.get("DROPOUT", _pick(payload, "dropout", 0.1)),
        "dim_feedforward": payload.get("DIM_FEEDFORWARD", _pick(payload, "dim_feedforward", 2048)),
        "hidden_dim": payload.get("HIDDEN_DIM", _pick(payload, "hidden_dim", 512)),
        "nheads": payload.get("NHEADS", _pick(payload, "nheads", 8)),
        "num_queries": payload.get("NUM_QUERIES", _pick(payload, "num_queries", payload.get("FUTURE_STEPS", 10))),
        "state_dim": payload.get("STATE_DIM", _pick(payload, "state_dim", 6)),
        "masks": payload.get("MASKS", _pick(payload, "masks", False)),
        "lr_drop": payload.get("LR_DROP", _pick(payload, "lr_drop", 200)),
        "clip_max_norm": payload.get("CLIP_MAX_NORM", _pick(payload, "clip_max_norm", 0.1)),
        "chunk_size": payload.get("CHUNK_SIZE", _pick(payload, "chunk_size", payload.get("FUTURE_STEPS", 10))),
        "temporal_agg": payload.get("TEMPORAL_AGG", _pick(payload, "temporal_agg", False)),
        "eval": payload.get("EVAL", _pick(payload, "eval", False)),
        "onscreen_render": payload.get("ONSCREEN_RENDER", _pick(payload, "onscreen_render", False)),
        "ckpt_dir": payload.get("CKPT_DIR", _pick(payload, "ckpt_dir", "")),
        "policy_class": payload.get("POLICY_CLASS", _pick(payload, "policy_class", "ACTPolicy")),
        "task_name": payload.get("TASK_NAME", _pick(payload, "task_name", "custom_dataset")),
        "seed": payload.get("SEED", _pick(payload, "seed", 42)),
        "num_epochs": payload.get("NUM_EPOCHS", _pick(payload, "num_epochs", 1)),
        "batch_size": payload.get("BATCH_SIZE", _pick(payload, "batch_size", 1)),
        "epochs": payload.get("NUM_EPOCHS", _pick(payload, "epochs", 1)),
    }


def load_act_policy(checkpoint_path: str, device: torch.device) -> Tuple[ACTPolicy, Dict]:
    """Rebuild ACTPolicy from a checkpoint and load its weights strictly."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_override = normalize_act_config(checkpoint["config"])
    policy = ACTPolicy(args_override, device=device).eval()
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return policy, checkpoint["config"]


def _trace_and_save(module: torch.nn.Module, example_inputs: Tuple[torch.Tensor, ...], output_path: str) -> None:
    """Trace, freeze, and save one TorchScript module."""
    traced = torch.jit.trace(module.eval(), example_inputs, strict=False)
    traced = torch.jit.freeze(traced)
    traced.save(output_path)


def write_deploy_config(path: str, payload: Dict) -> None:
    """Write C++-readable deployment metadata using OpenCV FileStorage."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open deploy config for writing: {path}")
    for key, value in payload.items():
        if isinstance(value, bool):
            fs.write(key, int(value))
        else:
            fs.write(key, value)
    fs.release()


def export_artifacts(args: argparse.Namespace) -> Dict:
    """Export ACT and optional me_block modules plus deploy_config.yml."""
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    act_policy, act_config = load_act_policy(args.act_checkpoint, device=device)
    use_memory_image_input = bool(normalize_act_config(act_config)["use_memory_image_input"])
    joint_stats = get_fixed_joint_stats()

    if args.me_block_checkpoint and not use_memory_image_input:
        raise ValueError("ACT checkpoint is a single-image model. Do not export me_block for a baseline ACT deployment.")
    if use_memory_image_input and not args.me_block_checkpoint:
        raise ValueError(
            "ACT checkpoint expects memory_image input. Current deploy pipeline requires --me-block-checkpoint "
            "to produce memory_image online."
        )

    if use_memory_image_input:
        act_wrapper = ACTDualImageInferenceWrapper(
            act_policy.model,
            joint_min=joint_stats["min"],
            joint_rng=joint_stats["rng"],
        ).to(device)
        act_examples = (
            torch.zeros(1, act_policy.model.action_head.out_features, device=device),
            torch.zeros(1, 4, TARGET_HEIGHT, TARGET_WIDTH, device=device),
            torch.zeros(1, 4, TARGET_HEIGHT, TARGET_WIDTH, device=device),
        )
    else:
        act_wrapper = ACTSingleImageInferenceWrapper(
            act_policy.model,
            joint_min=joint_stats["min"],
            joint_rng=joint_stats["rng"],
        ).to(device)
        act_examples = (
            torch.zeros(1, act_policy.model.action_head.out_features, device=device),
            torch.zeros(1, 4, TARGET_HEIGHT, TARGET_WIDTH, device=device),
        )

    act_output_path = str(output_dir / "act_inference.pt")
    _trace_and_save(act_wrapper, act_examples, act_output_path)

    has_me_block = False
    if args.me_block_checkpoint:
        me_block_model = load_me_block_model(args.me_block_checkpoint, device=device).eval()
        me_block_wrapper = MEBlockInferenceWrapper(me_block_model).to(device)
        num_me_classes = len(me_block_model.config.importance.class_names)
        me_examples = (
            torch.zeros(1, 4, TARGET_HEIGHT, TARGET_WIDTH, device=device),
            torch.zeros(1, num_me_classes, 4, TARGET_HEIGHT, TARGET_WIDTH, device=device),
            torch.zeros(1, num_me_classes, TARGET_HEIGHT, TARGET_WIDTH, device=device),
        )
        _trace_and_save(me_block_wrapper, me_examples, str(output_dir / "me_block_inference.pt"))
        has_me_block = True
    else:
        num_me_classes = 0

    deploy_payload = {
        "target_width": int(TARGET_WIDTH),
        "target_height": int(TARGET_HEIGHT),
        "pad_left": int(PAD_LEFT),
        "pad_top": int(PAD_TOP),
        "depth_clip_min": float(DEPTH_CLIP_MIN),
        "depth_clip_max": float(DEPTH_CLIP_MAX),
        "state_dim": int(act_policy.model.action_head.out_features),
        "num_queries": int(act_policy.model.num_queries),
        "use_memory_image_input": use_memory_image_input,
        "has_me_block": has_me_block,
        "me_block_num_classes": int(num_me_classes),
        "preprocessed_channel_order": "BGRA",
        "act_wrapper_expects": "BGRA_float_0_1",
        "me_block_wrapper_expects": "BGRA_float_0_1",
    }
    write_deploy_config(str(output_dir / "deploy_config.yml"), deploy_payload)
    return {
        "device": str(device),
        "output_dir": str(output_dir),
        "act_output_path": act_output_path,
        "use_memory_image_input": use_memory_image_input,
        "has_me_block": has_me_block,
        "me_block_num_classes": int(num_me_classes),
    }


def smoke_test(output_dir: str, use_memory_image_input: bool, has_me_block: bool, me_block_num_classes: int) -> None:
    """Run CPU forward passes through exported modules to catch shape errors."""
    act_module = torch.jit.load(str(Path(output_dir) / "act_inference.pt"), map_location="cpu").eval()
    qpos = torch.zeros(1, 6)
    image = torch.zeros(1, 4, TARGET_HEIGHT, TARGET_WIDTH)
    if use_memory_image_input:
        memory = torch.zeros_like(image)
        actions = act_module(qpos, image, memory)
    else:
        actions = act_module(qpos, image)
    print(f"[smoke] ACT output shape: {tuple(actions.shape)}")

    if has_me_block:
        me_block = torch.jit.load(str(Path(output_dir) / "me_block_inference.pt"), map_location="cpu").eval()
        memory_image, memory_state, score_state = me_block(
            image,
            torch.zeros(1, me_block_num_classes, 4, TARGET_HEIGHT, TARGET_WIDTH),
            torch.zeros(1, me_block_num_classes, TARGET_HEIGHT, TARGET_WIDTH),
        )
        print(f"[smoke] me_block outputs: memory_image={tuple(memory_image.shape)}, memory_state={tuple(memory_state.shape)}, score_state={tuple(score_state.shape)}")


def main() -> None:
    args = parse_args()
    result = export_artifacts(args)
    print(f"Exported deployment artifacts to: {result['output_dir']}")
    print(f"ACT wrapper: {result['act_output_path']}")
    if result["has_me_block"]:
        print(f"me_block wrapper: {Path(result['output_dir']) / 'me_block_inference.pt'}")
    if args.smoke_test:
        smoke_test(
            output_dir=result["output_dir"],
            use_memory_image_input=result["use_memory_image_input"],
            has_me_block=result["has_me_block"],
            me_block_num_classes=result["me_block_num_classes"],
        )


if __name__ == "__main__":
    main()
