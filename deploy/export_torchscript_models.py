"""Export trained ACT checkpoints to TorchScript deployment artifacts."""

from __future__ import annotations

import argparse
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
from data_process.data_loader import get_fixed_joint_stats
from deploy.deploy_wrappers import ACTSingleImageInferenceWrapper


DEPTH_CLIP_MIN = 0.0
DEPTH_CLIP_MAX = 800.0
PAD_LEFT = 0
PAD_TOP = 40
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ACT checkpoint to TorchScript for deployment.")
    parser.add_argument("--act-checkpoint", type=str, required=True, help="Path to ACT checkpoint (.pth).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for exported deployment artifacts.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _pick(payload: Dict, key: str, default=None):
    model_params = payload.get("MODEL_PARAMS") or {}
    if key in payload:
        return payload[key]
    return model_params.get(key, default)


def normalize_act_config(payload: Dict) -> Dict:
    image_channels = payload.get("IMAGE_CHANNELS", _pick(payload, "image_channels"))
    if image_channels is None:
        image_channels = 4 if bool(payload.get("DEPTH_CHANNEL", _pick(payload, "depth_channel", True))) else 3
    image_channels = int(image_channels)
    if image_channels not in (3, 4):
        raise ValueError(f"Unsupported ACT image_channels={image_channels}; expected 3 or 4.")

    return {
        "lr": payload.get("LR", 1e-4),
        "lr_backbone": payload.get("LR_BACKBONE", 1e-5),
        "weight_decay": payload.get("WEIGHT_DECAY", 1e-4),
        "kl_weight": payload.get("KL_WEIGHT", _pick(payload, "kl_weight", 1.0)),
        "pca_coord_loss_weight": payload.get("PCA_COORD_LOSS_WEIGHT", _pick(payload, "pca_coord_loss_weight", _pick(payload, "prototype_loss_weight", 0.1))),
        "residual_loss_weight": payload.get("RESIDUAL_LOSS_WEIGHT", _pick(payload, "residual_loss_weight", 1.0)),
        "recon_loss_weight": payload.get("RECON_LOSS_WEIGHT", _pick(payload, "recon_loss_weight", 1.0)),
        "predict_delta_qpos": bool(payload.get("PREDICT_DELTA_QPOS", _pick(payload, "predict_delta_qpos", False))),
        "delta_qpos_scale": float(payload.get("DELTA_QPOS_SCALE", _pick(payload, "delta_qpos_scale", 10.0))),
        "camera_names": payload.get("CAMERA_NAMES", _pick(payload, "camera_names", ["gemini"])),
        "image_channels": image_channels,
        "depth_channel": image_channels == 4,
        "backbone": payload.get("BACKBONE", _pick(payload, "backbone", "resnet18")),
        "position_embedding": payload.get("POSITION_EMBEDDING", _pick(payload, "position_embedding", "sine")),
        "dilation": payload.get("DILATION", _pick(payload, "dilation", False)),
        "pre_norm": payload.get("PRE_NORM", _pick(payload, "pre_norm", True)),
        "enc_layers_enc": payload.get("ENC_LAYERS_ENC", _pick(payload, "enc_layers_enc", 3)),
        "enc_layers": payload.get("ENC_LAYERS", _pick(payload, "enc_layers", 5)),
        "dec_layers": payload.get("DEC_LAYERS", _pick(payload, "dec_layers", 5)),
        "dropout": payload.get("DROPOUT", _pick(payload, "dropout", 0.1)),
        "dim_feedforward": payload.get("DIM_FEEDFORWARD", _pick(payload, "dim_feedforward", 2048)),
        "hidden_dim": payload.get("HIDDEN_DIM", _pick(payload, "hidden_dim", 512)),
        "nheads": payload.get("NHEADS", _pick(payload, "nheads", 8)),
        "num_queries": payload.get("NUM_QUERIES", _pick(payload, "num_queries", payload.get("FUTURE_STEPS", 10))),
        "state_dim": payload.get("STATE_DIM", _pick(payload, "state_dim", 6)),
        "use_phase_pca_supervision": payload.get("USE_PHASE_PCA_SUPERVISION", _pick(payload, "use_phase_pca_supervision", True)),
        "use_phase_token": payload.get("USE_PHASE_TOKEN", _pick(payload, "use_phase_token", True)),
        "phase_bank_path": payload.get("PHASE_BANK_PATH", _pick(payload, "phase_bank_path", "")),
        "phase_pca_dim": payload.get("PHASE_PCA_DIM", _pick(payload, "phase_pca_dim", 16)),
        "pca_head_hidden_dim": payload.get("PCA_HEAD_HIDDEN_DIM", _pick(payload, "pca_head_hidden_dim", 1024)),
        "pca_head_depth": payload.get("PCA_HEAD_DEPTH", _pick(payload, "pca_head_depth", 3)),
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_override = normalize_act_config(checkpoint["config"])
    policy = ACTPolicy(args_override, device=device).eval()
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return policy, checkpoint["config"]


def _trace_and_save(module: torch.nn.Module, example_inputs: Tuple[torch.Tensor, ...], output_path: str) -> None:
    traced = torch.jit.trace(module.eval(), example_inputs, strict=False)
    traced = torch.jit.freeze(traced)
    traced.save(output_path)


def write_deploy_config(path: str, payload: Dict) -> None:
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
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    act_policy, act_config = load_act_policy(args.act_checkpoint, device=device)
    normalized_act_config = normalize_act_config(act_config)
    image_channels = int(normalized_act_config["image_channels"])
    joint_stats = get_fixed_joint_stats()

    act_wrapper = ACTSingleImageInferenceWrapper(
        act_policy.model,
        joint_min=joint_stats["min"],
        joint_rng=joint_stats["rng"],
        image_channels=image_channels,
        predict_delta_qpos=bool(normalized_act_config["predict_delta_qpos"]),
        delta_qpos_scale=float(normalized_act_config["delta_qpos_scale"]),
    ).to(device)
    act_examples = (
        torch.zeros(1, act_policy.model.residual_head.out_features, device=device),
        torch.zeros(1, 4, TARGET_HEIGHT, TARGET_WIDTH, device=device),
    )
    act_output_path = str(output_dir / "act_inference.pt")
    _trace_and_save(act_wrapper, act_examples, act_output_path)

    deploy_payload = {
        "target_width": int(TARGET_WIDTH),
        "target_height": int(TARGET_HEIGHT),
        "pad_left": int(PAD_LEFT),
        "pad_top": int(PAD_TOP),
        "depth_clip_min": float(DEPTH_CLIP_MIN),
        "depth_clip_max": float(DEPTH_CLIP_MAX),
        "state_dim": int(act_policy.model.residual_head.out_features),
        "num_queries": int(act_policy.model.num_queries),
        "image_channels": int(image_channels),
        "use_phase_pca_supervision": bool(normalized_act_config["use_phase_pca_supervision"]),
        "use_phase_token": bool(normalized_act_config["use_phase_token"]),
        "phase_pca_dim": int(normalized_act_config["phase_pca_dim"]),
        "predict_delta_qpos": bool(normalized_act_config["predict_delta_qpos"]),
        "delta_qpos_scale": float(normalized_act_config["delta_qpos_scale"]),
        "preprocessed_channel_order": "BGRA",
        "act_wrapper_expects": "BGRA_float_0_1",
    }
    write_deploy_config(str(output_dir / "deploy_config.yml"), deploy_payload)
    return {"device": str(device), "output_dir": str(output_dir), "act_output_path": act_output_path}


def smoke_test(output_dir: str) -> None:
    act_module = torch.jit.load(str(Path(output_dir) / "act_inference.pt"), map_location="cpu").eval()
    qpos = torch.zeros(1, 6)
    image = torch.zeros(1, 4, TARGET_HEIGHT, TARGET_WIDTH)
    actions = act_module(qpos, image)
    print(f"[smoke] ACT output shape: {tuple(actions.shape)}")


def main() -> None:
    args = parse_args()
    result = export_artifacts(args)
    print(f"Exported deployment artifacts to: {result['output_dir']}")
    print(f"ACT wrapper: {result['act_output_path']}")
    if args.smoke_test:
        smoke_test(output_dir=result["output_dir"])


if __name__ == "__main__":
    main()
