"""Generate offline memory images for ACT training.

The generator replays each task in frame order. It carries `prev_memory` and
`prev_scores` across frames inside a task, then resets them at the next task.
The saved `memory_image_four_channel` files keep the same filenames as source
`four_channel` frames so ACT can align them by basename.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .me_block_config import (
    MemoryGenerationConfig,
    me_block_config_from_dict,
)
from .importance_dataset import list_task_dirs, read_four_channel
from .memory_gate_model import ImportanceMemoryModel


def parse_args(default_data_root: str = "") -> argparse.Namespace:
    """Parse CLI options for offline memory-image generation."""
    parser = argparse.ArgumentParser(description="Generate memory_image_four_channel from a trained importance model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth or latest_model.pth.")
    parser.add_argument("--data-root", type=str, default=default_data_root, help="Root containing task* folders. Overrides checkpoint config.")
    parser.add_argument("--task-filter", type=str, default="", help="Only process tasks whose name contains this text.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing memory_image_four_channel directories.")
    parser.add_argument("--debug", action="store_true", help="Also save score/mask/intermediate debug outputs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--keep-top-ratio-target", type=float, default=None, help="Override checkpoint target memory keep ratio.")
    parser.add_argument("--keep-top-ratio-goal", type=float, default=None, help="Override checkpoint goal memory keep ratio.")
    parser.add_argument("--keep-top-ratio-arm", type=float, default=None, help="Override checkpoint arm memory keep ratio.")
    return parser.parse_args()


def apply_memory_overrides(model: ImportanceMemoryModel, args: argparse.Namespace) -> None:
    """Apply generation-only memory overrides without modifying checkpoint files."""
    overrides = {
        "keep_top_ratio_target": args.keep_top_ratio_target,
        "keep_top_ratio_goal": args.keep_top_ratio_goal,
        "keep_top_ratio_arm": args.keep_top_ratio_arm,
    }
    for name, value in overrides.items():
        if value is None:
            continue
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
        setattr(model.config.memory, name, float(value))


def load_model_from_checkpoint(path: str, device: torch.device) -> ImportanceMemoryModel:
    """Restore model weights and config from a me_block checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = me_block_config_from_dict(checkpoint["config"])

    model = ImportanceMemoryModel(config=config).to(device)
    model_state = checkpoint["model_state_dict"]
    allowed_keys = model.state_dict()
    filtered_state = {key: value for key, value in model_state.items() if key in allowed_keys}
    model.load_state_dict(filtered_state, strict=True)
    model.eval()
    return model


def task_output_complete(task_dir: str, config: MemoryGenerationConfig, debug: bool) -> bool:
    """Check whether a task already has all requested output files."""
    image_dir = os.path.join(task_dir, config.image_dirname)
    output_dir = os.path.join(task_dir, config.output_dirname)
    src_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    out_paths = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    if not src_paths or len(src_paths) != len(out_paths):
        return False
    if not debug:
        return True

    meta_path = os.path.join(task_dir, "memory_image_meta.json")
    score_dir = os.path.join(task_dir, config.save_score_dirname)
    mask_dir = os.path.join(task_dir, config.save_mask_dirname)
    importance_dir = os.path.join(task_dir, "importance_scores")
    write_mask_dir = os.path.join(task_dir, "write_masks")
    extra_counts = [
        len(glob.glob(os.path.join(score_dir, "*.png"))),
        len(glob.glob(os.path.join(mask_dir, "*.png"))),
        len(glob.glob(os.path.join(importance_dir, "*.png"))),
        len(glob.glob(os.path.join(write_mask_dir, "*.png"))),
    ]
    return os.path.exists(meta_path) and all(count == len(src_paths) for count in extra_counts)


def save_png_uint8(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def collapse_score_for_debug(score_state: torch.Tensor) -> torch.Tensor:
    """Collapse per-class score state into one debug image."""
    if score_state.size(1) == 1:
        return score_state
    return torch.max(score_state, dim=1, keepdim=True).values


def clear_png_dir(path: str) -> None:
    if not os.path.isdir(path):
        return
    for png_path in glob.glob(os.path.join(path, "*.png")):
        os.remove(png_path)


def clear_debug_outputs(task_dir: str, config: MemoryGenerationConfig) -> None:
    """Remove debug-only artifacts when generating a clean memory-image set."""
    for dirname in (config.save_score_dirname, config.save_mask_dirname, "importance_scores", "write_masks"):
        clear_png_dir(os.path.join(task_dir, dirname))
    meta_path = os.path.join(task_dir, "memory_image_meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)


@torch.no_grad()
def generate_for_task(
    model: ImportanceMemoryModel,
    task_dir: str,
    device: torch.device,
    checkpoint_path: str,
    force: bool,
    debug: bool,
) -> None:
    """Generate recurrent memory images for one task directory."""
    config = model.config.generation
    image_dir = os.path.join(task_dir, config.image_dirname)
    output_dir = os.path.join(task_dir, config.output_dirname)
    score_dir = os.path.join(task_dir, config.save_score_dirname)
    mask_dir = os.path.join(task_dir, config.save_mask_dirname)
    importance_dir = os.path.join(task_dir, "importance_scores")
    write_mask_dir = os.path.join(task_dir, "write_masks")

    if task_output_complete(task_dir, config, debug=debug) and not force:
        print(f"Skip completed task: {os.path.basename(task_dir)}")
        return

    os.makedirs(output_dir, exist_ok=True)
    if debug:
        os.makedirs(score_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(importance_dir, exist_ok=True)
        os.makedirs(write_mask_dir, exist_ok=True)
    else:
        clear_debug_outputs(task_dir, config)

    clear_png_dir(output_dir)
    if debug:
        clear_png_dir(score_dir)
        clear_png_dir(mask_dir)
        clear_png_dir(importance_dir)
        clear_png_dir(write_mask_dir)

    prev_memory = None
    prev_scores = None
    frame_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    for frame_path in tqdm(frame_paths, desc=os.path.basename(task_dir), leave=False):
        frame_name = os.path.basename(frame_path)
        frame = read_four_channel(frame_path).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(np.transpose(frame, (2, 0, 1))).unsqueeze(0).to(device)

        step = model(image_tensor, prev_memory=prev_memory, prev_scores=prev_scores)
        prev_memory = step.memory_state
        prev_scores = step.score_state

        memory_uint8 = (step.memory_image.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        save_png_uint8(os.path.join(output_dir, frame_name), memory_uint8)
        if debug:
            debug_score = collapse_score_for_debug(step.score_state)
            score_uint8 = (debug_score.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            mask_uint8 = (step.output_mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255)
            importance_uint8 = (step.importance_score.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            write_mask_uint8 = (step.write_mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255)
            save_png_uint8(os.path.join(score_dir, frame_name), score_uint8)
            save_png_uint8(os.path.join(mask_dir, frame_name), mask_uint8)
            save_png_uint8(os.path.join(importance_dir, frame_name), importance_uint8)
            save_png_uint8(os.path.join(write_mask_dir, frame_name), write_mask_uint8)

    if debug:
        meta = {
            "task": os.path.basename(task_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": os.path.abspath(checkpoint_path),
            "output_dirname": config.output_dirname,
            "score_dirname": config.save_score_dirname,
            "mask_dirname": config.save_mask_dirname,
            "importance_dirname": "importance_scores",
            "write_mask_dirname": "write_masks",
            "class_names": model.config.importance.class_names,
            "memory": {
                "score_decay": model.config.memory.score_decay,
                "tau_up": model.config.memory.tau_up,
                "keep_top_ratio_target": model.config.memory.keep_top_ratio_target,
                "keep_top_ratio_goal": model.config.memory.keep_top_ratio_goal,
                "keep_top_ratio_arm": model.config.memory.keep_top_ratio_arm,
            },
        }
        with open(os.path.join(task_dir, "memory_image_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def main(default_data_root: str = "") -> None:
    """Entry point used by both module execution and root-level run script."""
    args = parse_args(default_data_root=default_data_root)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_model_from_checkpoint(args.checkpoint, device)
    apply_memory_overrides(model, args)

    if args.data_root:
        model.config.generation.data_root = args.data_root

    print(
        "Memory ratios: "
        f"target={model.config.memory.keep_top_ratio_target} "
        f"goal={model.config.memory.keep_top_ratio_goal} "
        f"arm={model.config.memory.keep_top_ratio_arm}"
    )

    task_dirs = list_task_dirs(
        model.config.generation.data_root,
        image_dirname=model.config.generation.image_dirname,
    )
    if args.task_filter:
        task_dirs = [task_dir for task_dir in task_dirs if args.task_filter in os.path.basename(task_dir)]

    if not task_dirs:
        raise FileNotFoundError(f"No task directories found under {model.config.generation.data_root}")

    for task_dir in task_dirs:
        generate_for_task(
            model,
            task_dir,
            device,
            checkpoint_path=args.checkpoint,
            force=args.force,
            debug=args.debug,
        )

    print("Memory image generation finished.")


if __name__ == "__main__":
    main()
