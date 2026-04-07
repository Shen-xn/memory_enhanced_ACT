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
    ImportanceModelConfig,
    MEBlockConfig,
    ImportanceTrainingConfig,
    MemoryGenerationConfig,
    memory_update_config_from_dict,
)
from .importance_dataset import list_task_dirs, read_four_channel
from .memory_gate_model import ImportanceMemoryModel


def parse_args(default_data_root: str = "") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate memory_image_four_channel from a trained importance model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth or latest_model.pth.")
    parser.add_argument("--data-root", type=str, default=default_data_root, help="Root containing task* folders. Overrides checkpoint config.")
    parser.add_argument("--task-filter", type=str, default="", help="Only process tasks whose name contains this text.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing memory_image_four_channel directories.")
    parser.add_argument("--debug", action="store_true", help="Also save score/mask/intermediate debug outputs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser.parse_args()


def load_model_from_checkpoint(path: str, device: torch.device) -> ImportanceMemoryModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config_payload = checkpoint["config"]
    config = MEBlockConfig(
        importance=ImportanceModelConfig(**config_payload["importance"]),
        memory=memory_update_config_from_dict(config_payload["memory"]),
        training=ImportanceTrainingConfig(**config_payload.get("training", {})),
        generation=MemoryGenerationConfig(**config_payload.get("generation", {})),
    )

    model = ImportanceMemoryModel(config=config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def task_output_complete(task_dir: str, config: MemoryGenerationConfig, debug: bool) -> bool:
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


def clear_png_dir(path: str) -> None:
    if not os.path.isdir(path):
        return
    for png_path in glob.glob(os.path.join(path, "*.png")):
        os.remove(png_path)


def clear_debug_outputs(task_dir: str, config: MemoryGenerationConfig) -> None:
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
            score_uint8 = (step.score_state.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
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
            "class_weights": model.config.importance.normalized_class_weights(),
            "memory": {
                "score_decay": model.config.memory.score_decay,
                "tau_up": model.config.memory.tau_up,
                "keep_top_ratio": model.config.memory.keep_top_ratio,
            },
        }
        with open(os.path.join(task_dir, "memory_image_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def main(default_data_root: str = "") -> None:
    args = parse_args(default_data_root=default_data_root)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_model_from_checkpoint(args.checkpoint, device)

    if args.data_root:
        model.config.generation.data_root = args.data_root

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
