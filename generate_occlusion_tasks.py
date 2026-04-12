#!/usr/bin/env python3
"""
Automatic occlusion generation for four-channel task data.

Source tasks:
    task_*

Generated tasks:
    task_obst_*

This script only reads and writes `four_channel/*.png`, plus copies CSV metadata
so the generated tasks remain compatible with the current data loader.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np


DEFAULT_TRIGGER_PROB = 0.01
DEFAULT_CANCEL_PROB = 0.05
DEFAULT_SEED = 42
MIN_OCCLUSION_START_FRAME = 1


@dataclass
class Occluder:
    center: np.ndarray
    velocity: np.ndarray
    relative_polygon: np.ndarray
    bgr_color: np.ndarray
    base_radius: float
    spawn_frame: int
    patch_offset: np.ndarray
    rgb_texture: np.ndarray
    depth_texture: np.ndarray
    alpha_texture: np.ndarray

    def polygon(self) -> np.ndarray:
        pts = self.relative_polygon + self.center[None, :]
        return np.round(pts).astype(np.int32)

    def advance(self) -> None:
        self.center = self.center + self.velocity

    def as_meta(self) -> dict:
        return {
            "center": [float(self.center[0]), float(self.center[1])],
            "velocity": [float(self.velocity[0]), float(self.velocity[1])],
            "relative_polygon": self.relative_polygon.round(3).tolist(),
            "bgr_color": [int(v) for v in self.bgr_color.tolist()],
            "base_radius": float(self.base_radius),
            "spawn_frame": int(self.spawn_frame),
            "patch_offset": [int(self.patch_offset[0]), int(self.patch_offset[1])],
            "depth_range": [int(self.depth_texture.min()), int(self.depth_texture.max())],
        }


def get_data_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "data_process", "data")


def list_source_tasks(data_root: str, task_filter: str | None = None) -> list[str]:
    candidates = sorted(glob.glob(os.path.join(data_root, "task_*")))
    tasks = []
    for path in candidates:
        name = os.path.basename(path)
        if not os.path.isdir(path):
            continue
        if name.startswith("task_obst_"):
            continue
        if task_filter and task_filter not in name:
            continue
        if os.path.isdir(os.path.join(path, "four_channel")):
            tasks.append(path)
    return tasks


def target_task_dir(source_task_dir: str) -> str:
    parent = os.path.dirname(source_task_dir)
    name = os.path.basename(source_task_dir)
    return os.path.join(parent, name.replace("task_", "task_obst_", 1))


def frame_paths(task_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(task_dir, "four_channel", "*.png")))


def task_complete(source_task_dir: str, generated_task_dir: str) -> bool:
    src_frames = frame_paths(source_task_dir)
    dst_frames = frame_paths(generated_task_dir)
    meta_path = os.path.join(generated_task_dir, "occlusion_meta.json")
    return bool(src_frames) and len(src_frames) == len(dst_frames) and os.path.exists(meta_path)


def ensure_clean_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for old_png in glob.glob(os.path.join(path, "*.png")):
        os.remove(old_png)


def copy_csv_metadata(source_task_dir: str, generated_task_dir: str) -> None:
    for csv_path in glob.glob(os.path.join(source_task_dir, "*.csv")):
        shutil.copy2(csv_path, os.path.join(generated_task_dir, os.path.basename(csv_path)))


def read_four_channel_png(path: str) -> np.ndarray:
    frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if frame.ndim != 3 or frame.shape[2] != 4:
        raise RuntimeError(f"Expected 4-channel PNG, got shape {frame.shape} from {path}")
    return frame


def _low_frequency_noise(
    rng: np.random.Generator,
    height: int,
    width: int,
    coarse_h: int,
    coarse_w: int,
) -> np.ndarray:
    coarse = rng.normal(0.0, 1.0, size=(coarse_h, coarse_w)).astype(np.float32)
    noise = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(1.0, width * 0.015), sigmaY=max(1.0, height * 0.015))
    noise -= noise.mean()
    noise /= max(noise.std(), 1e-6)
    return noise


def _build_occluder_textures(
    polygon: np.ndarray,
    base_radius: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    soft_edge = max(2, int(base_radius * 0.035))
    min_xy = np.floor(polygon.min(axis=0)).astype(np.int32) - soft_edge * 2
    max_xy = np.ceil(polygon.max(axis=0)).astype(np.int32) + soft_edge * 2
    size = np.maximum(max_xy - min_xy + 1, 1)
    patch_w, patch_h = int(size[0]), int(size[1])

    local_polygon = np.round(polygon - min_xy[None, :]).astype(np.int32)
    mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon], 255)

    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=max(0.8, soft_edge * 0.55), sigmaY=max(0.8, soft_edge * 0.55))
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[mask > 0] = np.maximum(alpha[mask > 0], 0.92)

    yy, xx = np.mgrid[0:patch_h, 0:patch_w].astype(np.float32)
    center_x = (patch_w - 1) * 0.5
    center_y = (patch_h - 1) * 0.5
    nx = (xx - center_x) / max(center_x, 1.0)
    ny = (yy - center_y) / max(center_y, 1.0)
    radial = np.sqrt(nx * nx + ny * ny)
    radial = np.clip(radial, 0.0, 1.0)

    light_angle = rng.uniform(0.0, 2.0 * np.pi)
    directional = 0.5 * (nx * np.cos(light_angle) + ny * np.sin(light_angle) + 1.0)

    noise_large = _low_frequency_noise(rng, patch_h, patch_w, 5, 5)
    noise_small = _low_frequency_noise(rng, patch_h, patch_w, 10, 10)

    palette = np.array(
        [
            [60, 60, 60],
            [85, 80, 72],
            [92, 88, 84],
            [70, 78, 92],
            [56, 68, 74],
            [96, 92, 78],
        ],
        dtype=np.float32,
    )
    base_color = palette[int(rng.integers(0, len(palette)))]
    shading = 0.88 + directional[..., None] * 0.18 - radial[..., None] * 0.08
    texture_noise = noise_large[..., None] * 7.0 + noise_small[..., None] * 3.0
    rgb_texture = np.clip(base_color[None, None, :] * shading + texture_noise, 28.0, 185.0).astype(np.uint8)

    depth_base = float(rng.uniform(6.0, 32.0))
    depth_slope = directional * rng.uniform(-4.0, 4.0)
    depth_noise = noise_large * 1.6 + noise_small * 0.8
    depth_texture = np.clip(depth_base + depth_slope + depth_noise, 1.0, 80.0).astype(np.uint8)

    mean_alpha = max(alpha.sum(), 1e-6)
    mean_color = (rgb_texture.astype(np.float32) * alpha[..., None]).sum(axis=(0, 1)) / mean_alpha
    return min_xy.astype(np.float32), rgb_texture, depth_texture, alpha.astype(np.float32), mean_color.astype(np.uint8)


def build_random_occluder(
    spawn_point: tuple[int, int],
    width: int,
    height: int,
    frame_idx: int,
    rng: np.random.Generator,
) -> Occluder:
    min_dim = min(width, height)
    base_radius = float(rng.uniform(0.03, 0.7) * min_dim)
    num_vertices = int(rng.integers(7, 13))

    angles = np.linspace(0.0, 2.0 * np.pi, num_vertices, endpoint=False)
    angles += rng.uniform(-np.pi / num_vertices * 0.45, np.pi / num_vertices * 0.45, size=num_vertices)
    angles = np.sort(angles)

    radii = base_radius * rng.uniform(0.65, 1.35, size=num_vertices)
    rel_x = np.cos(angles) * radii
    rel_y = np.sin(angles) * radii
    polygon = np.stack([rel_x, rel_y], axis=1).astype(np.float32)
    patch_offset, rgb_texture, depth_texture, alpha_texture, mean_color = _build_occluder_textures(
        polygon,
        base_radius,
        rng,
    )

    speed = float(rng.uniform(0.004, 0.012) * min_dim)
    direction = float(rng.uniform(0.0, 2.0 * np.pi))
    velocity = np.array([np.cos(direction) * speed, np.sin(direction) * speed], dtype=np.float32)

    return Occluder(
        center=np.array(spawn_point, dtype=np.float32),
        velocity=velocity,
        relative_polygon=polygon,
        bgr_color=mean_color,
        base_radius=base_radius,
        spawn_frame=frame_idx,
        patch_offset=patch_offset,
        rgb_texture=rgb_texture,
        depth_texture=depth_texture,
        alpha_texture=alpha_texture,
    )


def apply_occluder(frame: np.ndarray, occluder: Occluder, rng: np.random.Generator) -> np.ndarray:
    output = frame.copy()
    height, width = frame.shape[:2]

    patch_h, patch_w = occluder.alpha_texture.shape
    top_left = np.round(occluder.center + occluder.patch_offset).astype(np.int32)
    x0, y0 = int(top_left[0]), int(top_left[1])
    x1, y1 = x0 + patch_w, y0 + patch_h

    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(width, x1)
    dst_y1 = min(height, y1)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return output

    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    alpha = occluder.alpha_texture[src_y0:src_y1, src_x0:src_x1][..., None]
    occ_rgb = occluder.rgb_texture[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    occ_depth = occluder.depth_texture[src_y0:src_y1, src_x0:src_x1].astype(np.float32)

    roi_rgb = output[dst_y0:dst_y1, dst_x0:dst_x1, :3].astype(np.float32)
    roi_depth = output[dst_y0:dst_y1, dst_x0:dst_x1, 3].astype(np.float32)

    blended_rgb = occ_rgb * alpha + roi_rgb * (1.0 - alpha)
    hard_mask = (alpha[..., 0] > 0.35).astype(np.uint8)
    blended_depth = roi_depth.copy()
    blended_depth[hard_mask > 0] = occ_depth[hard_mask > 0]

    # Simulate a shallow boundary band with missing / jittered depth, but still cover the original depth.
    edge_radius = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_radius * 2 + 1, edge_radius * 2 + 1))
    dilated = cv2.dilate(hard_mask, kernel)
    eroded = cv2.erode(hard_mask, kernel)
    outer_band = (dilated > 0) & (hard_mask == 0)
    inner_band = (hard_mask > 0) & (eroded == 0)

    if np.any(outer_band):
        missing_prob = 0.55
        outer_missing = rng.random(outer_band.shape) < missing_prob
        blended_depth[outer_band & outer_missing] = 0.0
        outer_jitter = rng.normal(loc=4.0, scale=2.0, size=outer_band.shape).astype(np.float32)
        outer_values = np.clip(occ_depth + outer_jitter, 0.0, 255.0)
        blended_depth[outer_band & ~outer_missing] = outer_values[outer_band & ~outer_missing]

    if np.any(inner_band):
        inner_missing = rng.random(inner_band.shape) < 0.18
        blended_depth[inner_band & inner_missing] = 0.0
        inner_jitter = rng.normal(loc=0.0, scale=2.0, size=inner_band.shape).astype(np.float32)
        inner_values = np.clip(occ_depth + inner_jitter, 0.0, 255.0)
        blended_depth[inner_band & ~inner_missing] = inner_values[inner_band & ~inner_missing]

    output[dst_y0:dst_y1, dst_x0:dst_x1, :3] = np.clip(blended_rgb, 0.0, 255.0).astype(np.uint8)
    output[dst_y0:dst_y1, dst_x0:dst_x1, 3] = np.clip(blended_depth, 0.0, 255.0).astype(np.uint8)
    return output


def occluder_out_of_frame(occluder: Occluder, width: int, height: int) -> bool:
    polygon = occluder.polygon()
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    return x_max < 0 or y_max < 0 or x_min >= width or y_min >= height


def sample_spawn_point(width: int, height: int, rng: np.random.Generator) -> tuple[int, int]:
    return int(rng.integers(0, width)), int(rng.integers(0, height))


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def process_task(
    source_task_dir: str,
    generated_task_dir: str,
    rng: np.random.Generator,
    trigger_prob: float,
    cancel_prob: float,
) -> None:
    source_name = os.path.basename(source_task_dir)
    target_name = os.path.basename(generated_task_dir)
    source_frames = frame_paths(source_task_dir)

    os.makedirs(generated_task_dir, exist_ok=True)
    generated_four_channel_dir = os.path.join(generated_task_dir, "four_channel")
    ensure_clean_dir(generated_four_channel_dir)
    copy_csv_metadata(source_task_dir, generated_task_dir)

    occluder: Occluder | None = None
    current_event: dict | None = None
    events: list[dict] = []

    for frame_idx, source_frame_path in enumerate(source_frames):
        frame_name = os.path.basename(source_frame_path)
        frame = read_four_channel_png(source_frame_path)
        height, width = frame.shape[:2]

        can_start_occlusion = frame_idx >= MIN_OCCLUSION_START_FRAME
        if can_start_occlusion and occluder is None and rng.random() < trigger_prob:
            spawn_point = sample_spawn_point(width, height, rng)
            occluder = build_random_occluder(spawn_point, width, height, frame_idx, rng)
            current_event = {
                "start_frame": frame_idx,
                "start_image": frame_name,
                "spawn_point": [int(spawn_point[0]), int(spawn_point[1])],
                "occluder": occluder.as_meta(),
            }

        if occluder is not None:
            output_frame = apply_occluder(frame, occluder, rng)
        else:
            output_frame = frame.copy()

        cv2.imwrite(os.path.join(generated_four_channel_dir, frame_name), output_frame)

        if occluder is not None:
            occluder.advance()
            end_reason = None
            if occluder_out_of_frame(occluder, width, height):
                end_reason = "out_of_frame"
            elif rng.random() < cancel_prob:
                end_reason = "random_cancel"

            if end_reason is not None and current_event is not None:
                current_event["end_frame"] = frame_idx
                current_event["end_image"] = frame_name
                current_event["duration_frames"] = frame_idx - current_event["start_frame"] + 1
                current_event["end_reason"] = end_reason
                events.append(current_event)
                current_event = None
                occluder = None

    if current_event is not None:
        current_event["end_frame"] = len(source_frames) - 1
        current_event["end_image"] = os.path.basename(source_frames[-1])
        current_event["duration_frames"] = len(source_frames) - current_event["start_frame"]
        current_event["end_reason"] = "task_end"
        events.append(current_event)

    meta = {
        "source_task": source_name,
        "generated_task": target_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_count": len(source_frames),
        "min_occlusion_start_frame": MIN_OCCLUSION_START_FRAME,
        "trigger_probability": trigger_prob,
        "cancel_probability": cancel_prob,
        "events": events,
    }
    save_json(os.path.join(generated_task_dir, "occlusion_meta.json"), meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate automatic occluded task_* datasets.")
    parser.add_argument("--data-root", type=str, default=get_data_root(), help="Root directory containing task_* folders.")
    parser.add_argument("--task-filter", type=str, default=None, help="Only process source tasks whose name contains this text.")
    parser.add_argument("--force", action="store_true", help="Regenerate tasks even if matching task_obst_* already exists.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--trigger-prob", type=float, default=DEFAULT_TRIGGER_PROB, help="Per-frame probability of triggering a new occluder.")
    parser.add_argument("--cancel-prob", type=float, default=DEFAULT_CANCEL_PROB, help="Per-frame probability of canceling an active occluder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    tasks = list_source_tasks(args.data_root, task_filter=args.task_filter)

    if not tasks:
        print("No source task_* folders found.")
        return

    print(f"Found {len(tasks)} source tasks under: {args.data_root}")

    processed = 0
    skipped = 0

    for source_task_dir in tasks:
        generated_task_dir = target_task_dir(source_task_dir)
        source_name = os.path.basename(source_task_dir)
        target_name = os.path.basename(generated_task_dir)

        if not args.force and task_complete(source_task_dir, generated_task_dir):
            print(f"Skip completed task: {source_name} -> {target_name}")
            skipped += 1
            continue

        print(f"Processing: {source_name} -> {target_name}")
        process_task(
            source_task_dir=source_task_dir,
            generated_task_dir=generated_task_dir,
            rng=rng,
            trigger_prob=args.trigger_prob,
            cancel_prob=args.cancel_prob,
        )
        processed += 1
        print(f"Finished: {source_name}")

    print(f"Summary: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
