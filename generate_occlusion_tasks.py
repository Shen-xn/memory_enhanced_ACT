#!/usr/bin/env python3
"""
Interactive occlusion generation for four-channel task data.

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


WINDOW_NAME = "Occlusion Task Generator"
DEFAULT_TRIGGER_PROB = 0.01
DEFAULT_CANCEL_PROB = 0.05
DEFAULT_FRAME_DELAY_MS = 30
DEFAULT_SEED = 42


@dataclass
class ClickState:
    frame_width: int = 0
    frame_height: int = 0
    point: tuple[int, int] | None = None
    awaiting: bool = False

    def reset(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.point = None
        self.awaiting = True


@dataclass
class Occluder:
    center: np.ndarray
    velocity: np.ndarray
    relative_polygon: np.ndarray
    bgr_color: np.ndarray
    base_radius: float
    spawn_frame: int

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
        }


def on_mouse(event: int, x: int, y: int, _flags: int, state: ClickState) -> None:
    if not state.awaiting or event != cv2.EVENT_LBUTTONDOWN:
        return

    if x >= state.frame_width:
        x -= state.frame_width

    x = int(np.clip(x, 0, state.frame_width - 1))
    y = int(np.clip(y, 0, state.frame_height - 1))
    state.point = (x, y)
    state.awaiting = False


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


def depth_preview(depth_channel: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(depth_channel, cv2.COLORMAP_TURBO)


def compose_preview(
    frame: np.ndarray,
    task_name: str,
    frame_idx: int,
    total_frames: int,
    active: bool,
    paused_for_click: bool = False,
) -> np.ndarray:
    rgb = frame[:, :, :3].copy()
    depth = depth_preview(frame[:, :, 3])
    preview = np.hstack([rgb, depth])

    lines = [
        f"{task_name}  frame {frame_idx + 1}/{total_frames}",
        "Left: RGB   Right: depth",
        "Q/ESC: quit   S: skip task",
    ]

    if paused_for_click:
        lines.append("Click to place occluder   C: cancel trigger")
    elif active:
        lines.append("Occluder active")
    else:
        lines.append("Playing")

    y = 24
    for line in lines:
        cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        y += 24

    return preview


def sample_occluder_color(rng: np.random.Generator) -> np.ndarray:
    hsv = np.array(
        [[[rng.integers(0, 180), rng.integers(35, 110), rng.integers(55, 150)]]],
        dtype=np.uint8,
    )
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return color.astype(np.uint8)


def build_random_occluder(
    click_point: tuple[int, int],
    width: int,
    height: int,
    frame_idx: int,
    rng: np.random.Generator,
) -> Occluder:
    min_dim = min(width, height)
    base_radius = float(rng.uniform(0.08, 0.2) * min_dim)
    num_vertices = int(rng.integers(7, 13))

    angles = np.linspace(0.0, 2.0 * np.pi, num_vertices, endpoint=False)
    angles += rng.uniform(-np.pi / num_vertices * 0.45, np.pi / num_vertices * 0.45, size=num_vertices)
    angles = np.sort(angles)

    radii = base_radius * rng.uniform(0.65, 1.35, size=num_vertices)
    rel_x = np.cos(angles) * radii
    rel_y = np.sin(angles) * radii
    polygon = np.stack([rel_x, rel_y], axis=1).astype(np.float32)

    speed = float(rng.uniform(0.004, 0.012) * min_dim)
    direction = float(rng.uniform(0.0, 2.0 * np.pi))
    velocity = np.array([np.cos(direction) * speed, np.sin(direction) * speed], dtype=np.float32)

    return Occluder(
        center=np.array(click_point, dtype=np.float32),
        velocity=velocity,
        relative_polygon=polygon,
        bgr_color=sample_occluder_color(rng),
        base_radius=base_radius,
        spawn_frame=frame_idx,
    )


def apply_occluder(frame: np.ndarray, occluder: Occluder) -> tuple[np.ndarray, np.ndarray]:
    output = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    polygon = occluder.polygon()
    cv2.fillPoly(mask, [polygon], 255)

    output[mask > 0, :3] = occluder.bgr_color
    output[mask > 0, 3] = 0
    return output, mask


def occluder_out_of_frame(occluder: Occluder, width: int, height: int) -> bool:
    polygon = occluder.polygon()
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    return x_max < 0 or y_max < 0 or x_min >= width or y_min >= height


def wait_for_click(preview: np.ndarray, state: ClickState) -> str:
    while True:
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(20) & 0xFF
        if state.point is not None:
            return "clicked"
        if key in (ord("q"), 27):
            return "quit"
        if key in (ord("c"), ord("n")):
            state.awaiting = False
            return "cancel"


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def process_task(
    source_task_dir: str,
    generated_task_dir: str,
    rng: np.random.Generator,
    trigger_prob: float,
    cancel_prob: float,
    frame_delay_ms: int,
    click_state: ClickState,
) -> str:
    source_name = os.path.basename(source_task_dir)
    target_name = os.path.basename(generated_task_dir)
    source_frames = frame_paths(source_task_dir)

    os.makedirs(generated_task_dir, exist_ok=True)
    generated_four_channel_dir = os.path.join(generated_task_dir, "four_channel")
    generated_mask_dir = os.path.join(generated_task_dir, "occlusion_masks")
    ensure_clean_dir(generated_four_channel_dir)
    ensure_clean_dir(generated_mask_dir)
    copy_csv_metadata(source_task_dir, generated_task_dir)

    occluder: Occluder | None = None
    current_event: dict | None = None
    events: list[dict] = []

    for frame_idx, source_frame_path in enumerate(source_frames):
        frame_name = os.path.basename(source_frame_path)
        frame = read_four_channel_png(source_frame_path)
        height, width = frame.shape[:2]

        if occluder is None and rng.random() < trigger_prob:
            paused_preview = compose_preview(
                frame=frame,
                task_name=source_name,
                frame_idx=frame_idx,
                total_frames=len(source_frames),
                active=False,
                paused_for_click=True,
            )
            click_state.reset(width, height)
            result = wait_for_click(paused_preview, click_state)
            if result == "quit":
                return "quit"
            if result == "clicked" and click_state.point is not None:
                occluder = build_random_occluder(click_state.point, width, height, frame_idx, rng)
                current_event = {
                    "start_frame": frame_idx,
                    "start_image": frame_name,
                    "click_point": [int(click_state.point[0]), int(click_state.point[1])],
                    "occluder": occluder.as_meta(),
                }

        if occluder is not None:
            output_frame, mask = apply_occluder(frame, occluder)
        else:
            output_frame = frame.copy()
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        preview = compose_preview(
            frame=output_frame,
            task_name=source_name,
            frame_idx=frame_idx,
            total_frames=len(source_frames),
            active=occluder is not None,
        )
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(frame_delay_ms) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "skip"

        cv2.imwrite(os.path.join(generated_four_channel_dir, frame_name), output_frame)
        cv2.imwrite(os.path.join(generated_mask_dir, frame_name), mask)

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
        "trigger_probability": trigger_prob,
        "cancel_probability": cancel_prob,
        "frame_delay_ms": frame_delay_ms,
        "events": events,
    }
    save_json(os.path.join(generated_task_dir, "occlusion_meta.json"), meta)
    return "done"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interactive occluded task_* datasets.")
    parser.add_argument("--data-root", type=str, default=get_data_root(), help="Root directory containing task_* folders.")
    parser.add_argument("--task-filter", type=str, default=None, help="Only process source tasks whose name contains this text.")
    parser.add_argument("--force", action="store_true", help="Regenerate tasks even if matching task_obst_* already exists.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--trigger-prob", type=float, default=DEFAULT_TRIGGER_PROB, help="Per-frame probability of triggering a new occluder.")
    parser.add_argument("--cancel-prob", type=float, default=DEFAULT_CANCEL_PROB, help="Per-frame probability of canceling an active occluder.")
    parser.add_argument("--frame-delay-ms", type=int, default=DEFAULT_FRAME_DELAY_MS, help="Playback delay between frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    tasks = list_source_tasks(args.data_root, task_filter=args.task_filter)

    if not tasks:
        print("No source task_* folders found.")
        return

    print(f"Found {len(tasks)} source tasks under: {args.data_root}")
    print("Controls: left click = place occluder, C = cancel trigger, S = skip task, Q/ESC = quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    click_state = ClickState()
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, click_state)

    processed = 0
    skipped = 0

    try:
        for source_task_dir in tasks:
            generated_task_dir = target_task_dir(source_task_dir)
            source_name = os.path.basename(source_task_dir)
            target_name = os.path.basename(generated_task_dir)

            if not args.force and task_complete(source_task_dir, generated_task_dir):
                print(f"Skip completed task: {source_name} -> {target_name}")
                skipped += 1
                continue

            print(f"Processing: {source_name} -> {target_name}")
            result = process_task(
                source_task_dir=source_task_dir,
                generated_task_dir=generated_task_dir,
                rng=rng,
                trigger_prob=args.trigger_prob,
                cancel_prob=args.cancel_prob,
                frame_delay_ms=args.frame_delay_ms,
                click_state=click_state,
            )

            if result == "quit":
                print("Stopped by user.")
                break
            if result == "skip":
                print(f"Skipped current task: {source_name}")
                skipped += 1
                continue

            processed += 1
            print(f"Finished: {source_name}")
    finally:
        cv2.destroyAllWindows()

    print(f"Summary: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
