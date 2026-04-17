#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider


DEFAULT_JOINT_COLUMNS = ["j1", "j2", "j3", "j4", "j5", "j10"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive dataset reviewer: drag the trajectory progress bar to inspect the matching image."
    )
    parser.add_argument("--task-dir", required=True, help="Path to one task directory containing states.csv and rgb/")
    parser.add_argument(
        "--image-dir-name",
        default="rgb",
        help="Image subdirectory name inside task-dir. Default: rgb",
    )
    return parser.parse_args()


def find_image_path(image_dir: Path, frame_idx: int) -> Path:
    stem = f"{frame_idx:06d}"
    for ext in (".jpg", ".png", ".jpeg", ".bmp"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for frame {frame_idx} under {image_dir}")


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def infer_joint_columns(rows) -> List[str]:
    fieldnames = list(rows[0].keys())
    inferred = [c for c in DEFAULT_JOINT_COLUMNS if c in fieldnames]
    if inferred:
        return inferred
    fallback = [c for c in fieldnames if c.startswith("j")]
    if fallback:
        return fallback
    raise ValueError("Could not infer joint columns from states.csv")


def load_image_rgb(image_path: Path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    task_dir = Path(args.task_dir).expanduser().resolve()
    csv_path = task_dir / "states.csv"
    image_dir = task_dir / args.image_dir_name

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing states.csv: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    rows = load_rows(csv_path)
    joint_columns = infer_joint_columns(rows)
    frame_ids = [int(row["frame"]) for row in rows]
    joint_values = {joint: [float(row[joint]) for row in rows] for joint in joint_columns}

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(
        nrows=len(joint_columns) + 2,
        ncols=2,
        width_ratios=[1.2, 1.8],
        height_ratios=[1] * len(joint_columns) + [0.35, 0.35],
        figure=fig,
    )

    image_ax = fig.add_subplot(gs[: len(joint_columns), 0])
    curve_axes = [fig.add_subplot(gs[i, 1]) for i in range(len(joint_columns))]
    slider_ax = fig.add_subplot(gs[len(joint_columns), :])
    help_ax = fig.add_subplot(gs[len(joint_columns) + 1, :])
    help_ax.axis("off")

    image_ax.set_title(task_dir.name)
    image_ax.axis("off")

    vertical_lines = []
    markers = []

    for ax, joint in zip(curve_axes, joint_columns):
        ax.plot(frame_ids, joint_values[joint], color="#2a6fbb", linewidth=1.5)
        marker, = ax.plot([frame_ids[0]], [joint_values[joint][0]], "o", color="#d62828", markersize=6)
        vline = ax.axvline(frame_ids[0], color="#d62828", linestyle="--", linewidth=1.2)
        ax.set_ylabel(joint)
        ax.grid(True, alpha=0.25)
        markers.append(marker)
        vertical_lines.append(vline)

    curve_axes[-1].set_xlabel("Frame")
    slider = Slider(
        ax=slider_ax,
        label="Frame Index",
        valmin=0,
        valmax=len(rows) - 1,
        valinit=0,
        valstep=1,
    )

    help_ax.text(
        0.01,
        0.5,
        "Drag the slider or click any trajectory subplot to jump. Left/Right arrows step one frame.",
        fontsize=11,
        va="center",
    )

    image_artist = image_ax.imshow(load_image_rgb(find_image_path(image_dir, frame_ids[0])))
    title_artist = image_ax.text(
        0.02,
        0.98,
        "",
        transform=image_ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=11,
        bbox=dict(facecolor="black", alpha=0.55, boxstyle="round,pad=0.25"),
    )

    def update(index: int):
        index = int(max(0, min(len(rows) - 1, index)))
        frame_id = frame_ids[index]
        image_artist.set_data(load_image_rgb(find_image_path(image_dir, frame_id)))
        for ax, joint, marker, vline in zip(curve_axes, joint_columns, markers, vertical_lines):
            marker.set_data([frame_id], [joint_values[joint][index]])
            vline.set_xdata([frame_id, frame_id])
            ax.set_title("" if ax is not curve_axes[0] else f"Frame {frame_id} / row {index}")

        row = rows[index]
        extras = []
        for key in ("rgb_depth_physical_skew_ms", "img_servo_physical_skew_max_abs_ms", "match_ready_latency_from_physical_max_ms"):
            if key in row and row[key] != "":
                extras.append(f"{key}={float(row[key]):.2f}")
        title_artist.set_text("\n".join([f"frame={frame_id}"] + extras))
        fig.canvas.draw_idle()

    def on_slider_change(val):
        update(int(val))

    def on_click(event):
        for ax in curve_axes:
            if event.inaxes is ax and event.xdata is not None:
                nearest_idx = min(range(len(frame_ids)), key=lambda i: abs(frame_ids[i] - event.xdata))
                slider.set_val(nearest_idx)
                break

    def on_key(event):
        current = int(slider.val)
        if event.key == "right":
            slider.set_val(min(len(rows) - 1, current + 1))
        elif event.key == "left":
            slider.set_val(max(0, current - 1))

    slider.on_changed(on_slider_change)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update(0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
