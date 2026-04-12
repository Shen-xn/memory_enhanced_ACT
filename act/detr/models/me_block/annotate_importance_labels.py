from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np

from .importance_dataset import list_image_files, list_task_dirs, read_four_channel, read_rgb_image, resolve_task_image_dir
from .me_block_config import default_me_block_config

WINDOW_NAME = "Importance Label Annotator"
HELP_LINES = [
    "Paint: left mouse | Erase: right mouse",
    "Classes: 0 background, 1-9 foreground, TAB cycle fg, E erase",
    "Frames: A prev, D next, Z/X jump -/+10, U next unlabeled",
    "Order: global random order across all tasks",
    "Edit: S save only, C clear, R reload, P copy previous frame",
    "View: O overlay toggle, [ ] brush size, Q/ESC quit",
]


@dataclass
class TaskRecord:
    task_dir: str
    name: str
    image_dirname: str
    frame_paths: List[str]
    label_dir: str
    labeled_count: int
    allow_copy_prev: bool


def parse_args(default_data_root: str = "", default_label_dirname: str = "importance_labels") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive labeling tool for me_block importance masks.")
    parser.add_argument("--data-root", type=str, default=default_data_root, help="Root containing task* folders.")
    parser.add_argument(
        "--image-dirname",
        type=str,
        default="auto",
        help="Per-task image directory used for annotation. Default uses four_channel, with special_data falling back to rgb.",
    )
    parser.add_argument(
        "--label-dirname",
        type=str,
        default=default_label_dirname,
        help="Per-task directory used to store label PNGs.",
    )
    parser.add_argument("--task-filter", type=str, default="", help="Only annotate tasks whose name contains this text.")
    parser.add_argument("--task-name", type=str, default="", help="Only annotate this exact task_* directory name or full task path.")
    parser.add_argument("--brush-radius", type=int, default=12, help="Initial brush radius.")
    parser.add_argument("--jump-size", type=int, default=10, help="Frame jump size for Z/X shortcuts.")
    parser.add_argument("--relabel", action="store_true", help="Include tasks that already have labels for every frame.")
    parser.add_argument(
        "--no-copy-prev",
        action="store_true",
        help="Do not initialize new frames from the previous frame label.",
    )
    parser.add_argument(
        "--start-first-frame",
        action="store_true",
        help="Start each task from frame 0 instead of the first unlabeled frame.",
    )
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help="Only visit frames that already have saved labels. Useful for batch cleanup/revision.",
    )
    return parser.parse_args()


def read_label(path: str) -> np.ndarray:
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Failed to read label: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint8)


def save_label(path: str, label: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, label)


def build_palette(class_names: List[str]) -> Dict[int, tuple[int, int, int]]:
    fixed = [
        (70, 70, 255),
        (0, 220, 255),
        (255, 180, 0),
        (120, 255, 120),
        (255, 120, 220),
        (255, 255, 120),
        (160, 120, 255),
        (120, 255, 255),
        (255, 160, 120),
    ]
    palette = {0: (90, 90, 90)}
    for idx, _name in enumerate(class_names, start=1):
        if idx - 1 < len(fixed):
            palette[idx] = fixed[idx - 1]
            continue
        hue = int(((idx - 1) * 33) % 180)
        hsv = np.array([[[hue, 180, 255]]], dtype=np.uint8)
        palette[idx] = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    return palette


def colorize_label(
    label: np.ndarray,
    palette: Dict[int, tuple[int, int, int]],
    unlabeled_index: int,
) -> np.ndarray:
    colored = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for class_index, color in palette.items():
        colored[label == class_index] = color
    unlabeled_mask = label == unlabeled_index
    if np.any(unlabeled_mask):
        colored[unlabeled_mask] = (18, 18, 18)
        yy, xx = np.indices(label.shape)
        checker = ((xx // 12 + yy // 12) % 2 == 0) & unlabeled_mask
        colored[checker] = (35, 35, 35)
    return colored


def label_overlay(
    image_bgr: np.ndarray,
    label: np.ndarray,
    palette: Dict[int, tuple[int, int, int]],
    unlabeled_index: int,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image_bgr.copy()
    colored = colorize_label(label, palette, unlabeled_index)
    mask = label != unlabeled_index
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0.0)
    overlay[mask] = blended[mask]
    return overlay


class ImportanceLabelAnnotator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = default_me_block_config()
        self.unlabeled_index = int(self.config.importance.unlabeled_index)
        self.class_names = list(self.config.importance.class_names)
        self.class_to_index = {name: idx + 1 for idx, name in enumerate(self.class_names)}
        self.palette = build_palette(self.class_names)
        self.selected_class = 1 if self.class_names else 0
        self.brush_radius = max(1, int(args.brush_radius))
        self.copy_prev = not args.no_copy_prev
        self.overlay_enabled = True
        self.status_message = ""
        self.cursor_position = None

        self.tasks = self._build_tasks()
        if not self.tasks:
            raise FileNotFoundError("No matching tasks available for annotation.")

        self.current_task_idx = 0
        self.current_frame_idx = 0
        self.frame_order = self._build_frame_order()
        if not self.frame_order:
            raise FileNotFoundError("No matching frames available for annotation.")
        self.current_order_idx = 0
        self.current_frame = None
        self.current_label = None
        self.current_frame_path = ""
        self.current_label_path = ""
        self.current_label_exists = False
        self.modified = False

        self.current_order_idx = self._start_order_index()
        task_idx, frame_idx = self.frame_order[self.current_order_idx]
        self.current_task_idx = task_idx
        self._load_frame(frame_idx)

    def _build_tasks(self) -> List[TaskRecord]:
        tasks = []
        requested_task_name = os.path.basename(os.path.normpath(self.args.task_name)) if self.args.task_name else ""
        for task_dir in list_task_dirs(self.args.data_root, image_dirname=self.args.image_dirname):
            name = os.path.basename(task_dir)
            if requested_task_name and requested_task_name != name:
                continue
            if self.args.task_filter and self.args.task_filter not in name:
                continue
            image_dirname = resolve_task_image_dir(task_dir, self.args.image_dirname)
            if not image_dirname:
                continue

            frame_paths = list_image_files(os.path.join(task_dir, image_dirname))
            if not frame_paths:
                continue

            label_dir = os.path.join(task_dir, self.args.label_dirname)
            labeled_count = sum(1 for frame_path in frame_paths if os.path.exists(self._label_path(label_dir, frame_path)))
            if self.args.labeled_only and labeled_count == 0:
                continue
            if not self.args.labeled_only and labeled_count == len(frame_paths) and not self.args.relabel:
                continue

            tasks.append(
                TaskRecord(
                    task_dir=task_dir,
                    name=name,
                    image_dirname=image_dirname,
                    frame_paths=frame_paths,
                    label_dir=label_dir,
                    labeled_count=labeled_count,
                    allow_copy_prev=name.startswith("task"),
                )
            )
        return tasks

    def _read_display_image(self, frame_path: str) -> np.ndarray:
        if self._current_task().image_dirname == "rgb":
            return read_rgb_image(frame_path)
        image = read_four_channel(frame_path)
        return image[:, :, :3].copy()

    def _label_path(self, label_dir: str, frame_path: str) -> str:
        frame_stem = os.path.splitext(os.path.basename(frame_path))[0]
        return os.path.join(label_dir, f"{frame_stem}.png")

    def _build_frame_order(self) -> List[tuple[int, int]]:
        order = []
        for task_idx, task in enumerate(self.tasks):
            for frame_idx in range(len(task.frame_paths)):
                if self.args.labeled_only and not self._label_exists_for_task_frame(task, frame_idx):
                    continue
                if not self.args.labeled_only and not self.args.relabel and self._label_exists_for_task_frame(task, frame_idx):
                    continue
                order.append((task_idx, frame_idx))

        rng = np.random.default_rng(self.config.training.seed)
        if order:
            rng.shuffle(order)
        return order

    def _current_task(self) -> TaskRecord:
        return self.tasks[self.current_task_idx]

    def _frame_name(self) -> str:
        return os.path.basename(self.current_frame_path)

    def _label_exists_for_task_frame(self, task: TaskRecord, frame_idx: int) -> bool:
        return os.path.exists(self._label_path(task.label_dir, task.frame_paths[frame_idx]))

    def _start_order_index(self) -> int:
        if self.args.labeled_only:
            return 0
        if self.args.start_first_frame:
            return 0
        for order_idx, (task_idx, frame_idx) in enumerate(self.frame_order):
            task = self.tasks[task_idx]
            if not self._label_exists_for_task_frame(task, frame_idx):
                return order_idx
        return 0

    def _load_frame(self, frame_idx: int) -> None:
        task = self._current_task()
        frame_idx = int(np.clip(frame_idx, 0, len(task.frame_paths) - 1))
        frame_path = task.frame_paths[frame_idx]
        label_path = self._label_path(task.label_dir, frame_path)
        frame = self._read_display_image(frame_path)
        height, width = frame.shape[:2]

        if os.path.exists(label_path):
            label = read_label(label_path)
            label_exists = True
            modified = False
            self.status_message = f"Loaded saved label: {os.path.basename(label_path)}"
        else:
            label = np.full((height, width), self.unlabeled_index, dtype=np.uint8)
            label_exists = False
            modified = False
            if self.copy_prev and task.allow_copy_prev and frame_idx > 0:
                prev_label_path = self._label_path(task.label_dir, task.frame_paths[frame_idx - 1])
                if os.path.exists(prev_label_path):
                    prev_label = read_label(prev_label_path)
                    if prev_label.shape == label.shape:
                        label = prev_label.copy()
                        modified = True
                        self.status_message = "Initialized from previous frame label."
            if not modified:
                self.status_message = "Started with empty label."

        self.current_frame_idx = frame_idx
        self.current_frame = frame
        self.current_label = label
        self.current_frame_path = frame_path
        self.current_label_path = label_path
        self.current_label_exists = label_exists
        self.modified = modified

    def _save_current_label(self) -> None:
        if self.current_label is None:
            return
        task = self._current_task()
        if not self.current_label_exists:
            task.labeled_count += 1
        save_label(self.current_label_path, self.current_label)
        preview_dir = os.path.join(task.task_dir, f"{self.args.label_dirname}_preview")
        preview_path = os.path.join(preview_dir, os.path.basename(self.current_label_path))
        preview = colorize_label(self.current_label, self.palette, self.unlabeled_index)
        os.makedirs(preview_dir, exist_ok=True)
        cv2.imwrite(preview_path, preview)
        self.current_label_exists = True
        self.modified = False
        self._write_task_meta(task)
        self.status_message = f"Saved {self._frame_name()}"

    def _write_task_meta(self, task: TaskRecord) -> None:
        payload = {
            "task": task.name,
            "label_dirname": self.args.label_dirname,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "class_names": self.class_names,
            "class_to_index": self.class_to_index,
        }
        with open(os.path.join(task.task_dir, "importance_labels_meta.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _reset_from_saved(self) -> None:
        if self.current_label_exists:
            self.current_label = read_label(self.current_label_path)
            self.modified = False
            self.status_message = "Reverted to saved label."
            return
        self.current_label = np.full_like(self.current_label, self.unlabeled_index)
        self.modified = False
        self.status_message = "Cleared unsaved label."

    def _copy_previous_frame_label(self) -> None:
        if self.current_frame_idx == 0:
            self.status_message = "No previous frame in this task."
            return
        task = self._current_task()
        prev_path = self._label_path(task.label_dir, task.frame_paths[self.current_frame_idx - 1])
        if not os.path.exists(prev_path):
            self.status_message = "Previous frame has no saved label."
            return
        prev_label = read_label(prev_path)
        if prev_label.shape != self.current_label.shape:
            self.status_message = "Previous label shape mismatch."
            return
        self.current_label = prev_label
        self.modified = True
        self.status_message = "Copied previous frame label."

    def _move_to_order(self, order_idx: int) -> bool:
        if not (0 <= order_idx < len(self.frame_order)):
            return False
        had_unsaved = self.modified
        task_idx, frame_idx = self.frame_order[order_idx]
        self.current_order_idx = order_idx
        self.current_task_idx = task_idx
        self._load_frame(frame_idx)
        if had_unsaved:
            self.status_message = f"Unsaved edits discarded. {self.status_message}"
        return True

    def _move_frame(self, delta: int) -> bool:
        target_order_idx = self.current_order_idx + delta
        if not (0 <= target_order_idx < len(self.frame_order)):
            self.status_message = "Already at the boundary of the random frame order."
            return False
        return self._move_to_order(target_order_idx)

    def _next_unlabeled(self) -> bool:
        for order_idx in range(self.current_order_idx + 1, len(self.frame_order)):
            task_idx, frame_idx = self.frame_order[order_idx]
            task = self.tasks[task_idx]
            if not self._label_exists_for_task_frame(task, frame_idx):
                return self._move_to_order(order_idx)
        self.status_message = "No further unlabeled frames found."
        return False

    def _cycle_class(self) -> None:
        if not self.class_names:
            self.selected_class = 0
            return
        self.selected_class += 1
        if self.selected_class > len(self.class_names):
            self.selected_class = 1

    def _paint(self, x: int, y: int, value: int) -> None:
        if self.current_label is None:
            return
        height, width = self.current_label.shape
        canvas_h = height * 2
        canvas_w = width * 2
        if not (0 <= x < canvas_w and 0 <= y < canvas_h):
            return

        img_x = x % width
        img_y = y % height
        cv2.circle(self.current_label, (img_x, img_y), self.brush_radius, int(value), thickness=-1)
        self.cursor_position = (img_x, img_y)
        self.modified = True

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        if self.current_label is None:
            return
        height, width = self.current_label.shape
        if 0 <= x < width * 2 and 0 <= y < height * 2:
            self.cursor_position = (x % width, y % height)
        else:
            self.cursor_position = None

        if event == cv2.EVENT_LBUTTONDOWN:
            self._paint(x, y, self.selected_class)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._paint(x, y, self.unlabeled_index)
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self._paint(x, y, self.selected_class)
            elif flags & cv2.EVENT_FLAG_RBUTTON:
                self._paint(x, y, self.unlabeled_index)

    def _render(self) -> np.ndarray:
        frame = self.current_frame
        label = self.current_label
        image_bgr = frame.copy()
        overlay = (
            label_overlay(image_bgr, label, self.palette, self.unlabeled_index)
            if self.overlay_enabled
            else image_bgr.copy()
        )
        label_bgr = colorize_label(label, self.palette, self.unlabeled_index)

        if self.cursor_position is not None:
            cv2.circle(overlay, self.cursor_position, self.brush_radius, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(label_bgr, self.cursor_position, self.brush_radius, (255, 255, 255), 1, cv2.LINE_AA)

        top = np.hstack([overlay, image_bgr])
        bottom = np.hstack([label_bgr, image_bgr])
        canvas = np.vstack([top, bottom])

        self._draw_titles(canvas, image_bgr.shape[1], image_bgr.shape[0])
        self._draw_status(canvas)
        return canvas

    def _draw_titles(self, canvas: np.ndarray, width: int, height: int) -> None:
        titles = [
            ("Overlay", 8, 26),
            ("RGB", width + 8, 26),
            ("Label", 8, height + 26),
            ("RGB Ref", width + 8, height + 26),
        ]
        for text, x, y in titles:
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 1, cv2.LINE_AA)

    def _draw_status(self, canvas: np.ndarray) -> None:
        task = self._current_task()
        status_lines = [
            f"Random frame {self.current_order_idx + 1}/{len(self.frame_order)} | Task {task.name}",
            f"Task-local frame {self.current_frame_idx + 1}/{len(task.frame_paths)}: {self._frame_name()}",
            f"Labeled frames in task: {task.labeled_count}/{len(task.frame_paths)}",
            f"Selected class: {self.selected_class_name()}  |  Brush: {self.brush_radius}px  |  Overlay: {'ON' if self.overlay_enabled else 'OFF'}",
            f"Modified: {'YES' if self.modified else 'NO'}  |  Existing label: {'YES' if self.current_label_exists else 'NO'}",
            self.status_message,
        ]

        y = 24
        for line in status_lines:
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
            y += 24

        legend_y = y + 10
        legend_x = 16
        legend_items = [("0", "background", self.palette[0]), ("E", "erase/unlabeled", (40, 40, 40))]
        legend_items.extend((str(idx), name, self.palette[idx]) for idx, name in enumerate(self.class_names, start=1))
        for hotkey, name, color in legend_items:
            cv2.rectangle(canvas, (legend_x, legend_y - 14), (legend_x + 16, legend_y + 2), color, thickness=-1)
            text = f"{hotkey}:{name}"
            cv2.putText(canvas, text, (legend_x + 24, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (legend_x + 24, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            legend_x += 170

        help_y = canvas.shape[0] - (len(HELP_LINES) * 22) - 12
        for line in HELP_LINES:
            cv2.putText(canvas, line, (16, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (16, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 1, cv2.LINE_AA)
            help_y += 22

    def selected_class_name(self) -> str:
        if self.selected_class == self.unlabeled_index:
            return "erase/unlabeled"
        if self.selected_class == 0:
            return "background"
        index = self.selected_class - 1
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return f"class_{self.selected_class}"

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        try:
            while True:
                canvas = self._render()
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(30) & 0xFF
                if key == 255:
                    continue
                if not self._handle_key(key):
                    break
        finally:
            cv2.destroyAllWindows()

    def _handle_key(self, key: int) -> bool:
        if key in (27, ord("q")):
            return False
        if key == ord("s"):
            self._save_current_label()
        elif key == ord("c"):
            self.current_label.fill(self.unlabeled_index)
            self.modified = True
            self.status_message = "Cleared current label."
        elif key == ord("r"):
            self._reset_from_saved()
        elif key == ord("p"):
            self._copy_previous_frame_label()
        elif key == ord("o"):
            self.overlay_enabled = not self.overlay_enabled
            self.status_message = f"Overlay {'enabled' if self.overlay_enabled else 'disabled'}."
        elif key == ord("a"):
            self._move_frame(-1)
        elif key == ord("d"):
            self._move_frame(1)
        elif key == ord("z"):
            self._move_frame(-self.args.jump_size)
        elif key == ord("x"):
            self._move_frame(self.args.jump_size)
        elif key == ord("u"):
            self._next_unlabeled()
        elif key == ord("["):
            self.brush_radius = max(1, self.brush_radius - 2)
            self.status_message = f"Brush radius: {self.brush_radius}"
        elif key == ord("]"):
            self.brush_radius += 2
            self.status_message = f"Brush radius: {self.brush_radius}"
        elif key == 9:
            self._cycle_class()
            self.status_message = f"Selected class: {self.selected_class_name()}"
        elif key == ord("0"):
            self.selected_class = 0
            self.status_message = "Selected class: background"
        elif key == ord("e"):
            self.selected_class = self.unlabeled_index
            self.status_message = "Selected class: erase/unlabeled"
        elif ord("1") <= key <= ord("9"):
            class_index = key - ord("0")
            if class_index <= len(self.class_names):
                self.selected_class = class_index
                self.status_message = f"Selected class: {self.selected_class_name()}"
        return True


def main(default_data_root: str = "", default_label_dirname: str = "importance_labels") -> None:
    args = parse_args(default_data_root=default_data_root, default_label_dirname=default_label_dirname)
    annotator = ImportanceLabelAnnotator(args)
    print(f"Loaded {len(annotator.tasks)} tasks from: {args.data_root}")
    print("Launching annotation window...")
    annotator.run()


if __name__ == "__main__":
    main()
