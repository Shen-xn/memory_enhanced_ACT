#!/usr/bin/env python3
"""One-command ACT dataset preparation.

This is the preferred entry point before training:

1. Clean each raw ``states.csv`` into ``states_clean.csv``.
2. Run the trajectory/image synchronization and filtering stage.
3. Rebuild ``four_channel/*.png`` from RGB and normalized depth.
4. Amplify gripper (``j10``) motion around the dataset mean.
5. Validate the final training contract used by ``data_process.data_loader``.

The old scripts in ``data_process/`` are still the implementation pieces, but
this wrapper keeps the official order and final checks in one place.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_process import data_process_1, data_process_2
from data_process.data_loader import FIXED_JOINT_MAX, FIXED_JOINT_MIN
from data_process.exclusions import EXCLUSION_FILENAME, write_excluded_tasks


JOINT_COLS = ["j1", "j2", "j3", "j4", "j5", "j10"]
CSV_HEADER = ["frame", *JOINT_COLS]
IMAGE_SHAPE = (480, 640, 4)
GRIPPER_COL = "j10"
GRIPPER_SCALE = 1.2
GRIPPER_BACKUP_NAME = "states_filtered.pre_gripper_amp.csv"
GRIPPER_META_NAME = "gripper_amplification.json"


@dataclass
class ValidationResult:
    task_name: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exclude_reason: str | None = None

    def error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def exclude(self, message: str) -> None:
        self.ok = False
        self.exclude_reason = message
        self.warnings.append(message)


@dataclass
class GripperAmplifyResult:
    task_name: str
    ok: bool
    skipped: bool = False
    rows: int = 0
    before_min: float | None = None
    before_max: float | None = None
    before_mean: float | None = None
    after_min: float | None = None
    after_max: float | None = None
    after_mean: float | None = None
    clipped_low: int = 0
    clipped_high: int = 0
    message: str = ""


def natural_sort(paths: Iterable[Path]) -> list[Path]:
    def key(path: Path):
        parts = re.split(r"(\d+)", path.name)
        return [int(part) if part.isdigit() else part.lower() for part in parts]

    return sorted(paths, key=key)


def frame_number_from_path(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"Filename has no frame number: {path}")
    return int(matches[0])


def discover_tasks(data_root: Path) -> list[Path]:
    tasks = [
        path
        for path in natural_sort(data_root.glob("task_*"))
        if path.is_dir() and "task_copy" not in path.name
    ]
    if not tasks:
        raise FileNotFoundError(f"No task_* directories found under {data_root}")
    return tasks


def has_raw_inputs(task_dir: Path) -> bool:
    return (task_dir / "states.csv").exists() and (task_dir / "rgb").exists() and (task_dir / "depth").exists()


def parse_states_line(line: str) -> list[str]:
    """Parse one raw states.csv row into frame + six joints.

    Raw files commonly look like:
    ``0,array('H', [624]),array('H', [512]),...``.
    Empty reads appear as ``array('H')``. We preserve those as blank fields so
    the synchronization stage can delete the corresponding frame explicitly.
    """

    frame_match = re.match(r"\s*(-?\d+)", line)
    frame = frame_match.group(1) if frame_match else ""

    joints = re.findall(r"\[\s*(-?\d+)\s*\]", line)[: len(JOINT_COLS)]
    joints = [*joints, *([""] * (len(JOINT_COLS) - len(joints)))]
    return [frame, *joints[: len(JOINT_COLS)]]


def clean_states_csv(task_dir: Path, strict: bool = False) -> tuple[int, int]:
    raw_path = task_dir / "states.csv"
    clean_path = task_dir / "states_clean.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{task_dir.name}: missing states.csv")

    rows: list[list[str]] = []
    incomplete_rows = 0
    with raw_path.open("r", encoding="utf-8", errors="replace", newline="") as src:
        lines = src.readlines()

    for line_no, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parsed = parse_states_line(line)
        if not parsed[0] or any(value == "" for value in parsed[1:]):
            incomplete_rows += 1
            if strict:
                raise ValueError(f"{task_dir.name}: incomplete joint row at raw line {line_no}: {line.strip()}")
        rows.append(parsed)

    with clean_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    print(f"[CSV] {task_dir.name}: wrote states_clean.csv rows={len(rows)} incomplete={incomplete_rows}")
    return len(rows), incomplete_rows


def validate_frame_files(
    result: ValidationResult,
    task_dir: Path,
    subdir: str,
    suffix: str,
    required: bool = True,
) -> set[int]:
    directory = task_dir / subdir
    if not directory.exists():
        if required:
            result.error(f"missing {subdir}/")
        return set()

    frame_ids: dict[int, Path] = {}
    for path in natural_sort(directory.glob(f"*{suffix}")):
        try:
            frame_id = frame_number_from_path(path)
        except ValueError as exc:
            result.error(str(exc))
            continue
        if frame_id in frame_ids:
            result.error(f"duplicate frame id {frame_id} in {subdir}/")
        frame_ids[frame_id] = path
    return set(frame_ids)


def validate_task(task_dir: Path, future_steps: int, sample_images: bool = True) -> ValidationResult:
    result = ValidationResult(task_name=task_dir.name)
    csv_path = task_dir / "states_filtered.csv"
    four_dir = task_dir / "four_channel"

    if not csv_path.exists():
        result.error("missing states_filtered.csv")
        return result
    if not four_dir.exists():
        result.error("missing four_channel/")
        return result

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        result.error(f"failed to read states_filtered.csv: {exc}")
        return result

    missing_cols = [col for col in CSV_HEADER if col not in df.columns]
    if missing_cols:
        result.error(f"states_filtered.csv missing columns: {missing_cols}")
        return result

    if len(df) <= future_steps:
        result.error(f"too few frames for FUTURE_STEPS={future_steps}: {len(df)}")

    frames = pd.to_numeric(df["frame"], errors="coerce")
    if frames.isnull().any():
        result.error("frame column contains non-numeric values")
    else:
        frame_list = frames.astype(int).tolist()
        expected = list(range(len(frame_list)))
        if frame_list != expected:
            result.error("frame column must be continuous 0..N-1 after preprocessing")

    joints = df[JOINT_COLS].apply(pd.to_numeric, errors="coerce")
    if joints.isnull().any().any():
        bad_rows = joints[joints.isnull().any(axis=1)].index.tolist()[:10]
        result.error(f"joint columns contain NaN/non-numeric rows: {bad_rows}")
    else:
        values = joints.to_numpy(dtype=np.float32)
        below = values < FIXED_JOINT_MIN.reshape(1, -1)
        above = values > FIXED_JOINT_MAX.reshape(1, -1)
        if below.any() or above.any():
            bad_cols = sorted({JOINT_COLS[idx] for idx in np.where(below | above)[1]})
            result.exclude(f"joint values outside fixed physical range in columns: {bad_cols}")

    four_frames = validate_frame_files(result, task_dir, "four_channel", ".png")
    csv_frames = set(range(len(df)))
    if four_frames != csv_frames:
        result.error(
            "four_channel frames do not match states_filtered frame column: "
            f"missing_images={sorted(csv_frames - four_frames)[:10]} "
            f"extra_images={sorted(four_frames - csv_frames)[:10]}"
        )

    rgb_frames = validate_frame_files(result, task_dir, "rgb", ".jpg", required=False)
    depth_frames = validate_frame_files(result, task_dir, "depth", ".png", required=False)
    depth_norm_frames = validate_frame_files(result, task_dir, "depth_normalized", ".png", required=False)
    if rgb_frames and rgb_frames != csv_frames:
        result.warn("rgb frames do not match states_filtered frame column")
    if depth_frames and depth_frames != csv_frames:
        result.warn("depth frames do not match states_filtered frame column")
    if depth_norm_frames and depth_norm_frames != csv_frames:
        result.warn("depth_normalized frames do not match states_filtered frame column")

    if sample_images and four_frames:
        sample_ids = sorted(four_frames)
        sample_ids = sorted(set([sample_ids[0], sample_ids[len(sample_ids) // 2], sample_ids[-1]]))
        for frame_id in sample_ids:
            path = four_dir / f"{frame_id:06d}.png"
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                result.error(f"failed to read {path.name}")
                continue
            if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
                result.error(f"{path.name} expected shape={IMAGE_SHAPE} uint8, got shape={image.shape} dtype={image.dtype}")
            elif image[:, :, 3].max() == 0:
                result.warn(f"{path.name} depth channel is all zero")

    return result


def _gripper_source_csv(task_dir: Path, refresh_backup: bool) -> tuple[Path, Path]:
    csv_path = task_dir / "states_filtered.csv"
    backup_path = task_dir / GRIPPER_BACKUP_NAME
    if refresh_backup or not backup_path.exists():
        shutil.copy2(csv_path, backup_path)
    return csv_path, backup_path


def amplify_gripper_trajectories(
    task_dirs: list[Path],
    scale: float,
    refresh_backup_tasks: set[Path] | None = None,
) -> list[GripperAmplifyResult]:
    """Amplify j10 motion around the global dataset mean without compounding.

    `states_filtered.pre_gripper_amp.csv` is the stable source. The training
    CSV is rewritten from that source on every run, so running the preparation
    command repeatedly does not multiply the gripper range again and again.
    """

    if scale <= 0:
        raise ValueError(f"gripper scale must be positive, got {scale}")

    refresh_backup_tasks = refresh_backup_tasks or set()
    gripper_index = JOINT_COLS.index(GRIPPER_COL)
    lower = float(FIXED_JOINT_MIN[gripper_index])
    upper = float(FIXED_JOINT_MAX[gripper_index])
    loaded: list[tuple[Path, Path, Path, pd.DataFrame, np.ndarray]] = []
    results: list[GripperAmplifyResult] = []

    for task_dir in task_dirs:
        csv_path = task_dir / "states_filtered.csv"
        if not csv_path.exists():
            results.append(
                GripperAmplifyResult(
                    task_name=task_dir.name,
                    ok=False,
                    skipped=True,
                    message="missing states_filtered.csv",
                )
            )
            continue

        try:
            csv_path, backup_path = _gripper_source_csv(task_dir, refresh_backup=task_dir in refresh_backup_tasks)
            df = pd.read_csv(backup_path)
            if GRIPPER_COL not in df.columns:
                raise ValueError(f"missing {GRIPPER_COL} column")
            values = pd.to_numeric(df[GRIPPER_COL], errors="coerce")
            if values.isnull().any():
                bad_rows = values[values.isnull()].index.tolist()[:10]
                raise ValueError(f"{GRIPPER_COL} contains NaN/non-numeric rows: {bad_rows}")
            loaded.append((task_dir, csv_path, backup_path, df, values.to_numpy(dtype=np.float32)))
        except Exception as exc:
            results.append(GripperAmplifyResult(task_name=task_dir.name, ok=False, message=str(exc)))

    if not loaded:
        return results

    all_before = np.concatenate([values for *_prefix, values in loaded])
    center = float(all_before.mean())
    all_after_parts: list[np.ndarray] = []
    total_low = 0
    total_high = 0

    print(
        "[GRIPPER] global before "
        f"min={all_before.min():.3f} max={all_before.max():.3f} mean={center:.3f}; "
        f"scale={scale:.3f}; physical=[{lower:.0f}, {upper:.0f}]"
    )

    for task_dir, csv_path, backup_path, df, before in loaded:
        expanded = center + scale * (before - center)
        clipped_low = int((expanded < lower).sum())
        clipped_high = int((expanded > upper).sum())
        after = np.clip(expanded, lower, upper)
        after_int = np.rint(after).astype(int)

        out_df = df.copy()
        out_df[GRIPPER_COL] = after_int
        out_df.to_csv(csv_path, index=False)

        after_float = after_int.astype(np.float32)
        all_after_parts.append(after_float)
        total_low += clipped_low
        total_high += clipped_high

        meta = {
            "source_csv": backup_path.name,
            "target_csv": csv_path.name,
            "joint": GRIPPER_COL,
            "scale": scale,
            "center_mean": center,
            "physical_min": lower,
            "physical_max": upper,
            "before": {
                "min": float(before.min()),
                "max": float(before.max()),
                "mean": float(before.mean()),
            },
            "after": {
                "min": float(after_float.min()),
                "max": float(after_float.max()),
                "mean": float(after_float.mean()),
            },
            "clipped_low": clipped_low,
            "clipped_high": clipped_high,
            "rows": int(len(before)),
        }
        with (task_dir / GRIPPER_META_NAME).open("w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2, ensure_ascii=False)

        results.append(
            GripperAmplifyResult(
                task_name=task_dir.name,
                ok=True,
                rows=len(before),
                before_min=float(before.min()),
                before_max=float(before.max()),
                before_mean=float(before.mean()),
                after_min=float(after_float.min()),
                after_max=float(after_float.max()),
                after_mean=float(after_float.mean()),
                clipped_low=clipped_low,
                clipped_high=clipped_high,
            )
        )

    all_after = np.concatenate(all_after_parts)
    print(
        "[GRIPPER] global after  "
        f"min={all_after.min():.3f} max={all_after.max():.3f} mean={all_after.mean():.3f}; "
        f"clipped_low={total_low} clipped_high={total_high}"
    )

    return results


def run_prepare(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).resolve()
    tasks = discover_tasks(data_root)
    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] found {len(tasks)} task directories")

    if not args.validate_only:
        raw_tasks = [task_dir for task_dir in tasks if has_raw_inputs(task_dir)]
        final_only_tasks = [task_dir for task_dir in tasks if not has_raw_inputs(task_dir)]
        if final_only_tasks:
            print(
                "[INFO] final-only task directories will be validated but not regenerated: "
                f"{len(final_only_tasks)}"
            )

        print("\n[STEP 1/5] Clean states.csv -> states_clean.csv")
        total_incomplete = 0
        for task_dir in raw_tasks:
            _, incomplete = clean_states_csv(task_dir, strict=args.strict_csv)
            total_incomplete += incomplete
        print(f"[CSV] total incomplete raw rows={total_incomplete}")

        print("\n[STEP 2/5] Sync/filter trajectories and raw images")
        for task_dir in raw_tasks:
            data_process_1.process_single_task(str(task_dir))

        print("\n[STEP 3/5] Rebuild four_channel images")
        for task_dir in raw_tasks:
            data_process_2.process_single_task(str(task_dir))

        print("\n[STEP 4/5] Amplify gripper trajectory")
        gripper_results = amplify_gripper_trajectories(
            tasks,
            scale=GRIPPER_SCALE,
            refresh_backup_tasks=set(raw_tasks),
        )
        ok_gripper = sum(1 for result in gripper_results if result.ok)
        failed_gripper = [result for result in gripper_results if not result.ok]
        print(f"[GRIPPER] updated={ok_gripper}/{len(gripper_results)}")
        for result in gripper_results:
            if result.ok:
                print(
                    f"[GRIPPER] {result.task_name}: "
                    f"{result.before_min:.1f}-{result.before_max:.1f} mean={result.before_mean:.1f} -> "
                    f"{result.after_min:.1f}-{result.after_max:.1f} mean={result.after_mean:.1f} "
                    f"clip=({result.clipped_low},{result.clipped_high})"
                )
            else:
                print(f"[GRIPPER][ERROR] {result.task_name}: {result.message}")
        if failed_gripper:
            print("[FAIL] Gripper amplification failed. Fix data before training.")
            return 1

    print("\n[STEP 5/5] Validate final ACT training data")
    results = [validate_task(task_dir, future_steps=args.future_steps, sample_images=not args.fast_validate) for task_dir in tasks]

    excluded = {
        result.task_name: result.exclude_reason
        for result in results
        if result.exclude_reason
    }
    if excluded:
        path = write_excluded_tasks(data_root, excluded)
        print(f"[WARN] excluded {len(excluded)} task(s) via {EXCLUSION_FILENAME}: {path}")
    else:
        path = write_excluded_tasks(data_root, {})
        print(f"[INFO] no excluded tasks; wrote empty {EXCLUSION_FILENAME}: {path}")

    ok_count = sum(1 for result in results if result.ok)
    excluded_count = len(excluded)
    warn_count = sum(len(result.warnings) for result in results)
    print(f"[VALIDATE] ok={ok_count}/{len(results)} excluded={excluded_count} warnings={warn_count}")

    for result in results:
        for warning in result.warnings:
            print(f"[WARN] {result.task_name}: {warning}")
        for error in result.errors:
            print(f"[ERROR] {result.task_name}: {error}")

    failed = [result for result in results if result.errors]
    if failed:
        print(f"[FAIL] {len(failed)} task(s) failed validation. Fix data before training.")
        return 1

    if excluded:
        print("[OK] Dataset is ready for ACT training; excluded tasks will be ignored by dataloaders.")
    else:
        print("[OK] Dataset is ready for ACT training.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate ACT task data in one command.")
    parser.add_argument("--data-root", default=str(PROJECT_DIR / "data_process" / "data"), help="Directory containing task_* folders.")
    parser.add_argument("--future-steps", type=int, default=10, help="Minimum trajectory length check, should match Config.FUTURE_STEPS.")
    parser.add_argument("--strict-csv", action="store_true", help="Fail immediately on incomplete raw states.csv rows.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing states_filtered/four_channel outputs.")
    parser.add_argument("--fast-validate", action="store_true", help="Skip sample image shape/depth checks.")
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run_prepare(parse_args()))


if __name__ == "__main__":
    main()
