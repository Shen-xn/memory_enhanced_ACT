"""Shared task exclusion manifest used by preprocessing and training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


EXCLUSION_FILENAME = "excluded_tasks.json"


def exclusion_path(data_root: str | Path) -> Path:
    return Path(data_root) / EXCLUSION_FILENAME


def load_excluded_tasks(data_root: str | Path) -> dict[str, str]:
    path = exclusion_path(data_root)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    tasks = payload.get("tasks", [])
    exclusions: dict[str, str] = {}
    for item in tasks:
        task_name = str(item.get("task", "")).strip()
        if not task_name:
            continue
        exclusions[task_name] = str(item.get("reason", "excluded by preprocessing"))
    return exclusions


def exclusion_reason_for_task(task_name: str, exclusions: Mapping[str, str]) -> str | None:
    """Return the exclusion reason for a task or its obstacle counterpart."""
    if task_name in exclusions:
        return exclusions[task_name]
    if task_name.startswith("task_obst_"):
        source_name = "task_" + task_name[len("task_obst_") :]
        return exclusions.get(source_name)
    return None


def is_task_excluded(task_name: str, exclusions: Mapping[str, str]) -> bool:
    """Whether a task should be ignored, including obstacle variants of excluded sources."""
    return exclusion_reason_for_task(task_name, exclusions) is not None


def write_excluded_tasks(data_root: str | Path, exclusions: Mapping[str, str]) -> Path:
    path = exclusion_path(data_root)
    payload = {
        "version": 1,
        "tasks": [
            {"task": task_name, "reason": reason}
            for task_name, reason in sorted(exclusions.items())
        ],
    }
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    return path
