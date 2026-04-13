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
