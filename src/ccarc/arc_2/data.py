"""Dataset path and task-resolution helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path


SPLIT_TO_DIR = {
    "public_eval": "evaluation",
    "evaluation": "evaluation",
    "eval": "evaluation",
    "public_train": "training",
    "training": "training",
    "train": "training",
}


def _looks_like_dataset_root(path: Path) -> bool:
    """Check whether a path appears to be an ARC dataset root."""
    return (
        (path / "data" / "evaluation").is_dir()
        or (path / "data" / "training").is_dir()
        or (path / "evaluation").is_dir()
        or (path / "training").is_dir()
    )


def _auto_detect_dataset_root() -> Path | None:
    """Best-effort dataset root discovery for local dev workflows."""
    package_anchor = Path(__file__).resolve().parents[3]  # repo root
    candidates = [
        Path.cwd(),
        Path.cwd() / "ARC-AGI-2",
        Path.cwd().parent / "ARC-AGI-2",
        package_anchor / "ARC-AGI-2",
        package_anchor.parent / "ARC-AGI-2",
    ]

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_dataset_root(candidate):
            return candidate.resolve()
    return None


def resolve_dataset_root(dataset_root: str | Path | None) -> Path:
    """Resolve dataset root from CLI arg or ARC_DATA_ROOT environment variable."""
    raw = str(dataset_root).strip() if dataset_root is not None else ""
    if not raw:
        raw = str(os.getenv("ARC_DATA_ROOT") or "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        return root

    detected = _auto_detect_dataset_root()
    if detected is not None:
        return detected

    raise ValueError(
        "Dataset root not provided. Set ARC_DATA_ROOT, pass --dataset-root, "
        "or run from a workspace containing ARC-AGI-2."
    )


def map_split_to_dir(split: str) -> str:
    """Map logical split name to on-disk directory name."""
    key = str(split or "public_eval").strip().lower()
    if key not in SPLIT_TO_DIR:
        known = ", ".join(sorted(SPLIT_TO_DIR))
        raise ValueError(f"Unknown split '{split}'. Expected one of: {known}")
    return SPLIT_TO_DIR[key]


def resolve_task_path(
    task: str,
    split: str = "public_eval",
    dataset_root: str | Path | None = None,
) -> Path:
    """Resolve a task id or path into a concrete JSON file path."""
    raw = str(task or "").strip()
    if not raw:
        raise ValueError("Task/path is empty")

    direct = Path(raw).expanduser()
    if direct.exists() and direct.is_file():
        return direct.resolve()

    task_file = raw if raw.endswith(".json") else f"{raw}.json"
    root = resolve_dataset_root(dataset_root)
    split_dir = map_split_to_dir(split)

    candidates = [
        root / "data" / split_dir / task_file,
        root / split_dir / task_file,
        root / task_file,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Task JSON not found. Checked: " + ", ".join(str(p) for p in candidates)
    )


def list_tasks(split: str = "public_eval", dataset_root: str | Path | None = None) -> list[str]:
    """List task IDs available for a split."""
    root = resolve_dataset_root(dataset_root)
    split_dir = map_split_to_dir(split)
    directories = [root / "data" / split_dir, root / split_dir]
    task_dir = next((d for d in directories if d.exists() and d.is_dir()), None)
    if task_dir is None:
        return []
    return sorted(path.stem for path in task_dir.glob("*.json"))


def load_task_json(path: str | Path) -> dict:
    """Load a task JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
