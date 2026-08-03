# coding: utf-8
"""On-disk manifest that lets a crashed training run finish its artifacts upload.

Written to ``work_dir`` (not ``output_dir``, so it is not uploaded) right before the upload
starts. A failed upload exits non-zero, the agent keeps the task data dir on the host, and a
relaunch of the same task finds the manifest and offers to resume instead of retraining.

Every function is best-effort: a write failure only disables resume; an unreadable manifest,
an unknown version or another task id is treated as absent.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import supervisely.io.fs as sly_fs
from supervisely.sly_logger import logger

MANIFEST_NAME = ".resume_upload.json"
VERSION = 1


def path(work_dir: str) -> str:
    """Path of the manifest inside the given work dir."""
    return os.path.join(work_dir, MANIFEST_NAME)


def save(work_dir: str, task_id: int, **fields: Any) -> bool:
    """Write the manifest, replacing any previous one. Returns False on failure."""
    manifest = {
        "version": VERSION,
        "task_id": task_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "attempts": 0,
        "finalize_done": [],
    }
    manifest.update(fields)
    return _write(work_dir, manifest)


def load(work_dir: str, task_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Read the manifest, or None if absent, unreadable, of another version or another task."""
    manifest_path = path(work_dir)
    if not sly_fs.file_exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except Exception:
        logger.warning(f"Resume manifest {manifest_path} is unreadable, ignoring", exc_info=True)
        return None
    if not isinstance(manifest, dict):
        logger.warning(f"Resume manifest {manifest_path} has unexpected content, ignoring")
        return None
    if manifest.get("version") != VERSION:
        logger.warning(
            f"Resume manifest version {manifest.get('version')} != {VERSION}, ignoring"
        )
        return None
    if task_id is not None and manifest.get("task_id") != task_id:
        logger.info(
            f"Resume manifest belongs to task {manifest.get('task_id')}, current is {task_id}"
        )
        return None
    return manifest


def update(work_dir: str, **fields: Any) -> Optional[Dict[str, Any]]:
    """Merge the given fields into the existing manifest. Returns the new manifest."""
    manifest = load(work_dir)
    if manifest is None:
        return None
    manifest.update(fields)
    _write(work_dir, manifest)
    return manifest


def mark_done(work_dir: str, step: str) -> None:
    """Mark a finalize step done, so a later resume does not repeat it."""
    manifest = load(work_dir)
    if manifest is None:
        return
    done: List[str] = manifest.get("finalize_done", [])
    if step in done:
        return
    done.append(step)
    manifest["finalize_done"] = done
    _write(work_dir, manifest)


def is_done(manifest: Optional[Dict[str, Any]], step: str) -> bool:
    """True if the step was already completed by a previous attempt."""
    if not manifest:
        return False
    return step in manifest.get("finalize_done", [])


def bump_attempt(work_dir: str) -> int:
    """Increment and return the resume attempt counter."""
    manifest = load(work_dir)
    if manifest is None:
        return 0
    attempts = int(manifest.get("attempts", 0)) + 1
    manifest["attempts"] = attempts
    _write(work_dir, manifest)
    return attempts


def remove(work_dir: str) -> None:
    """Delete the manifest: the run is complete."""
    sly_fs.silent_remove(path(work_dir))


def _write(work_dir: str, manifest: Dict[str, Any]) -> bool:
    manifest_path = path(work_dir)
    tmp_path = manifest_path + ".tmp"
    try:
        sly_fs.mkdir(work_dir)
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, manifest_path)
        return True
    except Exception:
        logger.warning(
            f"Failed to write {manifest_path}, resuming the upload will not be possible",
            exc_info=True,
        )
        sly_fs.silent_remove(tmp_path)
        return False
