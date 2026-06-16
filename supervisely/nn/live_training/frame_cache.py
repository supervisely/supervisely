"""Disk-backed video-frame cache for the live-training endpoints.

Video endpoints (``/predict-video``, ``/add-sample-video``,
``/highlight_key_frames`` ...) repeatedly download the same frames from the
Supervisely server. ``VideoFrameCache`` stores decoded frames on disk as
lossless PNGs so each frame is fetched at most once, with a background prefetch
kicked off on the first ``/add-sample-video`` request.

Prefetch sizing is disk-aware: if the whole video comfortably fits
(``frames_count * W * H * 3`` < 30% of free disk) every frame is downloaded and
kept until the app stops (FULL mode); otherwise a 50-frames-forward sliding
window is maintained around the frame the user is on (WINDOW mode).

Retention has two tiers:
  * frames that entered the training dataset are **pinned** (kept until stop);
  * everything else is **transient** and a janitor thread evicts it 10 minutes
    after it was cached (only relevant in WINDOW mode — FULL mode has the disk
    room to keep everything).

The cache directory is a ``tempfile`` dir removed on ``cleanup()`` (called from
the app's shutdown path).
"""

import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import supervisely as sly
from supervisely import logger

DEFAULT_TTL_SECONDS = 600  # transient frames live 10 minutes
DEFAULT_WINDOW_AHEAD = 50  # sliding-window look-ahead (frames)
DEFAULT_FREE_FRACTION = 0.30  # FULL-prefetch budget = 30% of free disk
DOWNLOAD_BATCH_SIZE = 50
JANITOR_INTERVAL = 60  # seconds between eviction sweeps

MODE_FULL = "full"
MODE_WINDOW = "window"


@dataclass
class _Entry:
    path: str
    expires_at: Optional[float]  # None => keep until stop (pinned / FULL mode)


class VideoFrameCache:
    """Thread-safe, disk-backed cache of video frames keyed by
    ``(video_id, frame_index)``."""

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        window_ahead: int = DEFAULT_WINDOW_AHEAD,
        free_fraction: float = DEFAULT_FREE_FRACTION,
    ):
        self._dir = tempfile.mkdtemp(prefix="lt_frame_cache_")
        self._ttl = ttl_seconds
        self._window_ahead = window_ahead
        self._free_fraction = free_fraction

        self._lock = threading.RLock()
        self._entries: Dict[Tuple[int, int], _Entry] = {}
        self._modes: Dict[int, str] = {}
        self._frames_count: Dict[int, int] = {}
        self._prefetch_started: Set[int] = set()
        self._cancel = threading.Event()

        self._janitor = threading.Thread(
            target=self._janitor_loop, daemon=True, name="FrameCacheJanitor"
        )
        self._janitor.start()
        logger.info(f"[frame-cache] initialized at {self._dir}")

    # ------------------------------------------------------------------ paths
    def _frame_path(self, video_id: int, idx: int) -> str:
        return os.path.join(self._dir, str(video_id), f"{idx}.png")

    # ------------------------------------------------------------------ reads
    def get_frame(self, video_id: int, idx: int, api: sly.Api, pin: bool = False) -> np.ndarray:
        return self.get_frames(video_id, [idx], api, pin=pin)[0]

    def get_frames(
        self, video_id: int, indices: List[int], api: sly.Api, pin: bool = False
    ) -> List[np.ndarray]:
        """Return frames for ``indices`` (in order). Hits are read from disk,
        misses are downloaded in one batch and stored. ``pin=True`` keeps the
        frames until stop (used for training-dataset frames)."""
        indices = list(indices)
        unique = list(dict.fromkeys(indices))  # de-dupe, keep order
        keep_forever = pin or self._modes.get(video_id) == MODE_FULL

        missing = []
        with self._lock:
            for idx in unique:
                entry = self._entries.get((video_id, idx))
                if entry is None or not os.path.exists(entry.path):
                    missing.append(idx)

        downloaded: Dict[int, np.ndarray] = {}
        if missing:
            arrays = api.video.frame.download_nps(video_id, missing)
            for idx, arr in zip(missing, arrays):
                self._store(video_id, idx, arr, keep_forever)
                downloaded[idx] = arr

        # Pinning an already-cached (possibly transient) entry promotes it.
        if pin:
            self.pin(video_id, unique)

        by_idx: Dict[int, np.ndarray] = {}
        for idx in unique:
            by_idx[idx] = downloaded[idx] if idx in downloaded else self._read(
                video_id, idx, api, keep_forever
            )
        return [by_idx[idx] for idx in indices]

    def pin(self, video_id: int, indices: List[int]) -> None:
        """Mark cached frames as keep-until-stop."""
        with self._lock:
            for idx in indices:
                entry = self._entries.get((video_id, idx))
                if entry is not None:
                    entry.expires_at = None

    # ------------------------------------------------------------- prefetching
    def start_prefetch(self, video_id: int, api: sly.Api, current_index: int = 0) -> None:
        """Start background prefetch for ``video_id`` (once per video). Picks
        FULL or WINDOW mode based on free disk, then downloads accordingly."""
        with self._lock:
            if video_id in self._prefetch_started:
                return
            self._prefetch_started.add(video_id)
        threading.Thread(
            target=self._prefetch_worker,
            args=(video_id, api, current_index),
            daemon=True,
            name=f"FrameCachePrefetch-{video_id}",
        ).start()

    def ensure_window(self, video_id: int, current_index: int, api: sly.Api) -> None:
        """In WINDOW mode, (re)fill the 50-frames-forward window around
        ``current_index`` in the background. No-op otherwise."""
        if self._modes.get(video_id) != MODE_WINDOW:
            return
        threading.Thread(
            target=self._ensure_window_blocking,
            args=(video_id, current_index, api),
            daemon=True,
            name=f"FrameCacheWindow-{video_id}",
        ).start()

    def cleanup(self) -> None:
        """Stop background work and remove the cache directory. Idempotent."""
        if self._cancel.is_set():
            return
        self._cancel.set()
        try:
            if os.path.exists(self._dir):
                sly.fs.remove_dir(self._dir)
            logger.info(f"[frame-cache] removed cache dir {self._dir}")
        except Exception as e:
            logger.warning(f"[frame-cache] failed to remove {self._dir}: {e}")

    # --------------------------------------------------------------- internals
    def _store(self, video_id: int, idx: int, arr: np.ndarray, keep_forever: bool) -> None:
        path = self._frame_path(video_id, idx)
        sly.fs.ensure_base_path(path)
        # Keep a ``.png`` suffix so sly.image.write accepts the extension.
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp.png"
        try:
            sly.image.write(tmp, arr)
            os.replace(tmp, path)  # atomic — readers never see a partial file
        except Exception:
            sly.fs.silent_remove(tmp)
            raise
        expires_at = None if keep_forever else time.monotonic() + self._ttl
        with self._lock:
            existing = self._entries.get((video_id, idx))
            # Never shorten the lifetime of an already keep-forever entry.
            if existing is not None and existing.expires_at is None:
                expires_at = None
            self._entries[(video_id, idx)] = _Entry(path=path, expires_at=expires_at)

    def _read(self, video_id: int, idx: int, api: sly.Api, keep_forever: bool) -> np.ndarray:
        path = self._frame_path(video_id, idx)
        try:
            return sly.image.read(path)
        except Exception:
            # File evicted/corrupt between the hit check and the read — refetch.
            arr = api.video.frame.download_nps(video_id, [idx])[0]
            self._store(video_id, idx, arr, keep_forever)
            return arr

    def _prefetch_worker(self, video_id: int, api: sly.Api, current_index: int) -> None:
        try:
            info = api.video.get_info_by_id(video_id)
            frames_count = info.frames_count
            with self._lock:
                self._frames_count[video_id] = frames_count

            est = info.frame_width * info.frame_height * 3 * frames_count
            free = shutil.disk_usage(self._dir).free
            budget = free * self._free_fraction
            mode = MODE_FULL if est < budget else MODE_WINDOW
            with self._lock:
                self._modes[video_id] = mode
            logger.info(
                f"[frame-cache] video {video_id}: {frames_count} frames, "
                f"~{est / 1e9:.2f}GB est vs {budget / 1e9:.2f}GB budget "
                f"({free / 1e9:.2f}GB free) -> {mode} mode"
            )

            if mode == MODE_FULL:
                self._download_range(
                    video_id, list(range(frames_count)), api, keep_forever=True
                )
            else:
                self._ensure_window_blocking(video_id, current_index, api)
        except Exception as e:
            logger.warning(f"[frame-cache] prefetch for video {video_id} failed: {e}")

    def _ensure_window_blocking(self, video_id: int, current_index: int, api: sly.Api) -> None:
        try:
            frames_count = self._frames_count.get(video_id)
            if frames_count is None:
                frames_count = api.video.get_info_by_id(video_id).frames_count
                with self._lock:
                    self._frames_count[video_id] = frames_count
            start = max(0, current_index)
            end = min(current_index + self._window_ahead + 1, frames_count)
            self._download_range(video_id, list(range(start, end)), api, keep_forever=False)
        except Exception as e:
            logger.warning(f"[frame-cache] window prefetch for video {video_id} failed: {e}")

    def _download_range(
        self, video_id: int, indices: List[int], api: sly.Api, keep_forever: bool
    ) -> None:
        for batch in sly.batched(indices, batch_size=DOWNLOAD_BATCH_SIZE):
            if self._cancel.is_set():
                return
            need = []
            with self._lock:
                for idx in batch:
                    entry = self._entries.get((video_id, idx))
                    if entry is None or not os.path.exists(entry.path):
                        need.append(idx)
            if not need:
                continue
            try:
                arrays = api.video.frame.download_nps(video_id, need)
            except Exception as e:
                logger.warning(
                    f"[frame-cache] batch download failed for video {video_id}: {e}"
                )
                continue
            for idx, arr in zip(need, arrays):
                if self._cancel.is_set():
                    return
                self._store(video_id, idx, arr, keep_forever)

    def _janitor_loop(self) -> None:
        while not self._cancel.is_set():
            self._cancel.wait(JANITOR_INTERVAL)
            if self._cancel.is_set():
                return
            now = time.monotonic()
            expired: List[Tuple[Tuple[int, int], str]] = []
            with self._lock:
                for key, entry in list(self._entries.items()):
                    if entry.expires_at is not None and entry.expires_at <= now:
                        expired.append((key, entry.path))
                        del self._entries[key]
            for _key, path in expired:
                sly.fs.silent_remove(path)
            if expired:
                logger.debug(f"[frame-cache] evicted {len(expired)} expired frames")
