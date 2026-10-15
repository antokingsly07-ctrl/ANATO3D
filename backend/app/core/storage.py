"""In-memory artifact store + temporary workspace management.

Design notes
------------
* All processed binary data lives on disk under a per-dataset directory inside
  ``ANATO3D_DATA_DIR``.  Only lightweight handles (ids, numpy arrays for small
  masks) are kept in memory.
* Segmentation masks are kept in memory as boolean arrays because they are the
  working set for mesh generation and are cheap to regenerate if evicted; large
  source volumes are *not* cached in RAM (they are re-read from the stored
  NumPy file when needed) to keep memory bounded on laptop-class machines.
* A background reaper deletes workspaces that have not been touched within
  ``workspace_ttl_minutes`` -- nothing is retained beyond the session.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from .logging_ import get_logger

logger = get_logger("storage")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class Workspace:
    """Filesystem sandbox for one uploaded dataset."""

    dataset_id: str
    root: Path
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    @property
    def uploads_dir(self) -> Path:
        return self.root / "uploads"

    @property
    def derived_dir(self) -> Path:
        return self.root / "derived"

    def touch(self) -> None:
        self.last_access = time.time()

    def initialize(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.derived_dir.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def delete(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class AppStore:
    """Process-wide registry of all artefacts produced during a session."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workspaces: dict[str, Workspace] = {}
        self.datasets: dict[str, Any] = {}
        self.volumes: dict[str, Any] = {}
        self.segmentations: dict[str, Any] = {}
        self.meshes: dict[str, Any] = {}
        self.validations: dict[str, Any] = {}
        self.analyses: dict[str, Any] = {}
        #: ground-truth masks uploaded for validation, keyed by dataset id
        self.ground_truth: dict[str, Any] = {}

    # -- workspaces --------------------------------------------------------
    def create_workspace(self) -> Workspace:
        settings = get_settings()
        dataset_id = new_id("ds")
        root = settings.work_root / "datasets" / dataset_id
        ws = Workspace(dataset_id=dataset_id, root=root)
        ws.initialize()
        with self._lock:
            self._workspaces[dataset_id] = ws
        logger.info("Created workspace %s", dataset_id)
        return ws

    def workspace(self, dataset_id: str) -> Workspace:
        with self._lock:
            ws = self._workspaces.get(dataset_id)
        if ws is None:
            from .errors import NotFoundError

            raise NotFoundError(
                "This dataset is no longer available. It may have expired -- please upload again."
            )
        ws.touch()
        return ws

    def touch(self, dataset_id: str) -> None:
        with self._lock:
            ws = self._workspaces.get(dataset_id)
        if ws:
            ws.touch()

    # -- artefact helpers --------------------------------------------------
    def put(self, bucket: str, key: str, value: Any) -> Any:
        with self._lock:
            getattr(self, bucket)[key] = value
        return value

    def get(self, bucket: str, key: str) -> Any:
        with self._lock:
            value = getattr(self, bucket).get(key)
        if value is None:
            from .errors import NotFoundError

            raise NotFoundError(f"{bucket[:-1].capitalize()} '{key}' was not found.")
        return value

    def find(self, bucket: str, key: str) -> Any:
        with self._lock:
            return getattr(self, bucket).get(key)

    def drop(self, bucket: str, key: str) -> None:
        with self._lock:
            getattr(self, bucket).pop(key, None)

    # -- volume persistence ------------------------------------------------
    def volume_path(self, dataset_id: str, volume_id: str, suffix: str = ".npy") -> Path:
        ws = self.workspace(dataset_id)
        path = ws.derived_dir / "volumes"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{volume_id}{suffix}"

    def save_volume_array(self, dataset_id: str, volume_id: str, array: np.ndarray) -> Path:
        path = self.volume_path(dataset_id, volume_id)
        np.save(path, array, allow_pickle=False)
        return path

    def load_volume_array(self, dataset_id: str, volume_id: str) -> np.ndarray:
        path = self.volume_path(dataset_id, volume_id)
        if not path.exists():
            from .errors import NotFoundError

            raise NotFoundError("The reconstructed volume data is no longer available.")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def save_mask(self, dataset_id: str, segmentation_id: str, mask: np.ndarray) -> Path:
        ws = self.workspace(dataset_id)
        path = ws.derived_dir / "masks"
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{segmentation_id}.npz"
        np.savez_compressed(target, mask=np.packbits(mask.reshape(-1)))
        return target

    def load_mask(self, dataset_id: str, segmentation_id: str, shape: tuple[int, int, int]) -> np.ndarray:
        ws = self.workspace(dataset_id)
        target = ws.derived_dir / "masks" / f"{segmentation_id}.npz"
        if not target.exists():
            from .errors import NotFoundError

            raise NotFoundError("The segmentation data is no longer available.")
        with np.load(target, allow_pickle=False) as data:
            packed = data["mask"]
        return np.unpackbits(packed)[: int(np.prod(shape))].astype(bool).reshape(shape)

    # -- maintenance -------------------------------------------------------
    def reap(self) -> int:
        settings = get_settings()
        deadline = time.time() - settings.workspace_ttl_minutes * 60
        removed = 0
        with self._lock:
            stale = [ws for ws in self._workspaces.values() if ws.last_access < deadline]
        for ws in stale:
            ws.delete()
            with self._lock:
                self._workspaces.pop(ws.dataset_id, None)
                for bucket in (
                    self.datasets,
                    self.volumes,
                    self.segmentations,
                    self.meshes,
                    self.validations,
                    self.analyses,
                    self.ground_truth,
                ):
                    for key in [k for k, v in bucket.items() if getattr(v, "dataset_id", None) == ws.dataset_id]:
                        bucket.pop(key, None)
            removed += 1
        return removed

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "workspaces": len(self._workspaces),
                "datasets": len(self.datasets),
                "volumes": len(self.volumes),
                "segmentations": len(self.segmentations),
                "meshes": len(self.meshes),
                "validations": len(self.validations),
                "analyses": len(self.analyses),
            }

    def reset(self) -> None:
        """Drop everything (used by tests)."""
        with self._lock:
            workspaces = list(self._workspaces.values())
            self._workspaces.clear()
            for bucket in (
                self.datasets,
                self.volumes,
                self.segmentations,
                self.meshes,
                self.validations,
                self.analyses,
                self.ground_truth,
            ):
                bucket.clear()
        for ws in workspaces:
            ws.delete()


store = AppStore()
