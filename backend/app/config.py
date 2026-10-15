"""Runtime configuration.

All values are overridable through environment variables (prefix ``ANATO3D_``)
or a local ``.env`` file.  Nothing here should contain hard-coded absolute
paths -- everything is derived from ``ANATO3D_DATA_DIR``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANATO3D_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- general -----------------------------------------------------------
    app_name: str = "ANATO3D"
    app_version: str = "0.1.0"
    debug: bool = False

    # -- storage -----------------------------------------------------------
    # Root directory for all temporary processing data.  Everything written by
    # the application lives under here so it can be wiped in one operation.
    data_dir: Path = Path("_work")
    # Workspaces that have not been touched for this many minutes are removed
    # by the background reaper.
    workspace_ttl_minutes: int = 240
    # Reaper interval in seconds.
    reaper_interval_seconds: int = 300

    # -- upload limits -----------------------------------------------------
    # Maximum size of a single uploaded file (bytes).
    max_file_bytes: int = 512 * 1024 * 1024
    # Maximum total size of one upload request (bytes).
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    # Maximum number of files accepted in a single upload request.
    max_upload_files: int = 4000

    # -- processing limits -------------------------------------------------
    # If a reconstructed volume exceeds this many voxels we downsample the
    # in-plane resolution before segmentation so the prototype stays usable.
    max_voxels_soft: int = 220_000_000
    # Absolute ceiling.  Above this we refuse rather than thrash.
    max_voxels_hard: int = 900_000_000

    # Default working resolution cap along the slice axis (0 = no cap).
    max_slices: int = 0

    # -- CORS --------------------------------------------------------------
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    @property
    def work_root(self) -> Path:
        return Path(self.data_dir).resolve()

    def ensure_dirs(self) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
