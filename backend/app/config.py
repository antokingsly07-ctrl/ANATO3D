"""Runtime configuration.

All values are overridable through environment variables (prefix ``ANATO3D_``)
or a local ``.env`` file.  Nothing here should contain hard-coded absolute
paths -- everything is derived from ``ANATO3D_DATA_DIR``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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
    # Accepted formats for ANATO3D_CORS_ORIGINS:
    #   - JSON array:        ["https://app.vercel.app","https://x.example"]
    #   - comma-separated:   https://app.vercel.app,https://x.example
    #   - a single origin:   https://app.vercel.app
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        raw = value.strip()
        # Tolerate a value pasted with surrounding quotes (e.g. from a shell example).
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            raw = raw[1:-1].strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in raw.split(",") if item.strip()]

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
