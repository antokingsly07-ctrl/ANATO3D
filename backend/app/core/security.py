"""Upload hygiene.

Defence in depth for a prototype that accepts arbitrary user files:

* filenames are sanitised and never used to build filesystem paths directly,
* only whitelisted extensions are accepted (harmless for DICOM which has no
  meaningful extension requirement, but blocks ``.exe``/``.sh``/... ),
* every stored file gets a generated opaque name inside an isolated
  per-dataset directory,
* uploads are size-limited before and during writing,
* uploaded content is *never* executed or imported.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from ..core.errors import UnsupportedFileError, ValidationError

# Files without an extension are extremely common in DICOM exports, so an empty
# suffix is allowed.  Extensions that are obviously executable/script are
# rejected outright.
BLOCKED_EXTENSIONS = {
    ".exe", ".dll", ".so", ".dylib", ".bat", ".cmd", ".com", ".msi", ".scr",
    ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse", ".jar", ".sh", ".bash",
    ".py", ".pyc", ".pyo", ".rb", ".pl", ".php", ".asp", ".aspx", ".jsp",
    ".hta", ".cpl", ".inf", ".reg", ".lnk", ".pif", ".gadget", ".wsf", ".wsh",
}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str, *, max_length: int = 120) -> str:
    """Return a filesystem-safe version of ``name``.

    The result is *only* used for display and for the ``original_name`` column;
    files are always written under generated names.
    """
    if not name:
        return "unnamed"
    # Strip any directory component the client may have sent (incl. Windows).
    name = name.replace("\\", "/").split("/")[-1]
    # Normalise unicode and drop control characters.
    name = unicodedata.normalize("NFKC", name)
    name = "".join(ch for ch in name if ch.isprintable())
    name = _SAFE_NAME.sub("_", name).strip("._")
    if not name:
        name = "unnamed"
    return name[:max_length]


def validate_upload_name(name: str) -> str:
    """Sanitise a filename and reject obviously unsafe extensions."""
    safe = sanitize_filename(name)
    suffix = Path(safe).suffix.lower()
    if suffix in BLOCKED_EXTENSIONS:
        raise UnsupportedFileError(
            f"File type '{suffix}' is not allowed.",
            detail="Only DICOM files, NIfTI masks and archives may be uploaded.",
        )
    return safe


def safe_join(root: Path, *parts: str) -> Path:
    """Join ``parts`` onto ``root``, refusing anything that escapes ``root``."""
    root = root.resolve()
    candidate = root.joinpath(*[sanitize_filename(p) for p in parts]).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValidationError("Resolved path escapes the workspace directory.")
    return candidate


def check_upload_size(total_bytes: int, *, limit: int) -> None:
    from ..core.errors import PayloadTooLargeError

    if total_bytes > limit:
        raise PayloadTooLargeError(
            f"Upload exceeds the {limit / (1024 ** 3):.1f} GB limit for a single request."
        )
