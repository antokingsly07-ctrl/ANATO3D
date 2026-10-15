"""Logging helpers.

Privacy rules enforced here:

* Patient identifying fields (name, ID, birth date, ...) are never logged.
* Accession numbers and UIDs are truncated / hashed when logged.

Every log entry produced by a processing step is also appended to the job's
in-memory log so the UI can display a real processing trace.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from typing import Any

_CONFIGURED = False

# Tags whose values must never reach a log sink.
SENSITIVE_TAGS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "AccessionNumber",
    "InstitutionName",
    "ReferringPhysicianName",
    "OperatorsName",
    "PhysiciansOfRecord",
)


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    root = logging.getLogger("anato3d")
    root.handlers = [handler]
    root.setLevel(level.upper())
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"anato3d.{name}")


def redact_uid(uid: str | None) -> str:
    """Return a short, stable, non-reversible fingerprint of a DICOM UID."""
    if not uid:
        return "<none>"
    digest = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:12]
    return f"uid:{digest}"


def scrub_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Remove patient-identifying keys from a metadata dictionary."""
    return {k: v for k, v in meta.items() if k not in SENSITIVE_TAGS}
