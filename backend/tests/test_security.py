"""Upload hygiene and path handling."""

from __future__ import annotations

import json

import pytest

from app.core.errors import PayloadTooLargeError, UnsupportedFileError, ValidationError
from app.core.security import (
    check_upload_size,
    safe_join,
    sanitize_filename,
    validate_upload_name,
)


def test_sanitize_strips_directory_components():
    assert sanitize_filename("../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert "/" not in sanitize_filename("a/b/c.dcm")
    assert "\\" not in sanitize_filename("a\\b\\c.dcm")


def test_sanitize_removes_control_characters():
    assert sanitize_filename("bad\x00name.dcm") == "badname.dcm"
    assert sanitize_filename("tab\tname.dcm") == "tabname.dcm"


def test_sanitize_handles_empty_and_dot_only_names():
    assert sanitize_filename("") == "unnamed"
    assert sanitize_filename("..") == "unnamed"
    assert sanitize_filename(".") == "unnamed"


def test_sanitize_truncates_long_names():
    assert len(sanitize_filename("x" * 500)) <= 120


def test_sanitize_keeps_normal_dicom_names():
    assert sanitize_filename("IM_0001.dcm") == "IM_0001.dcm"
    assert sanitize_filename("1.2.840.113619.dcm") == "1.2.840.113619.dcm"


@pytest.mark.parametrize("name", ["evil.exe", "script.ps1", "x.sh", "a.py", "drop.bat", "lib.dll"])
def test_executable_extensions_are_rejected(name):
    with pytest.raises(UnsupportedFileError):
        validate_upload_name(name)


def test_allowed_extensions_pass():
    for name in ["slice.dcm", "mask.nii.gz", "study.zip", "no_extension"]:
        assert validate_upload_name(name)


def test_safe_join_never_escapes_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    joined = safe_join(root, "..", "..", "etc", "passwd")
    assert joined == root or root in joined.parents


def test_safe_join_stays_inside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    joined = safe_join(root, "series", "slice.dcm")
    assert root in joined.parents


def test_upload_size_limit():
    check_upload_size(10, limit=100)
    with pytest.raises(PayloadTooLargeError):
        check_upload_size(1000, limit=100)


def test_non_dicom_upload_is_rejected_by_the_api(client, tmp_path):
    """The API must refuse a file that carries no DICOM content."""
    payload = tmp_path / "notes.txt"
    payload.write_text("not a dicom file at all", encoding="utf-8")
    with payload.open("rb") as handle:
        response = client.post(
            "/api/dicom/upload",
            files={"files": ("notes.txt", handle, "text/plain")},
        )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "no_valid_dicom_series"
    # No internal detail (paths, stack frames) must leak to the client.
    assert "Traceback" not in json.dumps(body)


def test_blocked_extension_upload_is_rejected(client, tmp_path):
    payload = tmp_path / "payload.exe"
    payload.write_bytes(b"MZ\x90\x00not-really-an-executable")
    with payload.open("rb") as handle:
        response = client.post(
            "/api/dicom/upload",
            files={"files": ("payload.exe", handle, "application/octet-stream")},
        )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_file"
