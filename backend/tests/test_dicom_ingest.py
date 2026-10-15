"""DICOM ingestion: scanning, metadata extraction and series grouping."""

from __future__ import annotations

from pathlib import Path

from app.services.dicom import reader


def _first_metadata(dataset):
    record = dataset.dicom_files[0]
    ds = reader.read_dataset(Path(dataset.root) / record.path)
    return reader.extract_metadata(ds)


def test_scan_finds_all_synthetic_files(phantom, dataset):
    # The dataset directory is DICOM-only (the ground-truth .npz is kept out).
    assert dataset.file_count == len(phantom["result"].files)
    assert len(dataset.dicom_files) == dataset.file_count


def test_metadata_contains_geometry(dataset):
    meta = _first_metadata(dataset)
    assert meta.rows == 64
    assert meta.columns == 64
    assert meta.modality == "CT"
    assert meta.image_position_patient is not None
    assert meta.image_orientation_patient is not None
    assert meta.pixel_spacing is not None


def test_metadata_has_no_patient_identifiers(dataset):
    meta = _first_metadata(dataset)
    forbidden = [
        "patient_name",
        "patient_id",
        "patient_birth_date",
        "patient_sex",
        "patient_age",
        "institution_name",
        "accession_number",
    ]
    for attribute in forbidden:
        assert not hasattr(meta, attribute), f"{attribute} must never be extracted"


def test_rescale_is_preserved(dataset):
    meta = _first_metadata(dataset)
    assert meta.rescale_intercept == -1024.0
    assert meta.rescale_slope == 1.0


def test_scan_reports_no_dicom_for_random_bytes(tmp_path):
    (tmp_path / "not_dicom.txt").write_text("this is not a dicom file", encoding="utf-8")
    scanned = reader.scan_dataset(tmp_path)
    assert scanned.file_count == 1
    assert scanned.dicom_files == []
