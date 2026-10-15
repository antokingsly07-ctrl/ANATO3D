"""Domain error hierarchy.

Every error carries a stable machine-readable ``code`` and a message written for
a clinician/researcher rather than a Python developer.  The API layer converts
these into structured JSON responses and never leaks stack traces.
"""

from __future__ import annotations


class AnatoError(Exception):
    """Base class for all expected (user-facing) failures."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ValidationError(AnatoError):
    """Input failed validation (bad request payload, bad parameters)."""

    code = "invalid_request"
    http_status = 400


class NotFoundError(AnatoError):
    code = "not_found"
    http_status = 404


class UnsupportedModalityError(AnatoError):
    code = "unsupported_modality"
    http_status = 422


class DicomError(AnatoError):
    code = "dicom_error"
    http_status = 422


class NoValidDicomSeriesError(DicomError):
    code = "no_valid_dicom_series"


class IncompleteOrientationError(DicomError):
    code = "incomplete_orientation"


class ReconstructionError(AnatoError):
    code = "reconstruction_error"
    http_status = 422


class PreprocessingError(AnatoError):
    code = "preprocessing_error"
    http_status = 422


class SegmentationError(AnatoError):
    code = "segmentation_error"
    http_status = 422


class EmptyMaskError(SegmentationError):
    code = "empty_mask"


class SegmentationUnavailableError(SegmentationError):
    code = "segmentation_unavailable"
    http_status = 501


class MeshGenerationError(AnatoError):
    code = "mesh_generation_error"
    http_status = 422


class PayloadTooLargeError(AnatoError):
    code = "payload_too_large"
    http_status = 413


class UnsupportedFileError(AnatoError):
    code = "unsupported_file"
    http_status = 415
