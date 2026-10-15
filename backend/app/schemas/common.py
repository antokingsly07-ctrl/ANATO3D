"""Shared API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str = Field(..., description="Stable machine-readable error code.")
    message: str = Field(..., description="Message intended for a clinician/researcher, not a stack trace.")
    detail: str | None = Field(None, description="Optional extra context.")
    job_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    application: str
    version: str
    ai_segmentation_available: bool
    ai_unavailable_reason: str | None = None
    limits: dict[str, Any]
    storage: dict[str, Any]


class JobEnvelope(BaseModel):
    """Returned by every long-running endpoint.

    ``result`` is populated immediately when the request used ``wait=true``;
    otherwise poll ``GET /api/jobs/{job_id}`` until ``status`` is
    ``completed`` or ``failed``.
    """

    job_id: str
    kind: str
    status: Literal["pending", "running", "completed", "failed"]
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    warnings: list[str] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: ErrorResponse | None = None
    duration_s: float | None = None
    logs: list[dict[str, Any]] = Field(default_factory=list)


class JobStatusResponse(JobEnvelope):
    result_ready: bool = False


class LogEntryModel(BaseModel):
    time: float
    level: str
    message: str
    stage: str | None = None


class FindingModel(BaseModel):
    code: str
    severity: Literal["error", "warning", "info"]
    title: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class ValidationSummaryModel(BaseModel):
    total: int
    errors: int
    warnings: int
    info: int
    status: Literal["ok", "warning", "error"]
