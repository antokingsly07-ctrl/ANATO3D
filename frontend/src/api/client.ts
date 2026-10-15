// Typed fetch client for the ANATO3D FastAPI backend.
//
// Errors are normalised into ApiRequestError so the UI can display the
// backend's human-readable `message` and `code` (never a raw stack trace).

import type {
  AnalysisPayload,
  AnalyzeResult,
  DatasetSummary,
  ExportManifest,
  GroundTruthUploadResponse,
  Health,
  JobEnvelope,
  MaskSlice,
  MeshResult,
  Plane,
  SegmentationEngine,
  SegmentationPreset,
  SegmentationResult,
  SliceResponse,
  SyntheticOptions,
  SyntheticResponse,
  ValidationResult,
  VolumeGeometry,
} from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

export class ApiRequestError extends Error {
  readonly code: string;
  readonly status: number;
  readonly detail: string | null;

  constructor(status: number, code: string, message: string, detail: string | null = null) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), init);
  } catch (cause) {
    throw new ApiRequestError(0, "network_error", "Could not reach the ANATO3D backend.", String(cause));
  }

  const text = await response.text();
  const body = text ? (safeJson(text) as Record<string, unknown> | null) : null;

  if (!response.ok) {
    const code = typeof body?.code === "string" ? body.code : `http_${response.status}`;
    const message =
      typeof body?.message === "string"
        ? body.message
        : `Request failed with status ${response.status}.`;
    const detail = typeof body?.detail === "string" ? body.detail : null;
    throw new ApiRequestError(response.status, code, message, detail);
  }

  return body as T;
}

function jsonInit(method: string, payload: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}

// ---------------------------------------------------------------------------
// Job polling
// ---------------------------------------------------------------------------

export interface PollOptions<T = Record<string, unknown>> {
  intervalMs?: number;
  timeoutMs?: number;
  onUpdate?: (job: JobEnvelope<T>) => void;
  signal?: AbortSignal;
}

/**
 * Poll `GET /api/jobs/{id}` until the job finishes.
 *
 * Resolves with the final envelope (status `completed` or `failed`).
 */
export async function pollJob<T>(
  jobId: string,
  { intervalMs = 350, timeoutMs = 15 * 60 * 1000, onUpdate, signal }: PollOptions<T> = {},
): Promise<JobEnvelope<T>> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (signal?.aborted) {
      throw new ApiRequestError(0, "aborted", "The operation was cancelled.");
    }
    const job = await request<JobEnvelope<T>>(`/api/jobs/${encodeURIComponent(jobId)}`);
    onUpdate?.(job);
    if (job.status === "completed" || job.status === "failed") {
      return job;
    }
    if (Date.now() > deadline) {
      throw new ApiRequestError(0, "timeout", "The operation timed out before completing.");
    }
    await sleep(intervalMs);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const api = {
  health(): Promise<Health> {
    return request<Health>("/api/health");
  },

  listJobs(limit = 25): Promise<JobEnvelope[]> {
    return request<JobEnvelope[]>(`/api/jobs?limit=${limit}`);
  },

  getJob<T = Record<string, unknown>>(jobId: string): Promise<JobEnvelope<T>> {
    return request<JobEnvelope<T>>(`/api/jobs/${encodeURIComponent(jobId)}`);
  },

  async upload(files: File[]): Promise<DatasetSummary> {
    const form = new FormData();
    for (const file of files) {
      form.append("files", file, file.name);
    }
    return request<DatasetSummary>("/api/dicom/upload", { method: "POST", body: form });
  },

  getDataset(datasetId: string): Promise<DatasetSummary> {
    return request<DatasetSummary>(`/api/dicom/${encodeURIComponent(datasetId)}`);
  },

  createSynthetic(options: Partial<SyntheticOptions>): Promise<SyntheticResponse> {
    return request<SyntheticResponse>("/api/dicom/synthetic", jsonInit("POST", options));
  },

  analyze(datasetId: string, wait = true): Promise<JobEnvelope<AnalyzeResult>> {
    return request<JobEnvelope<AnalyzeResult>>(
      `/api/dicom/analyze?wait=${wait ? "true" : "false"}`,
      jsonInit("POST", { dataset_id: datasetId }),
    );
  },

  createReconstruction(
    body: {
      dataset_id: string;
      series_instance_uid?: string | null;
      allow_instance_number_fallback?: boolean;
      uniform_spacing_mm?: number[] | null;
      isotropic?: boolean;
    },
    wait = false,
  ): Promise<JobEnvelope<ReconstructionLike>> {
    return request<JobEnvelope<ReconstructionLike>>(
      `/api/reconstruction/create?wait=${wait ? "true" : "false"}`,
      jsonInit("POST", body),
    );
  },

  getVolume(volumeId: string): Promise<Record<string, unknown>> {
    return request<Record<string, unknown>>(`/api/reconstruction/${encodeURIComponent(volumeId)}`);
  },

  getGeometry(volumeId: string): Promise<VolumeGeometry> {
    return request<VolumeGeometry>(
      `/api/reconstruction/${encodeURIComponent(volumeId)}/geometry`,
    );
  },

  getSlice(
    volumeId: string,
    plane: Plane,
    index: number,
    window?: { center?: number | null; width?: number | null },
  ): Promise<SliceResponse> {
    const params = new URLSearchParams({ plane, index: String(index) });
    if (window?.center != null) params.set("window_center", String(window.center));
    if (window?.width != null) params.set("window_width", String(window.width));
    return request<SliceResponse>(
      `/api/reconstruction/${encodeURIComponent(volumeId)}/slice?${params.toString()}`,
    );
  },

  thumbnailUrl(
    volumeId: string,
    plane: Plane,
    options: { index?: number; maxSize?: number; center?: number | null; width?: number | null } = {},
  ): string {
    const params = new URLSearchParams({ plane, max_size: String(options.maxSize ?? 256) });
    if (options.index != null) params.set("index", String(options.index));
    if (options.center != null) params.set("window_center", String(options.center));
    if (options.width != null) params.set("window_width", String(options.width));
    return apiUrl(`/api/reconstruction/${encodeURIComponent(volumeId)}/thumbnail?${params.toString()}`);
  },

  windowPresets(modality?: string): Promise<Array<Record<string, unknown>>> {
    const query = modality ? `?modality=${encodeURIComponent(modality)}` : "";
    return request<Array<Record<string, unknown>>>(`/api/reconstruction/window-presets${query}`);
  },

  listEngines(volumeId?: string): Promise<SegmentationEngine[]> {
    const query = volumeId ? `?volume_id=${encodeURIComponent(volumeId)}` : "";
    return request<SegmentationEngine[]>(`/api/segmentation/engines${query}`);
  },

  listPresets(params: { volumeId?: string; modality?: string } = {}): Promise<SegmentationPreset[]> {
    const search = new URLSearchParams();
    if (params.volumeId) search.set("volume_id", params.volumeId);
    if (params.modality) search.set("modality", params.modality);
    const query = search.toString() ? `?${search.toString()}` : "";
    return request<SegmentationPreset[]>(`/api/segmentation/presets${query}`);
  },

  runSegmentation(
    body: {
      volume_id: string;
      engine?: string | null;
      preset?: string | null;
      params?: Record<string, unknown>;
      label?: string | null;
      color?: string | null;
    },
    wait = true,
  ): Promise<JobEnvelope<SegmentationResult>> {
    return request<JobEnvelope<SegmentationResult>>(
      `/api/segmentation/run?wait=${wait ? "true" : "false"}`,
      jsonInit("POST", body),
    );
  },

  getSegmentation(segmentationId: string): Promise<SegmentationResult> {
    return request<SegmentationResult>(
      `/api/segmentation/${encodeURIComponent(segmentationId)}`,
    );
  },

  getMaskSlice(segmentationId: string, plane: Plane, index: number): Promise<MaskSlice> {
    const params = new URLSearchParams({ plane, index: String(index) });
    return request<MaskSlice>(
      `/api/segmentation/${encodeURIComponent(segmentationId)}/mask?${params.toString()}`,
    );
  },

  deleteSegmentation(segmentationId: string): Promise<Record<string, unknown>> {
    return request(`/api/segmentation/${encodeURIComponent(segmentationId)}`, { method: "DELETE" });
  },

  generateMesh(
    body: { segmentation_id: string; options?: Record<string, unknown> },
    wait = false,
  ): Promise<JobEnvelope<MeshResult>> {
    return request<JobEnvelope<MeshResult>>(
      `/api/mesh/generate?wait=${wait ? "true" : "false"}`,
      jsonInit("POST", body),
    );
  },

  getMesh(meshId: string): Promise<MeshResult> {
    return request<MeshResult>(`/api/mesh/${encodeURIComponent(meshId)}`);
  },

  meshGlbUrl(meshId: string): string {
    return apiUrl(`/api/mesh/${encodeURIComponent(meshId)}/glb`);
  },

  meshDownloadUrl(meshId: string, format = "glb"): string {
    return apiUrl(`/api/mesh/${encodeURIComponent(meshId)}/download?format=${format}`);
  },

  runValidation(
    body: {
      dataset_id?: string | null;
      volume_id?: string | null;
      segmentation_id?: string | null;
      reference_segmentation_id?: string | null;
      ground_truth_id?: string | null;
    },
    wait = true,
  ): Promise<JobEnvelope<ValidationResult>> {
    return request<JobEnvelope<ValidationResult>>(
      `/api/validation/run?wait=${wait ? "true" : "false"}`,
      jsonInit("POST", body),
    );
  },

  getValidation(validationId: string): Promise<ValidationResult> {
    return request<ValidationResult>(`/api/validation/${encodeURIComponent(validationId)}`);
  },

  async uploadGroundTruth(datasetId: string, file: File): Promise<GroundTruthUploadResponse> {
    const form = new FormData();
    form.append("dataset_id", datasetId);
    form.append("file", file, file.name);
    return request<GroundTruthUploadResponse>("/api/validation/ground-truth", {
      method: "POST",
      body: form,
    });
  },

  getAnalysis(id: string): Promise<AnalysisPayload> {
    return request<AnalysisPayload>(`/api/analysis/${encodeURIComponent(id)}`);
  },

  exportManifest(
    datasetId: string,
    refs: {
      volumeId?: string | null;
      segmentationId?: string | null;
      meshId?: string | null;
      validationId?: string | null;
    } = {},
  ): Promise<ExportManifest> {
    const params = new URLSearchParams();
    if (refs.volumeId) params.set("volume_id", refs.volumeId);
    if (refs.segmentationId) params.set("segmentation_id", refs.segmentationId);
    if (refs.meshId) params.set("mesh_id", refs.meshId);
    if (refs.validationId) params.set("validation_id", refs.validationId);
    const query = params.toString() ? `?${params.toString()}` : "";
    return request<ExportManifest>(`/api/export/${encodeURIComponent(datasetId)}/manifest${query}`);
  },

  exportUrl(datasetId: string, key: string, query: string): string {
    return apiUrl(`/api/export/${encodeURIComponent(datasetId)}/file/${key}${query}`);
  },
};

// The reconstruction envelope result is the ReconstructionResponse shape; typed
// loosely here and narrowed in the workflow layer to avoid a circular import.
export interface ReconstructionLike {
  volume_id: string;
  dataset_id: string;
  report: Record<string, unknown>;
  geometry: VolumeGeometry;
  findings: import("../types").Finding[];
  summary: import("../types").ValidationSummary;
  slice_preview: import("../types").SlicePreview[];
}
