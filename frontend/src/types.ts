// Types mirroring the FastAPI response models. Kept hand-written (no codegen)
// so the contract stays readable and reviewable.

export type Plane = "axial" | "coronal" | "sagittal";
export type Severity = "error" | "warning" | "info";
export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface ApiErrorBody {
  code: string;
  message: string;
  detail?: string | null;
}

export interface Health {
  status: "ok";
  application: string;
  version: string;
  ai_segmentation_available: boolean;
  ai_unavailable_reason: string | null;
  limits: Record<string, number>;
  storage: Record<string, unknown>;
}

export interface DicomFileRecord {
  name: string;
  stored_name: string;
  size_bytes: number;
  is_dicom: boolean;
  error: string | null;
}

export interface DatasetSummary {
  dataset_id: string;
  file_count: number;
  dicom_file_count: number;
  non_image_file_count: number;
  unreadable_file_count: number;
  total_bytes: number;
  is_synthetic: boolean;
  synthetic_notes: string[];
  files: DicomFileRecord[];
  warnings: string[];
}

export interface SyntheticResponse {
  dataset_id: string;
  slice_count: number;
  dimensions: number[];
  spacing_mm: number[];
  orientation: string;
  ground_truth_available: boolean;
  ground_truth_id: string | null;
  bone_volume_cm3: number;
  notes: string[];
  label: "SYNTHETIC";
}

export interface SyntheticOptions {
  rows: number;
  columns: number;
  slices: number;
  row_spacing_mm: number;
  column_spacing_mm: number;
  slice_spacing_mm: number;
  orientation: "axial" | "coronal" | "sagittal" | "oblique";
  irregular_spacing: boolean;
  include_lesion: boolean;
  shuffle_filenames: boolean;
}

export interface Finding {
  code: string;
  severity: Severity;
  title: string;
  message: string;
  context: Record<string, unknown>;
}

export interface ValidationSummary {
  total: number;
  errors: number;
  warnings: number;
  info: number;
  status: "ok" | "warning" | "error";
}

export interface ValidationSummaryModel extends ValidationSummary {}

export interface SliceOrderEntry {
  index: number;
  sop_instance_uid: string | null;
  instance_number: number | null;
  position_mm: number[] | null;
  projection_mm: number | null;
  file: string | null;
}

export interface SeriesSummary {
  series_instance_uid: string;
  study_instance_uid: string | null;
  modality: string;
  series_number: number | null;
  series_description: string | null;
  body_part_examined: string | null;
  rows: number | null;
  columns: number | null;
  slice_count: number;
  orientation: string | null;
  slice_spacing_mm: number | null;
  irregular_spacing: boolean;
  has_complete_geometry: boolean;
  is_reconstructable: boolean;
  issues: string[];
  warnings: string[];
}

export interface StudySummary {
  study_instance_uid: string;
  series_count: number;
  slice_count: number;
  study_description: string | null;
  modalities: string[];
  series_instance_uids: string[];
}

export interface AnalyzeResult {
  dataset_id: string;
  studies: StudySummary[];
  series: SeriesSummary[];
  primary_series_uid: string | null;
  metadata_report: Record<string, unknown>;
  findings: Finding[];
  summary: ValidationSummary;
  ordering_preview: SliceOrderEntry[];
  frame_description: Record<string, string>;
  non_image_files: number;
  unreadable_files: number;
  elapsed_s: number;
  warnings: string[];
}

export interface SlicePreview {
  plane: Plane;
  index: number;
  count: number;
  position_label: string;
  position_mm: number;
  png_base64: string;
}

export interface VolumeGeometry {
  volume_id: string;
  dataset_id: string | null;
  modality: string;
  series_instance_uid: string;
  series_description: string | null;
  dimensions: number[];
  spacing_mm: number[];
  origin_lps_mm: number[];
  direction: number[][];
  orientation: string;
  patient_position: string | null;
  physical_size_mm: number[];
  voxel_volume_mm3: number;
  intensity_units: string;
  window_center: number | null;
  window_width: number | null;
  is_resampled: boolean;
  is_normalized: boolean;
  plane_slice_counts: Record<string, number>;
  warnings: string[];
  notes: string[];
}

export interface ReconstructionResult {
  volume_id: string;
  dataset_id: string;
  report: Record<string, unknown>;
  geometry: VolumeGeometry;
  findings: Finding[];
  summary: ValidationSummary;
  slice_preview: SlicePreview[];
}

export interface SliceResponse {
  volume_id: string;
  plane: Plane;
  index: number;
  count: number;
  rows: number;
  columns: number;
  row_spacing_mm: number;
  column_spacing_mm: number;
  position_mm: number;
  position_label: string;
  labels: Record<string, string>;
  dtype: "int16" | "float32";
  encoding: "base64";
  data: string;
  intensity_min: number;
  intensity_max: number;
  intensity_units: string;
  window_center: number | null;
  window_width: number | null;
  warnings: string[];
}

export interface MaskSlice {
  segmentation_id: string;
  plane: Plane;
  index: number;
  rows: number;
  columns: number;
  encoding: "packbits-base64";
  data: string;
  shape: [number, number];
}

export interface ParameterSpec {
  key: string;
  label: string;
  type: string;
  default: unknown;
  minimum?: number | null;
  maximum?: number | null;
  step?: number | null;
  unit?: string | null;
  choices?: string[] | null;
  help?: string | null;
  advanced?: boolean;
}

export interface EngineAvailability {
  available: boolean;
  reason: string | null;
}

export interface SegmentationEngine {
  key: string;
  display_name: string;
  description: string;
  category: string;
  uses_hounsfield: boolean;
  parameters: ParameterSpec[];
  availability: EngineAvailability;
}

export interface SegmentationPreset {
  key: string;
  label: string;
  engine: string;
  modalities: string[];
  params: Record<string, unknown>;
  description: string;
  color: string;
  applicable?: boolean;
  applicability_note?: string;
}

export interface SegmentationResult {
  segmentation_id: string;
  volume_id: string;
  label: string;
  color: string;
  engine: string;
  requested_engine: string | null;
  statistics: {
    voxel_count: number;
    volume_mm3: number;
    volume_cm3: number;
    components: number;
    bounding_box_index?: number[][] | null;
  };
  params: Record<string, unknown>;
  notes: string[];
  warnings: string[];
  findings: Finding[];
  duration_s: number;
}

export interface MeasurementEntry {
  key: string;
  label: string;
  value: number | null;
  unit: string;
  detail?: string;
  direction?: string;
}

export interface MeshResult {
  mesh_id: string;
  segmentation_id: string;
  volume_id: string;
  label: string;
  color: string;
  vertex_count: number;
  face_count: number;
  measurements: Record<string, unknown>;
  measurement_summary: MeasurementEntry[];
  refinement: Record<string, unknown>;
  extraction: Record<string, unknown>;
  available_formats: string[];
  notes: string[];
  warnings: string[];
  duration_s: number;
}

export interface MetricValue {
  key: string;
  label: string;
  value: number | null;
  unit: string;
  interpretation: string;
  direction: string;
}

export interface GroundTruthBlock {
  state: "computed" | "unavailable";
  message: string;
  detail?: string | null;
  reference_source?: string;
  metrics?: MetricValue[];
  counts?: Record<string, number> | null;
  notes?: string[];
  warnings?: string[];
}

export interface ValidationResult {
  validation_id: string;
  summary: ValidationSummary;
  findings: Finding[];
  ground_truth: GroundTruthBlock;
  dataset_information: Record<string, unknown>;
  interpretation_notes: string[];
  disclaimer: string;
  duration_s: number;
}

export interface AnalysisPayload {
  id: string;
  kind: string;
  mesh_id: string;
  segmentation_id: string;
  volume_id: string;
  label: string;
  color: string;
  engine: string | null;
  measurements: Record<string, any>;
  measurement_summary: MeasurementEntry[];
  mesh: Record<string, any>;
  voxel: Record<string, any>;
  intensity: Record<string, any>;
  segmentation_statistics: Record<string, any>;
  volume_geometry: VolumeGeometry;
  notes: string[];
  warnings: string[];
  disclaimer: string;
}

export interface LogEntry {
  time: number;
  level: string;
  message: string;
  stage: string | null;
}

export interface JobEnvelope<T = Record<string, unknown>> {
  job_id: string;
  kind: string;
  status: JobStatus;
  stage: string;
  progress: number;
  message: string;
  warnings: string[];
  result: T | null;
  error: ApiErrorBody | null;
  duration_s: number | null;
  logs: LogEntry[];
}

export interface ExportItem {
  key: string;
  label: string;
  description: string;
  url: string | null;
  available: boolean;
  size_bytes: number | null;
  media_type: string | null;
}

export interface ExportManifest {
  dataset_id: string;
  items: ExportItem[];
  privacy_note: string;
}

export interface GroundTruthUploadResponse {
  ground_truth_id: string;
  dataset_id: string;
  filename: string;
  shape: number[];
  voxel_count: number;
  notes: string[];
  warnings: string[];
}
