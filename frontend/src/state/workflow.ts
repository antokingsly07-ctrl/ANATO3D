// The 7-step workflow state machine.
//
// Kept as a pure reducer so the step gating and log handling can be unit-tested
// without React, and so a single source of truth drives every panel.

import type {
  AnalysisPayload,
  AnalyzeResult,
  DatasetSummary,
  ExportManifest,
  Health,
  LogEntry,
  MeshResult,
  Plane,
  ReconstructionResult,
  SegmentationResult,
  ValidationResult,
  VolumeGeometry,
} from "../types";

export type StepKey =
  | "upload"
  | "inspect"
  | "reconstruct"
  | "segment"
  | "mesh"
  | "analyze"
  | "export";

export interface StepDefinition {
  key: StepKey;
  label: string;
  blurb: string;
}

export const STEPS: StepDefinition[] = [
  { key: "upload", label: "Upload", blurb: "Load DICOM or generate a synthetic phantom" },
  { key: "inspect", label: "Inspect", blurb: "Scan headers, series and patient-space ordering" },
  { key: "reconstruct", label: "Reconstruct", blurb: "Build the 3D volume from geometry" },
  { key: "segment", label: "Segment", blurb: "Label anatomy with a classical or AI engine" },
  { key: "mesh", label: "3D Model", blurb: "Marching cubes surface + refinement" },
  { key: "analyze", label: "Analyze", blurb: "Measurements and validation checks" },
  { key: "export", label: "Export", blurb: "Download volume, mask, model and reports" },
];

export type StepStatus = "locked" | "available" | "complete";

export interface WorkflowState {
  step: StepKey;
  health: Health | null;
  dataset: DatasetSummary | null;
  analyze: AnalyzeResult | null;
  volume: ReconstructionResult | null;
  geometry: VolumeGeometry | null;
  segmentation: SegmentationResult | null;
  mesh: MeshResult | null;
  validation: ValidationResult | null;
  analysis: AnalysisPayload | null;
  manifest: ExportManifest | null;
  groundTruthId: string | null;
  activePlane: Plane;
  sliceIndex: Record<Plane, number>;
  windowCenter: number | null;
  windowWidth: number | null;
  running: boolean;
  progress: number;
  stage: string;
  logs: LogEntry[];
  notices: { severity: "error" | "warning" | "info"; message: string }[];
}

export const initialState: WorkflowState = {
  step: "upload",
  health: null,
  dataset: null,
  analyze: null,
  volume: null,
  geometry: null,
  segmentation: null,
  mesh: null,
  validation: null,
  analysis: null,
  manifest: null,
  groundTruthId: null,
  activePlane: "axial",
  sliceIndex: { axial: 0, coronal: 0, sagittal: 0 },
  windowCenter: null,
  windowWidth: null,
  running: false,
  progress: 0,
  stage: "",
  logs: [],
  notices: [],
};

let logSeq = 0;
export function makeLog(level: string, message: string, stage = ""): LogEntry {
  logSeq += 1;
  return { time: Date.now() / 1000 + logSeq * 1e-6, level, message, stage: stage || null };
}

export type WorkflowAction =
  | { type: "patch"; payload: Partial<WorkflowState> }
  | { type: "setStep"; step: StepKey }
  | { type: "setPlane"; plane: Plane }
  | { type: "setSliceIndex"; plane: Plane; index: number; count?: number }
  | { type: "setWindow"; center: number | null; width: number | null }
  | { type: "log"; entry: LogEntry }
  | { type: "logs"; entries: LogEntry[] }
  | { type: "notice"; severity: "error" | "warning" | "info"; message: string }
  | { type: "clearNotices" }
  | { type: "runStart"; stage: string }
  | { type: "runProgress"; progress?: number; stage?: string }
  | { type: "runEnd" }
  | { type: "reset" };

const MAX_LOGS = 400;

export function reducer(state: WorkflowState, action: WorkflowAction): WorkflowState {
  switch (action.type) {
    case "patch":
      return { ...state, ...action.payload };
    case "setStep":
      return { ...state, step: action.step };
    case "setPlane":
      return { ...state, activePlane: action.plane };
    case "setSliceIndex": {
      const raw = Math.max(0, Math.floor(action.index));
      const max = action.count != null ? Math.max(0, action.count - 1) : raw;
      const index = Math.min(raw, max);
      return {
        ...state,
        sliceIndex: { ...state.sliceIndex, [action.plane]: index },
      };
    }
    case "setWindow":
      return { ...state, windowCenter: action.center, windowWidth: action.width };
    case "log":
      return { ...state, logs: [...state.logs, action.entry].slice(-MAX_LOGS) };
    case "logs":
      return { ...state, logs: [...state.logs, ...action.entries].slice(-MAX_LOGS) };
    case "notice":
      return {
        ...state,
        notices: [...state.notices, { severity: action.severity, message: action.message }].slice(-20),
      };
    case "clearNotices":
      return { ...state, notices: [] };
    case "runStart":
      return { ...state, running: true, progress: 0, stage: action.stage };
    case "runProgress":
      return {
        ...state,
        progress: action.progress ?? state.progress,
        stage: action.stage ?? state.stage,
      };
    case "runEnd":
      return { ...state, running: false, progress: 1 };
    case "reset":
      return { ...initialState, health: state.health, logs: state.logs };
    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Selectors
// ---------------------------------------------------------------------------

export function stepStatus(state: WorkflowState, key: StepKey): StepStatus {
  const completed: Record<StepKey, boolean> = {
    upload: state.dataset != null,
    inspect: state.analyze != null,
    reconstruct: state.volume != null,
    segment: state.segmentation != null,
    mesh: state.mesh != null,
    analyze: state.analysis != null || state.validation != null,
    export: state.manifest != null,
  };
  if (completed[key]) return "complete";

  const order = STEPS.map((step) => step.key);
  const target = order.indexOf(key);
  const prerequisitesMet = order.slice(0, target).every((prior) => completed[prior]);
  return prerequisitesMet ? "available" : "locked";
}

/** Which step should be offered as the suggested next action. */
export function suggestedStep(state: WorkflowState): StepKey {
  for (const step of STEPS) {
    if (stepStatus(state, step.key) !== "complete") return step.key;
  }
  return "export";
}

export function collectFindings(state: WorkflowState): {
  errors: number;
  warnings: number;
  info: number;
} {
  const pools = [
    state.analyze?.findings ?? [],
    state.volume?.findings ?? [],
    state.segmentation?.findings ?? [],
    state.validation?.findings ?? [],
  ];
  let errors = 0;
  let warnings = 0;
  let info = 0;
  for (const pool of pools) {
    for (const finding of pool) {
      if (finding.severity === "error") errors += 1;
      else if (finding.severity === "warning") warnings += 1;
      else info += 1;
    }
  }
  return { errors, warnings, info };
}
