// React context that binds the pure workflow reducer to the real API.
//
// Every long-running operation is dispatched as a real backend job and polled;
// progress and log lines shown in the UI come from the backend's own job
// records, never from timers.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";

import { api, ApiRequestError, pollJob } from "../api/client";
import type {
  JobEnvelope,
  LogEntry,
  Plane,
  ReconstructionResult,
  SegmentationResult,
  SyntheticOptions,
  MeshResult,
  ValidationResult,
} from "../types";
import {
  collectFindings,
  initialState,
  makeLog,
  reducer,
  stepStatus,
  suggestedStep,
  type StepKey,
  type WorkflowState,
} from "./workflow";

function errorMessage(err: unknown): { message: string; detail: string | null } {
  if (err instanceof ApiRequestError) {
    return { message: err.message, detail: err.detail };
  }
  if (err instanceof Error) {
    return { message: err.message, detail: null };
  }
  return { message: String(err), detail: null };
}

export interface WorkflowContextValue {
  state: WorkflowState;
  findings: { errors: number; warnings: number; info: number };
  stepStatus: (key: StepKey) => ReturnType<typeof stepStatus>;
  suggestedStep: StepKey;
  refreshHealth: () => Promise<void>;
  generateSynthetic: (options: Partial<SyntheticOptions>) => Promise<void>;
  uploadFiles: (files: File[]) => Promise<void>;
  analyze: () => Promise<void>;
  reconstruct: (options?: { allowInstanceNumberFallback?: boolean; isotropic?: boolean }) => Promise<void>;
  segment: (input: {
    engine?: string | null;
    preset?: string | null;
    params?: Record<string, unknown>;
    label?: string | null;
    color?: string | null;
  }) => Promise<void>;
  buildMesh: (options?: Record<string, unknown>) => Promise<void>;
  validate: () => Promise<void>;
  uploadGroundTruth: (file: File) => Promise<void>;
  loadManifest: () => Promise<void>;
  setPlane: (plane: Plane) => void;
  setSliceIndex: (plane: Plane, index: number) => void;
  setWindow: (center: number | null, width: number | null) => void;
  setStep: (step: StepKey) => void;
  clearNotices: () => void;
  reset: () => void;
}

const WorkflowContext = createContext<WorkflowContextValue | null>(null);

export function WorkflowProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const stateRef = useRef(state);
  stateRef.current = state;

  const log = useCallback((level: string, message: string, stage = "") => {
    dispatch({ type: "log", entry: makeLog(level, message, stage) });
  }, []);

  const report = useCallback(
    (err: unknown) => {
      const { message, detail } = errorMessage(err);
      dispatch({ type: "notice", severity: "error", message });
      log("error", detail ? `${message} (${detail})` : message);
    },
    [log],
  );

  /**
   * Submit a job and resolve with its result.
   *
   * Handles both `?wait=true` (result already present) and background jobs that
   * must be polled, streaming the backend's real progress and logs into state.
   */
  const runJob = useCallback(
    async <T,>(
      label: string,
      invoke: () => Promise<JobEnvelope<T>>,
    ): Promise<T | null> => {
      dispatch({ type: "runStart", stage: label });
      try {
        const started = await invoke();
        let seen = 0;
        const ingest = (job: JobEnvelope<T>) => {
          if (job.logs && job.logs.length > seen) {
            const fresh = job.logs.slice(seen) as LogEntry[];
            seen = job.logs.length;
            dispatch({ type: "logs", entries: fresh });
          }
          dispatch({
            type: "runProgress",
            progress: job.progress,
            stage: job.message || job.stage || label,
          });
        };

        ingest(started);

        let final = started;
        if (started.status !== "completed" && started.status !== "failed") {
          final = await pollJob<T>(started.job_id, { onUpdate: ingest });
        }

        if (final.status === "failed") {
          const message = final.error?.message ?? `${label} failed.`;
          const detail = final.error?.detail ?? null;
          dispatch({ type: "notice", severity: "error", message });
          log("error", detail ? `${message} (${detail})` : message);
          return null;
        }

        dispatch({ type: "runEnd" });
        return (final.result as T) ?? null;
      } catch (err) {
        dispatch({ type: "runEnd" });
        report(err);
        return null;
      }
    },
    [log, report],
  );

  const refreshHealth = useCallback(async () => {
    try {
      const health = await api.health();
      dispatch({ type: "patch", payload: { health } });
      log(
        "info",
        `Backend ${health.application} ${health.version} online. AI segmentation: ${
          health.ai_segmentation_available ? "available" : "unavailable"
        }.`,
      );
    } catch (err) {
      report(err);
    }
  }, [log, report]);

  useEffect(() => {
    void refreshHealth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const generateSynthetic = useCallback(
    async (options: Partial<SyntheticOptions>) => {
      dispatch({ type: "runStart", stage: "Generating synthetic phantom" });
      try {
        const response = await api.createSynthetic(options);
        dispatch({
          type: "patch",
          payload: {
            dataset: {
              dataset_id: response.dataset_id,
              file_count: response.slice_count,
              dicom_file_count: response.slice_count,
              non_image_file_count: 0,
              unreadable_file_count: 0,
              total_bytes: 0,
              is_synthetic: true,
              synthetic_notes: response.notes,
              files: [],
              warnings: [],
            },
            groundTruthId: response.ground_truth_id,
            analyze: null,
            volume: null,
            geometry: null,
            segmentation: null,
            mesh: null,
            validation: null,
            analysis: null,
            manifest: null,
            step: "inspect",
          },
        });
        log(
          "info",
          `Synthetic dataset ${response.dataset_id} created: ${response.slice_count} slices, ` +
            `${response.dimensions.join("x")} @ ${response.spacing_mm.map((v) => v.toFixed(2)).join("x")} mm.`,
        );
        for (const note of response.notes) log("warning", note);
        if (response.ground_truth_available) {
          log("info", "Reference (ground-truth) mask registered for validation.");
        }
        dispatch({ type: "runEnd" });
      } catch (err) {
        dispatch({ type: "runEnd" });
        report(err);
      }
    },
    [log, report],
  );

  const uploadFiles = useCallback(
    async (files: File[]) => {
      dispatch({ type: "runStart", stage: "Uploading files" });
      try {
        const summary = await api.upload(files);
        dispatch({
          type: "patch",
          payload: {
            dataset: summary,
            groundTruthId: null,
            analyze: null,
            volume: null,
            geometry: null,
            segmentation: null,
            mesh: null,
            validation: null,
            analysis: null,
            manifest: null,
            step: "inspect",
          },
        });
        log(
          "info",
          `Uploaded ${summary.file_count} file(s): ${summary.dicom_file_count} DICOM, ` +
            `${summary.non_image_file_count} non-image, ${summary.unreadable_file_count} unreadable.`,
        );
        for (const warning of summary.warnings) log("warning", warning);
        dispatch({ type: "runEnd" });
      } catch (err) {
        dispatch({ type: "runEnd" });
        report(err);
      }
    },
    [log, report],
  );

  const analyze = useCallback(async () => {
    const dataset = stateRef.current.dataset;
    if (!dataset) return;
    const result = await runJob("Analysing dataset", () => api.analyze(dataset.dataset_id, false));
    if (!result) return;
    dispatch({ type: "patch", payload: { analyze: result, step: "reconstruct" } });
    if (result.summary?.status === "error") {
      dispatch({
        type: "notice",
        severity: "error",
        message: "The analysis reported blocking errors; reconstruction may be unreliable.",
      });
    }
    log(
      "info",
      `Found ${result.series.length} series across ${result.studies.length} study(ies).`,
    );
  }, [log, runJob]);

  const reconstruct = useCallback(
    async (options?: { allowInstanceNumberFallback?: boolean; isotropic?: boolean }) => {
      const current = stateRef.current;
      if (!current.dataset) return;
      const seriesUid = current.analyze?.primary_series_uid ?? null;
      const result = await runJob("Reconstructing volume", () =>
        api.createReconstruction(
          {
            dataset_id: current.dataset!.dataset_id,
            series_instance_uid: seriesUid,
            allow_instance_number_fallback: options?.allowInstanceNumberFallback ?? false,
            isotropic: options?.isotropic ?? false,
          },
          false,
        ),
      );
      if (!result) return;
      const reconstruction = result as ReconstructionResult;
      const counts = reconstruction.geometry.plane_slice_counts ?? {};
      dispatch({
        type: "patch",
        payload: {
          volume: reconstruction,
          geometry: reconstruction.geometry,
          sliceIndex: {
            axial: Math.floor((counts.axial ?? 1) / 2),
            coronal: Math.floor((counts.coronal ?? 1) / 2),
            sagittal: Math.floor((counts.sagittal ?? 1) / 2),
          },
          segmentation: null,
          mesh: null,
          validation: null,
          analysis: null,
          manifest: null,
          windowCenter: null,
          windowWidth: null,
          step: "segment",
        },
      });
      log(
        "info",
        `Volume ${reconstruction.volume_id}: ${reconstruction.geometry.dimensions.join("x")} @ ` +
          `${reconstruction.geometry.spacing_mm.map((v) => v.toFixed(2)).join("x")} mm ` +
          `(${reconstruction.geometry.orientation}).`,
      );
      for (const warning of reconstruction.geometry.warnings) log("warning", warning);
    },
    [log, runJob],
  );

  const segment = useCallback<WorkflowContextValue["segment"]>(
    async (input) => {
      const volume = stateRef.current.volume;
      if (!volume) return;
      const result = await runJob("Segmenting volume", () =>
        api.runSegmentation(
          {
            volume_id: volume.volume_id,
            engine: input.engine ?? null,
            preset: input.preset ?? null,
            params: input.params ?? {},
            label: input.label ?? null,
            color: input.color ?? null,
          },
          false,
        ),
      );
      if (!result) return;
      const segmentation = result as SegmentationResult;
      dispatch({
        type: "patch",
        payload: {
          segmentation,
          mesh: null,
          validation: null,
          analysis: null,
          manifest: null,
          step: "mesh",
        },
      });
      log(
        "info",
        `Segmentation '${segmentation.label}' via ${segmentation.engine}: ` +
          `${segmentation.statistics.voxel_count} voxels (${segmentation.statistics.volume_cm3.toFixed(3)} cm\u00b3).`,
      );
      for (const note of segmentation.notes) log("info", note);
      for (const warning of segmentation.warnings) log("warning", warning);
    },
    [log, runJob],
  );

  const buildMesh = useCallback(
    async (options?: Record<string, unknown>) => {
      const segmentation = stateRef.current.segmentation;
      if (!segmentation) return;
      const result = await runJob("Generating 3D mesh", () =>
        api.generateMesh(
          { segmentation_id: segmentation.segmentation_id, options: options ?? {} },
          false,
        ),
      );
      if (!result) return;
      const mesh = result as MeshResult;
      dispatch({ type: "patch", payload: { mesh, manifest: null, step: "analyze" } });
      log(
        "info",
        `Mesh ${mesh.mesh_id}: ${mesh.vertex_count} vertices, ${mesh.face_count} faces ` +
          `(${mesh.available_formats.join(", ")}).`,
      );
      for (const warning of mesh.warnings) log("warning", warning);

      try {
        const analysis = await api.getAnalysis(mesh.mesh_id);
        dispatch({ type: "patch", payload: { analysis } });
      } catch (err) {
        report(err);
      }
    },
    [log, report, runJob],
  );

  const validate = useCallback(async () => {
    const current = stateRef.current;
    if (!current.dataset || !current.volume) return;
    const result = await runJob("Running validation", () =>
      api.runValidation(
        {
          dataset_id: current.dataset!.dataset_id,
          volume_id: current.volume!.volume_id,
          segmentation_id: current.segmentation?.segmentation_id ?? null,
          ground_truth_id: current.groundTruthId,
        },
        false,
      ),
    );
    if (!result) return;
    const validation = result as ValidationResult;
    dispatch({ type: "patch", payload: { validation } });
    if (validation.ground_truth.state === "computed") {
      log("info", "Validation: ground-truth metrics computed.");
    } else {
      log("warning", validation.ground_truth.message);
    }
  }, [log, runJob]);

  const uploadGroundTruth = useCallback(
    async (file: File) => {
      const dataset = stateRef.current.dataset;
      if (!dataset) return;
      dispatch({ type: "runStart", stage: "Uploading reference mask" });
      try {
        const response = await api.uploadGroundTruth(dataset.dataset_id, file);
        dispatch({ type: "patch", payload: { groundTruthId: response.ground_truth_id } });
        log(
          "info",
          `Reference mask '${response.filename}' registered (${response.voxel_count} voxels, shape ${response.shape.join("x")}).`,
        );
        for (const warning of response.warnings) log("warning", warning);
        dispatch({ type: "runEnd" });
      } catch (err) {
        dispatch({ type: "runEnd" });
        report(err);
      }
    },
    [log, report],
  );

  const loadManifest = useCallback(async () => {
    const current = stateRef.current;
    if (!current.dataset) return;
    try {
      const manifest = await api.exportManifest(current.dataset.dataset_id, {
        volumeId: current.volume?.volume_id ?? null,
        segmentationId: current.segmentation?.segmentation_id ?? null,
        meshId: current.mesh?.mesh_id ?? null,
        validationId: current.validation?.validation_id ?? null,
      });
      dispatch({ type: "patch", payload: { manifest, step: "export" } });
    } catch (err) {
      report(err);
    }
  }, [report]);

  const setPlane = useCallback((plane: Plane) => dispatch({ type: "setPlane", plane }), []);
  const setSliceIndex = useCallback(
    (plane: Plane, index: number) => dispatch({ type: "setSliceIndex", plane, index }),
    [],
  );
  const setWindow = useCallback(
    (center: number | null, width: number | null) =>
      dispatch({ type: "setWindow", center, width }),
    [],
  );
  const setStep = useCallback((step: StepKey) => dispatch({ type: "setStep", step }), []);
  const clearNotices = useCallback(() => dispatch({ type: "clearNotices" }), []);
  const reset = useCallback(() => dispatch({ type: "reset" }), []);

  const findings = useMemo(() => collectFindings(state), [state]);

  const value = useMemo<WorkflowContextValue>(
    () => ({
      state,
      findings,
      stepStatus: (key: StepKey) => stepStatus(state, key),
      suggestedStep: suggestedStep(state),
      refreshHealth,
      generateSynthetic,
      uploadFiles,
      analyze,
      reconstruct,
      segment,
      buildMesh,
      validate,
      uploadGroundTruth,
      loadManifest,
      setPlane,
      setSliceIndex,
      setWindow,
      setStep,
      clearNotices,
      reset,
    }),
    [
      state,
      findings,
      refreshHealth,
      generateSynthetic,
      uploadFiles,
      analyze,
      reconstruct,
      segment,
      buildMesh,
      validate,
      uploadGroundTruth,
      loadManifest,
      setPlane,
      setSliceIndex,
      setWindow,
      setStep,
      clearNotices,
      reset,
    ],
  );

  return <WorkflowContext.Provider value={value}>{children}</WorkflowContext.Provider>;
}

export function useWorkflow(): WorkflowContextValue {
  const context = useContext(WorkflowContext);
  if (!context) {
    throw new Error("useWorkflow must be used inside a <WorkflowProvider>.");
  }
  return context;
}
