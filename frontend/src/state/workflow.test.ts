import { describe, expect, it } from "vitest";

import type { Finding, ValidationSummary } from "../types";
import {
  collectFindings,
  initialState,
  makeLog,
  reducer,
  stepStatus,
  suggestedStep,
  type WorkflowState,
} from "./workflow";

const summary = (errors = 0, warnings = 0): ValidationSummary => ({
  total: errors + warnings,
  errors,
  warnings,
  info: 0,
  status: errors ? "error" : warnings ? "warning" : "ok",
});

function withFindings(state: WorkflowState, findings: Finding[]): WorkflowState {
  return {
    ...state,
    analyze: {
      dataset_id: "ds",
      studies: [],
      series: [],
      primary_series_uid: null,
      metadata_report: {},
      findings,
      summary: summary(),
      ordering_preview: [],
      frame_description: {},
      non_image_files: 0,
      unreadable_files: 0,
      elapsed_s: 0,
      warnings: [],
    },
  };
}

const finding = (severity: Finding["severity"], code: string): Finding => ({
  code,
  severity,
  title: code,
  message: code,
  context: {},
});

describe("reducer", () => {
  it("clamps the slice index to the plane's slice count", () => {
    const next = reducer(initialState, {
      type: "setSliceIndex",
      plane: "axial",
      index: 999,
      count: 32,
    });
    expect(next.sliceIndex.axial).toBe(31);
  });

  it("never lets a slice index go negative", () => {
    const next = reducer(initialState, { type: "setSliceIndex", plane: "coronal", index: -5 });
    expect(next.sliceIndex.coronal).toBe(0);
  });

  it("tracks a job's lifecycle through start, progress and end", () => {
    let state = reducer(initialState, { type: "runStart", stage: "Reconstructing" });
    expect(state.running).toBe(true);
    expect(state.progress).toBe(0);

    state = reducer(state, { type: "runProgress", progress: 0.5, stage: "marching cubes" });
    expect(state.progress).toBe(0.5);
    expect(state.stage).toBe("marching cubes");

    state = reducer(state, { type: "runEnd" });
    expect(state.running).toBe(false);
    expect(state.progress).toBe(1);
  });

  it("caps the log buffer so a long job cannot grow without bound", () => {
    let state = initialState;
    for (let i = 0; i < 500; i += 1) {
      state = reducer(state, { type: "log", entry: makeLog("info", `line ${i}`) });
    }
    expect(state.logs.length).toBe(400);
    expect(state.logs[state.logs.length - 1]!.message).toBe("line 499");
  });

  it("merges a partial patch without dropping unrelated artefacts", () => {
    const seeded: WorkflowState = {
      ...initialState,
      segmentation: { segmentation_id: "seg" } as never,
      windowCenter: 40,
      windowWidth: 400,
    };
    const next = reducer(seeded, {
      type: "patch",
      payload: { volume: { volume_id: "vol" } as never, windowCenter: null },
    });
    expect((next.volume as { volume_id: string }).volume_id).toBe("vol");
    expect(next.segmentation).not.toBeNull();
    expect(next.windowWidth).toBe(400);
    expect(next.windowCenter).toBeNull();
  });

  it("keeps the backend health across a reset but drops the data", () => {
    const seeded: WorkflowState = {
      ...initialState,
      dataset: { dataset_id: "ds" } as never,
      health: { status: "ok" } as never,
    };
    const next = reducer(seeded, { type: "reset" });
    expect(next.dataset).toBeNull();
    expect(next.health).not.toBeNull();
  });
});

describe("stepStatus", () => {
  it("offers only the first step on a pristine state", () => {
    expect(stepStatus(initialState, "upload")).toBe("available");
    expect(stepStatus(initialState, "inspect")).toBe("locked");
    expect(stepStatus(initialState, "reconstruct")).toBe("locked");
  });

  it("unlocks the next step once the previous one produced data", () => {
    const state: WorkflowState = { ...initialState, dataset: { dataset_id: "ds" } as never };
    expect(stepStatus(state, "upload")).toBe("complete");
    expect(stepStatus(state, "inspect")).toBe("available");
    expect(stepStatus(state, "reconstruct")).toBe("locked");
  });

  it("marks a step complete only when it has genuine output", () => {
    const state: WorkflowState = { ...initialState, volume: { volume_id: "vol" } as never };
    expect(stepStatus(state, "reconstruct")).toBe("complete");
  });
});

describe("suggestedStep", () => {
  it("points at the first incomplete step", () => {
    expect(suggestedStep(initialState)).toBe("upload");
    const state: WorkflowState = {
      ...initialState,
      dataset: { dataset_id: "ds" } as never,
      analyze: { dataset_id: "ds" } as never,
    };
    expect(suggestedStep(state)).toBe("reconstruct");
  });
});

describe("collectFindings", () => {
  it("counts findings by severity across every stage", () => {
    const state = withFindings(initialState, [
      finding("error", "ordering_unverified"),
      finding("warning", "irregular_spacing"),
      finding("info", "modality_supported"),
    ]);
    expect(collectFindings(state)).toEqual({ errors: 1, warnings: 1, info: 1 });
  });

  it("reports zeros when nothing has run", () => {
    expect(collectFindings(initialState)).toEqual({ errors: 0, warnings: 0, info: 0 });
  });
});
