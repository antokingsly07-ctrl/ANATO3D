import { useEffect, useMemo, useState } from "react";
import { Box, Layers, ShieldCheck } from "lucide-react";

import { api } from "../api/client";
import { useWorkflow } from "../state/WorkflowProvider";
import type { SegmentationEngine, SegmentationPreset } from "../types";
import { Badge, Button, Field, Panel, Spinner } from "./ui";

export function ActionPanel() {
  const { state, reconstruct, segment, buildMesh, validate, uploadGroundTruth } = useWorkflow();
  const volumeId = state.volume?.volume_id ?? null;

  const [engines, setEngines] = useState<SegmentationEngine[]>([]);
  const [presets, setPresets] = useState<SegmentationPreset[]>([]);
  const [presetKey, setPresetKey] = useState<string>("ct_bone");
  const [engineKey, setEngineKey] = useState<string>("");
  const [label, setLabel] = useState("");
  const [minThreshold, setMinThreshold] = useState<string>("");
  const [isotropic, setIsotropic] = useState(false);
  const [fallback, setFallback] = useState(false);
  const [minComponent, setMinComponent] = useState(25);
  const [smoothIterations, setSmoothIterations] = useState(8);
  const [maxFaces, setMaxFaces] = useState(0);

  useEffect(() => {
    if (!volumeId) return;
    let cancelled = false;
    Promise.all([api.listEngines(volumeId), api.listPresets({ volumeId })])
      .then(([engineList, presetList]) => {
        if (cancelled) return;
        setEngines(engineList);
        setPresets(presetList);
        const preferred = presetList.find((preset) => preset.applicable) ?? presetList[0];
        if (preferred) setPresetKey(preferred.key);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [volumeId]);

  const activePreset = useMemo(
    () => presets.find((preset) => preset.key === presetKey) ?? null,
    [presets, presetKey],
  );

  const selectedEngine = engineKey || activePreset?.engine || "threshold";

  return (
    <Panel title="Pipeline actions">
      <div className="space-y-3">
        {/* Reconstruct */}
        <div className="rounded-md border border-ink-700/60 bg-ink-850/50 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
              <Layers size={13} /> Reconstruct volume
            </span>
            {state.volume ? <Badge tone="ok">built</Badge> : null}
          </div>
          <div className="space-y-1.5">
            <label className="flex items-center gap-2 text-[11px] text-slate-400">
              <input
                type="checkbox"
                checked={isotropic}
                onChange={(event) => setIsotropic(event.target.checked)}
              />
              Resample to isotropic 1 mm
            </label>
            <label className="flex items-center gap-2 text-[11px] text-slate-400">
              <input
                type="checkbox"
                checked={fallback}
                onChange={(event) => setFallback(event.target.checked)}
              />
              Allow InstanceNumber fallback (flagged)
            </label>
            <Button
              variant={state.analyze && !state.volume ? "primary" : "ghost"}
              className="w-full"
              disabled={!state.analyze || state.running}
              onClick={() => void reconstruct({ isotropic, allowInstanceNumberFallback: fallback })}
            >
              {state.running ? <Spinner /> : null}
              {state.volume ? "Re-reconstruct" : "Reconstruct"}
            </Button>
            {fallback ? (
              <p className="text-[10px] text-severity-warning">
                Fallback ordering ignores patient geometry and will be flagged as spatially
                unverified.
              </p>
            ) : null}
          </div>
        </div>

        {/* Segment */}
        <div className="rounded-md border border-ink-700/60 bg-ink-850/50 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
              <Box size={13} /> Segment anatomy
            </span>
            {state.segmentation ? <Badge tone="ok">segmented</Badge> : null}
          </div>

          {!volumeId ? (
            <p className="text-[11px] text-slate-500">Reconstruct a volume first.</p>
          ) : (
            <div className="space-y-2">
              <Field label="Anatomical preset">
                <select
                  className="field-input"
                  value={presetKey}
                  onChange={(event) => setPresetKey(event.target.value)}
                >
                  {presets.map((preset) => (
                    <option key={preset.key} value={preset.key}>
                      {preset.label}
                      {preset.applicable === false ? " (not for this modality)" : ""}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="Engine">
                <select
                  className="field-input"
                  value={engineKey}
                  onChange={(event) => setEngineKey(event.target.value)}
                >
                  <option value="">
                    Preset default ({activePreset?.engine ?? "threshold"})
                  </option>
                  {engines.map((engine) => (
                    <option key={engine.key} value={engine.key} disabled={!engine.availability.available}>
                      {engine.display_name}
                      {engine.availability.available ? "" : " — unavailable"}
                    </option>
                  ))}
                </select>
              </Field>

              {selectedEngine === "threshold" ? (
                <Field label="Custom lower threshold (HU)" hint="Leave blank to use the preset.">
                  <input
                    className="field-input"
                    type="number"
                    placeholder={String(activePreset?.params?.threshold_min ?? 200)}
                    value={minThreshold}
                    onChange={(event) => setMinThreshold(event.target.value)}
                  />
                </Field>
              ) : null}

              <Field label="Label">
                <input
                  className="field-input"
                  placeholder={activePreset?.label ?? "Structure"}
                  value={label}
                  onChange={(event) => setLabel(event.target.value)}
                />
              </Field>

              {activePreset?.applicability_note ? (
                <p className="text-[10px] text-severity-warning">{activePreset.applicability_note}</p>
              ) : null}
              {activePreset?.description ? (
                <p className="text-[10px] text-slate-500">{activePreset.description}</p>
              ) : null}

              {engines.find((engine) => engine.key === "ai")?.availability.available === false ? (
                <p className="text-[10px] text-severity-info">
                  AI engine unavailable —{" "}
                  {engines.find((engine) => engine.key === "ai")?.availability.reason}
                </p>
              ) : null}

              <Button
                variant={volumeId && !state.segmentation ? "primary" : "ghost"}
                className="w-full"
                disabled={state.running}
                onClick={() =>
                  void segment({
                    preset: presetKey,
                    engine: engineKey || null,
                    label: label || null,
                    params: minThreshold
                      ? { mode: "absolute", threshold_min: Number(minThreshold) }
                      : {},
                  })
                }
              >
                {state.running ? <Spinner /> : null}
                Run segmentation
              </Button>
            </div>
          )}
        </div>

        {/* Mesh */}
        <div className="rounded-md border border-ink-700/60 bg-ink-850/50 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
              <Box size={13} /> Build 3D model
            </span>
            {state.mesh ? <Badge tone="ok">meshed</Badge> : null}
          </div>
          {!state.segmentation ? (
            <p className="text-[11px] text-slate-500">Segment a structure first.</p>
          ) : (
            <div className="space-y-2">
              <div className="grid grid-cols-3 gap-2">
                <Field label="Min vol (mm³)">
                  <input
                    className="field-input"
                    type="number"
                    min={0}
                    value={minComponent}
                    onChange={(event) => setMinComponent(Number(event.target.value))}
                  />
                </Field>
                <Field label="Smooth iter.">
                  <input
                    className="field-input"
                    type="number"
                    min={0}
                    max={100}
                    value={smoothIterations}
                    onChange={(event) => setSmoothIterations(Number(event.target.value))}
                  />
                </Field>
                <Field label="Max faces">
                  <input
                    className="field-input"
                    type="number"
                    min={0}
                    value={maxFaces}
                    onChange={(event) => setMaxFaces(Number(event.target.value))}
                  />
                </Field>
              </div>
              <Button
                variant="primary"
                className="w-full"
                disabled={state.running}
                onClick={() =>
                  void buildMesh({
                    min_component_volume_mm3: minComponent,
                    smooth_iterations: smoothIterations,
                    max_faces: maxFaces,
                  })
                }
              >
                {state.running ? <Spinner /> : null}
                Generate mesh
              </Button>
            </div>
          )}
        </div>

        {/* Validate */}
        <div className="rounded-md border border-ink-700/60 bg-ink-850/50 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
              <ShieldCheck size={13} /> Validate
            </span>
            {state.validation ? (
              <Badge tone={state.validation.ground_truth.state === "computed" ? "ok" : "warning"}>
                {state.validation.ground_truth.state}
              </Badge>
            ) : null}
          </div>
          <div className="space-y-2">
            {state.groundTruthId ? (
              <Badge tone="accent">reference mask registered</Badge>
            ) : (
              <label className="block">
                <span className="field-label">Reference mask (optional)</span>
                <input
                  type="file"
                  className="mt-1 block w-full text-[11px] text-slate-400"
                  accept=".npz,.npy,.nii,.gz,.zip,.dcm"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void uploadGroundTruth(file);
                    event.target.value = "";
                  }}
                />
              </label>
            )}
            <Button
              className="w-full"
              disabled={!state.volume || state.running}
              onClick={() => void validate()}
            >
              {state.running ? <Spinner /> : null}
              Run validation
            </Button>
          </div>
        </div>
      </div>
    </Panel>
  );
}
