import { useEffect, useState } from "react";

import { api } from "../api/client";
import { useWorkflow } from "../state/WorkflowProvider";
import { clamp, titleCase } from "../lib/format";
import type { Plane } from "../types";
import { Badge, Button, Panel } from "./ui";

const PLANES: Plane[] = ["axial", "coronal", "sagittal"];

interface WindowPreset {
  key: string;
  label: string;
  center: number;
  width: number;
}

export function ViewControls() {
  const { state, setPlane, setSliceIndex, setWindow } = useWorkflow();
  const geometry = state.geometry;
  const [presets, setPresets] = useState<WindowPreset[]>([]);

  useEffect(() => {
    if (!geometry) return;
    let cancelled = false;
    api
      .windowPresets(geometry.modality)
      .then((list) => {
        if (!cancelled) setPresets(list as unknown as WindowPreset[]);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [geometry]);

  if (!geometry) {
    return null;
  }

  const counts = geometry.plane_slice_counts ?? {};
  const activePlane = state.activePlane;
  const count = counts[activePlane] ?? 0;
  const index = clamp(state.sliceIndex[activePlane] ?? 0, 0, Math.max(count - 1, 0));

  return (
    <Panel title="Viewport">
      <div className="space-y-3">
        <div className="flex gap-1">
          {PLANES.map((plane) => (
            <Button
              key={plane}
              variant={plane === activePlane ? "primary" : "ghost"}
              className="flex-1"
              onClick={() => setPlane(plane)}
            >
              {titleCase(plane)}
            </Button>
          ))}
        </div>

        <div>
          <div className="mb-1 flex items-center justify-between">
            <span className="field-label">Slice</span>
            <span className="font-mono text-[10px] text-slate-500">
              {count ? index + 1 : 0} / {count}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={Math.max(count - 1, 0)}
            value={index}
            disabled={count === 0}
            className="w-full accent-accent"
            onChange={(event) => setSliceIndex(activePlane, Number(event.target.value))}
          />
        </div>

        <div>
          <div className="field-label mb-1">Window / level</div>
          <div className="flex flex-wrap gap-1">
            <Button
              variant={state.windowCenter == null ? "primary" : "ghost"}
              onClick={() => setWindow(null, null)}
            >
              Auto
            </Button>
            {presets.slice(0, 5).map((preset) => (
              <Button
                key={preset.key}
                variant={
                  state.windowCenter === preset.center && state.windowWidth === preset.width
                    ? "primary"
                    : "ghost"
                }
                title={`W ${preset.width} / L ${preset.center}`}
                onClick={() => setWindow(preset.center, preset.width)}
              >
                {preset.label}
              </Button>
            ))}
          </div>
          {state.windowCenter != null ? (
            <p className="mt-1 font-mono text-[10px] text-slate-500">
              W {state.windowWidth?.toFixed(0)} / L {state.windowCenter.toFixed(0)}{" "}
              {geometry.intensity_units}
            </p>
          ) : null}
        </div>

        <div className="flex flex-wrap gap-1.5">
          <Badge tone="neutral">{geometry.intensity_units}</Badge>
          {geometry.is_resampled ? <Badge tone="info">resampled</Badge> : null}
          {geometry.is_normalized ? <Badge tone="info">normalised</Badge> : null}
          <Badge tone="neutral">{geometry.orientation}</Badge>
        </div>
      </div>
    </Panel>
  );
}
