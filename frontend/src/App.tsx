import { useState } from "react";

import { ActionPanel } from "./components/ActionPanel";
import { DatasetPanel } from "./components/DatasetPanel";
import { ExportPanel } from "./components/ExportPanel";
import { InspectPanel } from "./components/InspectPanel";
import { LogConsole } from "./components/LogConsole";
import { NoticeStack } from "./components/NoticeStack";
import { RightPanel } from "./components/RightPanel";
import { SliceViewer } from "./components/SliceViewer";
import { StepRail } from "./components/StepRail";
import { TopStatusBar } from "./components/TopStatusBar";
import { ViewControls } from "./components/ViewControls";
import { Viewport3D } from "./components/Viewport3D";
import { Button, Empty } from "./components/ui";
import { useWorkflow } from "./state/WorkflowProvider";

function CenterViewport() {
  const { state } = useWorkflow();
  const [mode, setMode] = useState<"2d" | "3d">("2d");

  const hasVolume = state.volume != null;
  const hasMesh = state.mesh != null;
  const effectiveMode = mode === "3d" && !hasMesh ? "2d" : mode;

  return (
    <section className="panel h-full min-h-0">
      <header className="panel-title">
        <span>Viewer</span>
        <span className="flex items-center gap-1">
          <Button
            variant={effectiveMode === "2d" ? "primary" : "ghost"}
            onClick={() => setMode("2d")}
            disabled={!hasVolume}
          >
            Multiplanar
          </Button>
          <Button
            variant={effectiveMode === "3d" ? "primary" : "ghost"}
            onClick={() => setMode("3d")}
            disabled={!hasMesh}
            title={hasMesh ? "Show the generated surface" : "Generate a mesh first"}
          >
            3D surface
          </Button>
        </span>
      </header>
      <div className="min-h-0 flex-1 p-3">
        {effectiveMode === "3d" ? (
          <Viewport3D mesh={state.mesh} />
        ) : hasVolume ? (
          <SliceViewer
            volumeId={state.volume?.volume_id ?? null}
            segmentationId={state.segmentation?.segmentation_id ?? null}
            plane={state.activePlane}
            index={state.sliceIndex[state.activePlane] ?? 0}
            center={state.windowCenter}
            width={state.windowWidth}
          />
        ) : (
          <Empty>Upload a dataset, inspect it and reconstruct a volume to view anatomy.</Empty>
        )}
      </div>
    </section>
  );
}

export default function App() {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-ink-950">
      <TopStatusBar />

      <main className="grid min-h-0 flex-1 grid-cols-[280px_minmax(0,1fr)_320px] gap-3 p-3">
        {/* Left column */}
        <div className="flex min-h-0 flex-col gap-3 overflow-y-auto pr-1">
          <StepRail />
          <DatasetPanel />
          <InspectPanel />
          <ActionPanel />
          <ViewControls />
        </div>

        {/* Center column */}
        <div className="flex min-h-0 flex-col gap-3">
          <div className="min-h-0 flex-1">
            <CenterViewport />
          </div>
          <NoticeStack />
          <LogConsole />
        </div>

        {/* Right column */}
        <div className="flex min-h-0 flex-col gap-3 overflow-y-auto pr-1">
          <RightPanel />
          <ExportPanel />
        </div>
      </main>
    </div>
  );
}
