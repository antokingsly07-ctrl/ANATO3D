import { useState } from "react";
import { X } from "lucide-react";

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
import { useMediaQuery } from "./lib/useMediaQuery";
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

function LeftPanels() {
  return (
    <>
      <StepRail />
      <DatasetPanel />
      <InspectPanel />
      <ActionPanel />
      <ViewControls />
    </>
  );
}

function RightPanels() {
  return (
    <>
      <RightPanel />
      <ExportPanel />
    </>
  );
}

function PanelsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label="Panels">
      <button
        type="button"
        aria-label="Close panels"
        className="absolute inset-0 bg-black/60"
        onClick={onClose}
      />
      <aside className="absolute inset-y-0 right-0 flex w-[92%] max-w-md flex-col border-l border-ink-700 bg-ink-950 shadow-2xl">
        <header className="flex items-center justify-between border-b border-ink-700 px-3 py-2">
          <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
            Panels
          </span>
          <Button onClick={onClose} title="Close panels">
            <X size={14} />
          </Button>
        </header>
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
          <LeftPanels />
          <RightPanels />
          <LogConsole />
        </div>
      </aside>
    </div>
  );
}

export default function App() {
  const isDesktop = useMediaQuery("(min-width: 1024px)");
  const [panelsOpen, setPanelsOpen] = useState(false);

  if (!isDesktop) {
    return (
      <div className="flex h-[100dvh] flex-col overflow-hidden bg-ink-950">
        <TopStatusBar onTogglePanels={() => setPanelsOpen((value) => !value)} />

        <main className="flex min-h-0 flex-1 flex-col gap-3 p-3">
          <div className="min-h-0 flex-1">
            <CenterViewport />
          </div>
          <NoticeStack />
        </main>

        <PanelsDrawer open={panelsOpen} onClose={() => setPanelsOpen(false)} />
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-ink-950">
      <TopStatusBar />

      <main className="grid min-h-0 flex-1 grid-cols-[280px_minmax(0,1fr)_320px] gap-3 p-3">
        {/* Left column */}
        <div className="flex min-h-0 flex-col gap-3 overflow-y-auto pr-1">
          <LeftPanels />
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
          <RightPanels />
        </div>
      </main>
    </div>
  );
}
