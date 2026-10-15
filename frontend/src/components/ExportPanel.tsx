import { useEffect } from "react";
import { Download, RefreshCw } from "lucide-react";

import { apiUrl } from "../api/client";
import { useWorkflow } from "../state/WorkflowProvider";
import { formatBytes } from "../lib/format";
import { Badge, Button, Empty, Panel, Spinner } from "./ui";

export function ExportPanel() {
  const { state, loadManifest } = useWorkflow();

  // Refresh the manifest whenever the artefact set changes.
  useEffect(() => {
    if (state.dataset) void loadManifest();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    state.dataset?.dataset_id,
    state.volume?.volume_id,
    state.segmentation?.segmentation_id,
    state.mesh?.mesh_id,
    state.validation?.validation_id,
  ]);

  const manifest = state.manifest;
  const largestBytes = manifest
    ? Math.max(0, ...manifest.items.map((item) => item.size_bytes ?? 0))
    : 0;

  return (
    <Panel
      title="7 · Export"
      actions={
        <Button onClick={() => void loadManifest()} title="Refresh export list">
          <RefreshCw size={12} />
        </Button>
      }
    >
      {!manifest ? (
        <Empty>{state.running ? <Spinner /> : "Exports appear once data is available."}</Empty>
      ) : (
        <div className="space-y-2">
          <ul className="space-y-1.5">
            {manifest.items.map((item) => (
              <li
                key={item.key}
                className="flex items-center justify-between gap-2 rounded-md border border-ink-700/60 bg-ink-850/50 px-2 py-1.5"
              >
                <div className="min-w-0">
                  <div className="truncate text-xs text-slate-200">{item.label}</div>
                  <div className="truncate text-[10px] text-slate-500">{item.description}</div>
                </div>
                {item.available && item.url ? (
                  <a
                    href={apiUrl(item.url)}
                    className="btn-ghost shrink-0"
                    download
                  >
                    <Download size={12} /> Download
                  </a>
                ) : (
                  <Badge tone="neutral" className="shrink-0">
                    n/a
                  </Badge>
                )}
              </li>
            ))}
          </ul>
          <p className="text-[10px] text-slate-500">{manifest.privacy_note}</p>
          {largestBytes > 0 ? (
            <p className="font-mono text-[10px] text-slate-500">
              Largest reported artefact: {formatBytes(largestBytes)}
            </p>
          ) : null}
        </div>
      )}
    </Panel>
  );
}
