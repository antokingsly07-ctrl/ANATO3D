import { Activity, AlertTriangle, Cpu, RefreshCw } from "lucide-react";

import { useWorkflow } from "../state/WorkflowProvider";
import { Badge, Button, Spinner } from "./ui";

export function TopStatusBar() {
  const { state, findings, refreshHealth } = useWorkflow();
  const health = state.health;

  return (
    <header className="flex items-center justify-between gap-4 border-b border-ink-700/70 bg-ink-900/90 px-4 py-2.5">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="grid h-7 w-7 place-items-center rounded-md border border-accent/40 bg-accent/10 text-accent">
            <Activity size={15} />
          </span>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-wide text-slate-100">
              ANATO3D
              <span className="ml-2 text-[10px] font-normal uppercase tracking-[0.2em] text-slate-500">
                reconstruction workbench
              </span>
            </div>
            <div className="text-[10px] text-slate-500">
              Patient-specific 3D anatomical reconstruction · research prototype
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {state.running ? (
          <div className="flex items-center gap-2">
            <Spinner />
            <span className="max-w-[240px] truncate text-[11px] text-accent-soft">
              {state.stage || "Working…"}
            </span>
            <div className="h-1.5 w-32 overflow-hidden rounded-full bg-ink-700">
              <div
                className="h-full bg-accent transition-[width] duration-200"
                style={{ width: `${Math.round((state.progress ?? 0) * 100)}%` }}
              />
            </div>
            <span className="font-mono text-[10px] text-slate-500">
              {Math.round((state.progress ?? 0) * 100)}%
            </span>
          </div>
        ) : null}

        {findings.errors > 0 ? (
          <Badge tone="error">
            <AlertTriangle size={11} /> {findings.errors} error{findings.errors === 1 ? "" : "s"}
          </Badge>
        ) : null}
        {findings.warnings > 0 ? (
          <Badge tone="warning">
            <AlertTriangle size={11} /> {findings.warnings} warning
            {findings.warnings === 1 ? "" : "s"}
          </Badge>
        ) : null}

        {state.dataset ? <Badge tone="neutral">{state.dataset.dataset_id}</Badge> : null}

        <Badge tone={health ? "ok" : "error"}>
          <Cpu size={11} /> {health ? "backend online" : "backend offline"}
        </Badge>
        {health ? (
          <Badge tone={health.ai_segmentation_available ? "accent" : "neutral"}>
            AI {health.ai_segmentation_available ? "ready" : "off"}
          </Badge>
        ) : null}

        <Button onClick={() => void refreshHealth()} title="Re-check backend health">
          <RefreshCw size={12} />
        </Button>
      </div>
    </header>
  );
}
