import { useEffect, useMemo, useRef, useState } from "react";

import { useWorkflow } from "../state/WorkflowProvider";
import { formatClock, logLevelClasses } from "../lib/format";
import { Badge, Button } from "./ui";

const LEVELS = ["all", "info", "warning", "error"] as const;
type FilterLevel = (typeof LEVELS)[number];

export function LogConsole() {
  const { state } = useWorkflow();
  const [filter, setFilter] = useState<FilterLevel>("all");
  const [collapsed, setCollapsed] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const logs = useMemo(() => {
    if (filter === "all") return state.logs;
    if (filter === "warning") {
      return state.logs.filter((entry) => entry.level === "warning" || entry.level === "warn");
    }
    return state.logs.filter((entry) => entry.level === filter);
  }, [state.logs, filter]);

  useEffect(() => {
    if (!collapsed) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [logs.length, collapsed]);

  return (
    <section className="panel">
      <header className="panel-title">
        <span>Operation log</span>
        <span className="flex items-center gap-1">
          {LEVELS.map((level) => (
            <Button
              key={level}
              variant={filter === level ? "primary" : "ghost"}
              onClick={() => setFilter(level)}
            >
              {level}
            </Button>
          ))}
          <Badge tone="neutral">{state.logs.length}</Badge>
          <Button onClick={() => setCollapsed((value) => !value)}>
            {collapsed ? "Show" : "Hide"}
          </Button>
        </span>
      </header>
      {!collapsed ? (
        <div className="h-40 overflow-auto p-2 font-mono text-[11px] leading-relaxed">
          {logs.length === 0 ? (
            <p className="text-slate-600">No log entries yet.</p>
          ) : (
            logs.map((entry, index) => (
              <div key={`${entry.time}-${index}`} className="flex gap-2">
                <span className="shrink-0 text-slate-600">{formatClock(entry.time)}</span>
                <span className={`shrink-0 uppercase ${logLevelClasses(entry.level)}`}>
                  {entry.level.padEnd(7, " ")}
                </span>
                {entry.stage ? (
                  <span className="shrink-0 text-slate-600">[{entry.stage}]</span>
                ) : null}
                <span className="text-slate-300">{entry.message}</span>
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      ) : null}
    </section>
  );
}
