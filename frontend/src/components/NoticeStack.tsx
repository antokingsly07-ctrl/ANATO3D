import { AlertTriangle, Info, X } from "lucide-react";

import { useWorkflow } from "../state/WorkflowProvider";

export function NoticeStack() {
  const { state, clearNotices } = useWorkflow();
  if (!state.notices.length) return null;

  return (
    <div className="flex flex-col gap-1.5">
      {state.notices.slice(-3).map((notice, index) => (
        <div
          key={`${notice.message}-${index}`}
          className={`flex items-start justify-between gap-3 rounded-md border px-3 py-2 text-xs ${
            notice.severity === "error"
              ? "border-severity-error/40 bg-severity-error/10 text-severity-error"
              : notice.severity === "warning"
                ? "border-severity-warning/40 bg-severity-warning/10 text-severity-warning"
                : "border-severity-info/40 bg-severity-info/10 text-severity-info"
          }`}
          role="alert"
        >
          <span className="flex items-start gap-2">
            {notice.severity === "info" ? <Info size={14} /> : <AlertTriangle size={14} />}
            <span>{notice.message}</span>
          </span>
        </div>
      ))}
      <button
        type="button"
        className="btn-ghost self-start"
        onClick={clearNotices}
        aria-label="Dismiss notices"
      >
        <X size={12} /> Dismiss
      </button>
    </div>
  );
}
