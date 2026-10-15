import { CheckCircle2, Circle, Lock } from "lucide-react";

import { useWorkflow } from "../state/WorkflowProvider";
import { STEPS } from "../state/workflow";

export function StepRail() {
  const { state, stepStatus, setStep } = useWorkflow();

  return (
    <nav aria-label="Workflow steps" className="flex flex-col gap-1">
      {STEPS.map((step, index) => {
        const status = stepStatus(step.key);
        const active = state.step === step.key;
        const locked = status === "locked";
        return (
          <button
            key={step.key}
            type="button"
            disabled={locked}
            aria-current={active ? "step" : undefined}
            onClick={() => setStep(step.key)}
            className={`flex items-start gap-2 rounded-md border px-2.5 py-2 text-left transition-colors ${
              active
                ? "border-accent/50 bg-accent/10"
                : "border-transparent hover:border-ink-600/60 hover:bg-ink-800/60"
            } ${locked ? "cursor-not-allowed opacity-40" : ""}`}
          >
            <span className="mt-0.5">
              {status === "complete" ? (
                <CheckCircle2 size={15} className="text-severity-ok" />
              ) : locked ? (
                <Lock size={14} className="text-slate-600" />
              ) : (
                <Circle size={14} className={active ? "text-accent" : "text-slate-500"} />
              )}
            </span>
            <span className="min-w-0">
              <span className="flex items-center gap-2">
                <span className="font-mono text-[10px] text-slate-500">{index + 1}</span>
                <span
                  className={`text-xs font-semibold ${active ? "text-accent-soft" : "text-slate-200"}`}
                >
                  {step.label}
                </span>
              </span>
              <span className="mt-0.5 block text-[10px] leading-snug text-slate-500">
                {step.blurb}
              </span>
            </span>
          </button>
        );
      })}
    </nav>
  );
}
