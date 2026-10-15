// Small presentational primitives shared across panels.

import type { ReactNode } from "react";

export function Panel({
  title,
  actions,
  children,
  className = "",
  bodyClassName = "",
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {title ? (
        <header className="panel-title">
          <span>{title}</span>
          {actions ? <span className="flex items-center gap-1">{actions}</span> : null}
        </header>
      ) : null}
      <div className={`min-h-0 flex-1 overflow-auto p-3 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "ghost",
  type = "button",
  title,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost" | "danger";
  type?: "button" | "submit";
  title?: string;
  className?: string;
}) {
  const variantClass =
    variant === "primary" ? "btn-primary" : variant === "danger" ? "btn-danger" : "btn-ghost";
  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`${variantClass} ${className}`}
    >
      {children}
    </button>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className = "",
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "error" | "warning" | "info" | "ok";
  className?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "border-ink-600 bg-ink-800 text-slate-400",
    accent: "border-accent/40 bg-accent/10 text-accent-soft",
    error: "border-severity-error/40 bg-severity-error/10 text-severity-error",
    warning: "border-severity-warning/40 bg-severity-warning/10 text-severity-warning",
    info: "border-severity-info/40 bg-severity-info/10 text-severity-info",
    ok: "border-severity-ok/40 bg-severity-ok/10 text-severity-ok",
  };
  return <span className={`chip ${tones[tone]} ${className}`}>{children}</span>;
}

export function Stat({
  label,
  value,
  hint,
  mono = false,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="rounded-md border border-ink-700/60 bg-ink-850/60 px-2 py-1.5">
      <div className="field-label">{label}</div>
      <div className={`mt-0.5 text-sm text-slate-100 ${mono ? "font-mono" : ""}`}>{value}</div>
      {hint ? <div className="mt-0.5 text-[10px] text-slate-500">{hint}</div> : null}
    </div>
  );
}

export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="block">
      <span className="field-label">{label}</span>
      <div className="mt-1">{children}</div>
      {hint ? <span className="mt-1 block text-[10px] text-slate-500">{hint}</span> : null}
    </label>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-accent/30 border-t-accent ${className}`}
    />
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full items-center justify-center px-4 text-center text-xs text-slate-500">
      {children}
    </div>
  );
}

export function KeyValue({ items }: { items: Array<[string, ReactNode]> }) {
  return (
    <dl className="space-y-1">
      {items.map(([key, value]) => (
        <div key={key} className="flex items-baseline justify-between gap-3 text-xs">
          <dt className="text-slate-500">{key}</dt>
          <dd className="text-right font-mono text-[11px] text-slate-200">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
