// Presentation helpers. Pure functions so they are trivially unit-testable.

import type { Severity } from "../types";

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "-";
  if (!Number.isFinite(value)) return value > 0 ? "\u221e" : "-\u221e";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return value.toExponential(2);
  return value.toFixed(digits);
}

export function formatMm3ToCm3(mm3: number | null | undefined, digits = 3): string {
  if (mm3 == null || Number.isNaN(mm3)) return "-";
  return `${(mm3 / 1000).toFixed(digits)} cm\u00b3`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || Number.isNaN(bytes)) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || Number.isNaN(seconds)) return "-";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(2)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes} min ${rest} s`;
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value == null || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatDimensions(dimensions: number[] | null | undefined): string {
  if (!dimensions || dimensions.length < 3) return "-";
  return `${dimensions[0]} \u00d7 ${dimensions[1]} \u00d7 ${dimensions[2]}`;
}

export function formatSpacing(spacing: number[] | null | undefined): string {
  if (!spacing || spacing.length < 3) return "-";
  return spacing.map((value) => `${value.toFixed(2)}`).join(" \u00d7 ") + " mm";
}

export function severityRank(severity: Severity): number {
  return severity === "error" ? 0 : severity === "warning" ? 1 : 2;
}

export function severityClasses(severity: Severity): string {
  switch (severity) {
    case "error":
      return "border-severity-error/40 bg-severity-error/10 text-severity-error";
    case "warning":
      return "border-severity-warning/40 bg-severity-warning/10 text-severity-warning";
    default:
      return "border-severity-info/40 bg-severity-info/10 text-severity-info";
  }
}

export function summaryClasses(status: string): string {
  switch (status) {
    case "error":
      return "text-severity-error";
    case "warning":
      return "text-severity-warning";
    default:
      return "text-severity-ok";
  }
}

export function logLevelClasses(level: string): string {
  switch (level.toLowerCase()) {
    case "error":
      return "text-severity-error";
    case "warning":
    case "warn":
      return "text-severity-warning";
    case "debug":
      return "text-ink-500";
    default:
      return "text-accent-soft";
  }
}

export function clamp(value: number, minimum: number, maximum: number): number {
  if (maximum < minimum) return minimum;
  return Math.min(Math.max(value, minimum), maximum);
}

export function titleCase(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function formatClock(epochSeconds: number): string {
  const date = new Date(epochSeconds * 1000);
  return date.toLocaleTimeString(undefined, { hour12: false });
}
