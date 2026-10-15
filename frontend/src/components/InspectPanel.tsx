import { ScanSearch } from "lucide-react";

import { useWorkflow } from "../state/WorkflowProvider";
import { formatNumber, severityClasses, summaryClasses } from "../lib/format";
import type { Finding } from "../types";
import { Badge, Button, Empty, Panel, Spinner } from "./ui";

function FindingRow({ finding }: { finding: Finding }) {
  return (
    <li className={`rounded-md border px-2 py-1.5 text-[11px] ${severityClasses(finding.severity)}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="font-semibold">{finding.title}</span>
        <span className="uppercase tracking-wide opacity-80">{finding.severity}</span>
      </div>
      <p className="mt-0.5 text-slate-300/90">{finding.message}</p>
    </li>
  );
}

export function InspectPanel() {
  const { state, analyze } = useWorkflow();
  const analysis = state.analyze;

  return (
    <Panel
      title="2 · Inspect"
      actions={
        analysis ? (
          <span className={`text-[10px] font-semibold uppercase ${summaryClasses(analysis.summary.status)}`}>
            {analysis.summary.status}
          </span>
        ) : null
      }
    >
      <div className="space-y-3">
        <Button
          variant={state.dataset && !analysis ? "primary" : "ghost"}
          className="w-full"
          disabled={!state.dataset || state.running}
          onClick={() => void analyze()}
        >
          {state.running ? <Spinner /> : <ScanSearch size={13} />}
          {analysis ? "Re-analyse dataset" : "Analyse dataset"}
        </Button>

        {!analysis ? (
          <p className="text-[11px] text-slate-500">
            Scanning reads DICOM headers only (no pixels) to list studies, series and the
            patient-space slice ordering.
          </p>
        ) : (
          <>
            <div className="flex flex-wrap gap-1.5">
              <Badge tone="info">{analysis.series.length} series</Badge>
              <Badge tone="neutral">{analysis.studies.length} studies</Badge>
              {analysis.summary.errors > 0 ? (
                <Badge tone="error">{analysis.summary.errors} errors</Badge>
              ) : null}
              {analysis.summary.warnings > 0 ? (
                <Badge tone="warning">{analysis.summary.warnings} warnings</Badge>
              ) : null}
              <Badge tone="neutral">{analysis.elapsed_s.toFixed(2)} s</Badge>
            </div>

            <div className="space-y-2">
              {analysis.series.map((series) => (
                <div
                  key={series.series_instance_uid}
                  className="rounded-md border border-ink-700/60 bg-ink-850/50 p-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-xs text-slate-200">
                      {series.series_description ?? series.series_instance_uid.slice(-12)}
                    </span>
                    <Badge tone={series.is_reconstructable ? "ok" : "warning"}>
                      {series.modality}
                    </Badge>
                  </div>
                  <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[10px] text-slate-500">
                    <span>{series.slice_count} slices</span>
                    <span>{series.orientation ?? "orientation ?"}</span>
                    <span>
                      {series.rows ?? "?"}×{series.columns ?? "?"}
                    </span>
                    <span>
                      {series.slice_spacing_mm != null
                        ? `${series.slice_spacing_mm.toFixed(2)} mm`
                        : "spacing ?"}
                    </span>
                  </div>
                  {series.irregular_spacing ? (
                    <div className="mt-1">
                      <Badge tone="warning">irregular spacing</Badge>
                    </div>
                  ) : null}
                  {series.issues.length ? (
                    <ul className="mt-1 list-inside list-disc text-[10px] text-severity-warning">
                      {series.issues.map((issue) => (
                        <li key={issue}>{issue}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              ))}
            </div>

            {analysis.ordering_preview.length ? (
              <div>
                <div className="field-label mb-1">
                  Patient-space ordering (first {Math.min(analysis.ordering_preview.length, 6)})
                </div>
                <div className="max-h-40 overflow-auto rounded-md border border-ink-700/60">
                  <table className="w-full border-collapse font-mono text-[10px]">
                    <thead>
                      <tr className="bg-ink-800/70 text-slate-500">
                        <th className="px-2 py-1 text-left font-medium">#</th>
                        <th className="px-2 py-1 text-left font-medium">Instance</th>
                        <th className="px-2 py-1 text-right font-medium">Projection (mm)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.ordering_preview.slice(0, 8).map((entry) => (
                        <tr key={entry.index} className="border-t border-ink-800/60">
                          <td className="px-2 py-0.5 text-slate-400">{entry.index}</td>
                          <td className="px-2 py-0.5 text-slate-400">{entry.instance_number ?? "-"}</td>
                          <td className="px-2 py-0.5 text-right text-slate-300">
                            {entry.projection_mm != null ? formatNumber(entry.projection_mm, 3) : "-"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-1 text-[10px] text-slate-500">
                  Sorted purely by projection onto the series normal — never by filename or
                  InstanceNumber.
                </p>
              </div>
            ) : null}

            {analysis.findings.length ? (
              <ul className="space-y-1.5">
                {analysis.findings.map((finding, index) => (
                  <FindingRow key={`${finding.code}-${index}`} finding={finding} />
                ))}
              </ul>
            ) : (
              <Empty>No structural issues detected during the header scan.</Empty>
            )}
          </>
        )}
      </div>
    </Panel>
  );
}
