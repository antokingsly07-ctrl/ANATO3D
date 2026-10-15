import { formatMm3ToCm3, formatNumber, severityClasses } from "../lib/format";
import { useWorkflow } from "../state/WorkflowProvider";
import { Badge, Empty, KeyValue, Panel, Stat } from "./ui";

function Structures() {
  const { state } = useWorkflow();
  const segmentation = state.segmentation;
  const mesh = state.mesh;

  return (
    <Panel title="Structures">
      {!segmentation ? (
        <Empty>No segmentation yet.</Empty>
      ) : (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span
              className="inline-block h-3 w-3 rounded-sm border border-white/20"
              style={{ backgroundColor: segmentation.color }}
            />
            <span className="text-sm font-medium text-slate-100">{segmentation.label}</span>
            <Badge tone="accent">{segmentation.engine}</Badge>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Voxels" value={segmentation.statistics.voxel_count.toLocaleString()} mono />
            <Stat label="Volume" value={formatMm3ToCm3(segmentation.statistics.volume_mm3)} mono />
          </div>
          <KeyValue
            items={[
              ["Components", segmentation.statistics.components],
              ["Engine time", `${segmentation.duration_s.toFixed(2)} s`],
              ["Mesh triangles", mesh ? mesh.face_count.toLocaleString() : "—"],
            ]}
          />
          {segmentation.notes.length ? (
            <ul className="list-inside list-disc space-y-0.5 text-[10px] text-slate-500">
              {segmentation.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          ) : null}
        </div>
      )}
    </Panel>
  );
}

function Measurements() {
  const { state } = useWorkflow();
  const analysis = state.analysis;

  return (
    <Panel title="Measurements">
      {!analysis ? (
        <Empty>Measurements appear after a mesh is generated.</Empty>
      ) : (
        <div className="space-y-2">
          {analysis.measurement_summary.length ? (
            <table className="w-full border-collapse text-[11px]">
              <tbody>
                {analysis.measurement_summary.map((entry) => (
                  <tr key={entry.key} className="border-b border-ink-800/60">
                    <td className="py-1 pr-2 text-slate-400">{entry.label}</td>
                    <td className="py-1 text-right font-mono text-slate-100">
                      {formatNumber(entry.value)} {entry.unit}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty>No summary measurements.</Empty>
          )}

          {analysis.intensity && Object.keys(analysis.intensity).length ? (
            <div>
              <div className="field-label mb-1">Intensity in structure</div>
              <KeyValue
                items={[
                  ["Mean", formatNumber(analysis.intensity.mean)],
                  ["Std", formatNumber(analysis.intensity.std)],
                  ["Min", formatNumber(analysis.intensity.min)],
                  ["Max", formatNumber(analysis.intensity.max)],
                ]}
              />
            </div>
          ) : null}
          <p className="text-[10px] text-slate-500">{analysis.disclaimer}</p>
        </div>
      )}
    </Panel>
  );
}

function Validation() {
  const { state } = useWorkflow();
  const validation = state.validation;

  return (
    <Panel
      title="Validation"
      actions={
        validation ? (
          <Badge tone={validation.ground_truth.state === "computed" ? "ok" : "warning"}>
            {validation.ground_truth.state}
          </Badge>
        ) : null
      }
    >
      {!validation ? (
        <Empty>Run validation to see consistency checks and accuracy metrics.</Empty>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-1.5">
            <Badge tone={validation.summary.errors ? "error" : "neutral"}>
              {validation.summary.errors} errors
            </Badge>
            <Badge tone={validation.summary.warnings ? "warning" : "neutral"}>
              {validation.summary.warnings} warnings
            </Badge>
            <Badge tone="neutral">{validation.summary.info} info</Badge>
          </div>

          <div
            className={`rounded-md border px-2 py-1.5 text-[11px] ${
              validation.ground_truth.state === "computed"
                ? "border-severity-ok/30 bg-severity-ok/5 text-severity-ok"
                : "border-severity-warning/30 bg-severity-warning/5 text-severity-warning"
            }`}
          >
            {validation.ground_truth.message}
          </div>

          {validation.ground_truth.state === "computed" && validation.ground_truth.metrics ? (
            <table className="w-full border-collapse text-[11px]">
              <tbody>
                {validation.ground_truth.metrics.map((metric) => (
                  <tr key={metric.key} className="border-b border-ink-800/60">
                    <td className="py-1 pr-2 text-slate-400">{metric.label}</td>
                    <td className="py-1 text-right font-mono text-slate-100">
                      {formatNumber(metric.value)} {metric.unit}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}

          {validation.interpretation_notes.length ? (
            <ul className="list-inside list-disc space-y-0.5 text-[10px] text-slate-500">
              {validation.interpretation_notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          ) : null}

          <ul className="space-y-1.5">
            {validation.findings.map((finding, index) => (
              <li
                key={`${finding.code}-${index}`}
                className={`rounded-md border px-2 py-1.5 text-[11px] ${severityClasses(finding.severity)}`}
              >
                <div className="font-semibold">{finding.title}</div>
                <p className="text-slate-300/90">{finding.message}</p>
              </li>
            ))}
          </ul>

          <p className="text-[10px] text-slate-500">{validation.disclaimer}</p>
        </div>
      )}
    </Panel>
  );
}

export function RightPanel() {
  return (
    <div className="flex flex-col gap-3">
      <Structures />
      <Measurements />
      <Validation />
    </div>
  );
}
