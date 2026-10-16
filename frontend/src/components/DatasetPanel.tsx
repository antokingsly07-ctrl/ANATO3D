import { useRef, useState } from "react";
import { FlaskConical, Upload } from "lucide-react";

import { useWorkflow } from "../state/WorkflowProvider";
import { formatBytes } from "../lib/format";
import { Badge, Button, Field, KeyValue, Panel, Spinner, Stat } from "./ui";

export function DatasetPanel() {
  const { state, uploadFiles, generateSynthetic } = useWorkflow();
  const fileInput = useRef<HTMLInputElement | null>(null);
  const [showSynthetic, setShowSynthetic] = useState(false);
  const [rows, setRows] = useState(256);
  const [columns, setColumns] = useState(256);
  const [slices, setSlices] = useState(96);
  const [irregular, setIrregular] = useState(false);
  const [shuffle, setShuffle] = useState(true);

  const dataset = state.dataset;

  return (
    <Panel
      title="1 · Dataset"
      actions={
        dataset ? (
          <Badge tone={dataset.is_synthetic ? "warning" : "accent"}>
            {dataset.is_synthetic ? "synthetic" : "dicom"}
          </Badge>
        ) : null
      }
    >
      <div className="space-y-3">
        <div className="flex gap-2">
          <Button
            variant="primary"
            className="flex-1"
            disabled={state.running}
            onClick={() => fileInput.current?.click()}
          >
            <Upload size={13} /> Upload DICOM / NIfTI
          </Button>
          <Button
            className="flex-1"
            disabled={state.running}
            onClick={() => setShowSynthetic((value) => !value)}
          >
            <FlaskConical size={13} /> Synthetic
          </Button>
        </div>

        <input
          ref={fileInput}
          type="file"
          multiple
          className="hidden"
          data-testid="dicom-file-input"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            if (files.length) void uploadFiles(files);
            event.target.value = "";
          }}
        />

        {showSynthetic ? (
          <div className="space-y-2 rounded-md border border-severity-warning/30 bg-severity-warning/5 p-2.5">
            <p className="text-[10px] leading-relaxed text-severity-warning">
              Generates a clearly-labelled <strong>synthetic</strong> CT phantom with a known
              ground-truth mask. It is not a real patient and only exists to exercise the pipeline
              without uploading data.
            </p>
            <div className="grid grid-cols-3 gap-2">
              <Field label="Rows">
                <input
                  className="field-input"
                  type="number"
                  min={16}
                  max={512}
                  value={rows}
                  onChange={(event) => setRows(Number(event.target.value))}
                />
              </Field>
              <Field label="Columns">
                <input
                  className="field-input"
                  type="number"
                  min={16}
                  max={512}
                  value={columns}
                  onChange={(event) => setColumns(Number(event.target.value))}
                />
              </Field>
              <Field label="Slices">
                <input
                  className="field-input"
                  type="number"
                  min={4}
                  max={512}
                  value={slices}
                  onChange={(event) => setSlices(Number(event.target.value))}
                />
              </Field>
            </div>
            <label className="flex items-center gap-2 text-[11px] text-slate-400">
              <input
                type="checkbox"
                checked={shuffle}
                onChange={(event) => setShuffle(event.target.checked)}
              />
              Shuffle filenames (proves ordering ignores filenames)
            </label>
            <label className="flex items-center gap-2 text-[11px] text-slate-400">
              <input
                type="checkbox"
                checked={irregular}
                onChange={(event) => setIrregular(event.target.checked)}
              />
              Add realistic slice-spacing jitter
            </label>
            <Button
              variant="primary"
              className="w-full"
              disabled={state.running}
              onClick={() =>
                void generateSynthetic({
                  rows,
                  columns,
                  slices,
                  irregular_spacing: irregular,
                  shuffle_filenames: shuffle,
                })
              }
            >
              {state.running ? <Spinner /> : null} Generate phantom
            </Button>
          </div>
        ) : null}

        {dataset ? (
          <div className="space-y-2">
            <div className="grid grid-cols-3 gap-2">
              <Stat label="DICOM" value={dataset.dicom_file_count} mono />
              <Stat label="Other" value={dataset.non_image_file_count} mono />
              <Stat label="Unreadable" value={dataset.unreadable_file_count} mono />
            </div>
            <KeyValue
              items={[
                ["Dataset", <span key="id">{dataset.dataset_id}</span>],
                ["Total size", formatBytes(dataset.total_bytes)],
              ]}
            />
            {dataset.synthetic_notes.length ? (
              <div className="rounded-md border border-severity-warning/30 bg-severity-warning/5 p-2 text-[10px] text-severity-warning">
                {dataset.synthetic_notes.join(" ")}
              </div>
            ) : null}
          </div>
        ) : (
          <p className="text-[11px] text-slate-500">
            Upload a DICOM series or a NIfTI volume (.nii/.nii.gz, converted to DICOM), or generate a
            synthetic phantom to begin. Nothing leaves this machine.
          </p>
        )}
      </div>
    </Panel>
  );
}
