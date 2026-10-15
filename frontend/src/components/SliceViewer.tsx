// Interactive 2D slice viewer.
//
// Fetches the raw slice pixels from the backend, applies a real window/level,
// and (when a segmentation exists) overlays its mask. Rendered on a <canvas>.

import { useEffect, useMemo, useRef, useState } from "react";

import { api, ApiRequestError } from "../api/client";
import { chooseWindow, decodeMask, decodeSlice, overlayMask, windowToRgba } from "../lib/window";
import type { Plane, SliceResponse } from "../types";
import { Badge, Empty, Spinner } from "./ui";

export interface SliceViewerProps {
  volumeId: string | null;
  segmentationId: string | null;
  plane: Plane;
  index: number;
  center: number | null;
  width: number | null;
  onSliceMeta?: (meta: SliceResponse) => void;
}

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const value = hex.replace("#", "");
  const full = value.length === 3 ? value.split("").map((c) => c + c).join("") : value;
  const int = parseInt(full || "8ab4f8", 16);
  return { r: (int >> 16) & 255, g: (int >> 8) & 255, b: int & 255 };
}

export function SliceViewer({
  volumeId,
  segmentationId,
  plane,
  index,
  center,
  width,
  onSliceMeta,
}: SliceViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [meta, setMeta] = useState<SliceResponse | null>(null);
  const [maskColor, setMaskColor] = useState("#ffc45c");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const metaCallback = useRef(onSliceMeta);
  metaCallback.current = onSliceMeta;

  const window = useMemo(() => {
    if (!meta) return null;
    if (center != null && width != null) return { center, width };
    return chooseWindow(meta);
  }, [meta, center, width]);

  // Fetch the slice whenever the volume/plane/index/window changes.
  useEffect(() => {
    if (!volumeId) {
      setMeta(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getSlice(volumeId, plane, index, { center, width })
      .then((response) => {
        if (cancelled) return;
        setMeta(response);
        metaCallback.current?.(response);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message = err instanceof ApiRequestError ? err.message : "Slice could not be loaded.";
        setError(message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [volumeId, plane, index, center, width]);

  // Draw the slice (and optional mask overlay).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !meta || !window) return;

    let cancelled = false;

    const draw = async () => {
      const values = decodeSlice(meta.data, meta.dtype);
      let rgba = windowToRgba(values, window);

      if (segmentationId) {
        try {
          const mask = await api.getMaskSlice(segmentationId, plane, meta.index);
          if (cancelled) return;
          const decoded = decodeMask(mask.data, mask.rows, mask.columns);
          rgba = overlayMask(rgba, decoded, hexToRgb(maskColor));
        } catch {
          // A missing mask must not blank the anatomical image.
        }
      }

      if (cancelled) return;
      canvas.width = meta.columns;
      canvas.height = meta.rows;
      const context = canvas.getContext("2d");
      if (!context) return;
      const image = new ImageData(meta.columns, meta.rows);
      image.data.set(rgba);
      context.putImageData(image, 0, 0);
    };

    void draw();
    return () => {
      cancelled = true;
    };
  }, [meta, window, segmentationId, plane, maskColor]);

  useEffect(() => {
    if (segmentationId) {
      api
        .getSegmentation(segmentationId)
        .then((seg) => setMaskColor(seg.color || "#ffc45c"))
        .catch(() => undefined);
    }
  }, [segmentationId]);

  if (!volumeId) {
    return <Empty>Reconstruct a volume to open the multiplanar viewer.</Empty>;
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between px-1 pb-2">
        <div className="flex items-center gap-2">
          <Badge tone="accent">{plane}</Badge>
          {meta ? (
            <span className="font-mono text-[11px] text-slate-400">
              {meta.position_label} · slice {meta.index + 1}/{meta.count}
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          {loading ? <Spinner /> : null}
          {meta ? (
            <span className="font-mono text-[10px] text-slate-500">
              {meta.columns}×{meta.rows} @ {meta.column_spacing_mm.toFixed(2)}mm
            </span>
          ) : null}
        </div>
      </div>

      <div className="viewport-grid relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-md border border-ink-700/60 bg-black">
        {error ? (
          <div className="p-4 text-center text-xs text-severity-error">{error}</div>
        ) : (
          <canvas
            ref={canvasRef}
            data-testid="slice-canvas"
            className="max-h-full max-w-full"
            style={{ imageRendering: "pixelated", objectFit: "contain" }}
          />
        )}

        {meta ? (
          <>
            <span className="pointer-events-none absolute left-1/2 top-1 -translate-x-1/2 font-mono text-[10px] text-accent-soft">
              {meta.labels.top ?? ""}
            </span>
            <span className="pointer-events-none absolute bottom-1 left-1/2 -translate-x-1/2 font-mono text-[10px] text-accent-soft">
              {meta.labels.bottom ?? ""}
            </span>
            <span className="pointer-events-none absolute left-1 top-1/2 -translate-y-1/2 font-mono text-[10px] text-accent-soft">
              {meta.labels.left ?? ""}
            </span>
            <span className="pointer-events-none absolute right-1 top-1/2 -translate-y-1/2 font-mono text-[10px] text-accent-soft">
              {meta.labels.right ?? ""}
            </span>
          </>
        ) : null}
      </div>

      {meta ? (
        <div className="mt-2 flex items-center justify-between font-mono text-[10px] text-slate-500">
          <span>
            intensity {meta.intensity_min.toFixed(0)} … {meta.intensity_max.toFixed(0)}{" "}
            {meta.intensity_units}
          </span>
          {window ? (
            <span>
              W {window.center.toFixed(0)} / L {window.width.toFixed(0)}
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
