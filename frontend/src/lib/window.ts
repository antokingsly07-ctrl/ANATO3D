// Slice decoding and windowing.
//
// The backend sends raw little-endian int16/float32 pixels as base64. These are
// pure, dependency-free helpers so the CT window/level maths is testable without
// a canvas or a DOM.

import type { SliceResponse } from "../types";

export interface WindowLevel {
  center: number;
  width: number;
}

/** Decode a base64 slice payload into a Float32Array of intensities. */
export function decodeSlice(base64: string, dtype: "int16" | "float32"): Float32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }

  if (dtype === "int16") {
    // Copy into an aligned buffer: the Uint8Array backing store may be offset.
    const aligned = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const source = new Int16Array(aligned);
    const out = new Float32Array(source.length);
    for (let i = 0; i < source.length; i += 1) out[i] = source[i];
    return out;
  }

  const aligned = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(aligned);
}

/** Decode a bit-packed boolean mask slice (np.packbits) back to bytes. */
export function decodeMask(base64: string, rows: number, columns: number): Uint8Array {
  const binary = atob(base64);
  const packed = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) packed[i] = binary.charCodeAt(i);

  const total = rows * columns;
  const out = new Uint8Array(total);
  for (let i = 0; i < total; i += 1) {
    const byte = packed[i >> 3] ?? 0;
    out[i] = (byte >> (7 - (i & 7))) & 1;
  }
  return out;
}

/**
 * Choose a sensible display window.
 *
 * Prefers an explicit CT window; otherwise derives one robustly from the slice's
 * own min/max using the 1st/99th percentile of the visible pixel values (so a
 * few hot voxels do not wash out the image).
 */
export function chooseWindow(response: Pick<SliceResponse, "window_center" | "window_width" | "intensity_min" | "intensity_max">): WindowLevel {
  if (response.window_center != null && response.window_width != null && response.window_width > 0) {
    return { center: response.window_center, width: response.window_width };
  }
  const low = response.intensity_min;
  const high = response.intensity_max;
  if (!Number.isFinite(low) || !Number.isFinite(high) || high - low < 1e-6) {
    return { center: 0, width: 1 };
  }
  return { center: (low + high) / 2, width: high - low };
}

/**
 * Apply a window/level and return an RGBA buffer ready for `ImageData`.
 *
 * Uses the standard linear ramp: values at or below `center - width/2` map to
 * black, values at or above `center + width/2` map to white.
 */
export function windowToRgba(values: Float32Array, window: WindowLevel): Uint8ClampedArray {
  const { center, width } = window;
  const safeWidth = width > 1e-6 ? width : 1;
  const low = center - safeWidth / 2;
  const scale = 255 / safeWidth;

  const out = new Uint8ClampedArray(values.length * 4);
  for (let i = 0; i < values.length; i += 1) {
    const mapped = (values[i] - low) * scale;
    const offset = i * 4;
    out[offset] = mapped;
    out[offset + 1] = mapped;
    out[offset + 2] = mapped;
    out[offset + 3] = 255;
  }
  return out;
}

/** Overlay a boolean mask on an RGBA buffer in a distinct colour. */
export function overlayMask(
  rgba: Uint8ClampedArray,
  mask: Uint8Array,
  color: { r: number; g: number; b: number } = { r: 255, g: 196, b: 92 },
  alpha = 0.45,
): Uint8ClampedArray {
  for (let i = 0; i < mask.length; i += 1) {
    if (!mask[i]) continue;
    const offset = i * 4;
    rgba[offset] = rgba[offset] * (1 - alpha) + color.r * alpha;
    rgba[offset + 1] = rgba[offset + 1] * (1 - alpha) + color.g * alpha;
    rgba[offset + 2] = rgba[offset + 2] * (1 - alpha) + color.b * alpha;
  }
  return rgba;
}
