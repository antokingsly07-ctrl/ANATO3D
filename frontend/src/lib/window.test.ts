import { describe, expect, it } from "vitest";

import {
  chooseWindow,
  decodeMask,
  decodeSlice,
  overlayMask,
  windowToRgba,
} from "./window";

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function bytesOfInt16(values: number[]): Uint8Array {
  const typed = Int16Array.from(values);
  return new Uint8Array(typed.buffer);
}

function bytesOfFloat32(values: number[]): Uint8Array {
  const typed = Float32Array.from(values);
  return new Uint8Array(typed.buffer);
}

describe("decodeSlice", () => {
  it("round-trips signed 16-bit intensities including negatives", () => {
    const values = [-1000, 0, 45, 1250, -32768, 32767];
    const decoded = decodeSlice(toBase64(bytesOfInt16(values)), "int16");
    expect(Array.from(decoded)).toEqual(values);
  });

  it("round-trips float32 intensities", () => {
    const values = [1.5, -2.25, 0, 1024.75];
    const decoded = decodeSlice(toBase64(bytesOfFloat32(values)), "float32");
    expect(decoded).toBeInstanceOf(Float32Array);
    expect(Array.from(decoded)).toEqual(values);
  });

  it("produces one value per two bytes for int16", () => {
    const decoded = decodeSlice(toBase64(bytesOfInt16([1, 2, 3, 4])), "int16");
    expect(decoded.length).toBe(4);
  });
});

describe("decodeMask", () => {
  /** Pack bits MSB-first, matching numpy.packbits (the backend's encoding). */
  function pack(bits: number[]): string {
    const packed = new Uint8Array(Math.ceil(bits.length / 8));
    bits.forEach((bit, index) => {
      if (bit) packed[index >> 3] |= 1 << (7 - (index & 7));
    });
    return toBase64(packed);
  }

  it("unpacks MSB-first bits into a 0/1 byte mask", () => {
    const bits = [1, 0, 0, 0, 0, 0, 0, 1, 1, 1];
    const decoded = decodeMask(pack(bits), 2, 5);
    expect(Array.from(decoded)).toEqual(bits);
  });

  it("sizes the output to rows*columns", () => {
    const decoded = decodeMask(pack(new Array(32).fill(1)), 4, 8);
    expect(decoded.length).toBe(32);
    expect(Array.from(decoded).every((value) => value === 1)).toBe(true);
  });
});

describe("chooseWindow", () => {
  it("prefers an explicit CT window when the backend supplies one", () => {
    const window = chooseWindow({
      window_center: 40,
      window_width: 400,
      intensity_min: -1000,
      intensity_max: 3000,
    });
    expect(window).toEqual({ center: 40, width: 400 });
  });

  it("falls back to the slice's own range", () => {
    const window = chooseWindow({
      window_center: null,
      window_width: null,
      intensity_min: -1000,
      intensity_max: 1000,
    });
    expect(window).toEqual({ center: 0, width: 2000 });
  });

  it("never returns a zero or negative width", () => {
    const window = chooseWindow({
      window_center: null,
      window_width: null,
      intensity_min: 50,
      intensity_max: 50,
    });
    expect(window.width).toBeGreaterThan(0);
  });
});

describe("windowToRgba", () => {
  it("maps values to an opaque grayscale RGBA buffer", () => {
    const rgba = windowToRgba(Float32Array.from([-200, 0, 200]), { center: 0, width: 200 });
    expect(rgba.length).toBe(12);
    expect(rgba[0]).toBe(0); // below the window -> black
    expect(rgba[3]).toBe(255); // opaque
    expect(rgba[8]).toBe(255); // above the window -> white
    expect(rgba[4]).toBeGreaterThan(120);
    expect(rgba[4]).toBeLessThan(136);
  });

  it("treats a degenerate width as a unit ramp rather than dividing by zero", () => {
    const rgba = windowToRgba(Float32Array.from([0]), { center: 0, width: 0 });
    expect(rgba.length).toBe(4);
    expect(Number.isFinite(rgba[0])).toBe(true);
  });
});

describe("overlayMask", () => {
  it("blends the highlight colour where the mask is set and leaves other pixels alone", () => {
    const rgba = windowToRgba(Float32Array.from([0, 0]), { center: 0, width: 200 });
    const baseGreen = rgba[1];

    overlayMask(rgba, Uint8Array.from([1, 0]), { r: 255, g: 0, b: 0 }, 0.5);

    expect(rgba[0]).toBeGreaterThan(baseGreen); // red channel boosted
    expect(rgba[1]).toBeLessThan(baseGreen); // green channel suppressed
    expect(rgba[4]).toBe(baseGreen); // second pixel untouched
  });
});
