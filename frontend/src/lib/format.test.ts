import { describe, expect, it } from "vitest";

import {
  clamp,
  formatBytes,
  formatDimensions,
  formatDuration,
  formatMm3ToCm3,
  formatNumber,
  formatSpacing,
  severityRank,
  titleCase,
} from "./format";

describe("numeric formatting", () => {
  it("formats plain numbers and handles missing values", () => {
    expect(formatNumber(1.2345, 2)).toBe("1.23");
    expect(formatNumber(null)).toBe("-");
    expect(formatNumber(Number.NaN)).toBe("-");
  });

  it("uses exponential notation for extreme magnitudes", () => {
    expect(formatNumber(0.0001)).toMatch(/e-4/);
    expect(formatNumber(5_000_000)).toMatch(/e\+6/);
  });

  it("converts cubic millimetres to cubic centimetres", () => {
    expect(formatMm3ToCm3(18819)).toBe("18.819 cm\u00b3");
    expect(formatMm3ToCm3(null)).toBe("-");
  });

  it("formats byte sizes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });

  it("formats durations", () => {
    expect(formatDuration(0.5)).toBe("500 ms");
    expect(formatDuration(2.5)).toBe("2.50 s");
    expect(formatDuration(75)).toBe("1 min 15 s");
  });

  it("formats dimensions and spacing", () => {
    expect(formatDimensions([32, 64, 64])).toBe("32 \u00d7 64 \u00d7 64");
    expect(formatSpacing([2, 1.5, 1.5])).toBe("2.00 \u00d7 1.50 \u00d7 1.50 mm");
    expect(formatDimensions(null)).toBe("-");
  });
});

describe("misc helpers", () => {
  it("clamps values", () => {
    expect(clamp(5, 0, 3)).toBe(3);
    expect(clamp(-2, 0, 3)).toBe(0);
    expect(clamp(2, 0, 3)).toBe(2);
  });

  it("orders severities error < warning < info", () => {
    expect(severityRank("error")).toBeLessThan(severityRank("warning"));
    expect(severityRank("warning")).toBeLessThan(severityRank("info"));
  });

  it("title-cases words", () => {
    expect(titleCase("axial")).toBe("Axial");
  });
});
