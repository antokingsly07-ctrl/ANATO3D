import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => {
  class ApiRequestError extends Error {
    code: string;
    status: number;
    detail: string | null;
    constructor(status: number, code: string, message: string, detail: string | null = null) {
      super(message);
      this.name = "ApiRequestError";
      this.status = status;
      this.code = code;
      this.detail = detail;
    }
  }
  return {
    ApiRequestError,
    api: {
      getSlice: vi.fn(),
      getMaskSlice: vi.fn(),
      getSegmentation: vi.fn(),
    },
  };
});

import { api, ApiRequestError } from "../api/client";
import type { SliceResponse } from "../types";
import { SliceViewer } from "./SliceViewer";

const getSlice = api.getSlice as unknown as ReturnType<typeof vi.fn>;

function sliceFixture(overrides: Partial<SliceResponse> = {}): SliceResponse {
  const pixels = Int16Array.from([0, 100, 200, 300]);
  const bytes = new Uint8Array(pixels.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);

  return {
    volume_id: "vol-1",
    plane: "axial",
    index: 3,
    count: 32,
    rows: 2,
    columns: 2,
    row_spacing_mm: 1.5,
    column_spacing_mm: 1.5,
    position_mm: 12.5,
    position_label: "z = 12.5 mm",
    labels: { top: "A", bottom: "P", left: "R", right: "L" },
    dtype: "int16",
    encoding: "base64",
    data: btoa(binary),
    intensity_min: 0,
    intensity_max: 300,
    intensity_units: "HU",
    window_center: null,
    window_width: null,
    warnings: [],
    ...overrides,
  };
}

beforeEach(() => {
  getSlice.mockReset();
});

describe("SliceViewer", () => {
  it("prompts for a volume when none is loaded", () => {
    render(
      <SliceViewer
        volumeId={null}
        segmentationId={null}
        plane="axial"
        index={0}
        center={null}
        width={null}
      />,
    );
    expect(screen.getByText(/reconstruct a volume/i)).toBeInTheDocument();
  });

  it("requests the slice the user asked for and renders it with orientation labels", async () => {
    getSlice.mockResolvedValue(sliceFixture());

    render(
      <SliceViewer
        volumeId="vol-1"
        segmentationId={null}
        plane="axial"
        index={3}
        center={null}
        width={null}
      />,
    );

    await waitFor(() => expect(screen.getByTestId("slice-canvas")).toBeInTheDocument());

    expect(getSlice).toHaveBeenCalledWith("vol-1", "axial", 3, { center: null, width: null });
    expect(screen.getByText(/slice 4\/32/)).toBeInTheDocument();
    expect(screen.getByText("A")).toBeInTheDocument();
    expect(screen.getByText("L")).toBeInTheDocument();
    expect(screen.getByText(/z = 12.5 mm/)).toBeInTheDocument();
  });

  it("shows the backend's message when the slice cannot be extracted", async () => {
    getSlice.mockRejectedValue(
      new ApiRequestError(400, "validation_error", "Slice index is out of range."),
    );

    render(
      <SliceViewer
        volumeId="vol-1"
        segmentationId={null}
        plane="sagittal"
        index={99}
        center={null}
        width={null}
      />,
    );

    expect(await screen.findByText("Slice index is out of range.")).toBeInTheDocument();
  });
});
