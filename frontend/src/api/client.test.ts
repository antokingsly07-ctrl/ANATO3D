import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, ApiRequestError, pollJob } from "./client";
import type { JobEnvelope } from "../types";

function fakeResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => (body === undefined ? "" : JSON.stringify(body)),
  } as unknown as Response;
}

const originalFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("api request handling", () => {
  it("returns the parsed body on success", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      fakeResponse(200, { status: "ok", application: "ANATO3D", version: "0.1.0" }),
    );
    const health = await api.health();
    expect(health.application).toBe("ANATO3D");
  });

  it("normalises an error payload into ApiRequestError with code and detail", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      fakeResponse(400, {
        code: "validation_error",
        message: "The series has no usable geometry.",
        detail: "enable allow_instance_number_fallback",
      }),
    );

    await expect(api.analyze("ds-1")).rejects.toMatchObject({
      name: "ApiRequestError",
      status: 400,
      code: "validation_error",
      message: "The series has no usable geometry.",
      detail: "enable allow_instance_number_fallback",
    });
  });

  it("reports a network failure distinctly from an HTTP error", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("ECONNREFUSED"));
    const error = await api.health().catch((err: unknown) => err);
    expect(error).toBeInstanceOf(ApiRequestError);
    expect((error as ApiRequestError).code).toBe("network_error");
    expect((error as ApiRequestError).status).toBe(0);
  });

  it("survives a non-JSON error body without throwing a parse error", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => "<html>Internal Server Error</html>",
    } as unknown as Response);

    const error = await api.health().catch((err: unknown) => err);
    expect(error).toBeInstanceOf(ApiRequestError);
    expect((error as ApiRequestError).code).toBe("http_500");
    expect((error as ApiRequestError).message).toContain("500");
  });
});

describe("pollJob", () => {
  function envelope(overrides: Partial<JobEnvelope>): JobEnvelope {
    return {
      job_id: "job-1",
      kind: "reconstruction",
      status: "running",
      stage: "building",
      progress: 0.5,
      message: "working",
      warnings: [],
      result: null,
      error: null,
      duration_s: null,
      logs: [],
      ...overrides,
    };
  }

  it("polls until the job completes and surfaces every update", async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock
      .mockResolvedValueOnce(fakeResponse(200, envelope({ status: "running", progress: 0.25 })))
      .mockResolvedValueOnce(fakeResponse(200, envelope({ status: "running", progress: 0.75 })))
      .mockResolvedValueOnce(
        fakeResponse(
          200,
          envelope({ status: "completed", progress: 1, result: { volume_id: "vol-1" } }),
        ),
      );

    const updates: number[] = [];
    const final = await pollJob<{ volume_id: string }>("job-1", {
      intervalMs: 1,
      onUpdate: (job) => updates.push(job.progress),
    });

    expect(final.status).toBe("completed");
    expect(final.result?.volume_id).toBe("vol-1");
    expect(updates).toEqual([0.25, 0.75, 1]);
  });

  it("returns a failed job rather than throwing", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      fakeResponse(
        200,
        envelope({
          status: "failed",
          error: { code: "dicom_error", message: "No usable slices." },
        }),
      ),
    );

    const final = await pollJob("job-1", { intervalMs: 1 });
    expect(final.status).toBe("failed");
    expect(final.error?.message).toBe("No usable slices.");
  });
});
