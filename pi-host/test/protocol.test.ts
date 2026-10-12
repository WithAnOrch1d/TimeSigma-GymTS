import { describe, expect, it } from "vitest";
import { ALLOWED_TOOLS, createRequest, parseToolResult } from "../src/protocol.js";

describe("domain-tool schema", () => {
  it("creates a strict versioned request envelope", () => {
    const request = createRequest("system.health", {}, {
      timeoutMs: 1_000,
    });
    expect(request.protocol_version).toBe("timesigma.bridge.v1");
    expect(request.schema_version).toBe(1);
    expect(ALLOWED_TOOLS).toEqual([
      "experiment.run",
      "experiment.preview",
      "memory.search",
      "memory.read",
      "memory.compare",
      "capability.lookup",
      "recommendation.context",
      "recommendation.evaluate",
      "recommendation.freeze",
      "research.complete_step",
      "research.context",
      "research.finish_episode",
      "system.health",
      "delivery.context",
      "delivery.preview",
      "delivery.freeze",
    ]);
    expect(Object.keys(request).sort()).toEqual([
      "deadline_unix_ms",
      "idempotency_key",
      "payload",
      "protocol_version",
      "request_id",
      "schema_version",
      "tool",
    ]);
    expect(() =>
      createRequest("system.health", {}, { timeoutMs: Number.MAX_SAFE_INTEGER }),
    ).toThrow("deadline_unix_ms must be a positive safe integer");
  });

  it("rejects result fields outside the frozen cross-language schema", () => {
    expect(() =>
      parseToolResult({
        protocol_version: "timesigma.bridge.v1",
        schema_version: 1,
        request_id: "request-1",
        status: "success",
        result: {},
        error: null,
        labels: [0, 1],
      }),
    ).toThrow("fields do not match schema");
  });
});
