import { randomUUID } from "node:crypto";

export const PROTOCOL_VERSION = "timesigma.bridge.v1" as const;
export const SCHEMA_VERSION = 1 as const;
export const ALLOWED_TOOLS = [
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
] as const;

export type ToolName = (typeof ALLOWED_TOOLS)[number];
export type ToolStatus = "success" | "method_error" | "infra_error" | "cancelled";
export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

export interface ToolRequestEnvelope {
  protocol_version: typeof PROTOCOL_VERSION;
  schema_version: typeof SCHEMA_VERSION;
  request_id: string;
  tool: ToolName;
  deadline_unix_ms: number;
  idempotency_key: string;
  payload: Record<string, JsonValue>;
}

export interface ToolResultEnvelope {
  protocol_version: typeof PROTOCOL_VERSION;
  schema_version: typeof SCHEMA_VERSION;
  request_id: string;
  status: ToolStatus;
  result: Record<string, JsonValue> | null;
  error: Record<string, JsonValue> | null;
}

export interface ToolCallOptions {
  timeoutMs?: number;
  idempotencyKey?: string;
}

const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

export function createRequest(
  tool: ToolName,
  payload: Record<string, JsonValue>,
  options: ToolCallOptions = {},
): ToolRequestEnvelope {
  const timeoutMs = options.timeoutMs ?? 30_000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) {
    throw new Error("timeoutMs must be a positive safe integer");
  }
  const request: ToolRequestEnvelope = {
    protocol_version: PROTOCOL_VERSION,
    schema_version: SCHEMA_VERSION,
    request_id: `req-${randomUUID()}`,
    tool,
    deadline_unix_ms: Date.now() + timeoutMs,
    idempotency_key: options.idempotencyKey ?? `idem-${randomUUID()}`,
    payload,
  };
  validateRequest(request);
  return request;
}

export function validateRequest(request: ToolRequestEnvelope): void {
  expectExactKeys(
    request as unknown as Record<string, unknown>,
    [
      "protocol_version",
      "schema_version",
      "request_id",
      "tool",
      "deadline_unix_ms",
      "idempotency_key",
      "payload",
    ],
    "ToolRequestEnvelope",
  );
  if (request.protocol_version !== PROTOCOL_VERSION || request.schema_version !== SCHEMA_VERSION) {
    throw new Error("unsupported bridge protocol");
  }
  if (!IDENTIFIER.test(request.request_id)) throw new Error("request_id is not a valid identifier");
  if (!IDENTIFIER.test(request.idempotency_key)) throw new Error("idempotency_key is not a valid identifier");
  if (!(ALLOWED_TOOLS as readonly string[]).includes(request.tool)) throw new Error("tool is not allowlisted");
  if (!Number.isSafeInteger(request.deadline_unix_ms) || request.deadline_unix_ms < 1) {
    throw new Error("deadline_unix_ms must be a positive safe integer");
  }
  expectObject(request.payload, "payload");
}

export function parseToolResult(value: unknown): ToolResultEnvelope {
  const result = expectObject(value, "ToolResultEnvelope");
  expectExactKeys(
    result,
    ["protocol_version", "schema_version", "request_id", "status", "result", "error"],
    "ToolResultEnvelope",
  );
  if (result.protocol_version !== PROTOCOL_VERSION || result.schema_version !== SCHEMA_VERSION) {
    throw new Error("worker returned an unsupported protocol version");
  }
  if (typeof result.request_id !== "string" || !IDENTIFIER.test(result.request_id)) {
    throw new Error("worker returned an invalid request_id");
  }
  if (!(["success", "method_error", "infra_error", "cancelled"] as unknown[]).includes(result.status)) {
    throw new Error("worker returned an invalid status");
  }
  const status = result.status as ToolStatus;
  if (status === "success") {
    expectObject(result.result, "result");
    if (result.error !== null) throw new Error("successful result cannot contain error");
  } else {
    expectObject(result.error, "error");
    if (result.result !== null) throw new Error("failed result cannot contain result");
  }
  return result as unknown as ToolResultEnvelope;
}

function expectObject(value: unknown, name: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function expectExactKeys(value: Record<string, unknown>, expected: readonly string[], name: string): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new Error(`${name} fields do not match schema`);
  }
}
