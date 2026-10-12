import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import type { ToolCallOptions } from "../src/protocol.js";
import { PythonWorkerClient } from "../src/python-worker.js";

const REPOSITORY_ROOT = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const PYTHON = join(REPOSITORY_ROOT, ".venv", "bin", "python");

const clients: PythonWorkerClient[] = [];

afterEach(async () => {
  await Promise.all(clients.splice(0).map(async (client) => await client.stop()));
});

async function fixture(maxInflight = 4): Promise<PythonWorkerClient> {
  const client = new PythonWorkerClient({
    pythonExecutable: PYTHON,
    repositoryRoot: REPOSITORY_ROOT,
    maxInflight,
  });
  clients.push(client);
  return client;
}

function options(): ToolCallOptions {
  return { timeoutMs: 30_000 };
}

describe("persistent Python transport", () => {
  it("frames concurrent requests and correlates results by request ID", async () => {
    const client = await fixture();
    const [left, right] = await Promise.all([
      client.callSuccess("system.health", {}, options()),
      client.callSuccess("system.health", {}, options()),
    ]);

    expect(left.request_id).not.toBe(right.request_id);
    expect(left.result?.protocol_version).toBe("timesigma.bridge.v1");
    expect(right.result?.protocol_version).toBe("timesigma.bridge.v1");
    expect(left.result?.worker_pid).toBe(right.result?.worker_pid);
  });

  it("replays an identical idempotency key and rejects conflicting reuse", async () => {
    const client = await fixture();
    const fixed = { ...options(), idempotencyKey: "idem-fixed" };
    const first = await client.callSuccess("system.health", {}, fixed);
    const replay = await client.callSuccess("system.health", {}, fixed);
    expect(replay.request_id).not.toBe(first.request_id);
    expect(replay.result).toEqual(first.result);

    const conflict = await client.call("system.health", { probe: "different" }, fixed);
    expect(conflict.status).toBe("method_error");
    expect(conflict.error?.code).toBe("IDEMPOTENCY_CONFLICT");
  });

  it("rejects duplicate and conflicting uses of an in-flight idempotency key", async () => {
    const client = await fixture(2);
    const fixed = { ...options(), idempotencyKey: "idem-inflight" };

    const first = client.call("system.health", {}, fixed);
    const duplicate = client.call("system.health", {}, fixed);
    const conflict = client.call("system.health", { probe: "different" }, fixed);
    const [firstResult, duplicateResult, conflictResult] = await Promise.all([
      first,
      duplicate,
      conflict,
    ]);

    expect(firstResult.status).toBe("success");
    expect(duplicateResult.status).toBe("method_error");
    expect(duplicateResult.error?.code).toBe("IDEMPOTENCY_IN_FLIGHT");
    expect(conflictResult.status).toBe("method_error");
    expect(conflictResult.error?.code).toBe("IDEMPOTENCY_CONFLICT");
  });

  it("reports explicit backpressure instead of exceeding max inflight", async () => {
    const client = await fixture(1);
    const first = client.call("system.health", {}, options());
    const second = await client.call("system.health", {}, options());
    expect(second.status).toBe("infra_error");
    expect(second.error?.code).toBe("WORKER_BUSY");
    expect((await first).status).toBe("success");
  });

  it("cancels an in-flight request through the Python worker", async () => {
    const client = await fixture();
    const controller = new AbortController();
    controller.abort();

    const cancelled = await client.call("system.health", {}, options(), controller.signal);

    expect(cancelled.status).toBe("cancelled");
    expect(cancelled.error?.code).toBe("CANCELLED");
  });

  it("starts a fresh worker after a crash without replaying the old request", async () => {
    const client = await fixture();
    const before = await client.callSuccess("system.health", {}, options());
    const oldPid = before.result?.worker_pid;
    client.killForRestartTest();
    await waitUntil(() => client.workerPid === undefined);

    const after = await client.callSuccess("system.health", {}, options());
    expect(after.result?.worker_pid).not.toBe(oldPid);
  }, 15_000);
});

async function waitUntil(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 3_000;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("timed out waiting for worker state");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
