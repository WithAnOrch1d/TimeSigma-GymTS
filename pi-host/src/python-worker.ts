import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface, type Interface as ReadlineInterface } from "node:readline";
import { once } from "node:events";
import type { JsonValue, ToolCallOptions, ToolName, ToolResultEnvelope } from "./protocol.js";
import { createRequest, parseToolResult } from "./protocol.js";

export interface PythonWorkerOptions {
  pythonExecutable: string;
  repositoryRoot: string;
  env?: NodeJS.ProcessEnv;
  maxInflight?: number;
}

interface PendingCall {
  resolve: (result: ToolResultEnvelope) => void;
  reject: (error: Error) => void;
  cleanup: () => void;
}

export class WorkerExitedError extends Error {}
export class WorkerProtocolError extends Error {}

export class DomainToolError extends Error {
  constructor(readonly envelope: ToolResultEnvelope) {
    const details = envelope.error?.message;
    super(typeof details === "string" ? details : `domain tool failed as ${envelope.status}`);
  }
}

export class PythonWorkerClient {
  private child: ChildProcessWithoutNullStreams | undefined;
  private lines: ReadlineInterface | undefined;
  private readonly pending = new Map<string, PendingCall>();
  private stopping = false;
  private stderrTail: string[] = [];

  constructor(readonly options: PythonWorkerOptions) {
    if (!Number.isSafeInteger(options.maxInflight ?? 4) || (options.maxInflight ?? 4) < 1) {
      throw new Error("maxInflight must be a positive safe integer");
    }
  }

  get workerPid(): number | undefined {
    return this.child?.pid;
  }

  start(): void {
    if (this.child !== undefined) return;
    this.stopping = false;
    const args = [
      "-m",
      "timesigma.bridge.worker",
      "--max-inflight",
      String(this.options.maxInflight ?? 4),
    ];
    const child = spawn(this.options.pythonExecutable, args, {
      cwd: this.options.repositoryRoot,
      env: { ...(this.options.env ?? process.env), PYTHONUNBUFFERED: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    this.stderrTail = [];
    this.lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
    this.lines.on("line", (line) => this.onLine(line));
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      this.stderrTail.push(...chunk.split(/\r?\n/).filter(Boolean));
      this.stderrTail = this.stderrTail.slice(-20);
    });
    child.once("error", (error) => this.onExit(error));
    child.once("exit", (code, signal) => {
      const suffix = this.stderrTail.length > 0 ? `: ${this.stderrTail.join(" | ")}` : "";
      this.onExit(new WorkerExitedError(`Python worker exited (${code ?? signal})${suffix}`));
    });
  }

  async call(
    tool: ToolName,
    payload: Record<string, JsonValue>,
    options: ToolCallOptions = {},
    signal?: AbortSignal,
  ): Promise<ToolResultEnvelope> {
    this.start();
    const child = this.child;
    if (child === undefined) throw new WorkerExitedError("Python worker did not start");
    const request = createRequest(tool, payload, options);
    return await new Promise<ToolResultEnvelope>((resolve, reject) => {
      let cancelled = false;
      const cancel = (reason: string) => {
        if (cancelled || !this.pending.has(request.request_id)) return;
        cancelled = true;
        this.write({
          control: "cancel",
          request_id: request.request_id,
          reason,
        });
      };
      const onAbort = () => cancel("caller aborted the tool request");
      signal?.addEventListener("abort", onAbort, { once: true });
      const timeout = setTimeout(
        () => cancel("tool request exceeded its deadline"),
        Math.max(1, request.deadline_unix_ms - Date.now()),
      );
      const cleanup = () => {
        clearTimeout(timeout);
        signal?.removeEventListener("abort", onAbort);
      };
      this.pending.set(request.request_id, { resolve, reject, cleanup });
      try {
        this.write(request);
        if (signal?.aborted) cancel("caller aborted the tool request");
      } catch (error) {
        this.pending.delete(request.request_id);
        cleanup();
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  async callSuccess(
    tool: ToolName,
    payload: Record<string, JsonValue>,
    options: ToolCallOptions = {},
    signal?: AbortSignal,
  ): Promise<ToolResultEnvelope> {
    const result = await this.call(tool, payload, options, signal);
    if (result.status !== "success") throw new DomainToolError(result);
    return result;
  }

  async restart(): Promise<void> {
    await this.stop();
    this.start();
  }

  async stop(): Promise<void> {
    const child = this.child;
    if (child === undefined) return;
    this.stopping = true;
    this.child = undefined;
    this.lines?.close();
    this.lines = undefined;
    child.stdin.end();
    const exited = once(child, "exit").then(() => undefined);
    const forced = new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        child.kill("SIGTERM");
        resolve();
      }, 2_000);
      timer.unref();
    });
    await Promise.race([exited, forced]);
    this.rejectPending(new WorkerExitedError("Python worker was stopped"));
    this.stopping = false;
  }

  killForRestartTest(): void {
    this.child?.kill("SIGKILL");
  }

  private write(payload: object): void {
    const child = this.child;
    if (child === undefined || child.stdin.destroyed) throw new WorkerExitedError("Python worker is not writable");
    child.stdin.write(`${JSON.stringify(payload)}\n`);
  }

  private onLine(line: string): void {
    let parsed: ToolResultEnvelope;
    try {
      parsed = parseToolResult(JSON.parse(line));
    } catch (error) {
      const failure = new WorkerProtocolError(
        `Invalid JSONL response: ${error instanceof Error ? error.message : String(error)}`,
      );
      this.rejectPending(failure);
      this.child?.kill("SIGTERM");
      return;
    }
    const pending = this.pending.get(parsed.request_id);
    if (pending === undefined) return;
    this.pending.delete(parsed.request_id);
    pending.cleanup();
    pending.resolve(parsed);
  }

  private onExit(error: Error): void {
    const child = this.child;
    if (child === undefined) return;
    this.child = undefined;
    this.lines?.close();
    this.lines = undefined;
    if (!this.stopping) this.rejectPending(error);
  }

  private rejectPending(error: Error): void {
    for (const pending of this.pending.values()) {
      pending.cleanup();
      pending.reject(error);
    }
    this.pending.clear();
  }
}
