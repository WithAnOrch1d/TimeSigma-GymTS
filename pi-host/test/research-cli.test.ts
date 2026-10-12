import { rmSync } from "node:fs";
import { mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createModels, fauxProvider } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ToolName, ToolResultEnvelope } from "../src/protocol.js";
import type { PythonWorkerOptions } from "../src/python-worker.js";
import { runResearchCli } from "../src/research-cli.js";
import type {
  ResearchFeedbackOptions,
  ResearchTurn,
} from "../src/research-feedback.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("research CLI", () => {
  it("rejects a missing frozen budget before constructing a Provider", async () => {
    const directory = await temporaryDirectory();
    const args = commandArgs(directory);
    const budgetFlag = args.indexOf("--max-provider-tokens");
    args.splice(budgetFlag, 2);
    const createProviderRuntime = vi.fn();

    await expect(
      runResearchCli(args, { createProviderRuntime }),
    ).rejects.toThrow("--max-provider-tokens is required");
    expect(createProviderRuntime).not.toHaveBeenCalled();
  });

  it("rejects a target metric mismatch before constructing a Provider", async () => {
    const directory = await fixtureScope({ target_metric: "event-wise-f1" });
    const createProviderRuntime = vi.fn();

    await expect(
      runResearchCli(commandArgs(directory), { createProviderRuntime }),
    ).rejects.toThrow("--target-metric must match scope.target_metric");
    expect(createProviderRuntime).not.toHaveBeenCalled();
  });

  it("rejects an invalid Planning Unit before constructing a Provider or worker", async () => {
    const directory = await fixtureScope({
      planning_unit: { curve: "0", training_curves: ["1"], held_out_curves: [] },
    });
    const createProviderRuntime = vi.fn();
    const createWorker = vi.fn();

    await expect(
      runResearchCli(commandArgs(directory), { createProviderRuntime, createWorker }),
    ).rejects.toThrow("naive planning requires one curve");
    expect(createProviderRuntime).not.toHaveBeenCalled();
    expect(createWorker).not.toHaveBeenCalled();
  });

  it("does not construct a Provider when the Python scope preflight fails", async () => {
    const directory = await fixtureScope();
    const worker = workerFixture();
    worker.callSuccess.mockRejectedValueOnce(new Error("Profile manifest is missing"));
    const createProviderRuntime = vi.fn();

    await expect(
      runResearchCli(commandArgs(directory), {
        env: {},
        createProviderRuntime,
        createWorker: () => worker,
      }),
    ).rejects.toThrow("Profile manifest is missing");
    expect(createProviderRuntime).not.toHaveBeenCalled();
    expect(worker.stop).toHaveBeenCalledOnce();
  });

  it("assembles persisted Pi sessions and prints only result, frozen budgets, and usage", async () => {
    const directory = await fixtureScope();
    const output = captureOutput();
    const worker = workerFixture();
    const runtime = providerFixture();
    const createWorker = vi.fn((_options: PythonWorkerOptions) => worker);
    const runFeedback = vi.fn(async (options: ResearchFeedbackOptions) => {
      await options.createSession(1);
      await options.createSession(2);
      return feedbackResult();
    });

    await expect(
      runResearchCli(commandArgs(directory), {
        env: {
          TIMESIGMA_OPENAI_API_KEY: "openai-unit-test-secret",
          TIMESIGMA_DEEPSEEK_API_KEY: "deepseek-unit-test-secret",
          DEEPSEEK_API_KEY: "deepseek-fallback-secret",
          KEEP_FOR_PYTHON: "visible",
        },
        stdout: output.stream,
        createProviderRuntime: runtime.create,
        createWorker,
        runFeedback,
      }),
    ).resolves.toBe(0);

    const summary = JSON.parse(output.value()) as Record<string, unknown>;
    expect(summary).toMatchObject({
      status: "completed",
      target_metric: "point-wise-f1",
      budgets: {
        max_experiments: 3,
        max_wall_seconds: 60,
        max_provider_calls: 4,
        max_provider_tokens: 100_000,
      },
      usage: { logicalCalls: 0, providerAttempts: 0, tokens: 0, paused: false },
    });
    expect(output.value()).not.toContain("openai-unit-test-secret");
    expect(output.value()).not.toContain("deepseek-unit-test-secret");
    expect(output.value()).not.toContain("deepseek-fallback-secret");
    expect(runtime.create).toHaveBeenCalledOnce();
    expect(createWorker).toHaveBeenCalledOnce();
    const workerEnvironment = createWorker.mock.calls[0]?.[0].env;
    expect(workerEnvironment).toMatchObject({ KEEP_FOR_PYTHON: "visible" });
    expect(workerEnvironment).not.toHaveProperty("TIMESIGMA_OPENAI_API_KEY");
    expect(workerEnvironment).not.toHaveProperty("TIMESIGMA_DEEPSEEK_API_KEY");
    expect(workerEnvironment).not.toHaveProperty("DEEPSEEK_API_KEY");
    expect(runFeedback).toHaveBeenCalledOnce();
    expect(runFeedback.mock.calls[0]?.[0].scope).toMatchObject({
      dataset: "TODS",
      schema: "naive",
      planning_unit: { curve: "0", training_curves: ["0"], held_out_curves: [] },
      dataset_dir: "/data/UTS/TODS",
      profile_dir: "/derived/profiles/TODS",
      agent_memory_root: join(directory, "agent-memory"),
      experiment_root: join(directory, "experiments"),
      experiment_index: 1,
      max_experiments: 3,
      max_wall_seconds: 60,
    });
    expect(worker.stop).toHaveBeenCalledOnce();
    const sessionFiles = await readdir(join(directory, "sessions"), { recursive: true });
    expect(sessionFiles.filter((path) => path.endsWith(".jsonl"))).toHaveLength(2);
    const state = JSON.parse(await readFile(join(directory, "research-run.json"), "utf8"));
    expect(state).toMatchObject({
      status: "completed",
      provider_log: join(directory, "provider.jsonl"),
      session_directory: join(directory, "sessions"),
      experiment_root: join(directory, "experiments"),
    });
  });

  it("starts a fresh Research Run when the same frozen scope uses a new output directory", async () => {
    const scopeDirectory = await fixtureScope();
    const firstRun = await temporaryDirectory();
    const secondRun = await temporaryDirectory();
    const observedScopes: unknown[] = [];
    const runtime = providerFixture();

    const runFeedback = vi.fn(async (options: ResearchFeedbackOptions) => {
      observedScopes.push(options.scope);
      return feedbackResult();
    });

    await runResearchCli(commandArgs(scopeDirectory, firstRun), {
      env: {},
      stdout: captureOutput().stream,
      createProviderRuntime: runtime.create,
      createWorker: () => workerFixture(),
      runFeedback,
    });
    await runResearchCli(commandArgs(scopeDirectory, secondRun), {
      env: {},
      stdout: captureOutput().stream,
      createProviderRuntime: runtime.create,
      createWorker: () => workerFixture(),
      runFeedback,
    });

    expect(observedScopes).toHaveLength(2);
    expect(observedScopes[0]).toMatchObject({
      experiment_root: join(firstRun, "experiments"),
      agent_memory_root: join(scopeDirectory, "agent-memory"),
      experiment_index: 1,
      max_experiments: 3,
      max_wall_seconds: 60,
    });
    expect(observedScopes[1]).toMatchObject({
      experiment_root: join(secondRun, "experiments"),
      agent_memory_root: join(scopeDirectory, "agent-memory"),
      experiment_index: 1,
      max_experiments: 3,
      max_wall_seconds: 60,
    });
    expect(runtime.create).toHaveBeenCalledTimes(2);
  });

  it("requires run-local provider log and session outputs to be fresh", async () => {
    const directory = await fixtureScope();
    await writeFile(join(directory, "provider.jsonl"), "{\"sequence\":1}\n");
    const createProviderRuntime = vi.fn();

    await expect(
      runResearchCli(commandArgs(directory), { env: {}, createProviderRuntime }),
    ).rejects.toThrow("provider log already exists");
    expect(createProviderRuntime).not.toHaveBeenCalled();
  });

  it("stops the Python worker when the feedback loop fails", async () => {
    const directory = await fixtureScope();
    const worker = workerFixture();
    const runtime = providerFixture();

    await expect(
      runResearchCli(commandArgs(directory), {
        env: {},
        stdout: captureOutput().stream,
        createProviderRuntime: runtime.create,
        createWorker: () => worker,
        runFeedback: async () => {
          throw new Error("feedback failed");
        },
      }),
    ).rejects.toThrow("feedback failed");
    expect(worker.stop).toHaveBeenCalledOnce();
    const state = JSON.parse(await readFile(join(directory, "research-run.json"), "utf8"));
    expect(state).toMatchObject({
      status: "aborted",
      details: { reason: "feedback failed" },
    });
  });
});

function providerFixture() {
  const faux = fauxProvider();
  const models = createModels();
  models.setProvider(faux.provider);
  return {
    create: vi.fn(async (config) => ({
      config,
      model: faux.getModel(),
      models,
      authSource: "test",
    })),
  };
}

function workerFixture() {
  return {
    callSuccess: vi.fn(async (tool: ToolName): Promise<ToolResultEnvelope> => ({
      protocol_version: "timesigma.bridge.v1",
      schema_version: 1,
      request_id: "request-preflight",
      status: "success",
      result: tool === "research.context"
        ? { component_cases: [], pipeline_cases: [] }
        : { unexpected_tool: tool },
      error: null,
    })),
    stop: vi.fn(async () => undefined),
  };
}

function feedbackResult(): { experiments: ResearchTurn[] } {
  return {
    experiments: [
      {
        experiment_id: "research-experiment-1",
        evidence_id: "research-step/TODS/naive/0/research-experiment-1",
        baseline_evidence_id: "component/LOF/naive/TODS/0",
        action: { dimension: "hyperparameter", parameter: "n_neighbors", value: 10 },
        diagnosis_status: "supported",
        next_action: { dimension: "hyperparameter", parameter: "n_neighbors", value: 15 },
        pipeline_job: "/runs/research/research-experiment-1",
        target_metric: "point-wise-f1",
        baseline_value: 0.2,
        candidate_value: 0.3,
        delta: 0.1,
      },
      {
        experiment_id: "research-experiment-2",
        evidence_id: "research-step/TODS/naive/0/research-experiment-2",
        baseline_evidence_id: "research-step/TODS/naive/0/research-experiment-1",
        action: { dimension: "hyperparameter", parameter: "n_neighbors", value: 15 },
        diagnosis_status: "supported",
        next_action: { dimension: "hyperparameter", parameter: "n_neighbors", value: 20 },
        pipeline_job: "/runs/research/research-experiment-2",
        target_metric: "point-wise-f1",
        baseline_value: 0.3,
        candidate_value: 0.35,
        delta: 0.05,
      },
    ],
  };
}

async function fixtureScope(
  overrides: Record<string, unknown> = {},
): Promise<string> {
  const directory = await temporaryDirectory();
  await writeFile(
    join(directory, "scope.json"),
    `${JSON.stringify({
      dataset: "TODS",
      schema: "naive",
      target_metric: "point-wise-f1",
      planning_unit: { curve: "0", training_curves: ["0"], held_out_curves: [] },
      dataset_root: "/data",
      dataset_source_root: "/sources",
      dataset_dir: "/data/UTS/TODS",
      dataset_source: "CSTCloudOps/datasets",
      dataset_revision: "0c6dfa195a5af8ef75938bc1ca6b870e49c7b14a",
      profile_dir: "/derived/profiles/TODS",
      agent_memory_root: join(directory, "agent-memory"),
      zero_shot_membership: null,
      ...overrides,
    })}\n`,
  );
  return directory;
}

function commandArgs(scopeDirectory: string, runDirectory = scopeDirectory): string[] {
  return [
    "--scope", join(scopeDirectory, "scope.json"),
    "--target-metric", "point-wise-f1",
    "--max-experiments", "3",
    "--max-wall-seconds", "60",
    "--max-provider-calls", "4",
    "--max-provider-tokens", "100000",
    "--provider-log", join(runDirectory, "provider.jsonl"),
    "--session-dir", join(runDirectory, "sessions"),
    "--repository-root", runDirectory,
    "--python", join(runDirectory, ".venv", "bin", "python"),
  ];
}

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "timesigma-research-cli-"));
  temporaryDirectories.push(directory);
  return directory;
}

function captureOutput() {
  let content = "";
  return {
    stream: {
      write(chunk: string): boolean {
        content += chunk;
        return true;
      },
    },
    value: () => content,
  };
}
