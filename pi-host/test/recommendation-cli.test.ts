import { rmSync } from "node:fs";
import { mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createModels, fauxProvider } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  JsonValue,
  ToolCallOptions,
  ToolName,
  ToolResultEnvelope,
} from "../src/protocol.js";
import type { PythonWorkerOptions } from "../src/python-worker.js";
import { runRecommendationCli } from "../src/recommendation-cli.js";
import type { RecommendationRunOptions } from "../src/recommendation-runner.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("recommendation CLI", () => {
  it("rejects iterative experiment budget before constructing a Provider", async () => {
    const directory = await fixtureScope();
    const args = commandArgs(directory);
    args[args.indexOf("--max-experiments") + 1] = "2";
    const createProviderRuntime = vi.fn();

    await expect(
      runRecommendationCli(args, { createProviderRuntime }),
    ).rejects.toThrow("--max-experiments must be 1");
    expect(createProviderRuntime).not.toHaveBeenCalled();
  });

  it("freezes one recommendation and evaluates it outside the Agent session", async () => {
    const directory = await fixtureScope();
    const output = captureOutput();
    const worker = workerFixture();
    const runtime = providerFixture();
    const createWorker = vi.fn((_options: PythonWorkerOptions) => worker);
    const runRecommendation = vi.fn(async (options: RecommendationRunOptions) => {
      await options.createSession();
      return {
        recommendation: {
          recommendation_id: "rec-ar-default",
          recommendation_job: join(directory, "recommendations/rec-ar-default"),
          action: { dimension: "default", parameter: null, value: "official_default" },
          pipeline: {
            method: "AR",
            resolved_configuration: {
              Data_Params: { preprocess: "min-max", diff_order: 1 },
              Model_Params: { p: 16 },
            },
          },
          rationale: "AR is selected from profile and historical evidence.",
        },
      };
    });

    await expect(
      runRecommendationCli(commandArgs(directory), {
        env: {
          TIMESIGMA_OPENAI_API_KEY: "openai-unit-test-secret",
          TIMESIGMA_DEEPSEEK_API_KEY: "deepseek-unit-test-secret",
          DEEPSEEK_API_KEY: "deepseek-fallback-secret",
          KEEP_FOR_PYTHON: "visible",
        },
        stdout: output.stream,
        createProviderRuntime: runtime.create,
        createWorker,
        runRecommendation,
      }),
    ).resolves.toBe(0);

    const summary = JSON.parse(output.value()) as Record<string, unknown>;
    expect(summary).toMatchObject({
      status: "completed",
      target_metric: "point-wise-f1",
      budgets: {
        max_experiments: 1,
        max_wall_seconds: 60,
        max_provider_calls: 2,
        max_provider_tokens: 100_000,
      },
      recommendation: {
        recommendation_id: "rec-ar-default",
      },
      evaluation: {
        recommendation_id: "rec-ar-default",
        target_metric: "point-wise-f1",
        target_value: 0.9,
      },
    });
    expect(output.value()).not.toContain("openai-unit-test-secret");
    expect(output.value()).not.toContain("deepseek-unit-test-secret");
    expect(output.value()).not.toContain("deepseek-fallback-secret");
    expect(runtime.create).toHaveBeenCalledOnce();
    expect(createWorker).toHaveBeenCalledOnce();
    expect(createWorker.mock.calls[0]?.[0].env).toMatchObject({
      KEEP_FOR_PYTHON: "visible",
    });
    expect(createWorker.mock.calls[0]?.[0].env).not.toHaveProperty("DEEPSEEK_API_KEY");
    expect(runRecommendation).toHaveBeenCalledOnce();
    expect(runRecommendation.mock.calls[0]?.[0].scope).toMatchObject({
      dataset: "TODS",
      schema: "naive",
      planning_unit: { curve: "0", training_curves: ["0"], held_out_curves: [] },
      experiment_root: join(directory, "recommendations"),
      experiment_index: 1,
      max_experiments: 1,
      max_wall_seconds: 60,
    });
    expect(worker.callSuccess.mock.calls.map((call) => call[0])).toEqual([
      "recommendation.context",
      "recommendation.evaluate",
    ]);
    expect(worker.callSuccess.mock.calls[1]?.[1]).toMatchObject({
      recommendation_id: "rec-ar-default",
    });
    expect(worker.callSuccess.mock.calls[1]?.[2]).toEqual({ timeoutMs: 60_000 });
    expect(worker.stop).toHaveBeenCalledOnce();
    const sessionFiles = await readdir(join(directory, "sessions"), { recursive: true });
    const sessionJsonlFiles = sessionFiles.filter((path) => path.endsWith(".jsonl"));
    expect(sessionJsonlFiles).toHaveLength(1);
    expect(JSON.stringify(sessionJsonlFiles)).not.toContain(directory);
    const sessionHeader = (
      await readFile(join(directory, "sessions", sessionJsonlFiles[0] as string), "utf8")
    ).split("\n")[0];
    expect(sessionHeader).toContain('"cwd":"timesigma-recommendation"');
    expect(sessionHeader).not.toContain(directory);
    const state = JSON.parse(await readFile(join(directory, "recommendation-run.json"), "utf8"));
    expect(state).toMatchObject({
      status: "completed",
      provider_log: join(directory, "provider.jsonl"),
      session_directory: join(directory, "sessions"),
      recommendation_root: join(directory, "recommendations"),
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
    callSuccess: vi.fn(async (
      tool: ToolName,
      payload: Record<string, JsonValue>,
      _options?: ToolCallOptions,
    ): Promise<ToolResultEnvelope> => ({
      protocol_version: "timesigma.bridge.v1",
      schema_version: 1,
      request_id: "request-preflight",
      status: "success",
      result: tool === "recommendation.context"
        ? {
            task_contract: {
              schema: "naive",
              target_metric: "point-wise-f1",
              input_state: "raw_train",
              planning_scope: "curve",
              training_member_count: 1,
              held_out_member_count: 0,
              one_shot_constraints: { current_task_metrics: "withheld" },
            },
            profile: { input_state: "raw_train", curve_profile: {}, aggregate: null },
            catalog: [],
            historical_pipeline_evidence: [],
          }
        : tool === "recommendation.evaluate"
          ? {
              recommendation_id: typeof payload.recommendation_id === "string"
                ? payload.recommendation_id
                : "",
              target_metric: "point-wise-f1",
              target_value: 0.9,
              official_metrics: { "point-wise-f1": 0.9 },
            }
          : { unexpected_tool: tool },
      error: null,
    })),
    stop: vi.fn(async () => undefined),
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
    "--max-experiments", "1",
    "--max-wall-seconds", "60",
    "--max-provider-calls", "2",
    "--max-provider-tokens", "100000",
    "--provider-log", join(runDirectory, "provider.jsonl"),
    "--session-dir", join(runDirectory, "sessions"),
    "--repository-root", runDirectory,
    "--python", join(runDirectory, ".venv", "bin", "python"),
  ];
}

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "timesigma-recommendation-cli-"));
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
