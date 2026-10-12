import { access, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { JsonlSessionRepo } from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { createAuditedModels } from "./audited-models.js";
import { createTimeSigmaProviderRuntime } from "./provider-client.js";
import { providerDefinition } from "./provider-catalog.js";
import { loadTimeSigmaProviderConfig } from "./provider-config.js";
import { ProviderLog } from "./provider-log.js";
import { createAuditedProviderController } from "./provider-runtime.js";
import {
  PythonWorkerClient,
  type PythonWorkerOptions,
} from "./python-worker.js";
import {
  runOneShotRecommendation,
} from "./recommendation-runner.js";
import type { ResearchScope, ResearchWorker } from "./research-agent.js";

interface TextOutput {
  write(chunk: string): unknown;
}

interface RecommendationCliWorker extends ResearchWorker {
  stop(): Promise<void>;
}

export interface RecommendationCliOptions {
  env?: NodeJS.ProcessEnv;
  stdout?: TextOutput;
  readScopeFile?: (path: string) => Promise<string>;
  createProviderRuntime?: typeof createTimeSigmaProviderRuntime;
  createWorker?: (options: PythonWorkerOptions) => RecommendationCliWorker;
  runRecommendation?: typeof runOneShotRecommendation;
}

export interface FrozenRecommendationCliArgs {
  scopePath: string;
  targetMetric: string;
  maxExperiments: 1;
  maxWallSeconds: number;
  maxProviderCalls: number;
  maxProviderTokens: number;
  providerLogPath: string;
  sessionDirectory: string;
  repositoryRoot: string;
  pythonExecutable: string;
}

const flags = {
  "--scope": "scopePath",
  "--target-metric": "targetMetric",
  "--max-experiments": "maxExperiments",
  "--max-wall-seconds": "maxWallSeconds",
  "--max-provider-calls": "maxProviderCalls",
  "--max-provider-tokens": "maxProviderTokens",
  "--provider-log": "providerLogPath",
  "--session-dir": "sessionDirectory",
  "--repository-root": "repositoryRoot",
  "--python": "pythonExecutable",
} as const;

const OFFICIAL_METRICS = new Set([
  "point-wise-f1",
  "reduced-length-f1",
  "event-wise-f1",
  "delay-reduced-length-f1",
  "point-wise-auprc",
  "reduced-length-auprc",
  "event-wise-auprc",
  "delay-reduced-length-auprc",
]);
const FROZEN_SCOPE_FIELDS = [
  "dataset",
  "schema",
  "target_metric",
  "planning_unit",
  "dataset_root",
  "dataset_source_root",
  "dataset_dir",
  "dataset_source",
  "dataset_revision",
  "profile_dir",
  "zero_shot_membership",
] as const;
const OPTIONAL_FROZEN_SCOPE_FIELDS = [
  "agent_memory_root",
] as const;
const RECOMMENDATION_SESSION_CWD = "timesigma-recommendation";
const PROVIDER_PROFILES = ["openai", "deepseek"] as const;

type FrozenRecommendationScope = Omit<
  ResearchScope,
  | "experiment_root"
  | "experiment_index"
  | "max_experiments"
  | "max_wall_seconds"
>;

interface RecommendationRunPaths {
  runRoot: string;
  statePath: string;
  providerLogPath: string;
  sessionDirectory: string;
  recommendationRoot: string;
}

export async function runRecommendationCli(
  args: readonly string[],
  options: RecommendationCliOptions = {},
): Promise<number> {
  if (args.length === 1 && ["help", "--help", "-h"].includes(args[0] as string)) {
    (options.stdout ?? process.stdout).write(usage());
    return 0;
  }
  const frozen = parseRecommendationCliArgs(args);
  const scopePath = resolve(frozen.scopePath);
  const scopeText = await (options.readScopeFile ?? readUtf8)(scopePath);
  const parsedScope = record(JSON.parse(scopeText), "scope");
  const scope = parseFrozenRecommendationScope(parsedScope);
  validateFrozenScope(scope, frozen);

  const environment = options.env ?? process.env;
  const config = loadTimeSigmaProviderConfig(environment);
  const repositoryRoot = resolve(frozen.repositoryRoot);
  const executionEnvironment = new NodeExecutionEnv({ cwd: repositoryRoot });
  const runPaths = recommendationRunPaths(frozen);
  await assertNewRecommendationRun(runPaths);
  const runScope = createRunScope(scope, frozen, runPaths);
  await writeRecommendationRunState(runPaths, "running", {
    scope_path: scopePath,
    target_metric: frozen.targetMetric,
    budgets: {
      max_experiments: frozen.maxExperiments,
      max_wall_seconds: frozen.maxWallSeconds,
      max_provider_calls: frozen.maxProviderCalls,
      max_provider_tokens: frozen.maxProviderTokens,
    },
  });
  let worker: RecommendationCliWorker | undefined;
  try {
    worker = (options.createWorker ?? createPythonWorker)({
      pythonExecutable: resolve(frozen.pythonExecutable),
      repositoryRoot,
      env: pythonWorkerEnvironment(environment),
    });
    await worker.callSuccess("recommendation.context", { ...runScope });

    const providerRuntime = await (
      options.createProviderRuntime ?? createTimeSigmaProviderRuntime
    )(config, environment);
    const providerLog = new ProviderLog(runPaths.providerLogPath);
    const controller = createAuditedProviderController({
      config,
      model: providerRuntime.model,
      source: providerRuntime.models.streamSimple.bind(providerRuntime.models),
      log: providerLog,
      budget: {
        maxCalls: frozen.maxProviderCalls,
        maxTokens: frozen.maxProviderTokens,
      },
    });
    const models = createAuditedModels(providerRuntime.models, controller);
    const sessions = new JsonlSessionRepo({
      fs: executionEnvironment,
      sessionsRoot: runPaths.sessionDirectory,
    });
    const recommendation = await (options.runRecommendation ?? runOneShotRecommendation)({
      models,
      model: providerRuntime.model,
      worker,
      scope: runScope,
      createSession: async () =>
        await sessions.create({
          cwd: RECOMMENDATION_SESSION_CWD,
          metadata: {
            purpose: "timesigma-recommendation",
            target_metric: frozen.targetMetric,
          },
        }),
      prompt:
        "Make one frozen one-shot Pipeline recommendation from the frozen scope and historical evidence.",
    });
    const evaluationEnvelope = await worker.callSuccess(
      "recommendation.evaluate",
      {
        ...runScope,
        recommendation_id: recommendation.recommendation.recommendation_id,
      },
      { timeoutMs: frozen.maxWallSeconds * 1_000 },
    );
    if (evaluationEnvelope.result === null) {
      throw new Error("recommendation.evaluate returned no result");
    }
    const report = `${JSON.stringify({
      status: "completed",
      target_metric: frozen.targetMetric,
      budgets: {
        max_experiments: frozen.maxExperiments,
        max_wall_seconds: frozen.maxWallSeconds,
        max_provider_calls: frozen.maxProviderCalls,
        max_provider_tokens: frozen.maxProviderTokens,
      },
      provider: {
        id: providerRuntime.model.provider,
        model: providerRuntime.model.id,
      },
      recommendation: recommendation.recommendation,
      evaluation: evaluationEnvelope.result,
      usage: controller.stats(),
      recommendation_run: {
        status: "completed",
        output_dir: runPaths.runRoot,
        state: runPaths.statePath,
        recommendation_root: runPaths.recommendationRoot,
      },
      provider_log: providerLog.path,
      session_directory: runPaths.sessionDirectory,
    }, null, 2)}\n`;
    await writeRecommendationRunState(runPaths, "completed", {
      recommendation: recommendation.recommendation,
      evaluation: evaluationEnvelope.result,
      usage: controller.stats(),
    });
    (options.stdout ?? process.stdout).write(report);
  } catch (error) {
    await writeRecommendationRunState(runPaths, "aborted", {
      reason: error instanceof Error ? error.message : String(error),
    });
    throw error;
  } finally {
    try {
      await worker?.stop();
    } finally {
      await executionEnvironment.cleanup();
    }
  }
  return 0;
}

export function parseRecommendationCliArgs(args: readonly string[]): FrozenRecommendationCliArgs {
  const values = new Map<keyof FrozenRecommendationCliArgs, string>();
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index];
    const value = args[index + 1];
    if (flag === undefined || !(flag in flags)) {
      throw new Error(`Unknown recommendation option '${flag ?? ""}'.\n${usage()}`);
    }
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`${flag} requires a value.`);
    }
    const name = flags[flag as keyof typeof flags];
    if (values.has(name)) throw new Error(`${flag} may only be provided once.`);
    values.set(name, value);
  }

  const maxExperiments = positiveInteger(required(values, "maxExperiments"), "--max-experiments");
  if (maxExperiments !== 1) {
    throw new Error("--max-experiments must be 1 for one-shot recommendation.");
  }
  const maxProviderCalls = positiveInteger(
    required(values, "maxProviderCalls"),
    "--max-provider-calls",
  );
  if (maxProviderCalls < 2) {
    throw new Error("--max-provider-calls must allow recommendation context and freeze calls.");
  }
  const targetMetric = required(values, "targetMetric");
  requireOfficialMetric(targetMetric, "--target-metric");
  return {
    scopePath: required(values, "scopePath"),
    targetMetric,
    maxExperiments: 1,
    maxWallSeconds: positiveInteger(
      required(values, "maxWallSeconds"),
      "--max-wall-seconds",
    ),
    maxProviderCalls,
    maxProviderTokens: positiveInteger(
      required(values, "maxProviderTokens"),
      "--max-provider-tokens",
    ),
    providerLogPath: required(values, "providerLogPath"),
    sessionDirectory: required(values, "sessionDirectory"),
    repositoryRoot: required(values, "repositoryRoot"),
    pythonExecutable: required(values, "pythonExecutable"),
  };
}

function parseFrozenRecommendationScope(scope: Record<string, unknown>): FrozenRecommendationScope {
  expectNoUnknownFields(
    scope,
    [
      ...FROZEN_SCOPE_FIELDS,
      ...OPTIONAL_FROZEN_SCOPE_FIELDS,
    ],
    "scope",
  );
  const planning = record(scope.planning_unit, "planning_unit");
  const curve = planning.curve;
  if (curve !== null && (typeof curve !== "string" || !curve.trim())) {
    throw new Error("planning_unit.curve must be a non-empty string or null.");
  }
  const schema = stringField(scope, "schema");
  if (!["naive", "all_in_one", "zero_shot"].includes(schema)) {
    throw new Error("scope.schema must be naive, all_in_one, or zero_shot.");
  }
  const membership = scope.zero_shot_membership;
  if (membership !== null && (typeof membership !== "string" || !membership.trim())) {
    throw new Error("scope.zero_shot_membership must be a non-empty string or null.");
  }
  const trainingCurves = stringArray(planning, "training_curves");
  const heldOutCurves = stringArray(planning, "held_out_curves");
  validatePlanningUnit(schema, curve, trainingCurves, heldOutCurves);
  if (schema === "zero_shot" && membership === null) {
    throw new Error("zero_shot recommendation requires frozen membership.");
  }
  if (schema !== "zero_shot" && membership !== null) {
    throw new Error("zero_shot membership requires schema zero_shot.");
  }
  const targetMetric = stringField(scope, "target_metric");
  requireOfficialMetric(targetMetric, "scope.target_metric");
  const agentMemoryRoot = optionalStringField(scope, "agent_memory_root");
  return {
    dataset: stringField(scope, "dataset"),
    schema,
    target_metric: targetMetric,
    planning_unit: {
      curve,
      training_curves: trainingCurves,
      held_out_curves: heldOutCurves,
    },
    dataset_root: stringField(scope, "dataset_root"),
    dataset_source_root: stringField(scope, "dataset_source_root"),
    dataset_dir: stringField(scope, "dataset_dir"),
    dataset_source: stringField(scope, "dataset_source"),
    dataset_revision: stringField(scope, "dataset_revision"),
    profile_dir: stringField(scope, "profile_dir"),
    ...(agentMemoryRoot === undefined ? {} : { agent_memory_root: agentMemoryRoot }),
    zero_shot_membership: membership,
  };
}

function validateFrozenScope(
  scope: FrozenRecommendationScope,
  frozen: FrozenRecommendationCliArgs,
): void {
  if (scope.dataset.toUpperCase() === "AIOPS") {
    throw new Error("AIOPS cannot enter one-shot recommendation prototype.");
  }
  if (scope.target_metric !== frozen.targetMetric) {
    throw new Error("--target-metric must match scope.target_metric.");
  }
}

function createRunScope(
  scope: FrozenRecommendationScope,
  frozen: FrozenRecommendationCliArgs,
  paths: RecommendationRunPaths,
): ResearchScope {
  return {
    ...scope,
    experiment_root: paths.recommendationRoot,
    experiment_index: 1,
    max_experiments: frozen.maxExperiments,
    max_wall_seconds: frozen.maxWallSeconds,
  };
}

function recommendationRunPaths(frozen: FrozenRecommendationCliArgs): RecommendationRunPaths {
  const providerLogPath = resolve(frozen.providerLogPath);
  const runRoot = dirname(providerLogPath);
  return {
    runRoot,
    statePath: join(runRoot, "recommendation-run.json"),
    providerLogPath,
    sessionDirectory: resolve(frozen.sessionDirectory),
    recommendationRoot: join(runRoot, "recommendations"),
  };
}

async function assertNewRecommendationRun(paths: RecommendationRunPaths): Promise<void> {
  await assertMissing(paths.providerLogPath, "provider log");
  await assertMissing(paths.statePath, "Recommendation Run state");
  await assertMissing(paths.recommendationRoot, "Recommendation output");
  await assertEmptyDirectory(paths.sessionDirectory, "Pi session directory");
}

async function assertMissing(path: string, label: string): Promise<void> {
  try {
    await access(path);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  throw new Error(`${label} already exists; choose a new Recommendation Run output path.`);
}

async function assertEmptyDirectory(path: string, label: string): Promise<void> {
  let entries: string[];
  try {
    entries = await readdir(path);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  if (entries.length > 0) {
    throw new Error(`${label} already exists and is not empty; choose a new Recommendation Run output path.`);
  }
}

async function writeRecommendationRunState(
  paths: RecommendationRunPaths,
  status: "running" | "completed" | "aborted",
  details: Record<string, unknown>,
): Promise<void> {
  await mkdir(paths.runRoot, { recursive: true });
  await writeFile(
    paths.statePath,
    `${JSON.stringify({
      status,
      provider_log: paths.providerLogPath,
      session_directory: paths.sessionDirectory,
      recommendation_root: paths.recommendationRoot,
      details,
    }, null, 2)}\n`,
    "utf8",
  );
}

function validatePlanningUnit(
  schema: string,
  curve: string | null,
  trainingCurves: readonly string[],
  heldOutCurves: readonly string[],
): void {
  if (trainingCurves.length === 0) {
    throw new Error("planning_unit.training_curves must be non-empty.");
  }
  if (
    new Set(trainingCurves).size !== trainingCurves.length ||
    new Set(heldOutCurves).size !== heldOutCurves.length
  ) {
    throw new Error("planning unit curve members must be unique.");
  }
  if (trainingCurves.some((member) => heldOutCurves.includes(member))) {
    throw new Error("planning unit training and held-out curves must be disjoint.");
  }
  if (schema === "naive") {
    if (curve === null || trainingCurves.length !== 1 || trainingCurves[0] !== curve) {
      throw new Error("naive planning requires one curve as its sole training member.");
    }
    if (heldOutCurves.length > 0) {
      throw new Error("naive planning does not accept held-out curves.");
    }
  } else if (schema === "all_in_one") {
    if (curve !== null || heldOutCurves.length > 0) {
      throw new Error("all_in_one planning requires dataset scope without held-out curves.");
    }
  } else if (curve !== null || heldOutCurves.length === 0) {
    throw new Error("zero_shot planning requires dataset scope and held-out curves.");
  }
}

function requireOfficialMetric(value: string, name: string): void {
  if (!OFFICIAL_METRICS.has(value)) {
    throw new Error(`${name} must be one of the eight official EasyTSAD metrics.`);
  }
}

function expectNoUnknownFields(
  source: Record<string, unknown>,
  expected: readonly string[],
  name: string,
): void {
  const unknown = Object.keys(source).filter((field) => !expected.includes(field));
  if (unknown.length > 0) throw new Error(`Unknown ${name} field: ${unknown[0]}.`);
  for (const field of FROZEN_SCOPE_FIELDS) {
    if (!(field in source)) throw new Error(`Missing ${name} field: ${field}.`);
  }
}

function required(
  values: ReadonlyMap<keyof FrozenRecommendationCliArgs, string>,
  name: keyof FrozenRecommendationCliArgs,
): string {
  const value = values.get(name)?.trim();
  if (!value) {
    const flag = Object.entries(flags).find(([, field]) => field === name)?.[0] ?? name;
    throw new Error(`${flag} is required.`);
  }
  return value;
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive safe integer.`);
  }
  return parsed;
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function stringField(source: Record<string, unknown>, name: string): string {
  const value = source[name];
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`scope.${name} must be a non-empty string.`);
  }
  return value;
}

function optionalStringField(source: Record<string, unknown>, name: string): string | undefined {
  const value = source[name];
  if (value === undefined) return undefined;
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`scope.${name} must be a non-empty string when provided.`);
  }
  return value;
}

function stringArray(source: Record<string, unknown>, name: string): string[] {
  const value = source[name];
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || !item.trim())) {
    throw new Error(`scope.${name} must be an array of non-empty strings.`);
  }
  return [...value] as string[];
}

async function readUtf8(path: string): Promise<string> {
  return await readFile(path, "utf8");
}

function createPythonWorker(options: PythonWorkerOptions): RecommendationCliWorker {
  return new PythonWorkerClient(options);
}

function pythonWorkerEnvironment(environment: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const workerEnvironment = { ...environment };
  for (const profile of PROVIDER_PROFILES) {
    for (const name of providerDefinition(profile).keyEnvCandidates) {
      delete workerEnvironment[name];
    }
  }
  return workerEnvironment;
}

function usage(): string {
  return [
    "Usage: npm run recommend --",
    "  --scope <scope.json> --target-metric <official-metric>",
    "  --max-experiments 1 --max-wall-seconds <seconds>",
    "  --max-provider-calls <calls> --max-provider-tokens <tokens>",
    "  --provider-log <new-run-dir>/provider.jsonl --session-dir <new-run-dir>/sessions",
    "  --repository-root <directory> --python <python-executable>",
    "",
  ].join("\n");
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runRecommendationCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`TimeSigma recommendation failed: ${message}\n`);
    process.exitCode = 1;
  });
}
