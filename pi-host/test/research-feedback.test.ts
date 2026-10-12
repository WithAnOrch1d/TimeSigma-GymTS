import { rmSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemorySessionRepo } from "@earendil-works/pi-agent-core";
import {
  createModels,
  fauxAssistantMessage,
  fauxProvider,
  fauxToolCall,
  type Context,
} from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import { createAuditedModels } from "../src/audited-models.js";
import { loadTimeSigmaProviderConfig } from "../src/provider-config.js";
import { ProviderLog } from "../src/provider-log.js";
import { createAuditedProviderController } from "../src/provider-runtime.js";
import { DomainToolError } from "../src/python-worker.js";
import { runResearchFeedback } from "../src/research-feedback.js";
import type { ResearchScope, ResearchWorker } from "../src/research-agent.js";
import type { JsonValue, ToolCallOptions, ToolName, ToolResultEnvelope } from "../src/protocol.js";

const temporaryDirectories: string[] = [];
type FauxResponse = Parameters<ReturnType<typeof fauxProvider>["setResponses"]>[0][number];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("Evidence-first research feedback", () => {
  it("matches a recovered pending execution by backend ID and preserves its actual lineage", async () => {
    class AliasWorker extends FeedbackWorker {
      override async callSuccess(toolName: ToolName, payload: Record<string, JsonValue>, options?: ToolCallOptions) {
        return super.callSuccess(toolName, toolName === "experiment.run" ? {
          ...payload, experiment_id: "persisted-id", baseline_evidence_id: "component/AR/naive/TODS/0",
        } : payload, options);
      }
    }
    const fixture = await feedbackFixture([
      tool("experiment_run", runActionArgs("new-request-id", "component/LOF/naive/TODS/0", { dimension: "window", value: 16 }), "recover"),
      tool("research_complete_step", finishStopArgs("persisted-id"), "commit"),
    ], new AliasWorker(), { maxCalls: 2, maxExperiments: 1 });
    await expect(fixture.result).resolves.toMatchObject({
      stop_reason: "experiment_budget", experiments: [{
        experiment_id: "persisted-id", baseline_evidence_id: "component/AR/naive/TODS/0",
        evidence_id: "research-step/TODS/naive/0/persisted-id",
      }],
    });
    expect(fixture.worker.executionCount).toBe(1);
  });

  it("enforces the episode wall budget while a Memory tool is pending", async () => {
    const faux = fauxProvider();
    const models = createModels();
    models.setProvider(faux.provider);
    faux.setResponses([tool("memory_search", {
      query: "", filters: [], order: "evidence_id", offset: 0, limit: 2,
    }, "slow-memory")]);
    let aborted = false;
    const worker: ResearchWorker = {
      async callSuccess(_tool, _payload, _options, signal) {
        return await new Promise((_resolve, reject) => {
          const fallback = setTimeout(() => reject(new Error("tool did not receive cancellation")), 4000);
          signal?.addEventListener("abort", () => {
            aborted = true;
            clearTimeout(fallback);
            reject(new Error("cancelled Memory read"));
          }, { once: true });
        });
      },
    };
    const sessions = new InMemorySessionRepo();
    await expect(runResearchFeedback({
      models, model: faux.getModel(), worker,
      scope: { ...researchScope(), max_wall_seconds: 1 },
      createSession: async () => await sessions.create({ id: "wall-budget" }),
      firstPrompt: "Inspect Memory within the frozen time budget.",
    })).resolves.toEqual({ experiments: [], stop_reason: "wall_time_budget", unfinished_experiments: [] });
    expect(aborted).toBe(true);
  });

  it("returns the provider budget stop without launching an experiment or fabricating diagnosis", async () => {
    const fixture = await feedbackFixture([
      tool("research_context", {}, "context-before-budget"),
      tool("research_finish_episode", { reason: "no_reasonable_next_step", rationale: "Cannot fund another request." }, "would-exceed-budget"),
    ], new FeedbackWorker(), { maxCalls: 1 });
    await expect(fixture.result).resolves.toMatchObject({ experiments: [], stop_reason: "provider_budget", unfinished_experiments: [] });
    expect(fixture.controller.stats().providerAttempts).toBe(1);
    expect(fixture.worker.calls.map(call => call.tool)).toEqual(["research.context"]);
  });

  it("reports an evaluated but undiagnosed experiment when Provider budget ends", async () => {
    const fixture = await feedbackFixture([
      tool("experiment_run", runActionArgs("awaiting-diagnosis", "component/LOF/naive/TODS/0", { dimension: "hyperparameter", parameter: "n_neighbors", value: 10 }), "run-at-limit"),
      tool("research_complete_step", finishStopArgs("awaiting-diagnosis"), "unfunded-diagnosis"),
    ], new FeedbackWorker(), { maxCalls: 1 });
    await expect(fixture.result).resolves.toMatchObject({ experiments: [], stop_reason: "provider_budget", unfinished_experiments: ["awaiting-diagnosis"] });
    expect(fixture.worker.calls.map(call => call.tool)).toEqual(["experiment.run"]);
    expect(fixture.controller.stats().providerAttempts).toBe(1);
  });

  it("runs a generic loop without binding the next turn to the previous next_action", async () => {
    const secondTurnContexts: Context[] = [];
    const firstNextAction = { dimension: "window", value: 15 };
    const independentSecondAction = {
      dimension: "hyperparameter",
      parameter: "n_neighbors",
      value: 30,
    };
    const fixture = await feedbackFixture([
      tool("research_context", {}, "context-1"),
      tool(
        "experiment_run",
        runActionArgs("experiment-1", "component/LOF/naive/TODS/0", {
          dimension: "hyperparameter",
          parameter: "n_neighbors",
          value: 10,
        }),
        "run-1",
      ),
      tool("research_complete_step", finishActionArgs("experiment-1", firstNextAction), "finish-1"),
      (context) => {
        secondTurnContexts.push(context);
        return tool("research_context", {}, "context-2");
      },
      tool(
        "experiment_run",
        runActionArgs(
          "experiment-2",
          "component/LOF/naive/TODS/0",
          independentSecondAction,
        ),
        "run-2",
      ),
      tool("research_complete_step", finishStopArgs("experiment-2"), "finish-2"),
      tool("research_finish_episode", finishEpisodeArgs(), "finish-episode"),
    ], new FeedbackWorker(), { maxCalls: 7 });

    await expect(fixture.result).resolves.toMatchObject({
      experiments: [
        {
          experiment_id: "experiment-1",
          baseline_evidence_id: "component/LOF/naive/TODS/0",
          next_action: { dimension: "window", parameter: null, value: 15 },
        },
        {
          experiment_id: "experiment-2",
          baseline_evidence_id: "component/LOF/naive/TODS/0",
          action: independentSecondAction,
        },
      ],
    });
    expect(fixture.createdSessions).toEqual([1, 2, 3]);
    expect(JSON.stringify(secondTurnContexts[0]?.messages)).toContain(
      "Do not treat the recorded next_action as mandatory Harness routing.",
    );
    expect(JSON.stringify(secondTurnContexts[0]?.messages)).not.toContain("next_action exactly");
    expect(fixture.worker.calls.map((call) => call.tool)).toEqual([
      "research.context", "experiment.run", "research.complete_step",
      "research.context", "experiment.run", "research.complete_step",
      "research.finish_episode",
    ]);
    expect(fixture.worker.calls[3]?.payload.experiment_index).toBe(2);
    expect(fixture.controller.stats()).toMatchObject({
      logicalCalls: 7,
      providerAttempts: 7,
      paused: false,
    });
  });

  it("reconsiders duplicate reuse without consuming an execution slot or recording a turn", async () => {
    const fixture = await feedbackFixture([
      tool("research_context", {}, "context-1"),
      tool(
        "experiment_run",
        runActionArgs("try-existing-ar", "component/LOF/naive/TODS/0", {
          dimension: "algorithm",
          value: "AR",
        }),
        "run-reuse",
      ),
      tool("research_context", {}, "context-2"),
      tool(
        "experiment_run",
        runActionArgs("experiment-1", "component/AR/naive/TODS/0", {
          dimension: "window",
          value: 8,
        }),
        "run-new",
      ),
      tool("research_complete_step", finishStopArgs("experiment-1"), "finish-1"),
      tool("research_finish_episode", finishEpisodeArgs(), "finish-episode"),
    ], new FeedbackWorker({ reuseExperimentIds: ["try-existing-ar"] }), { maxCalls: 6 });

    await expect(fixture.result).resolves.toMatchObject({
      experiments: [
        {
          experiment_id: "experiment-1",
          baseline_evidence_id: "component/AR/naive/TODS/0",
        },
      ],
    });
    const runCalls = fixture.worker.calls.filter((call) => call.tool === "experiment.run");
    expect(runCalls).toHaveLength(2);
    expect(fixture.worker.executionCount).toBe(1);
    expect(runCalls[1]?.payload.baseline_evidence_id).toBe("component/AR/naive/TODS/0");
    expect(fixture.worker.calls[2]?.payload.experiment_index).toBe(1);
  });

  it("treats ResearchStep completion as separate from episode finish", async () => {
    const fixture = await feedbackFixture([
      tool("research_context", {}, "context-1"),
      tool(
        "experiment_run",
        runActionArgs("experiment-1", "component/LOF/naive/TODS/0", {
          dimension: "hyperparameter",
          parameter: "n_neighbors",
          value: 10,
        }),
        "run-1",
      ),
      tool("research_complete_step", finishStopArgs("experiment-1"), "finish-1"),
      tool("research_finish_episode", finishEpisodeArgs(), "finish-episode"),
    ], new FeedbackWorker(), { maxCalls: 4, maxExperiments: 3 });

    await expect(fixture.result).resolves.toMatchObject({
      experiments: [{ experiment_id: "experiment-1" }],
    });
    expect(fixture.createdSessions).toEqual([1, 2]);
    expect(fixture.worker.calls.map((call) => call.tool)).toEqual([
      "research.context", "experiment.run", "research.complete_step",
      "research.finish_episode",
    ]);
  });

  it("records every completed ResearchStep in one provider tool batch", async () => {
    const fixture = await feedbackFixture([
      fauxAssistantMessage([
        fauxToolCall("research_context", {}, { id: "context-1" }),
        fauxToolCall(
          "experiment_run",
          runActionArgs("experiment-1", "component/LOF/naive/TODS/0", {
            dimension: "hyperparameter",
            parameter: "n_neighbors",
            value: 10,
          }),
          { id: "run-1" },
        ),
        fauxToolCall(
          "research_complete_step",
          finishActionArgs("experiment-1", {
            dimension: "hyperparameter",
            parameter: "n_neighbors",
            value: 15,
          }),
          { id: "finish-1" },
        ),
        fauxToolCall(
          "experiment_run",
          runActionArgs("experiment-2", "component/LOF/naive/TODS/0", {
            dimension: "hyperparameter",
            parameter: "n_neighbors",
            value: 15,
          }),
          { id: "run-2" },
        ),
        fauxToolCall(
          "research_complete_step",
          finishStopArgs("experiment-2"),
          { id: "finish-2" },
        ),
      ], { stopReason: "toolUse" }),
      fauxAssistantMessage("Done."),
    ], new FeedbackWorker(), { maxCalls: 2, maxExperiments: 2 });

    await expect(fixture.result).resolves.toMatchObject({
      experiments: [
        { experiment_id: "experiment-1" },
        { experiment_id: "experiment-2" },
      ],
    });
    expect(fixture.createdSessions).toEqual([1]);
    expect(fixture.worker.executionCount).toBe(2);
  });

  it("can complete more than two Research Experiments when budget allows", async () => {
    const fixture = await feedbackFixture([
      tool("research_context", {}, "context-1"),
      tool(
        "experiment_run",
        runActionArgs("experiment-1", "component/LOF/naive/TODS/0", {
          dimension: "hyperparameter",
          parameter: "n_neighbors",
          value: 10,
        }),
        "run-1",
      ),
      tool("research_complete_step", finishActionArgs("experiment-1", {
        dimension: "hyperparameter",
        parameter: "n_neighbors",
        value: 15,
      }), "finish-1"),
      tool("research_context", {}, "context-2"),
      tool(
        "experiment_run",
        runActionArgs("experiment-2", "component/LOF/naive/TODS/0", {
          dimension: "hyperparameter",
          parameter: "n_neighbors",
          value: 15,
        }),
        "run-2",
      ),
      tool("research_complete_step", finishActionArgs("experiment-2", {
        dimension: "hyperparameter",
        parameter: "n_neighbors",
        value: 25,
      }), "finish-2"),
      tool("research_context", {}, "context-3"),
      tool(
        "experiment_run",
        runActionArgs("experiment-3", "component/LOF/naive/TODS/0", {
          dimension: "hyperparameter",
          parameter: "n_neighbors",
          value: 25,
        }),
        "run-3",
      ),
      tool("research_complete_step", finishStopArgs("experiment-3"), "finish-3"),
    ], new FeedbackWorker(), { maxCalls: 9, maxExperiments: 3 });

    await expect(fixture.result).resolves.toMatchObject({
      experiments: [
        { experiment_id: "experiment-1" },
        { experiment_id: "experiment-2" },
        { experiment_id: "experiment-3" },
      ],
    });
    expect(fixture.createdSessions).toEqual([1, 2, 3]);
    expect(fixture.worker.executionCount).toBe(3);
  });

  it("rejects execution output that is not diagnosed in the same turn", async () => {
    const fixture = await feedbackFixture([
      tool("research_context", {}, "context-1"),
      tool(
        "experiment_run",
        runActionArgs("experiment-1", "component/LOF/naive/TODS/0", {
          dimension: "hyperparameter",
          parameter: "n_neighbors",
          value: 10,
        }),
        "run-1",
      ),
      fauxAssistantMessage("Stop before diagnosis."),
    ]);

    await expect(fixture.result).rejects.toThrow(
      "produced execution output without research_complete_step diagnosis",
    );
    expect(fixture.worker.executionCount).toBe(1);
  });

  it("allows a corrected proposal after a domain validation failure", async () => {
    const worker = new FeedbackWorker({ failFirstExperimentRun: true });
    const fixture = await feedbackFixture([
      tool("research_context", {}, "context-1"),
      tool(
        "experiment_run",
        runActionArgs("experiment-1", "component/LOF/naive/TODS/0", {
          dimension: "algorithm",
          value: "LOF",
        }),
        "run-invalid",
      ),
      tool(
        "experiment_run",
        runActionArgs("experiment-1", "component/LOF/naive/TODS/0", {
          dimension: "hyperparameter",
          parameter: "n_neighbors",
          value: 10,
        }),
        "run-corrected",
      ),
      tool("research_complete_step", finishStopArgs("experiment-1"), "finish-1"),
      tool("research_finish_episode", finishEpisodeArgs(), "finish-episode"),
    ], worker, { maxCalls: 5 });

    await expect(fixture.result).resolves.toMatchObject({
      experiments: [{ experiment_id: "experiment-1" }],
    });
    const runCalls = worker.calls.filter((call) => call.tool === "experiment.run");
    expect(runCalls).toHaveLength(2);
    expect(worker.executionCount).toBe(1);
  });

  it("rejects ordinary assistant text as an implicit episode finish", async () => {
    const fixture = await feedbackFixture([
      tool("research_context", {}, "context-1"),
      fauxAssistantMessage("Finish the research episode."),
    ], new FeedbackWorker(), { maxCalls: 2 });

    await expect(fixture.result).rejects.toThrow(
      "ended without research_finish_episode",
    );
    expect(fixture.worker.calls.map((call) => call.tool)).toEqual([
      "research.context",
    ]);
  });

  it("rejects AIOPS before opening a Provider turn", async () => {
    const fixture = await feedbackFixture(
      [fauxAssistantMessage("should not be read")],
      new FeedbackWorker(),
      { scope: { ...researchScope(), dataset: "AIOPS" } },
    );

    await expect(fixture.result).rejects.toThrow("AIOPS cannot enter iterative research");
    expect(fixture.createdSessions).toEqual([]);
    expect(fixture.worker.calls).toEqual([]);
  });
});

function tool(name: string, args: Record<string, JsonValue>, id: string) {
  return fauxAssistantMessage(fauxToolCall(name, args, { id }), { stopReason: "toolUse" });
}

function runActionArgs(
  experimentId: string,
  baseline: string,
  action: Record<string, JsonValue>,
): Record<string, JsonValue> {
  return {
    experiment_id: experimentId,
    preview_id: "unit-preview",
    hypothesis: "Apply one Evidence-first research action.",
    baseline_evidence_id: baseline,
    action,
    stopping_condition: "Stop after one EasyTSAD evaluation.",
  };
}

function finishActionArgs(
  experimentId: string,
  nextAction: Record<string, JsonValue>,
  diagnosisStatus: "supported" | "refuted" | "inconclusive" = "supported",
): Record<string, JsonValue> {
  return {
    experiment_id: experimentId,
    diagnosis_status: diagnosisStatus,
    diagnosis: "The target metric supports the recorded action.",
    next_action_rationale: "The result supports one bounded follow-up.",
    next_action: nextAction,
  };
}

function finishStopArgs(experimentId: string): Record<string, JsonValue> {
  return {
    experiment_id: experimentId,
    diagnosis_status: "supported",
    diagnosis: "The target metric reached the model's stopping judgment.",
    next_action_rationale: "No further experiment is warranted by this diagnosis.",
    next_action: { dimension: "stop", value: "target_saturated" },
  };
}

function finishEpisodeArgs(): Record<string, JsonValue> {
  return {
    reason: "target_saturated",
    rationale: "The episode is complete after the committed step.",
  };
}

interface RunOutcome {
  baselineValue: number;
  candidateValue: number;
}

interface WorkerOptions {
  failFirstExperimentRun?: boolean;
  reuseExperimentIds?: readonly string[];
  runOutcomes?: readonly RunOutcome[];
}

class FeedbackWorker implements ResearchWorker {
  readonly calls: Array<{ tool: ToolName; payload: Record<string, JsonValue> }> = [];
  private readonly reuseExperimentIds: Set<string>;
  private readonly runOutcomes: readonly RunOutcome[];
  private readonly runPayloads = new Map<string, Record<string, JsonValue>>();
  private readonly outcomes = new Map<string, RunOutcome>();
  private failFirstExperimentRun: boolean;
  private successfulRuns = 0;

  constructor(options: WorkerOptions = {}) {
    this.failFirstExperimentRun = options.failFirstExperimentRun ?? false;
    this.reuseExperimentIds = new Set(options.reuseExperimentIds ?? []);
    this.runOutcomes = options.runOutcomes ?? [
      { baselineValue: 0.2, candidateValue: 0.3 },
      { baselineValue: 0.3, candidateValue: 0.35 },
      { baselineValue: 0.35, candidateValue: 0.4 },
    ];
  }

  get executionCount(): number {
    return this.successfulRuns;
  }

  async callSuccess(
    tool: ToolName,
    payload: Record<string, JsonValue>,
    _options?: ToolCallOptions,
  ): Promise<ToolResultEnvelope> {
    this.calls.push({ tool, payload });
    if (tool === "experiment.run" && this.failFirstExperimentRun) {
      this.failFirstExperimentRun = false;
      throw new DomainToolError({
        protocol_version: "timesigma.bridge.v1",
        schema_version: 1,
        request_id: `request-${this.calls.length}`,
        status: "method_error",
        result: null,
        error: {
          code: "invalid_action",
          message: "action must change the baseline",
        },
      });
    }
    return {
      protocol_version: "timesigma.bridge.v1",
      schema_version: 1,
      request_id: `request-${this.calls.length}`,
      status: "success",
      result: this.resultFor(tool, payload),
      error: null,
    };
  }

  private resultFor(
    tool: ToolName,
    payload: Record<string, JsonValue>,
  ): Record<string, JsonValue> {
    if (tool === "research.context") return researchContext(this.successfulRuns);
    if (tool === "experiment.run") return this.runResult(payload);
    if (tool === "research.complete_step") return this.finishResult(payload);
    if (tool === "research.finish_episode") return this.finishEpisodeResult(payload);
    return {};
  }

  private runResult(payload: Record<string, JsonValue>): Record<string, JsonValue> {
    const experimentId = String(payload.experiment_id);
    if (this.reuseExperimentIds.has(experimentId)) {
      return {
        status: "reused",
        reused_evidence_id: "component/AR/naive/TODS/0",
        baseline_evidence_id: jsonField(payload, "baseline_evidence_id"),
        hypothesis: jsonField(payload, "hypothesis"),
        action: jsonField(payload, "action"),
        target_metric: "point-wise-f1",
        baseline_value: 0.2,
        candidate_value: 0.9,
        delta: 0.7,
        candidate_pipeline: { method: "AR", resolved_configuration: {} },
        official_metrics: { "point-wise-f1": 0.9 },
        execution_budget: this.budgetView(),
      };
    }
    const fallback = this.runOutcomes[this.runOutcomes.length - 1] ?? {
      baselineValue: 0.2,
      candidateValue: 0.3,
    };
    const outcome = this.runOutcomes[this.successfulRuns] ?? fallback;
    this.successfulRuns += 1;
    this.runPayloads.set(experimentId, payload);
    this.outcomes.set(experimentId, outcome);
    return {
      status: "executed_pending_diagnosis",
      experiment_id: experimentId,
      baseline_evidence_id: jsonField(payload, "baseline_evidence_id"),
      hypothesis: jsonField(payload, "hypothesis"),
      action: jsonField(payload, "action"),
      baseline_pipeline: { method: "LOF", resolved_configuration: {} },
      candidate_pipeline: { method: "LOF", resolved_configuration: {} },
      target_metric: "point-wise-f1",
      baseline_value: outcome.baselineValue,
      candidate_value: outcome.candidateValue,
      delta: outcome.candidateValue - outcome.baselineValue,
      official_metrics: { "point-wise-f1": outcome.candidateValue },
      execution_budget: this.budgetView(),
    };
  }

  private finishResult(payload: Record<string, JsonValue>): Record<string, JsonValue> {
    const experimentId = String(payload.experiment_id);
    const runPayload = this.runPayloads.get(experimentId);
    const outcome = this.outcomes.get(experimentId) ?? {
      baselineValue: 0.2,
      candidateValue: 0.3,
    };
    return {
      recorded: true,
      status: "committed",
      experiment_id: experimentId,
      evidence_id: `research-step/TODS/naive/0/${experimentId}`,
      baseline_evidence_id: runPayload?.baseline_evidence_id ?? "component/LOF/naive/TODS/0",
      action: runPayload?.action ?? null,
      diagnosis_status: jsonField(payload, "diagnosis_status"),
      target_metric: "point-wise-f1",
      baseline_value: outcome.baselineValue,
      candidate_value: outcome.candidateValue,
      delta: outcome.candidateValue - outcome.baselineValue,
      official_metrics: { "point-wise-f1": outcome.candidateValue },
      next_action: payload.next_action ?? null,
      next_action_rationale: payload.next_action_rationale ?? null,
      execution_budget: this.budgetView(),
    };
  }

  private finishEpisodeResult(payload: Record<string, JsonValue>): Record<string, JsonValue> {
    return {
      status: "finished",
      finish_reason: payload.reason ?? null,
      rationale: payload.rationale ?? null,
      execution_budget: this.budgetView(),
    };
  }

  private budgetView(): Record<string, JsonValue> {
    return {
      experiments_used: this.successfulRuns,
      max_experiments: 3,
      experiments_remaining: Math.max(0, 3 - this.successfulRuns),
    };
  }
}

async function feedbackFixture(
  responses: FauxResponse[],
  worker = new FeedbackWorker(),
  options: {
    maxCalls?: number;
    maxExperiments?: number;
    scope?: ResearchScope;
  } = {},
) {
  const faux = fauxProvider();
  const sourceModels = createModels();
  sourceModels.setProvider(faux.provider);
  faux.setResponses(responses);
  const directory = await mkdtemp(join(tmpdir(), "timesigma-feedback-"));
  temporaryDirectories.push(directory);
  const model = faux.getModel();
  const controller = createAuditedProviderController({
    config: loadTimeSigmaProviderConfig({}),
    model,
    source: sourceModels.streamSimple.bind(sourceModels),
    log: new ProviderLog(join(directory, "provider.jsonl")),
    budget: { maxCalls: options.maxCalls ?? 4, maxTokens: 200_000 },
  });
  const sessions = new InMemorySessionRepo();
  const createdSessions: number[] = [];
  const scope = {
    ...(options.scope ?? researchScope()),
    max_experiments: options.maxExperiments ?? options.scope?.max_experiments ?? 3,
  };
  const result = runResearchFeedback({
    models: createAuditedModels(sourceModels, controller),
    model,
    worker,
    scope,
    createSession: async (turnNumber) => {
      createdSessions.push(turnNumber);
      return await sessions.create({ id: `research-turn-${turnNumber}` });
    },
    firstPrompt: "Run the first frozen Evidence-first research hypothesis.",
    providerBudgetStopped: () => controller.budgetState().stopped,
  });
  return { controller, createdSessions, result, worker };
}

function jsonField(payload: Record<string, JsonValue>, name: string): JsonValue {
  const value = payload[name];
  if (value === undefined) throw new Error(`${name} is required`);
  return value;
}

function researchContext(experimentsUsed: number): Record<string, JsonValue> {
  return {
    task: {
      dataset: "TODS",
      schema: "naive",
      target_metric: "point-wise-f1",
      input_state: "raw_train",
      unit: { curve: "0", training_curves: ["0"], held_out_curves: [] },
    },
    profile: {
      input_state: "raw_train",
      curve_profile: { lag1_autocorrelation: 0.9, roughness: 0.1 },
      aggregate: null,
      zero_shot_split: null,
    },
    component_landscape: {
      fact_type: "current_exact_task_component_evidence",
      components: [
        { rank: 1, evidence_id: "component/AR/naive/TODS/0", method: "AR", target_metric_value: 0.9 },
        { rank: 2, evidence_id: "component/LOF/naive/TODS/0", method: "LOF", target_metric_value: 0.2 },
      ],
      best_value: 0.9,
      best_evidence_ids: ["component/AR/naive/TODS/0"],
      default_initial_baseline_evidence_id: "component/AR/naive/TODS/0",
    },
    research_state: {
      initial_baseline: {
        evidence_id: "component/AR/naive/TODS/0",
        pipeline: { method: "AR", data_params: {}, model_params: {} },
        target_value: 0.9,
      },
      global_best: {
        value: 0.9,
        evidence_ids: ["component/AR/naive/TODS/0"],
        incumbent_evidence_id: "component/AR/naive/TODS/0",
      },
      research_step_count: experimentsUsed,
      research_step_evidence_ids: [],
      research_steps: [],
      lineage_edges: [],
      runtime_budget: {
        experiment_index: experimentsUsed + 1,
        experiments_used: experimentsUsed,
        max_experiments: 3,
        experiments_remaining: Math.max(0, 3 - experimentsUsed),
        max_wall_seconds: 60,
      },
      stop_status: {
        can_continue: experimentsUsed < 3,
        stop_source: null,
        stop_reason: null,
      },
    },
    episode_feedback: [],
    relevant_history: [],
    executable_capabilities: [],
  };
}

function researchScope(): ResearchScope {
  return {
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
    agent_memory_root: "/runs/agent-memory",
    experiment_root: "/runs/pipelines",
    zero_shot_membership: null,
    experiment_index: 1,
    max_experiments: 3,
    max_wall_seconds: 60,
  };
}
