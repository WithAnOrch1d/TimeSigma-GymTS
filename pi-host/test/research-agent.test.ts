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
import {
  createRecommendationAgentHarness,
  createResearchAgentHarness,
  type ResearchScope,
  type ResearchWorker,
} from "../src/research-agent.js";
import type { JsonValue, ToolName, ToolResultEnvelope } from "../src/protocol.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("TimeSigma research AgentHarness", () => {
  it("passes a complete Pipeline through Pi and commits its research diagnosis", async () => {
    const faux = fauxProvider();
    const models = createModels();
    models.setProvider(faux.provider);
    const pipeline = {
      preprocessing: { scaling: { method: "z_score", fit_scope: "per_curve_train" }, temporal_transform: { method: "difference", order: 1 }, smoothing: { method: "none" } },
      detector: { method: "LOF", parameters: [{ name: "n_neighbors", value: 7 }] },
      ensemble: { method: "mean", detectors: [{ method: "OCSVM", parameters: [] }] },
    };
    const calls = [
      fauxToolCall("research_context", {}, { id: "context" }),
      fauxToolCall("experiment_run", {
        experiment_id: "mean-fusion",
        preview_id: "unit-preview",
        hypothesis: "A second frozen detector may add complementary scores.",
        baseline_evidence_id: "research-step/TODS/naive/0/lof-branch",
        action: { dimension: "pipeline", value: pipeline },
        stopping_condition: "One pre-registered fusion candidate.",
      }, { id: "experiment" }),
      fauxToolCall("research_complete_step", {
        experiment_id: "mean-fusion",
        diagnosis_status: "inconclusive",
        diagnosis: "The controlled result does not establish a transfer benefit.",
        next_action: { dimension: "pipeline", value: { ...pipeline, ensemble: { method: "none", detectors: [] } } },
        next_action_rationale: "Compare the standalone branch using existing Evidence.",
      }, { id: "complete" }),
    ];
    faux.setResponses(calls.map(call => fauxAssistantMessage(call, { stopReason: "toolUse" })));
    const worker = new RecordingWorker();
    const harness = createResearchAgentHarness({
      session: await new InMemorySessionRepo().create({ id: "pipeline-planning" }),
      models,
      model: faux.getModel(),
      toolContext: { worker, scope: researchScope() },
    });
    await harness.prompt("Plan and evaluate one complete Pipeline.");
    expect(worker.calls.map(call => call.tool)).toEqual([
      "research.context", "experiment.run", "research.complete_step",
    ]);
    expect(worker.calls[1]?.payload.enforce_preprocessing_categories).toBe(true);
    expect(worker.calls[1]?.payload.action).toEqual({
      dimension: "pipeline", parameter: null, value: pipeline,
    });
    expect(worker.calls[2]?.payload.next_action).toMatchObject({
      dimension: "pipeline", parameter: null,
      value: { ensemble: { method: "none", detectors: [] } },
    });
  });

  it("runs context, one experiment, and terminating diagnosis through a real Pi Session", async () => {
    const faux = fauxProvider();
    const sourceModels = createModels();
    sourceModels.setProvider(faux.provider);
    const model = faux.getModel();
    const observedContexts: Context[] = [];
    faux.setResponses([
      (context) => {
        observedContexts.push(context);
        return fauxAssistantMessage(fauxToolCall("research_context", {}, { id: "context-call" }), {
          stopReason: "toolUse",
        });
      },
      (context) => {
        observedContexts.push(context);
        return fauxAssistantMessage(
          fauxToolCall(
            "experiment_run",
            {
              experiment_id: "lof-neighbors-25",
              preview_id: "unit-preview",
              hypothesis: "A longer window improves the target metric.",
              baseline_evidence_id: "component/LOF/naive/TODS/0",
              action: {
                dimension: "window",
                value: 25,
              },
              stopping_condition: "Stop after one EasyTSAD run.",
            },
            { id: "experiment-call" },
          ),
          { stopReason: "toolUse" },
        );
      },
      (context) => {
        observedContexts.push(context);
        return fauxAssistantMessage(
          fauxToolCall(
            "research_complete_step",
            {
              experiment_id: "lof-neighbors-25",
              diagnosis_status: "supported",
              diagnosis: "The frozen target metric improved against the baseline.",
              next_action_rationale: "The improvement supports testing normalization next.",
              next_action: {
                dimension: "preprocess",
                value: "z-score",
              },
            },
            { id: "finish-call" },
          ),
          { stopReason: "toolUse" },
        );
      },
    ]);

    const directory = await mkdtemp(join(tmpdir(), "timesigma-research-agent-"));
    temporaryDirectories.push(directory);
    const controller = createAuditedProviderController({
      config: loadTimeSigmaProviderConfig({}),
      model,
      source: sourceModels.streamSimple.bind(sourceModels),
      log: new ProviderLog(join(directory, "provider.jsonl")),
      budget: { maxCalls: 3, maxTokens: 100_000 },
    });
    const models = createAuditedModels(sourceModels, controller);
    const session = await new InMemorySessionRepo().create({ id: "research-session" });
    const worker = new RecordingWorker();
    const scope = researchScope();
    const harness = createResearchAgentHarness({
      session,
      models,
      model,
      toolContext: { worker, scope },
    });
    const toolEvents: string[] = [];
    harness.on("tool_call", (event) => {
      toolEvents.push(event.toolName);
      return undefined;
    });

    const finalMessage = await harness.prompt("Test one bounded LOF hypothesis.");

    expect(finalMessage.stopReason).toBe("toolUse");
    expect(faux.state.callCount).toBe(3);
    expect(faux.getPendingResponseCount()).toBe(0);
    expect(controller.stats()).toMatchObject({ logicalCalls: 3, providerAttempts: 3, paused: false });
    expect(toolEvents).toEqual([
      "research_context",
      "experiment_run",
      "research_complete_step",
    ]);
    expect(worker.calls.map((call) => call.tool)).toEqual([
      "research.context",
      "experiment.run",
      "research.complete_step",
    ]);
    expect(worker.calls[0]?.payload).toEqual(scope);
    expect(worker.calls[1]?.payload).toMatchObject({
      ...scope,
      experiment_id: "lof-neighbors-25",
      action: { dimension: "window", parameter: null, value: 25 },
    });
    expect(worker.calls[1]?.options).toEqual({ timeoutMs: 60_000 });
    expect(worker.calls[2]?.payload).toMatchObject({
      ...scope,
      experiment_id: "lof-neighbors-25",
      diagnosis_status: "supported",
      next_action_rationale: "The improvement supports testing normalization next.",
      next_action: { dimension: "preprocess", parameter: null, value: "z-score" },
    });

    const sessionContext = await session.buildContext();
    expect(JSON.stringify(sessionContext.messages)).not.toContain('"parameter":null');
    expect(sessionContext.messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "toolResult",
      "assistant",
      "toolResult",
      "assistant",
      "toolResult",
    ]);
    expect(JSON.stringify(sessionContext.messages)).toContain("component_landscape");
    expect(JSON.stringify(sessionContext.messages)).toContain("research_state");
    expect(JSON.stringify(sessionContext.messages)).toContain("episode_feedback");
    expect(JSON.stringify(sessionContext.messages)).toContain("research_capabilities");
    expect(JSON.stringify(sessionContext.messages)).not.toContain("evidence_policy");
    expect(JSON.stringify(sessionContext.messages)).not.toContain(
      "remaining_improvement_headroom",
    );
    expect(
      harness.getTools().find((tool) => tool.name === "experiment_run")?.executionMode,
    ).toBe("sequential");

    const visibleTools = observedContexts[0]?.tools ?? [];
    expect(visibleTools.map((tool) => tool.name)).toEqual([
      "experiment_preview",
      "memory_search",
      "memory_read",
      "memory_compare",
      "capability_lookup",
      "research_context",
      "experiment_run",
      "research_complete_step",
      "research_finish_episode",
    ]);
    expect(visibleTools.every((tool) => /^[A-Za-z0-9_-]+$/.test(tool.name))).toBe(true);
    for (const tool of visibleTools) {
      expect((tool.parameters as { type?: string }).type).toBe("object");
      expect(deepSeekStrictSchemaIssues(tool.parameters)).toEqual([]);
    }
    const contextTool = visibleTools.find((tool) => tool.name === "research_context");
    const experimentTool = visibleTools.find((tool) => tool.name === "experiment_run");
    expect(contextTool?.parameters).toMatchObject({ properties: {} });
    expect(JSON.stringify(experimentTool?.parameters)).not.toContain("dataset_root");
    expect(observedContexts[0]?.systemPrompt).toContain("non-AIOPS");
    expect(observedContexts[0]?.systemPrompt).toContain("single frozen anomaly score");
    expect(observedContexts[0]?.systemPrompt).toContain(
      "must differ from the baseline's current value",
    );
    expect(observedContexts[0]?.systemPrompt).toContain("component_landscape");
    expect(observedContexts[0]?.systemPrompt).toContain("research_state");
    expect(observedContexts[0]?.systemPrompt).toContain("research_capabilities");
    expect(observedContexts[0]?.systemPrompt).toContain(
      "Historical other-task Evidence is reasoning context only",
    );
    expect(observedContexts[0]?.systemPrompt).toContain(
      "first genuinely new execution must use research_state.initial_baseline.evidence_id",
    );
    expect(observedContexts[0]?.systemPrompt).not.toContain("evidence_policy");
    expect(observedContexts[0]?.systemPrompt).not.toContain(
      "remaining improvement headroom",
    );
    expect(observedContexts[0]?.systemPrompt).not.toContain(
      "Current direct Evidence has priority",
    );
    expect(observedContexts[0]?.systemPrompt).toContain(
      "why the baseline was chosen",
    );
    expect(observedContexts[0]?.systemPrompt).toContain("Pi Session is only conversational");
  });

  it("passes preprocessing PipelineAction through experiment_run", async () => {
    const faux = fauxProvider();
    const sourceModels = createModels();
    sourceModels.setProvider(faux.provider);
    const model = faux.getModel();
    const observedContexts: Context[] = [];
    faux.setResponses([
      (context) => {
        observedContexts.push(context);
        return fauxAssistantMessage(fauxToolCall("research_context", {}, { id: "context-call" }), {
          stopReason: "toolUse",
        });
      },
      (context) => {
        observedContexts.push(context);
        return fauxAssistantMessage(
          fauxToolCall(
            "experiment_run",
            {
              experiment_id: "robust-lof",
              preview_id: "unit-preview",
              hypothesis: "Robust scaling may reduce outlier leverage before LOF.",
              baseline_evidence_id: "component/LOF/naive/TODS/0",
              action: {
                operation: "add_component",
                stage: "preprocessing",
                component_id: "robust_scaler",
              },
              stopping_condition: "Stop after one Pipeline MVP evaluation.",
            },
            { id: "pipeline-experiment-call" },
          ),
          { stopReason: "toolUse" },
        );
      },
      (context) => {
        observedContexts.push(context);
        return fauxAssistantMessage(
          fauxToolCall(
            "research_finish_episode",
            {
              reason: "no_reasonable_next_step",
              rationale: "Pipeline MVP smoke completed.",
            },
            { id: "finish-episode-call" },
          ),
          { stopReason: "toolUse" },
        );
      },
    ]);

    const directory = await mkdtemp(join(tmpdir(), "timesigma-pipeline-agent-"));
    temporaryDirectories.push(directory);
    const controller = createAuditedProviderController({
      config: loadTimeSigmaProviderConfig({}),
      model,
      source: sourceModels.streamSimple.bind(sourceModels),
      log: new ProviderLog(join(directory, "provider.jsonl")),
      budget: { maxCalls: 3, maxTokens: 100_000 },
    });
    const models = createAuditedModels(sourceModels, controller);
    const session = await new InMemorySessionRepo().create({ id: "pipeline-session" });
    const worker = new RecordingWorker();
    const scope = researchScope();
    const harness = createResearchAgentHarness({
      session,
      models,
      model,
      toolContext: { worker, scope },
    });

    await harness.prompt("Test one preprocessing PipelineAction.");

    expect(worker.calls.map((call) => call.tool)).toEqual([
      "research.context",
      "experiment.run",
      "research.finish_episode",
    ]);
    expect(worker.calls[1]?.payload).toMatchObject({
      ...scope,
      experiment_id: "robust-lof",
      pipeline_action: {
        operation: "add_component",
        stage: "preprocessing",
        component_id: "robust_scaler",
      },
    });
    expect(worker.calls[1]?.payload).not.toHaveProperty("action");
    expect(JSON.stringify(await session.buildContext())).toContain(
      "changed_research_dimension",
    );

    const visibleTools = observedContexts[0]?.tools ?? [];
    const experimentTool = visibleTools.find((tool) => tool.name === "experiment_run");
    expect(JSON.stringify(experimentTool?.parameters)).toContain("add_component");
    for (const tool of visibleTools) {
      expect(deepSeekStrictSchemaIssues(tool.parameters)).toEqual([]);
    }
  });

  it("runs one-shot recommendation without exposing experiment tools or current metrics", async () => {
    const faux = fauxProvider();
    const sourceModels = createModels();
    sourceModels.setProvider(faux.provider);
    const model = faux.getModel();
    const observedContexts: Context[] = [];
    faux.setResponses([
      (context) => {
        observedContexts.push(context);
        return fauxAssistantMessage(fauxToolCall("recommendation_context", {}, { id: "context-call" }), {
          stopReason: "toolUse",
        });
      },
      (context) => {
        observedContexts.push(context);
        return fauxAssistantMessage(
          fauxToolCall(
            "recommendation_freeze",
            {
              method: "AR",
              action: {
                dimension: "default",
                value: "official_default",
              },
              rationale: "The train-side Profile shows temporal persistence, so AR is the one-shot recommendation.",
            },
            { id: "freeze-call" },
          ),
          { stopReason: "toolUse" },
        );
      },
    ]);

    const directory = await mkdtemp(join(tmpdir(), "timesigma-recommendation-agent-"));
    temporaryDirectories.push(directory);
    const controller = createAuditedProviderController({
      config: loadTimeSigmaProviderConfig({}),
      model,
      source: sourceModels.streamSimple.bind(sourceModels),
      log: new ProviderLog(join(directory, "provider.jsonl")),
      budget: { maxCalls: 2, maxTokens: 100_000 },
    });
    const models = createAuditedModels(sourceModels, controller);
    const session = await new InMemorySessionRepo().create({ id: "recommendation-session" });
    const worker = new RecordingWorker();
    const scope = recommendationScope();
    const harness = createRecommendationAgentHarness({
      session,
      models,
      model,
      toolContext: { worker, scope },
    });
    const toolEvents: string[] = [];
    harness.on("tool_call", (event) => {
      toolEvents.push(event.toolName);
      return undefined;
    });

    await harness.prompt("Recommend one Pipeline.");

    expect(faux.state.callCount).toBe(2);
    expect(toolEvents).toEqual(["recommendation_context", "recommendation_freeze"]);
    expect(worker.calls.map((call) => call.tool)).toEqual([
      "recommendation.context",
      "recommendation.freeze",
    ]);
    expect(worker.calls[1]?.payload).toMatchObject({
      ...scope,
      method: "AR",
      action: {
        dimension: "default",
        parameter: null,
        value: "official_default",
      },
    });
    expect(String(worker.calls[1]?.payload.recommendation_id)).toMatch(
      /^recommendation-[0-9a-f-]{36}$/,
    );

    const sessionContext = await session.buildContext();
    const messages = JSON.stringify(sessionContext.messages);
    expect(messages).toContain("historical_pipeline_evidence");
    expect(messages).toContain("current_task_metrics");
    expect(messages).not.toContain("target_metric_value");
    expect(messages).not.toContain('"parameter":null');
    expect(messages).not.toContain("recommendation_job");
    expect(messages).not.toContain("dataset_family_1");
    expect(messages).not.toContain("unit_current");
    expect(messages).not.toContain("TODS");
    expect(messages).not.toContain("/runs/recommendations");

    const visibleTools = observedContexts[0]?.tools ?? [];
    expect(visibleTools.map((tool) => tool.name)).toEqual([
      "recommendation_context",
      "recommendation_freeze",
    ]);
    expect(visibleTools.every((tool) => /^[A-Za-z0-9_-]+$/.test(tool.name))).toBe(true);
    for (const tool of visibleTools) {
      expect(deepSeekStrictSchemaIssues(tool.parameters)).toEqual([]);
    }
    expect(JSON.stringify(visibleTools)).not.toContain("experiment_run");
    expect(visibleTools.map((tool) => tool.name)).not.toContain("research_complete_step");
    expect(visibleTools.map((tool) => tool.name)).not.toContain("research_finish_episode");
    expect(
      JSON.stringify(
        visibleTools.find((tool) => tool.name === "recommendation_freeze")
          ?.parameters,
      ),
    ).not.toContain("recommendation_id");
    expect(observedContexts[0]?.systemPrompt).toContain("one-shot");
    expect(observedContexts[0]?.systemPrompt).toContain(
      "dataset names, curve names, member names, paths, revisions",
    );
    expect(observedContexts[0]?.systemPrompt).toContain("source_unit_id");
    expect(observedContexts[0]?.systemPrompt).not.toContain("source_relations");
    expect(observedContexts[0]?.systemPrompt).toContain("Current task Component metrics");
    expect(observedContexts[0]?.systemPrompt).toContain("after this Agent session terminates");
  });

  it("withholds Python error details from the Provider session", async () => {
    const faux = fauxProvider();
    const sourceModels = createModels();
    sourceModels.setProvider(faux.provider);
    const model = faux.getModel();
    const observedContexts: Context[] = [];
    faux.setResponses([
      (context) => {
        observedContexts.push(context);
        return fauxAssistantMessage(fauxToolCall("recommendation_context", {}, { id: "context-call" }), {
          stopReason: "toolUse",
        });
      },
      (context) => {
        observedContexts.push(context);
        return fauxAssistantMessage("Stop after sanitized tool error.");
      },
    ]);

    const directory = await mkdtemp(join(tmpdir(), "timesigma-recommendation-error-"));
    temporaryDirectories.push(directory);
    const controller = createAuditedProviderController({
      config: loadTimeSigmaProviderConfig({}),
      model,
      source: sourceModels.streamSimple.bind(sourceModels),
      log: new ProviderLog(join(directory, "provider.jsonl")),
      budget: { maxCalls: 2, maxTokens: 100_000 },
    });
    const models = createAuditedModels(sourceModels, controller);
    const session = await new InMemorySessionRepo().create({ id: "recommendation-error-session" });
    const worker: ResearchWorker = {
      async callSuccess() {
        throw new Error(
          "research evidence is missing: /home/user/projects/TimeSigma/runs/Yahoo/A3Benchmark-TS49/run.json",
        );
      },
    };
    const harness = createRecommendationAgentHarness({
      session,
      models,
      model,
      toolContext: { worker, scope: recommendationScope() },
    });

    await harness.prompt("Load context.");

    const sessionContext = await session.buildContext();
    const messages = JSON.stringify(sessionContext.messages);
    expect(messages).toContain("Internal diagnostics were withheld");
    expect(messages).not.toContain("/home/user/projects/TimeSigma");
    expect(messages).not.toContain("Yahoo");
    expect(messages).not.toContain("A3Benchmark-TS49");
    expect(observedContexts).toHaveLength(2);
  });
});

function deepSeekStrictSchemaIssues(value: unknown, path = "$"): string[] {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => deepSeekStrictSchemaIssues(item, `${path}[${index}]`));
  }
  if (value === null || typeof value !== "object") return [];

  const schema = value as Record<string, unknown>;
  const issues: string[] = [];
  const unsupported = new Set([
    "$defs",
    "$id",
    "const",
    "maxLength",
    "minItems",
    "maxItems",
    "minLength",
    "patternProperties",
  ]);
  for (const [key, child] of Object.entries(schema)) {
    if (unsupported.has(key)) issues.push(`${path}.${key}`);
    issues.push(...deepSeekStrictSchemaIssues(child, `${path}.${key}`));
  }
  if (schema.type === "null") issues.push(`${path}.type:null`);
  if (Array.isArray(schema.anyOf)) {
    for (const branch of schema.anyOf as Array<Record<string, unknown>>) {
      if (branch.type === undefined) issues.push(path + ".anyOf.branch.type");
    }
  }
  if (schema.type === "object") {
    const properties = Object.keys((schema.properties ?? {}) as Record<string, unknown>).sort();
    const required = [...((schema.required ?? []) as string[])].sort();
    if (schema.additionalProperties !== false) issues.push(`${path}.additionalProperties`);
    if (JSON.stringify(properties) !== JSON.stringify(required)) issues.push(`${path}.required`);
  }
  return issues;
}

class RecordingWorker implements ResearchWorker {
  readonly calls: Array<{
    tool: ToolName;
    payload: Record<string, JsonValue>;
    options: { timeoutMs?: number; idempotencyKey?: string } | undefined;
  }> = [];

  async callSuccess(
    tool: ToolName,
    payload: Record<string, JsonValue>,
    options?: { timeoutMs?: number; idempotencyKey?: string },
  ): Promise<ToolResultEnvelope> {
    this.calls.push({ tool, payload, options });
    return {
      protocol_version: "timesigma.bridge.v1",
      schema_version: 1,
      request_id: `request-${this.calls.length}`,
      status: "success",
      result:
        tool === "research.context"
          ? {
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
                  {
                    rank: 1,
                    evidence_id: "component/LOF/naive/TODS/0",
                    method: "LOF",
                    pipeline: {
                      method: "LOF",
                      data_params: { preprocess: "z-score", diff_order: 0 },
                      model_params: { n_neighbors: 20 },
                    },
                    target_metric_value: 0.8,
                  },
                ],
                best_value: 0.8,
                best_evidence_ids: ["component/LOF/naive/TODS/0"],
                default_initial_baseline_evidence_id: "component/LOF/naive/TODS/0",
              },
              research_state: {
                initial_baseline: {
                  evidence_id: "component/LOF/naive/TODS/0",
                  pipeline: {
                    method: "LOF",
                    data_params: { preprocess: "z-score", diff_order: 0 },
                    model_params: { n_neighbors: 20 },
                  },
                  target_value: 0.8,
                },
                global_best: {
                  value: 0.8,
                  evidence_ids: ["component/LOF/naive/TODS/0"],
                  incumbent_evidence_id: "component/LOF/naive/TODS/0",
                },
                research_step_count: 0,
                research_step_evidence_ids: [],
                research_steps: [],
                lineage_edges: [],
                runtime_budget: {
                  experiment_index: 1,
                  experiments_used: 0,
                  max_experiments: 2,
                  experiments_remaining: 2,
                  max_wall_seconds: 60,
                },
                stop_status: {
                  can_continue: true,
                  stop_source: null,
                  stop_reason: null,
                },
              },
              episode_feedback: [],
              relevant_history: [],
              executable_capabilities: [
                {
                  method: "LOF",
                  schema_supported: true,
                  fit_scope: "per_curve",
                  default_pipeline: {
                    method: "LOF",
                    data_params: { preprocess: "z-score", diff_order: 0 },
                    model_params: { n_neighbors: 20 },
                  },
                  allowed_action_dimensions: [
                    "algorithm",
                    "preprocess",
                    "diff_order",
                    "hyperparameter",
                  ],
                  parameters: {
                    preprocess: {
                      legal_values: ["raw", "min-max", "z-score"],
                      default_value: "z-score",
                    },
                    diff_order: { minimum: 0, default_value: 0 },
                    window: null,
                    hyperparameters: [
                      {
                        name: "n_neighbors",
                        primitive_type: "integer",
                        default_value: 20,
                      },
                    ],
                  },
                },
              ],
              research_capabilities: {
                detectors: ["LOF"],
                preprocessing: [
                  {
                    operation: "add_component",
                    stage: "preprocessing",
                    component_id: "robust_scaler",
                    parameters: {},
                  },
                ],
                mvp_constraints: [
                  "only preprocessing dimension changes are executable in this MVP",
                  "pipeline_action must keep the detector unchanged",
                ],
              },
            }
          : tool === "recommendation.context"
            ? {
                task_contract: {
                  schema: "naive",
                  target_metric: "point-wise-f1",
                  input_state: "raw_train",
                  planning_scope: "curve",
                  training_member_count: 1,
                  held_out_member_count: 0,
	                  one_shot_constraints: {
	                    current_task_metrics: "withheld",
	                    current_task_pipeline_evidence: "withheld",
	                    exact_self_evidence: "forbidden",
	                    current_task_capability_results: "not_required",
	                    evaluation_feedback: "withheld_until_after_session",
	                    pipeline_contract: "choose one Catalog-supported method",
	                  },
                },
                profile: {
                  input_state: "raw_train",
                  curve_profile: {
                    lag1_autocorrelation: 0.9,
                    roughness: 0.1,
                  },
                  aggregate: null,
                  zero_shot_split: null,
                },
                catalog: [
                  {
                    method: "AR",
                    fit_scope: "per_curve",
                    default_pipeline: {
                      method: "AR",
                      resolved_configuration: {
                        Data_Params: { preprocess: "min-max", diff_order: 1 },
                        Model_Params: { p: 16 },
                      },
                    },
                    allowed_actions: {
                      default: "official_default",
                      preprocess: ["raw", "min-max", "z-score"],
                      diff_order: { minimum: 0 },
                      window: { parameter: "p" },
                      hyperparameters: ["batch_size"],
                    },
                    parameter_semantics: {},
                  },
                ],
	                historical_pipeline_evidence: [],
	              }
          : tool === "experiment.run"
            ? "pipeline_action" in payload
              ? {
                  status: "pipeline_executed",
                  experiment_id: payload.experiment_id ?? "robust-lof",
                  pipeline_action: payload.pipeline_action ?? null,
                  changed_research_dimension: "preprocessing",
                  delta: 0.04,
                }
              : {
                  status: "executed_pending_diagnosis",
                  experiment_id: "lof-neighbors-25",
                  action: payload.action ?? null,
                  delta: 0.04,
                }
            : tool === "recommendation.freeze"
              ? {
                  recorded: true,
                  recommendation_id: payload.recommendation_id ?? null,
                  recommendation_job: "/runs/recommendations/rec-ar-default",
                  action: payload.action ?? null,
                  pipeline: {
                    method: payload.method ?? "AR",
                    resolved_configuration: {
                      Data_Params: { preprocess: "min-max", diff_order: 1 },
                      Model_Params: { p: 16 },
                    },
                  },
                  rationale: payload.rationale ?? "recommendation rationale",
                }
              : { recorded: true, next_action: payload.next_action ?? null },
      error: null,
    };
  }
}

function researchScope(): ResearchScope {
  return {
    dataset: "TODS",
    schema: "naive",
    target_metric: "point-wise-f1",
    planning_unit: {
      curve: "0",
      training_curves: ["0"],
      held_out_curves: [],
    },
    dataset_root: "/data",
    dataset_source_root: "/sources",
    dataset_dir: "TODS",
    dataset_source: "CSTCloudOps/datasets",
    dataset_revision: "0c6dfa195a5af8ef75938bc1ca6b870e49c7b14a",
    profile_dir: "/derived/profiles/TODS",
    experiment_root: "/runs/pipelines",
    zero_shot_membership: null,
    experiment_index: 1,
    max_experiments: 2,
    max_wall_seconds: 60,
  };
}

function recommendationScope(): ResearchScope {
  return {
    ...researchScope(),
    experiment_root: "/runs/recommendations",
    max_experiments: 1,
  };
}
