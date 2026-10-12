import type { Api, Model, Models } from "@earendil-works/pi-ai";
import type { Session } from "@earendil-works/pi-agent-core";
import type { JsonValue, ToolCallOptions, ToolName, ToolResultEnvelope } from "./protocol.js";
import {
  createRecommendationAgentHarness,
  type ResearchScope,
  type ResearchWorker,
} from "./research-agent.js";

export interface RecommendationRunOptions {
  models: Models;
  model: Model<Api>;
  worker: ResearchWorker;
  scope: ResearchScope;
  createSession: () => Promise<Session>;
  prompt: string;
}

export interface FrozenRecommendation {
  recommendation_id: string;
  recommendation_job: string;
  action: Record<string, JsonValue>;
  pipeline: Record<string, JsonValue>;
  rationale: string;
}

export async function runOneShotRecommendation(
  options: RecommendationRunOptions,
): Promise<{ recommendation: FrozenRecommendation }> {
  if (options.scope.dataset.toUpperCase() === "AIOPS") {
    throw new Error("AIOPS cannot enter one-shot recommendation prototype");
  }
  if (options.scope.experiment_index !== 1 || options.scope.max_experiments !== 1) {
    throw new Error("one-shot recommendation requires one frozen evaluation");
  }
  const observed = new ObservedRecommendationWorker(options.worker);
  const harness = createRecommendationAgentHarness({
    session: await options.createSession(),
    models: options.models,
    model: options.model,
    toolContext: { worker: observed, scope: options.scope },
  });
  await harness.prompt(options.prompt);
  observed.assertComplete();
  const freeze = observed.calls[1];
  if (freeze === undefined) {
    throw new Error("one-shot recommendation did not freeze a Pipeline");
  }
  return {
    recommendation: {
      recommendation_id: requiredString(
        freeze.result.recommendation_id,
        "recommendation_id",
      ),
      recommendation_job: requiredString(
        freeze.result.recommendation_job,
        "recommendation_job",
      ),
      action: requiredObject(freeze.result.action, "action"),
      pipeline: requiredObject(freeze.result.pipeline, "pipeline"),
      rationale: requiredString(freeze.result.rationale, "rationale"),
    },
  };
}

interface ObservedCall {
  tool: ToolName;
  payload: Record<string, JsonValue>;
  result: Record<string, JsonValue>;
}

const REQUIRED_TOOL_SEQUENCE: readonly ToolName[] = [
  "recommendation.context",
  "recommendation.freeze",
];

class ObservedRecommendationWorker implements ResearchWorker {
  readonly calls: ObservedCall[] = [];
  private position = 0;
  private violation: Error | undefined;

  constructor(private readonly worker: ResearchWorker) {}

  assertComplete(): void {
    if (this.violation !== undefined) throw this.violation;
    if (
      this.position !== REQUIRED_TOOL_SEQUENCE.length ||
      this.calls.length !== REQUIRED_TOOL_SEQUENCE.length
    ) {
      throw new Error("one-shot recommendation did not complete the required tool sequence");
    }
  }

  async callSuccess(
    tool: ToolName,
    payload: Record<string, JsonValue>,
    options?: ToolCallOptions,
    signal?: AbortSignal,
  ): Promise<ToolResultEnvelope> {
    if (this.violation !== undefined) throw this.violation;
    const expected = REQUIRED_TOOL_SEQUENCE[this.position];
    if (tool !== expected) {
      this.violation = new Error(
        `one-shot recommendation expected ${expected ?? "no further tool"}, not ${tool}`,
      );
      throw this.violation;
    }
    const envelope = await this.worker.callSuccess(tool, payload, options, signal);
    if (envelope.status !== "success" || envelope.result === null) {
      throw new Error(`${tool} returned no successful result`);
    }
    this.calls.push({ tool, payload, result: envelope.result });
    this.position += 1;
    return envelope;
  }
}

function requiredString(value: JsonValue | undefined, name: string): string {
  if (typeof value !== "string" || value.length === 0) throw new Error(`${name} is required`);
  return value;
}

function requiredObject(value: JsonValue | undefined, name: string): Record<string, JsonValue> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
  return value;
}
