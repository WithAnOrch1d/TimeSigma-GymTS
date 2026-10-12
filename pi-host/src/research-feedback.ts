import { join } from "node:path";
import type { Api, Model, Models } from "@earendil-works/pi-ai";
import type { Session } from "@earendil-works/pi-agent-core";
import { installResearchCompaction } from "./research-context.js";
import type { JsonValue, ToolCallOptions, ToolName, ToolResultEnvelope } from "./protocol.js";
import {
  createResearchAgentHarness,
  type ResearchScope,
  type ResearchWorker,
} from "./research-agent.js";

export interface ResearchFeedbackOptions {
  models: Models;
  model: Model<Api>;
  worker: ResearchWorker;
  scope: ResearchScope;
  createSession: (turnNumber: number) => Promise<Session>;
  firstPrompt: string;
  remainingProviderBudget?: () => Record<string, JsonValue>;
  providerBudgetStopped?: () => boolean;
  /** A separately frozen recovery wall allowance; native experiment limits stay in scope. */
  episodeWallSeconds?: number;
}

export interface ResearchTurn {
  experiment_id: string;
  evidence_id: string;
  baseline_evidence_id: string;
  action: Record<string, JsonValue>;
  diagnosis_status: "supported" | "refuted" | "inconclusive";
  next_action: Record<string, JsonValue>;
  pipeline_job: string;
  target_metric: string;
  baseline_value: number;
  candidate_value: number;
  delta: number;
}

type TurnOutcome =
  | {
      kind: "completed";
      experiments: ResearchTurn[];
      experimentsUsed: number | null;
      finished: boolean;
    }
  | { kind: "continue"; reason: "reused"; experimentsUsed: number | null; evidenceId: string | null }
  | { kind: "finish" }
  | { kind: "budget_stop"; reason: "wall_time_budget" | "provider_budget"; experiments: ResearchTurn[]; unfinished: string[] };

export interface ResearchFeedbackResult {
  experiments: ResearchTurn[];
  stop_reason?: "agent_finish" | "experiment_budget" | "wall_time_budget" | "provider_budget";
  unfinished_experiments?: string[];
}

export async function runResearchFeedback(
  options: ResearchFeedbackOptions,
): Promise<ResearchFeedbackResult> {
  if (options.scope.dataset.toUpperCase() === "AIOPS") {
    throw new Error("AIOPS cannot enter iterative research");
  }

  const experiments: ResearchTurn[] = [];
  let experimentsUsed = Math.max(0, options.scope.experiment_index - 1);
  let prompt = options.firstPrompt;
  let turnNumber = 1;
  const wallSeconds = options.episodeWallSeconds ?? options.scope.max_wall_seconds;
  if (!Number.isFinite(wallSeconds) || wallSeconds <= 0) throw new Error("invalid episode wall budget");
  const deadlineUnixMs = Date.now() + wallSeconds * 1_000;

  while (experimentsUsed < options.scope.max_experiments) {
    const scope: ResearchScope = {
      ...options.scope,
      experiment_index: Math.min(experimentsUsed + 1, options.scope.max_experiments),
    };
    if (Date.now() >= deadlineUnixMs) return { experiments, stop_reason: "wall_time_budget", unfinished_experiments: [] };
    const outcome = await runTurn(options, scope, turnNumber, prompt, deadlineUnixMs);
    if (outcome.kind === "budget_stop") return { experiments: [...experiments, ...outcome.experiments], stop_reason: outcome.reason, unfinished_experiments: outcome.unfinished };
    if (outcome.kind === "finish") return { experiments, stop_reason: "agent_finish" };
    if (outcome.experimentsUsed !== null) {
      experimentsUsed = outcome.experimentsUsed;
    }
    if (outcome.kind === "completed") {
      experiments.push(...outcome.experiments);
      experimentsUsed = Math.max(experimentsUsed, experiments.length);
      if (outcome.finished) return { experiments, stop_reason: "agent_finish" };
      const latestExperiment = last(outcome.experiments);
      if (latestExperiment === undefined) {
        throw new Error("completed turn did not include a ResearchStep");
      }
      prompt = nextResearchPrompt(latestExperiment);
    } else {
      prompt = reuseReconsiderationPrompt(outcome.evidenceId);
    }
    turnNumber += 1;
  }

  return { experiments, stop_reason: "experiment_budget" };
}

async function runTurn(
  options: ResearchFeedbackOptions,
  scope: ResearchScope,
  turnNumber: number,
  prompt: string,
  deadlineUnixMs: number,
): Promise<TurnOutcome> {
  const observed = new ObservedWorker(options.worker);
  const session = await options.createSession(turnNumber);
  const compressionAbort = new AbortController();
  const harness = createResearchAgentHarness({
    session,
    models: options.models,
    model: options.model,
    toolContext: {
      worker: observed, scope, deadlineUnixMs,
      ...(options.remainingProviderBudget === undefined ? {} : {
        remainingProviderBudget: options.remainingProviderBudget,
      }),
    },
  });
  installResearchCompaction(harness, session, { signal: compressionAbort.signal });
  let timedOut = false;
  let abortError: unknown;
  let aborting: Promise<void> | undefined;
  let turnError: unknown;
  const timer = setTimeout(() => {
    timedOut = true;
    compressionAbort.abort();
    aborting = harness.abort().then(() => undefined, error => { abortError = error; });
  }, Math.max(1, deadlineUnixMs - Date.now()));
  try {
    await harness.prompt(prompt);
  } catch (error) {
    turnError = error;
  } finally {
    clearTimeout(timer);
    await aborting;
  }
  if (abortError !== undefined) throw abortError;
  const budgetStop = timedOut ? "wall_time_budget" : options.providerBudgetStopped?.() ? "provider_budget" : null;
  if (budgetStop !== null) {
    const commits = observed.calls.filter(call => call.tool === "research.complete_step");
    const committedIds = new Set(commits.map(call => executionId(call)));
    const pending = observed.calls.filter(call => call.tool === "experiment.run" && call.result.status === "executed_pending_diagnosis" && !committedIds.has(executionId(call)));
    return { kind: "budget_stop", reason: budgetStop, experiments: commits.map(call => completedTurn(scope, call, matchingRun(observed.calls, call))), unfinished: [...new Set([...pending.map(call => executionId(call)), ...observed.interruptedExperiments])] };
  }
  if (turnError !== undefined) throw turnError;

  const episodeFinish = last(
    observed.calls.filter((call) => call.tool === "research.finish_episode"),
  );
  const finishes = observed.calls.filter((call) => call.tool === "research.complete_step");
  if (finishes.length > 0) {
    const completedExperimentIds = new Set(
      finishes.map((finish) =>
        executionId(finish),
      ),
    );
    const pendingExecution = last(
      observed.calls.filter(
        (call) =>
          call.tool === "experiment.run" &&
          call.result.status === "executed_pending_diagnosis" &&
          !completedExperimentIds.has(
            executionId(call),
          ),
      ),
    );
    if (pendingExecution !== undefined) {
      throw new Error(
        `Research turn ${turnNumber} produced execution output without research_complete_step diagnosis`,
      );
    }
    return {
      kind: "completed",
      experiments: finishes.map((finish) =>
        completedTurn(scope, finish, matchingRun(observed.calls, finish)),
      ),
      experimentsUsed: experimentsUsedFrom(finishes[finishes.length - 1]!.result),
      finished: episodeFinish !== undefined,
    };
  }

  if (episodeFinish !== undefined) return { kind: "finish" };

  const executed = last(
    observed.calls.filter(
      (call) => call.tool === "experiment.run" && call.result.status === "executed_pending_diagnosis",
    ),
  );
  if (executed !== undefined) {
    throw new Error(
      `Research turn ${turnNumber} produced execution output without research_complete_step diagnosis`,
    );
  }

  const reused = last(
    observed.calls.filter(
      (call) => call.tool === "experiment.run" && call.result.status === "reused",
    ),
  );
  if (reused !== undefined) {
    return {
      kind: "continue",
      reason: "reused",
      experimentsUsed: experimentsUsedFrom(reused.result),
      evidenceId: optionalString(reused.result.reused_evidence_id),
    };
  }

  throw new Error(
    `Research turn ${turnNumber} ended without research_finish_episode, research_complete_step, or duplicate reuse`,
  );
}

interface ObservedCall {
  tool: ToolName;
  payload: Record<string, JsonValue>;
  result: Record<string, JsonValue>;
}

class ObservedWorker implements ResearchWorker {
  readonly calls: ObservedCall[] = [];
  readonly interruptedExperiments = new Set<string>();

  constructor(private readonly worker: ResearchWorker) {}

  async callSuccess(
    tool: ToolName,
    payload: Record<string, JsonValue>,
    options?: ToolCallOptions,
    signal?: AbortSignal,
  ): Promise<ToolResultEnvelope> {
    if (tool === "experiment.run") this.interruptedExperiments.add(requiredString(payload.experiment_id, "experiment_id"));
    const envelope = await this.worker.callSuccess(tool, payload, options, signal);
    if (envelope.status !== "success" || envelope.result === null) {
      throw new Error(`${tool} returned no successful result`);
    }
    this.calls.push({ tool, payload, result: envelope.result });
    if (tool === "experiment.run") this.interruptedExperiments.delete(requiredString(payload.experiment_id, "experiment_id"));
    return envelope;
  }
}

function executionId(call: ObservedCall): string {
  // A request may recover an existing pending Pipeline under its persisted ID.
  return requiredString(call.result.experiment_id ?? call.payload.experiment_id, "experiment_id");
}

function completedTurn(
  scope: ResearchScope,
  finish: ObservedCall,
  run: ObservedCall | undefined,
): ResearchTurn {
  const experimentId = executionId(finish);
  if (run !== undefined && executionId(run) !== experimentId) {
    throw new Error("Research turn finish identity is mismatched");
  }
  return {
    experiment_id: experimentId,
    evidence_id: requiredString(finish.result.evidence_id, "evidence_id"),
    baseline_evidence_id: requiredString(
      finish.result.baseline_evidence_id ?? run?.result.baseline_evidence_id,
      "baseline_evidence_id",
    ),
    action: requiredObject(run?.result.action ?? finish.result.action, "action"),
    diagnosis_status: requiredDiagnosis(
      finish.payload.diagnosis_status,
      "diagnosis_status",
    ),
    next_action: requiredObject(finish.result.next_action, "next_action"),
    pipeline_job: join(scope.experiment_root, experimentId),
    target_metric: requiredString(finish.result.target_metric, "target_metric"),
    baseline_value: requiredNumber(finish.result.baseline_value, "baseline_value"),
    candidate_value: requiredNumber(finish.result.candidate_value, "candidate_value"),
    delta: requiredNumber(finish.result.delta, "delta"),
  };
}

function matchingRun(
  calls: readonly ObservedCall[],
  finish: ObservedCall,
): ObservedCall | undefined {
  const experimentId = executionId(finish);
  return last(
    calls.filter(
      (call) => call.tool === "experiment.run" && executionId(call) === experimentId,
    ),
  );
}

function experimentsUsedFrom(result: Record<string, JsonValue>): number | null {
  const budget = result.execution_budget;
  if (budget === null || typeof budget !== "object" || Array.isArray(budget)) return null;
  const used = budget.experiments_used;
  return typeof used === "number" && Number.isFinite(used) ? used : null;
}

function nextResearchPrompt(experiment: ResearchTurn): string {
  return [
    `ResearchStepEvidence ${JSON.stringify(experiment.evidence_id)} is committed.`,
    "Load research_context again.",
    "Then either propose one legal new ResearchProposal from any active current-task Evidence, or finish the episode if no further experiment is warranted.",
    "Do not treat the recorded next_action as mandatory Harness routing.",
  ].join(" ");
}

function reuseReconsiderationPrompt(evidenceId: string | null): string {
  const reused = evidenceId === null
    ? "The previous proposal reused existing active Evidence."
    : `The previous proposal reused existing active Evidence ${JSON.stringify(evidenceId)}.`;
  return [
    reused,
    "Load research_context again.",
    "Then either branch from any legal active current-task Evidence with a genuinely new Candidate Pipeline, or finish the episode.",
  ].join(" ");
}

function optionalString(value: JsonValue | undefined): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function last<T>(values: readonly T[]): T | undefined {
  return values.length === 0 ? undefined : values[values.length - 1];
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

function requiredDiagnosis(
  value: JsonValue | undefined,
  name: string,
): "supported" | "refuted" | "inconclusive" {
  if (value !== "supported" && value !== "refuted" && value !== "inconclusive") {
    throw new Error(`${name} is invalid`);
  }
  return value;
}

function requiredNumber(value: JsonValue | undefined, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}
