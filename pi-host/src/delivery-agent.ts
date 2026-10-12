import { AgentHarness, type AgentHarnessTool, type Session } from "@earendil-works/pi-agent-core";
import { Type, type Api, type Model, type Models } from "@earendil-works/pi-ai";
import type { JsonValue, ToolName } from "./protocol.js";
import { capabilityLookupSchema, completePipelineActionSchema, memoryReadSchema, memorySearchSchema, type ResearchWorker } from "./research-agent.js";
import { withStrictToolArguments } from "./tool-arguments.js";
import { DECISION_FACTS_PROMPT, PLANNING_REVIEW_PROMPT, MEMORY_COMPARE_DESCRIPTION, factClaimsSchema, memoryCompareSchema } from "./evidence-tools.js";

export const DELIVERY_SYSTEM_PROMPT = `You are TimeSigma planning one initial complete anomaly-detection Pipeline for the Optimization / Leaderboard objective. Maximize the frozen primary metric using allowed train-side Profile and frozen non-target Evidence within budget. A simple detector is valid; do not exercise capabilities for their own sake. The Harness does not recommend a winner.
${DECISION_FACTS_PROMPT}
${PLANNING_REVIEW_PROMPT}
Before the first legal delivery_preview, include your factual premises in fact_claims: exact source metric values, same-task metric comparisons and any Catalog eligibility assertions used to include/exclude a Method. Use [] if relying on no such premises. Incorrect claims return invalid before a legal candidate is locked, within the existing invalid-plan budget. Keep checked Evidence IDs in delivery_freeze. This verifies only submitted structured facts, not completeness of your rationale or the correctness of transfer predictions; separate those hypotheses explicitly. Do not invent metrics to satisfy a check or interpret a passing check as confidence in target performance.
Load delivery_context first. There is NO target-result initial baseline and NO controlled ResearchStep transition here. Choose all parts of one initial complete Pipeline using the existing capability. The development one-semantic-change rule continues to apply to research transitions, not construction of this initial plan.
Memory is optional and instrumental: retrieve enough useful evidence, then commit your best current hypothesis. More searches are not inherently better; exhaustive Memory research is not the goal. delivery_context lists eligible historical datasets and exclusions. AIOPS history is intentionally unavailable, not hidden behind different keywords. You may search/read frozen non-target Memory and inspect capability. Search free text uses AND semantics, including mentions in reasoning; zero hits do not prove absence. Shorten keywords or use structured component/action filters or profile_similarity. Decide yourself which positive, zero or negative experiences apply. Read details when exact configurations, branch context or evidence directly supporting a plan are needed; no mandatory read each round. Cite full Evidence IDs in rationale and evidence_ids. Observed configurations/metrics differ from historical diagnosis and current hypotheses. Do not invent labels, TP/FP/FN, rankings, secondary metrics or error locations. Equal F1 does not imply equal errors. A local result is not a global optimum; one failed configuration does not disprove a method category. Historical mixed transitions do not isolate detector effects.
No target labels, test statistics, detector results, metrics, checkpoints or prior attempts are available. Do not request them. All targets are planned in independent sessions before ANY execution. No evaluation tool or research loop exists in this session.
Preprocessing uses scaling {method,fit_scope}, temporal_transform {method,order when difference}, smoothing {method,window when moving_average}. Use exactly one scaler (raw/min_max/z_score/robust), and zero or one temporal transform and smoothing. fit_scope is capability-derived, not a choice: raw=not_applicable; native scalers=per_curve_train; robust=per_curve_train for naive or source_train_members for shared Schemas. Native scalers include zero-shot held-out unlabeled train; do not equate their fit scope with robust. Existing legacy chained scaling is replay-only. Submit your chosen complete Pipeline to delivery_preview. Omitted private parameters use pinned defaults; preview exposes exact resolution and initialization. Before a legal preview only bounded invalid-plan repairs are allowed. The FIRST legal preview is the sole candidate: inspect it and freeze it with delivery_freeze; you cannot compare several legal candidates or silently change a resolved plan. No execution or evaluation feedback is used for repair. Ensemble uses only existing average-rank/equal-mean fusion with matching native suffix; record rationale for both branches. Centered MA and rank fusion are offline operations. No weight search or later fallback to a better branch.
Respect finalization_budget in tool results: it estimates the remaining proposal/preview, freeze and normal-completion turns, including repeated transcript input and output/framing growth. When status is finalize_now, prioritize choosing, previewing and freezing your best current Pipeline; avoid optional Memory exploration. A budget ceiling is not a usage target. Prefer small search pages, read only 1–3 IDs per call, and use structured filters for exact executed components. After successful freeze, provide a brief completion and stop. Do not request extra experiments or manufacture a diagnosis.`;

export interface DeliveryToolContext {
  worker: ResearchWorker;
  deliveryRoot: string;
  curve: string;
  deadline: number;
  maxOutputTokens?: number;
  requestTimeoutMs?: number;
  remainingBudget: () => Record<string, JsonValue>;
}

export function createDeliveryAgent(options: {
  session: Session; models: Models; model: Model<Api>; toolContext: DeliveryToolContext;
}): AgentHarness<DeliveryToolContext> {
  let phase: "proposal" | "freeze" | "completion" = "proposal";
  let requestMarker: JsonValue | undefined;
  let pendingToolBytes = 0;
  let largestToolBytes = 0;
  let largestToolWallMs = 0;
  let reserveEntered = false;
  const finalization = (context: DeliveryToolContext) => {
    const budget = context.remainingBudget();
    if (requestMarker !== budget.remainingCalls) {
      requestMarker = budget.remainingCalls;
      pendingToolBytes = 0;
    }
    const turns = phase === "proposal" ? 3 : phase === "freeze" ? 2 : 1;
    const output = context.maxOutputTokens ?? 4096;
    const nextInput = Number(budget.nextInputEstimate ?? budget.lastInputEstimate ?? 0) + pendingToolBytes + 512;
    // Pending tool text is bounded by UTF-8 bytes. Repeated input grows by the
    // output ceiling and observed tool payload/framing for each completion turn.
    const growth = output + Math.max(largestToolBytes, output) + 512;
    const tokens = turns * (nextInput + output) + growth * turns * (turns - 1) / 2;
    const secondsPerTurn = (Number(budget.lastRequestWallMs ?? context.requestTimeoutMs ?? 120000) + largestToolWallMs) / 1000;
    const remainingWall = Math.max(0, (context.deadline - Date.now()) / 1000);
    const threatened = Number(budget.remainingCalls ?? Infinity) <= turns + 1
      || Number(budget.remainingTokens ?? Infinity) <= tokens + nextInput + growth
      || remainingWall <= (turns + 1) * secondsPerTurn;
    reserveEntered ||= threatened;
    return {
      status: phase === "completion" ? "complete" : reserveEntered ? "finalize_now" : "exploration_available",
      phase, required_provider_turns: turns, estimated_next_input_tokens: nextInput,
      reserved_tokens: tokens, reserved_wall_seconds: turns * secondsPerTurn,
      note: phase === "completion" ? "Plan frozen; give a brief completion without further tools."
        : reserveEntered ? "Prioritize proposing, previewing and freezing your best current Pipeline. Avoid optional Memory exploration. No detector or Pipeline is recommended by this budget safeguard."
        : "Memory is optional; freeze when evidence is sufficient. Estimates are conservative, not extra budget or a guarantee of Provider latency.",
    };
  };
  const definitions = [
    ["delivery_context", "delivery.context", "Allowed train Profile, frozen objective, capability and budgets; no target results or baseline.", Type.Object({}, { additionalProperties: false })],
    ["memory_search", "memory.search", "Search frozen non-target Evidence, compact paginated summaries; text uses substring/AND mentions and metadata, NOT exact detector selection. Use detector/component/dataset/schema/action_dimension/delta_sign filters for structured intent. kind=component|research_step. Page limit 1–8. protocol_excluded cannot be solved by reformulating AIOPS history.", memorySearchSchema],
    ["memory_read", "memory.read", "Read 1–3 selected full Evidence IDs per call (split larger lists); limit/offset page lineage, not the number of Evidence IDs; historical interpretations are not facts or causal proof.", memoryReadSchema],
    ["memory_compare", "memory.compare", MEMORY_COMPARE_DESCRIPTION, memoryCompareSchema],
    ["capability_lookup", "capability.lookup", "Inspect executable capability, defaults and compatibility, not performance recommendation.", capabilityLookupSchema],
    ["delivery_preview", "delivery.preview", "Check submitted structured facts, then resolve the one chosen Pipeline. Invalid facts do not lock a candidate. The first legal preview locks both Pipeline and claims; no target-performance judgement.", Type.Object({ pipeline: completePipelineActionSchema.properties.value, fact_claims: factClaimsSchema }, { additionalProperties: false })],
    ["delivery_freeze", "delivery.freeze", "Freeze the matching resolved preview as this target's sole final plan and stop.", Type.Object({ preview_id: Type.String(), rationale: Type.String({ pattern: "\\S" }), evidence_ids: Type.Array(Type.String()) }, { additionalProperties: false })],
  ] as const;
  const tools: AgentHarnessTool<DeliveryToolContext>[] = definitions.map(([name, tool, description, parameters]) => ({
    name, label: name, description, parameters,
    constrainedSampling: { type: "json_schema", strict: "require" },
    async execute(_id, params, signal, _update, context) {
      const ids = (params as Record<string, JsonValue>).evidence_ids;
      if (tool === "memory.read" && (!Array.isArray(ids) || ids.length < 1 || ids.length > 3)) {
        throw new Error("memory_read accepts 1–3 Evidence IDs per call; split larger lists into separate calls");
      }
      const remaining = context.deadline - Date.now();
      if (remaining <= 0) throw new Error("frozen planning wall-time exhausted");
      const before = finalization(context);
      const started = Date.now();
      const optionalMemory = tool === "memory.search" || tool === "memory.read" || tool === "memory.compare";
      const response = optionalMemory && before.status === "finalize_now"
        ? { result: { status: "finalization_required", reason: before.note } }
        : await context.worker.callSuccess(tool as ToolName, {
          delivery_root: context.deliveryRoot, curve: context.curve, ...(params as Record<string, JsonValue>),
        }, { timeoutMs: Math.min(remaining, 60_000) }, signal);
      largestToolWallMs = Math.max(largestToolWallMs, Date.now() - started);
      if (tool === "delivery.preview" && response.result?.status === "ready") phase = "freeze";
      if (tool === "delivery.freeze" && response.result?.status === "frozen") phase = "completion";
      const result = { ...response.result, remaining_provider_budget: context.remainingBudget(), remaining_wall_seconds: Math.max(0, (context.deadline - Date.now()) / 1000), finalization_budget: before };
      const bytes = Buffer.byteLength(JSON.stringify(result), "utf8");
      pendingToolBytes += bytes + 64;
      largestToolBytes = Math.max(largestToolBytes, bytes);
      result.finalization_budget = finalization(context);
      return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
    },
  }));
  return new AgentHarness({ ...options, systemPrompt: DELIVERY_SYSTEM_PROMPT, tools: tools.map(tool => withStrictToolArguments(tool, options.session)) });
}
