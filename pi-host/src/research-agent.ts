import {
  AgentHarness,
  type AgentHarnessTool,
  type Session,
} from "@earendil-works/pi-agent-core";
import { randomUUID } from "node:crypto";
import { Type, type Api, type Model, type Models } from "@earendil-works/pi-ai";
import type {
  JsonValue,
  ToolCallOptions,
  ToolName,
  ToolResultEnvelope,
} from "./protocol.js";
import { actionForProvider, canonicalizeProviderAction } from "./provider-action.js";
import { withStrictToolArguments } from "./tool-arguments.js";
import { DECISION_FACTS_PROMPT, MEMORY_COMPARE_DESCRIPTION, memoryCompareSchema } from "./evidence-tools.js";

export interface ResearchScope {
  dataset: string;
  schema: string;
  target_metric: string;
  planning_unit: {
    curve: string | null;
    training_curves: string[];
    held_out_curves: string[];
  };
  dataset_root: string;
  dataset_source_root: string;
  dataset_dir: string;
  dataset_source: string;
  dataset_revision: string;
  profile_dir: string;
  agent_memory_root?: string;
  memory_mode?: "on" | "off";
  experiment_root: string;
  zero_shot_membership: string | null;
  experiment_index: number;
  max_experiments: number;
  max_wall_seconds: number;
}

export interface ResearchWorker {
  callSuccess(
    tool: ToolName,
    payload: Record<string, JsonValue>,
    options?: ToolCallOptions,
    signal?: AbortSignal,
  ): Promise<ToolResultEnvelope>;
}

export interface ResearchToolContext {
  worker: ResearchWorker;
  scope: ResearchScope;
  deadlineUnixMs?: number;
  remainingProviderBudget?: () => Record<string, JsonValue>;
}

export interface ResearchAgentOptions {
  session: Session;
  models: Models;
  model: Model<Api>;
  toolContext: ResearchToolContext;
}

const hyperparameterValueSchema = Type.Union([
  Type.Boolean(),
  Type.Number(),
  Type.String(),
  Type.Array(Type.Union([Type.Boolean(), Type.Number(), Type.String()])),
]);
const nonBlankStringSchema = Type.String({ pattern: "\\S" });
const experimentIdSchema = Type.String({
  pattern: "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
});

const algorithmActionSchema = Type.Object(
  {
    dimension: Type.String({ enum: ["algorithm"] }),
    value: nonBlankStringSchema,
  },
  { additionalProperties: false },
);
const preprocessActionSchema = Type.Object(
  {
    dimension: Type.String({ enum: ["preprocess"] }),
    value: Type.String({ enum: ["raw", "min-max", "z-score"] }),
  },
  { additionalProperties: false },
);
const diffOrderActionSchema = Type.Object(
  {
    dimension: Type.String({ enum: ["diff_order"] }),
    value: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);
const windowActionSchema = Type.Object(
  {
    dimension: Type.String({ enum: ["window"] }),
    value: Type.Integer({ minimum: 1 }),
  },
  { additionalProperties: false },
);
const hyperparameterActionSchema = Type.Object(
  {
    dimension: Type.String({ enum: ["hyperparameter"] }),
    parameter: nonBlankStringSchema,
    value: hyperparameterValueSchema,
  },
  { additionalProperties: false },
);
const robustScalerPipelineActionSchema = Type.Object(
  {
    operation: Type.String({ enum: ["add_component"] }),
    stage: Type.String({ enum: ["preprocessing"] }),
    component_id: Type.String({ enum: ["robust_scaler"] }),
  },
  { additionalProperties: false },
);
const movingAveragePipelineActionSchema = Type.Object(
  {
    operation: Type.String({ enum: ["add_component"] }),
    stage: Type.String({ enum: ["preprocessing"] }),
    component_id: Type.String({ enum: ["moving_average"] }),
    parameters: Type.Object(
      {
        window: Type.Integer({ minimum: 2 }),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);
const pipelineActionSchema = Type.Union([
  robustScalerPipelineActionSchema,
  movingAveragePipelineActionSchema,
]);
const plannedDetectorSchema = Type.Object(
  {
    method: nonBlankStringSchema,
    parameters: Type.Array(Type.Object(
      { name: nonBlankStringSchema, value: hyperparameterValueSchema },
      { additionalProperties: false },
    )),
  },
  { additionalProperties: false },
);
export const preprocessingSpecSchema = Type.Object({
  scaling: Type.Object({
    method: Type.String({ enum: ["raw", "min_max", "z_score", "robust"] }),
    fit_scope: Type.String({ enum: ["not_applicable", "per_curve_train", "source_train_members"], description: "Derived from capability: raw=not_applicable; native scalers=per_curve_train; robust=per_curve_train in naive, source_train_members in shared Schemas. Not a tunable scope." }),
  }, { additionalProperties: false }),
  temporal_transform: Type.Union([
    Type.Object({ method: Type.String({ enum: ["none"] }) }, { additionalProperties: false }),
    Type.Object({ method: Type.String({ enum: ["difference"] }), order: Type.Integer({ minimum: 1 }) }, { additionalProperties: false }),
  ]),
  smoothing: Type.Union([
    Type.Object({ method: Type.String({ enum: ["none"] }) }, { additionalProperties: false }),
    Type.Object({ method: Type.String({ enum: ["moving_average"] }), window: Type.Integer({ minimum: 2 }) }, { additionalProperties: false }),
  ]),
}, { additionalProperties: false });
export const completePipelineActionSchema = Type.Object(
  {
    dimension: Type.String({ enum: ["pipeline"] }),
    value: Type.Object(
      {
        preprocessing: preprocessingSpecSchema,
        detector: plannedDetectorSchema,
        ensemble: Type.Object(
          {
            method: Type.String({ enum: ["none", "mean"] }),
            detectors: Type.Array(plannedDetectorSchema),
          },
          { additionalProperties: false },
        ),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);
const executableActionSchema = Type.Union([
  algorithmActionSchema,
  preprocessActionSchema,
  diffOrderActionSchema,
  windowActionSchema,
  hyperparameterActionSchema,
  completePipelineActionSchema,
]);
const defaultRecommendationActionSchema = Type.Object(
  {
    dimension: Type.String({ enum: ["default"] }),
    value: Type.String({ enum: ["official_default"] }),
  },
  { additionalProperties: false },
);
const recommendationActionSchema = Type.Union([
  defaultRecommendationActionSchema,
  preprocessActionSchema,
  diffOrderActionSchema,
  windowActionSchema,
  hyperparameterActionSchema,
]);
const stopActionSchema = Type.Object(
  {
    dimension: Type.String({ enum: ["stop"] }),
    value: Type.String({
      enum: ["target_saturated", "no_reasonable_next_step"],
    }),
  },
  { additionalProperties: false },
);
const finishActionSchema = Type.Union([
  algorithmActionSchema,
  preprocessActionSchema,
  diffOrderActionSchema,
  windowActionSchema,
  hyperparameterActionSchema,
  stopActionSchema,
  completePipelineActionSchema,
]);

const contextSchema = Type.Object({}, { additionalProperties: false });
export const memorySearchSchema = Type.Object(
  {
    query: Type.String({ description: "AND of space-separated substrings in text/metadata/interpretations, NOT an exact detector selector. For executed AR use filters [{field: 'detector', value: 'AR'}], not query='AR'. Zero matches: shorten terms or use structured filters. Empty means no text filter." }),
    filters: Type.Array(Type.Object(
      {
        field: Type.String({ enum: [
          "kind", "task_scope", "dataset", "schema", "planning_unit", "detector",
          "preprocessing", "diff_order", "window", "parameter", "parameter_value",
          "component", "ensemble", "action_dimension", "delta_sign", "diagnosis", "metric",
        ] }),
        value: { ...hyperparameterValueSchema, description: "String except integer diff_order/window and typed parameter_value (requires parameter). kind: component|research_step; task_scope: current|other|all; delta_sign: positive|zero|negative; ensemble: none|mean. Each field at most once." },
      },
      { additionalProperties: false },
    )),
    order: Type.String({ enum: ["evidence_id", "profile_similarity"] }),
    offset: Type.Integer({ minimum: 0 }),
    limit: Type.Integer({ minimum: 1, maximum: 8 }),
  },
  { additionalProperties: false },
);
export const memoryReadSchema = Type.Object(
  {
    evidence_ids: Type.Array(nonBlankStringSchema, { description: "Read 1–3 full Evidence IDs. Split larger lists into separate calls. Cardinality is enforced before worker dispatch; this Provider's strict schema does not support minItems/maxItems." }),
    include_lineage: Type.Boolean(),
    offset: Type.Integer({ minimum: 0 }),
    limit: Type.Integer({ minimum: 1, maximum: 8 }),
  },
  { additionalProperties: false },
);
export const capabilityLookupSchema = Type.Object(
  {
    kind: Type.String({ enum: ["all", "detector", "preprocessing", "pipeline", "ensemble"] }),
    name: Type.String({ description: "Exact capability name, or empty to list the requested kind." }),
  },
  { additionalProperties: false },
);
const experimentSchema = Type.Object(
  {
    experiment_id: experimentIdSchema,
    preview_id: nonBlankStringSchema,
    hypothesis: nonBlankStringSchema,
    baseline_evidence_id: nonBlankStringSchema,
    action: Type.Union([...executableActionSchema.anyOf, ...pipelineActionSchema.anyOf]),
    stopping_condition: nonBlankStringSchema,
  },
  { additionalProperties: false },
);
const previewSchema = Type.Object(
  {
    baseline_evidence_id: nonBlankStringSchema,
    action: Type.Union([...executableActionSchema.anyOf, ...pipelineActionSchema.anyOf]),
  },
  { additionalProperties: false },
);
const finishSchema = Type.Object(
  {
    experiment_id: experimentIdSchema,
    diagnosis_status: Type.String({
      enum: ["supported", "refuted", "inconclusive"],
    }),
    diagnosis: nonBlankStringSchema,
    next_action: finishActionSchema,
    next_action_rationale: nonBlankStringSchema,
  },
  { additionalProperties: false },
);
const finishEpisodeSchema = Type.Object(
  {
    reason: Type.String({
      enum: ["target_saturated", "no_reasonable_next_step"],
    }),
    rationale: nonBlankStringSchema,
  },
  { additionalProperties: false },
);
const recommendationFreezeSchema = Type.Object(
  {
    method: nonBlankStringSchema,
    action: recommendationActionSchema,
    rationale: nonBlankStringSchema,
  },
  { additionalProperties: false },
);

const RESEARCH_SYSTEM_PROMPT = `You are TimeSigma's non-AIOPS time-series anomaly detection research agent.

${DECISION_FACTS_PROMPT}

Use research_context before proposing an experiment. Ground factual claims in the returned task, train-side profile, component_landscape, research_state, episode_feedback, relevant_history, executable_capabilities, research_capabilities, and facts you actively read through memory_search, memory_read or capability_lookup. Reason from these facts to form hypotheses. Current exact-task Evidence and historical other-task Evidence are distinct factual inputs. Historical other-task Evidence is reasoning context only; do not use it as an executable baseline. Never request, infer, or use test labels. Never use AIOPS results, checkpoints, decisions, or feedback to form a hypothesis, diagnosis, or next action.

You own the research question and decide whether Memory is useful. You may actively call memory_search, memory_read and capability_lookup before choosing a hypothesis or branch. Memory contains actual ComponentEvidence and ResearchStepEvidence; Registry/Catalog describes executable capabilities, not empirical success. The current context intentionally contains only recent steps. Use memory_search with task_scope=current to inspect other current-task branches, or task_scope=other for independent historical tasks. Filters are field/value pairs: kind=component|research_step; delta_sign=positive|zero|negative; ensemble=none|mean. Other filters include detector, component, preprocessing, diff_order, window, parameter and parameter_value, action_dimension, diagnosis, metric, dataset, schema and planning_unit. Delta sign uses the historical step's declared target_metric. Empty query and filters are allowed. Search returns summaries and next_offset; you decide what to read and whether to request another page. Profile similarity is an optional search order, not an algorithm recommendation or a guarantee of transfer.

Use memory_read to inspect exact Pipelines, hypothesis, metrics, diagnosis and parent/sibling/child context. Historical diagnosis is the model's recorded interpretation, not an established law. Consider counterexamples when relevant, but do not search mechanically or fabricate missing evidence. When you adopt or reject an experience, cite its Evidence ID and explain applicability or rejection in the existing hypothesis, diagnosis or next_action_rationale. Explain what you expect the proposed single change to reveal. These query tools do not prescribe a next action. You may form a new hypothesis without relevant history. You may query capability_lookup for exact defaults and constraints instead of guessing parameters. Keep all reads and decisions within the Runner-owned call, token and time budgets.

Keep three kinds of reasoning distinct in your existing hypothesis, diagnosis and rationale text: Observed fact (the Pipeline, metrics, delta and Artifact facts actually recorded); Historical interpretation (a previous model's hypothesis or diagnosis, which may be wrong); Current hypothesis (your testable prediction, including what would contradict it). Cite the full Evidence ID when a particular record supports a research decision. A search summary can provide a weak prior; label it as such instead of promoting it to a factual law. Consider memory_read when you need an exact historical configuration or branch context, directly support NextAction with a record, resolve an ambiguous summary, make a causal explanation, or reject a class of methods. Reading is your decision, not a required ritual on every turn.

Equal F1 does not establish equal TP/FP/FN, identical errors, score rankings or operating points. One failed configuration does not invalidate its algorithm class. One gain does not establish a mechanism, and a local parameter comparison does not establish a global optimum. Never present unavailable labels, test details or inferred anomaly locations as observed facts. Check each secondary metric separately; mixed directions are not uniform corroboration. A supported diagnosis means the stated prediction was supported, which can include a predicted zero or negative delta; it does not mean that performance improved. Keep unverified mechanism explanations explicitly tentative.

Free-text memory_search uses AND semantics: every space-separated literal term must match somewhere in the searchable record. A zero-result query is not evidence that relevant experience does not exist. You may reformulate by shortening keywords, removing text and using structured filters, or requesting profile_similarity order. A text mention in a hypothesis, diagnosis or proposed next action does not prove that a component was executed. For exact executed-component experiments, prefer component/preprocessing and action_dimension filters, then inspect the record when needed. The Runner may restrict visible Memory to the initial baseline and this episode; an empty cross-task search under that scope says nothing about the full historical store.

The context exposes known active Evidence references and compact Pipeline facts. The first genuinely new execution must use research_state.initial_baseline.evidence_id as baseline_evidence_id. Later turns may select any legal active current-task baseline_evidence_id exposed by research_state/component_landscape, then propose exactly one executable action. Do not treat recorded prior diagnosis or next_action as Harness authority; they are historical Model reasoning only.

Current ComponentEvidence records are factual results for their official-default Pipelines. Use experiments to validate a Pipeline that is not already represented by active current-task Evidence. The hypothesis must briefly state why the baseline was chosen and why the Candidate is the new untested Pipeline to validate. An action's new value must differ from the baseline's current value. A single frozen anomaly score must produce all eight official metrics.

Before every experiment_run, call experiment_preview with the chosen baseline_evidence_id and action. Inspect its exact resolved candidate, actual semantic diff and detector-private initialization. It consumes no detector budget. If it is incompatible, choose a different action or baseline and preview again; do not infer hidden defaults or ask execution to repair it. A ready preview returns preview_id. Repeat that baseline and action with preview_id in experiment_run; write the hypothesis for the candidate actually shown. Do not execute a candidate that does not test your intended controlled configuration.

One experiment means one ACTUAL semantic research change. Algorithm changes preserve all shared Data_Params, preprocessing components, context window length and sampling stride. Target detector-private architecture/training parameters use the target Method's declared defaults, even when parameter names overlap; these initializations belong to the detector replacement and are not separate findings. Window means number of input/subsequence observations: AR/AE p, AnomalyTransformer win_size, MatrixProfile window; LOF/OCSVM are fixed at one observation. MatrixProfile needs at least four. Equal context length is not equal native score support: horizon and scoring semantics remain method-specific. Only stride=1 is supported by the current continuous-suffix evaluator. Incompatible shared context is rejected, never silently reset. Native scaling and optional robust_scaler are separate executed stages; changing both is two changes. Adding/removing/changing two preprocessing components is also two changes. An ensemble addition/removal freezes existing detectors and preprocessing; existing ensemble branches must be researched standalone before a new composition.

The initial baseline is the best known already evaluated legal default Component for this current task, not the only detector that exists. Its selection is the frozen development baseline policy, not a Memory recommendation. A Component default comparison is not automatically a controlled detector experiment. When reading historical ResearchSteps, check semantic_diff/single_semantic_change: old bundled transitions remain actual Pipeline comparisons and cannot support isolated factor claims. If an intended controlled configuration was not actually tested, explicitly say so and narrow the diagnosis; do not conclusively support/refute the untested detector-level hypothesis.

Read memory_visibility first. When mode=off, use only current-task facts from the selected view: do not issue cross-task searches or reads. When mode=on, use historical Evidence only within visibility. Check secondary metric comparisons against both baseline_official_metrics and official_metrics (or an exact memory_read) before citing them. Never infer TP/FP/FN, error counts, or locations from F1, even as a plausible fractional example. Different event-wise and point-wise F1 do not identify missed events. Equal F1 does not imply equal other metrics. A local plateau bounds the tested settings only, not global saturation or a whole method category.

Remaining budgets cover experiments, detector branches, Provider calls/tokens and wall time. Preview includes the execution budget; tool replies include Provider/time limits. Total token remainder must pay for the next input as well as output; the displayed input estimate is conservative and the next context may grow. Avoid repeatedly looking up information already in the current context. Leave capacity to diagnose an executed result and finish; do not consume the entire budget in optional searches. Budget limits are Runner-owned and cannot be increased.

The Provider tool experiment_run always has an object at the top level. Its required preview_id binds the reviewed candidate. Its action field accepts either a NextAction or a PipelineAction such as {operation:"add_component", stage:"preprocessing", component_id:"robust_scaler"}. The adapter forwards PipelineAction through the existing domain pipeline_action field. The domain experiment.run API still supports both original payload forms.

Pre-register exactly one changed research dimension for each experiment. Existing NextAction and add_component interfaces remain available. To propose a complete Pipeline use action {dimension:"pipeline", value:{preprocessing:{scaling:{method:<raw|min_max|z_score|robust>,fit_scope:<capability-declared scope>},temporal_transform:{method:"none"}|{method:"difference",order:<positive integer>},smoothing:{method:"none"}|{method:"moving_average",window:<integer greater than one>}}, detector:{method:<Catalog name>, parameters:[{name:<parameter>, value:<value>}]}, ensemble:{method:<none|mean>, detectors:[<zero or one additional detector>]}}}. Empty detector parameters use the native Method defaults. No preprocessing means scaling={method:"raw",fit_scope:"not_applicable"}, temporal_transform={method:"none"}, smoothing={method:"none"}. Each category chooses at most one operation; categories can compose. Native scalers fit each curve train, including zero-shot held-out train; robust fits only training members in shared Schemas. Existing explicit operator orders remain unchanged and chained-scaling layouts remain replayable, but do not introduce new scaling chains through add_component. Legacy research dimensions are unchanged; switching native scaling to robust can require two controlled transitions. The default complete-plan order is native split, differencing, native scaling, optional robust scaling, optional moving average, detector, then optional mean score fusion. Explicit legacy operator order is preserved by preview and replay; category grouping is not permission to reorder it. Mean fusion shares preprocessing, requires equal native suffix support, and uses frozen offline per-curve average-rank normalization. The complete proposed Pipeline must differ from the baseline in exactly one actual dimension; do not hide simultaneous parameter or component changes inside a full proposal. The Runner already froze the target metric, Planning Unit, maximum experiment count, wall-time and Provider budget; never increase them. Each ensemble has at most two detector executions. Keep the baseline, hypothesis, configuration, seed and stopping condition frozen. Call experiment_run sequentially. The same final score produces all eight EasyTSAD metrics.

After evaluation, diagnose the target-metric result as supported, refuted, or inconclusive. If the target metric is saturated or no reasonable next step remains, record next_action {dimension:"stop", value:"target_saturated"} or {dimension:"stop", value:"no_reasonable_next_step"}. Otherwise choose exactly one executable next action, which may contain a complete Pipeline. Record next_action_rationale as the brief research reason for the next action that is not recoverable from numeric Evidence alone. Call research_complete_step for every executed experiment, including preprocessing and ensemble, to commit its ResearchStepEvidence. Never finish with an undiagnosed execution. A later turn may select any legal active current-task Evidence as baseline; next_action is historical reasoning, not Harness routing authority. To modify a detector already inside an ensemble, use its standalone research branch. An ensemble addition may use an unchanged Component baseline. Pi Session is only conversational working state, not scientific Memory.`;
const RECOMMENDATION_SYSTEM_PROMPT = `You are TimeSigma's non-AIOPS one-shot time-series anomaly detection recommendation agent.

Use recommendation_context before making a recommendation. Base the recommendation only on the returned Task Contract, train-side Profile, Catalog, and Historical Pipeline Evidence from other tasks. Provider-facing dataset names, curve names, member names, paths, revisions, source metadata, and public benchmark names are omitted. Identical historical source_unit_id values mean those Historical Cases came from the same historical planning unit or research trajectory; do not infer real benchmark names. Current task Component metrics, Pipeline metrics, exact-self Evidence, test labels, and evaluation feedback are intentionally withheld. Never request, infer, or use them.

Freeze exactly one Pipeline recommendation with recommendation_freeze. Choose one Catalog-supported method and either action {dimension:"default", value:"official_default"} or exactly one preprocess/diff_order/window/hyperparameter override to that method's official default. The system generates the recommendation_id; do not provide or ask for one. Do not call experiment_run, research_complete_step, or research_finish_episode. Do not ask for a second trial, HPO, Ensemble, scoring/post-processing, or code changes.

The Runner evaluates the frozen recommendation once after this Agent session terminates. That evaluation result is not returned to the Agent and must not influence the recommendation rationale.`;

export function createResearchAgentHarness(
  options: ResearchAgentOptions,
): AgentHarness<ResearchToolContext> {
  return new AgentHarness<ResearchToolContext>({
    session: options.session,
    models: options.models,
    model: options.model,
    systemPrompt: RESEARCH_SYSTEM_PROMPT,
    toolContext: options.toolContext,
    tools: researchTools().map(tool => withStrictToolArguments(tool, options.session)),
  });
}

export function createRecommendationAgentHarness(
  options: ResearchAgentOptions,
): AgentHarness<ResearchToolContext> {
  return new AgentHarness<ResearchToolContext>({
    session: options.session,
    models: options.models,
    model: options.model,
    systemPrompt: RECOMMENDATION_SYSTEM_PROMPT,
    toolContext: options.toolContext,
    tools: recommendationTools().map(tool => withStrictToolArguments(tool, options.session)),
  });
}

function researchTools(): AgentHarnessTool<ResearchToolContext>[] {
  return [
    {
      name: "experiment_preview",
      label: "Preview controlled experiment",
      description: "Resolve a candidate before training: exact baseline, requested action, candidate, actual semantic diff, private-default initialization and compatibility. Consumes no detector budget. A ready result issues preview_id; execution must repeat that baseline/action and preview_id. An incompatible result explains the shared value or extra change; the LLM chooses the next action.",
      parameters: previewSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      async execute(_toolCallId, params, signal, _onUpdate, context) {
        const { action, ...request } = params;
        const actionPayload = "operation" in action ? { pipeline_action: action } : { action: canonicalizeProviderAction(action) };
        return await callDomainTool(context, "experiment.preview", { ...context.scope, ...request, ...actionPayload } as Record<string, JsonValue>, signal);
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof previewSchema>,
    {
      name: "memory_search",
      label: "Search research Evidence",
      description: "Search visible Evidence with compact pages. Text uses AND literal matching, including historical reasoning; mentions do not prove execution. For executed components use component/action_dimension filters. Zero hits may require fewer keywords, structured filters or profile_similarity order, and do not establish absence. task_scope=current|other|all; delta_sign=positive|zero|negative; kind=component|research_step. No winner is recommended.",
      parameters: memorySearchSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      async execute(_toolCallId, params, signal, _onUpdate, context) {
        return await callDomainTool(context, "memory.search", { ...context.scope, ...params }, signal);
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof memorySearchSchema>,
    {
      name: "memory_read",
      label: "Read selected Evidence",
      description: "Read up to three Evidence IDs with exact Pipelines, official metrics, Artifact availability and optional parent/sibling/child context. Consider reading for exact configurations, ambiguous summaries or records supporting NextAction and causal claims. Hypothesis/diagnosis are historical model interpretations, not observed mechanisms. Cite full IDs. Visibility is checked for IDs and relatives.",
      parameters: memoryReadSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      async execute(_toolCallId, params, signal, _onUpdate, context) {
        return await callDomainTool(context, "memory.read", { ...context.scope, ...params }, signal);
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof memoryReadSchema>,
    {
      name: "memory_compare",
      label: "Compare observed Evidence",
      description: MEMORY_COMPARE_DESCRIPTION,
      parameters: memoryCompareSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      async execute(_toolCallId, params, signal, _onUpdate, context) {
        return await callDomainTool(context, "memory.compare", { ...context.scope, ...params }, signal);
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof memoryCompareSchema>,
    {
      name: "capability_lookup",
      label: "Inspect executable capability",
      description: "Inspect Catalog detectors, preprocessing, Pipeline operations, parameters, Schema support and Ensemble compatibility. Empty name lists the requested kind. Capability information is not experimental Evidence.",
      parameters: capabilityLookupSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      async execute(_toolCallId, params, signal, _onUpdate, context) {
        return await callDomainTool(context, "capability.lookup", { ...context.scope, ...params }, signal);
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof capabilityLookupSchema>,
    {
      name: "research_context",
      label: "Load research context",
      description: "Load the fixed non-AIOPS task context, Profile, active Evidence views, and executable capability facts.",
      parameters: contextSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      async execute(_toolCallId, _params, signal, _onUpdate, context) {
        return await callDomainTool(context, "research.context", { ...context.scope }, signal);
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof contextSchema>,
    {
      name: "experiment_run",
      label: "Run frozen experiment",
      description: "Execute the exact baseline/action accepted by experiment_preview using its preview_id. The semantic diff is checked again; changed or stale previews do not consume detector budget. Returns both baseline and candidate official metrics for fact checking.",
      parameters: experimentSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      executionMode: "sequential",
      async execute(_toolCallId, params, signal, _onUpdate, context) {
        const { action, ...request } = params;
        const actionPayload =
          "operation" in action
            ? { pipeline_action: action }
            : { action: canonicalizeProviderAction(action) };
        return await callDomainTool(
          context,
          "experiment.run",
          {
            ...context.scope,
            ...request,
            ...actionPayload,
          } as Record<string, JsonValue>,
          signal,
          { timeoutMs: context.scope.max_wall_seconds * 1_000 },
        );
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof experimentSchema>,
    {
      name: "research_complete_step",
      label: "Complete research step",
      description: "Commit the completed experiment diagnosis and its single next action as ResearchStepEvidence.",
      parameters: finishSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      async execute(_toolCallId, params, signal, _onUpdate, context) {
        const domainResult = await callDomainTool(
          context,
          "research.complete_step",
          {
            ...context.scope,
            ...params,
            next_action: canonicalizeProviderAction(params.next_action),
          } as Record<string, JsonValue>,
          signal,
        );
        return { ...domainResult, terminate: true };
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof finishSchema>,
    {
      name: "research_finish_episode",
      label: "Finish research episode",
      description: "End the research episode without committing a ResearchStepEvidence.",
      parameters: finishEpisodeSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      async execute(_toolCallId, params, signal, _onUpdate, context) {
        const domainResult = await callDomainTool(
          context,
          "research.finish_episode",
          {
            ...context.scope,
            ...params,
          } as Record<string, JsonValue>,
          signal,
        );
        return { ...domainResult, terminate: true };
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof finishEpisodeSchema>,
  ];
}

function recommendationTools(): AgentHarnessTool<ResearchToolContext>[] {
  return [
    {
      name: "recommendation_context",
      label: "Load recommendation context",
      description: "Load the one-shot task Profile, Catalog, and other-task Historical Memory.",
      parameters: contextSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      async execute(_toolCallId, _params, signal, _onUpdate, context) {
        return await callDomainTool(context, "recommendation.context", { ...context.scope }, signal);
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof contextSchema>,
    {
      name: "recommendation_freeze",
      label: "Freeze recommendation",
      description: "Persist one frozen Pipeline recommendation without running evaluation.",
      parameters: recommendationFreezeSchema,
      constrainedSampling: { type: "json_schema", strict: "require" },
      async execute(_toolCallId, params, signal, _onUpdate, context) {
        const domainResult = await callDomainTool(
          context,
          "recommendation.freeze",
          {
            ...context.scope,
            ...params,
            recommendation_id: systemRecommendationId(),
            action: canonicalizeProviderAction(params.action),
          } as Record<string, JsonValue>,
          signal,
        );
        return { ...domainResult, terminate: true };
      },
    } satisfies AgentHarnessTool<ResearchToolContext, typeof recommendationFreezeSchema>,
  ];
}

async function callDomainTool(
  context: ResearchToolContext,
  tool: ToolName,
  payload: Record<string, JsonValue>,
  signal: AbortSignal | undefined,
  callOptions?: ToolCallOptions,
) {
  try {
    if (tool === "experiment.preview" || tool === "experiment.run") {
      payload = { ...payload, enforce_preprocessing_categories: true };
    }
    if (tool === "memory.read" && (!Array.isArray(payload.evidence_ids) || payload.evidence_ids.length < 1 || payload.evidence_ids.length > 3)) {
      throw new Error("memory_read accepts 1–3 Evidence IDs per call; split larger lists into separate calls");
    }
    const remainingMs = context.deadlineUnixMs === undefined
      ? undefined : context.deadlineUnixMs - Date.now();
    if (remainingMs !== undefined && remainingMs <= 0) {
      throw new Error("Frozen research wall-time budget exhausted");
    }
    const envelope = await context.worker.callSuccess(
      tool,
      payload,
      remainingMs === undefined ? callOptions : {
        ...callOptions,
        timeoutMs: Math.min(callOptions?.timeoutMs ?? 30_000, remainingMs),
      },
      signal,
    );
    const result = providerToolResult(
      tool,
      envelope.result as Record<string, JsonValue>,
    );
    if (context.remainingProviderBudget !== undefined || remainingMs !== undefined) {
      result.remaining_budget = {
        ...(context.remainingProviderBudget?.() ?? {}),
        ...(context.deadlineUnixMs === undefined ? {} : {
          wall_seconds_remaining: Math.max(0, Math.floor((context.deadlineUnixMs - Date.now()) / 1000)),
        }),
      };
    }
    return {
      content: [{ type: "text" as const, text: JSON.stringify(result) }],
      details: result,
    };
  } catch (error) {
    throw providerFacingToolError(tool, error);
  }
}

function providerToolResult(
  tool: ToolName,
  result: Record<string, JsonValue>,
): Record<string, JsonValue> {
  if (tool === "research.context") {
    return { ...result };
  }
  if (tool === "recommendation.context") {
    return { ...result };
  }
  if (tool === "experiment.run") {
    const visible = { ...result };
    if (result.action !== undefined) {
      visible.action = actionForProvider(
        result.action as Record<string, JsonValue>,
      );
    }
    return visible;
  }
  if (tool === "recommendation.freeze") {
    const pipeline = result.pipeline as Record<string, JsonValue>;
    return {
      method: requiredProviderField(pipeline.method, "pipeline.method"),
      action: actionForProvider(
        result.action as Record<string, JsonValue>,
      ),
      rationale: requiredProviderField(result.rationale, "rationale"),
    };
  }
  if (tool === "research.complete_step") {
    return {
      ...result,
      next_action: actionForProvider(
        result.next_action as Record<string, JsonValue>,
      ),
    };
  }
  return {
    ...result,
  };
}

function systemRecommendationId(): string {
  return `recommendation-${randomUUID()}`;
}

function requiredProviderField(
  value: JsonValue | undefined,
  field: string,
): JsonValue {
  if (value === undefined) throw new Error(`missing Provider result field: ${field}`);
  return value;
}

function providerFacingToolError(tool: ToolName, cause: unknown): Error {
  const error = new Error(
    `${tool} failed. Internal diagnostics were withheld from the Provider context.`,
  ) as Error & { cause?: unknown };
  Object.defineProperty(error, "cause", {
    value: cause,
    enumerable: false,
    configurable: true,
  });
  return error;
}
