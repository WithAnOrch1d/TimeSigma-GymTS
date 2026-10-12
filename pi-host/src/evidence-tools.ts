import { Type } from "@earendil-works/pi-ai";

// Strict Provider schemas omit unsupported array cardinality keywords; Python
// enforces the same bounds at the tool boundary before reading any Evidence.
const text = Type.String({ pattern: "\\S" });
export const selectedEvidenceCompareSchema = Type.Object({
  evidence_ids: Type.Array(text, { description: "2–12 distinct full Evidence IDs selected by you. No automatic retrieval or target ranking." }),
  metric: text,
}, { additionalProperties: false });

export const memoryCompareSchema = Type.Object({
  evidence_ids: Type.Array(text, { description: "If candidate_methods is nonempty: 1–4 visible Evidence anchors, exactly one per frozen source task. Otherwise 2–12 distinct Evidence IDs." }),
  candidate_methods: Type.Array(text, { description: "1–12 target-eligible Methods to return a complete source-task × canonical Component table, including missing cells. [] compares only the selected Evidence IDs." }),
  metric: text,
}, { additionalProperties: false });

export const factClaimsSchema = Type.Array(Type.Union([
  Type.Object({ kind: Type.String({ enum: ["metric"] }), evidence_id: text, metric: text, value: Type.Number() }, { additionalProperties: false }),
  Type.Object({ kind: Type.String({ enum: ["comparison"] }), left_id: text, right_id: text, metric: text,
    relation: Type.String({ enum: ["gt", "eq", "lt"] }) }, { additionalProperties: false }),
  Type.Object({ kind: Type.String({ enum: ["capability"] }), method: text, available_in_current_schema: Type.Boolean() }, { additionalProperties: false }),
]), { description: "At most 24 factual premises used by this plan: exact recorded metric values, same-source-task comparisons, or Catalog Schema eligibility (not Artifact selection status). Copy full precision; [] if no such factual premises. Transfer predictions and causal interpretations belong in rationale, not these claims." });

export const SELECTED_EVIDENCE_COMPARE_DESCRIPTION = "Compare 2–12 caller-selected visible Evidence IDs under one official metric. Returns exact Pipelines, same-task score differences, requested coverage/missingness and recorded Research transitions. Missing is unknown; method coverage does not merge configurations. No winner or transfer prediction. Use before claiming a historical ranking; use memory_read for original interpretations and lineage.";

export const MEMORY_COMPARE_DESCRIPTION = "Compare historical observations under one official metric. With candidate_methods=[], compare only 2–12 selected Evidence IDs. With 1–12 candidate_methods, evidence_ids are 1–4 frozen source-task anchors: return every task/candidate cell from the visible Component collection, full Pipeline identities, configuration differences, Artifact semantics and train-Profile distance. Absent cells stay unknown; no Research substitution, source expansion, winner or transfer prediction. Use memory_read for original interpretations and lineage.";

export const DECISION_FACTS_PROMPT = `Use decision_semantics in context to distinguish Method eligibility, Artifact checkpoint selection and loading. not_applicable is not Method ineligibility. Curve IDs are opaque; NAIVE held_out_curves=[] does not mean test-on-train. FCVAE's synthetic training signal describes its recorded results, not a reason to discard them. Different official metrics may disagree without invalidating either. Missing Evidence is unknown, not inferior. memory_compare performs historical arithmetic on selected IDs, not target selection; you decide which configurations and counterexamples apply.`;

// Shared by one-shot delivery and its matched-information acceptance. This is
// a review procedure, not a numeric selector or a new Evidence authority.
export const PLANNING_REVIEW_PROMPT = `Before the first legal preview, review your tentative choice against the strongest alternative supported by the available facts. Compare complete Pipelines under the declared metric on the same source tasks; distinguish configuration differences from detector effects, and score ties from strict advantages. Missing results do not count against a candidate.
Check the already available sources for observations that oppose your tentative choice. Report the strongest relevant counterevidence, or explicitly say none was observed; do not manufacture a counterexample or assume its absence proves transfer. Related sources are not automatically independent confirmations. Profile proximity, agreement and stable source rankings do not establish target success probabilities.
State the assumption connecting the source observations to this target and explain why you still prefer your choice over the alternative. Treat that explanation as a hypothesis: do not infer an anomaly mechanism from a curve name or train-only summary. Do not ban a method because of one failure, or default to a method merely because transfer is uncertain. No averaging, nearest-source or fallback rule is mandatory.
Use an additional allowed tool call only to resolve a specific missing fact that could change the choice; do not repeat a complete table already available or seek target outcomes. Keep this review brief in the existing rationale/support, counterevidence and unknowns fields as applicable; no hidden reasoning trace or new report is required. Complete the review before preview locks the candidate, then preserve the existing freeze and budget rules.`;
