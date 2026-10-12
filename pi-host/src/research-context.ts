import {
  compact, estimateTokens, prepareCompaction,
  type AgentHarness, type AgentMessage, type Session,
} from "@earendil-works/pi-agent-core";

/** Pi owns summarization and the session transform; raw session facts stay append-only. */
export function installResearchCompaction<T extends object | undefined>(
  harness: AgentHarness<T>, session: Session,
  options: { thresholdTokens?: number; keepRecentTokens?: number; signal?: AbortSignal } = {},
): void {
  const threshold = options.thresholdTokens ?? 24000;
  let compacted = false;
  let lastCompactedLeaf: string | null = null;
  const anchors = new Map<string, Extract<AgentMessage, { role: "toolResult" }>>();
  harness.on("context", async event => {
    // The loop's in-memory array still has old messages after a context transform.
    // Pi persisted each message_end before this hook; rebuild from the compacted branch.
    let messages = compacted ? (await session.buildContext()).messages : event.messages;
    for (const message of messages) {
      if (message.role === "toolResult" && ["research_context", "experiment_preview", "experiment_run", "research_complete_step"].includes(message.toolName)) anchors.set(message.toolName, message);
    }
    if (messages.reduce((total, message) => total + estimateTokens(message), 0) < threshold) return { messages };
    const leaf = await session.getLeafId();
    if (leaf === lastCompactedLeaf) return { messages };
    const preparation = prepareCompaction(await session.getBranch(), {
      enabled: true, reserveTokens: 4096, keepRecentTokens: options.keepRecentTokens ?? 8000,
    });
    if (!preparation.ok) throw preparation.error;
    if (!preparation.value) return { messages };
    const result = await compact(preparation.value, harness.models, harness.getModel(),
      "Summarize research reasoning with full Evidence IDs and both positive and negative findings. " +
      "Preserve unresolved work, exact task/schema, baseline, candidate configuration and preview binding, " +
      "actual changed dimension, observed metrics versus historical interpretation, and remaining budgets. " +
      "Do not invent facts, select a new experiment, or treat historical diagnoses as proven science.",
      options.signal, harness.getThinkingLevel());
    if (!result.ok) throw result.error;
    const value = result.value;
    await session.appendCompaction(value.summary, value.firstKeptEntryId, value.tokensBefore,
      value.details, false, value.usage, value.retainedTail);
    // Exact current scientific state supplements the lossy reasoning summary.
    await session.appendMessage({ role: "user", timestamp: Date.now(), content: [{ type: "text", text:
      "Exact latest tool snapshots retained across Pi compaction (historical observations, not new tool calls). " +
      "Use remaining_budget from the next live tool response; these snapshots may contain older counters.\n" +
      JSON.stringify([...anchors].map(([tool, message]) => ({ tool, content: message.content }))),
    }] });
    compacted = true;
    lastCompactedLeaf = await session.getLeafId();
    messages = (await session.buildContext()).messages;
    return { messages };
  });
}
