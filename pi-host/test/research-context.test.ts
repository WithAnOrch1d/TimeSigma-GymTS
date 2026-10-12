import { AgentHarness, InMemorySessionRepo } from "@earendil-works/pi-agent-core";
import { createModels, fauxProvider, fauxAssistantMessage, fauxToolCall, type Context } from "@earendil-works/pi-ai";
import { expect, it } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createAuditedModels } from "../src/audited-models.js";
import { createAuditedProviderController } from "../src/provider-runtime.js";
import { loadTimeSigmaProviderConfig } from "../src/provider-config.js";
import { ProviderLog } from "../src/provider-log.js";
import { installResearchCompaction } from "../src/research-context.js";

it("uses native Pi compaction while preserving exact science snapshots and raw session entries", async () => {
  const session = await new InMemorySessionRepo().create({ id: "compression" });
  const oldText = "old verbose reasoning ".repeat(7000);
  await session.appendMessage({ role: "user", content: oldText, timestamp: 1 });
  await session.appendMessage(fauxAssistantMessage("Historical interpretation only."));
  await session.appendMessage({ role: "user", content: "Inspect current state", timestamp: 2 });
  await session.appendMessage(fauxAssistantMessage(fauxToolCall("research_context", {}, { id: "state" }), { stopReason: "toolUse" }));
  const exact = { baseline: "research-step/TODS/naive/0/old", preview_id: "frozen-preview", schema: "naive", window: 128, pending: "needs-diagnosis", notes: "recent measured observations ".repeat(200) };
  await session.appendMessage({ role: "toolResult", toolName: "research_context", toolCallId: "state", isError: false, timestamp: 3, content: [{ type: "text", text: JSON.stringify(exact) }] });
  const faux = fauxProvider(); const models = createModels(); models.setProvider(faux.provider);
  faux.setResponses([fauxAssistantMessage("Compact historical reasoning; retain unresolved diagnosis."), fauxAssistantMessage("Done")]);
  const calls: Context[] = [];
  const original = models.streamSimple.bind(models);
  models.streamSimple = (model, context, opts) => { calls.push(context); return original(model, context, opts); };
  const directory = await mkdtemp(join(tmpdir(), "pi-compaction-test-"));
  const controller = createAuditedProviderController({ config: loadTimeSigmaProviderConfig({}), model: faux.getModel(),
    source: models.streamSimple.bind(models), log: new ProviderLog(join(directory, "provider.jsonl")),
    budget: { maxCalls: 8, maxTokens: 1000000 } });
  const harness = new AgentHarness({ session, models: createAuditedModels(models, controller), model: faux.getModel(), tools: [] });
  installResearchCompaction(harness, session, { thresholdTokens: 3000, keepRecentTokens: 1000 });
  await harness.prompt("Continue without refitting.");
  const entries = await session.getBranch();
  expect(entries.some(entry => entry.type === "compaction")).toBe(true);
  expect(JSON.stringify(await session.getEntries()).includes(oldText)).toBe(true);
  const context = await session.buildContext();
  expect(JSON.stringify(context.messages)).toContain("frozen-preview");
  expect(JSON.stringify(context.messages)).toContain("needs-diagnosis");
  expect(JSON.stringify(context.messages).includes(oldText)).toBe(false);
  expect(calls.length).toBeGreaterThanOrEqual(2); // Native summary AND following research request.
  expect(controller.stats().providerAttempts).toBe(calls.length);
  expect(controller.stats().tokens).toBeGreaterThan(0);
  await rm(directory, { recursive: true, force: true });
});
