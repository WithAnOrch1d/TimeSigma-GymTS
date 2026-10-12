import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { Type, type Context, type ToolCall } from "@earendil-works/pi-ai";
import { createTimeSigmaProviderRuntime, providerProvenance } from "./provider-client.js";
import { loadTimeSigmaProviderConfig } from "./provider-config.js";
import { ProviderLog } from "./provider-log.js";
import { createAuditedProviderController } from "./provider-runtime.js";

export async function runProviderPreflight(
  env: NodeJS.ProcessEnv = process.env,
): Promise<Record<string, unknown>> {
  const config = loadTimeSigmaProviderConfig(env);
  const runtime = await createTimeSigmaProviderRuntime(config, env);
  const logPath = resolve(env.TIMESIGMA_LLM_LOG_PATH ?? ".artifacts/provider-events.jsonl");
  const log = new ProviderLog(logPath);
  await log.append("provider_preflight_started", {
    provenance: providerProvenance(runtime),
    purpose: "authentication, mapping, Pi strict-tool, usage, and request-trace qualification",
  });
  const controller = createAuditedProviderController({
    config,
    model: runtime.model,
    source: runtime.models.streamSimple.bind(runtime.models),
    log,
    budget: { maxCalls: 1, maxTokens: Math.min(8_000, config.maxOutputTokens + 4_000) },
  });
  const nonce = randomUUID();
  const context: Context = {
    systemPrompt:
      "This is a TimeSigma provider qualification. Call the only tool exactly once with the supplied nonce. Do not answer in prose.",
    messages: [
      {
        role: "user",
        content: `Call timesigma_provider_preflight with nonce '${nonce}'.`,
        timestamp: Date.now(),
      },
    ],
    tools: [
      {
        name: "timesigma_provider_preflight",
        description: "Return the qualification nonce without changing it.",
        parameters: Type.Object({ nonce: Type.String() }, { additionalProperties: false }),
        constrainedSampling: { type: "json_schema", strict: "require" },
      },
    ],
  };
  const stream = await controller.streamFn(runtime.model, context, { maxTokens: 512 });
  const result = await stream.result();
  const toolCall = result.content.find(
    (block): block is ToolCall => block.type === "toolCall",
  );
  const checks = {
    stop_reason_tool_use: result.stopReason === "toolUse",
    strict_tool_name: toolCall?.name === "timesigma_provider_preflight",
    nonce_round_trip: toolCall?.arguments.nonce === nonce,
    usage_reported: result.usage.totalTokens > 0,
    response_trace_reported: Boolean(result.responseId),
    requested_model_mapping:
      result.provider === runtime.model.provider && result.model === runtime.model.id,
    not_paused: !controller.stats().paused,
  };
  const passed = Object.values(checks).every(Boolean);
  await log.append("provider_preflight_finished", {
    passed,
    checks,
    exposed_model: result.model,
    response_model: result.responseModel ?? null,
    response_model_observability:
      result.responseModel === undefined
        ? `not_exposed_by_pi_${config.api.replaceAll("-", "_")}_0.82.1`
        : "provider_reported",
    response_id: result.responseId ?? null,
    usage: result.usage,
    log_path: logPath,
  });
  if (!passed) {
    throw new Error(`TimeSigma provider preflight failed: ${JSON.stringify(checks)}`);
  }
  return {
    passed,
    provider: runtime.model.provider,
    exposed_model: runtime.model.id,
    response_model: result.responseModel,
    response_model_observability:
      result.responseModel === undefined
        ? `not_exposed_by_pi_${config.api.replaceAll("-", "_")}_0.82.1`
        : "provider_reported",
    response_id: result.responseId,
    usage: result.usage,
    log_path: logPath,
    checks,
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  runProviderPreflight()
    .then((summary) => {
      process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
    })
    .catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      const hint = message.includes("Missing TIMESIGMA_")
        ? " Configure the active provider key in .env; never paste it into logs or chat."
        : "";
      process.stderr.write(`TimeSigma provider preflight failed: ${message}.${hint}\n`);
      process.exitCode = 1;
    });
}
