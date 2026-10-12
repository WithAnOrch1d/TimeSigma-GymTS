import { randomUUID } from "node:crypto";
import {
  createAssistantMessageEventStream,
  isRetryableAssistantError,
  retryAssistantCall,
  type AssistantMessageEvent,
  type Api,
  type Context,
  type Model,
} from "@earendil-works/pi-ai";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import type { TimeSigmaProviderConfig } from "./provider-config.js";
import type { ProviderLog } from "./provider-log.js";
import {
  providerAssistantLog,
  providerErrorMessage,
  providerResponseHeaders,
  pushProviderMessage,
  replayProviderEvents,
} from "./provider-messages.js";

export interface LlmBudget {
  maxCalls: number;
  maxTokens: number;
}

export interface AuditedProviderStats {
  logicalCalls: number;
  providerAttempts: number;
  tokens: number;
  cost: number;
  consecutiveFailures: number;
  paused: boolean;
  pauseReason: string | null;
}

export interface AuditedProviderOptions {
  config: TimeSigmaProviderConfig;
  model: Model<Api>;
  source: StreamFn;
  log: ProviderLog;
  budget: LlmBudget;
  estimateInputTokens?: (context: Context) => number;
  previousInput?: { context: Context; tokens: number };
}

export interface AuditedProviderController {
  streamFn: StreamFn;
  stats: () => Readonly<AuditedProviderStats>;
  pause: (reason: string) => Promise<void>;
  budgetState: () => { stopped: boolean; remainingCalls: number; remainingTokens: number; lastInputEstimate: number; nextInputEstimate: number; lastRequestWallMs: number; maxCalls: number; maxTokens: number };
}

interface AttemptCapture {
  events: AssistantMessageEvent[];
}

export function createAuditedProviderController(
  options: AuditedProviderOptions,
): AuditedProviderController {
  const stats: AuditedProviderStats = {
    logicalCalls: 0,
    providerAttempts: 0,
    tokens: 0,
    cost: 0,
    consecutiveFailures: 0,
    paused: false,
    pauseReason: null,
  };
  let measuredInput = options.previousInput === undefined ? undefined : { frames: inputFrames(options.previousInput.context), tokens: options.previousInput.tokens };
  const inputEstimate = options.estimateInputTokens ?? ((context: Context) => {
    const frames = inputFrames(context);
    if (measuredInput && measuredInput.frames.length <= frames.length && measuredInput.frames.every((frame, index) => frame === frames[index])) {
      // Reuse actual billed prompt tokens only for an unchanged prefix. Bound appended
      // content by UTF-8 bytes plus framing, rather than recharging session metadata.
      return measuredInput.tokens + 64 + frames.slice(measuredInput.frames.length).reduce((sum, frame) => sum + Buffer.byteLength(frame, "utf8") + 64, 0);
    }
    return conservativeInputTokenEstimate(context);
  });
  let budgetStopped = false;
  let lastInputEstimate = 0;
  let nextInputEstimate = 0;
  let lastRequestWallMs = 0;

  const pause = async (reason: string): Promise<void> => {
    if (!stats.paused) {
      stats.paused = true;
      stats.pauseReason = reason;
      await options.log.append("batch_paused", { reason, stats });
    }
  };

  const streamFn: StreamFn = (requestModel, context, callOptions = {}) => {
    const outer = createAssistantMessageEventStream();
    void (async () => {
      const logicalCallId = `llm-${randomUUID()}`;
      stats.logicalCalls += 1;
      if (stats.paused) {
        pushProviderMessage(
          outer,
          providerErrorMessage(options.model, `LLM_INFRA_ERROR: ${stats.pauseReason}`),
        );
        return;
      }
      if (
        requestModel.provider !== options.model.provider ||
        requestModel.id !== options.model.id ||
        requestModel.api !== options.model.api
      ) {
        await pause("The Agent attempted an unregistered provider or model switch.");
        pushProviderMessage(
          outer,
          providerErrorMessage(options.model, "LLM_INFRA_ERROR: model switch blocked"),
        );
        return;
      }

      const estimatedInput = inputEstimate(context);
      lastInputEstimate = estimatedInput;
      const remainingTokens = options.budget.maxTokens - stats.tokens;
      const maxTokens = Math.min(
        options.config.maxOutputTokens,
        callOptions.maxTokens ?? options.config.maxOutputTokens,
        Math.max(0, remainingTokens - estimatedInput),
      );
      if (maxTokens < 1 || stats.providerAttempts >= options.budget.maxCalls) {
        budgetStopped = true;
        await pause("The frozen LLM call or token budget was exhausted before dispatch.");
        pushProviderMessage(outer, providerErrorMessage(options.model, "LLM_BUDGET_EXHAUSTED"));
        return;
      }

      const frozenRequest = {
        logical_call_id: logicalCallId,
        provider: options.model.provider,
        model: options.model.id,
        api: options.model.api,
        context,
        options: {
          reasoning: options.config.reasoning,
          max_tokens: maxTokens,
          cache_retention: "none",
          timeout_ms: options.config.timeoutMs,
          provider_internal_retries: 0,
          temperature: callOptions.temperature ?? null,
        },
      };
      await options.log.append("logical_call_started", {
        ...frozenRequest,
        estimated_input_tokens: estimatedInput,
        budget: options.budget,
        stats_before: stats,
      });

      let attemptNumber = 0;
      let finalCapture: AttemptCapture | undefined;
      const message = await retryAssistantCall(
        async () => {
          attemptNumber += 1;
          if (stats.providerAttempts >= options.budget.maxCalls) {
            budgetStopped = true;
            await pause("The frozen LLM call budget was exhausted during retry.");
            return providerErrorMessage(options.model, "LLM_BUDGET_EXHAUSTED during retry");
          }
          const attemptMaxTokens = Math.min(maxTokens, Math.max(0, options.budget.maxTokens - stats.tokens - estimatedInput));
          if (attemptMaxTokens < 1) {
            budgetStopped = true;
            await pause("The frozen LLM token budget was exhausted during retry.");
            return providerErrorMessage(options.model, "LLM_BUDGET_EXHAUSTED during retry");
          }
          stats.providerAttempts += 1;
          const startedAt = Date.now();
          let payload: unknown = null;
          let response: { status: number; headers: Record<string, string> } | null = null;
          const events: AssistantMessageEvent[] = [];
          const sourceStream = await options.source(options.model, context, {
            ...callOptions,
            reasoning: options.config.reasoning,
            maxTokens: attemptMaxTokens,
            cacheRetention: "none",
            timeoutMs: options.config.timeoutMs,
            maxRetries: 0,
            maxRetryDelayMs: options.config.maxRetryDelayMs,
            onPayload: async (value, model) => {
              payload = value;
              return callOptions.onPayload?.(value, model);
            },
            onResponse: async (value, model) => {
              response = { status: value.status, headers: providerResponseHeaders(value.headers) };
              await callOptions.onResponse?.(value, model);
            },
          });
          for await (const event of sourceStream) events.push(event);
          const result = await sourceStream.result();
          stats.tokens += result.usage.totalTokens;
          stats.cost += result.usage.cost.total;
          const promptTokens = result.usage.input + result.usage.cacheRead + result.usage.cacheWrite;
          if (promptTokens > 0 && result.stopReason !== "error" && result.stopReason !== "aborted") {
            measuredInput = { frames: inputFrames(context), tokens: promptTokens };
            nextInputEstimate = inputEstimate({ ...context, messages: [...context.messages, result] });
          }
          lastRequestWallMs = Date.now() - startedAt;
          finalCapture = { events };
          await options.log.append("provider_attempt_finished", {
            logical_call_id: logicalCallId,
            attempt: attemptNumber,
            actual_payload: payload,
            response,
            latency_ms: Date.now() - startedAt,
            assistant: providerAssistantLog(result),
            usage: result.usage,
            cumulative_stats: stats,
          });
          return result;
        },
        {
          enabled: options.config.maxRetries > 0,
          maxRetries: options.config.maxRetries,
          baseDelayMs: options.config.retryBaseDelayMs,
        },
        callOptions.signal,
        {
          onRetryScheduled: async (attempt, maxAttempts, delayMs, error) => {
            await options.log.append("retry_scheduled", {
              logical_call_id: logicalCallId,
              attempt,
              max_attempts: maxAttempts,
              delay_ms: delayMs,
              error,
            });
          },
        },
      );

      if (stats.tokens > options.budget.maxTokens) {
        budgetStopped = true;
        await pause("Provider-reported usage exceeded the frozen LLM token budget.");
        pushProviderMessage(outer, providerErrorMessage(options.model, "LLM_BUDGET_EXCEEDED"));
        return;
      }
      if (budgetStopped) {
        pushProviderMessage(outer, message);
        return;
      }
      if (message.stopReason !== "error") {
        stats.consecutiveFailures = 0;
        replayProviderEvents(outer, finalCapture?.events, message);
        return;
      }
      if (!isRetryableAssistantError(message)) {
        await pause(`Non-retryable provider failure: ${message.errorMessage ?? "unknown"}`);
        replayProviderEvents(outer, finalCapture?.events, message);
        return;
      }

      stats.consecutiveFailures += 1;
      if (stats.consecutiveFailures >= options.config.systemicFailureThreshold) {
        await pause("Transient provider failures reached the systemic-failure threshold.");
      }
      replayProviderEvents(outer, finalCapture?.events, message);
    })().catch(async (error: unknown) => {
      const detail = error instanceof Error ? error.message : String(error);
      await pause(`Audited provider wrapper failure: ${detail}`).catch(() => undefined);
      pushProviderMessage(
        outer,
        providerErrorMessage(options.model, `LLM_INFRA_ERROR: ${detail}`),
      );
    });
    return outer;
  };

  return {
    streamFn, stats: () => ({ ...stats }), pause,
    budgetState: () => ({ stopped: budgetStopped, remainingCalls: Math.max(0, options.budget.maxCalls - stats.providerAttempts), remainingTokens: Math.max(0, options.budget.maxTokens - stats.tokens), lastInputEstimate, nextInputEstimate, lastRequestWallMs, maxCalls: options.budget.maxCalls, maxTokens: options.budget.maxTokens }),
  };
}

function conservativeInputTokenEstimate(context: Context): number {
  // Count model-visible content, not session details/usage or transport JSON escaping.
  let characters = (context.systemPrompt?.length ?? 0) + JSON.stringify(context.tools ?? []).length;
  for (const message of context.messages) {
    characters += 64; // Conservative role/tool framing allowance.
    if (typeof message.content === "string") {
      characters += message.content.length;
      continue;
    }
    for (const block of message.content) {
      if (block.type === "text") characters += block.text.length;
      else if (block.type === "thinking") characters += block.thinking.length;
      else if (block.type === "toolCall") characters += block.name.length + JSON.stringify(block.arguments).length;
      else characters += JSON.stringify(block).length;
    }
  }
  return Math.ceil(characters / 2);
}

function inputFrames(context: Context): string[] {
  return [
    JSON.stringify({ systemPrompt: context.systemPrompt, tools: context.tools?.map(({ name, description, parameters }) => ({ name, description, parameters })) }),
    ...context.messages.map((message) => JSON.stringify({
      role: message.role, content: message.content,
      ...(message.role === "toolResult" ? { toolName: message.toolName, toolCallId: message.toolCallId } : {}),
    })),
  ];
}
