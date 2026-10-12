import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type AssistantMessageEvent,
  type Api,
  type Model,
} from "@earendil-works/pi-ai";

type ProviderEventStream = ReturnType<typeof createAssistantMessageEventStream>;

export function replayProviderEvents(
  target: ProviderEventStream,
  events: readonly AssistantMessageEvent[] | undefined,
  message: AssistantMessage,
): void {
  if (events === undefined || events.length === 0) {
    pushProviderMessage(target, message);
    return;
  }
  for (const event of events) target.push(event);
}

export function pushProviderMessage(stream: ProviderEventStream, message: AssistantMessage): void {
  stream.push({ type: "start", partial: message });
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    stream.push({ type: "error", reason: message.stopReason, error: message });
  } else {
    stream.push({ type: "done", reason: message.stopReason, message });
  }
}

export function providerErrorMessage(model: Model<Api>, error: string): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: emptyUsage(),
    stopReason: "error",
    errorMessage: error,
    timestamp: Date.now(),
  };
}

export function providerAssistantLog(message: AssistantMessage): Record<string, unknown> {
  return {
    provider: message.provider,
    model: message.model,
    response_model: message.responseModel ?? null,
    response_id: message.responseId ?? null,
    stop_reason: message.stopReason,
    error_message: message.errorMessage ?? null,
    content: message.content.map((block) =>
      block.type === "thinking"
        ? { type: "thinking", length: block.thinking.length }
        : block,
    ),
  };
}

export function providerResponseHeaders(headers: Record<string, string>): Record<string, string> {
  const allowed = new Set([
    "date",
    "x-request-id",
    "openai-request-id",
    "request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
  ]);
  return Object.fromEntries(
    Object.entries(headers).filter(([key]) => allowed.has(key.toLowerCase())),
  );
}

function emptyUsage(): AssistantMessage["usage"] {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}
