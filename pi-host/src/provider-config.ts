import type { ThinkingLevel } from "@earendil-works/pi-ai";
import {
  providerDefinition,
  type TimeSigmaProviderApi,
  type TimeSigmaProviderProfile,
} from "./provider-catalog.js";

export interface TimeSigmaProviderConfig {
  profile: TimeSigmaProviderProfile;
  providerId: string;
  providerName: string;
  api: TimeSigmaProviderApi;
  baseUrl: string;
  keyEnvCandidates: readonly string[];
  modelId: string;
  reasoning: ThinkingLevel;
  maxOutputTokens: number;
  timeoutMs: number;
  maxRetries: number;
  retryBaseDelayMs: number;
  maxRetryDelayMs: number;
  systemicFailureThreshold: number;
  cacheRetention: "none";
  requestStore: false;
  gatewayOperator: string;
  authorizationScope: string;
  modelSnapshot: string | null;
  dataFlow: string;
  retentionPolicy: string;
  pricingSource: string;
}

export function loadTimeSigmaProviderConfig(
  env: NodeJS.ProcessEnv = process.env,
): TimeSigmaProviderConfig {
  if (env.NODE_TLS_REJECT_UNAUTHORIZED === "0") {
    throw new Error(
      "TimeSigma refuses NODE_TLS_REJECT_UNAUTHORIZED=0 because it would expose experiment credentials.",
    );
  }
  const profile = providerProfile(env.TIMESIGMA_LLM_PROVIDER ?? "openai");
  const definition = providerDefinition(profile);
  const baseUrl = normalizeHttpsUrl(
    env[definition.baseUrlEnv] ?? definition.defaultBaseUrl,
    definition.baseUrlEnv,
  );
  const modelId = text(
    env.TIMESIGMA_LLM_MODEL_ID ?? definition.defaultModelId,
    "TIMESIGMA_LLM_MODEL_ID",
  );
  const reasoning = reasoningLevel(
    env.TIMESIGMA_LLM_REASONING ?? definition.defaultReasoning,
  );
  return {
    profile,
    providerId: definition.providerId,
    providerName: definition.providerName,
    api: definition.api,
    baseUrl,
    keyEnvCandidates: definition.keyEnvCandidates,
    modelId,
    reasoning,
    maxOutputTokens: integer(env.TIMESIGMA_LLM_MAX_OUTPUT_TOKENS, 4096, 16, 128_000),
    timeoutMs: integer(env.TIMESIGMA_LLM_TIMEOUT_MS, 120_000, 1_000, 600_000),
    maxRetries: integer(env.TIMESIGMA_LLM_MAX_RETRIES, 1, 0, 3),
    retryBaseDelayMs: integer(env.TIMESIGMA_LLM_RETRY_BASE_DELAY_MS, 1_000, 0, 60_000),
    maxRetryDelayMs: integer(env.TIMESIGMA_LLM_MAX_RETRY_DELAY_MS, 10_000, 0, 60_000),
    systemicFailureThreshold: integer(
      env.TIMESIGMA_LLM_SYSTEMIC_FAILURE_THRESHOLD,
      2,
      1,
      10,
    ),
    cacheRetention: "none",
    requestStore: false,
    gatewayOperator: optionalText(env.TIMESIGMA_LLM_GATEWAY_OPERATOR) ?? "unconfirmed",
    authorizationScope:
      optionalText(env.TIMESIGMA_LLM_AUTHORIZATION_SCOPE) ?? "approved TimeSigma experiments",
    modelSnapshot: optionalText(env.TIMESIGMA_LLM_MODEL_SNAPSHOT),
    dataFlow: optionalText(env.TIMESIGMA_LLM_DATA_FLOW) ?? "unconfirmed",
    retentionPolicy: optionalText(env.TIMESIGMA_LLM_RETENTION_POLICY) ?? "unconfirmed",
    pricingSource:
      optionalText(env.TIMESIGMA_LLM_PRICING_SOURCE) ??
      (profile === "deepseek" && modelId === "deepseek-flash"
        ? "DeepSeek official pricing 2026-09-10 peak USD reference; off-peak half; actual billing unconfirmed"
        : `Pi ${definition.catalogProvider} catalog reference; actual billing unconfirmed`),
  };
}

function providerProfile(value: string): TimeSigmaProviderProfile {
  const normalized = value.trim().toLowerCase();
  if (normalized !== "openai" && normalized !== "deepseek") {
    throw new Error("TIMESIGMA_LLM_PROVIDER must be 'openai' or 'deepseek'.");
  }
  return normalized;
}

function normalizeHttpsUrl(value: string, name: string): string {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch (error) {
    throw new Error(`${name} must be a valid absolute URL.`, { cause: error });
  }
  if (parsed.protocol !== "https:") throw new Error("TimeSigma provider base URL must use HTTPS.");
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error("TimeSigma provider base URL cannot contain credentials, query, or fragment.");
  }
  return parsed.toString().replace(/\/$/, "");
}

function reasoningLevel(value: string): ThinkingLevel {
  const levels: ThinkingLevel[] = ["minimal", "low", "medium", "high", "xhigh", "max"];
  if (!levels.includes(value as ThinkingLevel)) {
    throw new Error("TIMESIGMA_LLM_REASONING is not a supported reasoning level.");
  }
  return value as ThinkingLevel;
}

function text(value: string, name: string): string {
  if (!value.trim()) throw new Error(`${name} must not be empty.`);
  return value.trim();
}

function optionalText(value: string | undefined): string | null {
  return value === undefined || !value.trim() ? null : value.trim();
}

function integer(
  value: string | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const parsed = value === undefined ? fallback : Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new Error(`Integer provider setting must be within [${minimum}, ${maximum}].`);
  }
  return parsed;
}
