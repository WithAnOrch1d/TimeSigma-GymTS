import {
  createModels,
  createProvider,
  envApiKeyAuth,
  getSupportedThinkingLevels,
  type Api,
  type Model,
  type MutableModels,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { openAIResponsesApi } from "@earendil-works/pi-ai/api/openai-responses.lazy";
import { providerCatalogModel, providerDefinition } from "./provider-catalog.js";
import type { TimeSigmaProviderConfig } from "./provider-config.js";

export interface TimeSigmaProviderRuntime {
  config: TimeSigmaProviderConfig;
  model: Model<Api>;
  models: MutableModels;
  authSource: string;
}

export async function createTimeSigmaProviderRuntime(
  config: TimeSigmaProviderConfig,
  env: NodeJS.ProcessEnv = process.env,
): Promise<TimeSigmaProviderRuntime> {
  const definition = providerDefinition(config.profile);
  if (
    config.providerId !== definition.providerId ||
    config.providerName !== definition.providerName ||
    config.api !== definition.api ||
    !sameStrings(config.keyEnvCandidates, definition.keyEnvCandidates)
  ) {
    throw new Error("Provider config does not match its registered profile.");
  }
  const template = providerCatalogModel(definition, config.modelId);
  const supported = getSupportedThinkingLevels(template);
  if (!supported.includes(config.reasoning)) {
    throw new Error(
      `Reasoning level '${config.reasoning}' is unsupported by '${config.modelId}'; choose one of ${supported.filter((level) => level !== "off").join(", ")}.`,
    );
  }
  const model: Model<Api> = {
    ...template,
    provider: config.providerId,
    baseUrl: config.baseUrl,
    maxTokens: Math.min(template.maxTokens, config.maxOutputTokens),
  };
  const provider = createProvider({
    id: config.providerId,
    name: config.providerName,
    baseUrl: config.baseUrl,
    auth: {
      apiKey: envApiKeyAuth("TimeSigma experiment API key", [...config.keyEnvCandidates]),
    },
    models: [model],
    api: config.api === "openai-responses" ? openAIResponsesApi() : openAICompletionsApi(),
  });
  const models = createModels({
    authContext: {
      env: async (name) => env[name],
      fileExists: async () => false,
    },
  });
  models.setProvider(provider);
  const auth = await models.checkAuth(config.providerId);
  if (auth === undefined) {
    throw new Error(
      `Missing ${config.keyEnvCandidates.join(" or ")}; inject a credential through the environment.`,
    );
  }
  const resolvedModel = models.getModel(config.providerId, config.modelId);
  if (resolvedModel === undefined || resolvedModel.api !== config.api) {
    throw new Error("TimeSigma provider model mapping was not registered correctly.");
  }
  return {
    config,
    model: resolvedModel,
    models,
    authSource: auth.source ?? config.keyEnvCandidates[0] ?? "environment",
  };
}

export function providerProvenance(runtime: TimeSigmaProviderRuntime): Record<string, unknown> {
  const { config, model } = runtime;
  return {
    provider_profile: config.profile,
    provider_id: config.providerId,
    provider_name: config.providerName,
    api: config.api,
    base_url: config.baseUrl,
    exposed_model_id: config.modelId,
    declared_model_snapshot: config.modelSnapshot,
    model_mapping_status: config.modelSnapshot === null ? "rolling_or_proxy_unconfirmed" : "declared",
    model_catalog: {
      context_window: model.contextWindow,
      max_output_tokens: model.maxTokens,
      reasoning: model.reasoning,
      thinking_level_map: model.thinkingLevelMap ?? {},
      reference_cost_per_million_tokens: model.cost,
    },
    reasoning_level: config.reasoning,
    effective_provider_reasoning: model.thinkingLevelMap?.[config.reasoning] ?? config.reasoning,
    timeout_ms: config.timeoutMs,
    max_retries: config.maxRetries,
    retry_base_delay_ms: config.retryBaseDelayMs,
    max_retry_delay_ms: config.maxRetryDelayMs,
    cache_retention: config.cacheRetention,
    request_store: config.requestStore,
    auth_source: runtime.authSource,
    authorization_scope: config.authorizationScope,
    gateway_operator: config.gatewayOperator,
    data_flow: config.dataFlow,
    retention_policy: config.retentionPolicy,
    pricing_source: config.pricingSource,
  };
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
