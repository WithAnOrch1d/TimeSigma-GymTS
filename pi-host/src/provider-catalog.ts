import {
  getSupportedThinkingLevels,
  type Api,
  type Model,
  type ThinkingLevel,
} from "@earendil-works/pi-ai";
import { getBuiltinModels } from "@earendil-works/pi-ai/providers/all";

export type TimeSigmaProviderProfile = "openai" | "deepseek";
export type TimeSigmaProviderApi = "openai-responses" | "openai-completions";

export const OPENAI_PROVIDER_ID = "timesigma-openai";
export const DEEPSEEK_PROVIDER_ID = "timesigma-deepseek";
export const OPENAI_API_KEY_ENV = "TIMESIGMA_OPENAI_API_KEY";
export const DEEPSEEK_API_KEY_ENV = "TIMESIGMA_DEEPSEEK_API_KEY";

export interface ProviderDefinition {
  profile: TimeSigmaProviderProfile;
  providerId: string;
  providerName: string;
  api: TimeSigmaProviderApi;
  catalogProvider: "openai" | "deepseek";
  defaultBaseUrl: string;
  baseUrlEnv: string;
  keyEnvCandidates: readonly string[];
  defaultModelId: string;
  defaultReasoning: ThinkingLevel;
  modelIds: readonly string[];
}

const PROVIDERS: Readonly<Record<TimeSigmaProviderProfile, ProviderDefinition>> = {
  openai: {
    profile: "openai",
    providerId: OPENAI_PROVIDER_ID,
    providerName: "TimeSigma OpenAI Responses Provider",
    api: "openai-responses",
    catalogProvider: "openai",
    defaultBaseUrl: "https://api.openai.com/v1",
    baseUrlEnv: "TIMESIGMA_OPENAI_BASE_URL",
    keyEnvCandidates: [OPENAI_API_KEY_ENV],
    defaultModelId: "gpt-5.5",
    defaultReasoning: "xhigh",
    modelIds: ["gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
  },
  deepseek: {
    profile: "deepseek",
    providerId: DEEPSEEK_PROVIDER_ID,
    providerName: "TimeSigma DeepSeek Provider",
    api: "openai-completions",
    catalogProvider: "deepseek",
    // Strict JSON-schema tools use DeepSeek's beta route.
    defaultBaseUrl: "https://api.deepseek.com/beta",
    baseUrlEnv: "TIMESIGMA_DEEPSEEK_BASE_URL",
    keyEnvCandidates: [DEEPSEEK_API_KEY_ENV, "DEEPSEEK_API_KEY"],
    defaultModelId: "deepseek-flash",
    defaultReasoning: "high",
    modelIds: ["deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro"],
  },
};

export interface ProviderModelChoice {
  profile: TimeSigmaProviderProfile;
  modelId: string;
  modelName: string;
  api: TimeSigmaProviderApi;
  defaultReasoning: ThinkingLevel;
  supportedReasoning: readonly ThinkingLevel[];
}

export function supportedProviderModels(): readonly ProviderModelChoice[] {
  return Object.values(PROVIDERS).flatMap((definition) =>
    definition.modelIds.map((modelId) => {
      const model = providerCatalogModel(definition, modelId);
      return {
        profile: definition.profile,
        modelId,
        modelName: model.name,
        api: definition.api,
        defaultReasoning: definition.defaultReasoning,
        supportedReasoning: getSupportedThinkingLevels(model).filter(
          (level): level is ThinkingLevel => level !== "off",
        ),
      };
    }),
  );
}

export function providerDefinition(
  profile: TimeSigmaProviderProfile,
): Readonly<ProviderDefinition> {
  return PROVIDERS[profile];
}

export function providerCatalogModel(
  definition: ProviderDefinition,
  modelId: string,
): Model<Api> {
  if (!definition.modelIds.includes(modelId)) {
    throw new Error(
      `Model '${modelId}' is not registered for provider '${definition.profile}'.`,
    );
  }
  // Pi 0.82.1 predates V4.1. Reuse its DeepSeek transport with explicit current metadata.
  const flash = definition.profile === "deepseek" && modelId === "deepseek-flash";
  const catalogId = flash ? "deepseek-v4-flash" : modelId;
  const model = getBuiltinModels(definition.catalogProvider).find(
    (candidate) => candidate.id === catalogId && candidate.api === definition.api,
  );
  if (model === undefined) {
    throw new Error(
      `Pi has no audited ${definition.api} model mapping for '${modelId}'.`,
    );
  }
  if (flash) {
    return {
      ...model,
      id: modelId,
      name: "DeepSeek V4.1 Flash",
      input: ["text", "image"],
      thinkingLevelMap: { ...model.thinkingLevelMap, low: "low" },
      // Official 2026-09-10 peak USD reference; off-peak billing is half.
      cost: { input: 0.3, output: 1.2, cacheRead: 0.006, cacheWrite: 0 },
    };
  }
  return model;
}
