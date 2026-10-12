import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { ThinkingLevel } from "@earendil-works/pi-ai";
import { readEnvironmentFile, updateEnvironmentFile } from "./environment-file.js";
import {
  providerDefinition,
  supportedProviderModels,
  type TimeSigmaProviderProfile,
} from "./provider-catalog.js";
import { loadTimeSigmaProviderConfig } from "./provider-config.js";

export interface ModelCliOptions {
  envPath?: string;
  runtimeEnv?: NodeJS.ProcessEnv;
  stdout?: Pick<NodeJS.WriteStream, "write">;
  readSecret?: (prompt: string) => Promise<string>;
}

export async function runModelCli(
  args: readonly string[],
  options: ModelCliOptions = {},
): Promise<number> {
  const envPath = resolve(options.envPath ?? ".env");
  const stdout = options.stdout ?? process.stdout;
  const command = args[0] ?? "show";
  if (command === "help" || command === "--help" || command === "-h") {
    stdout.write(usage());
    return 0;
  }
  if (command === "list") {
    const fileEnv = await readEnvironmentFile(envPath);
    let activeModel: string | undefined;
    try {
      activeModel = loadTimeSigmaProviderConfig(fileEnv).modelId;
    } catch {
      activeModel = undefined;
    }
    const lines = supportedProviderModels().map((choice) => {
      const active = choice.modelId === activeModel ? "*" : " ";
      return `${active} ${choice.profile.padEnd(8)} ${choice.modelId.padEnd(23)} ${choice.api.padEnd(20)} ${choice.supportedReasoning.join(",")}`;
    });
    stdout.write(`  provider model                   api                  reasoning\n${lines.join("\n")}\n`);
    return 0;
  }
  if (command === "show") {
    const fileEnv = await readEnvironmentFile(envPath);
    const effectiveEnv = { ...fileEnv, ...(options.runtimeEnv ?? process.env) };
    const config = loadTimeSigmaProviderConfig(effectiveEnv);
    const choice = supportedProviderModels().find(
      (candidate) => candidate.profile === config.profile && candidate.modelId === config.modelId,
    );
    if (choice === undefined) {
      throw new Error(`Model '${config.modelId}' is not registered for provider '${config.profile}'.`);
    }
    if (!choice.supportedReasoning.includes(config.reasoning)) {
      throw new Error(
        `Reasoning '${config.reasoning}' is unsupported by '${config.modelId}'; choose ${choice.supportedReasoning.join(", ")}.`,
      );
    }
    const credentialSource = config.keyEnvCandidates.find(
      (name) => Boolean(effectiveEnv[name]?.trim()),
    );
    stdout.write(
      [
        `env: ${envPath}`,
        `provider: ${config.profile}`,
        `model: ${config.modelId}`,
        `api: ${config.api}`,
        `base_url: ${config.baseUrl}`,
        `reasoning: ${config.reasoning}`,
        `credential: ${credentialSource === undefined ? "missing" : `configured (${credentialSource})`}`,
        "",
      ].join("\n"),
    );
    return 0;
  }
  if (command === "use") {
    const selection = parseSelection(args.slice(1));
    const choices = supportedProviderModels();
    const choice = choices.find(
      (candidate) =>
        candidate.modelId === selection.modelId &&
        (selection.profile === undefined || candidate.profile === selection.profile),
    );
    if (choice === undefined) {
      throw new Error(`Unknown registered model selection '${selection.display}'. Run 'npm run model -- list'.`);
    }
    const fileEnv = await readEnvironmentFile(envPath);
    const existingReasoning = fileEnv.TIMESIGMA_LLM_REASONING as ThinkingLevel | undefined;
    const requestedReasoning = selection.reasoning ?? existingReasoning ?? choice.defaultReasoning;
    const reasoning = choice.supportedReasoning.includes(requestedReasoning)
      ? requestedReasoning
      : choice.defaultReasoning;
    const definition = providerDefinition(choice.profile);
    const updates: Record<string, string> = {
      TIMESIGMA_LLM_PROVIDER: choice.profile,
      TIMESIGMA_LLM_MODEL_ID: choice.modelId,
      TIMESIGMA_LLM_REASONING: reasoning,
    };
    if (!fileEnv[definition.baseUrlEnv]?.trim()) {
      updates[definition.baseUrlEnv] = definition.defaultBaseUrl;
    }
    loadTimeSigmaProviderConfig({ ...fileEnv, ...updates });
    await updateEnvironmentFile(envPath, updates);
    stdout.write(
      `Switched TimeSigma to ${choice.profile}/${choice.modelId} with reasoning=${reasoning}.\n`,
    );
    if (reasoning !== requestedReasoning) {
      stdout.write(
        `Reasoning changed from ${requestedReasoning} because ${choice.modelId} supports ${choice.supportedReasoning.join(", ")}.\n`,
      );
    }
    stdout.write("Run 'npm run preflight' when you are ready to make one paid qualification call.\n");
    return 0;
  }
  if (command === "set-key") {
    const profile = parseProfile(args[1]);
    if (args.length !== 2) throw new Error("Usage: npm run model -- set-key <openai|deepseek>");
    const readSecret = options.readSecret ?? readHiddenSecret;
    const secret = (await readSecret(`Enter ${profile} API key: `)).trim();
    if (secret.length < 8) throw new Error("API key is too short.");
    const keyEnv = providerDefinition(profile).keyEnvCandidates[0];
    if (keyEnv === undefined) throw new Error(`Provider '${profile}' has no credential setting.`);
    const definition = providerDefinition(profile);
    const fileEnv = await readEnvironmentFile(envPath);
    await updateEnvironmentFile(envPath, {
      [keyEnv]: secret,
      ...(fileEnv[definition.baseUrlEnv]?.trim()
        ? {}
        : { [definition.baseUrlEnv]: definition.defaultBaseUrl }),
    });
    stdout.write(`Stored ${profile} credential as ${keyEnv} in ${envPath} with mode 0600.\n`);
    return 0;
  }
  throw new Error(`Unknown model command '${command}'.\n${usage()}`);
}

interface Selection {
  profile?: TimeSigmaProviderProfile;
  modelId: string;
  reasoning?: ThinkingLevel;
  display: string;
}

function parseSelection(args: readonly string[]): Selection {
  const values = [...args];
  let reasoning: ThinkingLevel | undefined;
  const reasoningIndex = values.indexOf("--reasoning");
  if (reasoningIndex >= 0) {
    const value = values[reasoningIndex + 1];
    if (value === undefined) throw new Error("--reasoning requires a value.");
    reasoning = parseReasoning(value);
    values.splice(reasoningIndex, 2);
  }
  if (values.length === 1 && values[0]?.includes("/")) {
    const [provider, modelId, ...extra] = values[0].split("/");
    if (provider === undefined || modelId === undefined || extra.length > 0) {
      throw new Error("Model selection must be <model> or <provider>/<model>.");
    }
    return {
      profile: parseProfile(provider),
      modelId,
      ...(reasoning === undefined ? {} : { reasoning }),
      display: values[0],
    };
  }
  if (values.length === 2) {
    return {
      profile: parseProfile(values[0]),
      modelId: values[1] as string,
      ...(reasoning === undefined ? {} : { reasoning }),
      display: values.join("/"),
    };
  }
  if (values.length === 1 && values[0] !== undefined) {
    return {
      modelId: values[0],
      ...(reasoning === undefined ? {} : { reasoning }),
      display: values[0],
    };
  }
  throw new Error("Usage: npm run model -- use [provider/]model [--reasoning level]");
}

function parseProfile(value: string | undefined): TimeSigmaProviderProfile {
  if (value !== "openai" && value !== "deepseek") {
    throw new Error("Provider must be 'openai' or 'deepseek'.");
  }
  return value;
}

function parseReasoning(value: string): ThinkingLevel {
  const levels: ThinkingLevel[] = ["minimal", "low", "medium", "high", "xhigh", "max"];
  if (!levels.includes(value as ThinkingLevel)) {
    throw new Error(`Unsupported reasoning level '${value}'.`);
  }
  return value as ThinkingLevel;
}

async function readHiddenSecret(prompt: string): Promise<string> {
  if (!process.stdin.isTTY || !process.stdout.isTTY || process.stdin.setRawMode === undefined) {
    throw new Error("set-key requires an interactive terminal so the credential can be hidden.");
  }
  process.stdout.write(prompt);
  process.stdin.setEncoding("utf8");
  process.stdin.setRawMode(true);
  process.stdin.resume();
  return new Promise<string>((resolveSecret, reject) => {
    let secret = "";
    const finish = (error?: Error): void => {
      process.stdin.off("data", onData);
      process.stdin.setRawMode?.(false);
      process.stdin.pause();
      process.stdout.write("\n");
      if (error === undefined) resolveSecret(secret);
      else reject(error);
    };
    const onData = (chunk: string | Buffer): void => {
      for (const character of String(chunk)) {
        if (character === "\u0003") {
          finish(new Error("Credential entry cancelled."));
          return;
        }
        if (character === "\r" || character === "\n") {
          finish();
          return;
        }
        if (character === "\u007f" || character === "\b") secret = secret.slice(0, -1);
        else if (character >= " ") secret += character;
      }
    };
    process.stdin.on("data", onData);
  });
}

function usage(): string {
  return [
    "TimeSigma model configuration",
    "  npm run model -- list",
    "  npm run model -- show",
    "  npm run model -- use gpt-5.6-sol",
    "  npm run model -- use deepseek/deepseek-flash --reasoning max",
    "  npm run model -- set-key <openai|deepseek>",
    "",
  ].join("\n");
}

function isMainModule(): boolean {
  const entry = process.argv[1];
  return entry !== undefined && import.meta.url === pathToFileURL(resolve(entry)).href;
}

if (isMainModule()) {
  runModelCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`TimeSigma model configuration failed: ${message}\n`);
    process.exitCode = 1;
  });
}
