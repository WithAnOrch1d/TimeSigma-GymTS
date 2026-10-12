import { rmSync } from "node:fs";
import { mkdtemp, readFile, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
  createAssistantMessageEventStream,
  createModels,
  createProvider,
  envApiKeyAuth,
  Type,
  type AssistantMessage,
  type Context,
  type Model,
  type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { openAIResponsesApi } from "@earendil-works/pi-ai/api/openai-responses.lazy";
import { getBuiltinModels } from "@earendil-works/pi-ai/providers/all";
import { afterEach, describe, expect, it } from "vitest";
import {
  DEEPSEEK_API_KEY_ENV,
  DEEPSEEK_PROVIDER_ID,
  OPENAI_API_KEY_ENV,
  OPENAI_PROVIDER_ID,
  providerCatalogModel,
  providerDefinition,
} from "../src/provider-catalog.js";
import { createTimeSigmaProviderRuntime } from "../src/provider-client.js";
import { loadTimeSigmaProviderConfig } from "../src/provider-config.js";
import { importCodexCredential } from "../src/configure-auth.js";
import { ProviderLog } from "../src/provider-log.js";
import { createAuditedProviderController } from "../src/provider-runtime.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("multi-provider config", () => {
  it("defaults DeepSeek to V4.1 Flash while retaining max effort and the output ceiling", async () => {
    const env = {
      TIMESIGMA_LLM_PROVIDER: "deepseek",
      TIMESIGMA_LLM_REASONING: "max",
      [DEEPSEEK_API_KEY_ENV]: "flash-unit-test-secret",
    };
    const config = loadTimeSigmaProviderConfig(env);
    const runtime = await createTimeSigmaProviderRuntime(config, env);
    expect(config.modelId).toBe("deepseek-flash");
    expect(config.pricingSource).toContain("peak USD reference");
    expect(runtime.model).toMatchObject({
      id: "deepseek-flash", name: "DeepSeek V4.1 Flash", api: "openai-completions",
      maxTokens: 4096, contextWindow: 1_000_000,
      thinkingLevelMap: { low: "low", high: "high", max: "max" },
      cost: { input: 0.3, output: 1.2, cacheRead: 0.006, cacheWrite: 0 },
    });
    expect(JSON.stringify(config)).not.toContain("flash-unit-test-secret");
  });

  it("maps the default OpenAI profile to Pi's audited gpt-5.5 Responses model without storing a key", async () => {
    const env = { [OPENAI_API_KEY_ENV]: "unit-test-secret" };
    const config = loadTimeSigmaProviderConfig(env);
    expect(config.profile).toBe("openai");
    expect(config.baseUrl).toBe("https://api.openai.com/v1");
    expect(config.modelId).toBe("gpt-5.5");
    expect(config.api).toBe("openai-responses");
    expect(JSON.stringify(config)).not.toContain("unit-test-secret");
    const runtime = await createTimeSigmaProviderRuntime(config, env);
    expect(runtime.model.provider).toBe(OPENAI_PROVIDER_ID);
    expect(runtime.model.id).toBe("gpt-5.5");
  });

  it("supports GPT-5.6 Sol and DeepSeek V4 through their distinct Pi APIs", async () => {
    const openaiEnv = {
      TIMESIGMA_LLM_PROVIDER: "openai",
      TIMESIGMA_LLM_MODEL_ID: "gpt-5.6-sol",
      TIMESIGMA_LLM_REASONING: "xhigh",
      [OPENAI_API_KEY_ENV]: "openai-unit-test-secret",
    };
    const openai = await createTimeSigmaProviderRuntime(
      loadTimeSigmaProviderConfig(openaiEnv),
      openaiEnv,
    );
    expect(openai.model).toMatchObject({ id: "gpt-5.6-sol", api: "openai-responses" });

    const deepseekEnv = {
      TIMESIGMA_LLM_PROVIDER: "deepseek",
      TIMESIGMA_LLM_MODEL_ID: "deepseek-v4-flash",
      TIMESIGMA_LLM_REASONING: "high",
      [DEEPSEEK_API_KEY_ENV]: "deepseek-unit-test-secret",
    };
    const deepseek = await createTimeSigmaProviderRuntime(
      loadTimeSigmaProviderConfig(deepseekEnv),
      deepseekEnv,
    );
    expect(deepseek.config.baseUrl).toBe("https://api.deepseek.com/beta");
    expect(deepseek.model).toMatchObject({
      id: "deepseek-v4-flash",
      api: "openai-completions",
      provider: DEEPSEEK_PROVIDER_ID,
    });
    expect(JSON.stringify(deepseek.config)).not.toContain("deepseek-unit-test-secret");
  });

  it("requires an isolated experiment key and rejects unsafe provider URLs", async () => {
    const config = loadTimeSigmaProviderConfig({});
    await expect(createTimeSigmaProviderRuntime(config, {})).rejects.toThrow(OPENAI_API_KEY_ENV);
    expect(() =>
      loadTimeSigmaProviderConfig({ TIMESIGMA_OPENAI_BASE_URL: "http://example.test" }),
    ).toThrow("HTTPS");
    expect(() =>
      loadTimeSigmaProviderConfig({ TIMESIGMA_OPENAI_BASE_URL: "https://key@example.test?q=secret" }),
    ).toThrow("credentials");
    expect(() =>
      loadTimeSigmaProviderConfig({ NODE_TLS_REJECT_UNAUTHORIZED: "0" }),
    ).toThrow("NODE_TLS_REJECT_UNAUTHORIZED");
  });

  it("rejects models absent from Pi's audited catalog and unsupported reasoning", async () => {
    const env = { [OPENAI_API_KEY_ENV]: "unit-test-secret" };
    const unknown = loadTimeSigmaProviderConfig({
      ...env,
      TIMESIGMA_LLM_MODEL_ID: "unregistered-model",
    });
    await expect(createTimeSigmaProviderRuntime(unknown, env)).rejects.toThrow("not registered");
    const unsupportedReasoning = loadTimeSigmaProviderConfig({
      TIMESIGMA_LLM_PROVIDER: "deepseek",
      TIMESIGMA_LLM_REASONING: "xhigh",
    });
    await expect(
      createTimeSigmaProviderRuntime(unsupportedReasoning, {
        [DEEPSEEK_API_KEY_ENV]: "unit-test-secret",
      }),
    ).rejects.toThrow("unsupported");
  });
});

describe("local credential import", () => {
  it("copies Codex API auth once into a Git-ignored 0600 env file without returning the key", async () => {
    const directory = await temporaryDirectory();
    const sourcePath = join(directory, "codex-auth.json");
    const targetPath = join(directory, ".env");
    await writeFile(
      sourcePath,
      JSON.stringify({ auth_mode: "apikey", OPENAI_API_KEY: "unit-test-key" }),
    );
    const result = await importCodexCredential({ sourcePath, targetPath });
    expect(result).toEqual({ sourcePath, targetPath, targetMode: "0600" });
    expect(JSON.stringify(result)).not.toContain("unit-test-key");
    expect((await stat(targetPath)).mode & 0o777).toBe(0o600);
    const content = await readFile(targetPath, "utf8");
    expect(content).toContain('TIMESIGMA_OPENAI_API_KEY="unit-test-key"');
    expect(content).toContain("TIMESIGMA_OPENAI_BASE_URL=https://yyt.s-flag.com");
    await expect(importCodexCredential({ sourcePath, targetPath })).rejects.toThrow(
      "refusing to overwrite",
    );
  });

  it("does not create a credential file from malformed or non-API-key Codex auth", async () => {
    const directory = await temporaryDirectory();
    const sourcePath = join(directory, "codex-auth.json");
    const targetPath = join(directory, ".env");
    await writeFile(sourcePath, JSON.stringify({ auth_mode: "oauth" }));
    await expect(importCodexCredential({ sourcePath, targetPath })).rejects.toThrow("auth_mode");
    await expect(readFile(targetPath, "utf8")).rejects.toMatchObject({ code: "ENOENT" });
  });
});

describe("Pi OpenAI Responses integration", () => {
  it("sends authenticated strict-tool requests and parses trace, tool call, usage, and cost", async () => {
    let requestBody: Record<string, unknown> | undefined;
    let requestAuthorization: string | undefined;
    let requestPath: string | undefined;
    const nonce = "protocol-nonce";
    const server = createServer(async (request, response) => {
      requestPath = request.url;
      requestAuthorization = request.headers.authorization;
      const chunks: Buffer[] = [];
      for await (const chunk of request) chunks.push(Buffer.from(chunk));
      requestBody = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
      const toolArguments = JSON.stringify({ nonce });
      const item = {
        type: "function_call",
        id: "fc_timesigma_preflight",
        call_id: "call_timesigma_preflight",
        name: "timesigma_provider_preflight",
        arguments: toolArguments,
        status: "completed",
      };
      const terminal = {
        id: "resp_timesigma_protocol_test",
        status: "completed",
        model: "gpt-5.5-mock-snapshot",
        output: [item],
        usage: {
          input_tokens: 20,
          input_tokens_details: { cached_tokens: 4 },
          output_tokens: 7,
          output_tokens_details: { reasoning_tokens: 2 },
          total_tokens: 27,
        },
      };
      const events = [
        { type: "response.created", response: terminal },
        { type: "response.output_item.added", output_index: 0, item },
        {
          type: "response.function_call_arguments.done",
          output_index: 0,
          item_id: item.id,
          arguments: toolArguments,
        },
        { type: "response.output_item.done", output_index: 0, item },
        { type: "response.completed", response: terminal },
      ];
      response.writeHead(200, {
        "content-type": "text/event-stream",
        "x-request-id": "request-timesigma-protocol-test",
      });
      for (const event of events) {
        response.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
      }
      response.end();
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    try {
      const address = server.address();
      if (address === null || typeof address === "string") throw new Error("mock server has no port");
      const model = {
        ...testModel(),
        provider: "timesigma-protocol-test",
        baseUrl: `http://127.0.0.1:${address.port}/v1`,
      };
      const provider = createProvider({
        id: model.provider,
        auth: { apiKey: envApiKeyAuth("protocol key", ["TIMESIGMA_PROTOCOL_TEST_KEY"]) },
        models: [model],
        api: openAIResponsesApi(),
      });
      const models = createModels({
        authContext: {
          env: async (name) =>
            name === "TIMESIGMA_PROTOCOL_TEST_KEY" ? "protocol-test-key" : undefined,
          fileExists: async () => false,
        },
      });
      models.setProvider(provider);
      let responseTrace: string | undefined;
      const result = await models.completeSimple(model, {
        systemPrompt: "Call the strict preflight tool.",
        messages: [{ role: "user", content: nonce, timestamp: 1 }],
        tools: [
          {
            name: "timesigma_provider_preflight",
            description: "Return the nonce.",
            parameters: Type.Object(
              { nonce: Type.String() },
              { additionalProperties: false },
            ),
            constrainedSampling: { type: "json_schema", strict: "require" },
          },
        ],
      }, {
        reasoning: "xhigh",
        maxTokens: 128,
        cacheRetention: "none",
        maxRetries: 0,
        onResponse: ({ headers }) => {
          responseTrace = headers["x-request-id"];
        },
      });

      expect(requestPath).toBe("/v1/responses");
      expect(requestAuthorization).toBe("Bearer protocol-test-key");
      expect(requestBody).toMatchObject({
        model: "gpt-5.5",
        stream: true,
        store: false,
        max_output_tokens: 128,
        reasoning: { effort: "xhigh" },
      });
      const tools = requestBody?.tools as Array<Record<string, unknown>>;
      expect(tools[0]).toMatchObject({
        type: "function",
        name: "timesigma_provider_preflight",
        strict: true,
      });
      expect(result.stopReason).toBe("toolUse");
      expect(result.responseId).toBe("resp_timesigma_protocol_test");
      expect(responseTrace).toBe("request-timesigma-protocol-test");
      expect(result.content).toContainEqual(
        expect.objectContaining({
          type: "toolCall",
          name: "timesigma_provider_preflight",
          arguments: { nonce },
        }),
      );
      expect(result.usage).toMatchObject({
        input: 16,
        output: 7,
        cacheRead: 4,
        reasoning: 2,
        totalTokens: 27,
      });
      expect(result.usage.cost.total).toBeGreaterThan(0);
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error === undefined ? resolve() : reject(error))),
      );
    }
  });
});

describe("Pi DeepSeek Chat Completions integration", () => {
  it.each(["deepseek-v4-flash", "deepseek-flash"])("sends %s thinking-mode strict tools and parses the streamed call", async (modelId) => {
    let requestBody: Record<string, unknown> | undefined;
    let requestAuthorization: string | undefined;
    let requestPath: string | undefined;
    const nonce = "deepseek-protocol-nonce";
    const server = createServer(async (request, response) => {
      requestPath = request.url;
      requestAuthorization = request.headers.authorization;
      const chunks: Buffer[] = [];
      for await (const chunk of request) chunks.push(Buffer.from(chunk));
      requestBody = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
      const base = {
        id: "chatcmpl-timesigma-deepseek",
        object: "chat.completion.chunk",
        created: 1,
        model: modelId,
      };
      const events = [
        {
          ...base,
          choices: [
            {
              index: 0,
              delta: { role: "assistant", reasoning_content: "I should call the tool." },
              finish_reason: null,
            },
          ],
        },
        {
          ...base,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: "call_timesigma_deepseek",
                    type: "function",
                    function: {
                      name: "timesigma_provider_preflight",
                      arguments: JSON.stringify({ nonce }),
                    },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        {
          ...base,
          choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
          usage: {
            prompt_tokens: 20,
            completion_tokens: 8,
            total_tokens: 28,
            prompt_cache_hit_tokens: 3,
          },
        },
      ];
      response.writeHead(200, {
        "content-type": "text/event-stream",
        "x-request-id": "request-timesigma-deepseek-test",
      });
      for (const event of events) response.write(`data: ${JSON.stringify(event)}\n\n`);
      response.end("data: [DONE]\n\n");
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    try {
      const address = server.address();
      if (address === null || typeof address === "string") throw new Error("mock server has no port");
      const template = providerCatalogModel(providerDefinition("deepseek"), modelId);
      const model = {
        ...template,
        provider: "timesigma-deepseek-protocol-test",
        baseUrl: `http://127.0.0.1:${address.port}/beta`,
      };
      const provider = createProvider({
        id: model.provider,
        auth: { apiKey: envApiKeyAuth("protocol key", ["TIMESIGMA_DEEPSEEK_TEST_KEY"]) },
        models: [model],
        api: openAICompletionsApi(),
      });
      const models = createModels({
        authContext: {
          env: async (name) =>
            name === "TIMESIGMA_DEEPSEEK_TEST_KEY" ? "deepseek-protocol-key" : undefined,
          fileExists: async () => false,
        },
      });
      models.setProvider(provider);
      const result = await models.completeSimple(
        model,
        {
          systemPrompt: "Call the strict preflight tool.",
          messages: [{ role: "user", content: nonce, timestamp: 1 }],
          tools: [
            {
              name: "timesigma_provider_preflight",
              description: "Return the nonce.",
              parameters: Type.Object({ nonce: Type.String() }, { additionalProperties: false }),
              constrainedSampling: { type: "json_schema", strict: "require" },
            },
          ],
        },
        {
          reasoning: "high",
          maxTokens: 128,
          cacheRetention: "none",
          maxRetries: 0,
        },
      );

      expect(requestPath).toBe("/beta/chat/completions");
      expect(requestAuthorization).toBe("Bearer deepseek-protocol-key");
      expect(requestBody).toMatchObject({
        model: modelId,
        stream: true,
        thinking: { type: "enabled" },
        reasoning_effort: "high",
      });
      const tools = requestBody?.tools as Array<Record<string, unknown>>;
      expect(tools[0]).toMatchObject({
        type: "function",
        function: { name: "timesigma_provider_preflight", strict: true },
      });
      expect(result.stopReason).toBe("toolUse");
      expect(result.responseId).toBe("chatcmpl-timesigma-deepseek");
      expect(result.content).toContainEqual(
        expect.objectContaining({
          type: "toolCall",
          name: "timesigma_provider_preflight",
          arguments: { nonce },
        }),
      );
      expect(result.usage).toMatchObject({ totalTokens: 28, cacheRead: 3 });
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error === undefined ? resolve() : reject(error))),
      );
    }
  });
});

describe("provider event log", () => {
  it("records ordered events and redacts secrets", async () => {
    const path = await logPath();
    const log = new ProviderLog(path);
    const shared = { allowed: "shared" };
    const circular: { self?: unknown } = {};
    circular.self = circular;
    await log.append("request", {
      authorization: "Bearer secret",
      nested: { api_key: "secret", allowed: "value" },
      first: shared,
      second: shared,
      circular,
      execute: () => "not serializable",
      missing: undefined,
    });
    const rawRecords = (await readFile(path, "utf8"))
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as unknown);
    expect(rawRecords).toHaveLength(1);
    await log.append("response", { status: 200 });
    const records = await log.records();
    expect(records[0]?.payload).toEqual({
      authorization: "[redacted]",
      nested: { api_key: "[redacted]", allowed: "value" },
      first: { allowed: "shared" },
      second: { allowed: "shared" },
      circular: { self: "[circular]" },
      execute: null,
      missing: null,
    });
    expect(records.map((record) => record.sequence)).toEqual([1, 2]);
    expect(records[1]).toMatchObject({ event: "response", payload: { status: 200 } });
  });
});

describe("audited provider runtime", () => {
  it("retries a transient 429 with the same frozen request and records usage", async () => {
    const harness = await runtimeHarness([
      assistant("error", "429 rate limit"),
      assistant("stop", undefined, "ok", 23),
    ]);
    const result = await complete(harness.controller.streamFn, harness.model, SMALL_CONTEXT);
    expect(result.stopReason).toBe("stop");
    expect(harness.controller.stats()).toMatchObject({
      providerAttempts: 2,
      tokens: 23,
      paused: false,
    });
    const records = await harness.log.records();
    const attempts = records.filter((record) => record.event === "provider_attempt_finished");
    expect(attempts).toHaveLength(2);
    expect(records.some((record) => record.event === "retry_scheduled")).toBe(true);
    expect(attempts.map((record) => field(record.payload, "actual_payload"))).toEqual([
      { store: false, model: harness.config.modelId },
      { store: false, model: harness.config.modelId },
    ]);
    expect(field(attempts[1]?.payload, "usage")).toMatchObject({ totalTokens: 23 });
    expect(JSON.stringify(attempts)).not.toContain("authorization");
  });

  it("classifies a timeout as transient and retries the identical call", async () => {
    const harness = await runtimeHarness([
      assistant("error", "Request timed out"),
      assistant("stop", undefined, "recovered", 11),
    ]);
    const result = await complete(harness.controller.streamFn, harness.model, SMALL_CONTEXT);
    expect(result.stopReason).toBe("stop");
    expect(harness.controller.stats()).toMatchObject({ providerAttempts: 2, paused: false });
  });

  it("does not retry quota or billing exhaustion and pauses the batch", async () => {
    const harness = await runtimeHarness([assistant("error", "insufficient_quota billing")]);
    const result = await complete(harness.controller.streamFn, harness.model, SMALL_CONTEXT);
    expect(result.stopReason).toBe("error");
    expect(harness.controller.stats()).toMatchObject({
      providerAttempts: 1,
      paused: true,
    });
  });

  it("returns provider errors and opens the circuit on systemic failure", async () => {
    const harness = await runtimeHarness(
      [assistant("error", "503 service unavailable"), assistant("error", "503 service unavailable")],
      { TIMESIGMA_LLM_MAX_RETRIES: "0", TIMESIGMA_LLM_SYSTEMIC_FAILURE_THRESHOLD: "2" },
    );
    const first = await complete(harness.controller.streamFn, harness.model, SMALL_CONTEXT);
    expect(first.stopReason).toBe("error");
    expect(first.provider).toBe(harness.model.provider);
    expect(first.content).toEqual([]);
    const second = await complete(harness.controller.streamFn, harness.model, SMALL_CONTEXT);
    expect(second.stopReason).toBe("error");
    expect(harness.controller.stats()).toMatchObject({
      providerAttempts: 2,
      consecutiveFailures: 2,
      paused: true,
    });
    const records = await harness.log.records();
    expect(records.filter((record) => record.event === "batch_paused")).toHaveLength(1);
  });

  it("enforces the actual-call budget, including retries", async () => {
    const harness = await runtimeHarness(
      [assistant("error", "429 rate limit"), assistant("stop", undefined, "should not run")],
      {},
      { maxCalls: 1, maxTokens: 1_000 },
    );
    const result = await complete(harness.controller.streamFn, harness.model, SMALL_CONTEXT);
    expect(result.stopReason).toBe("error");
    expect(harness.sourceCalls()).toBe(1);
    expect(harness.controller.stats()).toMatchObject({ providerAttempts: 1, paused: true });
  });

  it("blocks any provider or model switch before network dispatch", async () => {
    const harness = await runtimeHarness([assistant("stop", undefined, "must not run")]);
    const switched = { ...harness.model, id: "different-model" };
    const result = await complete(harness.controller.streamFn, switched, SMALL_CONTEXT);
    expect(result.stopReason).toBe("error");
    expect(harness.sourceCalls()).toBe(0);
    expect(harness.controller.stats().paused).toBe(true);
  });

  it("pauses when provider-reported tokens exceed the frozen token budget", async () => {
    const harness = await runtimeHarness(
      [assistant("stop", undefined, "oversized", 101)],
      {},
      { maxCalls: 1, maxTokens: 100 },
    );
    const result = await complete(harness.controller.streamFn, harness.model, SMALL_CONTEXT);
    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toBe("LLM_BUDGET_EXCEEDED");
    expect(harness.controller.stats()).toMatchObject({ tokens: 101, paused: true });
  });

  it("budgets transmitted tool content without charging session-only details twice", async () => {
    const harness = await runtimeHarness([assistant("stop", undefined, "accepted", 20)], {}, { maxCalls: 4, maxTokens: 1000 }, true);
    const toolResult = {
      role: "toolResult" as const, toolCallId: "memory-read", toolName: "memory_read",
      content: [{ type: "text" as const, text: "A compact observed fact." }],
      details: { largeSessionRecord: "x".repeat(50000) }, isError: false, timestamp: 2,
    };
    const context: Context = { ...SMALL_CONTEXT, messages: [...SMALL_CONTEXT.messages, toolResult] };
    expect((await complete(harness.controller.streamFn, harness.model, context)).stopReason).toBe("stop");
    expect(harness.sourceCalls()).toBe(1);
    const oversized: Context = { ...context, messages: [...SMALL_CONTEXT.messages, { ...toolResult, content: [{ type: "text", text: "x".repeat(5000) }] }] };
    expect((await complete(harness.controller.streamFn, harness.model, oversized)).errorMessage).toBe("LLM_BUDGET_EXHAUSTED");
    expect(harness.sourceCalls()).toBe(1);
    expect(harness.controller.stats().tokens).toBe(20);
  });

  it("uses measured prompt usage for an unchanged prefix and rejects changed oversized input", async () => {
    const first = assistant("stop", undefined, "ok", 200);
    first.usage.input = 50;
    first.usage.cacheRead = 150;
    const harness = await runtimeHarness([first, assistant("stop", undefined, "ok", 200)], {}, { maxCalls: 4, maxTokens: 1000 }, true);
    const context: Context = { ...SMALL_CONTEXT, messages: [{ role: "user", content: "x".repeat(1600), timestamp: 1 }] };
    expect((await complete(harness.controller.streamFn, harness.model, context)).stopReason).toBe("stop");
    expect((await complete(harness.controller.streamFn, harness.model, context)).stopReason).toBe("stop");
    expect(harness.controller.budgetState().lastInputEstimate).toBeGreaterThanOrEqual(200);
    expect(harness.controller.stats().tokens).toBe(400);
    expect((await complete(harness.controller.streamFn, harness.model, { ...context, systemPrompt: "y".repeat(5000) })).errorMessage).toBe("LLM_BUDGET_EXHAUSTED");
    expect(harness.sourceCalls()).toBe(2);
  });

  it("restores measured input across logged runtime-only tool fields", async () => {
    const tool = { name: "lookup", description: "Read facts", parameters: {} as NonNullable<Context["tools"]>[number]["parameters"], execute: () => null, label: "UI label" };
    const context: Context = { ...SMALL_CONTEXT, tools: [tool], messages: [{ role: "user", content: "x".repeat(1600), timestamp: 1 }] };
    const logged = JSON.parse(JSON.stringify(context, (_key, value) => typeof value === "function" ? null : value)) as Context;
    const harness = await runtimeHarness([assistant("stop", undefined, "ok", 200)], {}, { maxCalls: 2, maxTokens: 500 }, true, { context: logged, tokens: 200 });
    expect((await complete(harness.controller.streamFn, harness.model, context)).stopReason).toBe("stop");
    expect(harness.sourceCalls()).toBe(1);
    expect(harness.controller.stats().tokens).toBe(200);
  });
});

const SMALL_CONTEXT: Context = {
  systemPrompt: "Use the test protocol.",
  messages: [{ role: "user", content: "test", timestamp: 1 }],
};

async function runtimeHarness(
  responses: AssistantMessage[],
  overrides: NodeJS.ProcessEnv = {},
  budget = { maxCalls: 4, maxTokens: 1_000 },
  useDefaultEstimate = false,
  previousInput?: { context: Context; tokens: number },
) {
  const config = loadTimeSigmaProviderConfig({
    TIMESIGMA_LLM_MAX_RETRIES: "1",
    TIMESIGMA_LLM_RETRY_BASE_DELAY_MS: "0",
    ...overrides,
  });
  const model = testModel();
  const log = new ProviderLog(await logPath());
  let calls = 0;
  const source: StreamFn = (_model, _context, options) => {
    const response = responses[calls] ?? assistant("error", "500 missing fixture response");
    calls += 1;
    return messageStream(response, options, { store: false, model: config.modelId });
  };
  const controller = createAuditedProviderController({
    config,
    model,
    source,
    log,
    budget,
    ...(useDefaultEstimate ? {} : { estimateInputTokens: () => 10 }),
    ...(previousInput === undefined ? {} : { previousInput }),
  });
  return { config, controller, log, model, sourceCalls: () => calls };
}

function messageStream(
  message: AssistantMessage,
  options: SimpleStreamOptions | undefined,
  payload: unknown,
) {
  const stream = createAssistantMessageEventStream();
  void (async () => {
    await options?.onPayload?.(payload, testModel());
    await options?.onResponse?.(
      {
        status: message.stopReason === "error" ? 429 : 200,
        headers: { "x-request-id": `request-${message.timestamp}`, authorization: "secret" },
      },
      testModel(),
    );
    stream.push({ type: "start", partial: message });
    if (message.stopReason === "error" || message.stopReason === "aborted") {
      stream.push({ type: "error", reason: message.stopReason, error: message });
    } else {
      stream.push({ type: "done", reason: message.stopReason, message });
    }
  })();
  return stream;
}

function assistant(
  stopReason: AssistantMessage["stopReason"],
  errorMessage?: string,
  text = "",
  totalTokens = 0,
): AssistantMessage {
  return {
    role: "assistant",
    content: text ? [{ type: "text", text }] : [],
    api: "openai-responses",
    provider: OPENAI_PROVIDER_ID,
    model: "gpt-5.5",
    responseModel: "gpt-5.5-2026-07-01",
    responseId: `response-${Math.random()}`,
    usage: {
      input: totalTokens,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens,
      cost: { input: totalTokens / 1_000_000, output: 0, cacheRead: 0, cacheWrite: 0, total: totalTokens / 1_000_000 },
    },
    stopReason,
    ...(errorMessage === undefined ? {} : { errorMessage }),
    timestamp: Date.now() + Math.random(),
  };
}

function testModel(): Model<"openai-responses"> {
  const template = getBuiltinModels("openai").find(
    (model) => model.id === "gpt-5.5" && model.api === "openai-responses",
  );
  if (template === undefined) throw new Error("Pi gpt-5.5 test fixture is unavailable");
  return { ...template, provider: OPENAI_PROVIDER_ID };
}

async function complete(streamFn: StreamFn, model: Model<"openai-responses">, context: Context) {
  const stream = await streamFn(model, context);
  return stream.result();
}

async function logPath(): Promise<string> {
  const directory = await temporaryDirectory();
  return join(directory, "provider.jsonl");
}

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "timesigma-provider-"));
  temporaryDirectories.push(directory);
  return directory;
}

function field(value: unknown, name: string): unknown {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)[name]
    : undefined;
}
