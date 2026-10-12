import { rmSync } from "node:fs";
import { mkdtemp, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { readEnvironmentFile } from "../src/environment-file.js";
import { runModelCli } from "../src/model-cli.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("model configuration CLI", () => {
  it("switches Pro to the current Flash ID without changing max effort, limits or credentials", async () => {
    const envPath = await fixtureEnv([
      "TIMESIGMA_LLM_PROVIDER=deepseek",
      "TIMESIGMA_LLM_MODEL_ID=deepseek-v4-pro",
      "TIMESIGMA_LLM_REASONING=max",
      "TIMESIGMA_LLM_MAX_OUTPUT_TOKENS=4096",
      "TIMESIGMA_LLM_TIMEOUT_MS=120000",
      "TIMESIGMA_DEEPSEEK_BASE_URL=https://api.deepseek.com/beta",
      "TIMESIGMA_DEEPSEEK_API_KEY=unit-test-secret",
    ]);
    const before = await readEnvironmentFile(envPath);
    const output = captureOutput();
    await runModelCli(["use", "deepseek/deepseek-flash"], { envPath, runtimeEnv: {}, stdout: output.stream });
    expect(await readEnvironmentFile(envPath)).toEqual({ ...before, TIMESIGMA_LLM_MODEL_ID: "deepseek-flash" });
    await runModelCli(["show"], { envPath, runtimeEnv: {}, stdout: output.stream });
    expect(output.value()).toContain("model: deepseek-flash");
    expect(output.value()).not.toContain("unit-test-secret");
    expect((await stat(envPath)).mode & 0o777).toBe(0o600);
  });

  it("switches GPT-5.5 to GPT-5.6 Sol while preserving credentials and file permissions", async () => {
    const envPath = await fixtureEnv([
      "TIMESIGMA_LLM_PROVIDER=openai",
      "TIMESIGMA_LLM_MODEL_ID=gpt-5.5",
      "TIMESIGMA_LLM_REASONING=xhigh",
      "TIMESIGMA_OPENAI_API_KEY=unit-test-secret",
    ]);
    const output = captureOutput();
    await runModelCli(["use", "gpt-5.6-sol"], {
      envPath,
      runtimeEnv: {},
      stdout: output.stream,
    });
    const env = await readEnvironmentFile(envPath);
    expect(env).toMatchObject({
      TIMESIGMA_LLM_PROVIDER: "openai",
      TIMESIGMA_LLM_MODEL_ID: "gpt-5.6-sol",
      TIMESIGMA_LLM_REASONING: "xhigh",
      TIMESIGMA_OPENAI_API_KEY: "unit-test-secret",
    });
    expect(output.value()).not.toContain("unit-test-secret");
    expect((await stat(envPath)).mode & 0o777).toBe(0o600);
  });

  it("switches provider and uses a reasoning level supported by DeepSeek", async () => {
    const envPath = await fixtureEnv([
      "TIMESIGMA_LLM_PROVIDER=openai",
      "TIMESIGMA_LLM_MODEL_ID=gpt-5.5",
      "TIMESIGMA_LLM_REASONING=xhigh",
      "TIMESIGMA_OPENAI_API_KEY=openai-secret",
      "TIMESIGMA_DEEPSEEK_API_KEY=deepseek-secret",
    ]);
    const output = captureOutput();
    await runModelCli(["use", "deepseek-v4-flash"], {
      envPath,
      runtimeEnv: {},
      stdout: output.stream,
    });
    const env = await readEnvironmentFile(envPath);
    expect(env).toMatchObject({
      TIMESIGMA_LLM_PROVIDER: "deepseek",
      TIMESIGMA_LLM_MODEL_ID: "deepseek-v4-flash",
      TIMESIGMA_LLM_REASONING: "high",
      TIMESIGMA_DEEPSEEK_BASE_URL: "https://api.deepseek.com/beta",
      TIMESIGMA_OPENAI_API_KEY: "openai-secret",
      TIMESIGMA_DEEPSEEK_API_KEY: "deepseek-secret",
    });
    expect(output.value()).toContain("Reasoning changed from xhigh");
    expect(output.value()).not.toContain("openai-secret");
    expect(output.value()).not.toContain("deepseek-secret");
  });

  it("stores a provider key through an injected hidden prompt without printing it", async () => {
    const directory = await temporaryDirectory();
    const envPath = join(directory, ".env");
    const output = captureOutput();
    await runModelCli(["set-key", "deepseek"], {
      envPath,
      runtimeEnv: {},
      stdout: output.stream,
      readSecret: async () => "hidden-deepseek-key",
    });
    expect(await readEnvironmentFile(envPath)).toMatchObject({
      TIMESIGMA_DEEPSEEK_API_KEY: "hidden-deepseek-key",
      TIMESIGMA_DEEPSEEK_BASE_URL: "https://api.deepseek.com/beta",
    });
    expect(await readFile(envPath, "utf8")).not.toContain("TIMESIGMA_LLM_API_KEY");
    expect(output.value()).not.toContain("hidden-deepseek-key");
    expect((await stat(envPath)).mode & 0o777).toBe(0o600);
  });

  it("lists registered choices and rejects unregistered model names", async () => {
    const directory = await temporaryDirectory();
    const envPath = join(directory, ".env");
    const output = captureOutput();
    await runModelCli(["list"], { envPath, runtimeEnv: {}, stdout: output.stream });
    expect(output.value()).toContain("gpt-5.6-sol");
    expect(output.value()).toContain("deepseek-v4-flash");
    await expect(
      runModelCli(["use", "unregistered-model"], {
        envPath,
        runtimeEnv: {},
        stdout: output.stream,
      }),
    ).rejects.toThrow("Unknown registered model");
  });

  it("removes duplicate selected variables so an old trailing value cannot override the switch", async () => {
    const envPath = await fixtureEnv([
      "TIMESIGMA_LLM_PROVIDER=openai",
      "TIMESIGMA_LLM_MODEL_ID=gpt-5.5",
      "TIMESIGMA_LLM_MODEL_ID=gpt-5.6-luna",
      "TIMESIGMA_LLM_REASONING=xhigh",
    ]);
    await runModelCli(["use", "gpt-5.6-sol"], {
      envPath,
      runtimeEnv: {},
      stdout: captureOutput().stream,
    });
    const content = await readFile(envPath, "utf8");
    expect(content.match(/^TIMESIGMA_LLM_MODEL_ID=/gm)).toHaveLength(1);
    expect(await readEnvironmentFile(envPath)).toMatchObject({
      TIMESIGMA_LLM_MODEL_ID: "gpt-5.6-sol",
    });
  });
});

async function fixtureEnv(lines: string[]): Promise<string> {
  const directory = await temporaryDirectory();
  const path = join(directory, ".env");
  await writeFile(path, `${lines.join("\n")}\n`, { mode: 0o600 });
  return path;
}

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "timesigma-model-cli-"));
  temporaryDirectories.push(directory);
  return directory;
}

function captureOutput() {
  let content = "";
  return {
    stream: {
      write(chunk: string | Uint8Array): boolean {
        content += String(chunk);
        return true;
      },
    },
    value: () => content,
  };
}
