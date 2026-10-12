import { readFile, writeFile, mkdir, access } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { JsonlSessionRepo } from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import { createDeliveryAgent } from "./delivery-agent.js";
import { createAuditedModels } from "./audited-models.js";
import { createTimeSigmaProviderRuntime, providerProvenance } from "./provider-client.js";
import { loadTimeSigmaProviderConfig } from "./provider-config.js";
import { ProviderLog } from "./provider-log.js";
import { createAuditedProviderController } from "./provider-runtime.js";
import { PythonWorkerClient } from "./python-worker.js";

export async function planDelivery(root: string): Promise<void> {
  const protocol = JSON.parse(await readFile(join(root, "protocol.json"), "utf8"));
  if (await exists(join(root, "planning-closed.json"))) throw new Error("planning is already closed");
  const config = loadTimeSigmaProviderConfig();
  const effective = { profile: config.profile, model: config.modelId, reasoning: config.reasoning, max_output_tokens: config.maxOutputTokens, base_url: config.baseUrl, timeout_ms: config.timeoutMs, max_retries: config.maxRetries };
  for (const [key, value] of Object.entries(effective)) {
    if (protocol.provider[key] !== value) throw new Error(`frozen Provider mismatch: ${key}`);
  }
  const runtime = await createTimeSigmaProviderRuntime(config);
  const repositoryRoot = resolve(import.meta.dirname, "../..");
  const environment = new NodeExecutionEnv({ cwd: repositoryRoot });
  const worker = new PythonWorkerClient({ pythonExecutable: join(repositoryRoot, ".venv/bin/python"), repositoryRoot });
  try {
    for (const curve of (protocol.planning_units ?? protocol.curves) as string[]) {
      const directory = join(root, "plans", curve);
      if (await exists(join(directory, "provider-outcome.json"))) {
        const previous = JSON.parse(await readFile(join(directory, "provider-outcome.json"), "utf8"));
        if (previous.status === "frozen" && await exists(join(directory, "final.json"))) continue;
        throw new Error(`planning previously failed for ${curve}; no fresh budget or automatic replan`);
      }
      if (await exists(join(directory, "planning-start.json"))) throw new Error(`interrupted planning for ${curve}; retain logs, do not reset its budget`);
      await mkdir(directory, { recursive: true });
      const controller = createAuditedProviderController({
        config, model: runtime.model, source: runtime.models.streamSimple.bind(runtime.models),
        log: new ProviderLog(join(directory, "provider.jsonl")),
        budget: { maxCalls: protocol.planning.max_calls, maxTokens: protocol.planning.max_tokens },
      });
      const sessions = new JsonlSessionRepo({ fs: environment, sessionsRoot: join(directory, "sessions") });
      const started = Date.now();
      const harness = createDeliveryAgent({
        session: await sessions.create({ cwd: "timesigma-delivery", metadata: { purpose: "frozen-one-shot-planning" } }),
        models: createAuditedModels(runtime.models, controller), model: runtime.model,
        toolContext: { worker, deliveryRoot: root, curve, deadline: started + protocol.planning.max_wall_seconds * 1000, maxOutputTokens: config.maxOutputTokens, requestTimeoutMs: config.timeoutMs, remainingBudget: () => ({ ...controller.budgetState() }) },
      });
      const timer = setTimeout(() => harness.abort(), protocol.planning.max_wall_seconds * 1000);
      let failure: string | null = null;
      try {
        await harness.prompt("Load the allowed context, choose and preview one complete Pipeline, then freeze it. Do not execute or evaluate anything.");
        if (!(await exists(join(directory, "final.json")))) throw new Error("Provider stopped without a frozen plan");
      } catch (error) {
        failure = error instanceof Error ? error.message : String(error);
      } finally {
        clearTimeout(timer);
      }
      await writeFile(join(directory, "provider-outcome.json"), JSON.stringify({ status: failure === null ? "frozen" : "failed", failure, usage: controller.stats(), provider: providerProvenance(runtime), wall_seconds: (Date.now() - started) / 1000 }, null, 2) + "\n", { flag: "wx" });
      if (failure !== null) throw new Error(failure);
    }
  } finally {
    await worker.stop();
    await environment.cleanup();
  }
}

async function exists(path: string): Promise<boolean> {
  try { await access(path); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.length !== 3) throw new Error("Usage: node dist/delivery-cli.js RUN_ROOT");
  await planDelivery(resolve(process.argv[2]!));
}
