import { constants } from "node:fs";
import { access, chmod, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

export interface CodexCredentialImportOptions {
  sourcePath?: string;
  targetPath?: string;
}

export interface CodexCredentialImportResult {
  sourcePath: string;
  targetPath: string;
  targetMode: "0600";
}

export async function importCodexCredential(
  options: CodexCredentialImportOptions = {},
): Promise<CodexCredentialImportResult> {
  const sourcePath = resolve(options.sourcePath ?? `${homedir()}/.codex/auth.json`);
  const targetPath = resolve(options.targetPath ?? ".env");
  await assertTargetDoesNotExist(targetPath);

  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(sourcePath, "utf8"));
  } catch (error) {
    throw new Error(`Cannot read a valid Codex auth file at ${sourcePath}.`, { cause: error });
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Codex auth file must contain a JSON object.");
  }
  const auth = parsed as Record<string, unknown>;
  if (auth.auth_mode !== "apikey") {
    throw new Error("Codex auth_mode must be 'apikey' for TimeSigma import.");
  }
  if (typeof auth.OPENAI_API_KEY !== "string" || auth.OPENAI_API_KEY.trim().length < 8) {
    throw new Error("Codex auth file does not contain a usable OPENAI_API_KEY.");
  }

  const content = [
    "# Local TimeSigma experiment credential. Never commit this file.",
    "# Generated once from ~/.codex/auth.json by npm run auth:import-codex.",
    "TIMESIGMA_LLM_PROVIDER=openai",
    `TIMESIGMA_OPENAI_API_KEY=${JSON.stringify(auth.OPENAI_API_KEY.trim())}`,
    "TIMESIGMA_OPENAI_BASE_URL=https://yyt.s-flag.com",
    "TIMESIGMA_LLM_MODEL_ID=gpt-5.5",
    "TIMESIGMA_LLM_REASONING=xhigh",
    "",
  ].join("\n");
  await writeFile(targetPath, content, { encoding: "utf8", mode: 0o600, flag: "wx" });
  await chmod(targetPath, 0o600);
  return { sourcePath, targetPath, targetMode: "0600" };
}

async function assertTargetDoesNotExist(path: string): Promise<void> {
  try {
    await access(path, constants.F_OK);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  throw new Error(
    `${path} already exists; refusing to overwrite a local credential. Remove or rotate it explicitly.`,
  );
}

function isMainModule(): boolean {
  const entry = process.argv[1];
  return entry !== undefined && import.meta.url === pathToFileURL(resolve(entry)).href;
}

if (isMainModule()) {
  importCodexCredential()
    .then((result) => {
      process.stdout.write(
        `TimeSigma credential imported to ${result.targetPath} with mode ${result.targetMode}.\n`,
      );
    })
    .catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      process.stderr.write(`TimeSigma credential import failed: ${message}\n`);
      process.exitCode = 1;
    });
}
