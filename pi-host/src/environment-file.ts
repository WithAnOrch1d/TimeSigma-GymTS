import { randomUUID } from "node:crypto";
import { chmod, readFile, rename, unlink, writeFile } from "node:fs/promises";

export async function readEnvironmentFile(path: string): Promise<NodeJS.ProcessEnv> {
  let content: string;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return {};
    throw error;
  }
  const env: NodeJS.ProcessEnv = {};
  for (const line of content.split(/\r?\n/)) {
    const match = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/.exec(line);
    if (match?.[1] === undefined || match[2] === undefined) continue;
    env[match[1]] = decodeEnvironmentValue(match[2]);
  }
  return env;
}

export async function updateEnvironmentFile(
  path: string,
  updates: Readonly<Record<string, string>>,
): Promise<void> {
  let content = "";
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  const pending = new Map(Object.entries(updates));
  const updateKeys = new Set(pending.keys());
  const output: string[] = [];
  for (const line of content.split(/\r?\n/)) {
    const match = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=/.exec(line);
    const key = match?.[1];
    if (key === undefined || !updateKeys.has(key)) {
      output.push(line);
      continue;
    }
    if (!pending.has(key)) continue;
    output.push(`${key}=${encodeEnvironmentValue(pending.get(key) as string)}`);
    pending.delete(key);
  }
  while (output.length > 0 && output.at(-1) === "") output.pop();
  if (output.length > 0) output.push("");
  for (const [key, value] of pending) output.push(`${key}=${encodeEnvironmentValue(value)}`);
  output.push("");
  await writeEnvironmentFile(path, output.join("\n"));
}

async function writeEnvironmentFile(path: string, content: string): Promise<void> {
  const temporaryPath = `${path}.tmp-${process.pid}-${randomUUID()}`;
  try {
    await writeFile(temporaryPath, content, { encoding: "utf8", mode: 0o600, flag: "wx" });
    await rename(temporaryPath, path);
    await chmod(path, 0o600);
  } catch (error) {
    await unlink(temporaryPath).catch(() => undefined);
    throw error;
  }
}

function decodeEnvironmentValue(value: string): string {
  const trimmed = value.trim();
  if (trimmed.startsWith('"') && trimmed.endsWith('"')) {
    try {
      const parsed = JSON.parse(trimmed);
      return typeof parsed === "string" ? parsed : trimmed;
    } catch {
      return trimmed;
    }
  }
  if (trimmed.startsWith("'") && trimmed.endsWith("'")) return trimmed.slice(1, -1);
  return trimmed;
}

function encodeEnvironmentValue(value: string): string {
  return /^[A-Za-z0-9_./:@+-]+$/.test(value) ? value : JSON.stringify(value);
}
