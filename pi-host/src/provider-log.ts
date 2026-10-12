import { appendFile, mkdir, readFile } from "node:fs/promises";
import { dirname } from "node:path";

const SECRET_FIELD = /(authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|password|secret|cookie)/i;

export interface ProviderLogRecord {
  sequence: number;
  timestamp: string;
  event: string;
  payload: unknown;
}

export class ProviderLog {
  readonly path: string;
  #sequence = 0;
  #ready: Promise<void>;
  #writes: Promise<unknown> = Promise.resolve();

  constructor(path: string) {
    this.path = path;
    this.#ready = this.#restore();
  }

  append(eventType: string, payload: unknown): Promise<ProviderLogRecord> {
    const operation = this.#writes.then(async () => {
      await this.#ready;
      const record: ProviderLogRecord = {
        sequence: this.#sequence + 1,
        timestamp: new Date().toISOString(),
        event: eventType,
        payload: redactSecrets(payload),
      };
      await mkdir(dirname(this.path), { recursive: true });
      await appendFile(this.path, `${JSON.stringify(record)}\n`, { encoding: "utf8", mode: 0o600 });
      this.#sequence = record.sequence;
      return record;
    });
    // The caller still receives the original rejection; this only keeps later queued writes usable.
    this.#writes = operation.catch(() => undefined);
    return operation;
  }

  async records(): Promise<ProviderLogRecord[]> {
    await this.#writes;
    return readProviderLog(this.path);
  }

  async #restore(): Promise<void> {
    const records = await readProviderLog(this.path);
    this.#sequence = records.at(-1)?.sequence ?? 0;
  }
}

export async function readProviderLog(path: string): Promise<ProviderLogRecord[]> {
  let content: string;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
  return content
    .split("\n")
    .filter(Boolean)
    .map((line, index) => {
      const record = JSON.parse(line) as ProviderLogRecord;
      if (record.sequence !== index + 1) {
        throw new Error(`Provider log sequence mismatch at line ${index + 1}.`);
      }
      return record;
    });
}

export function redactSecrets(value: unknown, seen = new WeakSet<object>()): unknown {
  if (value === undefined || typeof value === "function" || typeof value === "symbol") return null;
  if (typeof value === "bigint") return value.toString();
  if (value !== null && typeof value === "object") {
    if (seen.has(value)) return "[circular]";
    seen.add(value);
    if (Array.isArray(value)) {
      const redacted = value.map((item) => redactSecrets(item, seen));
      seen.delete(value);
      return redacted;
    }
    const redacted: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      redacted[key] = SECRET_FIELD.test(key) ? "[redacted]" : redactSecrets(item, seen);
    }
    seen.delete(value);
    return redacted;
  }
  return value;
}
