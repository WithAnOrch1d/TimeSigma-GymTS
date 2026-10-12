import type { AgentHarnessTool, Session } from "@earendil-works/pi-agent-core";
import type { TSchema } from "@earendil-works/pi-ai";
import type { ResearchWorker } from "./research-agent.js";
import { redactSecrets } from "./provider-log.js";

export function strictToolParameters<T extends TSchema>(schema: T): T {
  // Pi 0.82.1 runs both TypeBox Convert and a legacy-symbol-gated JSON coercer.
  // A plain root drops TypeBox 1.x's non-enumerable ~kind (no Convert traversal);
  // the legacy marker skips the second coercer. Compile.Check still validates
  // the complete, unchanged JSON Schema. Symbols never enter Provider payloads.
  const parameters = { ...schema };
  Object.defineProperty(parameters, Symbol.for("TypeBox.Kind"), { value: "Object" });
  return parameters;
}

export function withStrictToolArguments<TContext extends { worker: ResearchWorker }>(
  tool: AgentHarnessTool<TContext>, session: Session,
): AgentHarnessTool<TContext> {
  return {
    ...tool,
    parameters: strictToolParameters(tool.parameters),
    async execute(id, args, signal, update, context) {
      // Native custom entries are audit-only: they do not enter model context.
      await session.appendCustomEntry("tool_argument_validation", {
        tool_call_id: id, tool: tool.name, validated_args: redactSecrets(args),
      });
      const worker: ResearchWorker = {
        async callSuccess(domainTool, payload, options, callSignal) {
          await session.appendCustomEntry("tool_argument_dispatch", {
            tool_call_id: id, tool: domainTool, normalized_args: redactSecrets(payload),
          });
          return context.worker.callSuccess(domainTool, payload, options, callSignal);
        },
      };
      return tool.execute(id, args, signal, update, { ...context, worker });
    },
  };
}
