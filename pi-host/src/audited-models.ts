import { lazyStream, type Models } from "@earendil-works/pi-ai";
import type { AuditedProviderController } from "./provider-runtime.js";

export function createAuditedModels(
  models: Models,
  controller: Pick<AuditedProviderController, "streamFn">,
): Models {
  const streamSimple: Models["streamSimple"] = (model, context, options) =>
    lazyStream(model, async () => await controller.streamFn(model, context, options));
  const completeSimple: Models["completeSimple"] = async (model, context, options) =>
    await streamSimple(model, context, options).result();

  return new Proxy(models, {
    get(target, property) {
      if (property === "streamSimple") return streamSimple;
      if (property === "completeSimple") return completeSimple;
      const value: unknown = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}
