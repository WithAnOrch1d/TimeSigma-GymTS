import type { JsonValue } from "./protocol.js";

export function canonicalizeProviderAction(action: {
  dimension: string;
  parameter?: string;
  value: JsonValue;
}): Record<string, JsonValue> {
  return {
    dimension: action.dimension,
    parameter: action.dimension === "hyperparameter" ? action.parameter as string : null,
    value: action.value,
  };
}

export function actionForProvider(
  action: Record<string, JsonValue>,
): Record<string, JsonValue> {
  const visible = { ...action };
  if (visible.dimension !== "hyperparameter") delete visible.parameter;
  return visible;
}
