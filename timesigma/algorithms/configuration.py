from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib.machinery import PathFinder
from importlib.util import find_spec
from pathlib import Path

import toml

_DATA_DEFAULTS = {"preprocess": "raw", "diff_order": 0}


def default_configuration_path(module: str) -> Path:
    parts = module.split(".")
    # The external adapter lives in the editable TimeSigma package, which can
    # be supplied by a meta-path finder when a CLI starts outside the repo root.
    spec = find_spec(parts[0])
    for index in range(1, len(parts)):
        if spec is None or spec.submodule_search_locations is None:
            break
        spec = PathFinder.find_spec(
            ".".join(parts[: index + 1]),
            spec.submodule_search_locations,
        )
    if spec is None or spec.origin is None:
        raise RuntimeError(f"EasyTSAD Method module has no source path: {module}")
    path = Path(spec.origin).resolve().with_name("config.toml")
    if not path.is_file():
        raise FileNotFoundError(f"EasyTSAD Method config is missing: {path}")
    return path


def resolve_configuration(
    module: str,
    datasets: Sequence[str],
    overrides: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    source = toml.load(default_configuration_path(module))
    override = {} if overrides is None else overrides
    source_data = _section(source, "Data_Params")
    source_models = _section(source, "Model_Params")
    override_data = _section(override, "Data_Params")
    override_models = _section(override, "Model_Params")

    data_params = _deep_merge(_DATA_DEFAULTS, source_data)
    data_params = _deep_merge(data_params, override_data)
    source_default = _section(source_models, "Default")
    override_default = _section(override_models, "Default")

    resolved: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        model_params = _deep_merge(
            source_default,
            _section(source_models, dataset),
        )
        model_params = _deep_merge(model_params, override_default)
        model_params = _deep_merge(
            model_params,
            _section(override_models, dataset),
        )
        resolved[dataset] = {
            "Data_Params": deepcopy(data_params),
            "Model_Params": model_params,
        }
        if module == "timesigma.algorithms.isolation_forest":
            from .isolation_forest.binding import validate_data_configuration

            import json

            pipeline = override.get("Pipeline", {})
            operators = json.loads(pipeline.get("operators_json", "[]"))
            validate_data_configuration(resolved[dataset], operators)
    return resolved


def freeze_configuration(
    path: Path,
    resolved: Mapping[str, Mapping[str, object]],
    *,
    pipeline: Mapping[str, object] | None = None,
) -> Path:
    if not resolved:
        raise ValueError("at least one resolved dataset is required")

    model_params: dict[str, object] = {}
    data_params: Mapping[str, object] | None = None
    for dataset, configuration in resolved.items():
        current_data = _section(configuration, "Data_Params")
        if data_params is None:
            data_params = current_data
        elif current_data != data_params:
            raise ValueError("EasyTSAD requires shared Data_Params across datasets")
        model_params[dataset] = deepcopy(
            _section(configuration, "Model_Params")
        )

    first_dataset = next(iter(resolved))
    payload = {
        "Data_Params": deepcopy(data_params),
        "Model_Params": {
            "Default": deepcopy(model_params[first_dataset]),
            **model_params,
        },
    }
    if pipeline is not None:
        payload["Pipeline"] = deepcopy(dict(pipeline))
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(toml.dumps(payload), encoding="utf-8")
    return destination


def validate_parameter_type(current: object, value: object, dimension: str) -> None:
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(f"{dimension} action must preserve boolean parameter type")
        return
    if isinstance(value, bool):
        raise TypeError(f"{dimension} action must not coerce a non-boolean parameter")
    if isinstance(current, int):
        if not isinstance(value, int):
            raise TypeError(f"{dimension} action must preserve integer parameter type")
    elif isinstance(current, float):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{dimension} action must preserve numeric parameter type")
    elif isinstance(current, (list, tuple)):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{dimension} action must preserve sequence parameter type")
        for index, item in enumerate(value):
            if current:
                validate_parameter_type(current[min(index, len(current) - 1)], item, dimension)
    elif isinstance(current, str):
        # OCSVM gamma natively accepts either a mode name or a numeric value.
        if dimension == "gamma" and isinstance(value, (int, float)):
            return
        if not isinstance(value, str):
            raise TypeError(f"{dimension} action must preserve string parameter type")
    elif current is None and value is not None:
        raise TypeError(f"{dimension} action must preserve null parameter type")


def _section(
    configuration: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    section = configuration.get(name, {})
    if not isinstance(section, Mapping):
        raise TypeError(f"configuration section must be a mapping: {name}")
    return section


def _deep_merge(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, object]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


__all__ = [
    "default_configuration_path",
    "freeze_configuration",
    "resolve_configuration",
]
