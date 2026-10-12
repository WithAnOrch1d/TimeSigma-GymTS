"""Qualification descriptor only: importing this file never registers a Catalog entry."""
from dataclasses import dataclass

from timesigma.algorithms.catalog import AlgorithmSpec, SampleSpec, SchemaCapability, BEST_UNLABELED_VALIDATION


@dataclass(frozen=True)
class OmniAnomalySpec(AlgorithmSpec):
    def score_support(self, length, parameters):
        width = parameters["window_length"]
        if type(length) is not int or length < width:
            raise ValueError("OmniAnomaly test length requires at least window_length points")
        return {"start": width - 1, "stop": length, "count": length - width + 1,
                "window_samples": length - width + 1, "native_tail_zeros": 0}


SPEC = OmniAnomalySpec(
    method_name="OmniAnomaly", module="third_party.adapters.omnianomaly", requirements=("numpy",),
    schemas={"naive": SchemaCapability(fit_scope="per_curve", selection_policy=BEST_UNLABELED_VALIDATION,
                                       selection_objective="native_weighted_validation_sgvb"),
             "all_in_one": SchemaCapability(None, unavailable_reason="not implemented"),
             "zero_shot": SchemaCapability(None, unavailable_reason="not implemented")},
    artifact_type="tensorflow_checkpoint", sample_spec=SampleSpec(window_parameter="window_length"),
    agent_available=False, searchable_parameters=(),
)
