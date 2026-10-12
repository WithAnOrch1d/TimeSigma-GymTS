from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "AgentMemoryMaterializationReport": ".agent_memory",
    "AgentMemoryValidationReport": ".agent_memory",
    "ComponentEvidence": ".canonical",
    "ComponentLandscape": ".landscape",
    "EvidenceCollection": ".collection",
    "PipelineDefinitionExperimentPlan": ".experiment",
    "PipelineDefinitionExperimentRun": ".experiment",
    "ExperimentPlan": ".experiment",
    "ExperimentRun": ".experiment",
    "NextAction": ".pipeline",
    "NormalizedPipeline": ".pipeline",
    "PipelineRecommendation": ".experiment",
    "PipelineAction": ".pipeline_action",
    "PipelineDefinitionDiff": ".pipeline_action",
    "PipelineComponent": ".pipeline_space",
    "PipelineDefinition": ".pipeline_space",
    "PipelineSpec": ".pipeline",
    "PreprocessingSpec": ".preprocessing",
    "PipelineStage": ".pipeline_space",
    "PlanningUnit": ".pipeline",
    "ProviderContextInputs": ".provider_context",
    "ProviderRuntimeMode": ".provider_context",
    "RecommendationRun": ".experiment",
    "ResearchDecision": ".decision",
    "ResearchRuntimeSnapshot": ".state",
    "ResearchState": ".state",
    "ResearchStepEvidence": ".canonical",
    "StopAction": ".decision",
    "TaskContext": ".pipeline",
    "agent_memory_manifest_payload": ".canonical",
    "apply_pipeline_action": ".pipeline_action",
    "apply_preprocessing_components": ".pipeline_transforms",
    "apply_score_postprocessing": ".pipeline_transforms",
    "build_component_landscape": ".landscape",
    "build_provider_context": ".provider_context",
    "build_research_state": ".state",
    "canonical_pipeline_key": ".collection",
    "canonical_pipeline_key_from_parts": ".collection",
    "component_evidence_id": ".canonical",
    "evidence_directory": ".collection",
    "fuse_scores_mean": ".pipeline_transforms",
    "load_evidence_collection": ".collection",
    "materialize_dataset_view": ".pipeline_transforms",
    "materialize_agent_memory": ".agent_memory",
    "normalize_pipeline": ".pipeline",
    "pipeline_identity": ".collection",
    "plan_pipeline": ".planning",
    "planning_unit_identity": ".canonical",
    "pipeline_component_registry": ".pipeline_space",
    "pipeline_definition_diff": ".pipeline_action",
    "pipeline_space_for_task": ".pipeline_space",
    "research_step_evidence_id": ".canonical",
    "retrieve_component_history": ".retrieval",
    "run_pipeline_experiment": ".experiment",
    "run_pipeline_definition_experiment": ".experiment",
    "run_recommendation_evaluation": ".experiment",
    "validate_agent_memory_store": ".agent_memory",
    "validate_pipeline_action_change": ".pipeline_action",
    "validate_pipeline_definition": ".pipeline_space",
    "zero_shot_membership_payload": ".canonical",
}


__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
