from copy import deepcopy
import json
from pathlib import Path

import pytest

from timesigma.algorithms import METHOD_CATALOG
from timesigma.research import NextAction, NormalizedPipeline, PipelineSpec, PlanningUnit, PreprocessingSpec, TaskContext
from timesigma.research.pipeline import candidate_pipeline_from_action
from timesigma.research.pipeline_space import PipelineDefinition
from timesigma.research.planning import plan_pipeline
from timesigma.research.preprocessing import preprocessing_capability, scaling_scope, validate_search_preprocessing
from timesigma.research.provider_context import lookup_capabilities


def task(schema="naive"):
    return TaskContext("TODS", schema, "point-wise-f1", PlanningUnit.for_schema(
        schema, curve="0" if schema == "naive" else None, training_curves=("0",),
        held_out_curves=("1",) if schema == "zero_shot" else (),
    ))


def proposal(method="raw", schema="naive"):
    return {
        "preprocessing": {
            "scaling": {"method": method, "fit_scope": scaling_scope(method, schema)},
            "temporal_transform": {"method": "difference", "order": 1},
            "smoothing": {"method": "moving_average", "window": 4},
        },
        "detector": {"method": "AR", "parameters": {"p": 4, "lr": 0.002}},
        "ensemble": {"method": "none", "detectors": []},
    }


@pytest.mark.parametrize("schema", ["naive", "all_in_one", "zero_shot"])
@pytest.mark.parametrize("method,legacy", [("raw", "none"), ("min_max", "min-max"), ("z_score", "z-score"), ("robust", "robust")])
def test_category_plan_and_flat_plan_have_identical_execution_identity(schema, method, legacy):
    value = proposal(method, schema)
    canonical = plan_pipeline(value, task(schema))
    old = plan_pipeline({**value, "preprocessing": {"differencing": 1, "moving_average": 4, "scaling": legacy}}, task(schema))
    assert canonical.identity_key() == old.identity_key()
    view = canonical.canonical_view(schema)
    assert view["preprocessing"] == value["preprocessing"]
    assert view["detector"]["parameters"]["p"] == 4
    assert view["detector"]["parameters"]["lr"] == 0.002
    assert NormalizedPipeline.from_canonical_view(view, schema) == old


@pytest.mark.parametrize("category,bad", [
    ("scaling", [{"method": "z_score"}, {"method": "robust"}]),
    ("scaling", {"method": "legacy_chain", "stages": []}),
    ("scaling", {"method": "robust", "fit_scope": "per_curve_train"}),
    ("scaling", {"method": "raw", "fit_scope": "per_curve_train"}),
    ("temporal_transform", {"method": "difference", "order": True}),
    ("temporal_transform", {"method": "difference", "order": 0}),
    ("temporal_transform", {"method": "log"}),
    ("smoothing", {"method": "moving_average", "window": 1}),
    ("smoothing", {"method": "none", "window": 5}),
])
def test_new_category_proposals_reject_ambiguous_or_changed_semantics(category, bad):
    value = proposal("robust", "zero_shot")
    value["preprocessing"][category] = bad
    with pytest.raises((TypeError, ValueError)):
        plan_pipeline(value, task("zero_shot"))


def test_legacy_chain_and_order_are_explicit_and_cannot_be_reinterpreted():
    native = PipelineSpec.official_default("AE", "TODS")
    operators = [{"component_id": "moving_average", "parameters": {"window": 3}}, {"component_id": "robust_scaler", "parameters": {}}]
    old = NormalizedPipeline(native.method, native.resolved_configuration, operators)
    view = old.canonical_view("zero_shot")
    spec = view["preprocessing"]
    assert spec["scaling"] == {"method": "legacy_chain", "stages": [
        {"method": "min_max", "fit_scope": "per_curve_train"},
        {"method": "robust", "fit_scope": "source_train_members"},
    ]}
    assert old.preprocessing_spec("zero_shot").execution_order()[-2:] == ["moving_average", "robust_scaler"]
    assert NormalizedPipeline.from_canonical_view(view, "zero_shot").identity_key() == old.identity_key()
    with pytest.raises(ValueError, match="replay-only"):
        PreprocessingSpec.from_dict(spec, "zero_shot")
    false_scope = deepcopy(spec)
    false_scope["scaling"]["stages"][1]["fit_scope"] = "per_curve_train"
    with pytest.raises(ValueError, match="disagrees"):
        PreprocessingSpec.from_dict(false_scope, "zero_shot", allow_legacy=True)
    candidate = candidate_pipeline_from_action(old, NextAction("hyperparameter", 0.002, "lr"), task())
    validate_search_preprocessing(old, candidate, "naive")


def test_category_names_do_not_merge_two_controlled_research_changes():
    baseline = plan_pipeline(proposal("min_max"), task())
    candidate = proposal("robust")
    with pytest.raises(ValueError, match="exactly one"):
        candidate_pipeline_from_action(baseline, NextAction("pipeline", candidate), task())


def test_single_scaler_outer_order_is_preserved_by_existing_action_api():
    baseline = plan_pipeline(proposal("robust"), task())
    reverse = list(reversed(baseline.canonical_payload()["operators"]))
    candidate = candidate_pipeline_from_action(baseline, NextAction("preprocessing", reverse), task())
    validate_search_preprocessing(baseline, candidate, "naive")
    assert candidate.preprocessing_spec("naive").execution_order()[-2:] == ["moving_average", "robust_scaler"]
    assert NormalizedPipeline.from_canonical_view(candidate.canonical_view("naive"), "naive") == candidate


def test_materialized_definition_keeps_effective_native_overrides_and_execution_layer():
    definition = PipelineDefinition.from_pipeline_spec(PipelineSpec.official_default("AR", "TODS"))
    raw = definition.as_dict()
    raw["preprocessing"] += [
        {"component_id": "moving_average", "parameters": {"window": 1}},
        {"component_id": "robust_scaler", "parameters": {}},
        {"component_id": "native_preprocess", "parameters": {"value": "z-score"}},
    ]
    definition = PipelineDefinition.from_dict(raw)
    view = definition.canonical_view("zero_shot")
    spec = definition.preprocessing_spec("zero_shot")
    assert spec.execution_order() == ["moving_average", "robust_scaler", "native_split", "difference", "z-score"]
    assert view["preprocessing"]["scaling"]["stages"][0]["fit_scope"] == "per_curve_full_train_before_split"
    assert PipelineDefinition.from_canonical_view(view, "zero_shot").canonical_view("zero_shot") == view
    with pytest.raises(ValueError, match="materialized"):
        NormalizedPipeline.from_canonical_view({**proposal(), "preprocessing": spec.as_dict()}, "zero_shot")


@pytest.mark.parametrize("schema", ["naive", "all_in_one", "zero_shot"])
def test_capability_reports_actual_schema_support_and_scaler_exposure(schema):
    cards = lookup_capabilities(task(schema), kind="all", name="")["items"]
    for method, registered in METHOD_CATALOG.items():
        if not registered.agent_available:
            assert not any(card["name"] == method for card in cards)
            continue
        card = next(card for card in cards if card["name"] == method)
        compatibility = card["preprocessing_compatibility"]
        assert compatibility == preprocessing_capability(method, schema)
        expected_scaling = ["raw"] if method == "IsolationForest" else ["raw", "min_max", "z_score", "robust"]
        assert compatibility["supported_scaling"] == (expected_scaling if registered.capability(schema).supported else [])
        if method == "IsolationForest":
            assert compatibility["native_required_preprocessing"] == "raw, diff_order=0, no components"
            assert compatibility["scaling_fit_scopes"] == {"raw": "not_applicable"}
            assert compatibility["supported_temporal_transform"] == ["none"]
            assert compatibility["supported_smoothing"] == ["none"]
        else:
            assert compatibility["native_required_preprocessing"] is None
            assert compatibility["scaling_fit_scopes"]["z_score"] == "per_curve_train"
            assert compatibility["scaling_fit_scopes"]["robust"] == ("per_curve_train" if schema == "naive" else "source_train_members")
    assert not any(card["name"] == "score_moving_average" for card in cards)


def test_agent_search_rejects_new_chain_without_changing_legacy_execution_or_spending_budget(tmp_path, monkeypatch):
    import shutil
    from timesigma.research.collection import EvidenceCollection
    from timesigma.research import tools

    # Keep the admission assertions, using shipped facts instead of the private
    # full-store helper. Provenance I/O is a fixture here; no detector is run.
    sample = Path(__file__).parents[1] / "examples/handoff/sample"
    shutil.copytree(sample, tmp_path / "memory")
    root = tmp_path / "memory/agent-memory"
    baseline = EvidenceCollection.from_directory(root).by_id("component/AR/naive/TODS/234")
    manifest = json.loads((root / "manifest.json").read_text())
    source, revision = manifest["dataset_sources"]["TODS"], manifest["dataset_revisions"]["TODS"]
    data = tmp_path / "data"
    (data / "UTS/TODS/234").mkdir(parents=True)
    monkeypatch.setattr(tools, "dataset_provenance", lambda _: {
        "source": source, "revision": revision, "clean": True,
    })
    scope = {
        "dataset": "TODS", "schema": "naive", "target_metric": "point-wise-f1",
        "planning_unit": {"curve": "234", "training_curves": ["234"], "held_out_curves": []},
        "dataset_root": str(data), "dataset_source_root": str(data),
        "dataset_dir": str(data / "UTS/TODS"), "dataset_source": source, "dataset_revision": revision,
        "profile_dir": str(tmp_path / "memory/profiles/TODS" / revision),
        "agent_memory_root": str(root), "experiment_root": str(tmp_path / "experiments"),
        "zero_shot_membership": None, "experiment_index": 1,
        "max_experiments": 2, "max_wall_seconds": 120,
    }
    action = {"dimension": "preprocessing", "parameter": None, "value": [{"component_id": "robust_scaler", "parameters": {}}]}
    payload = {**scope, "baseline_evidence_id": baseline.evidence_id, "action": action}
    legacy = tools.dispatch_research_tool("experiment.preview", payload)
    assert legacy["status"] == "ready"
    assert legacy["canonical_preprocessing"]["candidate"]["scaling"]["method"] == "legacy_chain"
    rejected = tools.dispatch_research_tool("experiment.preview", {**payload, "enforce_preprocessing_categories": True})
    assert rejected["status"] == "incompatible"
    assert "one scaling method" in rejected["reason"]
    assert rejected["execution_budget"]["experiments_used"] == 0
    monkeypatch.setattr(tools, "run_pipeline_experiment", lambda *a, **kw: pytest.fail("invalid preprocessing must not execute"))
    run = {
        **payload, "enforce_preprocessing_categories": True, "preview_id": legacy["preview_id"],
        "experiment_id": "category-rejection", "hypothesis": "Check admission only.", "stopping_condition": "No detector execution.",
    }
    with pytest.raises(ValueError, match="one scaling method"):
        tools.dispatch_research_tool("experiment.run", run)
    legal_action = {"dimension": "preprocessing", "parameter": None, "value": [{"component_id": "moving_average", "parameters": {"window": 3}}]}
    legal_preview = tools.dispatch_research_tool("experiment.preview", {**payload, "action": legal_action})
    assert legal_preview["status"] == "ready"
    rejected = tools.dispatch_research_tool("experiment.run", {**run, "action": legal_action, "preview_id": legal_preview["preview_id"]})
    assert rejected["status"] == "rejected"
    assert "Preview does not match" in rejected["reason"]
    assert rejected["execution_budget"]["experiments_used"] == 0


def test_packaged_naive_pipelines_roundtrip_and_validate_bundle_identity(tmp_path):
    import shutil
    from timesigma.algorithms.configuration import freeze_configuration
    from timesigma.research.pipeline_execution import file_digest, validate_pipeline_bundle

    root = Path(__file__).parents[1] / "examples/handoff/sample/agent-memory"
    files = sorted(root.glob("evidence/**/evidence.json"))
    assert len(files) == 6
    for index, file in enumerate(files):
        original = file.read_bytes()
        frozen = json.loads(original)
        pipeline = NormalizedPipeline.from_dict(frozen.get("pipeline", frozen.get("candidate_pipeline")))
        restored = NormalizedPipeline.from_canonical_view(pipeline.canonical_view("naive"), "naive")
        assert restored.identity_key() == pipeline.identity_key()
        # A test-only bundle exercises the same validator using real model bytes.
        # This is not a recreated historical AIOPS run or new scientific Evidence.
        artifact = tmp_path / str(index)
        artifact.mkdir()
        model = frozen["artifact"]["artifact_path"]
        shutil.copyfile(file.parent / model, artifact / model)
        freeze_configuration(
            artifact / "configuration.toml", {"TODS": restored.canonical_payload()["resolved_configuration"]},
            pipeline={"definition": json.dumps(restored.canonical_payload())},
        )
        (artifact / "pipeline-artifact.json").write_text(json.dumps({
            "pipeline": restored.canonical_payload(), "task": frozen["task"], "seed": 1,
            "branches": [{"directory": ".", "method": restored.method}],
            "files": [model, "configuration.toml"], "digests": {model: file_digest(artifact / model)},
        }))
        binding = validate_pipeline_bundle(artifact, restored.canonical_payload())
        assert len(binding["branches"]) == 1
        (artifact / model).write_bytes(b"corrupted test copy")
        with pytest.raises(ValueError, match="integrity mismatch"):
            validate_pipeline_bundle(artifact, restored.canonical_payload())
        assert file.read_bytes() == original


def test_frozen_component_and_research_evidence_views_roundtrip_without_rewriting_memory():
    import hashlib
    from timesigma.research import ComponentEvidence, ResearchStepEvidence

    root = Path(__file__).parents[1] / "examples/handoff"
    snapshot = json.loads((root / "sample-source.json").read_text())
    counts = {"component": 0, "research_step": 0}
    for record in snapshot["files"]:
        if not record["path"].endswith("/evidence.json"):
            continue
        file = root / "sample" / record["path"]
        original = file.read_bytes()
        assert hashlib.sha256(original).hexdigest() == record["sha256"]
        payload = json.loads(original)
        kind = payload["kind"]
        cls = ComponentEvidence if kind == "component" else ResearchStepEvidence
        evidence = cls.from_dict(payload)
        assert cls.from_canonical_view(evidence.canonical_view()).as_dict() == payload
        assert file.read_bytes() == original
        assert payload["task"]["dataset"].casefold() != "aiops"
        counts[kind] += 1
    assert counts == {"component": 4, "research_step": 2}
