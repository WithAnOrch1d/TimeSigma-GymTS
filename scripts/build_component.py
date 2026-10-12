"""One canonical task: dry-run, native execution, replay, candidate and explicit publish.

This uses existing Evidence/Artifact formats. It is not an algorithm onboarding
approval or a batch scheduler. Keep failed output directories for diagnosis.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep cold imports usable on CPU-quota containers; workers bind their own limits.
for _threads in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_threads, '1')

from timesigma.algorithms import EASYTSAD_REVISION, get_method, load_model_artifact
from timesigma.benchmark.manifest import dataset_provenance
from timesigma.benchmark.runtime import run_method, seed_runtime
from timesigma.benchmark.zero_shot import load_zero_shot_membership
from timesigma.research.agent_memory import _prepare_component_evidence, validate_evidence_artifact
from timesigma.research.canonical import ComponentEvidence, component_evidence_id
from timesigma.research.collection import (EvidenceCollection, evidence_directory,
    plain_json, read_json_mapping as read, validate_component_tasks,
    write_json_mapping_if_same as write)
from timesigma.research.pipeline import PipelineSpec
from timesigma.research.pipeline_execution import file_digest
from timesigma.research.result_parser import parse_validated_component_results


def source_binding():
    paths = [ROOT / 'pyproject.toml', ROOT / 'uv.lock', Path(__file__).resolve()]
    paths += sorted((ROOT / 'timesigma').rglob('*.py'))
    paths += sorted((ROOT / 'timesigma').rglob('*.toml'))
    return {str(p.relative_to(ROOT)): file_digest(p) for p in paths}


def prepare(memory: Path, data: Path, reference: str, method: str) -> dict:
    memory, data = memory.resolve(), data.resolve()
    manifest = read(memory / 'manifest.json')
    collection = EvidenceCollection.from_directory(memory)
    anchor = collection.by_id(reference)
    if not isinstance(anchor, ComponentEvidence):
        raise ValueError('Reference must be an existing canonical ComponentEvidence')
    task = plain_json(anchor.task)
    dataset, schema = task['dataset'], task['schema']
    if dataset.casefold() == 'aiops':
        raise ValueError('AIOPS cannot produce Component Memory')
    validate_component_tasks(memory, (anchor,))
    spec = get_method(method)
    if not spec.capability(schema).supported or not spec.supports_dataset(dataset):
        raise ValueError('Unsupported Method/Schema/dataset; never adapt members or split')
    if spec.artifact_type == 'tensorflow_checkpoint':
        raise ValueError('External TF worker is outside this Core Component entry')
    provenance = dataset_provenance(data)
    source = manifest.get('dataset_sources', {}).get(dataset, manifest.get('dataset_source'))
    if (manifest['easytsad_revision'] != EASYTSAD_REVISION or provenance['clean'] is not True
            or provenance['revision'] != manifest['dataset_revisions'][dataset]
            or provenance['source'] != source):
        raise ValueError('Dataset/source revision must match clean canonical Memory')
    identity = component_evidence_id(method=method, task=task)
    pipeline = PipelineSpec.official_default(method, dataset).normalized().canonical_payload()
    previous = [e for e in collection.evidence if e.evidence_id == identity]
    if previous:
        old, = previous
        if plain_json(old.task) != task or plain_json(old.pipeline) != pipeline:
            raise ValueError('Existing identity has different task/configuration; refuse overwrite')
        validate_evidence_artifact(memory, old, manifest=manifest)
        return {'status': 'SKIP_EXISTING', 'evidence_id': identity}
    members = list(task['unit']['training_curves'])
    if schema == 'zero_shot':
        split = next(s for s in load_zero_shot_membership(memory / manifest['zero_shot_membership']) if s.dataset == dataset)
        members = list(split.enumerated_curves)
    if schema != 'naive' and set(members) != {p.name for p in (data / 'UTS' / dataset).iterdir() if p.is_dir()}:
        raise ValueError('Shared task requires complete canonical dataset members')
    files = {str(p.relative_to(data)): file_digest(p) for curve in members
             for p in (data / 'UTS' / dataset / curve).iterdir() if p.is_file()}
    return {'status': 'TRAIN_REQUIRED', 'method': method, 'task': task, 'evidence_id': identity,
            'memory': str(memory), 'data': str(data), 'reference': reference,
            'pipeline': pipeline, 'seed': manifest['seed'], 'provenance': provenance,
            'members': members, 'data_files': files, 'source_files': source_binding(),
            'manifest_sha256': file_digest(memory / 'manifest.json'),
            'membership_file': manifest['zero_shot_membership'],
            'membership_sha256': file_digest(memory / manifest['zero_shot_membership']),
            'reference_sha256': file_digest(evidence_directory(memory, anchor) / 'evidence.json')}


def verify_inputs(plan):
    memory, data = Path(plan['memory']), Path(plan['data'])
    current = prepare(memory, data, plan['reference'], plan['method'])
    # An independently completed target must not be overwritten or retrained.
    if any(current.get(key) != plan[key] for key in ('status', 'task', 'pipeline', 'seed',
            'members', 'data_files', 'source_files', 'manifest_sha256', 'membership_sha256', 'reference_sha256')):
        raise ValueError('Frozen inputs/source changed or target appeared; use a reviewed new attempt')


def native_train(plan, output):
    task = plan['task']; schema = task['schema']; dataset = task['dataset']
    run_method(method_name=plan['method'], schema=schema, datasets=(dataset,),
        dataset_root=Path(plan['data']), dataset_source_root=Path(plan['data']),
        output_dir=output / 'job', seed=plan['seed'], model_artifacts=True,
        evaluate=False, label_free=True,
        curve_names=(task['unit']['curve'],) if schema == 'naive' else None,
        ordered_members={dataset: task['unit']['training_curves']} if schema == 'all_in_one' else None,
        zero_shot_membership=Path(plan['memory']) / plan['membership_file'] if schema == 'zero_shot' else None)


def replay(plan, output):
    import numpy as np
    from EasyTSAD.Controller import TSADController
    from EasyTSAD.DataFactory.LoadData import load_data
    from timesigma.algorithms.imports import import_method
    from timesigma.benchmark.label_free import label_free_loading
    from timesigma.benchmark.runner import _working_directory, bind_ordered_members
    task = plan['task']; schema = task['schema']; dataset = task['dataset']
    spec = get_method(plan['method'])
    configuration = plan['pipeline']['resolved_configuration']
    job = output / 'job'
    metadata, = job.glob('Results/ModelArtifacts/**/metadata.json')
    directory = output / 'reload'; directory.mkdir()
    with _working_directory(directory), label_free_loading(), bind_ordered_members(
            {dataset: plan['members']} if schema != 'naive' else None):
        c = TSADController(log_level='error')
        c.set_dataset(datasets=dataset, dirname=plan['data'], dataset_type='UTS',
            specify_curves=schema == 'naive', curve_names=[task['unit']['curve']] if schema == 'naive' else None)
        params = configuration['Data_Params']
        arrays = load_data(c.dc, params['preprocess'], params['diff_order'])[dataset]
        cls = getattr(import_method(spec.module), spec.method_name)
        with ExitStack() as stack:
            for phase in ('train_valid_phase', 'train_valid_phase_all_in_one'):
                stack.enter_context(patch.object(cls, phase, side_effect=AssertionError('Replay cannot fit')))
            seed_runtime(plan['seed'])
            model = cls(deepcopy(configuration['Model_Params']))
            loaded = load_model_artifact(model, spec, metadata, schema=schema, configuration=configuration)
            if not loaded and spec.artifact_type != 'replay_recipe':
                raise ValueError('Model Artifact was not restored')
            members = task['unit']['held_out_curves'] if schema == 'zero_shot' else task['unit']['training_curves']
            bindings = {}
            for curve in members:
                model.test_phase(arrays[curve])
                actual = model.anomaly_score()
                path = job / f'Results/Scores/{spec.method_name}/{schema}/{dataset}/{curve}.npy'
                expected = np.load(path, allow_pickle=False)
                if (actual.shape != expected.shape or actual.dtype != expected.dtype
                        or not np.isfinite(actual).all() or not np.array_equal(actual, expected)):
                    raise ValueError(f'Current-runtime independent replay differs: {curve}')
                bindings[str(path.relative_to(job))] = file_digest(path)
    write(output / 'replay.json', {'fresh_process': True, 'pid': os.getpid(),
        'bitwise_equal': True, 'training_calls': 0, 'scores': bindings})


def evaluate(plan, output):
    from EasyTSAD.Controller import TSADController
    from timesigma.benchmark.runner import _working_directory, official_evaluations
    task = plan['task']; schema = task['schema']; dataset = task['dataset']
    members = task['unit']['held_out_curves'] if schema == 'zero_shot' else task['unit']['training_curves']
    with _working_directory(output / 'job'):
        c = TSADController(log_level='error')
        c.set_dataset(datasets=dataset, dirname=plan['data'], dataset_type='UTS', specify_curves=True, curve_names=members)
        c.set_evals(official_evaluations())
        c.do_evals(method=plan['method'], training_schema=schema)


def materialize(plan, output):
    task = plan['task']; memory = Path(plan['memory']); job = output / 'job'
    parsed = parse_validated_component_results(job, Path(plan['data']) / 'UTS' / task['dataset'],
        plan['provenance'], profile_dir=memory.parent / 'profiles' / task['dataset'] / plan['provenance']['revision'])
    if len(parsed.units) != 1:
        raise ValueError('Expected exactly one canonical planning unit')
    prepared = _prepare_component_evidence(parsed, parsed.units[0], job)
    evidence = prepared.evidence
    if (evidence.evidence_id != plan['evidence_id'] or plain_json(evidence.task) != task
            or plain_json(evidence.pipeline) != plan['pipeline']):
        raise ValueError('Parsed task/configuration differs from frozen plan')
    candidate = output / 'candidate'; candidate.mkdir()
    if prepared.artifact.source_path is not None:
        shutil.copy2(prepared.artifact.source_path, candidate / prepared.artifact.artifact_name)
    if prepared.artifact.metadata is not None:
        write(candidate / 'artifact.json', prepared.artifact.metadata)
    write(candidate / 'evidence.json', evidence.as_dict())
    # Preserve scientific outputs and replay lineage; no epoch/temp log copies.
    for name in ('Scores', 'Evals'):
        shutil.copytree(job / 'Results' / name, candidate / 'execution' / 'Results' / name)
    for name in ('run.json', 'configuration.toml', 'score-support.json'):
        if (job / name).is_file():
            shutil.copy2(job / name, candidate / 'execution' / name)
    shutil.copy2(output / 'replay.json', candidate / 'execution/replay.json')
    shutil.copy2(output / 'plan.json', candidate / 'execution/plan.json')
    validate_component_tasks(memory, (evidence,))
    validate_evidence_artifact(memory, evidence, artifact_directory=candidate)
    write(output / 'candidate-binding.json', {str(p.relative_to(candidate)): file_digest(p)
          for p in candidate.rglob('*') if p.is_file()})
    return evidence


def publish(output):
    plan = read(output / 'plan.json'); memory = Path(plan['memory'])
    if not get_method(plan['method']).agent_available:
        raise ValueError('Unqualified Method: review onboarding before Catalog admission/publication')
    if read(output / 'status.json')['status'] not in ('ADMISSION_READY', 'PUBLISHED'):
        raise ValueError('Only a validated candidate can be published')
    candidate = output / 'candidate'; binding = read(output / 'candidate-binding.json')
    if binding != {str(p.relative_to(candidate)): file_digest(p) for p in candidate.rglob('*') if p.is_file()}:
        raise ValueError('Candidate files changed after validation')
    if read(candidate / 'execution/plan.json') != plan:
        raise ValueError('Candidate lineage differs from plan')
    evidence = ComponentEvidence.from_dict(read(candidate / 'evidence.json'))
    if evidence.evidence_id != plan['evidence_id'] or plain_json(evidence.task) != plan['task']:
        raise ValueError('Candidate identity differs from plan')
    destination = evidence_directory(memory, evidence)
    with (memory.parent / '.component-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if file_digest(memory / 'manifest.json') != plan['manifest_sha256'] or file_digest(memory / plan['membership_file']) != plan['membership_sha256']:
            raise ValueError('Canonical Memory protocol changed')
        validate_component_tasks(memory, (evidence,))
        validate_evidence_artifact(memory, evidence, artifact_directory=candidate)
        if destination.exists():
            if not all((destination / name).is_file() and file_digest(destination / name) == value for name, value in binding.items()):
                raise FileExistsError('Target exists; refuse to overwrite existing Evidence/Artifact')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = memory.parent / ('.component-staging-' + uuid.uuid4().hex)
            shutil.copytree(candidate, staging)
            if any(file_digest(staging / name) != value for name, value in binding.items()):
                raise ValueError('Publication copy differs; staging retained')
            os.rename(staging, destination)
    (output / 'status.json').write_text(json.dumps({'status': 'PUBLISHED', 'evidence_id': evidence.evidence_id}) + '\n')
    return {'status': 'PUBLISHED', 'evidence_id': evidence.evidence_id}


def execute(plan, output, *, seconds=600, memory_mb=4096, cpu_cores=1, device='cpu'):
    from timesigma.delivery.resources import bounded_process
    output = output.resolve(); output.mkdir(parents=True, exist_ok=False)
    plan = {**plan, 'limits': {'seconds_per_stage': seconds, 'memory_mb': memory_mb, 'cpu_cores': cpu_cores, 'device': device}}
    write(output / 'plan.json', plan)
    try:
        for stage in ('train', 'replay', 'evaluate'):
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', stage, '--output', str(output)]
            spec = get_method(plan['method'])
            if spec.shared_runtime:
                command[0] = str(ROOT / '.runtimes' / spec.shared_runtime / 'bin/python')
            result = bounded_process(command, output / f'{stage}.log', seconds=seconds,
                memory_mb=memory_mb, cpu_percent=cpu_cores * 100, cuda=device == 'cuda',
                backend='process_group', minimum_free_bytes=10 * 1024**3)
            write(output / f'{stage}-result.json', result)
            if result['returncode'] != 0 or result.get('limit_reason'):
                raise RuntimeError(f'{stage} failed; inspect {output / (stage + ".log")}; no automatic retry')
        materialize(plan, output)
        write(output / 'status.json', {'status': 'ADMISSION_READY', 'evidence_id': plan['evidence_id']})
    except Exception as error:
        write(output / 'status.json', {'status': 'FAILED', 'error': str(error)})
        raise
    return read(output / 'status.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory', type=Path); p.add_argument('--data', type=Path)
    p.add_argument('--reference'); p.add_argument('--method')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true'); p.add_argument('--publish', action='store_true')
    p.add_argument('--seconds-per-stage', type=int, default=600)
    p.add_argument('--memory-mb', type=int, default=4096); p.add_argument('--cpu-cores', type=int, default=1)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--worker', choices=('train', 'replay', 'evaluate'), help=argparse.SUPPRESS)
    a = p.parse_args(); output = a.output.resolve()
    if a.worker:
        plan = read(output / 'plan.json'); verify_inputs(plan)
        if plan['limits']['device'] == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA unavailable')
        {'train': native_train, 'replay': replay, 'evaluate': evaluate}[a.worker](plan, output)
        return
    if a.publish:
        if a.execute:
            p.error('Publish is a separate explicit step after candidate review')
        print(json.dumps(publish(output), indent=2)); return
    if not all((a.memory, a.data, a.reference, a.method)):
        p.error('Need --memory, --data, --reference and --method')
    if min(a.seconds_per_stage, a.memory_mb, a.cpu_cores) < 1:
        p.error('Resource limits must be positive')
    plan = prepare(a.memory, a.data, a.reference, a.method)
    if a.execute and plan['status'] != 'SKIP_EXISTING':
        plan = execute(plan, output, seconds=a.seconds_per_stage, memory_mb=a.memory_mb,
                       cpu_cores=a.cpu_cores, device=a.device)
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
