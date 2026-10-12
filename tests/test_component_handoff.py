"""Incremental handoff boundaries, using real canonical sample identities."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from scripts import build_component as build


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    memory = tmp_path / 'memory/agent-memory'
    shutil.copytree(build.ROOT / 'examples/handoff/sample', memory.parent)
    data = tmp_path / 'data'; curve = data / 'UTS/TODS/0'; curve.mkdir(parents=True)
    (curve / 'train.npy').write_bytes(b'input'); (curve / 'test.npy').write_bytes(b'test')
    manifest = build.read(memory / 'manifest.json')
    source = manifest.get('dataset_sources', {}).get('TODS', manifest.get('dataset_source'))
    monkeypatch.setattr(build, 'dataset_provenance', lambda _: {'source': source,
        'revision': manifest['dataset_revisions']['TODS'], 'clean': True})
    return memory, data


def plan(inputs, method='IsolationForest'):
    return build.prepare(*inputs, 'component/AR/naive/TODS/0', method)


def test_dry_run_is_read_only_and_binds_canonical_task(inputs):
    memory, data = inputs
    before = {p: p.read_bytes() for p in memory.parent.rglob('*') if p.is_file()}
    result = plan(inputs)
    assert result['status'] == 'TRAIN_REQUIRED'
    assert result['evidence_id'] == 'component/IsolationForest/naive/TODS/0'
    assert result['task']['unit'] == {'curve': '0', 'training_curves': ['0'], 'held_out_curves': []}
    assert before == {p: p.read_bytes() for p in before}
    assert not (memory / 'evidence/components/IsolationForest').exists()


def test_existing_component_skips_and_changed_default_refuses(inputs, monkeypatch):
    assert plan(inputs, 'AR')['status'] == 'SKIP_EXISTING'
    p = inputs[0] / 'evidence/components/AR/naive/TODS/0/evidence.json'
    payload = build.read(p); payload['pipeline']['resolved_configuration']['Model_Params']['p'] = 32
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='different task/configuration'):
        plan(inputs, 'AR')


def test_dirty_data_rejected_before_training(inputs, monkeypatch):
    monkeypatch.setattr(build, 'dataset_provenance', lambda _: {'clean': False})
    with pytest.raises(ValueError, match='Dataset/source'):
        plan(inputs)


def test_changed_numeric_inputs_and_membership_refused(inputs):
    frozen = plan(inputs)
    (inputs[1] / 'UTS/TODS/0/train.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Frozen inputs'):
        build.verify_inputs(frozen)


def test_researchstep_cannot_define_new_component_task(inputs):
    with pytest.raises(ValueError, match='Reference must'):
        build.prepare(*inputs, 'research-step/TODS/naive/234/tod234_exp1_ar_p32', 'IsolationForest')


def ready_sample(inputs, tmp_path):
    memory = inputs[0]; source = memory / 'evidence/components/AR/naive/TODS/0'
    output = tmp_path / 'attempt'; candidate = output / 'candidate'
    shutil.copytree(source, candidate)
    manifest = build.read(memory / 'manifest.json')
    evidence = build.read(candidate / 'evidence.json')
    payload = {'method': 'AR', 'memory': str(memory), 'task': evidence['task'],
               'evidence_id': evidence['evidence_id'], 'manifest_sha256': build.file_digest(memory / 'manifest.json'),
               'membership_file': manifest['zero_shot_membership'],
               'membership_sha256': build.file_digest(memory / manifest['zero_shot_membership'])}
    build.write(output / 'plan.json', payload); build.write(candidate / 'execution/plan.json', payload)
    build.write(output / 'status.json', {'status': 'ADMISSION_READY'})
    build.write(output / 'candidate-binding.json', {str(p.relative_to(candidate)): build.file_digest(p)
               for p in candidate.rglob('*') if p.is_file()})
    return output


def test_candidate_tamper_and_existing_target_never_overwritten(inputs, tmp_path):
    output = ready_sample(inputs, tmp_path)
    # Existing historical target has no matching candidate lineage and is refused.
    original = (inputs[0] / 'evidence/components/AR/naive/TODS/0/evidence.json').read_bytes()
    with pytest.raises(FileExistsError, match='refuse to overwrite'):
        build.publish(output)
    (output / 'candidate/best.pt').write_bytes(b'broken')
    with pytest.raises(ValueError, match='files changed'):
        build.publish(output)
    assert (inputs[0] / 'evidence/components/AR/naive/TODS/0/evidence.json').read_bytes() == original


def test_failed_stage_preserves_attempt_and_stops(inputs, tmp_path, monkeypatch):
    from timesigma.delivery import resources
    calls = []
    def fail(*args, **kwargs):
        calls.append(args[0]); return {'returncode': 1, 'limit_reason': None}
    monkeypatch.setattr(resources, 'bounded_process', fail)
    output = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='train failed'):
        build.execute(plan(inputs), output)
    assert len(calls) == 1
    assert build.read(output / 'status.json')['status'] == 'FAILED'
    assert not (output / 'candidate').exists()
    with pytest.raises(FileExistsError):
        build.execute(plan(inputs), output)


def test_cpu_wrapper_blocks_non_ar_and_unpreviewed_execution():
    subprocess.run(['node', '--input-type=module', '-e', '''
import assert from 'node:assert/strict';
import {cpuWorker} from './examples/cpu_research.mjs';
let ran=0;
const raw={stop:async()=>{},callSuccess:async(tool,p)=>{
 if(tool==='experiment.run'){ran++;return {result:{}};}
 return {result:{status:'ready',preview_id:p.id,candidate_pipeline:{method:p.method,operators:[],ensemble_ref:null}}};
}};
const guarded=cpuWorker(raw);
await assert.rejects(guarded.callSuccess('experiment.run',{preview_id:'unknown'}));
await guarded.callSuccess('experiment.preview',{id:'gpu',method:'FCVAE'});
await assert.rejects(guarded.callSuccess('experiment.run',{preview_id:'gpu'}));
await guarded.callSuccess('experiment.preview',{id:'cpu',method:'AR'});
await guarded.callSuccess('experiment.run',{preview_id:'cpu'});
assert.equal(ran,1);
'''], cwd=build.ROOT, check=True)


def test_publish_atomic_and_idempotent_without_changing_other_evidence(inputs, tmp_path):
    output = ready_sample(inputs, tmp_path)
    destination = inputs[0] / 'evidence/components/AR/naive/TODS/0'
    # Only this temporary fixture target is absent; OCSVM retains the task.
    shutil.rmtree(destination)
    before = {p: p.read_bytes() for p in inputs[0].rglob('*') if p.is_file()}
    assert build.publish(output)['status'] == 'PUBLISHED'
    assert build.publish(output)['status'] == 'PUBLISHED'
    assert before == {p: p.read_bytes() for p in before}
    assert not list(inputs[0].parent.glob('.component-staging-*'))
    build.validate_evidence_artifact(inputs[0], build.ComponentEvidence.from_dict(build.read(destination / 'evidence.json')))


@pytest.mark.parametrize('schema', ['all_in_one', 'zero_shot'])
def test_shared_plan_inherits_exact_canonical_members(inputs, schema):
    memory, data = inputs
    anchor = build.read(memory / 'evidence/components/AR/naive/TODS/0/evidence.json')
    if schema == 'zero_shot':
        manifest = build.read(memory / 'manifest.json')
        split = next(s for s in build.load_zero_shot_membership(memory / manifest['zero_shot_membership']) if s.dataset == 'TODS')
        training, held = list(split.training_curves), list(split.held_out_curves)
        members = list(split.enumerated_curves)
    else:
        training, held, members = ['234', '0'], [], ['234', '0']
    for member in members:
        path = data / 'UTS/TODS' / member; path.mkdir(exist_ok=True)
        for filename in ('train.npy', 'test.npy'):
            (path / filename).write_bytes(b'data')
    anchor['task'] = {'dataset': 'TODS', 'schema': schema, 'unit': {'curve': None,
                    'training_curves': training, 'held_out_curves': held}}
    anchor['evidence_id'] = build.component_evidence_id(method='AR', task=anchor['task'])
    obj = build.ComponentEvidence.from_dict(anchor)
    build.write(build.evidence_directory(memory, obj) / 'evidence.json', anchor)
    result = build.prepare(memory, data, anchor['evidence_id'], 'IsolationForest')
    assert result['members'] == members
    assert result['task'] == anchor['task']
    (data / 'UTS/TODS/unexpected-member').mkdir()
    with pytest.raises(ValueError, match='complete canonical'):
        build.prepare(memory, data, anchor['evidence_id'], 'IsolationForest')
