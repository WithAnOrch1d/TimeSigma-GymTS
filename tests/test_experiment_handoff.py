"""Handoff boundaries: relocation, global barriers and asset integrity."""
import io
import json
from pathlib import Path
import tarfile
import shutil
import pytest
from scripts import asset_bundle as assets
from scripts import run_delivery as delivery


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_preparation_rebases_without_editing_inputs(tmp_path, monkeypatch):
    templates = tmp_path / 'relocated repo/experiments'
    shutil.copytree(delivery.ROOT / 'experiments', templates)
    before = {p: p.read_bytes() for p in templates.rglob('*') if p.is_file()}
    captured = []
    def fake_prepare(path, root):
        config = json.loads(path.read_text()); captured.append(config)
        write(root / 'protocol.json', config)
        write(root / 'memory-snapshot.json', {'authority': config['memory_root']})
    monkeypatch.setattr(delivery, 'prepare_protocol', fake_prepare)
    output = tmp_path / 'other machine/run'
    data, memory = tmp_path / 'data', tmp_path / 'archive/runs/agent-memory'
    delivery.prepare(output, data, memory, templates)
    assert len(captured) == 3
    for config in captured:
        assert config['dataset_root'] == str(data)
        assert config['memory_root'] == str(memory)
        assert '/root/autodl-tmp/' not in json.dumps(config)
        assert '/home/withanorch1d/' not in json.dumps(config)
        assert Path(config['comparison']['snapshot']).is_file()
    assert captured[1]['memory_snapshot_source'] == captured[2]['memory_snapshot_source']
    assert captured[2]['zero_shot_membership_source_sha256'] == delivery.file_digest(Path(captured[2]['zero_shot_membership_source']))
    assert {p: p.read_bytes() for p in before} == before
    with pytest.raises(FileExistsError):
        delivery.prepare(output, data, memory, templates)


def mock_campaign(tmp_path, monkeypatch):
    calls, protocols = [], {}
    for schema in delivery.SCHEMAS:
        root = tmp_path / schema
        protocol = {'schema': schema, 'variant': 'TimeSigma_First_Demo', 'baselines': [{'name': 'IsolationForest'}], 'comparison': {'snapshot': 'unused'}}
        write(root / 'protocol.json', protocol); protocols[root] = protocol
    write(tmp_path / 'campaign.json', {'schemas': list(delivery.SCHEMAS), 'protocols': {s: delivery.file_digest(tmp_path / s / 'protocol.json') for s in delivery.SCHEMAS}})
    monkeypatch.setattr(delivery, 'load_protocol', lambda root: protocols[root])
    monkeypatch.setattr(delivery, 'frozen_memory', lambda root: None)
    def command(argv):
        stage = 'plan' if argv[0] == 'node' else 'evaluate'
        root = Path(argv[-1]); calls.append((stage, root.name))
        if stage == 'evaluate':
            assert all((tmp_path / s / 'scores-frozen.json').exists() for s in delivery.SCHEMAS)
            write(root / 'evaluation-complete.json', {})
    monkeypatch.setattr(delivery, 'command', command)
    def close(root):
        calls.append(('close', root.name)); write(root / 'planning-closed.json', {})
    monkeypatch.setattr(delivery, 'close_planning', close)
    def verify(root):
        assert (root / 'planning-closed.json').is_file()
    monkeypatch.setattr(delivery, 'verify_closed_plans', verify)
    def execute(root, **kwargs):
        assert all((tmp_path / s / 'planning-closed.json').exists() for s in delivery.SCHEMAS)
        calls.append(('execute', root.name, kwargs['variant_names']))
        return {'status': 'scores_ready'}
    monkeypatch.setattr(delivery, 'execute_all', execute)
    def seal(root):
        calls.append(('seal', root.name)); write(root / 'scores-frozen.json', {})
    monkeypatch.setattr(delivery, 'seal_scores', seal)
    monkeypatch.setattr(delivery, 'leaderboard', lambda root: {'status': 'complete'})
    monkeypatch.setattr(delivery, 'write_outputs', lambda *args: None)
    monkeypatch.setattr(delivery, 'write_three_schema_outputs', lambda *args: calls.append(('html',)))
    return calls


def test_all_plans_then_all_scores_barriers_and_completed_resume(tmp_path, monkeypatch):
    calls = mock_campaign(tmp_path, monkeypatch); delivery.run(tmp_path)
    first_execute = next(i for i, row in enumerate(calls) if row[0] == 'execute')
    first_evaluate = next(i for i, row in enumerate(calls) if row[0] == 'evaluate')
    assert sum(row[0] == 'close' for row in calls[:first_execute]) == 3
    assert sum(row[0] == 'seal' for row in calls[:first_evaluate]) == 3
    computes = [row for row in calls if row[0] == 'execute']
    assert all(row[2] == ['TimeSigma_First_Demo'] for row in computes[:3])
    assert all(row[2] == ['IsolationForest'] for row in computes[3:])
    calls.clear(); delivery.run(tmp_path)
    assert not any(row[0] in ('plan', 'execute', 'evaluate') for row in calls)


def test_one_unclosed_schema_blocks_every_detector(tmp_path, monkeypatch):
    calls = mock_campaign(tmp_path, monkeypatch); old = delivery.close_planning
    def fail(root):
        if root.name == 'zero_shot': raise ValueError('invalid final plan')
        old(root)
    monkeypatch.setattr(delivery, 'close_planning', fail)
    with pytest.raises(ValueError, match='invalid final plan'): delivery.run(tmp_path)
    assert not any(row[0] in ('execute', 'evaluate', 'html') for row in calls)


def test_failed_score_blocks_all_evaluation(tmp_path, monkeypatch):
    calls = mock_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(delivery, 'execute_all', lambda *a, **k: {'status': 'execution_failed'})
    with pytest.raises(RuntimeError, match='Execution incomplete'): delivery.run(tmp_path)
    assert not any(row[0] in ('evaluate', 'html') for row in calls)


def test_changed_protocol_rejected_before_paid_calls(tmp_path, monkeypatch):
    calls = mock_campaign(tmp_path, monkeypatch)
    write(tmp_path / 'naive/protocol.json', {'changed': True})
    with pytest.raises(ValueError, match='protocol changed'): delivery.run(tmp_path)
    assert not calls


def test_asset_roundtrip_relocated_hardlinks_and_corruption(tmp_path):
    source = tmp_path / 'server'; path = source / 'runs/agent-memory/evidence.json'
    write(path, {'artifact_path': 'model.bin', 'provenance': '/old/server/location'})
    model = path.parent / 'model.bin'; model.write_bytes(b'weights')
    (path.parent / 'same.bin').hardlink_to(model)
    manifest = tmp_path / 'manifest.json'
    assets.inventory(source, manifest, ('runs/agent-memory',))
    archive = tmp_path / 'bundle.tar.gz'; assets.pack(source, manifest, archive)
    destination = tmp_path / 'different machine'
    assert assets.unpack(archive, destination)['status'] == 'PASS'
    assert (destination / 'runs/agent-memory/evidence.json').read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError): assets.unpack(archive, destination)
    (destination / 'runs/agent-memory/model.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Asset differs'): assets.verify(destination, destination / 'asset-manifest.json')


def test_export_refuses_configured_credential_and_escape(tmp_path):
    write(tmp_path / 'runs/data.json', {'accidental': 'private-value-12345678'})
    (tmp_path / 'pi-host').mkdir()
    (tmp_path / 'pi-host/.env').write_text('TIMESIGMA_DEEPSEEK_API_KEY=private-value-12345678\n')
    with pytest.raises(ValueError, match='credential'): assets.inventory(tmp_path, tmp_path / 'manifest.json', ('runs',))
    archive = tmp_path / 'malicious.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        info = tarfile.TarInfo('../escape'); info.size = 1; stream.addfile(info, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unsafe'): assets.unpack(archive, tmp_path / 'extract')
    assert not (tmp_path / 'escape').exists()


def test_real_shipped_memory_survives_archive_relocation(tmp_path):
    from timesigma.research.agent_memory import validate_agent_memory_store
    original = delivery.ROOT / 'examples/handoff/sample'
    manifest, archive = tmp_path / 'manifest.json', tmp_path / 'sample.tar.gz'
    assets.inventory(original, manifest, ('agent-memory', 'profiles'))
    assets.pack(original, manifest, archive)
    restored = tmp_path / 'new computer'
    assert assets.unpack(archive, restored)['status'] == 'PASS'
    report = validate_agent_memory_store(restored / 'agent-memory', load_artifacts=True)
    assert report.component_evidence == 4
    assert report.research_step_evidence == 2
    assert report.torch_artifacts_loaded == 4
    assert report.fitted_estimators_loaded == 2
    assert report.missing_artifacts == report.absolute_path_violations == report.aiops_evidence == 0
