"""Prepare/run sequential non-AIOPS Research using the existing Pi CLI."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from timesigma.benchmark.manifest import dataset_provenance
from timesigma.research.collection import write_json_mapping_if_same
from timesigma.research.tools import dispatch_research_tool
from timesigma.research.pipeline_execution import file_digest


def prepare(data, memory, output, tasks_file):
    data, memory, output = (p.resolve() for p in (data, memory, output))
    if output.exists():
        raise FileExistsError('Use a new Research run; do not overwrite previous attempts')
    provenance = dataset_provenance(data)
    manifest = json.loads((memory / 'manifest.json').read_text())
    tasks = json.loads(tasks_file.read_text())['tasks']
    ids = [t['task_id'] for t in tasks]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Need distinct task IDs')
    prepared = []
    for task in tasks:
        name, dataset, schema = task['task_id'], task['dataset'], task['schema']
        if Path(name).name != name or name in ('.', '..') or dataset.casefold() == 'aiops':
            raise ValueError('Unsafe task identity or target dataset in Research')
        revision = manifest['dataset_revisions'][dataset]
        if (provenance['clean'] is not True or provenance['revision'] != revision
                or provenance['source'] != 'https://github.com/CSTCloudOps/datasets.git'):
            raise ValueError('Dataset must match clean canonical Memory source')
        for key in ('max_experiments', 'max_wall_seconds', 'max_provider_calls', 'max_provider_tokens'):
            if type(task[key]) is not int or task[key] < 1:
                raise ValueError('Freeze positive task budgets before preparation')
        scope = {'dataset': dataset, 'schema': schema, 'target_metric': 'point-wise-f1',
                 'planning_unit': task['planning_unit'], 'dataset_root': str(data),
                 'dataset_source_root': str(data), 'dataset_dir': str(data / 'UTS' / dataset),
                 'dataset_source': provenance['source'], 'dataset_revision': revision,
                 'profile_dir': str(memory.parent / 'profiles' / dataset / revision),
                 'zero_shot_membership': str(memory / 'zero-shot-membership.json') if schema == 'zero_shot' else None,
                 'agent_memory_root': str(memory), 'memory_mode': 'on'}
        directory = output / name
        path = directory / 'scope.json'
        # Read-only real context validation, including complete membership/defaults.
        dispatch_research_tool('research.context', {**scope, 'experiment_root': str(directory / 'run/experiments'),
            'experiment_index': 1, 'max_experiments': task['max_experiments'], 'max_wall_seconds': task['max_wall_seconds']})
        write_json_mapping_if_same(path, scope)
        argv = ['node', f'--env-file-if-exists={ROOT / "pi-host/.env"}', str(ROOT / 'pi-host/dist/research-cli.js'),
                '--scope', str(path), '--target-metric', 'point-wise-f1',
                '--provider-log', str(directory / 'run/provider.jsonl'), '--session-dir', str(directory / 'run/sessions'),
                '--repository-root', str(ROOT), '--python', sys.executable]
        for key in ('max_experiments', 'max_wall_seconds', 'max_provider_calls', 'max_provider_tokens'):
            argv += ['--' + key.replace('_', '-'), str(task[key])]
        launch = {'argv': argv, 'scope_sha256': file_digest(path)}
        write_json_mapping_if_same(directory / 'launch.json', launch)
        prepared.append({'task_id': name, 'launch_sha256': file_digest(directory / 'launch.json')})
    write_json_mapping_if_same(output / 'campaign.json', {'status': 'PREPARED_NOT_STARTED', 'tasks': prepared})
    return {'tasks': len(prepared), 'status': 'PREPARED_NOT_STARTED'}


def run(output):
    output = output.resolve()
    campaign = json.loads((output / 'campaign.json').read_text())
    with (output / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in campaign['tasks']:
            directory = output / item['task_id']
            path = directory / 'launch.json'
            if file_digest(path) != item['launch_sha256']:
                raise ValueError('Frozen launch changed')
            launch = json.loads(path.read_text())
            if file_digest(directory / 'scope.json') != launch['scope_sha256']:
                raise ValueError('Frozen scope changed')
            if (directory / 'completed.json').exists():
                continue
            if (directory / 'started.json').exists():
                raise RuntimeError(f'Interrupted/failed Research needs budget-preserving recovery: {directory}')
            write_json_mapping_if_same(directory / 'started.json', {'status': 'STARTED'})
            # Existing CLI owns Provider, scientific/wall budgets and Evidence commit.
            with (directory / 'process.log').open('x') as log:
                result = subprocess.run(launch['argv'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write_json_mapping_if_same(directory / 'process-result.json', {'returncode': result.returncode})
            if result.returncode:
                raise RuntimeError(f'Research failed; preserve attempt: {directory}')
            write_json_mapping_if_same(directory / 'completed.json', {'status': 'SESSION_ENDED'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'run'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--memory', type=Path)
    p.add_argument('--tasks', type=Path, default=ROOT / 'experiments/research-tasks.json')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.stage == 'prepare':
        if not a.data or not a.memory:
            p.error('prepare requires --data and --memory')
        print(json.dumps(prepare(a.data, a.memory, a.output, a.tasks.resolve()), indent=2))
    else:
        if not a.execute:
            p.error('run requires --execute: paid calls, training and new ResearchStep commits')
        run(a.output)


if __name__ == '__main__':
    main()
