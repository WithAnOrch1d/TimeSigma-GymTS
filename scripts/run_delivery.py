"""Portable three-schema delivery over the existing TimeSigma lifecycle.

Preparation never calls a Provider or fits a detector. Execution is explicit.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from timesigma.delivery.protocol import prepare_protocol, load_protocol, frozen_memory
from timesigma.delivery.planning import close_planning
from timesigma.delivery.execution import execute_all, seal_scores, leaderboard, verify_closed_plans
from timesigma.research.collection import write_json_mapping_if_same
from timesigma.research.pipeline_execution import file_digest
from scripts.build_leaderboard_demo import write_outputs, write_three_schema_outputs

SCHEMAS = ('naive', 'all_in_one', 'zero_shot')


def read(path):
    return json.loads(Path(path).read_text())


def prepare(output: Path, data: Path, memory: Path, configs: Path) -> dict:
    output, data, memory, configs = (p.resolve() for p in (output, data, memory, configs))
    if output.exists():
        raise FileExistsError('Use a new output directory; never relocate/rewrite a frozen attempt.')
    # Resolve only new configuration paths. Never edit historical receipts.
    prepared = []
    for schema in SCHEMAS:
        config = read(configs / f'aiops-{schema}.json')
        if config['schema'] != schema or config['dataset'] != 'AIOPS':
            raise ValueError('Expected an AIOPS config for each official schema')
        config.update(dataset_root=str(data), dataset_source_root=str(data), memory_root=str(memory))
        config['comparison']['snapshot'] = str((configs / config['comparison']['snapshot']).resolve())
        if prepared:
            config['memory_snapshot_source'] = str(output / SCHEMAS[0] / 'memory-snapshot.json')
        if schema == 'zero_shot':
            membership = configs / 'inputs/aiops-zero-shot-membership.json'
            config.update(zero_shot_membership_source=str(membership),
                          zero_shot_membership_source_sha256=file_digest(membership))
        path = output / 'configs' / f'{schema}.json'
        write_json_mapping_if_same(path, config)
        prepare_protocol(path, output / schema)
        prepared.append(schema)
    record = {'status': 'PREPARED_NOT_STARTED', 'schemas': prepared,
              'protocols': {s: file_digest(output / s / 'protocol.json') for s in SCHEMAS}}
    write_json_mapping_if_same(output / 'campaign.json', record)
    return record


def command(argv):
    subprocess.run(argv, cwd=ROOT, check=True)


def run(output: Path, *, retry_failed: bool = False) -> None:
    output = output.resolve()
    record = read(output / 'campaign.json')
    if record['schemas'] != list(SCHEMAS):
        raise ValueError('All three schemas are required')
    roots = [output / s for s in SCHEMAS]
    for root in roots:
        if file_digest(root / 'protocol.json') != record['protocols'][root.name]:
            raise ValueError('Prepared protocol changed')
        load_protocol(root)
        frozen_memory(root)
    # A nonblocking lock prevents two orchestrators advancing the same run.
    with (output / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for root in roots:
            if not (root / 'planning-closed.json').exists():
                command(['node', f'--env-file-if-exists={ROOT / "pi-host/.env"}',
                         str(ROOT / 'pi-host/dist/delivery-cli.js'), str(root)])
                close_planning(root)
        # Global barrier: no detector runs until ALL schemas' plans validate.
        for root in roots:
            verify_closed_plans(root)
        write_json_mapping_if_same(output / 'all-plans-closed.json', {
            s: file_digest(output / s / 'planning-closed.json') for s in SCHEMAS})
        # TimeSigma first, then the exact frozen missing-public-method supplements.
        for supplements in (False, True):
            for root in roots:
                if (root / 'scores-frozen.json').exists():
                    continue
                protocol = load_protocol(root)
                names = [row['name'] for row in protocol['baselines']] if supplements else [protocol['variant']]
                if names:
                    result = execute_all(root, retry_failed=retry_failed, variant_names=names)
                    if result['status'] != 'scores_ready':
                        raise RuntimeError(f'Execution incomplete: {root}; preserve attempts and repair before retry')
        for root in roots:
            # seal_scores is idempotent and revalidates complete bundles and bytes.
            seal_scores(root)
            protocol = load_protocol(root)
            write_json_mapping_if_same(root / 'execution-audit.json', {
                'status': 'structural_bundle_and_score_binding_pass',
                'numerical_replay_performed': False,
                'zero_shot_provenance': protocol.get('zero_shot_provenance')})
        write_json_mapping_if_same(output / 'all-scores-frozen.json', {
            s: file_digest(output / s / 'scores-frozen.json') for s in SCHEMAS})
        # Global barrier: only now may the official evaluator read labels.
        for root in roots:
            if not (root / 'evaluation-complete.json').exists():
                argv = [sys.executable, '-B', '-m', 'timesigma.delivery.cli', 'evaluate', str(root)]
                if retry_failed:
                    argv.append('--retry-failed')
                command(argv)
            result = leaderboard(root)
            if result['status'] != 'complete':
                raise RuntimeError('No incomplete leaderboard may be published')
            write_outputs(root, Path(load_protocol(root)['comparison']['snapshot']), None)
        write_three_schema_outputs(roots, output / 'summary', output / 'TimeSigma_AIOPS_Leaderboard_3Schema.html')
        write_json_mapping_if_same(output / 'complete.json', {'status': 'COMPLETE'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'run'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--memory', type=Path)
    p.add_argument('--configs', type=Path, default=ROOT / 'experiments')
    p.add_argument('--execute', action='store_true', help='Allow paid planning and detector execution')
    p.add_argument('--retry-failed', action='store_true', help='Only within existing execution/evaluation attempt budgets')
    a = p.parse_args()
    if a.stage == 'prepare':
        if not a.data or not a.memory:
            p.error('prepare requires --data and --memory')
        print(json.dumps(prepare(a.output, a.data, a.memory, a.configs), indent=2))
    else:
        if not a.execute:
            p.error('run requires --execute; preparation alone never starts an experiment')
        run(a.output, retry_failed=a.retry_failed)


if __name__ == '__main__':
    main()
