"""Re-render archived official comparisons without dereferencing old server paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.asset_bundle import digest, safe_path
from scripts.build_leaderboard_demo import merge_leaderboard, render_html


def render(assets, output, manifest_path=None):
    if output.exists():
        raise FileExistsError('Use a new HTML path; do not replace a historical result')
    manifest = json.loads((manifest_path or assets / 'asset-manifest.json').read_text())

    def bound(name):
        path = safe_path(assets, name)
        if digest(path) != manifest['files'][name]['sha256']:
            raise ValueError(f'Historical result changed: {name}')
        return json.loads(path.read_text())

    schemas = {}
    for schema in ('naive', 'all_in_one', 'zero_shot'):
        base = f'runs/aiops-continuation-20260921/{schema}'
        suffix = '' if schema == 'naive' else '-' + schema.replace('_', '-')
        merged = bound(f'{base}/timesigma-public-leaderboard-merged{suffix}.json')
        local = bound(f'{base}/leaderboard.json')
        protocol = bound(f'{base}/protocol.json')
        snapshot = ROOT / f'experiments/inputs/{schema}-public-snapshot.json'
        if digest(snapshot) != protocol['comparison']['snapshot_sha256']:
            raise ValueError('Public snapshot differs from archived protocol')
        checked = merge_leaderboard(json.loads(snapshot.read_text()), local,
            tuple(row['name'] for row in protocol['baselines']), protocol.get('supplemental_not_applicable'))
        if merged['rows'] != checked['rows'] or merged['schema'].lower() != schema:
            raise ValueError('Archived chart rows disagree with official local results/public snapshot')
        # Keep original planning/provider statistics. Never resolve old Memory paths.
        schemas[schema] = merged
    render_html({'schemas': schemas}, ROOT / 'docs/demo/assets', output)
    return {'status': 'RENDERED_FROM_ARCHIVE', 'output': str(output), 'training': False, 'llm_calls': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, help='Optional separately stored export manifest')
    a = p.parse_args()
    print(json.dumps(render(a.assets.resolve(), a.output.resolve(), a.manifest), indent=2))


if __name__ == '__main__':
    main()
