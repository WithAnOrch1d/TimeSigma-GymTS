"""Merge frozen public numbers with official local results and build one offline HTML."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from copy import deepcopy
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path


def merge_leaderboard(snapshot: dict, local: dict, supplemental_baselines: tuple[str, ...] = (), not_applicable: dict[str, str] | None = None) -> dict:
    expected = {'TimeSigma_First_Demo', *supplemental_baselines}
    if local['status'] != 'complete' or len(local['rows']) != len(expected) or {row['method'] for row in local['rows']} != expected:
        raise ValueError('a complete frozen TimeSigma/supplement official evaluation is required')
    own = next(row for row in local['rows'] if row['method'] == 'TimeSigma_First_Demo')
    if own['method'] != 'TimeSigma_First_Demo' or own['coverage'] != own['required_curves'] or own['failures']:
        raise ValueError('incomplete or unexpected local evaluation row')
    if snapshot['dataset'] != local['dataset'] or snapshot['schema'].lower() != local['schema']:
        raise ValueError('snapshot and local task do not match')
    metric_ids = [m['id'] for m in snapshot['metrics']]
    if len(metric_ids) != 8 or set(own['official_metrics']) != set(metric_ids):
        raise ValueError('local evaluation must provide the same eight official metrics')
    rows = deepcopy(snapshot['rows'])
    if len({r['method'] for r in rows}) != len(rows) or any(r['method'] == 'TimeSigma' for r in rows):
        raise ValueError('duplicate public method identity')
    ours = {'method': 'TimeSigma', 'display_name': 'TimeSigma (Ours)', 'source': 'local_timesigma_evaluation',
            'status': 'locally_evaluated', 'metrics': dict(own['official_metrics']), 'display_values': {}}
    rows.append(ours)
    for method in supplemental_baselines:
        row = next(r for r in local['rows'] if r['method'] == method)
        if method in {r['method'] for r in rows} or row['status'] != 'complete' or row['failures'] or row['coverage'] != row['required_curves'] or set(row['official_metrics']) != set(metric_ids):
            raise ValueError('incomplete, duplicate or mismatched supplemental AIOPS row')
        rows.append({'method': method, 'display_name': method + ' (local default)',
                     'source': 'local_baseline_evaluation', 'status': 'locally_evaluated',
                     'metrics': dict(row['official_metrics']), 'display_values': {}})
    for method, reason in (not_applicable or {}).items():
        if method in {row['method'] for row in rows}:
            raise ValueError('N/A cannot replace a public or measured method')
        rows.append({'method': method, 'display_name': method + ' (N/A)',
                     'source': 'local_baseline_unavailable', 'status': 'not_applicable', 'reason': reason,
                     'metrics': {key: None for key in metric_ids}, 'display_values': {key: 'N/A' for key in metric_ids}})
    precision = snapshot['ranking_policy']['precision_decimals']
    if snapshot['ranking_policy']['rounding'] != 'ROUND_HALF_UP':
        raise ValueError('unsupported frozen comparison rounding')
    unit = Decimal(1).scaleb(-precision)
    best_public = {}
    for row in rows:
        row['ranks'] = {}
        row['comparison_values'] = {}
        for key in metric_ids:
            value = row['metrics'][key]
            if row['source'] in {'local_timesigma_evaluation', 'local_baseline_evaluation'} and value is None:
                raise ValueError('all eight local official metrics are required')
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError('metric values must be finite numbers or explicit missing values')
            row['comparison_values'][key] = None if value is None else str(Decimal(str(value)).quantize(unit, rounding=ROUND_HALF_UP))
            if row['source'] in {'local_timesigma_evaluation', 'local_baseline_evaluation'}:
                row['display_values'][key] = row['comparison_values'][key]
    for key in metric_ids:
        ranked = {r['method']: Decimal(r['comparison_values'][key]) for r in rows if r['comparison_values'][key] is not None}
        for row in rows:
            value = ranked.get(row['method'])
            row['ranks'][key] = None if value is None else 1 + sum(other > value for other in ranked.values())
        public = [r for r in rows if r['source'] == 'public_snapshot' and r['metrics'][key] is not None]
        if public:
            best = max(r['metrics'][key] for r in public)
            best_public[key] = {'methods': [r['method'] for r in public if r['metrics'][key] == best], 'value': best,
                                'delta': ours['metrics'][key] - best}
        else:
            best_public[key] = {'methods': [], 'value': None, 'delta': None}
    return {'title': 'TimeSigma Local Comparison Demo', 'comparison_status': 'Published leaderboard snapshot comparison',
            'dataset': snapshot['dataset'], 'schema': snapshot['schema'], 'primary_metric': snapshot['primary_metric'],
            'snapshot_timestamp': snapshot['snapshot_timestamp'], 'source_url': snapshot['source_url'],
            'metrics': snapshot['metrics'], 'ranking_policy': snapshot['ranking_policy'], 'rows': rows,
            'best_published': best_public, 'provider': local['provider'], 'local_evaluation': own,
            'notice': 'Public method values are preserved from the frozen TimeSeriesBench leaderboard snapshot. TimeSigma is locally evaluated and inserted for comparison; this page is not an official TimeSeriesBench submission.'}


def planning_details(root: Path, protocol: dict) -> dict:
    methods, preprocess, branches, tools = Counter(), Counter(), Counter(), Counter()
    citations = set()
    per_curve = []
    ensemble_count = 0
    invalid_plans = 0
    units = protocol.get('planning_units', protocol['curves'])
    per_unit = []
    for curve in units:
        directory = root / 'plans' / curve
        plan = json.loads((directory / 'final.json').read_text())
        pipe = plan['pipeline']
        methods[pipe['method']] += 1
        data = pipe['resolved_configuration']['Data_Params']
        operations = [op['component_id'] for op in pipe.get('operators', [])]
        preprocessing = f"diff={data['diff_order']} · {data['preprocess']}" + (' · ' + ', '.join(operations) if operations else '')
        preprocess[preprocessing] += 1
        detectors = [pipe] + (pipe['ensemble_ref']['detectors'] if pipe.get('ensemble_ref') else [])
        branches.update(d['method'] for d in detectors)
        ensemble_count += bool(pipe.get('ensemble_ref'))
        citations.update(plan['evidence_ids'])
        invalid_plans += len(list((directory / 'invalid-plans').glob('*.json')))
        for log in directory.glob('sessions/**/*.jsonl'):
            for line in log.open():
                record = json.loads(line)
                msg = record.get('message', {})
                if msg.get('role') == 'assistant':
                    tools.update(c['name'] for c in msg['content'] if c['type'] == 'toolCall')
        unit_record = {'curve': curve, 'detector': ' + '.join(d['method'] for d in detectors), 'preprocessing': preprocessing,
                          'branches': len(detectors), 'evidence_ids': plan['evidence_ids'], 'pipeline': pipe}
        per_unit.append(unit_record)
        task_unit = plan['task']['unit']
        members = task_unit['held_out_curves'] if protocol['schema'] == 'zero_shot' else task_unit['training_curves']
        per_curve.extend({**unit_record, 'curve': member} for member in members)
    return {'detector_distribution': dict(methods), 'preprocessing_distribution': dict(preprocess),
            'branch_distribution': dict(branches), 'ensemble_count': ensemble_count, 'tool_counts': dict(tools),
            'unique_cited_evidence': len(citations), 'evidence_ids': sorted(citations), 'invalid_plans': invalid_plans,
            'per_curve': per_curve, 'per_unit': per_unit}


def write_outputs(root: Path, snapshot_path: Path, html_path: Path | None) -> dict:
    protocol = json.loads((root / 'protocol.json').read_text())
    if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != protocol['comparison']['snapshot_sha256']:
        raise ValueError('public snapshot binding mismatch')
    if not (root / 'scores-frozen.json').is_file() or not (root / 'evaluation-complete.json').is_file():
        raise ValueError('score freeze and official evaluation barriers are required')
    snapshot = json.loads(snapshot_path.read_text())
    local = json.loads((root / 'leaderboard.json').read_text())
    supplements = tuple(row['name'] for row in protocol['baselines'])
    merged = merge_leaderboard(snapshot, local, supplements, protocol.get('supplemental_not_applicable'))
    for row in local['rows']:
        if set(row['per_curve']) != set(protocol['zero_shot_split']['held_out_curves'] if protocol['schema'] == 'zero_shot' else protocol['curves']):
            raise ValueError('supplemental evaluation population differs from the frozen protocol')
    merged['supplemental_baselines'] = list(supplements)
    merged['supplemental_not_applicable'] = protocol.get('supplemental_not_applicable', {})
    members = protocol['zero_shot_split']['held_out_curves'] if protocol['schema'] == 'zero_shot' else protocol['curves']
    if set(merged['local_evaluation']['per_curve']) != set(members):
        raise ValueError('every frozen evaluation member must have official metrics')
    merged.update(planning=planning_details(root, protocol), easytsad_revision=protocol['easytsad_revision'],
                  source_revision=protocol['source_revision'], source_binding='Exact frozen backend source copies in the formal run',
                  run_root=root.name, seed=protocol['seed'], model=protocol['provider'],
                  public_snapshot_sha256=protocol['comparison']['snapshot_sha256'],
                  evaluation=local['evaluation'], zero_shot_split=protocol.get('zero_shot_split'),
                  membership_sha256=protocol.get('zero_shot_membership_sha256'), zero_shot_provenance=protocol.get('zero_shot_provenance'))
    merged['planning']['memory_schema_usage'] = memory_schema_usage(root, protocol)
    for record in merged['planning']['per_curve']:
        record['metrics'] = merged['local_evaluation']['per_curve'][record['curve']]
    suffix = '' if protocol['schema'] == 'naive' else '-' + protocol['schema'].replace('_', '-')
    output_json = root / f'timesigma-public-leaderboard-merged{suffix}.json'
    output_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    keys = [m['id'] for m in snapshot['metrics']]
    with (root / f'timesigma-public-leaderboard-merged{suffix}.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'source', 'dataset', 'schema', *keys, *[k + '_rank' for k in keys]])
        for row in merged['rows']:
            writer.writerow([row['display_name'], row['source'], merged['dataset'], merged['schema'],
                             *[row['metrics'][k] if row['metrics'][k] is not None else row['display_values'][k] for k in keys],
                             *[row['ranks'][k] for k in keys]])
    if html_path is not None:
        fonts = snapshot_path.parent / 'raw'
        if not (fonts / 'Mooli-Regular.ttf').is_file():
            fonts = Path(__file__).resolve().parents[1] / 'docs/demo/assets'
        render_html(merged, fonts, html_path)
    return merged


def render_html(merged: dict, font_directory: Path, html_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    font = font_directory / 'Mooli-Regular.ttf'
    merged['font_license'] = (font_directory / 'Mooli-OFL.txt').read_text()
    payload = json.dumps(merged, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    template = (project / 'docs/demo/leaderboard-template.html').read_text()
    html = template.replace('__EMBEDDED_DATA__', payload).replace('__FONT_DATA__', base64.b64encode(font.read_bytes()).decode())
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html)


def memory_schema_usage(root: Path, protocol: dict) -> dict:
    snapshot = json.loads((root / 'memory-snapshot.json').read_text())
    schemas = {}
    kinds = Counter()
    for name in snapshot['evidence_files']:
        record = json.loads(Path(name).read_text())
        schemas[record['evidence_id']] = record['task']['schema']
        kinds[record.get('kind', 'unknown')] += 1
    cited, read_ids = set(), []
    search_filters = Counter()
    same_schema_searches = 0
    for unit in protocol.get('planning_units', protocol['curves']):
        directory = root / 'plans' / unit
        cited.update(json.loads((directory / 'final.json').read_text())['evidence_ids'])
        for path in directory.glob('sessions/**/*.jsonl'):
            for line in path.open():
                message = json.loads(line).get('message', {})
                if message.get('role') == 'assistant':
                    for call in message['content']:
                        if call['type'] == 'toolCall' and call['name'] == 'memory_search':
                            filters = {item['value'] for item in call['arguments'].get('filters', []) if item['field'] == 'schema'}
                            search_filters.update(filters or ['unfiltered'])
                            same_schema_searches += protocol['schema'] in filters
                if message.get('role') == 'toolResult' and message['toolName'] == 'memory_read' and not message.get('isError'):
                    content = '\n'.join(c['text'] for c in message['content'] if c['type'] == 'text')
                    details = json.loads(content)
                    read_ids.extend(item['evidence_id'] for item in details.get('items', []))
    same = sorted(eid for eid in cited if schemas[eid] == protocol['schema'])
    cross = sorted(cited - set(same))
    return {'memory_counts': dict(kinds), 'searches_by_schema_filter': dict(search_filters), 'same_schema_filtered_searches': same_schema_searches,
            'reads_by_evidence_schema': dict(Counter(schemas[eid] for eid in read_ids)),
            'same_schema_reads': sum(schemas[eid] == protocol['schema'] for eid in read_ids),
            'cross_schema_reads': sum(schemas[eid] != protocol['schema'] for eid in read_ids),
            'read_evidence_ids': sorted(set(read_ids)), 'same_schema_evidence_ids': same, 'cross_schema_evidence_ids': cross,
            'same_schema_citations': len(same), 'cross_schema_citations': len(cross),
            'counting': 'Reads count returned Evidence items (repeated reads included); citations count distinct IDs declared in frozen plans.'}


def write_three_schema_outputs(roots: list[Path], output_directory: Path, html_path: Path) -> dict:
    if len(roots) != 3:
        raise ValueError('three completed schema roots are required')
    schemas = {}
    for root in roots:
        protocol = json.loads((root / 'protocol.json').read_text())
        schema = protocol['schema']
        suffix = '' if schema == 'naive' else '-' + schema.replace('_', '-')
        merged = json.loads((root / f'timesigma-public-leaderboard-merged{suffix}.json').read_text())
        own = merged['local_evaluation']
        if merged['schema'].lower() != schema or own['status'] != 'complete':
            raise ValueError('each merged result must match its complete official schema evaluation')
        merged['planning']['memory_schema_usage'] = memory_schema_usage(root, protocol)
        if schema == 'zero_shot':
            merged['zero_shot_provenance'] = json.loads((root / 'execution-audit.json').read_text())['zero_shot_provenance']
        merged['notice'] = 'Public values are preserved from frozen snapshots. TimeSigma and rows marked local default are local AIOPS evaluations, not official leaderboard submissions. Historical V1 AIOPS exposure is disclosed; this is not a claim of project-wide unseen evaluation.'
        schemas[schema] = merged
    if set(schemas) != {'naive', 'all_in_one', 'zero_shot'}:
        raise ValueError('each of the three official schemas is required exactly once')
    summary = {'comparison_status': 'PUBLISHED-LEADERBOARD COMPARISON ONLY', 'schemas': {}}
    for schema, merged in schemas.items():
        own = next(row for row in merged['rows'] if row['source'] == 'local_timesigma_evaluation')
        local = merged['local_evaluation']
        root = next(root for root in roots if root.name == merged['run_root'])
        outcomes = [json.loads(path.read_text()) for path in root.glob('execution/*/*/outcome-*.json')]
        summary['schemas'][schema] = {
            'schema': schema, 'snapshot_timestamp': merged['snapshot_timestamp'], 'public_method_count': sum(row['source'] == 'public_snapshot' for row in merged['rows']),
            'supplemental_baselines': merged.get('supplemental_baselines', []),
            'comparison_row_count': len(merged['rows']), 'point_wise_f1': own['metrics']['point-wise-f1'],
            'point_wise_auprc': own['metrics']['point-wise-auprc'], 'metrics': own['metrics'], 'ranks': own['ranks'],
            'best_published': merged['best_published'], 'planning_unit_type': 'curve' if schema == 'naive' else 'dataset' if schema == 'all_in_one' else 'split',
            'planning': merged['planning'], 'provider': merged['provider'], 'execution_wall_seconds': local['execution_wall_seconds'],
            'evaluation': merged['evaluation'], 'detector_branches': local['detector_executions'],
            'coverage': local['coverage'], 'required_curves': local['required_curves'],
            'retries': sum(item['attempt'] > 1 for item in outcomes), 'zero_shot_split': merged.get('zero_shot_split'),
            'membership_sha256': merged.get('membership_sha256'), 'zero_shot_provenance': merged.get('zero_shot_provenance'),
            'run_root': str(root), 'comparison_status': merged['comparison_status'],
        }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / 'three-schema-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    font_directory = Path(__file__).resolve().parents[1] / 'docs/demo/assets'
    render_html({'schemas': schemas}, font_directory, html_path)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--all-in-one', type=Path)
    parser.add_argument('--zero-shot', type=Path)
    parser.add_argument('--summary-dir', type=Path, default=Path('runs/first-demo/three-schema-delivery-20260910'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.all_in_one is not None or args.zero_shot is not None:
        if args.all_in_one is None or args.zero_shot is None:
            parser.error('both shared Schema roots are required')
        args.output = args.output or Path('docs/demo/TimeSigma_AIOPS_Leaderboard_3Schema.html')
        result = write_three_schema_outputs([args.root, args.all_in_one, args.zero_shot], args.summary_dir, args.output)
        print(json.dumps({'html': str(args.output), 'schemas': list(result['schemas'])}))
    else:
        if args.snapshot is None:
            parser.error('single Schema merge requires --snapshot')
        args.output = args.output or Path('docs/demo/TimeSigma_AIOPS_Leaderboard.html')
        result = write_outputs(args.root, args.snapshot, args.output)
        ours = next(r for r in result['rows'] if r['source'] == 'local_timesigma_evaluation')
        print(json.dumps({'html': str(args.output), 'rows': len(result['rows']), 'primary_rank': ours['ranks'][result['primary_metric']]}))
