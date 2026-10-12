from copy import deepcopy

import pytest

from scripts.build_leaderboard_demo import merge_leaderboard
from timesigma.benchmark.leaderboard import OFFICIAL_METRIC_IDS


def fixture_inputs():
    metrics = [{'id': key, 'label': key} for key in OFFICIAL_METRIC_IDS]
    snapshot = {'dataset': 'fixture', 'schema': 'NAIVE', 'primary_metric': 'point-wise-f1',
                'snapshot_timestamp': '2026-09-09T00:00:00+00:00', 'source_url': 'https://example.invalid/fixture',
                'metrics': metrics, 'ranking_policy': {'precision_decimals': 4, 'rounding': 'ROUND_HALF_UP'},
                'rows': [{'method': name, 'display_name': name, 'source': 'public_snapshot',
                          'metrics': {key: score for key in OFFICIAL_METRIC_IDS},
                          'display_values': {key: 'N/A' if score is None else str(score) for key in OFFICIAL_METRIC_IDS}}
                         for name, score in [('Published A', 0.9), ('Published B', 0.8), ('Unavailable', None)]]}
    local = {'dataset': 'fixture', 'schema': 'naive', 'status': 'complete', 'provider': {'calls': 1, 'tokens': 100},
             'rows': [{'method': 'TimeSigma_First_Demo', 'coverage': 29, 'required_curves': 29, 'failures': [],
                       'official_metrics': {key: 0.90004 for key in OFFICIAL_METRIC_IDS}}]}
    return snapshot, local


def test_merge_preserves_facts_missing_rows_and_ties_at_published_precision():
    snapshot, local = fixture_inputs()
    original = deepcopy(snapshot)
    local['rows'][0]['official_metrics']['event-wise-f1'] = 0.7
    result = merge_leaderboard(snapshot, local)
    assert snapshot == original
    assert len(result['rows']) == 4
    ours = result['rows'][-1]
    assert ours['metrics']['point-wise-f1'] == 0.90004
    assert result['rows'][0]['metrics']['point-wise-f1'] == 0.9
    assert ours['ranks']['point-wise-f1'] == result['rows'][0]['ranks']['point-wise-f1'] == 1
    assert result['rows'][1]['ranks']['point-wise-f1'] == 3
    assert ours['ranks']['event-wise-f1'] == 3
    assert result['rows'][2]['ranks']['point-wise-f1'] is None
    assert result['rows'][2]['display_values']['point-wise-f1'] == 'N/A'
    assert result['best_published']['point-wise-f1']['delta'] == 0.90004 - 0.9
    assert ours['source'] == 'local_timesigma_evaluation'


@pytest.mark.parametrize('fault', ['incomplete', 'missing_metric', 'nonfinite', 'wrong_dataset'])
def test_merge_rejects_incomplete_or_unmatched_scientific_inputs(fault):
    snapshot, local = fixture_inputs()
    if fault == 'incomplete':
        local['rows'][0]['coverage'] = 28
    elif fault == 'missing_metric':
        local['rows'][0]['official_metrics'].pop('point-wise-f1')
    elif fault == 'nonfinite':
        local['rows'][0]['official_metrics']['point-wise-f1'] = float('nan')
    else:
        local['dataset'] = 'different'
    with pytest.raises(ValueError):
        merge_leaderboard(snapshot, local)


def test_shared_schema_merge_uses_its_own_snapshot_and_preserves_missing_rows():
    snapshot, local = fixture_inputs()
    snapshot['schema'] = 'ALL_IN_ONE'
    local['schema'] = 'all_in_one'
    snapshot['rows'][1]['metrics']['point-wise-f1'] = 0.92
    snapshot['rows'][1]['display_values']['point-wise-f1'] = '0.9200'
    result = merge_leaderboard(snapshot, local)
    assert result['schema'] == 'ALL_IN_ONE'
    assert result['rows'][-1]['ranks']['point-wise-f1'] == 2
    assert result['best_published']['point-wise-f1']['methods'] == ['Published B']
    assert result['rows'][1]['display_values']['point-wise-f1'] == '0.9200'
    assert result['rows'][2]['metrics']['point-wise-f1'] is None
    local['schema'] = 'zero_shot'
    with pytest.raises(ValueError, match='task do not match'):
        merge_leaderboard(snapshot, local)


def test_memory_usage_counts_returned_details_and_declared_schema_citations(tmp_path):
    import json
    from scripts.build_leaderboard_demo import memory_schema_usage

    evidence = {}
    for name, schema in [('same', 'zero_shot'), ('cross', 'naive')]:
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps({'evidence_id': name, 'task': {'schema': schema}}))
        evidence[str(path)] = 'unused-by-descriptive-view'
    (tmp_path / 'memory-snapshot.json').write_text(json.dumps({'evidence_files': evidence}))
    unit = tmp_path / 'plans/AIOPS'
    (unit / 'sessions').mkdir(parents=True)
    (unit / 'final.json').write_text(json.dumps({'evidence_ids': ['same', 'cross']}))
    messages = [
        {'role': 'assistant', 'content': [{'type': 'toolCall', 'name': 'memory_search', 'arguments': {'filters': [{'field': 'schema', 'value': 'zero_shot'}]}}]},
        {'role': 'assistant', 'content': [{'type': 'toolCall', 'name': 'memory_read', 'arguments': {'evidence_ids': ['same', 'cross']}}]},
        {'role': 'toolResult', 'toolName': 'memory_read', 'content': [{'type': 'text', 'text': json.dumps({'items': [{'evidence_id': 'same'}]})}]},
    ]
    (unit / 'sessions/events.jsonl').write_text(''.join(json.dumps({'message': message}) + '\n' for message in messages))
    result = memory_schema_usage(tmp_path, {'schema': 'zero_shot', 'planning_units': ['AIOPS'], 'curves': ['a', 'b']})
    assert result['same_schema_filtered_searches'] == 1
    assert result['same_schema_reads'] == 1 and result['cross_schema_reads'] == 0
    assert result['same_schema_evidence_ids'] == ['same'] and result['cross_schema_evidence_ids'] == ['cross']


def test_supplement_requires_frozen_membership_and_never_relabels_public_values():
    snapshot, local = fixture_inputs()
    supplement = deepcopy(local['rows'][0])
    supplement.update(method='IsolationForest', status='complete')
    local['rows'].append(supplement)
    with pytest.raises(ValueError):
        merge_leaderboard(snapshot, local)  # Historical snapshot-only contract stays strict.
    original = deepcopy(snapshot)
    result = merge_leaderboard(snapshot, local, ('IsolationForest',), {'LOF': 'shared fit unavailable'})
    assert snapshot == original
    row = next(row for row in result['rows'] if row['method'] == 'IsolationForest')
    assert row['source'] == 'local_baseline_evaluation'
    na = next(row for row in result['rows'] if row['method'] == 'LOF')
    assert all(value is None for value in na['ranks'].values())
    supplement['coverage'] = 28
    with pytest.raises(ValueError):
        merge_leaderboard(snapshot, local, ('IsolationForest',))
