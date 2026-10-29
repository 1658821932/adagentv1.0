"""Evaluate caption recall offline, optionally run real Kimi queries (API fees).

The labeled synthetic cases are regression checks, not a real-video benchmark.
Live match scores are model judgments; inspect returned videos separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mcp_servers.sentry_video_search.hybrid_recall import hybrid_rank
from mcp_servers.sentry_video_search.kimi_backend import KimiNoMatchError, recall_scores
from mcp_servers.sentry_video_search.config import SearchConfig


def caption_row(name, text, unknown=()):
    caption = {'summary_zh': text, 'summary_en': '', 'road_layout': '', 'ego_motion': '',
               'visible_actors': [], 'observed_interactions': [], 'tags': [], 'uncertainties': list(unknown)}
    return {'id': name, 'source_file': name + '.mp4', 'metadata': json.dumps({'scene_id': name}),
            'caption': json.dumps(caption, ensure_ascii=False), 'document': text + ' ' + ' '.join(unknown)}


def offline_evaluation():
    # Hand-authored expected matches precede ranking. These deliberately include
    # negative and unknown observations that the flat legacy index also stored.
    rows = [caption_row('bus', '前方有一辆公交车 bus passing through a junction'),
            caption_row('coach', 'A coach is approaching traffic lights.'),
            caption_row('unknown_bus', '城市道路', ['公交车 公共汽车 巴士 bus buses coach 无法确认是否存在'] * 4),
            caption_row('no_bus', '没有公交车，没有巴士，no bus or buses visible'),
            caption_row('pedestrian', '行人穿过人行横道 pedestrian crossing a crosswalk'),
            caption_row('unknown_pedestrian', '城市道路', ['行人 pedestrian pedestrians 是否出现无法确认'] * 4),
            caption_row('rain', '白天路口正在下雨 rainy intersection in daylight'),
            caption_row('dry', '白天城市路口，晴天，干燥路面，无下雨 daytime sunny junction without rain'),
            caption_row('trees', '白天道路两侧有树木，车流较少，路边有建筑 daytime tree-lined road light traffic buildings'),
            caption_row('night', '夜间城市道路 night urban road'),
            caption_row('truck', '货车沿道路行驶 a lorry driving on a road'),
            caption_row('tunnel', '车辆穿过隧道 tunnel traffic'),
            caption_row('roadworks', '前方道路施工 traffic passing a construction work zone'),
            caption_row('cyclist', '骑行者沿道路骑自行车 cyclist riding a bicycle'),
            caption_row('left', '自车在十字路口左转 ego turning left at an intersection'),
            caption_row('right', '自车在十字路口右转 ego turning right at an intersection')]
    cases = [('公交车', {'bus', 'coach'}), ('公共汽车', {'bus', 'coach'}),
             ('bus', {'bus', 'coach'}), ('行人', {'pedestrian'}), ('pedestrian', {'pedestrian'}),
             ('白天路口，不要下雨', {'dry'}), ('daylight junction without rain', {'dry'}),
             ('白天树木较多的道路，车流较少，路边有建筑', {'trees'}),
             ('夜间城市道路', {'night'}), ('卡车', {'truck'}), ('truck', {'truck'}),
             ('隧道', {'tunnel'}), ('道路施工', {'roadworks'}), ('roadworks', {'roadworks'}),
             ('骑行者', {'cyclist'}), ('十字路口左转', {'left'}), ('right turn at a junction', {'right'})]
    results = []
    for query, expected in cases:
        scores = recall_scores(query, [row['document'] for row in rows])
        baseline = [row['id'] for row, score in sorted(zip(rows, scores), key=lambda item: item[1], reverse=True) if score > 0][:4]
        selected, _ = hybrid_rank(query, rows)
        upgraded = [entry['row']['id'] for entry in selected]
        def metrics(ranking):
            ranks = [i + 1 for i, identity in enumerate(ranking) if identity in expected]
            return {'hit_at_1': bool(ranks and ranks[0] == 1), 'hit_at_4': bool(ranks), 'reciprocal_rank_at_4': 1 / ranks[0] if ranks else 0.}
        results.append({'query': query, 'expected': sorted(expected), 'baseline': baseline, 'upgraded': upgraded,
                        'baseline_metrics': metrics(baseline), 'upgraded_metrics': metrics(upgraded)})
    aggregate = {}
    for version in ('baseline', 'upgraded'):
        aggregate[version] = {metric: sum(row[version + '_metrics'][metric] for row in results) / len(results)
                              for metric in ('hit_at_1', 'hit_at_4', 'reciprocal_rank_at_4')}
    return {'scope': '17 hand-authored queries / 16 synthetic captions; regression suite only, not labeled real-video accuracy. Baseline is flat original-query BM25 without API expansion; upgraded recall adds local bilingual concepts and field evidence. No API calls.', 'aggregate': aggregate, 'cases': results}


def caption_snapshot(config):
    with sqlite3.connect(config.runtime_home / '.sentrysearch/kimi_caption.sqlite3') as db:
        rows = db.execute('SELECT * FROM chunks ORDER BY id').fetchall()
    return {'count': len(rows), 'sha256': hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()}


def live_evaluation(output):
    from dotenv import load_dotenv
    from mcp_servers.nuplan_data.manifest import load_manifest
    from mcp_servers.sentry_video_search.service import search_local
    load_dotenv(ROOT / '.env')
    config = SearchConfig.for_library('nuplan')
    before = caption_snapshot(config)
    cases = [
        ('白天树木较多的城市道路，车流较少，路边有建筑', 'match'),
        ('宽阔的城市路口，有交通信号灯和车辆通行', 'match'),
        ('城市道路上有公交车，前方有交通信号灯', 'match'),
        ('夜间下大雪，道路被积雪覆盖', 'no_match'),
    ]
    records = []
    def execute(query, expected, repeated=False):
        old = set(config.logs_dir.glob('run_*'))
        start = time.perf_counter()
        item = {'query': query, 'expected_outcome': expected, 'repeated_query': repeated, 'mock_used': False}
        try:
            result = search_local(query, config)
            manifest = load_manifest(result.scene_manifest_path)
            assert manifest['scene_id'] == result.scene_id
            assert Path(manifest['video_path']).resolve() == Path(result.video_path).resolve()
            assert hashlib.sha256(Path(result.video_path).read_bytes()).hexdigest() == manifest['video_sha256']
            item.update(outcome='match', result=result.to_dict(), annotation_frame_count=manifest['frame_count'])
            run_dir = Path(result.run_dir)
        except KimiNoMatchError:
            item['outcome'] = 'no_match'
            created = set(config.logs_dir.glob('run_*')) - old
            assert len(created) == 1
            run_dir = created.pop()
        except Exception as error:
            item.update(outcome='error', error_type=type(error).__name__, error=str(error))
            run_dir = None
        item['elapsed_seconds'] = round(time.perf_counter() - start, 3)
        item['outcome_as_expected'] = item['outcome'] == expected
        if run_dir:
            local = json.loads((run_dir / 'local_search.json').read_text())
            rerank = json.loads((run_dir / '09_rerank.json').read_text())
            item.update(retrieval_run_dir=str(run_dir), query_expansion_cache_hit=bool((local.get('trace') or {}).get('cache_hit')),
                        visual_cache_hits=rerank['cache_hits'], visual_candidate_count=rerank['candidate_count'],
                        visual_api_requests=sum(not trace.get('cache_hit', False) for trace in rerank['traces']),
                        condition_checks=[{'candidate_index': row['candidate_index'], 'matched': row['matched'], 'conditions': row.get('condition_checks')} for row in rerank['ranked_results']])
        records.append(item)
        (output / 'live_progress.json').write_text(json.dumps({'records': records}, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({key: item.get(key) for key in ('query', 'outcome', 'repeated_query', 'elapsed_seconds', 'visual_cache_hits', 'visual_api_requests')}, ensure_ascii=False), flush=True)
    for index, (query, expected) in enumerate(cases):
        execute(query, expected)
        if index == 0 and records[-1]['outcome'] == 'match':
            execute(query, expected, repeated=True)
    after = caption_snapshot(config)
    assert before == after, 'Existing paid caption rows changed during retrieval'
    return {'scope': 'Real queries over the 300-scene index, with actual Kimi API and video/manifest identity validation. Model matching decisions require separate visual review.',
            'caption_snapshot_before': before, 'caption_snapshot_after': after, 'all_outcomes_as_expected': all(row['outcome_as_expected'] for row in records), 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Also send actual sampled images to the configured gateway; incurs API charges.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'artifacts/nuplan_retrieval/retrieval_upgrade' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output = output.resolve()
    output.relative_to((ROOT / 'artifacts').resolve())
    output.mkdir(parents=True, exist_ok=True)
    report = {'offline': offline_evaluation()}
    (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'offline': report['offline']['aggregate']}, ensure_ascii=False), flush=True)
    if args.live:
        report['live'] = live_evaluation(output)
        (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        if not report['live']['all_outcomes_as_expected']:
            raise SystemExit('Some live outcomes differ; inspect the saved report before claiming acceptance.')


if __name__ == '__main__':
    main()
