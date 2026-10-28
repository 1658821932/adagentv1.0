"""Freeze a reproducible subset of official WorldEngine navtrain scenes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import tarfile

import yaml

from .common import ARTIFACT_ROOT, RAW_ROOT, nuplan_reader, write_json


def parse_config(text):
    # These official files are YAML objects. Retain primitive fields only;
    # never instantiate the Python objects referenced in the YAML tags.
    lines = text.splitlines()
    if lines and lines[0].startswith('!!python/object:'):
        lines.pop(0)
    return yaml.safe_load('\n'.join(lines).replace('!!python/tuple', ''))


def official_scenes(archive, revision):
    rows = []
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.yaml'):
                continue
            raw = tar.extractfile(member).read()
            item = parse_config(raw.decode('utf-8'))
            tokens = item['central_tokens']
            rows.append({
                'scene_id': item['road_block_name'], 'log_name': item['central_log'],
                'scenario_token': tokens[0], 'central_tokens': tokens, 'city': item['city'],
                'official_repo': 'OpenDriveLab/WorldEngine', 'official_revision': revision,
                'official_config_member': member.name,
                'official_config_sha256': hashlib.sha256(raw).hexdigest(),
                'official_road_block': item.get('road_block'), 'official_split': 'navtrain',
            })
    if len({r['scene_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate scene identities in official catalog')
    return rows


OFFICIAL_REVISION='e230eebd77e3add58b2845449592b6e729a69923'
OFFICIAL_ARCHIVE_SHA256='496f719e631c6980d42ae7bf747fa861fcdb66b9328cea876798a64d1ffe0e6c'


def stage_official_archive(path):
    source=Path(path).expanduser().resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest()!=OFFICIAL_ARCHIVE_SHA256:
        raise ValueError('Official navtrain archive does not match the pinned release SHA256')
    target=ARTIFACT_ROOT/'worldengine_navtrain_configs.tar.gz'
    ARTIFACT_ROOT.mkdir(parents=True,exist_ok=True)
    if source!=target.resolve():
        shutil.copy2(source,target)
    with tarfile.open(target,'r:gz') as archive:
        members=[{'name':m.name,'size':m.size} for m in archive if m.isfile() and m.name.endswith('.yaml')]
    write_json(ARTIFACT_ROOT/'official_navtrain_catalog.json',{
        'repo':'OpenDriveLab/WorldEngine','revision':OFFICIAL_REVISION,
        'archive_path':str(target),'archive_sha256':OFFICIAL_ARCHIVE_SHA256,'members':members})
    return len(members)


def select(count=300, seed=20261027, output=None):
    nup = nuplan_reader()
    source = json.loads((ARTIFACT_ROOT/'official_navtrain_catalog.json').read_text())
    archive = ARTIFACT_ROOT/'worldengine_navtrain_configs.tar.gz'
    official = official_scenes(archive, source['revision'])
    dbs = {}
    for part in sorted((RAW_ROOT/'plan_data').iterdir(), key=lambda p:(not p.name.startswith('train'), p.name)):
        if part.is_dir():
            for p in part.glob('*.db'):
                dbs.setdefault(p.stem, p)
    locator = nup.SensorLocator([RAW_ROOT/'sensor_blobs_full', RAW_ROOT/'sensor_blobs_mini'])
    bycity = defaultdict(list)
    for row in official:
        bycity[row['city']].append(row)
    rng = random.Random(seed)
    for rows in bycity.values():
        rows.sort(key=lambda r:r['scene_id'])
        rng.shuffle(rows)
    selected, rejected = [], []
    used_logs = set()
    # Round robin over cities, one scene per log, avoids a majority-city sample
    # and highly overlapping clips from a single drive.
    while len(selected)<count and any(bycity.values()):
        for city in sorted(bycity):
            rows = bycity[city]
            accepted = False
            while rows and not accepted:
                row = dict(rows.pop())
                if row['log_name'] in used_logs:
                    continue
                database = dbs.get(row['log_name'])
                if database is None:
                    rejected.append({'scene_id':row['scene_id'],'reason':'database_missing'})
                    continue
                try:
                    with nup.connect(database) as db:
                        anchor = nup.anchor_frame(db,row['scenario_token'])
                        start = int(anchor['lidar_timestamp'])-2_000_000
                        end = start+10_000_000
                        calibrations = nup.cameras(db)
                        for t in (start,start+5_000_000,end-100_000):
                            frame = nup.nearest_frames(db,[t])[0]
                            if locator.resolve(frame['filename']) is None:
                                raise FileNotFoundError('lidar_missing')
                            for calibration in calibrations.values():
                                image = nup.nearest_image(db,calibration['token'],frame['lidar_timestamp'])
                                if locator.resolve(image['filename_jpg']) is None:
                                    raise FileNotFoundError('camera_image_missing')
                        tags = [r[0] for r in db.execute('SELECT type FROM scenario_tag WHERE lidar_pc_token=?',(bytes.fromhex(row['scenario_token']),))]
                    row.update(database=str(database),db_partition=database.parent.name,
                        anchor_timestamp_us=int(anchor['lidar_timestamp']),start_timestamp_us=start,
                        end_timestamp_us_exclusive=end,fps=10.,frame_count=100,cameras=list(range(8)),
                        scenario_tags=sorted(set(tags)),availability='sampled_endpoints_and_midpoint_passed')
                    selected.append(row)
                    used_logs.add(row['log_name'])
                    accepted = True
                    print(f"SELECT {len(selected)}/{count} {city} {row['scene_id']}",flush=True)
                except (ValueError,FileNotFoundError) as exc:
                    rejected.append({'scene_id':row['scene_id'],'reason':str(exc)})
                if len(selected)>=count:
                    break
            if len(selected)>=count:
                break
    payload = {'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'official_revision':source['revision'],'official_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'official_scene_count':len(official),'official_city_counts':dict(Counter(r['city'] for r in official)),
        'selection_seed':seed,'requested_count':count,'selected_count':len(selected),
        'selection_policy':'city round-robin, one scene per log, deterministic shuffle, original images and lidar sampled at start/middle/end',
        'full_frame_validation':'performed by prepare; selection checks are not full-scene validation',
        'selected_city_counts':dict(Counter(r['city'] for r in selected)),
        'scenes':selected,'rejected':rejected}
    path = output or ARTIFACT_ROOT/'selection_300.json'
    write_json(path,payload)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('scenes','rejected')},ensure_ascii=False),flush=True)
    if len(selected)<count:
        raise RuntimeError(f'Only {len(selected)} usable official navtrain scenes found; see {path}')
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=300)
    parser.add_argument('--official-archive',type=Path,help='Pinned WorldEngine navtrain configs.tar.gz; checked before staging')
    parser.add_argument('--seed',type=int,default=20261027)
    args=parser.parse_args()
    if args.count<=0:
        parser.error('--count must be positive')
    if args.official_archive:
        stage_official_archive(args.official_archive)
    select(args.count,args.seed)
