"""Replace incomplete candidates with fully validated official scenes."""
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mcp_servers.nuplan_data.catalog import official_scenes
from mcp_servers.nuplan_data.common import ARTIFACT_ROOT,RAW_ROOT,nuplan_reader,write_json
from mcp_servers.nuplan_data.prepare import prepare_scene

base=json.loads((ARTIFACT_ROOT/'selection_300.json').read_text())
preparation=json.loads((ARTIFACT_ROOT/'preparation_300.json').read_text())
byid={row['scene_id']:row for row in base['scenes']}
successful={row['scene_id'] for row in preparation['records'] if row['status'] in ('prepared','reused')}
selected=[row for row in base['scenes'] if row['scene_id'] in successful]
needed=Counter(row['city'] for row in base['scenes'])
needed.subtract(row['city'] for row in selected)
print('REPLACEMENTS_NEEDED',dict(needed),flush=True)
used_logs={row['log_name'] for row in base['scenes']}
dbs={}
for part in sorted((RAW_ROOT/'plan_data').iterdir(),key=lambda p:(not p.name.startswith('train'),p.name)):
    if part.is_dir():
        for path in part.glob('*.db'):
            dbs.setdefault(path.stem,path)
official=official_scenes(ARTIFACT_ROOT/'worldengine_navtrain_configs.tar.gz',base['official_revision'])
rng=random.Random(20261028)
rng.shuffle(official)
nup=nuplan_reader()
records=[]
for city,count in needed.items():
    if count<=0:
        continue
    for row in official:
        if row['city']!=city or row['log_name'] in used_logs or row['log_name'] not in dbs:
            continue
        candidate=dict(row)
        database=dbs[row['log_name']]
        try:
            with nup.connect(database) as db:
                anchor=nup.anchor_frame(db,row['scenario_token'])
                tags=[r[0] for r in db.execute('SELECT type FROM scenario_tag WHERE lidar_pc_token=?',(bytes.fromhex(row['scenario_token']),))]
            start=int(anchor['lidar_timestamp'])-2000000
            candidate.update(database=str(database),db_partition=database.parent.name,
                anchor_timestamp_us=int(anchor['lidar_timestamp']),start_timestamp_us=start,
                end_timestamp_us_exclusive=start+10000000,fps=10.,frame_count=100,cameras=list(range(8)),
                scenario_tags=sorted(set(tags)),availability='full_validation_by_prepare')
            record=prepare_scene(candidate)
            selected.append(candidate)
            used_logs.add(candidate['log_name'])
            count-=1
            records.append(record)
            print('SUPPLEMENT_OK',candidate['scene_id'],'remaining_city',count,flush=True)
        except Exception as exc:
            records.append({'scene_id':candidate['scene_id'],'status':'error','error_type':type(exc).__name__,'error':str(exc)})
            print('SUPPLEMENT_SKIP',candidate['scene_id'],str(exc),flush=True)
        write_json(ARTIFACT_ROOT/'preparation_supplement.json',{'records':records,'current_complete_count':len(selected)})
        if count==0:
            break
    if count:
        raise RuntimeError(f'Could not supplement {count} scenes for {city}')
assert len(selected)==300
assert len({row['log_name'] for row in selected})==300
final={**{k:v for k,v in base.items() if k not in ('scenes','rejected')},
    'scenes':selected,'selected_count':len(selected),'selected_city_counts':dict(Counter(row['city'] for row in selected)),
    'supersedes_selection':'selection_300.json','full_validation':'passed in prepare_scene for all selected scenes',
    'excluded_initial_candidates':[row for row in preparation['records'] if row['status']=='error'],
    'supplement_seed':20261028}
write_json(ARTIFACT_ROOT/'selection_final_300.json',final)
print('FINAL_COMPLETE_CATALOG',len(selected),final['selected_city_counts'],flush=True)
