"""Check every referenced raw sensor file, without modifying or copying it."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/nuplan_retrieval'
selection_path=ART/'selection_final_300.json'
selection=json.loads(selection_path.read_text())

def audit(row):
    scene=ROOT/'mcp_servers/sentry_video_search/data_nuplan/scenes'/row['scene_id']/'scene_manifest.json'
    data=json.loads(scene.read_text())
    refs=[('database',row['database'])]
    for frame in data['frames']:
        refs.append(('lidar',frame['lidar_path']))
        refs.extend(('camera',image['image_path']) for image in frame['images'])
    counts={'camera':0,'lidar':0,'database':0}
    errors=[]
    for kind,name in refs:
        path=Path(name)
        try:
            stat=path.stat()
            if not path.is_file() or stat.st_size<=0:
                raise ValueError('Missing, empty or non-file source reference')
            counts[kind]+=1
        except (OSError,ValueError) as exc:
            errors.append({'kind':kind,'path':str(path),'type':type(exc).__name__,'error':str(exc)})
    return {'scene_id':row['scene_id'],'counts':counts,'errors':errors}

records=[]
with ThreadPoolExecutor(max_workers=8) as pool:
    for number,record in enumerate(pool.map(audit,selection['scenes']),1):
        records.append(record)
        if number%25==0:
            print('RAW_REFERENCE_AUDIT',number,'/300',flush=True)
errors=[{'scene_id':r['scene_id'],**e} for r in records for e in r['errors']]
report={'status':'passed' if not errors and len(records)==300 else 'error',
    'selected_scene_count':len(records),'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    'camera_references_checked':sum(r['counts']['camera'] for r in records),
    'lidar_references_checked':sum(r['counts']['lidar'] for r in records),
    'database_references_checked':sum(r['counts']['database'] for r in records),
    'check_scope':'Every referenced path exists as a nonempty regular file; raw file contents are not re-decoded or hashed.',
    'errors':errors,'records':records}
out=ART/'source_reference_audit.json'
tmp=out.with_suffix('.json.tmp')
tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
tmp.replace(out)
print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2),flush=True)
if report['status']!='passed':
    raise SystemExit(1)
