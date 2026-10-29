"""Run actual natural-language retrieval and optional new-scene reconstruction."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mcp_servers.nuplan_data.common import ARTIFACT_ROOT,write_json
from mcp_servers.nuplan_data.manifest import load_manifest
from tools.tool04_video_retriever.real import run as retrieve
from tools.tool06_reconstruction.real import run as reconstruct

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--query',required=True)
parser.add_argument('--iterations',type=int,default=3000)
parser.add_argument('--retrieval-only',action='store_true')
parser.add_argument('--retrieval-result',type=Path)
args=parser.parse_args()
directory=ARTIFACT_ROOT/'acceptance'/('run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
directory.mkdir(parents=True)
state={'query':args.query,'status':'retrieving','run_dir':str(directory),'mock_used':False,
       'preexisting_gaussian_assets_used':False,'retrieval_only':args.retrieval_only}
write_json(directory/'run.json',state)
t0=time.perf_counter()
print('ACCEPTANCE_RUN',directory,flush=True)
try:
    if args.retrieval_result:
        response=json.loads(args.retrieval_result.read_text())
        state['reused_retrieval_result']=str(args.retrieval_result.resolve())
    else:
        response=retrieve({'requirement':args.query,'library':'nuplan'})
    write_json(directory/'retrieval.json',response)
    result=response['result']
    if result.get('requirement')!=args.query:
        raise ValueError('Reused retrieval result belongs to a different natural-language query')
    scene=load_manifest(result['scene_manifest_path'])
    assert result['scene_id']==scene['scene_id']
    assert Path(result['video_path']).resolve()==Path(scene['video_path']).resolve()
    state.update(status='retrieved',scene_id=result['scene_id'],match_score=result['match_score'],
        scene_manifest_path=result['scene_manifest_path'],video_path=result['video_path'],
        retrieval_run_dir=result['run_dir'],retrieval_seconds=round(time.perf_counter()-t0,3),
        annotation_frame_count=scene['frame_count'],annotation_camera_count=len(scene['calibrations']))
    write_json(directory/'run.json',state)
    print('RETRIEVAL_PASSED',json.dumps({k:state[k] for k in ('scene_id','match_score','annotation_frame_count','retrieval_seconds')},ensure_ascii=False),flush=True)
    if not args.retrieval_only:
        state['status']='reconstructing'
        write_json(directory/'run.json',state)
        response=reconstruct({'backend':'adscenegraph','video_path':result['video_path'],
            'scene_manifest_path':result['scene_manifest_path'],'iterations':args.iterations,'cameras':[0],
            'output_dir':str(directory/'reconstruction')})
        write_json(directory/'reconstruction_result.json',response)
        result=response['result']
        assert Path(result['checkpoint_path']).is_file()
        assert Path(result['rendered_video']).is_file()
        assert result['scene_id']==state['scene_id']
        state.update(checkpoint_path=result['checkpoint_path'],rendered_video=result['rendered_video'],
            iterations=args.iterations,reconstruction_run_dir=result['run_dir'])
    state.update(status='retrieval_passed' if args.retrieval_only else 'reconstructed',elapsed_seconds=round(time.perf_counter()-t0,3))
    write_json(directory/'run.json',state)
    print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)
except Exception as exc:
    state.update(status='error',error_type=type(exc).__name__,error=str(exc),elapsed_seconds=round(time.perf_counter()-t0,3))
    write_json(directory/'run.json',state)
    print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)
    raise
