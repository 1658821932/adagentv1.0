"""Finish video export from an existing acceptance checkpoint, without training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mcp_servers.nuplan_data.common import write_json
from mcp_servers.nuplan_data.reconstruct import finalize_existing

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('acceptance_dir',type=Path)
args=parser.parse_args()
directory=args.acceptance_dir.resolve()
state=json.loads((directory/'run.json').read_text())
result=finalize_existing(directory/'reconstruction')
write_json(directory/'reconstruction_result.json',{'status':'success','tool':'reconstruction','result':result})
if state.get('error'):
    state['previous_failure']={'type':state.get('error_type'),'message':state['error']}
state.pop('error',None);state.pop('error_type',None)
state.update(status='reconstructed',checkpoint_path=result['checkpoint_path'],rendered_video=result['rendered_video'],
    iterations=result['iterations'],reconstruction_run_dir=result['run_dir'],
    recovery_action='Rendered the existing completed checkpoint; no retraining')
write_json(directory/'run.json',state)
print(json.dumps({k:v for k,v in state.items() if k not in ('previous_failure',)},ensure_ascii=False,indent=2))
