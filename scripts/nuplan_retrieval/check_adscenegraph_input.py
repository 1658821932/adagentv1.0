"""Load prepared data through the real ADSceneGraph loader, without training."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mcp_servers.nuplan_data.common import ADSCENEGRAPH_ROOT,runtime_environment,write_json

if os.environ.get('ADAGENT_LOCAL_RUNTIME_READY')!='1':
    environment=runtime_environment()
    environment['ADAGENT_LOCAL_RUNTIME_READY']='1'
    os.execve(sys.executable,[sys.executable,*sys.argv],environment)

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(ADSCENEGRAPH_ROOT))
from omegaconf import OmegaConf
from datasets.driving_dataset import DrivingDataset

config=OmegaConf.load(args.run_dir/'train_config.yaml')
dataset=DrivingDataset(data_cfg=config.data)
report={'status':'loaded','scene_id':config.data.scene_idx,'frames':dataset.num_img_timesteps,
    'cameras':dataset.pixel_source.num_cams,'train_images':len(dataset.train_image_set),
    'aabb':dataset.get_aabb().cpu().tolist(),'objects':int(dataset.pixel_source.instances_pose.shape[1]),
    'source':'real original nuPlan RGB, calibrated camera/ego poses, instance annotations and converted lidar',
    'training_executed':False}
assert report['frames']==100 and report['cameras']==1
write_json(args.run_dir/'loader_check.json',report)
print(json.dumps(report,indent=2))
