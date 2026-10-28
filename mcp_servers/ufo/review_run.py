"""Reproducible insertion/right-shift review through the actual Agent executor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconstruction-json',type=Path,required=True)
    parser.add_argument('--asset-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--edited-root',type=Path)
    parser.add_argument('--calibration-path',type=Path)
    parser.add_argument('--gpu',type=int,default=int(os.environ.get('UFO_GPU','0')))
    parser.add_argument('--forward-m',type=float,default=20)
    parser.add_argument('--left-m',type=float,default=3)
    parser.add_argument('--right-m',type=float,default=3)
    parser.add_argument('--start-seconds',type=float,default=0)
    parser.add_argument('--end-seconds',type=float)
    parser.add_argument('--motion-reference',choices=['world_fixed','ego_projected','ego_translation'],default='world_fixed')
    parser.add_argument('--road-proxy',action='store_true')
    parser.add_argument('--road-model',choices=['local_depth','motion_near_depth'],default='motion_near_depth')
    args=parser.parse_args()
    if args.gpu<0 or args.right_m<=0:parser.error('gpu must be nonnegative and right-m positive')
    import numpy as np
    from agent.executor import ToolExecutor
    from .metric_control import read_reconstruction
    from .metric_assets import load_asset
    from .review import audit_pair
    from .video_compare import compare_videos
    reconstruction=read_reconstruction(args.reconstruction_json.resolve())
    load_asset(args.asset_manifest.resolve())  # Fail before GPU work if units/axes/contact are missing.
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    executor=ToolExecutor({'scene_editor':'real','edit_config_generator':'real'},reconstruction_backend='ufo')
    executor.reconstruction=reconstruction

    def call(name,values,label):
        result=executor.execute(name,values)
        (args.output/f'tool_{label}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        if result.get('status')!='success':raise RuntimeError(json.dumps(result,ensure_ascii=False))
        return result['result']

    calibration=args.calibration_path or Path(reconstruction['workspace'])/'metric_scene_frontend_motion_v1.json'
    cal=call('scene_editor',{'operation':'calibrate_metric','calibration_method':'frontend_metric',
        'road_model':args.road_model,'output_path':str(calibration.resolve()),'gpu':args.gpu},'01_calibration')
    with np.load(reconstruction['camera_file'],allow_pickle=False) as cameras:
        end=args.end_seconds if args.end_seconds is not None else float(cameras['timestamps'][-1])*.9
    insert={'operation':'insert','object_id':'review_car','asset_manifest_path':str(args.asset_manifest.resolve()),
            'position_m':[args.forward_m,args.left_m,0],'yaw_degrees':0,'motion_reference':args.motion_reference}
    translation={'operation':'translate','target_id':'review_car','offset_m':[0,-args.right_m,0],
                 'start_seconds':args.start_seconds,'end_seconds':end}
    baseline=[insert]
    if args.start_seconds>0:
        baseline.append({**translation,'offset_m':[0,0,0]})
    results=[]
    for label,operations,root in [('baseline',baseline,args.baseline_root),('edited',[insert,translation],args.edited_root)]:
        config=call('edit_config_generator',{'operation':'metric_edit','calibration_path':cal['calibration_path'],
            'operations':operations,'road_geometry':'planar_proxy' if args.road_proxy else 'native',
            'output_path':str(args.output/f'{label}_config.json')},f'02_config_{label}')
        result=call('scene_editor',{'operation':'metric_edit','edit_config_path':config['saved_path'],
            'output_root':str((root or args.output/label).resolve()),'gpu':args.gpu},f'03_render_{label}')
        results.append(result)
    before,after=results
    audit=audit_pair(Path(before['result_path']),Path(after['result_path']),'review_car',args.output)
    comparison=compare_videos(Path(before['rendered_video']),Path(after['rendered_video']),args.output/'comparison.mp4')
    result={'audit':audit,'comparison':comparison,'validation_scope':cal['validation_scope'],
            'visual_status':'requires_human_review','absolute_accuracy_certified':False}
    (args.output/'review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
