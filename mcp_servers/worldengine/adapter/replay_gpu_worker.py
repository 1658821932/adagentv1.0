"""Render actual native WE states using its MTGS renderer and the existing GPU environment."""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import traceback
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--request',type=Path,required=True);args=parser.parse_args();request=json.loads(args.request.read_text())
    protocol=sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            import cv2
            root=Path(request['worldengine_root']);sys.path.insert(0,str(root/'projects/SimEngine'))
            # nuPlan state definitions are Python source; do not import simulator
            # site-packages or its Python-3.9 extension modules into this process.
            source=Path(request['nuplan_source'])
            spec=importlib.util.spec_from_file_location('nuplan',source/'__init__.py',submodule_search_locations=[str(source)])
            package=importlib.util.module_from_spec(spec);sys.modules['nuplan']=package;spec.loader.exec_module(package)
            if request.get('gsplat_extension'):
                import gsplat
                extension=Path(request['gsplat_extension']).resolve(strict=True)
                spec=importlib.util.spec_from_file_location('gsplat_cuda',extension);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);gsplat.csrc=backend
            with Path(request['staged_scenario']).open('rb') as stream:scene=pickle.load(stream)[request['scene_id']]
            import worldengine.engine.engine_utils as engine_utils
            engine_utils._global_config=SimpleNamespace(asset_folder_path=request['asset_folder_path'])
            engine_utils._engine_instance=SimpleNamespace(current_scene=scene,managers={'scenario_manager':SimpleNamespace(current_scene=scene)})
            from replay_pose import install
            install(renderer_only=True)
            from worldengine.render.mtgs.mtgs import MTGSRenderEngine
            renderer=MTGSRenderEngine(**request['renderer_options']);renderer.reset(request['scene_id'])
            import worldengine.render.mtgs.mtgs as mtgs
            original_raster=mtgs.rasterization
            def traced_raster(**kwargs):
                if not (Path(request['renderer_output_dir'])/'raster-input.pt').exists():
                    captured={k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in kwargs.items()}
                    torch.save(captured,Path(request['renderer_output_dir'])/'raster-input.pt')
                    print('raster shapes', {k:tuple(v.shape) for k,v in kwargs.items() if torch.is_tensor(v)},file=sys.stderr,flush=True)
                    print('camera matrices',kwargs['viewmats'].detach().cpu().tolist(),file=sys.stderr,flush=True)
                return original_raster(**kwargs)
            mtgs.rasterization=traced_raster
        print(json.dumps(dict(status='ready')),file=protocol,flush=True)
        output=Path(request['renderer_output_dir'])
        for line in sys.stdin:
            message=json.loads(line)
            if message.get('stop'):break
            state=message['render_state'];state['agent_state']={k:np.asarray(v,float) for k,v in state['agent_state'].items()};state['cameras']=scene['cameras']
            if message['sequence']==0:
                (output/'first-render-state.json').write_text(json.dumps(message,default=lambda value:value.tolist()))
            with contextlib.redirect_stdout(sys.stderr):
                result=renderer.render(state)
                paths={}
                for camera,images in result['cameras'].items():
                    path=output/(camera+'_'+str(message['sequence']).zfill(3)+'.png')
                    if not cv2.imwrite(str(path),images['image']):raise RuntimeError('GPU image could not be saved')
                    paths[camera]=str(path)
            print(json.dumps(dict(status='rendered',sequence=message['sequence'],camera_images=paths,ego2global=result['ego2global'].tolist())),file=protocol,flush=True)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps(dict(status='error',error=str(error)[:1000])),file=protocol,flush=True)
        raise
if __name__=='__main__':main()
