"""Inspect actual render videos against their matching training RGB frames."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mcp_servers.nuplan_data.common import write_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('acceptance_dir',type=Path)
args=parser.parse_args()
directory=args.acceptance_dir.resolve()
state=json.loads((directory/'run.json').read_text())
if state['status']!='reconstructed':
    raise SystemExit('New reconstruction has not completed')
render=Path(state['rendered_video'])
ground_truth=render.with_name(render.name.replace('_rgbs','_gt_rgbs',1))
if not ground_truth.is_file():
    raise FileNotFoundError('Matching undistorted training RGB video is required for comparison')
paths=[Path(state['video_path']),ground_truth,render]
videos=[]
for path in paths:
    capture=cv2.VideoCapture(str(path))
    assert capture.isOpened(),path
    frames=[]
    while True:
        ok,frame=capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    fps=capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    videos.append({'frames':frames,'fps':fps})
counts=[len(video['frames']) for video in videos]
assert len(set(counts))==1 and counts[0]==state['annotation_frame_count'],counts
assert all(abs(video['fps']-10)<1e-3 for video in videos)
assert all(a.shape==b.shape for a,b in zip(videos[1]['frames'],videos[2]['frames']))
errors=[float(np.mean((a.astype(np.float32)-b.astype(np.float32))**2)) for a,b in zip(videos[1]['frames'],videos[2]['frames'])]
psnr=[float(10*np.log10(255**2/max(error,1e-9))) for error in errors]
assert np.std(videos[2]['frames'][counts[0]//2])>5,'Render is unexpectedly flat or empty'
selected=[0,counts[0]//2,counts[0]-1]
sheet=Image.new('RGB',(480*3,294*3))
for row,index in enumerate(selected):
    for col,video in enumerate(videos):
        tile=Image.fromarray(video['frames'][index]).resize((480,270))
        sheet.paste(tile,(col*480,row*294+24))
        draw=ImageDraw.Draw(sheet)
        label=['Retrieved raw RGB','Training RGB (undistorted)','New ADSceneGraph reconstruction'][col]
        draw.text((col*480+4,row*294+5),f'{label} | frame {index}',fill='white')
preview=directory/'comparison.jpg'
sheet.save(preview,quality=90)
report={'status':'render_checked','scene_id':state['scene_id'],'query':state['query'],
    'frame_count':counts[0],'fps':videos[2]['fps'],'new_checkpoint':state['checkpoint_path'],
    'rendered_video':str(render),'training_rgb_video':str(ground_truth),'preview':str(preview),
    'decoded_training_view_psnr_mean_db':float(np.mean(psnr)),
    'metric_scope':'Decoded videos at training camera viewpoints; not held-out or novel-view quality.',
    'manual_visual_review':'pending'}
write_json(directory/'render_review.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
