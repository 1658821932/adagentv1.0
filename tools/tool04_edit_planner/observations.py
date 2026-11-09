"""Temporal contact sheets and bounded, model-requested extra-camera review."""
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from tools.llm_runtime import parse_json_object
from .schemas import ViewRequest

ROOT=Path(__file__).resolve().parents[2]
LOG_ROOT=ROOT/'logs/tool04_observations'
ACTIVE_OBSERVATION=ContextVar('edit_planner_visual_observation',default=None)


def settings():
    value=json.loads(Path(__file__).with_name('observation.json').read_text())
    for name,lower,upper in [('time_samples',2,48),('frames_per_sheet',1,8),('columns',1,4),
        ('tile_width',160,1280),('tile_height',90,720),('label_height',20,80),
        ('max_extra_rounds',0,3),('max_cameras_per_request',1,2)]:
        if type(value.get(name)) is not int or not lower<=value[name]<=upper:
            raise ValueError('Invalid visual observation setting: '+name)
    if not isinstance(value.get('initial_camera'),str) or not value['initial_camera']:
        raise ValueError('initial_camera is required')
    return value


def sample_indices(count,limit):
    if count<=limit:return list(range(count))
    return [round(i*(count-1)/(limit-1)) for i in range(limit)]


def source_entries(ref,frames,indices,camera,binding):
    result=[]
    for index in indices:
        frame=frames[index];info=frame.get('cams',{}).get(camera,{})
        name=next((info.get(k) for k in ['data_path','image_path','filename','image_file'] if info.get(k)),None)
        possible=[];bound=binding.get('cameras',{}).get(camera,{}).get('frames',[])
        if index<len(bound) and bound[index].get('timestamp_us')==frame['timestamp'] and bound[index].get('image_path'):
            possible.append(Path(bound[index]['image_path']))
        if isinstance(name,str):
            path=Path(name).expanduser()
            possible += [path] if path.is_absolute() else [Path(ref.sample_root)/'original_images'/path,Path(ref.sample_root)/path]
        path=next((p for p in possible if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}),None)
        result.append(dict(camera=camera,frame_index=index,timestamp=frame['timestamp'],
            time_s=round((frame['timestamp']-frames[0]['timestamp'])/1e6,3),image_path=str(path) if path else None))
    return result


def compose(entries,folder,cfg):
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    folder.mkdir(parents=True,exist_ok=True)
    width,height,label=cfg['tile_width'],cfg['tile_height'],cfg['label_height']
    per,columns=cfg['frames_per_sheet'],cfg['columns']
    rows=(per+columns-1)//columns
    try:font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    sheets=[]
    for offset in range(0,len(entries),per):
        batch=entries[offset:offset+per]
        board=Image.new('RGB',(columns*width,rows*(height+label)),(25,25,25));draw=ImageDraw.Draw(board)
        for index,item in enumerate(batch):
            x=(index%columns)*width;y=(index//columns)*(height+label)
            title=f"{item['camera']}  frame={item['frame_index']}  t={item['time_s']:.3f}s"
            draw.text((x+8,y+5),title,font=font,fill='white')
            if item['image_path']:
                with Image.open(item['image_path']) as source:
                    tile=ImageOps.contain(ImageOps.exif_transpose(source).convert('RGB'),(width,height),Image.Resampling.LANCZOS)
                board.paste(tile,(x+(width-tile.width)//2,y+label+(height-tile.height)//2))
            else:draw.text((x+12,y+label+20),'SOURCE FRAME MISSING',font=font,fill=(255,150,100))
        target=folder/f'sheet_{offset//per+1:02d}.jpg'
        board.save(target,quality=88)
        sheets.append(dict(path=str(target),camera=batch[0]['camera'],frame_indices=[v['frame_index'] for v in batch],
            available_frames=sum(v['image_path'] is not None for v in batch),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),delivered=False))
    return sheets


class Observation:
    def __init__(self,data,summary):
        self.data,self.summary=data,summary
        self.cfg=dict(summary['evidence']['visual_settings'])
        self.frames=json.loads(Path(data.scene_ref.frames_path).read_text())
        manifest=Path(data.scene_ref.manifest_path)
        self.binding=json.loads(manifest.read_text()) if manifest.is_file() else {}
        self.indices=summary['evidence']['frame_indices']
        self.available=sorted(set().union(*(set(frame.get('cams',{})) for frame in self.frames)))
        self.directory=LOG_ROOT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')+uuid4().hex[:10])
        self.directory.mkdir(parents=True,exist_ok=False)
        self.records=[];self.view_requests=[];self.model_calls=0
        summary['evidence']['visual_observation']=dict(record_dir=str(self.directory),initial_camera=self.cfg['initial_camera'],
            available_cameras=self.available,selected_frame_indices=self.indices,sheets=self.records,
            view_requests=self.view_requests,model_calls=0)
        self.initial=self.add_camera(self.cfg['initial_camera'])
        self.save()

    def save(self):
        evidence=self.summary['evidence']['visual_observation'];evidence['model_calls']=self.model_calls
        path=self.directory/'observation.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(scene_id=self.data.scene_ref.scene_id,scene_token=self.data.scene_ref.scene_token,
            settings=self.cfg,observation=evidence),ensure_ascii=False,indent=2));temporary.replace(path)

    def add_camera(self,camera):
        entries=source_entries(self.data.scene_ref,self.frames,self.indices,camera,self.binding)
        if not any(item['image_path'] for item in entries):
            return dict(camera=camera,available_frames=0,tool_error='No selected source images are available for this camera'),[]
        folder=self.directory/f'view_{len(self.records):02d}'
        sheets=compose(entries,folder,self.cfg)
        self.records.extend(sheets)
        # Retain raw source references for audit, including supplemental cameras.
        keys={(e['camera'],e['frame_index']) for e in self.summary['evidence']['keyframes']}
        for entry in entries:
            if entry['image_path'] and (entry['camera'],entry['frame_index']) not in keys:
                self.summary['evidence']['keyframes'].append(entry)
                self.summary['evidence']['source_images'].append(entry['image_path'])
        return dict(camera=camera,available_frames=sum(e['image_path'] is not None for e in entries),
                    frame_indices=self.indices,sheet_count=len(sheets)),sheets

    def content(self,review):
        information,sheets=review
        content=[dict(type='text',text=json.dumps(information,ensure_ascii=False))]
        for sheet in sheets:
            content.append(dict(type='text',text=json.dumps(dict(camera=sheet['camera'],frame_indices=sheet['frame_indices']),ensure_ascii=False)))
            content.append(dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(Path(sheet['path']).read_bytes()).decode())))
        return content

    def initial_content(self):
        self.summary['visual_review_options']=dict(available_cameras=self.available,max_extra_rounds=self.cfg['max_extra_rounds'],
            max_cameras_per_request=self.cfg['max_cameras_per_request'],example=dict(action='request_views',cameras=['CAM_L0'],purpose='Check the side entry geometry'))
        return [dict(type='text',text=json.dumps(self.summary,ensure_ascii=False))]+self.content(self.initial)

    def delivered(self):
        for sheet in self.records:sheet['delivered']=True
        self.save()


class ViewClient:
    def __init__(self,client,observation):
        if observation is None:raise ValueError('Visual observation was not prepared')
        self.client,self.observation=client,observation
        self.model=getattr(client,'model','injected-client');self.exchanges=[];self.rounds=0

    @property
    def last_usage(self):return getattr(self.client,'last_usage',None)

    def chat(self,messages):
        history=deepcopy(messages[:2])+deepcopy(self.exchanges)+deepcopy(messages[2:])
        while True:
            self.observation.model_calls+=1;self.observation.save()
            response=self.client.chat(deepcopy(history))
            self.observation.delivered()
            try:action=parse_json_object(response)
            except (ValueError,TypeError):return response
            if action.get('action')!='request_views':return response
            if self.rounds>=self.observation.cfg['max_extra_rounds']:
                return json.dumps(dict(suitability='needs_information',reasons=['Additional visual review budget exhausted; more views are needed.'],edit_plan=None,retrieval_suggestion=None))
            self.rounds+=1
            try:
                request=ViewRequest.model_validate(action)
                if len(set(request.cameras))!=len(request.cameras):raise ValueError('Request each camera only once')
                if len(request.cameras)>self.observation.cfg['max_cameras_per_request']:raise ValueError('Too many cameras in one request')
                unknown=set(request.cameras)-set(self.observation.available)
                if unknown:raise ValueError('Unknown source cameras: '+', '.join(sorted(unknown)))
                seen={sheet['camera'] for sheet in self.observation.records if sheet['delivered']}
                if set(request.cameras)&seen:raise ValueError('These cameras were already delivered; use the existing images')
                reviews=[self.observation.add_camera(camera) for camera in request.cameras]
                content=[dict(type='text',text=json.dumps(dict(purpose=request.purpose,remaining_extra_rounds=self.observation.cfg['max_extra_rounds']-self.rounds),ensure_ascii=False))]
                for review in reviews:content.extend(self.observation.content(review))
                self.observation.view_requests.append(dict(cameras=request.cameras,purpose=request.purpose,round=self.rounds))
            except (ValueError,TypeError,KeyError) as error:
                self.observation.view_requests.append(dict(request=action,round=self.rounds,status='invalid',error=str(error)))
                content=[dict(type='text',text=json.dumps(dict(tool_error=str(error),available_cameras=self.observation.available,
                    remaining_extra_rounds=self.observation.cfg['max_extra_rounds']-self.rounds),ensure_ascii=False))]
            pair=[dict(role='assistant',content=response),dict(role='user',content=content)]
            self.exchanges.extend(pair);history.extend(deepcopy(pair));self.observation.save()
