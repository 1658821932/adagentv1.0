"""Run identities, atomic states and append-only request records. No data hashing."""
import contextlib
import fcntl
import json
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import yaml

ROOT=Path(__file__).resolve().parents[1]
STORE=ROOT/'experiments/managed_runs/managed'
SENSITIVE_FIELDS={'api_key','apikey','authorization','password','access_token','refresh_token','secret','private_key'}


def experiments_root():
    return Path(os.environ.get('ADVERSARIAL_EXPERIMENTS_ROOT', ROOT/'experiments')).resolve()


def full_pipeline_root():
    return Path(os.environ.get('ADVERSARIAL_FULL_PIPELINE_ROOT', ROOT.parent/'experiments/full_pipeline')).resolve()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def no_credentials(value):
    if isinstance(value,dict):
        for key,child in value.items():
            if str(key).lower() in SENSITIVE_FIELDS:
                raise ValueError('Credentials belong in the private profile, not experiment configuration')
            no_credentials(child)
    elif isinstance(value,list):
        for child in value:
            no_credentials(child)


def within(path,parent):
    path=Path(path).resolve()
    if not path.is_relative_to(Path(parent).resolve()):
        raise ValueError('Path is outside the permitted experiment area')
    return path


def config_from(path):
    config=yaml.safe_load(Path(path).read_text())
    no_credentials(config)
    if config.get('schema_version')!=1:
        raise ValueError('Unsupported experiment config version')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',config['scene']['id']):
        raise ValueError('Scene id must be a simple directory name')
    subset=within(ROOT/config['scene']['subset'],ROOT)
    manifest=read_json(subset/'manifest.json')
    fps=float(config['scene']['approximate_fps'])
    if fps<=0 or fps!=manifest['approximate_fps']:
        raise ValueError('FPS must agree with the explicitly approximate subset')
    cameras=config['scene']['cameras']
    if not cameras or len(set(cameras))!=len(cameras) or any(c not in range(5) for c in cameras):
        raise ValueError('Invalid camera selection')
    p=config['planning']
    mode=p.get('decision_mode','fixed')
    source=p.get('trajectory_source','llm')
    if source not in {'llm','manual'} or (source=='manual' and mode!='fixed'):
        raise ValueError('Manual references require an explicitly fixed target and behavior')
    if source=='manual':
        if type(p.get('target_track_id')) is not int or not p.get('behavior') or not p.get('reference_limits'):
            raise ValueError('Manual reference requires target, behavior and explicit numerical limits')
        design=p['manual']
        if not 0<=design['normal_until_s']<design['lane_change_end_s']<=design['speed_recovery_s']<float(p['horizon_seconds']):
            raise ValueError('Manual phases must be ordered inside the observation window')
        if any(not math.isfinite(float(v)) for v in design.values()):
            raise ValueError('Manual trajectory parameters must be finite')
    if mode not in {'fixed','autonomous'}:
        raise ValueError('Unknown planning decision mode')
    if mode=='autonomous' and (p.get('target_track_id') is not None or p.get('behavior') is not None):
        raise ValueError('Autonomous runs must not preselect a target or behavior')
    count=int(round(float(p['horizon_seconds'])*fps))
    if count<2 or abs(count/fps-float(p['horizon_seconds']))>1e-6:
        raise ValueError('Planning horizon must have an integral sample count')
    if 'reference_limits' in p:
        from trajectory_checks import LIMITS
        limits=p['reference_limits']
        if set(limits)!=set(LIMITS) or any(type(v) not in {int,float} or not math.isfinite(v) or v<=0 for v in limits.values()):
            raise ValueError('Reference limits must specify all positive finite numerical screening thresholds')
    if p['analysis_frame']<0 or p['history_frames']<0 or p['analysis_frame']+count>=manifest['frame_count']:
        raise ValueError('Not enough frames for planner future ego lookup')
    dt=float(config['carla']['physics_dt'])
    if dt<=0 or abs(round(1/fps/dt)*dt-1/fps)>1e-6:
        raise ValueError('Output interval must be a multiple of physics_dt')
    if not 1<=config['llm']['max_attempts']<=5:
        raise ValueError('LLM max_attempts must be in 1..5')
    if config['training']['iterations']<=0:
        raise ValueError('Training iterations must be positive')
    stride=config['training'].get('colmap_frame_stride',1)
    if type(stride) is not int or stride<1:
        raise ValueError('COLMAP initialization frame stride must be a positive integer')
    # Freeze the full effective template, so later template edits cannot silently
    # change an already-created run or shared scene.
    template=yaml.safe_load((ROOT/config['training']['template']).read_text())
    no_credentials(template)
    config['training']['resolved_template']=template
    config['subset_manifest']=manifest
    return config


def provenance():
    result={}
    for name,path in [('project',ROOT),('carla_gs',ROOT/'CARLA-GS')]:
        try:
            result[name]={'commit':subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
                'dirty':bool(subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True))}
        except subprocess.CalledProcessError:
            result[name]={'commit':None}
    return result


class BusyError(RuntimeError):
    pass


@contextlib.contextmanager
def locked(folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.execution.lock').open('a+') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise BusyError('Another process is using this run/scene') from None
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


class StageStore:
    def __init__(self,folder):
        self.folder=Path(folder)
        self.path=self.folder/'state.json'

    def state(self):
        return read_json(self.path) if self.path.exists() else {'schema_version':1,'stages':{}}

    def execute(self,name,operation,validate):
        """Caller holds the run/scene lock for the entire workflow."""
        state=self.state()
        previous=state['stages'].get(name)
        if previous and previous['status']=='completed':
            try:
                validate()
            except Exception as error:
                previous.update(status='invalid',checked_at=utc(),error_type=type(error).__name__)
                atomic_json(self.path,state)
                raise RuntimeError(f'{name}: completed artifacts are no longer valid; inspect before resuming') from None
            return 'reused'
        if previous and previous['status']=='running':
            # We own the exclusive lock, so the previous process is no longer
            # executing this workflow. Preserve its attempt as interrupted.
            previous['status']='interrupted'
            previous['interrupted_at']=utc()
            old=Path(previous['attempt_dir'])/'status.json'
            atomic_json(old,previous)
        attempts=self.folder/'stages'/name
        attempts.mkdir(parents=True,exist_ok=True)
        attempt=attempts/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:10])
        attempt.mkdir()
        entry={'status':'running','started_at':utc(),'attempt_dir':str(attempt),'git':provenance()}
        state['stages'][name]=entry
        atomic_json(self.path,state);atomic_json(attempt/'status.json',entry)
        try:
            operation(attempt)
            validate()
        except BaseException as error:
            entry.update(status='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed',
                         finished_at=utc(),error_type=type(error).__name__)
            atomic_json(attempt/'status.json',entry);atomic_json(self.path,state)
            raise
        entry.update(status='completed',finished_at=utc())
        atomic_json(attempt/'status.json',entry);atomic_json(self.path,state)
        return 'completed'


class Experiment:
    def __init__(self,run):
        self.run=within(run,STORE/'runs')
        self.manifest=read_json(self.run/'run.json')
        self.config=self.manifest['config']
        self.scene=within(self.manifest['scene_dir'],STORE/'scenes')
        # Metadata alone must not redirect outputs outside the managed scene.
        self.data=self.scene/'data'
        self.model=self.scene/'model'
        self.train_config=self.scene/'train.yaml'
        self.stages=StageStore(self.run)
        self.lock_fds=()

    @property
    def reference_dir(self):
        return self.run/self.config['planning'].get('trajectory_source','llm')

    def planning(self):
        """Resolve autonomous choices without changing the frozen input config."""
        planning=dict(self.config['planning'])
        if planning.get('decision_mode','fixed')=='autonomous':
            analysis=read_json(self.run/'llm/analysis.json')
            candidates=read_json(self.run/'llm/candidates.json')['eligible_track_ids']
            target=analysis['selected_vehicle']
            if isinstance(target,bool) or not isinstance(target,int) or target not in candidates:
                raise ValueError('Selected target is not an eligible editable vehicle')
            if not isinstance(analysis.get('behavior_type'),str) or not analysis['behavior_type'].strip():
                raise ValueError('Missing model-selected behavior')
            if analysis.get('target_collision_zone') not in {'front','back','left','right'}:
                raise ValueError('Unknown model-selected collision zone')
            horizon=float(analysis['time_to_collision'])
            if abs(horizon-float(planning['horizon_seconds']))>1e-6:
                raise ValueError('Decision exceeds the fixed short-run planning horizon')
            planning.update(target_track_id=target,behavior=analysis['behavior_type'])
        return planning

    @classmethod
    def create(cls,config_path):
        config=config_from(config_path)
        scene=STORE/'scenes'/config['scene']['id']
        scene.mkdir(parents=True,exist_ok=True)
        spec={'scene':config['scene'],'training':config['training'],'subset_manifest':config['subset_manifest']}
        with locked(scene):
            if (scene/'scene.json').exists():
                if read_json(scene/'scene.json')['spec']!=spec:
                    raise ValueError('Scene id already belongs to another configuration; use a new scene id')
            else:
                atomic_json(scene/'scene.json',{'created_at':utc(),'spec':spec})
                training=dict(config['training']['resolved_template'])
                training['source_path']=str(scene/'data')
                training['model_path']=str(scene/'model')
                training['record_dir']=str(scene/'tensorboard')
                training['data']=dict(training['data'],selected_frames=[0,config['subset_manifest']['frame_count']-1],
                                      cameras=config['scene']['cameras'])
                n=config['training']['iterations']
                training['train']=dict(training['train'],iterations=n,save_iterations=[n],checkpoint_iterations=[n],test_iterations=[])
                (scene/'train.yaml').write_text(yaml.safe_dump(training,sort_keys=False,allow_unicode=True))
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:10]
        run=STORE/'runs'/run_id
        run.mkdir(parents=True)
        atomic_json(run/'run.json',{'schema_version':1,'run_id':run_id,'created_at':utc(),
            'scene_dir':str(scene),'config':config,'git':provenance()})
        (run/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
        (run/'README.md').write_text(
            '# Managed experiment '+run_id+'\n\n'
            'Frozen configuration: `config.yaml`; identity and Git provenance: `run.json`.\n'
            'Shared scene: `'+str(scene)+'`.\n'
            'State: `state.json`; each stage attempt has its own `stages/<stage>/<attempt>/` directory.\n'
            'LLM requests: `llm/requests/<request>/attempt_NNN/` (full messages, parameters, response and validation).\n'
            'Resume explicitly with `python scripts/experiment.py run --run '+str(run)+'`.\n'
            'Runtime artifacts are local-only and excluded from Git. API credentials are not archived.\n')
        if config['planning'].get('trajectory_source')=='manual':
            with (run/'README.md').open('a') as stream:
                stream.write('\nTrajectory source: human-designed analytic reference in `manual/`; this run skips all LLM stages and makes no API calls.\n')
        return cls(run)

    @classmethod
    def from_environment(cls):
        if not os.environ.get('RESEARCH_RUN_DIR'):
            raise RuntimeError('Use scripts/experiment.py; explicit run context is required')
        return cls(os.environ['RESEARCH_RUN_DIR'])

    def environment(self):
        return dict(os.environ,RESEARCH_RUN_DIR=str(self.run))


def redact(value,secrets):
    if isinstance(value,dict):
        return {k:('[REDACTED]' if str(k).lower() in SENSITIVE_FIELDS else redact(v,secrets)) for k,v in value.items()}
    if isinstance(value,list):
        return [redact(v,secrets) for v in value]
    if isinstance(value,str):
        for secret in secrets:
            if secret:
                value=value.replace(secret,'[REDACTED]')
    return value


class RequestArchive:
    def __init__(self,folder,client,profile,max_attempts=2):
        self.folder=Path(folder)
        self.client=client
        self.profile=profile
        self.max_attempts=max_attempts
        url=urlsplit(profile['base_url'])
        if url.scheme!='https' or url.username or url.password or url.query or url.fragment:
            raise ValueError('LLM endpoint must use HTTPS without embedded credentials/query')

    def complete(self,label,messages,**parameters):
        key=self.profile['api_key']
        if key and any(key in json.dumps(m,ensure_ascii=False) for m in messages):
            raise ValueError('Credential found in outgoing message; request blocked')
        request=self.folder/(label+'_'+uuid.uuid4().hex)
        request.mkdir(parents=True,exist_ok=False)
        for index in range(1,self.max_attempts+1):
            attempt=request/f'attempt_{index:03d}'
            attempt.mkdir()
            payload={'model':self.profile['model'],'messages':messages,**parameters}
            atomic_json(attempt/'request.json',redact({'started_at':utc(),'endpoint':self.profile['base_url'],
                'payload':payload},[key]))
            start=time.monotonic()
            try:
                response=self.client.chat.completions.create(**payload)
                safe=redact(response.model_dump(),[key])
                atomic_json(attempt/'response.json',safe)
                content=safe['choices'][0]['message'].get('content') or ''
                (attempt/'response.txt').write_text(content)
                atomic_json(attempt/'result.json',redact({'status':'received' if content.strip() else 'empty',
                    'finished_at':utc(),'elapsed_seconds':round(time.monotonic()-start,3),
                    'request_id':getattr(response,'_request_id',None),'finish_reason':safe['choices'][0].get('finish_reason')},[key]))
                if not content.strip():
                    if index<self.max_attempts:
                        continue
                    raise ValueError('LLM returned empty content')
                return content,attempt
            except Exception as error:
                # Preserve useful safe metadata, excluding exception strings,
                # response bodies and HTTP headers which can echo credentials.
                status=getattr(error,'status_code',None)
                request_id=getattr(error,'request_id',None)
                atomic_json(attempt/'error.json',redact({'error_type':type(error).__name__,
                    'http_status':status,'request_id':request_id,'finished_at':utc(),
                    'elapsed_seconds':round(time.monotonic()-start,3)},[key]))
                retry=status in {408,409,429,500,502,503,504} or type(error).__name__ in {'APIConnectionError','APITimeoutError'}
                if not retry or index==self.max_attempts:
                    raise RuntimeError(f'LLM request failed ({type(error).__name__}); see archived attempt') from None
        raise RuntimeError('No LLM response')
