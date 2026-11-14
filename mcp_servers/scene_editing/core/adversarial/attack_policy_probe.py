"""Editor-owned VADv2 candidate tests through the shared native WE backend."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT=Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from mcp_servers.runtime import write_json as atomic_json


class VADProbe:
    def __init__(self,editor):
        self.editor=editor
        self.root=Path(editor.constraints['vadv2_probe_root'])
        self.root.mkdir(parents=True,exist_ok=True)
        self.limit=editor.constraints.get('vadv2_probe_budget',3)
        self.policy=editor.constraints['vadv2_policy_config']
        self.state_path=self.root/'budget.json'
        if not self.state_path.exists():atomic_json(self.state_path,dict(calls=0,latest_feedback=None))
        self.results={}

    @staticmethod
    def _eligible(node):
        result=node['result'];goal=result.get('construction_goal')
        return result['reference_screen_passed'] and (goal is None or goal['goal_met'])

    def context(self):
        state=json.loads(self.state_path.read_text())
        return dict(enabled=True,calls=state['calls'],remaining=max(0,self.limit-state['calls']),
            latest_feedback=state['latest_feedback'],
            untested_eligible_candidate_ids=[key for key,node in self.editor.nodes.items()
                if self._eligible(node) and key not in self.results],
            candidates={key:self.public(value) for key,value in self.results.items()})

    @staticmethod
    def public(value):
        return {key:deepcopy(value[key]) for key in ('status','candidate_id','candidate_scope',
            'target_collision_detected','policy_attack_success','first_target_collision_time_s','min_target_clearance_m',
            'closest_approach','initial_history_target_contact','possible_interpolated_target_contact',
            'policy_behavior_summary','pdm_subscores',
            'elapsed_s','overlay_path','result_path','scope') if key in value}

    def snapshot(self,candidate_id):
        node=self.editor.nodes[candidate_id]
        return dict(target=deepcopy(self.editor.target),xy=node['xy'].tolist(),yaw=node['yaw'].tolist())

    def require_challenge(self,candidate_id):
        result=self.results.get(candidate_id)
        if result is None or result.get('status')!='completed':
            raise ValueError('Call probe_vadv2 on this exact candidate before submission')
        if result.get('policy_attack_success') is not True:
            raise ValueError('VADv2 has not hit the attack target in this candidate; edit within budget or finish_search')
        if result['candidate_snapshot']!=self.snapshot(candidate_id):
            raise ValueError('VADv2 test belongs to a different candidate version')
        return result

    def require_reselection_evidence(self):
        eligible=[key for key,node in self.editor.nodes.items() if self._eligible(node)]
        if eligible and not any(key in self.results for key in eligible):
            raise ValueError('Test an eligible current candidate with probe_vadv2 before object reselection')

    def _materialize(self,candidate_id,folder):
        editor=self.editor;node=editor.nodes[candidate_id]
        if editor.target['operation']=='add_vehicle':
            from add_vehicle_scene import add_vehicle
            detail=add_vehicle(editor.scene,folder/'materialized',0,editor.target['track_id'],editor.target['size_m'],
                trajectory_xy=node['xy'],trajectory_yaw=node['yaw'],official_min_margin_m=editor.constraints['official_margin_m'])
            detail.update(asset_path=editor.target['asset_path'],candidate_id=candidate_id)
            atomic_json(folder/'materialized/frozen_scene/freeze-info.json',detail)
        else:
            from freeze_attack_scene import freeze
            freeze(editor.scene,editor.run_dir,candidate_id,editor.target['track_id'],folder/'materialized/frozen_scene')
        return folder/'materialized/frozen_scene'

    def probe(self,candidate_id):
        editor=self.editor
        if not self._eligible(editor.nodes[candidate_id]):
            raise ValueError('VADv2 testing requires passing trajectory and task checks first')
        if candidate_id in self.results:
            result=self.results[candidate_id]
            if result['candidate_snapshot']!=self.snapshot(candidate_id):
                raise ValueError('Cached VADv2 evidence has a different candidate version')
            return dict(self.public(result),cached=True)
        state=json.loads(self.state_path.read_text())
        if state['calls']>=self.limit:
            return dict(status='budget_exhausted',candidate_id=candidate_id)
        state['calls']+=1;atomic_json(self.state_path,state)
        folder=self.root/f"probe_{state['calls']:03d}";folder.mkdir(exist_ok=False)
        started=time.perf_counter()
        try:
            frozen=self._materialize(candidate_id,folder)
            from native_scene_export import export
            exported=export(editor.constraints['policy_source_scene_ref'],frozen,folder/'native_attack',
                attack_result=dict(operation=editor.target['operation'],target_track=editor.target['track_id']))
            simulation=exported.get('simulation_input')
            if simulation is None:raise RuntimeError('VADv2 self-test has no native attack package; inspect '+str(folder/'native_attack'))
            policy=self.policy
            request=dict(execution_mode='native',config_path=policy['config_path'],checkpoint_path=policy['checkpoint_path'],
                model_name=policy.get('model_name','attack_vadv2'),data_type='nuplan_openscene',react_type='NR',
                collision_target_track=editor.target['track_id'],num_scenarios=1,num_history=policy.get('num_history',4),
                max_frame_per_scene=len(editor.times),num_future=len(editor.times)-policy.get('num_history',4),
                with_dense_reward_manager=False,timeout=policy.get('timeout_seconds',600),**simulation)
            atomic_json(folder/'input.json',request)
            root=Path(__file__).resolve().parents[4]
            import sys
            if str(root) not in sys.path:sys.path.insert(0,str(root))
            from mcp_servers.process import run_logged
            env=dict(os.environ,WORLDENGINE_ROOT=policy['worldengine_root'])
            completed=run_logged([policy['python'],'-m','mcp_servers.worldengine.adapter.policy_probe_runner'],
                input=json.dumps(request),cwd=root,env=env,log_path=folder/'native-policy.log',
                json_stdout=True,timeout=request['timeout']+60)
            if completed.returncode:
                raise RuntimeError('Native VADv2 self-test failed; inspect '+str(folder/'native-policy.log'))
            response=json.loads(completed.stdout)
            if response.get('status')!='success':raise RuntimeError('Native self-test returned no successful result')
            native=response['result'];collision=native.get('collision_report')
            if collision is None or collision['status']!='completed' or collision['target_collision_detected'] is None:
                raise RuntimeError('Native self-test collision evidence is incomplete; inspect '+str(folder/'native-policy.log'))
            overlay=self._plot(collision,folder)
            result=dict(self.public(collision),status='completed',candidate_id=candidate_id,
                policy_attack_success=collision['target_collision_detected'] is True and not collision['initial_history_target_contact'],
                policy_behavior_summary=native.get('policy_behavior_summary'),
                pdm_subscores=native.get('pdm_subscores'),
                candidate_scope=str(editor.run_dir/'candidates'/candidate_id),
                candidate_snapshot=self.snapshot(candidate_id),frozen_scene_path=str(frozen),
                frozen_frames_sha256=hashlib.sha256((frozen/'metadata/frames.json').read_bytes()).hexdigest(),
                native_result=native,collision_report=collision,overlay_path=str(overlay),
                elapsed_s=round(time.perf_counter()-started,3),result_path=str(folder/'result.json'))
            atomic_json(folder/'result.json',result)
            self.results[candidate_id]=result
            state['latest_feedback']=self.public(result);atomic_json(self.state_path,state)
            return self.public(result)
        except Exception as error:
            atomic_json(folder/'error.json',dict(status='execution_failed',error_type=type(error).__name__,
                error=str(error)[:600],elapsed_s=time.perf_counter()-started))
            raise RuntimeError('VADv2 self-test execution failed; inspect '+str(folder/'error.json')) from error

    def _plot(self,collision,folder):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        frames=[json.loads(line) for line in Path(collision['states_path']).read_text().splitlines() if line]
        target=self.editor.target['track_id']
        figure,axis=plt.subplots(figsize=(9,6))
        for track,label,color in (('ego','VADv2 actual body centre','tab:blue'),(target,'Attack vehicle','tab:red')):
            points=[f['actors'][track]['box'][:2] for f in frames if track in f['actors']]
            axis.plot([p[0] for p in points],[p[1] for p in points],'.-',label=label,color=color)
            for frame in frames[::4]:
                if track in frame['actors']:
                    p=frame['actors'][track]['box'];axis.annotate(f"{frame['time_s']:.1f}s",p[:2],fontsize=8)
        closest=collision.get('closest_approach')
        if closest:
            frame=next(f for f in frames if f['step']==closest['step'])
            import sys
            root=Path(__file__).resolve().parents[4]
            if str(root) not in sys.path:sys.path.insert(0,str(root))
            from mcp_servers.worldengine.adapter.collision_report import corners
            for track in ('ego',target):
                points=corners(frame['actors'][track]['box']);points.append(points[0])
                axis.plot([p[0] for p in points],[p[1] for p in points],color='black')
        axis.set_aspect('equal',adjustable='datalim');axis.grid(alpha=.3);axis.legend()
        axis.set(xlabel='WE local world x (m)',ylabel='WE local world y (m)',
            title=f"Actual VADv2 vs attack: contact={collision['target_collision_detected']}, gap={collision['min_target_clearance_m']:.2f}m")
        path=folder/'vadv2-feedback.png';figure.tight_layout();figure.savefig(path,dpi=130);plt.close(figure)
        return path
