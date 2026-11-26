"""Editor-owned VADv2 candidate tests through the shared native WE backend."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from backend_hooks import native_policy_horizon, native_policy_probe, start_prewarm
from box_geometry import corners
from experiment_runtime import atomic_json


class VADProbe:
    def __init__(self,editor):
        self.editor=editor
        self.root=Path(editor.constraints['vadv2_probe_root'])
        self.root.mkdir(parents=True,exist_ok=True)
        self.limit=editor.constraints.get('vadv2_probe_budget',3)
        self.policy=editor.constraints['vadv2_policy_config']
        self.prewarm=start_prewarm(self.policy,self.root/'prewarm.json')
        self.state_path=self.root/'budget.json'
        if not self.state_path.exists():atomic_json(self.state_path,dict(calls=0,latest_feedback=None))
        self.results={}

    @staticmethod
    def _eligible(node):
        from attack_tool_contract import candidate_eligibility
        return bool(candidate_eligibility(node['result'])['pdm_probe_eligible'])

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
            'target_collision_detected','policy_attack_success','policy_evaluation','first_target_collision_time_s','min_target_clearance_m',
            'closest_approach','initial_history_target_contact','possible_interpolated_target_contact',
            'policy_behavior_summary','evaluation_window_s','scene_feedback',
            'elapsed_s','overlay_path','result_path','scope') if key in value}

    def snapshot(self,candidate_id):
        node=self.editor.nodes[candidate_id]
        return dict(target=deepcopy(self.editor.target),xy=node['xy'].tolist(),yaw=node['yaw'].tolist())

    def require_challenge(self,candidate_id):
        result=self.results.get(candidate_id)
        if result is None or result.get('status')!='completed':
            raise ValueError('Call probe_vadv2 on this exact candidate before submission')
        if result.get('policy_attack_success') is not True:
            raise ValueError('VADv2 test has no confirmed target collision or drivable-area violation for this candidate')
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
        from ego_history import attach
        frozen=folder/'materialized/frozen_scene'
        attach(editor.scene,frozen)
        return frozen

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
        history_frames, rollout_frames = native_policy_horizon()
        if len(editor.times) < rollout_frames:
            raise ValueError(f'VADv2 self-test needs {rollout_frames} source frames for the 1.5–7.5s window')
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
                collision_target_track=editor.target['track_id'],num_scenarios=1,num_history=history_frames,
                max_frame_per_scene=rollout_frames,num_future=rollout_frames-history_frames,
                collision_only=True,visualize_video=False,visualize_BEV=False,
                with_dense_reward_manager=False,timeout=policy.get('timeout_seconds',600),**simulation)
            atomic_json(folder/'input.json',request)
            response = native_policy_probe(policy, request, folder/'native-policy.log')
            if response.get('status')!='success':raise RuntimeError('Native self-test returned no successful result')
            native=response['result'];collision=native.get('collision_report')
            if collision is None or collision['status']!='completed' or collision['target_collision_detected'] is None:
                raise RuntimeError('Native self-test collision evidence is incomplete; inspect '+str(folder/'native-policy.log'))
            from vad_scene_feedback import feedback as scene_feedback
            from vad_policy_evaluation import evaluate_policy, compact_policy_evaluation
            fixed=scene_feedback(editor,collision)
            atomic_json(folder/'scene-feedback.json',fixed)
            evaluation=evaluate_policy(collision,fixed,frozen/'metadata/map_drivable_official.json')
            atomic_json(folder/'policy-evaluation.json',evaluation)
            overlay=self._plot(collision,folder)
            result=dict(self.public(collision),status='completed',candidate_id=candidate_id,
                policy_attack_success=evaluation['policy_attack_success'],
                policy_evaluation=compact_policy_evaluation(evaluation),
                policy_behavior_summary=self._compact_behavior(native.get('policy_behavior_summary')),
                scene_feedback={key:value for key,value in fixed.items() if key!='full_precision_states'},
                evaluation_window_s=[1.5,7.5],
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

    @staticmethod
    def _compact_behavior(value):
        if value is None:return None
        keys=('status','behavior_classification','absolute_progress_m','progress_ratio_vs_recorded',
              'initial_speed_mps','final_speed_mps','stopped_detected','stopped_from_time_s',
              'signed_yaw_change_deg')
        return {key:deepcopy(value[key]) for key in keys if key in value}

    def _plot(self,collision,folder):
        from vad_scene_feedback import feedback,plot
        fixed=feedback(self.editor,collision)
        if fixed['status']=='measured':return plot(self.editor,collision,fixed,folder)
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
            for track in ('ego',target):
                points=corners(frame['actors'][track]['box']);points.append(points[0])
                axis.plot([p[0] for p in points],[p[1] for p in points],color='black')
        axis.set_aspect('equal',adjustable='datalim');axis.grid(alpha=.3);axis.legend()
        axis.set(xlabel='WE local world x (m)',ylabel='WE local world y (m)',
            title=f"Actual VADv2 vs attack: contact={collision['target_collision_detected']}, gap={collision['min_target_clearance_m']:.2f}m")
        path=folder/'vadv2-feedback.png';figure.tight_layout();figure.savefig(path,dpi=130);plt.close(figure)
        return path
