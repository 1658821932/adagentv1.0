"""Candidate-level PDM checks with a bounded shared budget and actual path feedback."""
from pathlib import Path
from copy import deepcopy
import csv
import json
import time

import numpy as np

from experiment_runtime import atomic_json
from pdm_tool import run_pdm_challenge
from run_ego_reference import EGO_FRONT, EGO_REAR, EGO_WIDTH, polygon
from workflow_timing import timed_operation


class PDMProbe:
    def __init__(self, editor):
        self.editor = editor
        self.root = Path(editor.constraints['pdm_probe_root']).resolve()
        self.run_dir = self.root
        self.limit = editor.constraints.get('pdm_probe_budget',2)
        self.root.mkdir(parents=True,exist_ok=True)
        self.state_path = self.root/'budget.json'
        if not self.state_path.exists():
            cache=None
            evidence=editor.constraints.get('revision_feedback',{}).get('evidence',{})
            previous=evidence.get('result_path')
            if previous and Path(previous).is_file():
                review=json.loads(Path(previous).read_text())
                candidate=Path(review['pdm_output']) if review.get('pdm_output') else None
                if review.get('status')=='completed' and candidate and (candidate/'binding.json').is_file():
                    # The runner compares the complete baseline semantics before reuse.
                    cache=str(candidate)
            atomic_json(self.state_path,dict(calls=0,baseline_cache=cache,latest_feedback=None))
        self.results = {}

    def context(self):
        state=json.loads(self.state_path.read_text())
        return dict(enabled=True,calls=state['calls'],remaining=max(0,self.limit-state['calls']),
            current_candidate_scopes={key:str(self.editor.run_dir/'candidates'/key) for key in self.editor.nodes},
            untested_eligible_candidate_ids=[key for key,node in self.editor.nodes.items()
                if self._eligible(node) and key not in self.results],
            latest_feedback=state['latest_feedback'],
            candidates={key:self.public(value) for key,value in self.results.items()})

    @staticmethod
    def public(value):
        return {key:value[key] for key in ('status','candidate_id','candidate_scope','target_initial_state','pdm_defense_succeeded',
            'baseline_reused','elapsed_s','attack','closest_approach','trajectory_samples',
            'overlay_path','result_path','defense_feasibility','traffic_light_policy','takeover_alignment','background_geometry_policy','scope') if key in value}

    @staticmethod
    def _eligible(node):
        return node['result']['reference_screen_passed'] and (node['result'].get('construction_goal') is None
            or node['result']['construction_goal']['goal_met'])

    def require_reselection_evidence(self):
        eligible=[key for key,node in self.editor.nodes.items() if self._eligible(node)]
        if eligible and not any(key in self.results for key in eligible):
            raise ValueError('Current branch has eligible but untested candidates: '+', '.join(eligible)+
                '. Call probe_pdm here before reselection; prior-run/branch feedback is not a current test.')

    def require_challenge(self, candidate_id):
        result=self.results.get(candidate_id)
        if result is None or result.get('status')!='completed':
            raise ValueError('Call probe_pdm for this candidate before submitting; geometry alone does not test PDM')
        if result['pdm_defense_succeeded']:
            raise ValueError('PDM defended this candidate safely; revise/reselect within budget or finish_search')
        self._check_snapshot(candidate_id,result)
        return result

    def _check_snapshot(self,candidate_id,result):
        node=self.editor.nodes[candidate_id]
        expected=dict(target=self.editor.target,xy=node['xy'].tolist(),yaw=node['yaw'].tolist())
        if result['candidate_snapshot']!=expected:
            raise ValueError('PDM probe belongs to a different candidate version')

    @timed_operation('attack.probe_pdm',None)
    def probe(self,candidate_id):
        editor=self.editor
        node=editor.nodes[candidate_id]
        if not node['result']['reference_screen_passed']:
            raise ValueError('PDM probe requires all reference checks to pass first')
        goal=node['result'].get('construction_goal')
        if goal is not None and not goal['goal_met']:
            raise ValueError('PDM probe requires the fixed task contract to pass first')
        if candidate_id in self.results:
            self._check_snapshot(candidate_id,self.results[candidate_id])
            return dict(self.public(self.results[candidate_id]),cached=True)
        state=json.loads(self.state_path.read_text())
        if state['calls']>=self.limit:
            return dict(status='budget_exhausted',candidate_id=candidate_id,
                guidance='No PDM probes remain; do not submit an untested or safely defended candidate')
        state['calls']+=1
        atomic_json(self.state_path,state)
        folder=self.root/f"probe_{state['calls']:03d}"
        folder.mkdir(exist_ok=False)
        started=time.perf_counter()
        try:
            frozen=self._materialize(candidate_id,folder)
            arguments=dict(challenge_baseline_scene_path=str(editor.scene),challenge_attack_scene_path=str(frozen),
                challenge_target_track=editor.target['track_id'],challenge_target_is_new=editor.target['operation']=='add_vehicle',
                challenge_history_s=editor.constraints.get('observation_end_s',3.),
                challenge_ignore_traffic_lights=editor.constraints.get('pdm_ignore_traffic_lights',False))
            if state['baseline_cache']:
                arguments['challenge_baseline_cache_path']=state['baseline_cache']
            atomic_json(folder/'input.json',arguments)
            result=run_pdm_challenge(arguments,folder/'pdm',folder/'pdm.log')
            details=self._feedback(folder,node,result)
            if editor.constraints.get('pdm_defense_feedback'):
                details['defense_feasibility']=self._defense_feedback(frozen,folder)
            report=dict(status='completed',candidate_id=candidate_id,
                candidate_scope=str(editor.run_dir/'candidates'/candidate_id),
                target_initial_state=deepcopy(editor.target.get('initial_state')),
                candidate_snapshot=dict(target=deepcopy(editor.target),xy=node['xy'].tolist(),yaw=node['yaw'].tolist()),
                frozen_scene_path=str(frozen),pdm_output=str(folder/'pdm'),result_path=str(folder/'result.json'),
                pdm_defense_succeeded=result['pdm_defense_succeeded'],
                baseline_reused=result['summary'].get('baseline_reused',False),
                elapsed_s=round(time.perf_counter()-started,3),
                traffic_light_policy=result['summary'].get('traffic_light_policy'),
                takeover_alignment=result['summary'].get('takeover_alignment'),
                background_geometry_policy=result['summary'].get('background_geometry_policy'),**details,
                scope='WE PDM + nuPlan LQR/kinematic rollout; fixed history and replayed background; not rigid-body physics or lane-line certification')
            atomic_json(folder/'result.json',report)
            self.results[candidate_id]=report
            state['baseline_cache']=str(folder/'pdm')
            state['latest_feedback']=self.public(report)
            atomic_json(self.state_path,state)
            return self.public(report)
        except Exception as error:
            atomic_json(folder/'error.json',dict(status='execution_failed',error_type=type(error).__name__,
                error=str(error)[:500],elapsed_s=time.perf_counter()-started))
            raise RuntimeError('PDM probe execution failed; inspect '+str(folder/'error.json')) from error

    def _defense_feedback(self, frozen, folder):
        from defense_probe_feedback import collect
        settings=json.loads((Path(__file__).resolve().parent/'backend_settings.json').read_text())
        try:
            return collect(frozen,self.editor.target['track_id'],
                self.editor.constraints.get('acceptance_policy'),folder/'defense_constructor',
                settings['defaults']['defense_constructor_candidates'])
        except Exception as error:
            # Optional planning evidence must not turn an execution error into 'no defense'.
            evidence=dict(status='unavailable',error_type=type(error).__name__,error=str(error)[:350],
                diagnostic_only=True,attack_acceptance_affected=False,global_feasibility_proven=False)
            atomic_json(folder/'defense-feedback-error.json',evidence)
            return evidence

    def _materialize(self,candidate_id,folder):
        editor=self.editor
        node=editor.nodes[candidate_id]
        if editor.target['operation']=='add_vehicle':
            from add_vehicle_scene import add_vehicle
            add_vehicle(editor.scene,folder/'materialized',0,editor.target['track_id'],editor.target['size_m'],
                trajectory_xy=node['xy'],trajectory_yaw=node['yaw'],
                official_min_margin_m=editor.constraints['official_margin_m'])
        else:
            from freeze_attack_scene import freeze
            freeze(editor.scene,editor.run_dir,candidate_id,editor.target['track_id'],folder/'materialized/frozen_scene')
        return folder/'materialized/frozen_scene'

    def _feedback(self,folder,node,result):
        def read(path):
            with path.open() as stream:
                return [{key:float(value) for key,value in row.items() if key!='source'} for row in csv.DictReader(stream)]
        baseline_path=folder/'pdm/baseline/trajectory.csv'
        baseline=read(baseline_path) if baseline_path.is_file() else []
        attack=read(folder/'pdm/attack/trajectory.csv')
        diagnostics=json.loads((folder/'pdm/attack/diagnostics.json').read_text())
        closest=min((row for row in diagnostics if row['target_gap_m'] is not None),key=lambda row:row['target_gap_m'])
        ego=min(attack,key=lambda row:abs(row['time_s']-closest['time_s']))
        index=closest['observation_frame']
        target_xy=node['xy'][index]
        heading=ego['ego_yaw_rad']
        center=np.array([ego['ego_x_m'],ego['ego_y_m']])+(EGO_FRONT-EGO_REAR)/2*np.array([np.cos(heading),np.sin(heading)])
        delta=target_xy-center
        approach=dict(time_s=closest['time_s'],target_gap_m=closest['target_gap_m'],ego_rear_pose={
            key:ego[key] for key in ('ego_x_m','ego_y_m','ego_yaw_rad','speed_mps')},
            target_center_xy_m=target_xy.tolist(),target_yaw_rad=float(node['yaw'][index]),
            target_relative_to_pdm_body_center=dict(forward_m=float(delta@np.array([np.cos(heading),np.sin(heading)])),
                left_m=float(delta@np.array([-np.sin(heading),np.cos(heading)]))))
        def samples(rows):
            if not rows:
                return []
            return [{key:rows[i][key] for key in ('time_s','ego_x_m','ego_y_m','ego_yaw_rad','speed_mps')}
                for i in sorted(set(np.linspace(0,len(rows)-1,10,dtype=int)))]
        plot=self._plot(folder,node,baseline,attack,ego,index)
        metrics=result['summary']['results']['attack']
        return dict(attack={key:metrics[key] for key in ('contact_samples','first_contact','road_violation_samples',
            'min_gap_m','min_target_gap_m','final_speed_mps')},closest_approach=approach,
            trajectory_samples=dict(position_reference='ego_rear_axle',coordinate_frame='initial_ego',
                body_center_shift_m=(EGO_FRONT-EGO_REAR)/2,ego_length_m=EGO_FRONT+EGO_REAR,ego_width_m=EGO_WIDTH,
                pdm_baseline=samples(baseline),pdm_attack=samples(attack)),overlay_path=str(plot))

    def _plot(self,folder,node,baseline,attack,ego,index):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        editor=self.editor
        def centers(rows):
            heading=np.array([row['ego_yaw_rad'] for row in rows])
            return np.array([[r['ego_x_m'],r['ego_y_m']] for r in rows])+(EGO_FRONT-EGO_REAR)/2*np.column_stack([np.cos(heading),np.sin(heading)])
        recorded=editor.ego+(EGO_FRONT-EGO_REAR)/2*np.column_stack([np.cos(editor.ego_yaw),np.sin(editor.ego_yaw)])
        bp=centers(baseline) if baseline else np.empty((0,2))
        ap=centers(attack)
        fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
        target=node['xy'][index]
        for ax in axes:
            from shapely.plotting import plot_polygon
            plot_polygon(editor.road.union,ax=ax,add_points=False,facecolor='#eeeeee',
                edgecolor='#bbbbbb',linewidth=.6,zorder=0)
            ax.plot(*recorded.T,'--',color='#777777',label='Recorded ego body center')
            if baseline:
                ax.plot(*bp.T,color='#377eb8',linewidth=3,label='Diagnostic PDM without attack')
            ax.plot(*ap.T,color='#ff7f00',linewidth=1.5,label='PDM with attack')
            ax.plot(*node['xy'].T,color='#e41a1c',label='Target center trajectory')
            shape=polygon(*target,node['yaw'][index],*editor.target['size_m'][:2]);x,y=shape.exterior.xy
            ax.fill(x,y,color='#e41a1c',alpha=.5,label='Target at closest approach')
            h=ego['ego_yaw_rad'];c=np.array([ego['ego_x_m'],ego['ego_y_m']])+(EGO_FRONT-EGO_REAR)/2*np.array([np.cos(h),np.sin(h)])
            shape=polygon(*c,h,EGO_FRONT+EGO_REAR,EGO_WIDTH);x,y=shape.exterior.xy
            ax.plot(x,y,color='#984ea3',linewidth=2,label='Ego at closest approach')
            ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('Initial ego frame x [m]');ax.set_ylabel('y [m]')
        all_xy=np.vstack([recorded,bp,ap,node['xy']]);lo=all_xy.min(axis=0)-5;hi=all_xy.max(axis=0)+5
        axes[0].set_xlim(lo[0],hi[0]);axes[0].set_ylim(lo[1],hi[1]);axes[0].set_title('Global trajectories')
        axes[1].set_xlim(target[0]-10,target[0]+10);axes[1].set_ylim(target[1]-10,target[1]+10)
        axes[1].set_title(f"Closest approach at {ego['time_s']:.2f} s")
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
        fig.suptitle('PDM candidate self-check: shaded area is drivable coverage, not lane-line certification',fontsize=10)
        path=folder/'pdm-overlay.png';fig.savefig(path,dpi=130);plt.close(fig)
        return path
