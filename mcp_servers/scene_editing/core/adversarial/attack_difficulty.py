"""Explicit attack difficulty modes and bounded, stationary path placement search.

Numerical attack generation and reference defense witnesses are separate from Agent runs.
"""
import argparse
import copy
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import acceptance_policy as policy
from attack_editor import AttackEditor,write_trajectory
from experiment_runtime import ROOT,atomic_json
from run_ego_reference import ego_path_from_frames


MODES=('free','defendable','curriculum')


def resolve_difficulty(value=None):
    if value is None:
        return dict(schema_version='attack_difficulty.v1',mode='free',source='default_existing_attack_behavior')
    if not isinstance(value,dict) or set(value)-{'schema_version','mode','source','placement_search','budget'}:
        raise ValueError('Unknown attack difficulty fields')
    result=copy.deepcopy(value)
    if result.get('schema_version')!='attack_difficulty.v1' or result.get('mode') not in MODES:
        raise ValueError('Explicit difficulty requires attack_difficulty.v1 and free/defendable/curriculum')
    if not isinstance(result.get('source'),str) or not result['source'].strip():
        raise ValueError('Explicit difficulty mode requires its configuration source')
    if result['mode']=='free' and 'placement_search' not in result and 'budget' not in result:
        return result
    search=result.get('placement_search',{})
    if set(search)!={'anchor_xy_m','anchor_yaw_rad','min_forward_m','max_forward_m','step_m'}:
        raise ValueError('placement_search requires anchor_xy_m, anchor_yaw_rad and min/max_forward_m, step_m')
    anchor=np.asarray(search['anchor_xy_m'],dtype=float)
    if anchor.shape!=(2,) or not np.isfinite(anchor).all():
        raise ValueError('anchor_xy_m must contain two finite coordinates')
    for key in ('anchor_yaw_rad','min_forward_m','max_forward_m','step_m'):
        if type(search[key]) not in (float,int) or not math.isfinite(search[key]):
            raise ValueError('Placement values must be finite numbers')
    if search['min_forward_m']<0 or search['max_forward_m']<search['min_forward_m'] or search['step_m']<=0:
        raise ValueError('Placement offsets must be ordered nonnegative and step positive')
    budget=result.get('budget',{})
    if set(budget)!={'max_positions','max_defense_candidates','per_probe_seconds','total_seconds'}:
        raise ValueError('Difficulty search requires explicit position/candidate/time budgets')
    for key,maximum in (('max_positions',64),('max_defense_candidates',64)):
        if type(budget[key]) is not int or not 1<=budget[key]<=maximum:
            raise ValueError('Invalid bounded integer budget: '+key)
    for key in ('per_probe_seconds','total_seconds'):
        if type(budget[key]) not in (float,int) or not math.isfinite(budget[key]) or budget[key]<=0:
            raise ValueError('Positive finite time budget required: '+key)
    return result


def placement_poses(ego_xy,ego_yaw,search,max_positions):
    """Carry an explicit anchor's path-relative pose forward; offset zero keeps the anchor."""
    distance=np.linalg.norm(np.diff(ego_xy,axis=0),axis=1)
    keep=np.r_[True,distance>1e-8]
    path=np.asarray(ego_xy)[keep];headings=np.unwrap(ego_yaw)[keep]
    if len(path)<2: raise ValueError('Recorded reference path has no usable length')
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    anchor=np.asarray(search['anchor_xy_m'],dtype=float)
    delta=np.diff(path,axis=0)
    fractions=(np.sum((anchor-path[:-1])*delta,axis=1)/np.sum(delta**2,axis=1)).clip(0.,1.)
    projected=path[:-1]+fractions[:,None]*delta
    index=int(np.argmin(np.linalg.norm(projected-anchor,axis=1)))
    origin=float(arc[index]+fractions[index]*(arc[index+1]-arc[index]))
    angle=float(np.interp(origin,arc,headings))
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    local=rotation.T@(anchor-projected[index])
    count=int(math.floor((search['max_forward_m']-search['min_forward_m'])/search['step_m']+1e-9))+1
    for i in range(min(count,max_positions)):
        offset=search['min_forward_m']+i*search['step_m']
        value=origin+offset
        if value>arc[-1]+1e-8: break
        heading=float(np.interp(value,arc,headings))
        turn=np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
        position=np.array([np.interp(value,arc,path[:,k]) for k in range(2)])+turn@local
        yield dict(forward_offset_m=float(offset),reference_arc_m=float(value),xy_m=position.tolist(),
            yaw_rad=float(search['anchor_yaw_rad']+heading-angle),placement_method='recorded_path_with_explicit_anchor_relative_pose')


def scene_binding(scene,target_track,effective):
    """Bind to actual annotation/map contents, without raw-data content hashes."""
    scene=Path(scene).resolve()
    files={}
    for name in ('frames.json','map_features.json','map_drivable_official.json','no_cross_lines.json'):
        path=scene/'metadata'/name
        files[name]=json.loads(path.read_text()) if path.is_file() else None
    return dict(scene_path=str(scene),target_track=target_track,acceptance_policy=effective,metadata=files)


def verify_binding(binding,scene,target_track,effective,allow_identical_relocation=False):
    actual=scene_binding(scene,target_track,effective)
    if not allow_identical_relocation and binding['scene_path']!=actual['scene_path']:
        raise ValueError('Witness scene path differs from its explicit binding')
    if any(binding[key]!=actual[key] for key in ('target_track','acceptance_policy','metadata')):
        raise ValueError('Witness scene/target/policy differs from its binding')
    return actual


def materialize_candidate(scene,output,target,node,candidate_id,attack_run,official_margin_m):
    from add_vehicle_scene import add_vehicle
    detail=add_vehicle(scene,output,0,target['track_id'],target['size_m'],trajectory_xy=node['xy'],trajectory_yaw=node['yaw'],official_min_margin_m=official_margin_m)
    frozen=Path(output)/'frozen_scene'
    detail.update(asset_path=target['asset_path'],candidate_id=candidate_id,attack_run=str(attack_run),
        branch_status='provisional_for_defense_probe',agent_generated=False)
    atomic_json(Path(output)/'asset.json',dict(asset_path=target['asset_path']))
    atomic_json(frozen/'freeze-info.json',detail)
    atomic_json(Path(output)/'summary.json',detail)
    return frozen


def invoke_probe(command,log,seconds):
    """One isolated process group keeps probe descendants inside the time budget."""
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
        raise


def run_search(scene,output,instruction,assets,constraints,required_operation='add_vehicle',required_track=None):
    scene,output=Path(scene).resolve(),Path(output).resolve()
    difficulty=resolve_difficulty(constraints.get('attack_difficulty'))
    if 'placement_search' not in difficulty:
        raise ValueError('Numerical placement entry requires an explicit placement search and budget')
    if required_operation!='add_vehicle' or required_track is not None or not constraints.get('stationary_target'):
        raise ValueError('Current bounded difficulty search supports explicit added stationary vehicles')
    if len(assets)!=1:
        raise ValueError('Numerical placement search requires exactly one explicitly selected asset')
    asset=assets[0];size=np.asarray(asset['size_m'],dtype=float)
    if not Path(asset['path']).is_file() or size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):
        raise ValueError('Asset must exist with finite positive dimensions')
    if (ROOT/'experiments').resolve() not in output.parents or output.exists():
        raise ValueError('Difficulty search output must be a new experiments directory')
    config=copy.deepcopy(constraints);config['attack_difficulty']=difficulty
    effective=policy.resolve(config.get('acceptance_policy'))
    for key in ('observation_end_s','official_margin_m'):
        if key in config and config[key]!=effective[key]:
            raise ValueError('Difficulty search requires task/policy agreement: '+key)
        config[key]=effective[key]
    config['acceptance_policy']=effective
    config.setdefault('max_ego_clearance_m',policy.ATTACK_EVENT_MAX_GAP_M)
    config.setdefault('min_travel_m',0.)
    frames=json.loads((scene/'metadata/frames.json').read_text())
    times,ego,yaw=ego_path_from_frames(frames)
    poses=list(placement_poses(ego,yaw,difficulty['placement_search'],difficulty['budget']['max_positions']))
    output.mkdir(parents=True)
    atomic_json(output/'input.json',dict(schema_version='attack_difficulty_search.v1',scene=str(scene),instruction=instruction,
        assets=assets,constraints=config,required_operation=required_operation,api_calls=0,physical_execution='not_run'))
    atomic_json(output/'effective-policy.json',effective)
    start=time.monotonic();budget=difficulty['budget'];deadline=start+budget['total_seconds']
    summary=dict(schema_version='attack_difficulty_search.v1',status='searching',mode=difficulty['mode'],mode_source=difficulty['source'],
        selection_rule='first_attack_and_verified_witness_pass_in_configured_order' if difficulty['mode']!='free' else 'first_attack_pass_in_configured_order',
        mode_requires_recorded_conflict=difficulty['mode']!='free',branches=[],selected_branch_id=None,global_feasibility_proven=False,
        agent_status='not_run',attack_generation='bounded_numerical_stationary_placement',api_calls=0,physical_execution='not_run',
        lane_identity='unknown',budget=budget,search_uses_complete_future_observations=True)
    try:
        for index,pose in enumerate(poses,1):
            if time.monotonic()>=deadline:
                summary['stop_reason']='total_time_budget';break
            branch=output/'branches'/f'position_{index:03d}';branch.mkdir(parents=True)
            record=dict(branch_id=branch.name,pose=pose,attack_status='not_checked',defense_status='not_searched',formal_frozen=False,
                lane_identity='unknown',timing_s={},witness=None)
            stage_start=time.monotonic()
            try:
                target=dict(operation='add_vehicle',track_id='synthetic_vehicle_001',size_m=size.tolist(),asset_path=str(Path(asset['path']).resolve()),asset_id=asset['asset_id'])
                original_tracks={str(track) for frame in frames for track in frame['anns']['track_tokens']}
                suffix=1
                while target['track_id'] in original_tracks:
                    suffix+=1;target['track_id']=f'synthetic_vehicle_{suffix:03d}'
                xy=np.tile(pose['xy_m'],(len(times),1));angles=np.full(len(times),pose['yaw_rad'])
                seed=branch/'object_selection/seed_trajectory.csv'
                write_trajectory(seed,times,xy,angles,size)
                selection=dict(target=target,seed_trajectory_path=str(seed),source='bounded_numerical_placement_not_agent')
                atomic_json(branch/'object_selection/output.json',selection)
                editor=AttackEditor(scene,selection,branch/'attack_editor',config)
                node=editor.nodes['seed_000'];result=node['result']
                attack_pass=result['reference_screen_passed'] and (result.get('construction_goal') is None or result['construction_goal']['goal_met'])
                record.update(attack_status='passed' if attack_pass else 'failed',attack_metrics=editor.public('seed_000'),
                    baseline_conflict=dict(present=result['first_contact_time_s'] is not None,first_contact_sample_s=result['first_contact_time_s'],
                        method='existing attack_time_grid and polygon contact checker',max_step_s=effective['max_check_step_s']))
                record['timing_s']['attack_check']=time.monotonic()-stage_start
                diagnostic_start=time.monotonic()
                try:
                    from ego_edit_evidence import braking_diagnostic
                    target_actor=dict(track=target['track_id'],label='vehicle',x=pose['xy_m'][0],y=pose['xy_m'][1],
                        yaw=pose['yaw_rad'],length=float(size[0]),width=float(size[1]))
                    diagnostic_actors=[list(group)+[dict(target_actor)] for group in editor.actors]
                    record['difficulty_diagnostic']=braking_diagnostic(times,ego,yaw,diagnostic_actors,target['track_id'],effective)
                    record['difficulty_diagnostic']['source']='original observations plus exact stationary candidate seed'
                except Exception as diagnostic_error:
                    record['difficulty_diagnostic']=dict(status='unavailable',error_type=type(diagnostic_error).__name__,
                        diagnostic_only=True,attack_acceptance_affected=False)
                record['timing_s']['difficulty_diagnostic']=time.monotonic()-diagnostic_start
                if not attack_pass or (difficulty['mode']!='free' and result['first_contact_time_s'] is None):
                    record['stop_reason']='attack_failed' if not attack_pass else 'mode_requires_recorded_conflict'
                elif time.monotonic()>=deadline:
                    record['defense_status']='budget_interrupted'
                else:
                    stage_start=time.monotonic()
                    candidate_scene=materialize_candidate(scene,branch/'candidate_scene',target,node,'seed_000',editor.run_dir,effective['official_margin_m'])
                    record['candidate_scene']=str(candidate_scene)
                    atomic_json(branch/'scene-binding.json',scene_binding(candidate_scene,target['track_id'],effective))
                    record['timing_s']['candidate_materialization']=time.monotonic()-stage_start
                    if difficulty['mode']=='free':
                        record['defense_status']='not_searched'
                        eligible=True
                    else:
                        stage_start=time.monotonic()
                        remaining=min(budget['per_probe_seconds'],deadline-time.monotonic())
                        if remaining<=0:
                            record['defense_status']='budget_interrupted';eligible=False
                        else:
                            command=[sys.executable,str(ROOT/'scripts/run_defense_probe.py'),'--input',str(candidate_scene),'--run',str(branch/'defense_probe'),
                                '--target-track',target['track_id'],'--policy',str(output/'effective-policy.json'),'--max-candidates',str(budget['max_defense_candidates'])]
                            try:
                                with (branch/'defense-probe.log').open('x') as log:
                                    returncode=invoke_probe(command,log,remaining)
                                probe_path=branch/'defense_probe/output.json'
                                if not probe_path.is_file():
                                    record['defense_status']='execution_error'
                                else:
                                    probe=json.loads(probe_path.read_text());record['defense_status']=probe['status'];record['defense_probe']=probe
                                if returncode: record['defense_status']='execution_error'
                                eligible=record['defense_status']=='witness_found' and returncode==0
                            except subprocess.TimeoutExpired:
                                record['defense_status']='budget_interrupted';eligible=False
                        record['timing_s']['defense_probe']=time.monotonic()-stage_start
                    if eligible and time.monotonic()<deadline:
                        stage_start=time.monotonic()
                        formal=branch/'materialized/frozen_scene'
                        if difficulty['mode']!='free':
                            binding=json.loads((branch/'defense_probe/witness-binding.json').read_text())
                            verify_binding(binding,candidate_scene,target['track_id'],effective)
                        shutil.copytree(candidate_scene,formal,symlinks=True)
                        verify_binding(scene_binding(candidate_scene,target['track_id'],effective),formal,target['track_id'],effective,True)
                        info=json.loads((formal/'freeze-info.json').read_text())
                        info.update(branch_status='formal_after_verified_witness' if difficulty['mode']!='free' else 'formal_free_attack',
                            source_candidate_scene=str(candidate_scene),difficulty_mode=difficulty['mode'])
                        atomic_json(formal/'freeze-info.json',info)
                        record.update(formal_frozen=True,frozen_scene_path=str(formal),attack_run=str(editor.run_dir),attack_candidate_id='seed_000',target_track=target['track_id'])
                        if difficulty['mode']!='free':
                            record['witness']=dict(probe['witness'],formal_scene_path=str(formal),identical_scene_relocation_verified=True)
                            atomic_json(branch/'formal-witness-binding.json',scene_binding(formal,target['track_id'],effective))
                            write_followup(branch,record,config)
                        record['timing_s']['formalization']=time.monotonic()-stage_start
                        if summary['selected_branch_id'] is None: summary['selected_branch_id']=branch.name
                    elif eligible:
                        record['defense_status']='budget_interrupted_before_formalization'
            except Exception as error:
                record.update(defense_status='execution_error',error_type=type(error).__name__,error=str(error)[:250])
            if record.get('difficulty_diagnostic'):
                status=record['defense_status']
                record['difficulty_diagnostic']['defense_search']=dict(status=status,
                    verified_passing_candidate_ids=[] if not record['witness'] else [record['witness']['candidate_id']],
                    global_feasibility_proven=False)
            record['elapsed_s']=sum(record['timing_s'].values())
            atomic_json(branch/'output.json',record);summary['branches'].append(record)
            atomic_json(output/'output.json',summary)
            if record['formal_frozen'] and difficulty['mode']!='curriculum':
                summary['stop_reason']='first_eligible_branch_selected';break
        else:
            requested=int(math.floor((difficulty['placement_search']['max_forward_m']-difficulty['placement_search']['min_forward_m'])/difficulty['placement_search']['step_m']+1e-9))+1
            summary['stop_reason']='position_budget' if len(poses)<requested else 'configured_positions_complete'
        selected=next((row for row in summary['branches'] if row['branch_id']==summary['selected_branch_id']),None)
        summary.update(status='selected_branch' if selected else 'search_complete_without_selected_branch',selected=selected,
            total_elapsed_s=time.monotonic()-start,stage_totals_s={key:sum(row['timing_s'].get(key,0.) for row in summary['branches']) for key in ('attack_check','difficulty_diagnostic','candidate_materialization','defense_probe','formalization')})
        atomic_json(output/'output.json',summary)
        return summary
    finally:
        # Interrupted children leave their candidate artifacts; they never become formal branches.
        atomic_json(output/'stage-timing.json',dict(total_elapsed_s=time.monotonic()-start,branches=[dict(branch_id=row['branch_id'],timing_s=row['timing_s']) for row in summary['branches']]))


def write_followup(branch,record,config):
    probe=Path(record['defense_probe']['run_dir'])
    followup=dict(status='not_run',scene=record['frozen_scene_path'],target_track=record['target_track'],attack_run=record['attack_run'],attack_candidate_id='seed_000',
        constructor_run=str(probe/'constructor'),acceptance_policy=config['acceptance_policy'],
        agent_pool_contains_validated_witness=True,full_future_information_used=True,
        evaluation_claim='Agent chooses/edits from a witness-containing numerical pool; independent rediscovery is not established.',
        witness_used_as_agent_output=False)
    atomic_json(branch/'followup-agent.json',followup)
    followup.update(output=str(branch/'future_agent_run'),attack_candidate_id='seed_000',
        binding_path=str(branch/'formal-witness-binding.json'),
        timeout_seconds=600,max_tokens=16384,max_turns=10,max_edit_candidates=5,max_inspections=3,
        image_scene=None,cameras=['CAM_F0'])
    atomic_json(branch/'followup-agent.json',followup)
    (branch/'followup-commands.txt').write_text(f'{sys.executable} scripts/run_ego_branch_agent.py --config {branch}/followup-agent.json\n')



def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    output=args.output.resolve()
    summary=run_search(config['scene'],output,config['instruction'],config['assets'],config['constraints'],config.get('required_operation','add_vehicle'),config.get('required_track'))
    print(json.dumps(dict(status=summary['status'],selected_branch_id=summary['selected_branch_id'],branches=len(summary['branches']))))


if __name__=='__main__':main()
