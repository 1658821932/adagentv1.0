"""Run the independent P0 ego-decision -> trajectory -> reference-check pipeline."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

from ego_avoidance_contract import validate_decision
from ego_avoidance_other_actors import evaluate_candidate,load_context
from ego_avoidance_scene import load_frozen_attack_scene,metadata_json,write_frozen_input
from ego_avoidance_trajectory import generate_ego_trajectory


from acceptance_policy import ego_limits
DEFAULT_LIMITS = ego_limits()


def manual_decision(horizon):
    return dict(version='defense_decision.v1',action_type='brake',start_s=8.,planning_horizon_s=horizon,
        reason='Executor-selected manual P0 probe: smooth braking before the archived interaction; not an LLM decision',
        evidence=['run04 CARLA actual attack trajectory','run04 edited attack video'],
        brake=dict(deceleration_mps2=3.,target_speed_mps=5.))


def rule_decisions(horizon):
    for start,deceleration,target in itertools.product([4.,6.,8.,10.],[2.,3.,4.,5.],[0.,5.,8.,10.]):
        yield dict(version='defense_decision.v1',action_type='brake',start_s=start,planning_horizon_s=horizon,
            reason='Bounded deterministic P0 braking grid; full future attack trajectory is offline privileged input',
            evidence=['frozen attack actual trajectory'],brake=dict(deceleration_mps2=deceleration,target_speed_mps=target))


def _write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def _write_trajectory(path,trajectory):
    fields=['sample_index','frame_id','time_s','x_m','y_m','z_m','heading_rad','speed_mps']
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i in range(len(trajectory.times_s)):
            writer.writerow(dict(sample_index=i,frame_id=f'{trajectory.frame_ids[i]:g}',time_s=f'{trajectory.times_s[i]:.12g}',
                x_m=f'{trajectory.xy[i,0]:.12g}',y_m=f'{trajectory.xy[i,1]:.12g}',z_m=f'{trajectory.z_m[i]:.12g}',
                heading_rad=f'{trajectory.heading_rad[i]:.12g}',speed_mps=f'{trajectory.speed_mps[i]:.12g}'))


def _check(scene,trajectory,limits,other_context=None):
    return evaluate_candidate(scene,trajectory,limits,other_context)


def run_offline(source_run,run_dir,mode='manual',decision_path=None,other_actor_path=None):
    if mode not in {'manual','rule'}:
        raise ValueError('P0 mode must be manual or rule')
    run_dir=Path(run_dir)
    if run_dir.exists():
        raise FileExistsError('Refusing to reuse or overwrite an existing ego avoidance run')
    source=load_frozen_attack_scene(source_run);scene=source.planning_view();write_frozen_input(source,run_dir)
    other_context=load_context(other_actor_path,scene) if other_actor_path else None
    if other_context is not None:_write_json(run_dir/'input/other-actors.json',other_context)
    _write_json(run_dir/'config.json',dict(version='ego_avoidance_p0.v1',mode=mode,limits=DEFAULT_LIMITS,
        source=metadata_json(source),api_calls=0,carla_executed=False,rendered=False,
        other_actor_context=str(Path(other_actor_path).resolve()) if other_actor_path else None,
        threshold_scope='Shared reference acceptance defaults; missing road/marking checks remain unvalidated'))
    if decision_path:
        if mode!='manual':raise ValueError('A decision file is only valid in manual mode')
        decisions=[json.loads(Path(decision_path).read_text())]
    elif mode=='manual':decisions=[manual_decision(float(scene.times_s[-1]))]
    else:decisions=list(rule_decisions(float(scene.times_s[-1])))
    ledger=[];passed=[]
    for index,payload in enumerate(decisions,1):
        candidate_id=f'candidate_{index:03d}';folder=run_dir/'candidates'/candidate_id;folder.mkdir(parents=True)
        decision=validate_decision(payload);_write_json(folder/'decision.json',decision)
        trajectory=generate_ego_trajectory(scene,decision);_write_trajectory(folder/'ego-reference.csv',trajectory)
        result=_check(scene,trajectory,DEFAULT_LIMITS,other_context);_write_json(folder/'reference-check.json',result)
        _write_json(folder/'other-actor-check.json',result['other_actor_check'])
        entry=dict(candidate_id=candidate_id,action_type=decision['action_type'],start_s=decision['start_s'],
            brake=decision['brake'],lane_change=decision['lane_change'],status=result['status'],
            avoidance_status=result['avoidance_status'],minimum_clearance_m=result['collision_check']['minimum_clearance_m'],
            progress_m=result['progress_check']['progress_m'],motion_peaks=result['motion_check']['peaks'],
            violations=result['violations'])
        ledger.append(entry)
        if result['reference_passed']:passed.append(entry)
    selected=None
    if passed:
        selected=min(passed,key=lambda entry:((entry['brake'] or {}).get('deceleration_mps2',0.),
            entry['motion_peaks']['max_jerk_mps3'],-entry['minimum_clearance_m'],entry['candidate_id']))['candidate_id']
    _write_json(run_dir/'search/rule-ledger.json',dict(mode=mode,candidates=ledger,evaluations=len(ledger),
        passed=len(passed),failed=len(ledger)-len(passed),selected_candidate_id=selected,
        selection_rule='Among reference-passed candidates: lowest deceleration, then lowest jerk, then greatest clearance',
        unavoidable=False,note='A bounded candidate failure or exhausted grid does not prove the scene unavoidable'))
    summary=dict(version='avoidance_result.v1',pipeline_status='reference_passed' if passed else 'reference_failed',
        avoidance_status='undetermined',generation_mode=mode,candidates=len(ledger),reference_passed=len(passed),
        selected_candidate_id=selected,baseline_actual_contact_time_s=scene.metadata['baseline_actual_contact_time_s'],
        carla_validated=False,render_validated=False,road_status='not_strictly_validated',
        other_actor_status=result['other_actor_check']['status'] if other_context is not None else 'not_checked_in_p0',
        limitations=['Fixed exogenous attack replay includes archived post-contact response',
            'P0 uses sampled reference geometry only; no CARLA execution or rendering',
            'Other vehicles are recorded observations only; missing observations remain unknown' if other_context is not None else 'Other vehicles were not checked in this run',
            'Source road/map and full secondary-accident physics are not validated'])
    _write_json(run_dir/'summary.json',summary)
    artifacts=dict(status='passed',input_files=5 if other_context is not None else 4,candidates=len(ledger),
        candidate_directories=sorted(p.name for p in (run_dir/'candidates').iterdir()),
        all_candidates_have_decision=all((run_dir/'candidates'/e['candidate_id']/'decision.json').is_file() for e in ledger),
        all_candidates_have_trajectory=all((run_dir/'candidates'/e['candidate_id']/'ego-reference.csv').is_file() for e in ledger),
        all_candidates_have_check=all((run_dir/'candidates'/e['candidate_id']/'reference-check.json').is_file() for e in ledger),
        all_candidates_have_other_actor_check=all((run_dir/'candidates'/e['candidate_id']/'other-actor-check.json').is_file() for e in ledger),
        summary_present=(run_dir/'summary.json').is_file())
    _write_json(run_dir/'artifact-check.json',artifacts)
    if not all([artifacts['all_candidates_have_decision'],artifacts['all_candidates_have_trajectory'],
        artifacts['all_candidates_have_check'],artifacts['all_candidates_have_other_actor_check'],artifacts['summary_present']]):
        raise RuntimeError('Incomplete P0 artifacts')
    return summary


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=['manual','rule'],default='manual')
    parser.add_argument('--decision',type=Path,help='Optional defense_decision.v1 JSON for manual mode')
    parser.add_argument('--other-actors',type=Path,help='Recorded other-vehicle JSON; observed samples are checked over the full ego window')
    args=parser.parse_args()
    summary=run_offline(args.source_run,args.run_dir,args.mode,args.decision,args.other_actors)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
