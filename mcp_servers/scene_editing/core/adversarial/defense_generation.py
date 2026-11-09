"""Agent-controlled construction and FOCI reuse the editor's candidate graph/checks."""
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from defense_generation_contract import TOOLS, validate_request, normalize_request
from experiment_runtime import atomic_json


class DefenseGeneration:
    def __init__(self, editor, candidate_limit, *, foci_enabled=True, foci_call_limit=2,
                 foci_iterations=250, foci_timeout=300):
        self.editor=editor
        self.candidate_limit=candidate_limit
        self.foci_enabled=foci_enabled
        self.foci_call_limit=foci_call_limit
        self.foci_iterations=foci_iterations
        self.foci_timeout=foci_timeout
        self.root=editor.run_dir/'generation'
        self.root.mkdir(exist_ok=True)
        self.path=self.root/'budget.json'
        self.config=dict(candidate_limit=candidate_limit,foci_enabled=foci_enabled,
                         foci_call_limit=foci_call_limit,foci_iterations=foci_iterations,foci_timeout=foci_timeout)
        if self.path.is_file():
            self.state=json.loads(self.path.read_text())
            if self.state['configuration']!=self.config:
                raise ValueError('Generation budget configuration differs from archived run')
        else:
            self.state=dict(configuration=self.config,rejected_attempts=0,foci_calls=0,calls=[])
            self.save()

    def save(self):
        atomic_json(self.path,self.state)

    def remaining(self):
        return max(0,self.candidate_limit-self.editor.edit_count-self.state['rejected_attempts'])

    def context(self):
        return dict(shared_new_candidates_remaining=self.remaining(),
                    rejected_generation_attempts=self.state['rejected_attempts'],
                    foci_calls_remaining=max(0,self.foci_call_limit-self.state['foci_calls']) if self.foci_enabled else 0,
                    foci_enabled=self.foci_enabled, budget_path=str(self.path),
                    scope='Construction, edits, repairs and FOCI candidates share the existing new-candidate limit. Rejected generation attempts are charged. Additional FOCI calls are tracked separately from the archived initial pool. Shared behavior objective; acceptance limits unchanged.')

    def tools(self):
        if not self.remaining():
            return []
        tools=[]
        for tool in deepcopy(TOOLS):
            function=tool['function'];schema=function['parameters']
            schema['properties']['max_candidates']['maximum']=min(3,self.remaining())
            if function['name']=='optimize_defense_candidate':
                if not self.foci_enabled or self.state['foci_calls']>=self.foci_call_limit:
                    continue
                schema['properties']['parent_candidate_id']['enum']=list(self.editor.nodes)
            else:
                schema['properties']['parent_trajectory_id']['enum']=['recorded_baseline',*self.editor.nodes]
            tools.append(tool)
        return tools

    def _parent(self,key):
        if key=='recorded_baseline':
            for cid,node in self.editor.nodes.items():
                if np.array_equal(node['xy'],self.editor.baseline_xy) and np.array_equal(node['yaw'],self.editor.baseline_yaw):
                    return cid
            raise ValueError('This legacy pool has no recorded-baseline candidate; choose an existing candidate')
        if key not in self.editor.nodes:
            raise ValueError('Unknown parent candidate')
        return key

    def execute(self,name,args):
        editor=self.editor
        args=normalize_request(name,args,float(editor.times[-1]),editor.effective_policy.get('time_rounding_s',.001))
        validate_request(name,args,boundary=editor.effective_policy['observation_end_s'],
                         end=float(editor.times[-1]),remaining=self.remaining())
        key=args.get('parent_trajectory_id',args.get('parent_candidate_id'))
        parent=self._parent(key)
        if name=='construct_defense_candidates':
            if args['target_speed_mps']>editor.effective_policy['max_speed_mps'] or args['deceleration_range_mps2'][1]>editor.effective_policy['max_acceleration_mps2']:
                raise ValueError('Requested search range exceeds the unchanged research policy')
        elif not self.foci_enabled or self.state['foci_calls']>=self.foci_call_limit:
            raise ValueError('FOCI call budget exhausted or FOCI disabled')
        if name=='optimize_defense_candidate' and args.get('terminal_behavior')=='stop' and args['edit_window_s'][0]>=editor.times[-3]:
            raise ValueError('terminal_behavior=stop needs a window starting before the last two sampled intervals; use edit_speed_profile for a shorter tail edit')
        if name=='optimize_defense_candidate' and any(abs(p['speed_mps'])>editor.effective_policy['max_speed_mps'] for p in args.get('speed_targets',[])):
            raise ValueError('speed_targets exceeds unchanged hard speed limit')
        call=dict(tool=name,arguments=args,status='running',candidate_ids=[])
        self.state['calls'].append(call)
        number=len(self.state['calls'])
        self.save()
        results=[];failures=[]
        try:
            if name=='construct_defense_candidates':
                count=args['max_candidates']
                # Sample the requested ranges; this bounded sample is not an exhaustive search.
                starts=np.linspace(*args['start_time_range_s'],count) if count>1 else [sum(args['start_time_range_s'])/2]
                strengths=np.linspace(*args['deceleration_range_mps2'],count) if count>1 else [sum(args['deceleration_range_mps2'])/2]
                for start,strength in zip(starts,strengths):
                    operation=dict(mode='point_edit',edit_mode='brake_profile',generation_source='constructor',
                        parent_candidate_id=parent,path_candidate_id=key,start_time_s=float(start),
                        deceleration_mps2=float(strength),target_speed_mps=args['target_speed_mps'],
                        motion_intent='forward_stop' if args['target_speed_mps']==0 else 'forward',
                        end_behavior=args['end_behavior'],reason='Agent-requested bounded construction')
                    result=editor.edit(operation)
                    if result['status']=='invalid':
                        self.state['rejected_attempts']+=1
                        failures.append(dict(start_time_s=float(start),deceleration_mps2=float(strength),reason=result['error']))
                    else:
                        results.append(editor._public(result['candidate_id']))
                        call['candidate_ids'].append(result['candidate_id'])
                    self.save()
            else:
                self.state['foci_calls']+=1
                self.save()
                run=self.root/f'foci_{number:03d}'
                script=Path(__file__).with_name('run_ego_foci_probe.py')
                cmd=[sys.executable,str(script),'--input',str(editor.input_dir),'--parent-run',str(editor.run_dir),
                     '--candidate',parent,'--run',str(run),'--target-track',editor.target_track,
                     '--freeze-before',str(editor.effective_policy['observation_end_s']),
                     '--edit-window',*[str(x) for x in args['edit_window_s']],
                     '--max-candidates',str(args['max_candidates']),'--max-iterations',str(self.foci_iterations)]
                if args.get('preserve_route',False):cmd.append('--preserve-route')
                cmd.extend(['--terminal-behavior',args.get('terminal_behavior','free'),
                    '--improvement-goal',args.get('improvement_goal','balanced')])
                if 'speed_targets' in args:
                    targets_path=self.root/f'foci_{number:03d}-speed-targets.json'
                    atomic_json(targets_path,args['speed_targets'])
                    cmd.extend(['--speed-targets',str(targets_path)])
                with (self.root/f'foci_{number:03d}.log').open('x') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=self.foci_timeout,check=True)
                source=json.loads((run/'summary.json').read_text())
                candidates=[r for r in source['candidates'] if r['candidate_id'].startswith('foci_')]
                if not candidates:
                    failures.append(dict(reason=source['gaussian_optimizer'],report_path=str(run/'summary.json')))
                    self.state['rejected_attempts']+=1
                for row in candidates:
                    result=editor.edit(dict(mode='foci',generation_source='foci',parent_candidate_id=parent,
                        foci_run=str(run),foci_candidate_id=row['candidate_id'],reason='Agent-requested shared progress/reserve FOCI objective'))
                    if result['status']=='invalid':
                        raise RuntimeError('Generated FOCI candidate could not be imported: '+result['error'])
                    public=editor._public(result['candidate_id'])
                    if row.get('optimization',{}).get('speed_target_feedback') is not None:
                        public['speed_target_feedback']=row['optimization']['speed_target_feedback']
                    results.append(public)
                    call['candidate_ids'].append(result['candidate_id'])
                    self.save()
            call['status']='completed'
        except Exception as error:
            call.update(status='execution_failed',error_type=type(error).__name__)
            raise
        finally:
            self.save()
        return dict(status='candidates_generated' if results else 'no_candidates_generated',
                    candidates=results,construction_failures=failures,generation_budget=self.context(),
                    evidence_path=str(self.path),scope='Bounded search results; failed candidates are editable and cannot be submitted until passing')


def prepare_initial_foci(scene, constructor, target_track, output, boundary, end, *, candidate_limit=2, timeout=300):
    """Shared default stage for registered and standalone loops; an empty pool is normal."""
    from defense_generation_contract import select_foci_parents
    constructor,output=Path(constructor),Path(output)
    pool=json.loads((constructor/'reference_screen/summary.json').read_text())
    parents=select_foci_parents(pool['candidates'],candidate_limit)
    parent=parents[0] if parents else None
    stage=output.parent/'foci-stage.json'
    if parent is None:
        atomic_json(stage,dict(status='no_parent',fallback_to_constructor=True))
        return None
    if pool['target_track']!=target_track or pool['acceptance_policy']['observation_end_s']!=boundary:
        raise ValueError('Initial FOCI scene/target/observation policy differs')
    script=Path(__file__).with_name('run_ego_foci_probe.py')
    command=[sys.executable,str(script),'--input',str(scene),'--parent-run',str(constructor),
        '--candidate',parent,'--run',str(output),'--target-track',target_track,
        '--freeze-before',str(boundary),'--edit-window',str(boundary),str(end),
        '--max-candidates',str(candidate_limit)]
    if len(parents)>1:
        command.extend(['--additional-parents',*parents[1:]])
    atomic_json(stage,dict(status='running',parent_candidate_id=parent,parent_candidate_ids=parents))
    try:
        with output.with_suffix('.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
        summary=json.loads((output/'summary.json').read_text())
    except Exception as error:
        atomic_json(stage,dict(status='execution_failed',parent_candidate_id=parent,error_type=type(error).__name__))
        raise
    has_candidates=any(row['candidate_id'].startswith('foci_') for row in summary['candidates'])
    atomic_json(stage,dict(status='completed' if has_candidates else 'no_candidates',parent_candidate_id=parent,parent_candidate_ids=parents,
        summary_path=str(output/'summary.json'),optimizer=summary['gaussian_optimizer'],fallback_to_constructor=not has_candidates))
    return output if has_candidates else None
