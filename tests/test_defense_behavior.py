"""Shared progress excludes detours, reserve does not reward distance, costs guide search only."""
from pathlib import Path
import importlib
import importlib.util
import unittest

import numpy as np

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


def objective(module):
    times=np.linspace(0,12,25);reference=np.column_stack([times*2,np.zeros(len(times))])
    actors=[[dict(track='t',x=100.,y=0.,yaw=0.,length=4.,width=2.)] for _ in times]
    policy=dict(observation_end_s=3.,comfort_reference_mps2=2.4)
    return module.DrivingObjective(times,reference,actors,'t',policy),times,reference


@unittest.skipUnless(importlib.util.find_spec('shapely'), 'Use existing geometry environment')
class BehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0,str(CORE))
        cls.behavior=importlib.import_module('defense_behavior')

    def test_net_route_progress_excludes_lateral_detours_and_counts_backtracking(self):
        behavior=self.behavior
        obj,times,reference=objective(behavior)
        base=obj.evaluate(reference,np.zeros(len(times)))
        detour=reference.copy();detour[:,1]=np.sin(times)*3;detour[times<=3]=reference[times<=3]
        observed=obj.evaluate(detour,np.zeros(len(times)))
        assert observed['net_reference_progress_m']==base['net_reference_progress_m']
        assert observed['path_offset_rms_m']>base['path_offset_rms_m']
        reverse=reference.copy();reverse[-4:,0]=[20.,19.,18.,17.]
        evidence=obj.evaluate(reverse,np.zeros(len(times)))
        assert evidence['net_reference_progress_m']<base['net_reference_progress_m']
        assert evidence['reverse_reference_distance_m']>0


    def test_surplus_gap_has_no_reward_but_shortfall_has_cost(self):
        behavior=self.behavior
        obj,times,reference=objective(behavior);yaw=np.zeros(len(times))
        near=obj.evaluate(reference,yaw,np.full(len(times),10.))
        far=obj.evaluate(reference,yaw,np.full(len(times),30.))
        insufficient=obj.evaluate(reference,yaw,np.full(len(times),.2))
        assert near['reserve_shortfall_cost']==far['reserve_shortfall_cost']==0
        assert near['optimization_cost']==far['optimization_cost']
        assert insufficient['optimization_cost']>near['optimization_cost']
        approaching=obj.evaluate(reference,yaw,np.linspace(30.,5.,len(times)))
        assert approaching['terminal_closing_speed_mps']>0
        assert approaching['terminal_closing_speed_mps']>near['terminal_closing_speed_mps']


    def test_advancement_is_preferred_with_identical_reserve_and_no_added_gate(self):
        behavior=self.behavior
        obj,times,reference=objective(behavior);yaw=np.zeros(len(times))
        stopped=reference.copy();stopped[times>6,0]=12.
        slow=obj.evaluate(stopped,yaw,np.full(len(times),20.))
        forward=obj.evaluate(reference,yaw,np.full(len(times),20.))
        assert forward['net_reference_progress_fraction']>slow['net_reference_progress_fraction']
        assert forward['optimization_cost']<slow['optimization_cost']
        assert 'passed' not in forward and 'reference_screen_passed' not in forward
        assert 'optimization_cost' not in behavior.agent_evidence(forward)


    def test_pair_differences_and_parent_coverage_do_not_reorder_candidates(self):
        behavior=self.behavior
        from defense_generation_contract import select_foci_parents
        rows=[]
        for cid,progress,cost,acceleration in [('early',.3,.8,2.),('later',.5,.5,2.1),('same',.51,.6,2.1)]:
            rows.append(dict(candidate_id=cid,status='reference_screen_passed',reference_screen_passed=True,
                max_acceleration_mps2=acceleration,min_target_clearance_m=30.-progress,
                behavior_metrics=dict(net_reference_progress_m=progress*100,net_reference_progress_fraction=progress,
                    optimization_cost=cost,max_reserve_shortfall_m=0.)))
        differences=behavior.candidate_differences(rows)
        assert differences[0]['b_minus_a_net_reference_progress_m']==20.
        assert select_foci_parents(rows,2)==['later','early']
        assert [r['candidate_id'] for r in rows]==['early','later','same']

if __name__=='__main__':unittest.main()
