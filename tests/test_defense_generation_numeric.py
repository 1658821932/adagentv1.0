"""Real numerical regression tests; runnable with the existing SciPy interpreter."""
import importlib.util
from pathlib import Path
import sys
import unittest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'Use the existing carla-gs-py310 numerical environment')
class DefenseNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(CORE))
        import numpy as np
        import ego_trajectory_edit as trajectory
        from run_ego_reference import conflict_decisions
        from run_ego_foci_probe import FociProblem
        cls.np=np;cls.trajectory=trajectory;cls.decisions=staticmethod(conflict_decisions);cls.problem=FociProblem

    def test_local_foci_zero_edit_and_outside_window_are_exactly_parent(self):
        np=self.np;times=np.linspace(0,12,25)
        parent=np.column_stack([times*2.,np.sin(times/5)*.1]);yaw=np.zeros(len(times))
        actors=[[dict(track='target',x=40.,y=0.,yaw=0.,length=4.5,width=2.)] for _ in times]
        problem=self.problem(times,parent,yaw,actors,target_track='target',freeze_before_s=3.,
                             edit_window_s=[4.,10.],observation_end_s=3.)
        xy,angles=problem.trajectory(np.zeros_like(problem.initial_control))
        self.assertTrue(np.array_equal(xy,parent));self.assertTrue(np.array_equal(angles,yaw))
        delta=np.full_like(problem.initial_control,.1);delta[problem.frozen]=0.
        xy,angles=problem.trajectory(delta)
        outside=(times<=4.)|(times>=10.)
        self.assertTrue(np.array_equal(xy[outside],parent[outside]))
        self.assertTrue(np.any(xy[~outside]!=parent[~outside]))

    def test_nondefault_history_boundary_is_protected(self):
        np=self.np;times=np.linspace(0,12,25);parent=np.column_stack([times,np.zeros(25)]);yaw=np.zeros(25)
        actors=[[dict(track='target',x=40.,y=0.,yaw=0.,length=4.5,width=2.)] for _ in times]
        with self.assertRaises(ValueError):
            self.problem(times,parent,yaw,actors,target_track='target',edit_window_s=[3.,10.],observation_end_s=4.)
        problem=self.problem(times,parent,yaw,actors,target_track='target',edit_window_s=[4.,10.],observation_end_s=4.)
        delta=np.full_like(problem.initial_control,.1);delta[problem.frozen]=0.
        xy,_=problem.trajectory(delta)
        self.assertTrue(np.array_equal(xy[times<=4.],parent[times<=4.]))

    def test_capped_seed_pool_covers_later_starts_and_moving_end_states(self):
        np=self.np
        policy=dict(observation_end_s=3.,max_acceleration_mps2=6.,comfort_reference_mps2=2.4)
        seed=self.decisions(np.linspace(0,11.5,24),8.,policy)[:24]
        self.assertEqual(seed[0][0],'baseline')
        self.assertGreater(max(item[1] for item in seed),3.5)
        self.assertTrue(any(isinstance(item[2],dict) and item[2].get('target_speed_fraction',0)>0 for item in seed))
        self.assertTrue(any(isinstance(item[2],dict) and item[2].get('deceleration_mps2',6)<3 for item in seed))
        self.assertTrue(any(isinstance(item[2],dict) and item[2].get('lateral_offset_m',0)!=0 for item in seed))


    def test_micro_timestamp_geometry_fix_preserves_history_and_legacy_replay(self):
        np=self.np
        times=np.arange(0.,12.,.5);times[5]=2.499733;times[6]=3.000015
        xy=np.column_stack([times*12.,np.zeros(len(times))]);yaw=np.full(len(times),.01)
        operation=dict(parent_candidate_id='recorded_baseline',path_candidate_id='recorded_baseline',
            start_time_s=3.,deceleration_mps2=2.,target_speed_mps=0.,motion_intent='forward_stop',end_behavior='hold')
        old_xy,old_yaw,old_curve=self.trajectory.brake_along_path(times,xy,yaw,xy,yaw,operation,3.,with_details=True)
        new_op=dict(operation,path_geometry_version='merge_micro_knots.v1')
        new_xy,new_yaw,new_curve=self.trajectory.brake_along_path(times,xy,yaw,xy,yaw,new_op,3.,with_details=True)
        self.assertLess(new_curve['max_acceleration_mps2'],old_curve['max_acceleration_mps2'])
        self.assertTrue(np.array_equal(new_xy[times<=3.],xy[times<=3.]))
        self.assertTrue(np.array_equal(new_yaw[times<=3.],yaw[times<=3.]))
        repeated=self.trajectory.verify_braking_candidate(times,xy,yaw,old_xy,old_yaw,operation,3.)
        self.assertEqual(repeated,old_curve)
        repeated=self.trajectory.verify_braking_candidate(times,xy,yaw,new_xy,new_yaw,new_op,3.)
        self.assertEqual(repeated,new_curve)


    def test_new_foci_objective_keeps_history_but_can_adjust_terminal_pose(self):
        np=self.np
        from defense_behavior import DrivingObjective
        times=np.linspace(0,12,25);reference=np.column_stack([times*2,np.zeros(len(times))]);yaw=np.zeros(len(times))
        parent=reference.copy();parent[times>8,0]=16.
        actors=[[dict(track='target',x=60.,y=0.,yaw=0.,length=4.5,width=2.)] for _ in times]
        behavior=DrivingObjective(times,reference,actors,'target',dict(observation_end_s=3.,comfort_reference_mps2=2.4))
        problem=self.problem(times,parent,yaw,actors,target_track='target',freeze_before_s=3.,
            edit_window_s=[3.,12.],observation_end_s=3.,behavior_objective=behavior)
        delta=np.zeros_like(problem.initial_control);delta[~problem.frozen,0]=.2
        xy,_=problem.trajectory(delta)
        self.assertTrue(np.array_equal(xy[times<=3.],parent[times<=3.]))
        self.assertGreater(xy[-1,0],parent[-1,0])
        zero,_=problem.trajectory(np.zeros_like(delta))
        self.assertTrue(np.array_equal(zero,parent))
        local=self.problem(times,parent,yaw,actors,target_track='target',freeze_before_s=3.,
            edit_window_s=[4.,10.],observation_end_s=3.,behavior_objective=behavior)
        changed,_=local.trajectory(delta)
        outside=(times<=4.)|(times>=10.)
        self.assertTrue(np.array_equal(changed[outside],parent[outside]))
        cost=problem.objective(delta.reshape(-1),1.,.03,1.,1.,0.)
        self.assertTrue(np.isfinite(cost))


if __name__=='__main__':unittest.main()
