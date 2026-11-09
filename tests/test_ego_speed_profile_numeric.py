"""Behavior goals, exact-state inheritance and sampled stop boundaries must agree."""
import importlib.util
from pathlib import Path
import sys
import unittest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'

@unittest.skipUnless(importlib.util.find_spec('scipy'),'Requires existing numerical environment')
class SpeedProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(CORE))
        import numpy as np
        import ego_trajectory_edit as edit
        import ego_speed_profile as speed
        import acceptance_policy
        cls.np=np;cls.edit=edit;cls.speed=speed;cls.policy=acceptance_policy.resolve()

    def fixture(self):
        np=self.np;times=np.arange(0.,12.5,.5)
        xy=np.column_stack([2*times,np.zeros(len(times))]);yaw=np.zeros(len(times))
        return times,xy,yaw

    def test_stop_between_last_samples_is_recognized_from_rebuilt_curve(self):
        times,xy,yaw=self.fixture()
        op=dict(edit_mode='speed_profile',parent_candidate_id='p',start_time_s=9.,goal='stop',
                reach_target_time_s=11.7,reason='Keep route; repair moving end')
        out,angles,curve=self.speed.generate_speed_edit(times,lambda _: (xy,yaw),op,3.,acceptance=self.policy)
        # It still moved during the last CSV interval, but the actual curve is at rest.
        self.assertGreater(self.np.linalg.norm(out[-1]-out[-2]),1e-6)
        result=dict(generation_curve=curve,checks={})
        self.edit.apply_intent(result,times,out,angles,op)
        self.assertTrue(result['motion_intent']['passed'])
        self.assertEqual(result['motion_intent']['terminal_state_method'],'reconstructed_generation_curve')
        sampled=self.edit.intent_evidence(times,out,angles,'forward_stop',9.,'hold',11.7)
        self.assertFalse(sampled['terminal_stopped'])
        self.assertTrue(self.np.array_equal(out[times<=9.],xy[times<=9.]))
        self.assertAlmostEqual(curve['peak_tangential_deceleration_mps2'],1.5*2/2.7)

    def test_arbitrary_time_uses_parent_derivative_and_replay_is_exact(self):
        times,xy,yaw=self.fixture()
        low=dict(parent_candidate_id='p',path_candidate_id='p',start_time_s=3.,
            deceleration_mps2=.5,target_speed_mps=0.,motion_intent='forward_stop',end_behavior='hold')
        px,pa,old_curve=self.edit.brake_along_path(times,xy,yaw,xy,yaw,low,3.,with_details=True)
        lookup=lambda key:(xy,yaw) if key=='recorded_baseline' else (px,pa)
        old_curve['reference_path_candidate_id']='recorded_baseline'
        continuous=self.speed.continuous_parent(times,lookup,'parent',old_curve,3.)
        state=self.edit.inherited_brake_state(times,px,pa,3.25,continuous)
        self.assertAlmostEqual(state['velocity_xy_mps'][0],1.875,places=7)
        self.assertAlmostEqual(state['xy_m'][0],6.+2*.25-.25*.25**2,places=7)
        op=dict(edit_mode='speed_profile',parent_candidate_id='parent',start_time_s=3.25,goal='stop',
            braking_style='constant_deceleration',reach_target_time_s=6.8,reason='Stop earlier on same route')
        first=self.edit.modern_edit(times,lookup,op,3.,parent_curve=old_curve,acceptance=self.policy)
        second=self.edit.modern_edit(times,lookup,op,3.,parent_curve=old_curve,acceptance=self.policy)
        self.assertTrue(self.np.array_equal(first[0],second[0]));self.assertEqual(first[2],second[2])
        self.assertEqual(first[2]['actual_start_state']['method'],'continuous_parent_curve')

    def test_unreachable_goal_reports_required_strength_and_preserves_history(self):
        times,xy,yaw=self.fixture()
        with self.assertRaisesRegex(ValueError,'peak tangential deceleration.*required'):
            self.speed.generate_speed_edit(times,lambda _: (xy,yaw),dict(parent_candidate_id='p',start_time_s=11.,
                goal='stop',reach_target_time_s=11.1,max_deceleration_mps2=2.4,reason='Test deadline'),3.,acceptance=self.policy)
        with self.assertRaisesRegex(ValueError,'observation history is fixed'):
            self.speed.generate_speed_edit(times,lambda _: (xy,yaw),dict(parent_candidate_id='p',start_time_s=2.9,
                goal='stop',reason='Invalid history'),3.,acceptance=self.policy)

    def test_implicit_deadline_fits_route_without_moving_an_explicit_deadline(self):
        times,xy,yaw=self.fixture()
        # A short remaining spatial route requires firmer braking than comfort.
        xy[times>=10.,0]=20.+.3*(times[times>=10.]-10.)
        op=dict(parent_candidate_id='p',start_time_s=10.,goal='stop',reason='Repair tail')
        out,angles,curve=self.speed.generate_speed_edit(times,lambda _: (xy,yaw),op,3.,acceptance=self.policy)
        self.assertLessEqual(curve['speed_edit_summary']['target_time_s'],times[-1])
        self.assertLessEqual(out[-1,0],xy[-1,0]+1e-7)
        self.assertGreater(curve['peak_tangential_deceleration_mps2'],2.4)
        with self.assertRaisesRegex(ValueError,'no extrapolation'):
            self.speed.generate_speed_edit(times,lambda _: (xy,yaw),dict(op,reach_target_time_s=11.8),3.,acceptance=self.policy)

    def test_foci_route_goal_keeps_curved_geometry_and_explicit_terminal_hold(self):
        from run_ego_foci_probe import FociProblem
        from shapely.geometry import LineString,Point
        np=self.np;times,xy,yaw=self.fixture()
        xy[:,1]=.05*times**2;yaw=np.arctan2(np.gradient(xy[:,1]),np.gradient(xy[:,0]))
        actors=[[dict(track='target',x=50.,y=0.,yaw=0.,length=4.5,width=2.)] for _ in times]
        problem=FociProblem(times,xy,yaw,actors,target_track='target',freeze_before_s=3.,edit_window_s=[3.,12.],
            edit_goals=dict(preserve_route=True,terminal_behavior='stop'))
        delta=np.zeros_like(problem.initial_control);delta[~problem.frozen,0]=-.3
        out,angles=problem.trajectory(delta)
        route=LineString(xy)
        self.assertLess(max(route.distance(Point(p)) for p in out),1e-10)
        self.assertTrue(np.array_equal(out[times<=3.],xy[times<=3.]))
        self.assertTrue(np.array_equal(out[-1],out[-2]));self.assertEqual(angles[-1],angles[-2])
        bounds=np.array(problem.bounds()).reshape(-1,3,2)
        self.assertTrue(np.all(bounds[:,1:]==0))

if __name__=='__main__':unittest.main()
