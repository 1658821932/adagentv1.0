"""Signed speed behavior, retiming and reference geometry must survive replay."""
import importlib.util
from pathlib import Path
import sys
import unittest
CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'

@unittest.skipUnless(importlib.util.find_spec('scipy'),'Use the existing numerical environment')
class ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(CORE))
        import numpy as np
        import ego_speed_schedule as schedule
        import ego_trajectory_edit as edit
        import acceptance_policy
        cls.np=np;cls.schedule=schedule;cls.edit=edit;cls.policy=acceptance_policy.resolve()

    def fixture(self):
        np=self.np;t=np.arange(0.,12.5,.5);xy=np.column_stack([2*t,np.zeros(len(t))]);yaw=np.zeros(len(t))
        return t,xy,yaw

    def request(self,**changes):
        d=dict(edit_mode='speed_schedule',operation='set_speed_targets',parent_candidate_id='p',start_time_s=3.,
            targets=[dict(time_s=5.,speed_mps=0.)],reason='Preserve route; change speed')
        d.update(changes);return d

    def test_stop_wait_restart_and_reverse_have_explicit_states_and_no_heading_flip(self):
        t,xy,yaw=self.fixture()
        op=self.request(targets=[dict(time_s=5.,speed_mps=0.),dict(time_s=6.,speed_mps=0.),
            dict(time_s=8.,speed_mps=-1.),dict(time_s=9.,speed_mps=0.),dict(time_s=10.,speed_mps=1.)])
        x,a,c=self.schedule.generate(t,lambda _: (xy,yaw),op,3.,acceptance=self.policy)
        self.assertTrue(self.np.array_equal(x[t<=3.],xy[t<=3.]))
        self.assertLess(abs(float(a.max())),1e-6)
        self.assertLess(c['max_acceleration_mps2'],3.)
        self.assertGreater(self.np.min(self.np.diff(c['continuous_state']['times_s'])),.001)
        self.assertTrue(self.np.array_equal(x[t==5.],x[t==6.]))
        self.assertLess(x[t==8.,0][0],x[t==6.,0][0])
        state=self.edit.inherited_brake_state(t,x,a,7.5,c)
        self.assertLess(state['signed_speed_mps'],0.);self.assertAlmostEqual(state['yaw_rad'],0.,places=7)
        self.assertAlmostEqual(c['speed_edit_summary']['target_evidence'][-1]['actual_speed_mps'],1.)
        result=dict(generation_curve=c,checks={})
        self.edit.apply_intent(result,t,x,a,op)
        self.assertTrue(result['motion_intent']['passed']);self.assertTrue(result['motion_intent']['declared_reverse'])
        self.assertFalse(result['motion_intent']['terminal_stopped'])
        # Child inherits reversing state with body heading still forward.
        child=self.request(start_time_s=7.5,targets=[dict(time_s=9.,speed_mps=0.)])
        y,b,d=self.schedule.generate(t,lambda _: (x,a),child,3.,parent_curve=c,acceptance=self.policy)
        self.assertLess(d['actual_start_state']['signed_speed_mps'],0.)
        self.assertTrue(self.np.array_equal(y[t<=7.5],x[t<=7.5]))

    def test_crossing_direction_without_zero_and_route_exhaustion_are_actionable(self):
        t,xy,yaw=self.fixture()
        with self.assertRaisesRegex(ValueError,'direction_change_requires_stop'):
            self.schedule.generate(t,lambda _: (xy,yaw),self.request(targets=[dict(time_s=6.,speed_mps=-1.)]),3.,acceptance=self.policy)
        with self.assertRaisesRegex(ValueError,'route_exhausted'):
            self.schedule.generate(t,lambda _: (xy,yaw),self.request(targets=[dict(time_s=5.,speed_mps=10.)]),3.,acceptance=self.policy)
        with self.assertRaisesRegex(ValueError,'after observation'):
            self.schedule.generate(t,lambda _: (xy,yaw),self.request(start_time_s=2.9),3.,acceptance=self.policy)

    def test_restart_from_actual_stop_uses_known_route_and_exact_child_replay(self):
        t,xy,yaw=self.fixture()
        low=dict(parent_candidate_id='p',path_candidate_id='recorded_baseline',start_time_s=3.,
            deceleration_mps2=1.,target_speed_mps=0.,motion_intent='forward_stop',end_behavior='hold')
        px,pa,c=self.edit.brake_along_path(t,xy,yaw,xy,yaw,low,3.,with_details=True)
        lookup=lambda k:(xy,yaw) if k=='recorded_baseline' else (px,pa)
        op=self.request(start_time_s=6.,path_source='recorded_baseline',targets=[dict(time_s=8.,speed_mps=1.)])
        x,a,d=self.edit.modern_edit(t,lookup,op,3.,parent_curve=c,acceptance=self.policy)
        self.assertEqual(d['actual_start_state']['velocity_xy_mps'],[0.,0.])
        self.assertGreater(x[-1,0],px[-1,0]);self.assertTrue(self.np.array_equal(x[t<=6],px[t<=6]))
        again=self.edit.modern_edit(t,lookup,op,3.,parent_curve=c,acceptance=self.policy)
        self.assertTrue(self.np.array_equal(x,again[0]));self.assertEqual(d,again[2])
        with self.assertRaisesRegex(ValueError,'route_exhausted'):
            self.edit.modern_edit(t,lookup,dict(op,path_source='parent'),3.,parent_curve=c,acceptance=self.policy)

    def test_retime_removes_original_braking_and_preserves_declared_stop_target(self):
        t,xy,yaw=self.fixture()
        low=dict(parent_candidate_id='p',path_candidate_id='recorded_baseline',start_time_s=3.,
            deceleration_mps2=.5,target_speed_mps=0.,motion_intent='forward_stop',end_behavior='hold')
        px,pa,c=self.edit.brake_along_path(t,xy,yaw,xy,yaw,low,3.,with_details=True)
        lookup=lambda k:(xy,yaw) if k=='recorded_baseline' else (px,pa)
        op=dict(edit_mode='speed_schedule',operation='retime_braking',parent_candidate_id='p',braking_start_shift_s=.5,
                preserve='stop_position',reason='Delay onset without moving the stopping location')
        x,a,d=self.schedule.generate(t,lookup,op,3.,parent_curve=c,acceptance=self.policy)
        self.assertAlmostEqual(x[-1,0],px[-1,0],places=5)
        self.assertLess(d['max_acceleration_mps2'],2.)
        self.assertAlmostEqual(x[t==3.5,0][0],7.,places=5) # Coasts, does not inherit old braking.
        self.assertEqual(d['speed_edit_summary']['preserved_until_s'],3.)
        self.assertTrue(self.np.array_equal(x[t<=3],px[t<=3]))
        timed=self.schedule.generate(t,lookup,dict(op,preserve='stop_time',path_source='recorded_baseline'),3.,parent_curve=c,acceptance=self.policy)[2]
        self.assertEqual(timed['hold_starts_at_s'],c['hold_starts_at_s'])
        with self.assertRaisesRegex(ValueError,'retiming_unavailable'):
            self.schedule.generate(t,lookup,op,3.,acceptance=self.policy)

    def test_curved_route_motion_bound_is_for_delivered_curve_not_tangential_cap(self):
        t,xy,yaw=self.fixture();np=self.np
        angle=.1*t;xy=np.column_stack([20*np.sin(angle),20*(1-np.cos(angle))]);yaw=angle
        op=self.request(targets=[dict(time_s=7.,speed_mps=0.)])
        x,a,c=self.schedule.generate(t,lambda _: (xy,yaw),op,3.,acceptance=self.policy)
        self.assertGreater(c['max_acceleration_mps2'],c['speed_edit_summary']['peak_tangential_acceleration_mps2'])
        self.assertLess(np.max(np.abs(np.diff(a))),.1)
        self.assertAlmostEqual(float(np.linalg.norm(c['terminal_velocity_xy_mps'])),0.)

    def test_foci_timed_speed_targets_are_explicit_soft_preferences(self):
        from run_ego_foci_probe import FociProblem
        t,xy,yaw=self.fixture()
        actors=[[dict(track='target',x=50.,y=0.,yaw=0.,length=4.5,width=2.)] for _ in t]
        common=dict(target_track='target',freeze_before_s=3.,edit_window_s=[3.,12.])
        target=[dict(time_s=10.,speed_mps=0.)]
        plain=FociProblem(t,xy,yaw,actors,**common)
        requested=FociProblem(t,xy,yaw,actors,edit_goals=dict(speed_targets=target),**common)
        zero=self.np.zeros(plain.initial_control.size)
        self.assertAlmostEqual(requested.objective(zero,1.,.03,1.,1.)-plain.objective(zero,1.,.03,1.,1.),4.)
        feedback=self.schedule.sampled_target_errors(t,xy,yaw,target)
        self.assertEqual(feedback['targets'][0]['sampled_speed_mps'],2.)
        self.assertIn('Soft',feedback['scope'])

    def test_body_heading_bias_cannot_create_lateral_bends_in_a_straight_route(self):
        t,xy,yaw=self.fixture();yaw[:]=.03
        op=self.request()
        out,angles,curve=self.schedule.generate(t,lambda _: (xy,yaw),op,3.,acceptance=self.policy)
        self.assertTrue(self.np.array_equal(out[:,1],xy[:,1]))
        self.assertLess(curve['acceleration_peak']['normal_mps2'],1e-8)
        self.assertEqual(curve['profile'],'signed_speed_schedule.v4')

    def test_retime_inherits_style_and_child_reuses_identical_saved_route(self):
        t,xy,yaw=self.fixture()
        low=dict(parent_candidate_id='p',path_candidate_id='recorded_baseline',start_time_s=3.,
            deceleration_mps2=.5,target_speed_mps=0.,motion_intent='forward_stop',end_behavior='hold')
        px,pa,parent=self.edit.brake_along_path(t,xy,yaw,xy,yaw,low,3.,with_details=True)
        lookup=lambda k:(xy,yaw) if k=='recorded_baseline' else (px,pa)
        op=dict(edit_mode='speed_schedule',operation='retime_braking',parent_candidate_id='p',braking_start_shift_s=.2,
                preserve='stop_position',reason='Delay onset, keep original braking style')
        out,angles,curve=self.schedule.generate(t,lookup,op,3.,parent_curve=parent,acceptance=self.policy)
        self.assertEqual(curve['speed_edit_summary']['transition_style'],'constant_acceleration')
        self.assertEqual(curve['speed_edit_summary']['transition_style_source'],'inherited_parent_braking')
        child=dict(op,braking_start_shift_s=.1)
        x,a,next_curve=self.schedule.generate(t,lambda _: (out,angles),child,3.,parent_curve=curve,acceptance=self.policy)
        self.assertEqual(next_curve['speed_route'],curve['speed_route'])
        self.assertEqual(next_curve['speed_edit_summary']['route_initialization']['method'],'reuse_saved_spatial_route')
        self.assertEqual(next_curve['speed_edit_summary']['transition_style'],'constant_acceleration')
        self.assertAlmostEqual(x[-1,0],out[-1,0],places=5)

    def test_archived_schedule_keeps_its_original_generation_and_new_version_is_explicit(self):
        t,xy,yaw=self.fixture();op=self.request()
        archived=self.edit.modern_edit(t,lambda _: (xy,yaw),op,3.,acceptance=self.policy)
        current=self.edit.modern_edit(t,lambda _: (xy,yaw),dict(op,speed_schedule_version='route_reuse.v2'),3.,acceptance=self.policy)
        self.assertEqual(archived[2]['profile'],'signed_speed_schedule.v1')
        self.assertEqual(current[2]['profile'],'signed_speed_schedule.v2')
        self.assertTrue(self.np.allclose(archived[0],current[0],atol=1e-8))
        with self.assertRaisesRegex(ValueError,'Unknown speed schedule'):
            self.edit.modern_edit(t,lambda _: (xy,yaw),dict(op,speed_schedule_version='future'),3.)

if __name__=='__main__':unittest.main()
