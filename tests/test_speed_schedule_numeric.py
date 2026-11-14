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
        with self.assertRaisesRegex(ValueError,'after edit_start_s'):
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

    def position_request(self,**changes):
        args=dict(edit_mode='speed_schedule',operation='stop_at_position',parent_candidate_id='p',
                  start_time_s=3.5,stop_target='parent_stop_position',reason='Keep actual stopping position')
        args.update(changes);return args

    def stopped_parent(self):
        t,xy,yaw=self.fixture()
        low=dict(parent_candidate_id='p',path_candidate_id='recorded_baseline',start_time_s=3.,
            deceleration_mps2=.5,target_speed_mps=0.,motion_intent='forward_stop',end_behavior='hold')
        px,pa,c=self.edit.brake_along_path(t,xy,yaw,xy,yaw,low,3.,with_details=True)
        return t,xy,yaw,px,pa,c

    def test_position_stop_preserves_actual_prefix_goal_and_has_exact_replay(self):
        t,xy,yaw,px,pa,c=self.stopped_parent()
        lookup=lambda k:(xy,yaw) if k=='recorded_baseline' else (px,pa)
        op=self.position_request()
        out,angles,curve=self.schedule.generate(t,lookup,op,3.,parent_curve=c,acceptance=self.policy)
        self.assertTrue(self.np.array_equal(out[t<=3.5],px[t<=3.5]))
        self.assertAlmostEqual(out[t==3.5,0][0],6.9375,places=5) # Existing braking is retained.
        self.assertAlmostEqual(out[-1,0],px[-1,0],places=5)
        self.assertAlmostEqual(curve['hold_starts_at_s'],7.,places=5)
        summary=curve['speed_edit_summary']
        self.assertLess(summary['stop_position']['position_error_m'],1e-4)
        self.assertAlmostEqual(summary['actual_start_speed_mps'],1.75,places=5)
        self.assertLess(curve['max_acceleration_mps2'],2.)
        from edit_feedback import intent_checks
        self.assertTrue(intent_checks(curve)['passed'])
        again=self.schedule.generate(t,lookup,op,3.,parent_curve=c,acceptance=self.policy)
        self.assertTrue(self.np.array_equal(out,again[0]));self.assertEqual(curve,again[2])
        child=self.position_request(start_time_s=4.,transition_style='constant_acceleration')
        x,a,next_curve=self.schedule.generate(t,lambda _: (out,angles),child,3.,parent_curve=curve,acceptance=self.policy)
        self.assertEqual(next_curve['speed_route'],curve['speed_route'])
        self.assertAlmostEqual(x[-1,0],out[-1,0],places=5)

    def test_sampled_foci_stop_needs_no_named_braking_segment(self):
        t,xy,yaw,px,pa,_=self.stopped_parent()
        out,angles,curve=self.schedule.generate(t,lambda _: (px,pa),self.position_request(),3.,acceptance=self.policy)
        self.assertTrue(self.np.array_equal(out[t<=3.5],px[t<=3.5]))
        self.assertAlmostEqual(out[-1,0],px[-1,0],places=5)
        self.assertEqual(curve['speed_edit_summary']['stop_position']['source'],'observed_stationary_tail')
        self.assertTrue(self.np.allclose(curve['terminal_velocity_xy_mps'],0.,atol=1e-8))

    def test_position_stop_does_not_invent_goal_for_a_moving_parent(self):
        from edit_feedback import EditFeasibilityError,failure_feedback
        t,xy,yaw=self.fixture()
        try:self.schedule.generate(t,lambda _: (xy,yaw),self.position_request(),3.,acceptance=self.policy)
        except EditFeasibilityError as error:
            self.assertEqual(error.code,'stop_position_unavailable')
            result=failure_feedback(error,self.position_request(preview_only=True),3.,12.)
            self.assertFalse(result['feasible']);self.assertFalse(result['candidate_created'])
            self.assertEqual(result['error_code'],'stop_position_unavailable')
        else:self.fail('Moving final frame was treated as a stopping point')

    def test_position_stop_rejects_fixed_history_excess_braking_and_off_route_goal(self):
        from edit_feedback import EditFeasibilityError
        t,xy,yaw,px,pa,c=self.stopped_parent()
        lookup=lambda k:(xy,yaw) if k=='recorded_baseline' else (px,pa)
        with self.assertRaisesRegex(EditFeasibilityError,'immutable_history'):
            self.schedule.generate(t,lookup,self.position_request(start_time_s=2.9),3.,parent_curve=c,acceptance=self.policy)
        with self.assertRaisesRegex(EditFeasibilityError,'target_unreachable'):
            self.schedule.generate(t,lookup,self.position_request(max_acceleration_mps2=.1),3.,parent_curve=c,acceptance=self.policy)
        from ego_speed_profile import continuous_parent
        c=continuous_parent(t,lookup,'p',c,3.)
        shifted=xy.copy();shifted[:,1]=.5
        lookup=lambda k:(shifted,yaw) if k=='recorded_baseline' else (px,pa)
        with self.assertRaisesRegex(EditFeasibilityError,'stop_position_off_route'):
            self.schedule.generate(t,lookup,self.position_request(path_source='recorded_baseline'),3.,parent_curve=c,acceptance=self.policy)

    def test_position_stop_reports_computed_time_when_it_exceeds_horizon(self):
        from edit_feedback import EditFeasibilityError
        t,xy,yaw=self.fixture()
        px,pa,c=self.schedule.generate(t,lambda _: (xy,yaw),self.request(
            targets=[dict(time_s=5.,speed_mps=4.),dict(time_s=7.,speed_mps=0.)]),3.,acceptance=self.policy)
        with self.assertRaises(EditFeasibilityError) as caught:
            self.schedule.generate(t,lambda _: (px,pa),self.position_request(start_time_s=3.),3.,parent_curve=c,acceptance=self.policy)
        self.assertEqual(caught.exception.code,'target_unreachable')
        self.assertGreater(caught.exception.limits['computed_stop_time_s'],t[-1])

    def test_curved_route_motion_bound_is_for_delivered_curve_not_tangential_cap(self):
        t,xy,yaw=self.fixture();np=self.np
        angle=.1*t;xy=np.column_stack([20*np.sin(angle),20*(1-np.cos(angle))]);yaw=angle
        op=self.request(targets=[dict(time_s=7.,speed_mps=0.)])
        x,a,c=self.schedule.generate(t,lambda _: (xy,yaw),op,3.,acceptance=self.policy)
        self.assertGreater(c['max_acceleration_mps2'],c['speed_edit_summary']['peak_tangential_acceleration_mps2'])
        self.assertLess(np.max(np.abs(np.diff(a))),.1)
        self.assertAlmostEqual(float(np.linalg.norm(c['terminal_velocity_xy_mps'])),0.)

    def test_sampled_curved_stop_reaches_route_endpoint_without_extrapolation(self):
        t,xy,yaw=self.fixture();np=self.np
        angle=.1*t;xy=np.column_stack([20*np.sin(angle),20*(1-np.cos(angle))]);yaw=angle
        px,pa,_=self.schedule.generate(t,lambda _: (xy,yaw),self.request(
            targets=[dict(time_s=7.,speed_mps=0.)]),3.,acceptance=self.policy)
        out,angles,curve=self.schedule.generate(t,lambda _: (px,pa),self.position_request(
            start_time_s=4.),3.,acceptance=self.policy)
        self.assertLess(float(np.linalg.norm(out[-1]-px[-1])),1e-4)
        from edit_feedback import intent_checks
        self.assertTrue(intent_checks(curve)['passed'])
        self.assertTrue(np.array_equal(out[t<=4.],px[t<=4.]))

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

    def test_current_schedule_version_is_explicit_and_removed_versions_are_rejected(self):
        t,xy,yaw=self.fixture();op=self.request()
        default=self.edit.modern_edit(t,lambda _: (xy,yaw),op,3.,acceptance=self.policy)
        explicit=self.edit.modern_edit(t,lambda _: (xy,yaw),dict(op,speed_schedule_version='state_continuity.v4'),3.,acceptance=self.policy)
        self.assertEqual(default[2]['profile'],'signed_speed_schedule.v4')
        self.assertEqual(default[2],explicit[2]);self.assertTrue(self.np.array_equal(default[0],explicit[0]))
        for version in ('route_reuse.v2','future'):
            with self.assertRaisesRegex(ValueError,'Unsupported speed schedule'):
                self.edit.modern_edit(t,lambda _: (xy,yaw),dict(op,speed_schedule_version=version),3.)

if __name__=='__main__':unittest.main()
