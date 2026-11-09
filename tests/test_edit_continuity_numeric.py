"""State continuity, body-centre semantics and acceptance across sample seams."""
import importlib.util
from pathlib import Path
import sys
import unittest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


@unittest.skipUnless(importlib.util.find_spec('scipy'),'requires numeric environment')
class ContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(CORE))
        import numpy as np
        import acceptance_policy
        cls.np=np;cls.policy=acceptance_policy.resolve()

    def fixture(self,speed=5.,radius=None):
        np=self.np;t=np.unique(np.r_[np.arange(0,12.5,.5),3.000015])
        dense=np.unique(np.r_[np.arange(0,12.01,.1),t,2.99989,3.,3.000011])
        def poses(q):
            if radius is None:return np.c_[speed*q,np.zeros(len(q))],np.full(len(q),.005)
            angle=speed*q/radius
            return np.c_[radius*np.sin(angle),radius*(1-np.cos(angle))],angle-np.arcsin(1.35/radius)
        x,y=poses(t);dx,dy=poses(dense)
        return t,x,y,dict(reference_point='target_body_center',rear_axle_offset_m=1.35,samples=(dense,dx,dy))

    def test_boundary_heading_steering_and_all_prefix_samples_are_continuous(self):
        from ego_speed_schedule import generate
        from scipy.interpolate import PPoly
        from vehicle_motion import check_bicycle
        np=self.np;t,x,y,context=self.fixture()
        args=dict(parent_candidate_id='p',edit_mode='speed_schedule',operation='set_speed_targets',start_time_s=3.,targets=[dict(time_s=7.,speed_mps=0.)],reason='Stop without changing the observed history')
        out,yaw,curve=generate(t,lambda _:(x,y),args,3.,acceptance=self.policy,parent_context=context)
        q=curve['continuous_state'];p=PPoly(np.asarray(q['xy_coefficients']),np.asarray(q['times_s']));a=PPoly(np.asarray(q['yaw_coefficients']),np.asarray(q['times_s']))
        dt,dx,dy=context['samples'];dense_x=dx.copy();dense_y=dy.copy();mask=dt>3.;dense_x[mask]=p(dt[mask]);dense_y[mask]=a(dt[mask])
        self.assertTrue(np.array_equal(out[t<=3],x[t<=3]));self.assertTrue(np.array_equal(dense_x[~mask],dx[~mask]))
        check=check_bicycle(dt,dense_x,dense_y,[4.5,2.,1.6])
        self.assertTrue(check['passed'],check)
        self.assertLess(abs(a(3.,1)),1e-7)
        self.assertEqual(curve['actual_start_state']['reference_point'],'target_body_center')

    def test_curved_centre_route_uses_body_heading_and_actual_targets(self):
        from ego_speed_schedule import generate
        from edit_feedback import intent_checks
        np=self.np;t,x,y,context=self.fixture(4.,40.)
        args=dict(parent_candidate_id='p',operation='set_speed_targets',start_time_s=3.,targets=[dict(time_s=7.,speed_mps=1.)],reason='Retain a curved known route')
        out,yaw,curve=generate(t,lambda _:(x,y),args,3.,acceptance=self.policy,parent_context=context)
        self.assertTrue(intent_checks(curve)['passed'])
        self.assertEqual(curve['body_heading']['method'],'centre_velocity_with_rear_axle_constraint')
        self.assertTrue(np.isfinite(yaw).all());self.assertLess(curve['max_acceleration_mps2'],6.)

    def test_arbitrary_time_route_does_not_mix_left_interval_and_current_speed(self):
        from ego_route_edit import generate
        np=self.np;t,x,y,context=self.fixture()
        def path(q):
            u=np.maximum(0.,q-3.);return np.c_[5*q-.2*u*u,np.zeros(len(q))]
        x=path(t);dt,_,dy=context['samples'];context['samples']=(dt,path(dt),dy)
        args=dict(parent_candidate_id='p',start_time_s=3.25,left_offset_m=.1,transition_duration_s=2.,return_duration_s=2.,end_behavior='return_to_route',speed_policy='preserve_progress',reason='Gentle return on a decelerating parent')
        out,yaw,curve=generate(t,lambda _:(x,y),args,3.,acceptance=self.policy,parent_context=context)
        self.assertLess(curve['max_acceleration_mps2'],2.)
        self.assertTrue(np.array_equal(out[t<=3.25],x[t<=3.25]))

    def test_shared_contract_rejects_target_and_hold_errors_and_reports_true_peak(self):
        from copy import deepcopy
        from ego_speed_schedule import generate
        from edit_feedback import apply_edit_contract,intent_checks
        t,x,y,context=self.fixture()
        args=dict(parent_candidate_id='p',operation='set_speed_targets',start_time_s=3.,targets=[dict(time_s=7.,speed_mps=0.)],reason='Check target acceptance')
        _,_,curve=generate(t,lambda _:(x,y),args,3.,acceptance=self.policy,parent_context=context)
        result=dict(checks={},reference_screen_passed=True,max_speed_mps=1.,max_acceleration_mps2=.1)
        apply_edit_contract(result,curve)
        self.assertEqual(result['max_acceleration_mps2'],curve['max_acceleration_mps2'])
        broken=deepcopy(curve);broken['speed_edit_summary']['target_evidence'][0]['actual_speed_mps']=.2
        self.assertFalse(intent_checks(broken)['passed'])
        broken=deepcopy(curve);broken['speed_edit_summary']['preservation_evidence']=dict(error=.1,tolerance=1e-4)
        self.assertFalse(intent_checks(broken)['passed'])

    def test_infeasible_feedback_has_numeric_limits_and_no_silent_target_change(self):
        from ego_speed_schedule import generate
        from edit_feedback import failure_feedback
        t,x,y,context=self.fixture()
        args=dict(parent_candidate_id='p',operation='set_speed_targets',start_time_s=3.,targets=[dict(time_s=3.01,speed_mps=0.)],reason='Unreachable deadline')
        with self.assertRaises(ValueError) as caught:generate(t,lambda _:(x,y),args,3.,acceptance=self.policy,parent_context=context)
        feedback=failure_feedback(caught.exception,args,3.,12.)
        self.assertEqual(feedback['error_code'],'target_unreachable')
        self.assertGreater(feedback['feasibility_limits']['minimum_transition_duration_s'],.01)
        self.assertEqual(args['targets'][0]['time_s'],3.01)

    def test_attack_preview_and_creation_use_the_same_peak_and_intent_gate(self):
        from copy import deepcopy
        from attack_editor import AttackEditor
        from ego_speed_schedule import generate
        t,x,y,context=self.fixture()
        args=dict(parent_candidate_id='p',operation='set_speed_targets',start_time_s=3.,targets=[dict(time_s=7.,speed_mps=0.)],reason='Check the attack acceptance gate')
        _,_,curve=generate(t,lambda _:(x,y),args,3.,acceptance=self.policy,parent_context=context)
        editor=object.__new__(AttackEditor);editor.constraints=dict(acceptance_policy=self.policy)
        curve['speed_edit_summary']['target_evidence'][0]['actual_speed_mps']=.2
        raw=dict(checks=dict(road=True),max_speed_mps=1.,max_acceleration_mps2=.1)
        preview=editor._motion_curve_checks(deepcopy(raw),deepcopy(curve))
        created=editor._motion_curve_checks(deepcopy(raw),deepcopy(curve))
        self.assertEqual(preview,created)
        self.assertFalse(preview['reference_screen_passed'])
        self.assertIn('edit_intent',preview['failed_checks'])
        self.assertEqual(preview['max_acceleration_mps2'],curve['max_acceleration_mps2'])

    def test_speed_radius_offset_sweep_preserves_goals_and_finite_motion(self):
        from ego_route_edit import generate
        from edit_feedback import intent_checks
        np=self.np
        for speed in (2.,5.,10.):
            for radius in (None,60.):
                for offset in (-.5,.5):
                    with self.subTest(speed=speed,radius=radius,offset=offset):
                        t,x,y,context=self.fixture(speed,radius)
                        args=dict(parent_candidate_id='p',start_time_s=3.25,left_offset_m=offset,transition_duration_s=2.,return_duration_s=2.,end_behavior='return_to_route',speed_policy='preserve_progress',reason='Check broad numerical capability')
                        out,yaw,curve=generate(t,lambda _:(x,y),args,3.,acceptance=self.policy,parent_context=context)
                        self.assertTrue(np.isfinite(out).all() and np.isfinite(yaw).all())
                        self.assertTrue(np.array_equal(out[t<=3.25],x[t<=3.25]))
                        self.assertTrue(intent_checks(curve)['passed'])
                        self.assertLess(curve['max_acceleration_mps2'],6.)


    def test_route_after_speed_keeps_exact_requested_times_without_roundoff_pieces(self):
        from ego_speed_schedule import generate as speed
        from ego_route_edit import generate as route
        np=self.np;t,x,y,context=self.fixture()
        op=dict(parent_candidate_id='p',operation='set_speed_targets',start_time_s=3.25,targets=[dict(time_s=7.,speed_mps=0.)],reason='Stop parent')
        px,pa,parent=speed(t,lambda _:(x,y),op,3.,acceptance=self.policy,parent_context=context)
        request=dict(parent_candidate_id='p',start_time_s=4.25,left_offset_m=.1,transition_duration_s=1.,return_duration_s=1.,end_behavior='return_to_route',reason='Return before stopping')
        out,yaw,curve=route(t,lambda _:(px,pa),request,3.,parent_curve=parent,acceptance=self.policy,parent_context=context)
        self.assertGreater(np.min(np.diff(curve['continuous_state']['times_s'])),1e-9)
        self.assertIn(5.25,curve['continuous_state']['times_s'])
        self.assertLess(curve['max_acceleration_mps2'],6.)

    def test_earlier_retiming_preserves_legacy_explicit_hold_beyond_curve_end(self):
        from ego_trajectory_edit import brake_along_path
        from ego_speed_schedule import generate
        from edit_feedback import intent_checks
        np=self.np;t=np.arange(0.,12.5,.5);xy=np.c_[2*t,np.zeros(len(t))];yaw=np.zeros(len(t))
        low=dict(parent_candidate_id='p',path_candidate_id='recorded_baseline',start_time_s=4.,deceleration_mps2=.5,target_speed_mps=0.,motion_intent='forward_stop',end_behavior='hold')
        px,pa,parent=brake_along_path(t,xy,yaw,xy,yaw,low,3.,with_details=True)
        lookup=lambda key:(xy,yaw) if key=='recorded_baseline' else (px,pa)
        op=dict(parent_candidate_id='p',operation='retime_braking',braking_start_shift_s=-.5,preserve='stop_position',reason='Earlier inherited braking')
        out,angles,curve=generate(t,lookup,op,3.,parent_curve=parent,acceptance=self.policy)
        self.assertTrue(intent_checks(curve)['passed']);self.assertLess(np.linalg.norm(out[-1]-px[-1]),1e-4)

    def test_real_attack_and_ego_method_boundaries_attach_the_shared_summary(self):
        from attack_editor import AttackEditor
        from ego_editor import EgoEditor
        arguments=dict(parent_candidate_id='p',edit_mode='speed_schedule',operation='set_speed_targets',
            start_time_s=3.,targets=[dict(time_s=7.,speed_mps=0.)],preview_only=True,reason='Check shared tool feedback')
        raw=dict(status='preview',feasible=True,candidate_created=False)
        attack=object.__new__(AttackEditor);attack._edit_motion=lambda value:dict(raw)
        ego=object.__new__(EgoEditor);ego._edit=lambda value:dict(raw)
        attack_result=AttackEditor.edit_motion(attack,arguments)
        ego_result=EgoEditor.edit(ego,arguments)
        self.assertEqual(attack_result['decision_summary']['outcome'],'preview_passed')
        self.assertEqual(ego_result['decision_summary'],attack_result['decision_summary'])


if __name__=='__main__':unittest.main()
