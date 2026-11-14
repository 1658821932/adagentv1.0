"""Regressions for identity edits, parked motion, direction and legacy joins."""
import importlib.util
from pathlib import Path
import sys,unittest
CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


@unittest.skipUnless(importlib.util.find_spec('scipy'),'requires numeric environment')
class RouteIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(CORE))
        import numpy as np
        import acceptance_policy
        from ego_route_edit import generate
        cls.np=np;cls.generate=staticmethod(generate);cls.policy=acceptance_policy.resolve()

    def request(self,**changes):
        args=dict(parent_candidate_id='p',start_time_s=3.25,left_offset_m=.1,
            transition_duration_s=1.,hold_duration_s=0.,return_duration_s=1.,
            end_behavior='return_to_route',reason='Verify route integrity')
        args.update(changes);return args

    def fixture(self):
        np=self.np;t=np.arange(0.,12.5,.5)
        return t,np.c_[5*t,np.zeros(len(t))],np.zeros(len(t))

    def test_zero_offset_preserves_positions_and_independent_body_yaw(self):
        from edit_feedback import intent_checks
        np=self.np;t,xy,yaw=self.fixture();angle=.08*t
        xy=np.c_[60*np.sin(angle),60*(1-np.cos(angle))];yaw=angle-.12
        for policy in ('preserve_speed','preserve_progress'):
            out,body,curve=self.generate(t,lambda _:(xy,yaw),self.request(left_offset_m=0.,speed_policy=policy),3.)
            self.assertTrue(np.array_equal(out,xy));self.assertTrue(np.array_equal(body,yaw))
            self.assertTrue(curve['route_edit_summary']['identity_edit'])
            self.assertTrue(intent_checks(curve)['passed'])

    def test_zero_offset_on_a_stationary_parent_is_an_identity(self):
        from edit_feedback import intent_checks
        np=self.np;t,xy,yaw=self.fixture();xy[:]=[20.,1.];yaw[:]=.3
        out,body,curve=self.generate(t,lambda _:(xy,yaw),self.request(left_offset_m=0.),3.)
        self.assertTrue(np.array_equal(out,xy));self.assertTrue(np.array_equal(body,yaw))
        self.assertTrue(intent_checks(curve)['passed'])

    def test_both_policies_reject_lateral_motion_after_stop_with_local_feedback(self):
        from ego_speed_schedule import generate as speed
        from edit_feedback import EditFeasibilityError,failure_feedback
        t,xy,yaw=self.fixture()
        px,pa,parent=speed(t,lambda _:(xy,yaw),dict(parent_candidate_id='p',operation='set_speed_targets',
            start_time_s=3.,targets=[dict(time_s=7.,speed_mps=0.)],reason='Prepare a stop'),3.)
        for policy in ('preserve_speed','preserve_progress'):
            args=self.request(start_time_s=8.,speed_policy=policy)
            with self.assertRaises(EditFeasibilityError) as caught:
                self.generate(t,lambda _:(px,pa),args,3.,parent_curve=parent)
            self.assertEqual(caught.exception.code,'stationary_lateral_motion')
            feedback=failure_feedback(caught.exception,dict(args,preview_only=True),3.,12.)
            self.assertFalse(feedback['candidate_created']);self.assertFalse(feedback['feasible'])
            self.assertEqual(feedback['error_code'],'stationary_lateral_motion')

    def test_lateral_transition_crossing_stop_is_rejected_but_completed_offset_can_hold(self):
        from ego_speed_schedule import generate as speed
        from edit_feedback import EditFeasibilityError,intent_checks
        t,xy,yaw=self.fixture()
        px,pa,parent=speed(t,lambda _:(xy,yaw),dict(parent_candidate_id='p',operation='set_speed_targets',
            start_time_s=3.,targets=[dict(time_s=7.,speed_mps=0.)],reason='Prepare a stop'),3.)
        for policy in ('preserve_speed','preserve_progress'):
            with self.assertRaises(EditFeasibilityError):
                self.generate(t,lambda _:(px,pa),self.request(start_time_s=6.5,speed_policy=policy),3.,parent_curve=parent)
            args=self.request(start_time_s=4.,speed_policy=policy,end_behavior='keep_offset')
            args.pop('hold_duration_s');args.pop('return_duration_s')
            out,body,curve=self.generate(t,lambda _:(px,pa),args,3.,parent_curve=parent)
            self.assertTrue(intent_checks(curve)['passed'])
            self.assertTrue(self.np.allclose(out[t>=7.],out[t==7.],atol=1e-8))

    def test_route_intent_detects_undeclared_reverse_even_with_matching_offsets(self):
        from copy import deepcopy
        from edit_feedback import intent_checks
        t,xy,yaw=self.fixture()
        _,_,curve=self.generate(t,lambda _:(xy,yaw),self.request(),3.)
        broken=deepcopy(curve)
        angles=self.np.array(broken['continuous_state']['yaw_coefficients'])
        angles[-1]+=self.np.pi
        broken['continuous_state']['yaw_coefficients']=angles.tolist()
        intent=intent_checks(broken)
        self.assertFalse(intent['passed']);self.assertFalse(intent['checks']['route_forward_motion'])

    def test_legacy_micro_join_is_C2_bounded_and_has_no_artificial_peak(self):
        from ego_trajectory_edit import brake_along_path
        from ego_speed_profile import continuous_parent
        from edit_continuity import parent_motion
        from scipy.interpolate import PPoly
        np=self.np;t=np.arange(0.,12.5,.5);t[9]=4.499962
        xy=np.c_[12*t+.2*t*t,np.zeros(len(t))];yaw=np.zeros(len(t))
        operation=dict(parent_candidate_id='p',path_candidate_id='recorded_baseline',start_time_s=4.5254096,
            deceleration_mps2=2.4,target_speed_mps=0.,motion_intent='forward_stop',end_behavior='hold')
        px,pa,curve=brake_along_path(t,xy,yaw,xy,yaw,operation,3.,with_details=True)
        lookup=lambda key:(xy,yaw) if key=='recorded_baseline' else (px,pa)
        motion=parent_motion(t,lookup,'p',curve,3.,3.25)
        connection=motion['parent_motion_source']['connection']
        self.assertLessEqual(connection['maximum_sample_adjustment_m'],.01)
        self.assertLess(connection['peak_acceleration_mps2'],6.)
        raw=continuous_parent(t,lookup,'p',curve,3.)['continuous_state']
        original=PPoly(np.array(raw['xy_coefficients']),np.array(raw['times_s']))
        cont=motion['continuous_state'];p=PPoly(np.array(cont['xy_coefficients']),np.array(cont['times_s']))
        join=original.x[0];before=np.nextafter(join,-np.inf)
        for derivative in (0,1,2):
            self.assertTrue(np.allclose(p(before,derivative),original(join,derivative),atol=1e-7))
        out,body,result=self.generate(t,lookup,self.request(left_offset_m=0.),3.,parent_curve=curve)
        self.assertTrue(np.array_equal(out,px));self.assertTrue(np.array_equal(body,pa))
        self.assertLess(result['max_acceleration_mps2'],6.)

    def test_quintic_extrema_and_arbitrary_time_crop_are_exact(self):
        from scipy.interpolate import BPoly,PPoly
        from ego_trajectory_edit import curve_evidence
        from edit_continuity import restrict_curve
        np=self.np
        p=PPoly.from_bernstein_basis(BPoly.from_derivatives([0.,2.],
            [[[0.,0.],[0.,0.],[0.,0.]],[[2.,1.],[0.,0.],[0.,0.]]]))
        evidence=curve_evidence(p,{},'Test quintic interior extrema')
        q=np.linspace(0.,2.,10001);dense=float(np.max(np.linalg.norm(p(q,2),axis=1)))
        self.assertAlmostEqual(evidence['max_acceleration_mps2'],dense,places=5)
        cropped=restrict_curve(p,.37);q=np.linspace(.37,2.,30)
        for derivative in (0,1,2):self.assertTrue(np.allclose(p(q,derivative),cropped(q,derivative),atol=1e-10))

    def test_legacy_identity_supplies_yaw_and_holds_through_scene_end(self):
        from ego_trajectory_edit import brake_along_path
        from scipy.interpolate import PPoly
        np=self.np;t,xy,yaw=self.fixture()
        op=dict(parent_candidate_id='p',path_candidate_id='recorded_baseline',start_time_s=4.,
            deceleration_mps2=2.,target_speed_mps=0.,motion_intent='forward_stop',end_behavior='hold')
        px,pa,parent=brake_along_path(t,xy,yaw,xy,yaw,op,3.,with_details=True)
        lookup=lambda key:(xy,yaw) if key=='recorded_baseline' else (px,pa)
        for start in (4.5,8.):
            out,body,curve=self.generate(t,lookup,self.request(start_time_s=start,left_offset_m=0.),3.,parent_curve=parent)
            self.assertTrue(np.array_equal(out,px));self.assertTrue(np.array_equal(body,pa))
            cont=curve['continuous_state'];p=PPoly(np.asarray(cont['xy_coefficients']),np.asarray(cont['times_s']))
            self.assertEqual(p.x[-1],t[-1]);self.assertLess(np.linalg.norm(p(t[-1],1)),1e-8)
            self.assertTrue(np.allclose(p(t[-1]),px[-1],atol=1e-8))

    def test_sampled_stop_does_not_invent_reverse_overshoot(self):
        from edit_continuity import parent_motion
        from scipy.interpolate import PPoly
        np=self.np;t=np.arange(0.,6.5,.5)
        x=2*t;x[t>=3.5]=6.1;x[t>=4.]=6.1001
        xy=np.c_[x,np.zeros(len(t))];yaw=np.zeros(len(t))
        curve=parent_motion(t,lambda _:(xy,yaw),'p',None,3.,3.25)
        cont=curve['continuous_state'];p=PPoly(np.asarray(cont['xy_coefficients']),np.asarray(cont['times_s']))
        q=np.linspace(3.5,6.,1001)
        self.assertGreaterEqual(float(p(q,1)[:,0].min()),-1e-8)
        self.assertTrue(np.allclose(p(t),xy,atol=1e-8))
        self.assertTrue(curve['parent_motion_source']['terminal_overshoot_limited'])


if __name__=='__main__':unittest.main()
