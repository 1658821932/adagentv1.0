"""Numerical semantics: lateral goals, speed tradeoff, smooth return and replay."""
import sys,unittest,importlib.util
from pathlib import Path
CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
@unittest.skipUnless(importlib.util.find_spec('scipy'),'requires numeric environment')
class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(CORE))
        import numpy as np
        import ego_route_edit
        import acceptance_policy
        cls.np=np;cls.edit=ego_route_edit;cls.policy=acceptance_policy.resolve()
    def fixture(self):
        np=self.np;t=np.arange(0.,12.5,.5);return t,np.column_stack([5*t,np.zeros(len(t))]),np.zeros(len(t))
    def request(self,**changes):
        d=dict(edit_mode='route_edit',parent_candidate_id='p',start_time_s=3.,left_offset_m=.5,transition_duration_s=2.,end_behavior='return_to_route',hold_duration_s=1.,return_duration_s=2.,reason='Clear obstacle and return')
        d.update(changes);return d
    def test_left_return_keeps_speed_but_spends_forward_progress_and_child_uses_edited_route(self):
        np=self.np;t,xy,yaw=self.fixture();x,a,c=self.edit.generate(t,lambda _: (xy,yaw),self.request(),3.,acceptance=self.policy)
        self.assertTrue(np.array_equal(x[t<=3],xy[t<=3]));self.assertAlmostEqual(x[t==5,1][0],.5,places=7)
        self.assertAlmostEqual(x[t==8,1][0],0.,places=7);self.assertLess(x[-1,0],xy[-1,0])
        self.assertLess(c['route_edit_summary']['max_speed_change_at_construction_knots_mps'],1e-6)
        self.assertLess(c['max_acceleration_mps2'],2.)
        from ego_speed_schedule import generate
        child=dict(edit_mode='speed_schedule',parent_candidate_id='p',operation='set_speed_targets',start_time_s=5.,targets=[dict(time_s=9.,speed_mps=0.)],reason='Brake on edited route')
        y,b,d=generate(t,lambda _: (x,a),child,3.,parent_curve=c,acceptance=self.policy)
        self.assertEqual(c['speed_route'],d['speed_route']);self.assertTrue(np.array_equal(y[t<=5],x[t<=5]))
        again=self.edit.generate(t,lambda _: (xy,yaw),self.request(),3.,acceptance=self.policy)
        self.assertTrue(np.array_equal(x,again[0]));self.assertEqual(c,again[2])
    def test_progress_policy_reports_actual_speed_increase_and_right_offset(self):
        t,xy,yaw=self.fixture();args=self.request(left_offset_m=-.5,speed_policy='preserve_progress')
        x,a,c=self.edit.generate(t,lambda _: (xy,yaw),args,3.,acceptance=self.policy)
        self.assertLess(x[t==5,1][0],0.);self.assertAlmostEqual(x[-1,0],xy[-1,0],places=5)
        self.assertGreater(c['route_edit_summary']['max_speed_change_at_construction_knots_mps'],.01)
    def test_lateral_movement_while_stopped_and_horizon_violation_are_rejected(self):
        t,xy,yaw=self.fixture();xy[:]=0
        with self.assertRaisesRegex(ValueError,'route_exhausted'):self.edit.generate(t,lambda _: (xy,yaw),self.request(),3.)
        t,xy,yaw=self.fixture()
        with self.assertRaisesRegex(ValueError,'target_unreachable'):self.edit.generate(t,lambda _: (xy,yaw),self.request(left_offset_m=4.,transition_duration_s=.1),3.)
        with self.assertRaisesRegex(ValueError,'beyond scene end'):self.edit.generate(t,lambda _: (xy,yaw),self.request(return_duration_s=20.),3.)
if __name__=='__main__':unittest.main()
