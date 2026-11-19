"""Editing time, WE history and attack-danger time are independent contracts."""
import importlib.util
from pathlib import Path
import sys
import unittest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


@unittest.skipUnless(importlib.util.find_spec('scipy') and importlib.util.find_spec('shapely'),'Requires numerical runtime')
class EditableObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(CORE))
        import numpy as np
        import acceptance_policy as policy
        import ego_trajectory_edit as edit
        from vehicle_trajectory import screen_attack,attack_time_grid
        from ego_road_checks import OfficialDrivableMap
        from shapely.geometry import box
        cls.np=np;cls.policy=policy;cls.edit=edit
        cls.screen=staticmethod(screen_attack);cls.grid=staticmethod(attack_time_grid)
        cls.road=OfficialDrivableMap([box(-100,-20,200,20)],{})

    def fixture(self):
        np=self.np;t=np.arange(0.,8.5,.5)
        return t,np.c_[2*t,np.zeros(len(t))],np.zeros(len(t))

    def test_fresh_policy_and_legacy_policy_have_distinct_edit_boundaries(self):
        fresh=self.policy.resolve();old=self.policy.resolve(dict(policy_version='reference_acceptance.v1',observation_end_s=3.))
        self.assertEqual(self.policy.edit_start(fresh),0.)
        self.assertEqual(fresh['observation_end_s'],1.5)
        self.assertEqual(self.policy.danger_start(fresh),1.5)
        self.assertEqual(self.policy.edit_start(old),3.)
        self.assertNotIn('edit_start_s',old)
        fresh_from_old=self.policy.for_new_run(old)
        self.assertEqual(self.policy.edit_start(fresh_from_old),0.)
        self.assertEqual(self.policy.danger_start(fresh_from_old),1.5)
        with self.assertRaises(ValueError):self.policy.resolve(dict(attack_danger_start_s=-.1))

    def test_speed_edit_at_zero_and_one_second_changes_history_and_preserves_initial_state(self):
        np=self.np;t,x,y=self.fixture()
        for start in (0.,1.):
            op=dict(edit_mode='speed_schedule',operation='set_speed_targets',parent_candidate_id='p',
                start_time_s=start,targets=[dict(time_s=start+3.,speed_mps=1.)],
                transition_style='smooth',reason='Smooth early preparation')
            out,angles,curve=self.edit.modern_edit(t,lambda _: (x,y),op,3.,acceptance=self.policy.resolve())
            self.assertTrue(np.array_equal(out[t<=start],x[t<=start]))
            self.assertGreater(np.max(abs(out[t<3]-x[t<3])),.01)
            self.assertEqual(curve['actual_start_state']['time_s'],start)

    def test_route_edit_can_start_during_observation(self):
        np=self.np;t,x,y=self.fixture()
        op=dict(edit_mode='route_edit',parent_candidate_id='p',start_time_s=.5,left_offset_m=.2,
            transition_duration_s=2.,end_behavior='keep_offset',speed_policy='preserve_speed',reason='Early smooth steering')
        out,angles,curve=self.edit.modern_edit(t,lambda _: (x,y),op,3.,acceptance=self.policy.resolve())
        self.assertTrue(np.array_equal(out[t<=.5],x[t<=.5]))
        self.assertGreater(abs(out[t==2.5,1][0]),.1)

    def attack(self,early_near=False,legacy=False,initial_distance=20.):
        np=self.np;t,x,y=self.fixture();dense=self.grid(t,3.)
        ego=np.c_[t*4,np.zeros(len(t))]
        seed=np.c_[t*(4 if early_near else 2)+(7 if early_near else initial_distance),np.zeros(len(t))]
        edited=np.c_[dense*(4 if early_near else 2)+(7 if early_near else initial_distance),np.zeros(len(dense))]
        if not early_near:edited[:,0]-=.1*dense**2
        actors=[[dict(track='target',x=float(p[0]),y=0.,yaw=0.,length=4.,width=2.,label='vehicle')] for p in seed]
        # These numerical scenarios explicitly request no danger delay. Default
        # production windows are verified separately from editability at t=0.
        acceptance=self.policy.resolve(dict(policy_version='reference_acceptance.v1',observation_end_s=3.)
                                       if legacy else dict(attack_danger_start_s=0.))
        return self.screen(t,ego,y,actors,dict(track_id='target',size_m=[4.,2.,1.5]),seed,y,
            dense,edited,np.zeros(len(dense)),self.road,
            dict(acceptance_policy=acceptance,observation_end_s=3.,max_ego_clearance_m=2.))

    def test_late_attack_remains_valid_with_editable_observation_history(self):
        result=self.attack()
        self.assertTrue(result['reference_screen_passed'],result['failed_checks'])
        self.assertTrue(result['checks']['observation_prefix'])
        self.assertNotIn('no_risk_before_danger_start',result['checks'])
        self.assertGreater(result['first_contact_time_s'],3.)
        self.assertGreater(result['geometry_feedback']['time_contract']['observation_trajectory_change_m'],.5)
        old=self.attack(legacy=True)
        self.assertFalse(old['checks']['observation_prefix'])
        self.assertNotIn('no_risk_before_danger_start',old['checks'])

    def test_near_interaction_during_observation_is_allowed_without_an_implicit_time_gate(self):
        result=self.attack(early_near=True)
        self.assertTrue(result['checks']['no_contact_before_explicit_event_start'])
        self.assertNotIn('no_risk_before_danger_start',result['checks'])
        self.assertTrue(result['reference_screen_passed'],result['failed_checks'])

    def test_foci_can_modify_history_and_zero_edit_is_still_exact_parent(self):
        from run_ego_foci_probe import FociProblem
        np=self.np;t,x,y=self.fixture()
        actors=[[dict(track='target',x=60.,y=0.,yaw=0.,length=4.,width=2.)] for _ in t]
        p=FociProblem(t,x,y,actors,target_track='target',freeze_before_s=0.,edit_window_s=[0.,8.],
            observation_end_s=3.,edit_start_s=0.)
        zero=np.zeros_like(p.initial_control)
        self.assertTrue(np.array_equal(p.trajectory(zero)[0],x))
        delta=zero.copy();delta[~p.frozen,0]=.1
        out,angles=p.trajectory(delta)
        self.assertTrue(np.array_equal(out[0],x[0]))
        self.assertGreater(np.max(abs(out[t<3]-x[t<3])),.01)

    def test_attack_edit_and_event_windows_can_both_start_before_three_seconds(self):
        from attack_objective import validate_objective
        t,_,_=self.fixture()
        op=dict(parent_candidate_id='p',objective=dict(interaction='either',interaction_window_s=[4.,8.]),
            edit_window_s=[0.,8.],max_candidates=1,reason='Prepare early, interact later')
        validate_objective(op,t,0.,edit_start_s=0.)
        op['objective']['interaction_window_s']=[0.,8.]
        validate_objective(op,t,0.,edit_start_s=0.)
        op['objective']['interaction_window_s']=[-.1,8.]
        with self.assertRaises(ValueError):validate_objective(op,t,0.,edit_start_s=0.)

    def test_attack_contact_before_three_seconds_is_allowed_in_new_runs(self):
        result=self.attack(initial_distance=12.)
        self.assertTrue(result['reference_screen_passed'],result['failed_checks'])
        self.assertGreater(result['first_contact_time_s'],0.)
        self.assertLess(result['first_contact_time_s'],3.)

    def test_constructor_controls_can_start_at_scene_zero(self):
        from vehicle_control_search import integrate_controls
        np=self.np;t,x,y=self.fixture();dense=self.grid(t,3.)
        controls=dict(times_s=[0.,8.],acceleration_mps2=[-.1,-.1],steering_rad=[0.,0.])
        arrays,state=integrate_controls(t,x,y,dense,[4.,2.,1.5],controls,0.,acceptance=self.policy.resolve())
        self.assertTrue(np.array_equal(arrays[0][0],x[0]))
        self.assertGreater(np.max(abs(arrays[0][t<3]-x[t<3])),.01)

    def test_behavior_metrics_include_the_entire_new_editable_horizon(self):
        from defense_behavior import DrivingObjective
        np=self.np;t,x,y=self.fixture()
        actors=[[dict(track='target',x=60.,y=0.,yaw=0.,length=4.,width=2.)] for _ in t]
        objective=DrivingObjective(t,x,actors,'target',self.policy.resolve())
        result=objective.evaluate(x,y)
        self.assertEqual(objective.boundary,0.)
        self.assertEqual(float(objective.grid[0]),0.)
        self.assertAlmostEqual(result['net_reference_progress_m'],16.)
        self.assertAlmostEqual(result['net_reference_progress_fraction'],1.)

    def test_freeze_contact_guard_accepts_the_current_explicit_boundary(self):
        from attack_timing import contact_after_observation
        self.assertTrue(contact_after_observation([0.,2.,4.],[1],earliest_contact_s=0.))
        self.assertTrue(contact_after_observation([0.,2.,4.],[1]))  # current 1.5s default
        self.assertFalse(contact_after_observation([0.,2.,4.],[1],earliest_contact_s=3.))


if __name__=='__main__':unittest.main()
