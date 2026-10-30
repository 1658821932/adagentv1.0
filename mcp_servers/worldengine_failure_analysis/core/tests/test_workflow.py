import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evidence import Evidence,steps
from vlm_review import run,validate_final


def final():
    return {'action':'final','observations':['核验后的现象'],
        'candidate_assessment':[{'id':'A','verdict':'supported','evidence_ids':['E1'],'explanation':{'required_observation':'自车速度','actual_observation':'速度上升','missing_evidence':'无'}},
                                {'id':'B','verdict':'rejected','evidence_ids':['E2'],'explanation':{'required_observation':'周车突然减速','actual_observation':'速度恒定','missing_evidence':'无'}}],
        'selected_candidate_id':'A','failure_reason':'测试行为原因','reason_category':'测试类别'}


class FakeEvidence:
    row={'token':'test-token'}
    def __init__(self):
        self.images={f'I_overview_{i:02d}':{'id':f'I_overview_{i:02d}'} for i in range(12)};self.queries=[]
    def content(self,images):return []
    def select_images(self,requests):return [{'id':'I_bev_07'}]
    def query(self,request):self.queries.append(request);return {'rows':[{'test':1}],'request':request}


class FakeClient:
    def __init__(self,early=False):self.early=early;self.agent_requests=[]
    def ask(self,messages,name):
        if not name.startswith('turn_'):return {'observations':['测试图像观察'],'uncertainties':[]}
        self.agent_requests.append(messages.copy());i=int(name.split('_')[-1])
        actions=[{'action':'images','purpose':'查看BEV','requests':[{'camera':'bev','steps':[7]}]},
            {'action':'visual_complete','observations':['测试现象'],'candidates':[{'id':'A','reason':'候选A'},{'id':'B','reason':'候选B'}]},
            {'action':'numbers','queries':[{'kind':'ego','steps':[6,7]},{'kind':'actors','steps':[6,7]}]},final()]
        if self.early:actions.insert(0,{'action':'numbers','queries':[{'kind':'ego','steps':[6]}]})
        return actions[i]


class WorkflowTests(unittest.TestCase):
    def test_vlm_selects_images_then_candidates_then_numbers_and_reason(self):
        with tempfile.TemporaryDirectory() as d:
            evidence=FakeEvidence();client=FakeClient()
            result=run(client,evidence,{'metrics':{},'source':'official.csv','source_sha256':'hash'},Path(d))
            self.assertEqual(result['failure_reason'],'测试行为原因')
            self.assertEqual([q['kind'] for q in evidence.queries],['ego','actors'])
            trace=json.loads((Path(d)/'trace.json').read_text())
            self.assertEqual([e['action']['action'] for e in trace],['images','visual_complete','numbers','final'])

    def test_early_numeric_query_is_rejected_and_vlm_can_correct(self):
        with tempfile.TemporaryDirectory() as d:
            evidence=FakeEvidence()
            run(FakeClient(early=True),evidence,{'metrics':{},'source':'official.csv','source_sha256':'hash'},Path(d))
            trace=json.loads((Path(d)/'trace.json').read_text())
            self.assertIn('tool_error',trace[0]['response']);self.assertEqual(len(evidence.queries),2)

    def test_final_cannot_skip_numeric_verification(self):
        with self.assertRaises(ValueError):validate_final(final(),{'A':{},'B':{}},set())

    def test_final_cannot_cite_invented_numeric_evidence(self):
        answer=final();answer['candidate_assessment'][0]['evidence_ids']=['E999']
        with self.assertRaises(ValueError):validate_final(answer,{'A':{},'B':{}},{'E1','E2'})

    def test_final_accepts_viewed_images_but_not_unseen_images(self):
        answer=final();answer['candidate_assessment'][0]['evidence_ids'].append('I_bev_07')
        validate_final(answer,{'A':{},'B':{}},{'E1','E2'},{'I_bev_07'})
        with self.assertRaises(ValueError):validate_final(answer,{'A':{},'B':{}},{'E1','E2'},set())

    def test_image_citation_does_not_replace_numeric_verification(self):
        answer=final();answer['candidate_assessment'][0]['evidence_ids']=['I_bev_07']
        with self.assertRaises(ValueError):validate_final(answer,{'A':{},'B':{}},{'E1','E2'},{'I_bev_07'})

    def test_final_cannot_select_rejected_candidate(self):
        answer=final();answer['selected_candidate_id']='B'
        with self.assertRaises(ValueError):validate_final(answer,{'A':{},'B':{}},{'E1','E2'})

    def test_verdict_needs_explicit_evidence_match(self):
        answer=final();answer['candidate_assessment'][0]['explanation']='泛泛的原因说明'
        with self.assertRaises(ValueError):validate_final(answer,{'A':{},'B':{}},{'E1','E2'})

    def test_invalid_steps_are_not_read(self):
        for values in ([-1],[12],[True],[],[3.5]):
            with self.assertRaises(ValueError):steps(values)

    def test_world_actor_speed_is_not_relative_gap_rate(self):
        e=Evidence(Path('.'),{'result_root':'.'}, {},[],Path('.'))
        e.data={'frames':[{'step':s,'heading':0} for s in (3,4)],'actor_measurements':[
            {'track_token':'car','step':3,'center_x_m':100,'center_y_m':0,'distance_m':10,'relative_longitudinal_velocity_fd_mps':0},
            {'track_token':'car','step':4,'center_x_m':105,'center_y_m':0,'distance_m':10,'relative_longitudinal_velocity_fd_mps':0}]}
        result=e.query({'kind':'actors','steps':[4],'track_token':'car'})
        self.assertEqual(result['rows'][0]['world_speed_fd_mps'],10)
        self.assertEqual(result['rows'][0]['relative_longitudinal_velocity_fd_mps'],0)

    def test_missing_actor_data_is_empty_not_stationary(self):
        e=Evidence(Path('.'),{'result_root':'.'}, {},[],Path('.'));e.data={'frames':[{'step':3,'heading':0}],'actor_measurements':[]}
        self.assertEqual(e.query({'kind':'actors','steps':[3],'track_token':'missing'})['rows'],[])

    def test_axis_overlap_is_not_a_full_collision(self):
        e=Evidence(Path('.'),{'result_root':'.'}, {},[{}, {}, {}, {'anns':{
            'track_tokens':['car'],'gt_boxes':[[5,0,0,4,2,1,0]]}}],Path('.'))
        e.data={'frames':[{'step':3,'heading':0}], 'vehicle_parameters':{'length':4,'width':2,'rear_axle_to_center':0},
                'actor_measurements':[{'track_token':'car','step':3,'center_x_m':5,'center_y_m':0,'distance_m':1}]}
        row=e.query({'kind':'actors','steps':[3]})['rows'][0]
        self.assertEqual(row['lateral_axis_gap_m'],-2)
        self.assertEqual(row['longitudinal_axis_gap_m'],1)
        self.assertEqual(row['distance_m'],1)

    def test_turn_budget_never_returns_a_fabricated_final(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):run(FakeClient(),FakeEvidence(),{'metrics':{},'source':'official.csv','source_sha256':'hash'},Path(d),max_turns=1)


if __name__=='__main__':unittest.main()
