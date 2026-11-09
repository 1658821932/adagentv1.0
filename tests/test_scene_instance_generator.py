"""Standalone compact tool02 tests; no external model calls."""
import json
from copy import deepcopy
from pathlib import Path
import pytest
from tools.tool02_scene_instance_generator import real, mock
from tools.tool02_scene_instance_generator.schemas import Input
from tools.registry import call_tool
from tools.tool03_video_retriever.schemas import Input as RetrievalInput

GROUP={'id':'G-cross','name':'Cross traffic','base_scene':'Urban intersection',
       'wrong_behavior':'Delayed yielding to cross traffic.',
       'cases':[{'case':'C-test','token':'source-token'}]}
FIELDS={'scene_requirement','target_effect','candidates'}

@pytest.fixture(autouse=True)
def records_in_tmp(monkeypatch,tmp_path):
    monkeypatch.setattr(real,'RECORDS_ROOT',tmp_path/'records')

class Client:
    model='offline-test'
    def __init__(self,*responses):self.responses=list(responses);self.calls=0;self.messages=[]
    def chat(self,messages):
        self.calls+=1;self.messages.append(deepcopy(messages))
        value=self.responses.pop(0)
        if isinstance(value,Exception):raise value
        return json.dumps(value)

def response():
    out=mock.run({'failure_analysis':GROUP})
    assert out['status']=='success',out
    return out['result']

def call(client,**kwargs):
    return real.run({'failure_analysis':deepcopy(GROUP),'_llm_client':client,**kwargs})

def test_four_fields_only_and_source_is_recorded_outside_result():
    before=deepcopy(GROUP);client=Client(response());out=call(client)
    assert out['status']=='success',out
    assert set(out['result'])==FIELDS
    path=Path(out['meta']['record_dir'])
    saved=json.loads((path/'input.json').read_text())
    assert saved['failure_analysis']==before==GROUP
    assert '_llm_client' not in saved
    assert json.loads((path/'output.json').read_text())==out
    assert 'source_analysis' not in out['result']

def test_successful_upstream_envelope_and_no_optional_inputs():
    analysis={'failure_reason':'Late yielding','failure_type':['cross traffic']}
    out=mock.run({'failure_analysis':{'status':'success','result':{'failure_analysis':analysis}}})
    assert out['status']=='success'
    assert set(out['result'])==FIELDS
    assert Input.model_validate({'failure_analysis':analysis}).variation_hints=={}

@pytest.mark.parametrize('value',[{}, {'anything':'text'}, {'failure_reason':''}, {'failure_reason':'x','failure_type':42}])
def test_invalid_input_before_model(value):
    client=Client();out=real.run({'failure_analysis':value,'_llm_client':client})
    assert out['error']['type']=='input_validation_error'
    assert client.calls==0

def test_failed_upstream_is_rejected():
    out=mock.run({'failure_analysis':{'status':'error','result':{'failure_analysis':{'failure_reason':'x'}}}})
    assert out['error']['type']=='input_validation_error'

def test_optional_hints_are_reflected_in_mock():
    out=mock.run({'failure_analysis':GROUP,'variation_hints':{'target_actor':'crossing_bus','lighting':'dusk'}})
    assert all('dusk' in item['base_video_requirement'] for item in out['result']['candidates'])
    assert 'crossing_bus' in out['result']['scene_requirement']
    assert 'dusk' in out['result']['scene_requirement']

def test_legacy_qualitative_alias():
    with pytest.raises(ValueError):
        Input.model_validate({'failure_analysis':GROUP,'parameter_assignment':{'lighting':'dusk'}})

def test_exact_duplicate_corrected_once_with_previous_envelope():
    item=response();new=deepcopy(item);new['scene_requirement']='New cross-traffic requirement with an initially occluded crossing vehicle.'
    old=mock.run({'failure_analysis':GROUP});client=Client(item,new)
    out=call(client,existing_instances=[old])
    assert out['status']=='success',out
    assert client.calls==2
    assert out['result']==new

def test_old_scene_instance_summary_is_rejected():
    with pytest.raises(ValueError):
        Input.model_validate({'failure_analysis':GROUP,'existing_instances':[{'status':'success','result':{'scene_instance':'legacy summary'}}]})

def test_duplicate_exhausts_bounded_retries():
    item=response();client=Client(item,item)
    out=call(client,existing_instances=[' '+item['scene_requirement'].upper()+' '])
    assert out['error']['type']=='output_validation_error'
    assert client.calls==2

@pytest.mark.parametrize('field',sorted(FIELDS))
def test_each_output_field_is_required(field):
    item=response();del item[field]
    assert call(Client(item,item))['error']['type']=='output_validation_error'

def test_removed_fields_are_rejected_instead_of_leaking():
    item=response();item['source_analysis']={'unwanted':'payload'}
    assert call(Client(item,item))['error']['type']=='output_validation_error'

def test_each_candidate_passes_directly_to_tool04():
    out=mock.run({'failure_analysis':GROUP})
    assert len(out['result']['candidates'])==3
    for candidate in out['result']['candidates']:
        args={'requirement':candidate['base_video_requirement']}
        RetrievalInput.model_validate(args)
        received=call_tool('video_retriever',args,mode='mock')
        assert received['status']=='success'
        assert received['result']['requirement']==candidate['base_video_requirement']

@pytest.mark.parametrize('count',[0,2,4])
def test_candidate_count_must_be_three(count):
    item=response();item['candidates']=(item['candidates']*2)[:count]
    assert call(Client(item,item))['error']['type']=='output_validation_error'

def test_repeated_candidate_pairs_are_rejected():
    item=response();item['candidates']=[item['candidates'][0]]*3
    assert call(Client(item,item))['error']['type']=='output_validation_error'


def test_transport_error_is_not_retried_as_json_error():
    client=Client(RuntimeError('Network unavailable'));out=call(client)
    assert out['error']['type']=='llm_request_error'
    assert client.calls==1

@pytest.mark.parametrize('extra',[{'cases':None},{'supporting_case_ids':'not-a-list'}])
def test_invalid_source_shapes_rejected(extra):
    client=Client();out=real.run({'failure_analysis':{**GROUP,**extra},'_llm_client':client})
    assert out['error']['type']=='input_validation_error'
    assert client.calls==0

def test_uncertainty_not_forwarded_and_input_not_mutated():
    analysis={**GROUP,'uncertainty':'legacy','forensics':{'uncertainty':'legacy','observations':['cross traffic']}}
    before=deepcopy(analysis);client=Client(response())
    out=real.run({'failure_analysis':analysis,'_llm_client':client})
    assert out['status']=='success'
    assert 'uncertainty' not in json.dumps(client.messages)
    saved=json.loads((Path(out['meta']['record_dir'])/'input.json').read_text())
    assert 'uncertainty' not in saved['failure_analysis']
    assert analysis==before
