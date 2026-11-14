"""Required wire mode preserves forced-phase functions and candidate enum gates."""
import pytest
from mcp_servers.scene_editing.core.adversarial.ego_tool_transport import require_native_call,validate_exposed_call


def tool(name,ids=None):
    spec=dict(type='string')
    if ids is not None:spec['enum']=ids
    return dict(function=dict(name=name,parameters=dict(properties=dict(candidate_id=spec),
        required=['candidate_id'],additionalProperties=False)))


def test_force_is_translated_to_one_required_tool():
    tools=[tool('inspect',['a']),tool('submit',['b'])]
    choice,exposed=require_native_call(dict(function=dict(name='inspect')),tools)
    assert choice=='required' and exposed==[tools[0]]
    exposed[0]['function']['parameters']['properties']['candidate_id']['enum'].append('other')
    assert tools[0]['function']['parameters']['properties']['candidate_id']['enum']==['a']


def test_unforced_phases_also_require_a_native_call():
    tools=[tool('inspect'),tool('submit')]
    assert require_native_call('auto',tools)==('required',tools)
    with pytest.raises(ValueError):require_native_call('none',tools)
    with pytest.raises(ValueError):require_native_call('required',[])


def test_auto_image_transport_requires_explicit_capability_and_retains_forced_gate():
    messages=[dict(role='user',content=[dict(type='text',text='diagram'),
        dict(type='image_url',image_url=dict(url='fixture'))])]
    tools=[tool('inspect',['a']),tool('submit',['b'])]
    assert require_native_call('auto',tools,messages=messages)[0]=='required'
    choice,exposed=require_native_call(dict(function=dict(name='inspect')),tools,
        messages=messages,image_tool_choice='auto')
    assert choice=='auto' and exposed==[tools[0]]
    assert messages[0]['content'][1]['type']=='image_url'
    assert require_native_call('auto',tools,messages=[dict(role='user',content='text')],image_tool_choice='auto')[0]=='required'


def test_other_tools_and_unreviewed_ids_are_rejected_locally():
    exposed=[tool('inspect',['a'])]
    with pytest.raises(ValueError,match='not exposed'):validate_exposed_call('submit',dict(candidate_id='a'),exposed)
    with pytest.raises(ValueError,match='whitelist'):validate_exposed_call('inspect',dict(candidate_id='b'),exposed)
    validate_exposed_call('inspect',dict(candidate_id='a'),exposed)


def test_missing_and_extra_fields_rejected():
    exposed=[tool('inspect')]
    with pytest.raises(ValueError,match='Missing'):validate_exposed_call('inspect',{},exposed)
    with pytest.raises(ValueError,match='Unexpected'):validate_exposed_call('inspect',dict(candidate_id='a',extra=1),exposed)
