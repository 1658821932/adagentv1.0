"""Auto wire mode preserves forced-phase functions and candidate enum gates."""
import pytest
from mcp_servers.scene_editing.core.adversarial.ego_tool_transport import constrained_auto,validate_exposed_call


def tool(name,ids=None):
    spec=dict(type='string')
    if ids is not None:spec['enum']=ids
    return dict(function=dict(name=name,parameters=dict(properties=dict(candidate_id=spec),
        required=['candidate_id'],additionalProperties=False)))


def test_force_is_translated_to_one_declared_tool_with_auto():
    tools=[tool('inspect',['a']),tool('submit',['b'])]
    choice,exposed=constrained_auto(dict(function=dict(name='inspect')),tools)
    assert choice=='auto' and exposed==[tools[0]]
    exposed[0]['function']['parameters']['properties']['candidate_id']['enum'].append('other')
    assert tools[0]['function']['parameters']['properties']['candidate_id']['enum']==['a']


def test_other_tools_and_unreviewed_ids_are_rejected_locally():
    exposed=[tool('inspect',['a'])]
    with pytest.raises(ValueError,match='not exposed'):validate_exposed_call('submit',dict(candidate_id='a'),exposed)
    with pytest.raises(ValueError,match='whitelist'):validate_exposed_call('inspect',dict(candidate_id='b'),exposed)
    validate_exposed_call('inspect',dict(candidate_id='a'),exposed)


def test_missing_and_extra_fields_rejected():
    exposed=[tool('inspect')]
    with pytest.raises(ValueError,match='Missing'):validate_exposed_call('inspect',{},exposed)
    with pytest.raises(ValueError,match='Unexpected'):validate_exposed_call('inspect',dict(candidate_id='a',extra=1),exposed)
