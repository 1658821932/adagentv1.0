"""The constructor accepts caller 'either' without weakening its event requirement."""
import ast
import types
from pathlib import Path

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial/attack_objective.py'


def functions(**extra):
    nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)
           and n.name in {'validate_objective','objective_evidence'}]
    namespace=dict(np=np,**extra)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace


def test_constructor_accepts_either_and_still_validates_unknown_modes():
    validate=functions()['validate_objective']
    arguments=dict(parent_candidate_id='seed_000',max_candidates=1,reason='event required',
                   objective=dict(interaction='either',interaction_window_s=[3.5,5.]))
    validate(arguments,np.array([0.,3.,4.,5.]),3.)
    assert arguments['objective']['interaction']=='either'
    arguments['objective']['interaction']='unknown'
    with pytest.raises(ValueError,match='interaction must'):
        validate(arguments,np.array([0.,3.,4.,5.]),3.)


@pytest.mark.parametrize('mode,gap,contact_index,expected',[
    ('either',.5,None,True),
    ('either',0.,2,True),
    ('either',0.,1,False),
    ('either',5.,None,False),
    ('either',.05,None,False),
    ('allowed',5.,None,True),
    ('near_miss',0.,2,False),
    ('contact',.5,None,False),
])
def test_either_requires_contact_or_real_near_miss(mode,gap,contact_index,expected):
    times=np.array([0.,3.,4.,5.])
    gaps=np.full(4,gap)
    overlaps=np.zeros(4)
    if contact_index is not None:
        overlaps[contact_index]=1.
    geometry=types.SimpleNamespace(distance=lambda *args:gaps,
        intersection=lambda *args:overlaps,area=lambda value:value)
    zero=np.zeros(4)
    namespace=functions(shapely=geometry,policy=types.SimpleNamespace(
        ATTACK_NEAR_MISS_MIN_GAP_M=.15,ATTACK_EVENT_MAX_GAP_M=2.),
        EGO_FRONT=4.,EGO_REAR=1.,EGO_WIDTH=2.,
        relative_state=lambda *args:(zero,zero,zero,zero),bodies=lambda *args:None)
    evidence=namespace['objective_evidence'](times,np.zeros((4,2)),zero,np.zeros((4,2)),zero,[4.,2.],
        dict(interaction=mode,interaction_window_s=[3.5,5.],keyframes=[]))
    assert evidence['goal_met'] is expected
    assert ('requested_interaction' in evidence['failed_checks']) is (not expected)
