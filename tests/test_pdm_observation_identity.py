"""Reference-ego overlap is a valid obstacle, not proof of ego identity."""
import ast
from pathlib import Path

import pytest

SOURCE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial/run_pdm_closed_loop.py'


def predicate():
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='is_ego_annotation')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace['is_ego_annotation']


def test_coincident_attack_vehicle_is_retained():
    assert not predicate()(dict(track='synthetic_vehicle_001',label='vehicle',x=0,y=0))


@pytest.mark.parametrize('obj',[dict(track='ego',label='vehicle'),dict(track='other',label='EGO')])
def test_explicit_ego_identity_is_excluded(obj):
    assert predicate()(obj)
