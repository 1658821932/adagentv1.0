import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.tool04_video_retriever import real
from mcp_servers.sentry_video_search.adapter import client


@pytest.mark.parametrize('library',['nuplan','waymo'])
def test_requirement_only_request_obeys_configured_library(tmp_path,library):
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/video_retriever.json').write_text(json.dumps({'default_library':library}))
    python=tmp_path/'python';python.write_text('fixture')
    proc=SimpleNamespace(returncode=0,stdout=json.dumps({'status':'success','result':{'video_path':'/matched.mp4'}}),stderr='')
    with patch.object(client,'PROJECT_ROOT',tmp_path),patch.object(client,'VIDEO_RETRIEVER_PYTHON',python),patch.object(client.subprocess,'run',return_value=proc) as send:
        result=real.run({'requirement':'urban road'})
    assert result['result']['video_path']=='/matched.mp4'
    assert json.loads(send.call_args.kwargs['input'])['library']==library


def test_public_tool_imports_do_not_load_chromadb_or_component_services():
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    code="""
import sys
class RejectServiceImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'chromadb' or fullname.startswith('chromadb.') or fullname == 'mcp_servers.sentry_video_search.service':
            raise AssertionError('Component dependency loaded inside caller: '+fullname)
sys.meta_path.insert(0,RejectServiceImports())
from tools.tool04_video_retriever.real import run as retrieve
from tools.tool06_reconstruction.real import run as reconstruct
assert callable(retrieve) and callable(reconstruct)
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
