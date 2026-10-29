import importlib.util
import json
from pathlib import Path
import subprocess

MODULE=Path(__file__).resolve().parents[1]/'tools/scripts/check_commit_safety.py'
spec=importlib.util.spec_from_file_location('commit_safety',MODULE)
checker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_staged_secret_is_reported_without_echoing_value(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    value='sk-'+'x'*40
    (tmp_path/'.env').write_text('KIMI_API_KEY='+value+'\n')
    (tmp_path/'unsafe.py').write_text('credential='+repr(value)+'\n')
    subprocess.run(['git','add','unsafe.py'],cwd=tmp_path,check=True)
    report=checker.check(tmp_path)
    assert report['status']=='findings_require_review'
    assert 'matches_configured_secret' in report['findings'][0]['rules']
    assert value not in json.dumps(report)


def test_ignored_file_cannot_be_smuggled_via_force_add(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'.gitignore').write_text('.env.local\n')
    (tmp_path/'.env.local').write_text('KIMI_API_KEY=\n')
    subprocess.run(['git','add','-f','.env.local'],cwd=tmp_path,check=True)
    report=checker.check(tmp_path)
    assert report['findings'][0]['rules']==['local_environment_file']


def test_empty_template_is_allowed(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'.env.example').write_text('KIMI_API_KEY=\n')
    subprocess.run(['git','add','.env.example'],cwd=tmp_path,check=True)
    assert checker.check(tmp_path)['status']=='passed'
