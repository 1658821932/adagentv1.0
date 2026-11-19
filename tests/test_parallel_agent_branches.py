"""Main Agent can start both branches and must collect them before finishing."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from agent.config import AgentConfig
from agent.main import Agent


def test_main_collects_both_background_jobs_and_rejects_premature_finish(tmp_path,monkeypatch):
    def call(tool,control):
        return dict(action='call_tool',tool_name=tool,arguments={'_execution':control},reason='offline')
    decisions=[call('ego_trajectory_editor',{'background':True}),call('policy_runner',{'background':True}),
        call('ego_trajectory_editor',{'job_id':'policy'}),
        dict(action='finish',reason='too early'),
        call('ego_trajectory_editor',{'job_id':'expert','background':False}),
        call('policy_runner',{'job_id':'policy','background':False}),dict(action='finish',reason='both collected')]
    observations=[]
    def decide(task,state):
        observations.append(state)
        return decisions.pop(0)
    fake=SimpleNamespace(decide=decide,llm_client=SimpleNamespace(model='offline',base_url='offline'))
    monkeypatch.setattr('agent.main.Orchestrator',lambda **kw:fake)
    monkeypatch.setattr('agent.main.prewarm.start',lambda:None)
    agent=Agent(replace(AgentConfig.load(),logs_root=tmp_path,max_steps=7))
    started=set()
    def execute(tool,args):
        key='expert' if tool=='ego_trajectory_editor' else 'policy'
        if args['_execution'].get('background'):
            assert key not in started
            started.add(key)
            return dict(status='running',tool=tool,job={'job_id':key})
        assert started=={'expert','policy'}
        if args['_execution'].get('job_id') != key:
            return dict(status='error', tool=tool, error=dict(type='execution_error', message='Job belongs to another tool'))
        return dict(status='success',tool=tool,result={'completed':True}, job={'job_id':key})
    agent.executor.execute=Mock(side_effect=execute)
    agent.executor.query=Mock(side_effect=AssertionError('Background startup must not block'))
    result=agent.run('Generate the paired policy and expert response',max_steps=7)
    assert result['status']=='finished' and result['background_jobs']=={}
    assert set(observations[2]['background_jobs'])=={'expert','policy'}
    assert set(observations[3]['background_jobs'])=={'expert','policy'}
    assert agent.executor.execute.call_count==5
