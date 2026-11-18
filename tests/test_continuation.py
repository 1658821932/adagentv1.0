"""Offline conversation continuation, old checkpoints and durable job recovery."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.config import AgentConfig
from agent.continuation import recover
from agent.jobs import JobStore
from agent.logger import RunLogger
from agent.main import Agent
from agent.orchestrator import Orchestrator
from agent.state import AgentState


class Client:
    model='offline-fixture'
    base_url='https://example.invalid/v1'
    def __init__(self, decisions):
        self.decisions=list(decisions)
        self.messages=[]
    def call_functions(self, messages, tools):
        self.messages.append(deepcopy(messages))
        name, payload=self.decisions.pop(0)
        return dict(message=dict(role='assistant',content=None,tool_calls=[
            dict(id='fixture',type='function',function=dict(name=name,arguments=json.dumps(payload)))]),
            finish_reason='tool_calls')


@pytest.fixture(autouse=True)
def no_prewarm(monkeypatch):
    monkeypatch.setattr('agent.main.prewarm.start',lambda:None)


def make_agent(root, decisions, max_steps=3):
    config=AgentConfig.load()
    config=replace(config,logs_root=root,max_steps=max_steps,
                   tool_modes={name:'mock' for name in config.tool_modes})
    client=Client(decisions)
    with patch('agent.main.Orchestrator',return_value=Orchestrator(client)):
        agent=Agent(config)
    return agent,client


def parent_run(root):
    logger=RunLogger('原任务：完整攻防并评估',10,'fixture','https://example.invalid',{},logs_root=root)
    state=AgentState('原任务：完整攻防并评估')
    state.execution_scope=logger.run_id
    for step,tool,result in [
        (0,'video_retriever',dict(scene_ref=dict(scene_id='fixture',scene_token='fixture',
            sample_root='/fixture',dataset='fixture',matched_camera='CAM_F0',frame_count=24,
            manifest_path='/fixture/manifest.json',frames_path='/fixture/frames.json',cameras_path='/fixture/cameras.json',
            images_complete=True,cameras=dict(CAM_F0=dict(frame_count=24,available_images=24,images_complete=True))),
            evidence=dict(frames=['bulk']*1000))),
        (1,'edit_planner',dict(suitability='ready',reasons=['fixture'],scene_id='A',scene_token='A',
            candidate_id='cand_1',target_scene='fixture scene',
            attack=dict(map_context='curve',ego_action='straight',ego_lane_keeping='keep',
                        other_origin='none',other_action='none',other_lane_keeping='none'),
            checks=dict(road='pass',lanes='pass',space='pass'),
            adapted_plan=dict(attack_edit_strategy=None,ego_input_edit_strategy=None,
                ego_correct_trajectory_strategy='安全减速并继续前进'),
            retrieval_suggestion=None,evidence=dict(frames=['bulk']*100))),
    ]:
        state.set_step(step)
        state.add_decision(step,dict(action='call_tool',tool_name=tool,arguments=dict(fixture=True),reason='fixture'))
        logger.start_tool_call(step,tool,dict(fixture=True))
        envelope=dict(status='success',tool=tool,result=result)
        reference=logger.finish_tool_call(step,tool,envelope)
        state.add_tool_result(tool,envelope,result_ref=reference)
    state.set_step(2)
    state.add_decision(2,dict(action='stop',reason='old backend error'))
    state.mark_error('old backend error')
    logger.finalize(state.to_dict())
    return logger


def fingerprint(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def test_followup_reuses_history_goal_references_and_original_execution_scope(tmp_path):
    parent=parent_run(tmp_path)
    before=fingerprint(parent.run_dir)
    reference=str(parent.run_dir/'tools/step_00_video_retriever/output.json')
    agent,client=make_agent(tmp_path,[
        ('attack_vehicle_editor',dict(reason='repair complete',arguments=dict(scene_ref=reference,
            operation='edit_existing',target_track='car',instruction='fixture'),execution=dict(retry=True))),
        ('finish',dict(reason='本轮只测试恢复后的攻击工具')),
    ])
    with patch.object(agent.executor,'execute',return_value=dict(status='success',tool='attack_vehicle_editor',
                                                                result=dict(status='fixture'))) as execute:
        state=agent.run('程序已修复，只重试攻击编辑',resume_from=parent.run_id)
    assert state['status']=='finished'
    context=json.loads(client.messages[0][1]['content'])
    assert context['task']=='原任务：完整攻防并评估'
    current=context['current_state']
    assert current['current_request']=='程序已修复，只重试攻击编辑'
    assert current['resume_context']['previous_error']=='old backend error'
    assert any(entry.get('result_ref')==reference for entry in current['history'])
    assert len(json.dumps(context,ensure_ascii=False))<18000
    assert state['user_messages']==[dict(content='程序已修复，只重试攻击编辑',after_run_id=parent.run_id)]
    assert execute.call_count==1 and execute.call_args.args[0]=='attack_vehicle_editor'
    assert execute.call_args.args[1]['_execution']['retry'] is True
    assert agent.executor.scope==parent.run_id
    metadata=json.loads(agent.logger.run_metadata_path.read_text())
    assert metadata['parent_run_id']==parent.run_id and metadata['conversation_id']==parent.run_id
    assert (agent.logger.run_dir/'initial_state.json').is_file()
    assert (agent.logger.run_dir/'tools/step_03_attack_vehicle_editor/input.json').is_file()
    assert fingerprint(parent.run_dir)==before


def test_second_followup_keeps_original_task_and_previous_messages_once(tmp_path):
    parent=parent_run(tmp_path)
    first,_=make_agent(tmp_path,[('finish',dict(reason='fixture'))])
    first.run('先只解释原因',resume_from=parent.run_id)
    second,client=make_agent(tmp_path,[('finish',dict(reason='fixture'))])
    state=second.run('现在继续',resume_from=first.logger.run_id)
    assert state['task']=='原任务：完整攻防并评估'
    assert [m['content'] for m in state['user_messages']]==['先只解释原因','现在继续']
    assert second.executor.scope==parent.run_id
    assert second.logger.run_metadata['conversation_id']==parent.run_id
    assert len(state['history'])==7
    assert json.loads(client.messages[0][1]['content'])['current_state']['current_request']=='现在继续'


def test_old_checkpoint_recovers_completed_tool_files_but_not_private_streams(tmp_path):
    parent=parent_run(tmp_path)
    saved=json.loads(parent.final_state_path.read_text())
    saved['history']=saved['history'][:2]
    saved['current_step']=0
    saved.pop('execution_scope')
    parent.final_state_path.unlink()
    parent.save_state(saved)
    (parent.run_dir/'thinking.jsonl').write_text('private provider data must not be restored')
    state,lineage,pending=recover(tmp_path,parent.run_id,'继续')
    assert pending is None
    assert len([e for e in state.history if e.get('tool')=='edit_planner'])==1
    assert state.current_step==1
    assert lineage['execution_scope']==parent.run_id
    assert 'private provider data' not in json.dumps(state.to_dict())


def test_reconnect_completed_job_after_parent_died_before_result_checkpoint(tmp_path):
    parent=RunLogger('分析一次',3,'fixture','https://example.invalid',{},logs_root=tmp_path/'runs')
    state=AgentState('分析一次');state.execution_scope=parent.run_id
    state.add_decision(0,dict(action='call_tool',tool_name='failure_analyzer',arguments=dict(token='fixture'),reason='fixture'))
    parent.start_tool_call(0,'failure_analyzer',dict(token='fixture'))
    parent.save_state(state.to_dict())
    jobs=JobStore(tmp_path/'jobs')
    job_id=jobs.submit('failure_analyzer','mock',dict(token='fixture'),parent.run_id)
    assert jobs.wait(job_id,10)['status']=='success'
    state.mark_error('process interrupted before collecting result')
    parent.finalize(state.to_dict())
    agent,client=make_agent(tmp_path/'runs',[('finish',dict(reason='已恢复真实结果'))])
    agent.executor.jobs=jobs
    with patch.object(jobs,'submit',wraps=jobs.submit) as submit:
        result=agent.run('继续',resume_from=parent.run_id)
    assert result['status']=='finished' and submit.call_count==0
    assert len(client.messages)==1
    context=json.loads(client.messages[0][1]['content'])['current_state']
    assert any(e.get('tool')=='failure_analyzer' and e['result']['status']=='success' for e in context['history'])
    recovered=json.loads((agent.logger.run_dir/'tools/step_01_failure_analyzer/input.json').read_text())
    assert recovered['_execution']['job_id']==job_id


def test_continuation_rejects_active_parent_and_escaping_paths(tmp_path):
    parent=RunLogger('fixture',1,'fixture','https://example.invalid',{},logs_root=tmp_path)
    with pytest.raises(ValueError,match='仍在执行'):
        recover(tmp_path,parent.run_id,'continue')
    for value in ('../outside','/tmp/run_20200101_000000_1','run_unknown'):
        with pytest.raises(ValueError):
            recover(tmp_path,value,'continue')
    parent.finalize(dict(status='finished',current_step=0))
    other=tmp_path.parent/(tmp_path.name+'-outside');other.mkdir()
    (tmp_path/'run_20990101_000000_1').symlink_to(other,target_is_directory=True)
    with pytest.raises(ValueError):
        recover(tmp_path,'run_20990101_000000_1','continue')


def test_unrecorded_pending_job_does_not_repeat_side_effects(tmp_path):
    parent=RunLogger('fixture',1,'fixture','https://example.invalid',{},logs_root=tmp_path)
    parent.start_tool_call(0,'policy_trainer',dict(fixture=True))
    parent.finalize(AgentState('fixture').to_dict())
    agent,_=make_agent(tmp_path,[])
    agent.executor.jobs=JobStore(tmp_path/'jobs')
    with patch.object(agent.executor,'execute') as execute:
        with pytest.raises(RuntimeError,match='no persistent job'):
            agent.run('继续',resume_from=parent.run_id)
    assert execute.call_count==0


def test_old_parent_cannot_fork_a_second_active_turn_in_the_same_conversation(tmp_path):
    parent=parent_run(tmp_path)
    _,lineage,_=recover(tmp_path,parent.run_id,'继续')
    child=RunLogger('继续',1,'fixture','https://example.invalid',{},logs_root=tmp_path,lineage=lineage)
    with pytest.raises(ValueError,match='仍在执行'):
        recover(tmp_path,parent.run_id,'再试一次')
    child.finalize(dict(status='finished',current_step=3))
    state,_,_=recover(tmp_path,parent.run_id,'已结束后可重试')
    assert state.current_request=='已结束后可重试'


def test_cli_resume_flag_preserves_literal_followup_without_network(tmp_path,monkeypatch):
    import runpy
    import sys
    from agent import prewarm
    parent=parent_run(tmp_path)
    config=replace(AgentConfig.load(),logs_root=tmp_path,max_steps=1)
    client=Client([('finish',dict(reason='CLI offline fixture'))])
    text='--literal-option 继续，不重新检索'
    monkeypatch.setattr(sys,'argv',['agent.main','--resume-from',parent.run_id,'--',text])
    monkeypatch.setattr(prewarm,'start',lambda:None)
    with patch('agent.config.AgentConfig.load',return_value=config), patch('agent.orchestrator.LLMClient',return_value=client):
        runpy.run_module('agent.main',run_name='__main__')
    context=json.loads(client.messages[0][1]['content'])
    assert context['task']=='原任务：完整攻防并评估'
    assert context['current_state']['current_request']==text
