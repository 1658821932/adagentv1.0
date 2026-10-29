import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from dataclasses import replace

import pytest

from mcp_servers.sentry_video_search.config import SearchConfig
from mcp_servers.sentry_video_search import kimi_backend as backend
from mcp_servers.sentry_video_search.service import index_local


def options():
    return {'base_url':'https://example.invalid/v1','model':'kimi-k3','api_key_env':'KIMI_API_KEY',
            'fallback_api_key_env':'LLM_API_KEY','timeout_seconds':30,'max_tokens':1000,
            'caption_frames':8,'rerank_frames':4,'frame_width':640,'candidate_count':4,
            'query_expansion':False,'minimum_relevance_score':.4}


def caption(text='公交车在城市道路上行驶'):
    return {'summary_zh':text,'summary_en':'bus on an urban road','road_layout':'urban road',
            'ego_motion':'forward','visible_actors':['公交车 bus'],'observed_interactions':[],
            'tags':['公交车','bus'],'uncertainties':[]}


def test_chat_uses_configured_gateway_model_and_image_url(monkeypatch,tmp_path):
    monkeypatch.setenv('KIMI_API_KEY','unit-test-secret')
    image=tmp_path/'frame.jpg';image.write_bytes(b'fake-jpeg-unit-fixture')
    response=Mock(status_code=200)
    response.json.return_value={'id':'req-test','model':'kimi-k3','usage':{'total_tokens':10},
        'choices':[{'message':{'content':'```json\n{"ok":true}\n```'},'finish_reason':'stop'}]}
    with patch.object(backend.requests,'post',return_value=response) as send:
        value,trace=backend.KimiClient(options()).chat('test prompt',[image])
    assert value=={'ok':True}
    assert send.call_args.args[0]=='https://example.invalid/v1/chat/completions'
    payload=send.call_args.kwargs['json']
    assert payload['model']=='kimi-k3'
    assert payload['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert trace['image_count']==1
    assert 'unit-test-secret' not in json.dumps(trace)


def test_forbidden_key_never_rotates_to_another_credential(monkeypatch):
    monkeypatch.setenv('KIMI_API_KEY','explicit-key')
    monkeypatch.setenv('LLM_API_KEY','different-key-must-not-be-tried')
    response=Mock(status_code=403)
    response.json.return_value={'error':{'message':'no access for explicit-key'}}
    with patch.object(backend.requests,'post',return_value=response) as send:
        with pytest.raises(backend.KimiAPIError,match='403') as error:
            backend.KimiClient(options()).chat('literal test')
    assert send.call_count==1
    assert send.call_args.kwargs['headers']['Authorization']=='Bearer explicit-key'
    assert 'explicit-key' not in str(error.value)
    assert 'different-key' not in str(error.value)


def test_empty_reasoning_only_response_is_not_a_caption(monkeypatch):
    monkeypatch.setenv('KIMI_API_KEY','key')
    response=Mock(status_code=200)
    response.json.return_value={'choices':[{'message':{'content':'','reasoning_content':'internal reasoning'},'finish_reason':'length'}]}
    with patch.object(backend.requests,'post',return_value=response):
        with pytest.raises(backend.KimiAPIError,match='no answer content'):
            backend.KimiClient(options()).chat('test')


def test_caption_index_resumes_and_invalidates_model_change(tmp_path):
    video=tmp_path/'front.mp4';video.write_bytes(b'unit video fixture')
    config=SearchConfig.for_library('nuplan')
    config.runtime_home=tmp_path/'runtime'
    client=Mock()
    client.chat.return_value=(caption(),{'request_id':'test','image_count':8})
    frame=tmp_path/'frame.jpg';frame.write_bytes(b'jpeg')
    config_values=options()
    with patch.object(backend,'settings',return_value=config_values),patch.object(backend,'KimiClient',return_value=client),\
         patch.object(backend,'_get_video_duration',return_value=10.),\
         patch.object(backend,'sample_frames',return_value=([frame]*8,list(range(8)))):
        first=backend.index_videos([video],config)
        second=backend.index_videos([video],config)
        assert first['indexed_file_count']==1
        assert second['indexed_file_count']==0
        assert client.chat.call_count==1
        config_values['model']='different-model'
        third=backend.index_videos([video],config)
        assert third['indexed_file_count']==1
        assert client.chat.call_count==2
    assert len(backend.CaptionStore(config.runtime_home).rows())==1


def test_local_chinese_english_recall_ranks_matching_scene_first():
    scores=backend.recall_scores('公交车 bus',['城市道路 公交车 bus','parking parked cars 停车场','rainy highway 下雨高速'])
    assert scores[0]>scores[1] and scores[0]>scores[2]
    assert scores[0]>0


def test_no_match_flags_override_high_model_scores():
    value={'ranked_results':[{'candidate_index':0,'relevance_score':.99,'matched':False,'reason':'missing pedestrian'},
        {'candidate_index':1,'relevance_score':.8,'matched':True,'reason':'pedestrian visible'}]}
    ranked=backend.validate_ranking(value,2)
    assert ranked[0]['candidate_index']==1
    assert ranked[1]['relevance_score']==0


@pytest.mark.parametrize('score',[float('nan'),float('inf'),1.1,-.1,True])
def test_invalid_model_scores_are_rejected(score):
    value={'ranked_results':[{'candidate_index':0,'relevance_score':score,'matched':True,'reason':'test'}]}
    with pytest.raises(ValueError,match='score'):
        backend.validate_ranking(value,1)


def test_repeated_or_missing_candidates_are_rejected():
    row={'candidate_index':0,'relevance_score':.7,'matched':True,'reason':'test'}
    with pytest.raises(ValueError,match='Repeated'):
        backend.validate_ranking({'ranked_results':[row,row]},2)
    with pytest.raises(ValueError,match='every candidate'):
        backend.validate_ranking({'ranked_results':[row]},2)


def test_nuplan_does_not_call_dashscope_indexer(tmp_path):
    video=tmp_path/'clip.mp4';video.write_bytes(b'unit fixture')
    result={'backend':'kimi_caption','indexed_file_count':1}
    with patch('mcp_servers.sentry_video_search.service._ensure_dashscope_key',side_effect=AssertionError('must not call DashScope')),\
         patch.object(backend,'index_videos',return_value=result) as index:
        value=index_local(video,replace(SearchConfig.for_library('nuplan'),requires_scene_manifest=False))
    assert value['backend']=='kimi_caption'
    assert index.call_args.args[0]==[video.resolve()]


def test_changed_source_is_rejected_before_visual_rerank(tmp_path):
    video=tmp_path/'front.mp4';video.write_bytes(b'unit video fixture')
    config=SearchConfig.for_library('nuplan');config.runtime_home=tmp_path/'runtime'
    client=Mock();client.chat.return_value=(caption(),{'request_id':'test'})
    values=options()
    with patch.object(backend,'settings',return_value=values),patch.object(backend,'KimiClient',return_value=client),\
         patch.object(backend,'_get_video_duration',return_value=10.),\
         patch.object(backend,'sample_frames',return_value=([],[])):
        backend.index_videos([video],config)
        video.write_bytes(b'changed after index')
        with pytest.raises(ValueError,match='changed since Kimi indexing'):
            backend.recall('公交车',config)
    assert client.chat.call_count==1


def test_visual_rerank_keeps_each_candidate_request_small_and_maps_ids(tmp_path):
    config=SearchConfig.for_library('nuplan');config.runtime_home=tmp_path/'runtime'
    frames=[tmp_path/f'{i}.jpg' for i in range(4)]
    values=options();values['rerank_batch_size']=1
    hits=[{'source_file':str(tmp_path/f'{i}.mp4'),'start_time':0.,'end_time':10.,'similarity_score':.5,'caption':caption()} for i in range(2)]
    client=Mock()
    client.chat.side_effect=[({'ranked_results':[{'candidate_index':0,'relevance_score':score,'matched':True,'reason':'visible'}]}, {'image_count':4}) for score in (.7,.9)]
    with patch.object(backend,'settings',return_value=values),patch.object(backend,'sample_frames',return_value=(frames,[0,3,6,9])):
        best,details=backend.rerank('公交车',hits,config,client)
    assert client.chat.call_count==2
    assert all(len(call.args[1])==4 for call in client.chat.call_args_list)
    assert best['source_file']==hits[1]['source_file']
    assert [row['candidate_index'] for row in details['ranked_results']]==[1,0]


def test_configured_thinking_option_is_forwarded_without_changing_model(monkeypatch):
    monkeypatch.setenv('KIMI_API_KEY','unit-test-key')
    values=options();values['thinking_mode']='disabled'
    response=Mock(status_code=200)
    response.json.return_value={'choices':[{'message':{'content':'{"ok":true}'},'finish_reason':'stop'}]}
    with patch.object(backend.requests,'post',return_value=response) as send:
        backend.KimiClient(values).chat('test')
    assert send.call_args.kwargs['json']['thinking']=={'type':'disabled'}
    assert send.call_args.kwargs['json']['model']=='kimi-k3'


def test_unannotated_video_cannot_enter_nuplan_library(tmp_path):
    video=tmp_path/'front.mp4';video.write_bytes(b'not a prepared annotated scene')
    with patch.object(backend,'index_videos',side_effect=AssertionError('must not send incomplete scene to API')):
        with pytest.raises(ValueError,match='No complete annotated'):
            index_local(video,SearchConfig.for_library('nuplan'))


def test_credentials_error_stops_new_batch_dispatch_and_keeps_progress(tmp_path):
    config=SearchConfig.for_library('nuplan');config.runtime_home=tmp_path/'runtime'
    values=options();values['index_workers']=1
    videos=[tmp_path/f'{i}.mp4' for i in range(3)]
    with patch.object(backend,'settings',return_value=values),\
         patch.object(backend,'_index_one_video',side_effect=backend.KimiAPIError('Kimi gateway HTTP 403: no access')) as process:
        with pytest.raises(backend.KimiAPIError,match='successful records were retained'):
            backend.index_videos(videos,config)
    assert process.call_count==1
    progress=json.loads((config.runtime_home/'.sentrysearch/kimi_index_progress.json').read_text())
    assert progress['stopped_for_credentials'] is True
    assert len(progress['errors'])==1


def test_directory_index_ignores_encoder_temporary_files(tmp_path):
    video=tmp_path/'front.mp4';video.write_bytes(b'complete')
    temporary=tmp_path/'front.tmp.mp4';temporary.write_bytes(b'incomplete')
    (tmp_path/'scene_manifest.json').write_text(json.dumps({'video_path':str(video)}))
    config=SearchConfig.for_library('nuplan')
    with patch('mcp_servers.nuplan_data.manifest.index_metadata',return_value={'dataset':'nuplan'}) as metadata,\
         patch.object(backend,'index_videos',return_value={}) as index:
        result=index_local(tmp_path,config)
    assert metadata.call_args.args[0]==video
    assert index.call_args.args[0]==[video]
    assert len(result['ignored_incomplete_videos'])==1


def test_public_retrieval_entry_keeps_kimi_in_local_scene_backend():
    from mcp_servers.sentry_video_search import service
    config=SearchConfig.for_library('nuplan')
    expected=object()
    with patch.object(service,'_ensure_dashscope_key',side_effect=AssertionError('must not call legacy API')),\
         patch.object(service,'search_local',return_value=expected) as local:
        assert service.retrieve_video('urban road',config) is expected
    assert local.call_args.args==('urban road',config)


def test_direct_transport_keeps_gateway_and_reuses_worker_session(monkeypatch):
    monkeypatch.setenv('KIMI_API_KEY','direct-unit-fixture')
    values=options();values['use_environment_proxy']=False
    response=Mock(status_code=200)
    response.json.return_value={'model':'kimi-k3','choices':[{'message':{'content':'{"ok":true}'}}]}
    session=Mock();session.post.return_value=response
    with patch.object(backend,'_direct_http',SimpleNamespace()), \
         patch.object(backend.requests,'Session',return_value=session) as factory, \
         patch.object(backend.requests,'post') as proxy_send:
        backend.KimiClient(values).chat('public test')
        backend.KimiClient(values).chat('public test')
    factory.assert_called_once()
    assert session.trust_env is False
    assert session.post.call_count==2 and proxy_send.call_count==0
    assert session.post.call_args.args[0]=='https://example.invalid/v1/chat/completions'
    assert session.post.call_args.kwargs['json']['model']=='kimi-k3'
