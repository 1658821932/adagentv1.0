import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from mcp_servers.sentry_video_search import kimi_backend as backend
from mcp_servers.sentry_video_search.hybrid_recall import evidence, hybrid_rank, split_conditions
from mcp_servers.sentry_video_search.config import SearchConfig


def row(identity, text, **fields):
    caption={'summary_zh':text,'summary_en':'','road_layout':'','ego_motion':'',
             'visible_actors':[],'observed_interactions':[],'tags':[],'uncertainties':[]}
    caption.update(fields)
    return {'id':identity,'caption':json.dumps(caption,ensure_ascii=False),
            'metadata':json.dumps({'scene_id':identity}),'source_file':identity+'.mp4'}


def first(query,rows,keywords=()):
    selected,_=hybrid_rank(query,rows,keywords)
    return selected[0]['row']['id'] if selected else None


def test_unknown_actor_does_not_outrank_observed_actor():
    rows=[row('unknown','城市路口',uncertainties=['行人 pedestrian pedestrians 无法确认 是否出现']),
          row('visible','路口行人穿过道路',visible_actors=['pedestrian walking'])]
    assert first('行人 pedestrian',rows)=='visible'
    assert evidence('无法确认是否有行人')['positive']==set()


def test_explicit_absence_is_not_positive_actor_evidence():
    rows=[row('absent','没有公交车，没有巴士，no buses'),row('present','前方有公交车')]
    assert first('公交车',rows)=='present'
    assert 'bus' not in evidence('no buses')['positive']


def test_negative_query_keeps_exclusion_out_of_positive_expansion():
    rows=[row('rain','白天城市路口，正在下雨 rain rainy'),row('dry','白天城市路口，天气晴朗')]
    selected,details=hybrid_rank('白天路口，不要下雨',rows,['rain','rainy','intersection'])
    assert selected[0]['row']['id']=='dry'
    assert details['excluded_concepts']==['rain']
    assert 'rain' not in details['positive_concepts']


def test_local_bilingual_synonyms_do_not_need_an_api():
    assert first('公共汽车',[row('car','urban sedan'),row('bus','coach at junction')])=='bus'


def test_query_expansion_cannot_overwhelm_original_actor_requirement():
    rows=[row('bus','公交车在路口行驶'),row('drift','night rainy fog snow highway construction')]
    assert first('公交车',rows,['night','rainy','fog','snow','highway','construction'])=='bus'


def test_joint_requirements_outrank_partial_caption():
    rows=[row('partial','公交车 bus buses 公交车'),row('both','公交车经过交通信号灯控制的十字路口')]
    assert first('公交车，交通信号灯路口',rows)=='both'


def test_duplicate_chunks_do_not_fill_visual_budget():
    one=row('same','公交车在路口行驶');two=dict(one,id='other_chunk')
    selected,_=hybrid_rank('公交车',[one,two,row('other_scene','公共汽车行驶')],candidate_count=3)
    assert len(selected)==2
    assert len({json.loads(x['row']['metadata'])['scene_id'] for x in selected})==2


def test_unrecognized_words_still_use_original_lexical_recall():
    assert first('隧道',[row('tunnel','隧道内车辆行驶'),row('other','路口车辆行驶')])=='tunnel'


def test_long_query_does_not_silently_drop_conditions():
    conditions=split_conditions('，'.join('condition'+str(i) for i in range(12)))
    assert len(conditions)==8
    assert all('condition'+str(i) in '；'.join(conditions) for i in range(12))


def decision(status='met',missing=()):
    return {'candidate_index':0,'relevance_score':.95,'matched':True,'reason':'visible',
            'missing_conditions':list(missing),
            'condition_checks':[{'condition_index':0,'status':status,'evidence':'Visible scene evidence'}]}


@pytest.mark.parametrize('status',['not_met','unknown'])
def test_high_model_score_cannot_override_unmet_or_unknown_condition(status):
    value=backend.validate_ranking({'ranked_results':[decision(status)]},1,['bus'])
    assert value[0]['relevance_score']==0 and value[0]['matched'] is False


def test_missing_conditions_override_matched_true_even_without_new_checks():
    value=backend.validate_ranking({'ranked_results':[decision(missing=['no visible bus'])]},1)
    assert value[0]['relevance_score']==0 and value[0]['matched'] is False


@pytest.mark.parametrize('checks',[[],[{'condition_index':True,'status':'met','evidence':'x'}],
    [{'condition_index':0,'status':'met','evidence':''}],[{'condition_index':0,'status':'maybe','evidence':'x'}]])
def test_incomplete_or_invalid_condition_evidence_is_rejected(checks):
    item=decision();item['condition_checks']=checks
    with pytest.raises(ValueError,match='condition'):
        backend.validate_ranking({'ranked_results':[item]},1,['bus'])


def options():
    return {'base_url':'https://example.invalid/v1','model':'kimi-k3','api_key_env':'KIMI_API_KEY',
            'fallback_api_key_env':'LLM_API_KEY','timeout_seconds':30,'max_tokens':1000,
            'caption_frames':4,'rerank_frames':4,'frame_width':512,'candidate_count':4,
            'query_expansion':False,'minimum_relevance_score':.4,'rerank_batch_size':1,
            'require_condition_checks':True,'rerank_workers':1,'rerank_cache_ttl_seconds':3600}


def hits(tmp_path,count=2):
    return [{'source_file':str(tmp_path/f'{i}.mp4'),'source_video_sha256':'video-'+str(i),
             'scene_manifest_sha256':'manifest-'+str(i),'start_time':0.,'end_time':10.,
             'similarity_score':.7,'caption':{'summary_zh':str(i)}} for i in range(count)]


def test_repeated_visual_query_is_served_without_api_or_frame_extraction(tmp_path):
    cfg=SimpleNamespace(runtime_home=tmp_path)
    client=Mock();client.chat.return_value=({'ranked_results':[decision()]},{'image_count':4})
    with patch.object(backend,'settings',return_value=options()),patch.object(backend,'sample_frames',return_value=([],[])):
        first,details=backend.rerank('bus',hits(tmp_path),cfg,client)
    with patch.object(backend,'settings',return_value=options()),patch.object(backend,'KimiClient',side_effect=AssertionError('API forbidden on cache hit')),patch.object(backend,'sample_frames',side_effect=AssertionError('frames forbidden on cache hit')):
        second,cached=backend.rerank('bus',hits(tmp_path),cfg)
    assert client.chat.call_count==2
    assert first==second and details['cache_hits']==0 and cached['cache_hits']==2


@pytest.mark.parametrize('change',['model','manifest','video','frames','query','policy'])
def test_visual_cache_invalidates_when_identity_or_policy_changes(tmp_path,change):
    values=options();items=hits(tmp_path,1);query='bus'
    cfg=SimpleNamespace(runtime_home=tmp_path)
    client=Mock();client.chat.side_effect=lambda *args:({'ranked_results':[decision()]},{'image_count':4})
    with patch.object(backend,'settings',return_value=values),patch.object(backend,'sample_frames',return_value=([],[])):
        backend.rerank(query,items,cfg,client)
        if change=='model':values['model']='changed'
        if change=='manifest':items[0]['scene_manifest_sha256']='changed'
        if change=='video':items[0]['source_video_sha256']='changed'
        if change=='frames':values['rerank_frames']=8
        if change=='query':query='公交车'
        if change=='policy':values['require_condition_checks']=False
        backend.rerank(query,items,cfg,client)
    assert client.chat.call_count==2


def test_expired_cache_is_not_used(tmp_path,monkeypatch):
    store=backend.CaptionStore(tmp_path)
    monkeypatch.setattr(backend.time,'time',lambda:100.)
    store.cache_response('n','k',{'ok':True},{})
    assert store.cached_response('n','k',50)[0]=={'ok':True}
    monkeypatch.setattr(backend.time,'time',lambda:151.)
    assert store.cached_response('n','k',50) is None


def test_visual_concurrency_is_bounded_and_candidate_mapping_is_stable(tmp_path):
    cfg=SimpleNamespace(runtime_home=tmp_path);values=options()
    values.update(rerank_workers=2,rerank_cache_ttl_seconds=0)
    lock=threading.Lock();barrier=threading.Barrier(2);active=0;maximum=0
    def chat(*args):
        nonlocal active,maximum
        with lock:active+=1;maximum=max(maximum,active)
        barrier.wait(timeout=3)
        with lock:active-=1
        return {'ranked_results':[decision()]},{'image_count':4}
    with patch.object(backend,'settings',return_value=values),patch.object(backend,'sample_frames',return_value=([],[])):
        best,details=backend.rerank('bus',hits(tmp_path,4),cfg,SimpleNamespace(chat=chat))
    assert maximum==2
    assert [r['candidate_index'] for r in details['ranked_results']]==[0,1,2,3]
    assert best['source_file']==str(tmp_path/'0.mp4')


def test_auth_failure_does_not_dispatch_remaining_visual_candidates(tmp_path):
    cfg=SimpleNamespace(runtime_home=tmp_path);values=options()
    values.update(rerank_workers=2,rerank_cache_ttl_seconds=0)
    client=Mock();client.chat.side_effect=backend.KimiAPIError('Kimi gateway HTTP 403: no access')
    with patch.object(backend,'settings',return_value=values),patch.object(backend,'sample_frames',return_value=([],[])):
        with pytest.raises(backend.KimiAPIError,match='403'):
            backend.rerank('bus',hits(tmp_path,4),cfg,client)
    assert client.chat.call_count<=2


def test_nuplan_export_does_not_write_rejected_candidate_videos(tmp_path):
    from mcp_servers.sentry_video_search import service
    details={'candidates':[{'candidate_index':0},{'candidate_index':1}],
             'ranked_results':[dict(decision(),candidate_index=0),dict(decision('unknown'),candidate_index=1,matched=False,relevance_score=0.)]}
    with patch.object(service,'_trim_hit',side_effect=[tmp_path/'compressed.mp4',tmp_path/'lossless.mp4']) as trim:
        winner=service._export_reranked_hits(details,'bus',SearchConfig.for_library('nuplan'),tmp_path/'run')
    assert trim.call_count==2 and winner==tmp_path/'lossless.mp4'
    assert 'video_path' not in details['ranked_results'][1]


def test_negation_scope_resets_at_a_new_positive_observation():
    observed=evidence('without pedestrians and with a bus')
    assert observed['negative']=={'pedestrian'}
    assert observed['positive']=={'bus'}
    assert evidence('not only buses')['positive']=={'bus'}


def test_possible_actor_is_unknown_not_positive_evidence():
    assert evidence('There might be a bus')['positive']==set()
    assert evidence('可能有行人')['positive']==set()


def indexed_config(tmp_path,values):
    import hashlib
    video=tmp_path/'front.mp4';video.write_bytes(b'test-video')
    cfg=SimpleNamespace(runtime_home=tmp_path/'runtime')
    store=backend.CaptionStore(cfg.runtime_home)
    item=row('one','公交车 bus')
    item.update(source_file=str(video),start_time=0.,end_time=10.,metadata='{}',
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),signature=backend.signature(values),
        document='公交车 bus',sampling='{}',updated_at='test')
    store.upsert(item)
    return cfg


def test_query_expansion_cache_reuses_response_without_api(tmp_path):
    values=options();values.update(query_expansion=True,query_cache_ttl_seconds=3600)
    cfg=indexed_config(tmp_path,values)
    client=Mock();client.chat.return_value=({'keywords':['公交车','bus']},{'image_count':0})
    with patch.object(backend,'settings',return_value=values):
        first,_=backend.recall('公交车',cfg,client)
        with patch.object(backend,'KimiClient',side_effect=AssertionError('API forbidden on cache hit')):
            second,details=backend.recall('公交车',cfg)
    assert first==second and client.chat.call_count==1 and details['trace']['cache_hit'] is True


def test_optional_expansion_network_failure_preserves_local_recall(tmp_path):
    values=options();values['query_expansion']=True
    cfg=indexed_config(tmp_path,values)
    client=Mock();client.chat.side_effect=backend.KimiAPIError('Kimi gateway Timeout after one retry')
    with patch.object(backend,'settings',return_value=values):
        found,details=backend.recall('公交车',cfg,client)
    assert found and details['expansion_fallback']=='KimiAPIError'


def test_expansion_auth_failure_is_not_hidden_by_local_fallback(tmp_path):
    values=options();values['query_expansion']=True
    cfg=indexed_config(tmp_path,values)
    client=Mock();client.chat.side_effect=backend.KimiAPIError('Kimi gateway HTTP 401: denied')
    with patch.object(backend,'settings',return_value=values):
        with pytest.raises(backend.KimiAPIError,match='401'):
            backend.recall('公交车',cfg,client)
