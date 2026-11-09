"""Kimi handles text/visual reasoning; Alibaba credentials stay in retrieval."""
import json
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace
import pytest

@pytest.fixture
def model_environment(monkeypatch):
    for key in ['OPENAI_MODEL','LLM_MODEL','OPENAI_BASE_URL','LLM_BASE_URL']:
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('OPENAI_API_KEY','test-kimi-key')
    monkeypatch.setenv('DASHSCOPE_API_KEY','test-retrieval-key')
    import api_config
    from mcp_servers.sentry_video_search.core.api_settings import bind_profiles
    with patch('dotenv.load_dotenv'), bind_profiles(api_config.get_profile):
        yield

@pytest.mark.parametrize('kind',['search_plan','precheck','first_person'])
def test_auxiliary_video_reasoning_uses_kimi_credentials(kind,model_environment,tmp_path):
    from mcp_servers.sentry_video_search.core.pipeline import search_query,prefilter
    from mcp_servers.sentry_video_search.core import first_person_detector
    received=[]
    def chat(**kwargs):
        received.append(kwargs)
        content={'accepted':True,'confidence':.9,'reason':'test','is_first_person':True,'is_clean_view':True}
        return {'choices':[{'message':{'content':json.dumps(content)}}]}
    frame=tmp_path/'frame.jpg';frame.write_bytes(b'frame')
    if kind=='search_plan':
        with patch.object(search_query,'chat_completion',side_effect=chat):
            search_query.generate_search_plan('urban road')
    elif kind=='precheck':
        with patch.object(prefilter,'chat_completion',side_effect=chat),patch.object(prefilter,'image_data_url',return_value='data:image/jpeg;base64,dGVzdA=='):
            prefilter.visual_precheck(frames=[frame],requirement='road',search_plan={},item={},config=prefilter.PrecheckConfig())
    else:
        with patch.object(first_person_detector,'chat_completion',side_effect=chat),patch.object(first_person_detector,'_data_url',return_value='data:image/jpeg;base64,dGVzdA=='):
            first_person_detector.classify_frames([frame],video_path=Path('/video'),model=None,base_url=None)
    assert received[0]['payload']['model']=='kimi-k3'
    assert received[0]['api_key']=='test-kimi-key'
    assert received[0]['base_url']=='https://apicz.boyuerichdata.com/v1'
    messages=received[0]["payload"]["messages"]
    assert [m["role"] for m in messages]==["system","user"]
    prompts=Path("mcp_servers/sentry_video_search/core/prompts")
    filename={"search_plan":"search_plan.md","precheck":"preview_review.md","first_person":"first_person.md"}[kind]
    assert messages[0]["content"]==(prompts/filename).read_text().strip()
    if kind=="precheck":
        context=json.loads(messages[1]["content"][0]["text"])
        assert context["requirement"]=="road"
        assert "accepted" not in context


def test_reasoning_cannot_borrow_retrieval_key(monkeypatch):
    from mcp_servers.sentry_video_search.core.pipeline.common import model_settings
    monkeypatch.delenv('OPENAI_API_KEY',raising=False);monkeypatch.delenv('LLM_API_KEY',raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY','retrieval-only')
    with patch('dotenv.load_dotenv'),pytest.raises(RuntimeError,match='OPENAI_API_KEY'):
        model_settings()


def test_effective_agent_defaults_to_kimi(model_environment):
    from agent.llm_client import LLMClient
    with patch.dict('sys.modules', {'openai':SimpleNamespace(OpenAI=Mock())}):
        assert LLMClient().model=='kimi-k3'

@pytest.fixture
def root_profile_override(tmp_path,monkeypatch):
    import api_config
    profiles=json.loads(api_config.API_CONFIG_PATH.read_text())
    profiles['kimi'].update(model='root-test-model',base_url='https://root.invalid/v1',api_key_env='ROOT_TEST_KEY')
    profiles['retrieval'].update(api_key_env='ROOT_RETRIEVAL_KEY',embedding_model='root-embedding',embedding_url='https://embedding.invalid/api',embedding_dimensions=3,rerank_model='root-rerank',rerank_url='https://rerank.invalid/api')
    path=tmp_path/'api.json';path.write_text(json.dumps(profiles))
    monkeypatch.setattr(api_config,'API_CONFIG_PATH',path)
    monkeypatch.setenv('ROOT_TEST_KEY','root-kimi-key')
    monkeypatch.setenv('ROOT_RETRIEVAL_KEY','root-retrieval-key')
    return profiles


def test_one_root_profile_controls_all_reasoning_clients(model_environment,root_profile_override):
    import api_config
    from agent.llm_client import LLMClient
    from mcp_servers.sentry_video_search.core.pipeline.common import model_settings
    with patch.dict('sys.modules',{'openai':SimpleNamespace(OpenAI=Mock())}):
        client=LLMClient()
    for config in [dict(model=client.model,base_url=client.base_url,api_key=client.api_key)]:
        assert config['model']=='root-test-model'
        assert config['base_url']=='https://root.invalid/v1'
        assert config['api_key']=='root-kimi-key'
    assert model_settings()==('root-test-model','https://root.invalid/v1','root-kimi-key')
    assert api_config.wfa_settings()=={'endpoint':'https://root.invalid/v1/chat/completions','model':'root-test-model','api_key':'root-kimi-key'}
    assert 'api_key' not in api_config.get_profile('kimi')


def test_embedding_request_reads_root_profile(model_environment,root_profile_override):
    from mcp_servers.sentry_video_search.core.vendor.sentrysearch.qwen_api_embedder import QwenAPIEmbedder
    backend=QwenAPIEmbedder()
    response=SimpleNamespace(status_code=200,json=lambda:{'output':{'embeddings':[{'embedding':[1.,2.,3.]}]}})
    with patch('mcp_servers.sentry_video_search.core.vendor.sentrysearch.qwen_api_embedder.requests.post',return_value=response) as post:
        backend._call([{'text':'urban road'}],enable_fusion=False)
    assert post.call_args.args[0]=='https://embedding.invalid/api'
    assert json.loads(post.call_args.kwargs['data'])['model']=='root-embedding'
    assert post.call_args.kwargs['headers']['Authorization']=='Bearer root-retrieval-key'


def test_rerank_request_reads_root_profile(model_environment,root_profile_override,tmp_path):
    from mcp_servers.sentry_video_search.core import reranker
    def trim(source,start,end,target):Path(target).write_bytes(b'x'*25000)
    response=SimpleNamespace(status_code=200,json=lambda:{'output':{'results':[{'index':0,'relevance_score':.9}]}})
    with patch.object(reranker,'_settings',return_value=('access','secret','bucket','endpoint','prefix','')),patch.object(reranker,'trim_clip',side_effect=trim),patch.object(reranker,'preprocess_chunk',side_effect=lambda p,**kwargs:p),patch.object(reranker,'_upload',return_value='https://staging.invalid/video'),patch.object(reranker,'_delete'),patch.object(reranker.requests,'post',return_value=response) as post:
        reranker.rerank_hits('urban road',[{'source_file':'/video.mp4','start_time':0,'end_time':3,'similarity_score':.8}],run_dir=tmp_path/'rerank')
    assert post.call_args.args[0]=='https://rerank.invalid/api'
    assert post.call_args.kwargs['json']['model']=='root-rerank'
    assert post.call_args.kwargs['headers']['Authorization']=='Bearer root-retrieval-key'



def test_adapter_injects_host_config_without_native_implicit_import(root_profile_override,tmp_path,monkeypatch):
    from mcp_servers.sentry_video_search.adapter import service
    from mcp_servers.sentry_video_search.core import service as native
    from mcp_servers.sentry_video_search.core.api_settings import get_profile
    monkeypatch.setenv("ADAGENT_RUNTIME_ROOT",str(tmp_path/"runtime"))
    def index(*args,**kwargs):
        assert get_profile()["model"]=="root-test-model"
        return {"status":"indexed"}
    with patch("dotenv.load_dotenv"),patch.object(native,"index_local",side_effect=index):
        assert service.index_local("/unused")["status"] == "indexed"
    assert get_profile()["model"] != "root-test-model"
