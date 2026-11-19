"""VLM-driven observation, candidate generation, and on-demand numeric verification."""
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from evidence import POSITIONS, save
from prompt_utils import load_prompt

SYSTEM = load_prompt("forensic_analysis.md")
IMAGE_SYSTEM = load_prompt("image_review.md")


class Client:
    def __init__(self,secret,out, audit_factory=None):
        self.secret,self.out=secret,out
        self.audit_factory = audit_factory

    def ask(self,messages,name):
        payload={'model':self.secret['model'],'max_tokens':self.secret['max_tokens'],'messages':messages}
        # Kimi K3 cannot disable thinking, so a hardcoded "disabled" was inert there
        # and the call silently used the provider default. Send the level instead.
        if self.secret.get('reasoning_effort'):
            payload['reasoning_effort']=self.secret['reasoning_effort']
        dialect=self.secret.get('request_dialect')
        if dialect=='qwen':
            payload['enable_thinking']=self.secret.get('thinking_mode')=='enabled'
        elif dialect=='thinking':
            payload['thinking']={'type':self.secret.get('thinking_mode') or 'enabled'}
        elif not dialect:
            payload['thinking']={'type':'disabled'}
        encoded=json.dumps(payload,ensure_ascii=False).encode()
        for attempt in range(3):
            started=time.monotonic()
            audit = self.audit_factory(payload, endpoint=self.secret['endpoint'], label=f'{name}/attempt_{attempt+1}',
                                       secrets=[self.secret['api_key']]) if self.audit_factory else None
            raw=None
            try:
                request=urllib.request.Request(self.secret['endpoint'],data=encoded,
                    headers={'Authorization':'Bearer '+self.secret['api_key'],'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=self.secret['timeout']) as response:
                    if audit is not None:
                        audit.first_response()
                    raw=json.load(response)
                if audit is not None:
                    audit.finish(response=raw, usage=raw.get('usage'))
                choice=raw['choices'][0]
                if choice.get('finish_reason')=='length':raise ValueError('Truncated model response')
                text=choice['message']['content'].strip()
                if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
                answer=json.loads(text)
                if not isinstance(answer,dict):raise ValueError('Expected JSON object')
                print('VLM',name,round(time.monotonic()-started,2),'seconds',flush=True)
                return answer
            except (TimeoutError,socket.timeout,urllib.error.URLError) as exc:
                if raw is None and audit is not None:
                    audit.finish(error=exc,http_status=getattr(exc,'code',None))
                retryable=not isinstance(exc,urllib.error.HTTPError) or exc.code in (429,500,502,503,504)
                if attempt==2 or not retryable:raise
                time.sleep(2*(attempt+1))
            except BaseException as exc:
                if raw is None and audit is not None:
                    audit.finish(error=exc)
                raise


def inspect_images(client,evidence,images,purpose,name,workers):
    def group(job):
        i,selected=job
        prompt={'purpose':purpose}
        content=[{'type':'text','text':json.dumps(prompt,ensure_ascii=False)}]+evidence.content(selected)
        review=client.ask([{'role':'system','content':IMAGE_SYSTEM},{'role':'user','content':content}],f'{name}_{i:02d}')
        return {'image_ids':[im['id'] for im in selected],'review':review}
    jobs=list(enumerate([images[i:i+3] for i in range(0,len(images),3)]))
    with ThreadPoolExecutor(max_workers=workers) as pool:return list(pool.map(group,jobs))


def validate_final(action,candidates,numeric_ids,image_ids=frozenset()):
    assessed=action.get('candidate_assessment')
    if not isinstance(assessed,list) or any(not isinstance(a,dict) for a in assessed) or {a.get('id') for a in assessed}!=set(candidates) or len(assessed)!=len(candidates):
        raise ValueError('Assess every candidate id exactly once')
    if not numeric_ids:raise ValueError('Query numeric evidence before finishing')
    for a in assessed:
        if a.get('verdict') not in ('supported','rejected','uncertain'):raise ValueError('Invalid verdict')
        if not isinstance(a.get('evidence_ids'),list) or any(i not in numeric_ids|image_ids|{'E0'} for i in a['evidence_ids']):
            raise ValueError('Cite only returned numeric ids or actually viewed image ids')
        if a['verdict'] in ('supported','rejected') and not set(a['evidence_ids'])&numeric_ids:
            raise ValueError('A supported/rejected candidate needs requested numeric evidence')
        explanation=a.get('explanation')
        if not isinstance(explanation,dict) or any(not explanation.get(k) for k in ('required_observation','actual_observation','missing_evidence')):
            raise ValueError('For each candidate explain required_observation, actual_observation and missing_evidence; use 无 only when justified')
    chosen=action.get('selected_candidate_id')
    if chosen is not None and not any(a['id']==chosen and a['verdict']=='supported' for a in assessed):
        raise ValueError('Choose a supported candidate, or null when undetermined')
    if not action.get('observations') or not action.get('failure_reason') or not action.get('reason_category'):
        raise ValueError('Final needs observations, failure_reason and reason_category')
    if chosen is None and action['reason_category']!='待确认':raise ValueError('No supported selection must be 待确认')


def run(client,evidence,official,out,max_turns=6,workers=4):
    raw_step=(official or {}).get('metrics',{}).get('first_violation_step')
    try:
        viol=int(float(raw_step)) if raw_step is not None else None
    except (ValueError,TypeError,OverflowError):
        viol=None
    picked=([max(0,min(11,viol+d)) for d in (-1,0,1)] if viol is not None else [0,5,11])
    steps_used=sorted(set(picked))
    initial=[evidence.images[f'I_overview_{i:02d}'] for i in steps_used]
    overview=inspect_images(client,evidence,initial,'关键帧逐组观察：%s' % steps_used,'overview',workers)
    candidates={};numeric_ids=set();image_ids={im['id'] for im in initial}
    numeric_cache={};trace=[];supplemental_remaining=6
    def numbers(queries):
        returned=[]
        for query in queries:
            signature=json.dumps(query,sort_keys=True,ensure_ascii=False)
            if signature in numeric_cache:
                returned.append(numeric_cache[signature]);continue
            try:
                value=evidence.query(query);identifier=f'E{len(numeric_ids)+1}'
                value={**value,'id':identifier};numeric_ids.add(identifier)
                numeric_cache[signature]=value;returned.append(value)
            except (ValueError,KeyError,TypeError) as exc:
                returned.append({'request':query,'tool_error':str(exc)})
        return returned
    prefetched=numbers([{'kind':'ego','steps':steps_used},
        {'kind':'actors','steps':steps_used,'nearest':3}])
    def checkpoint(status):
        save(out/'trace.json',trace)
        save(out/'evidence-budget.json',{'status':status,'initial_steps':steps_used,'prefetched_evidence':prefetched,
            'viewed_image_ids':sorted(image_ids),'supplemental_images_remaining':supplemental_remaining,
            'numeric_evidence':list(numeric_cache.values()),'turns':trace})
    checkpoint('running')
    contract={'official_evidence':{'id':'E0','metrics':official['metrics'],'source':official['source'],'sha256':official['source_sha256']},
       'layout':POSITIONS,'layout_note':'3x3布局，中央空白；八相机无旧BEV','overview_observations':overview,
       'numeric_tools':{'ego':'自车速度、纵向加速度、位置、朝向','actors':'对象ID、相对位置、几何间距、世界速度及速度变化；可选track_token、nearest',
                        'boundary':'车角边界余量、区域外车角','plans':'所选轨迹；steps表示输入帧，含executed字段'},
       'actions':{
          'images':{'action':'images','purpose':'想核实什么','requests':[{'camera':'CAM_F0 或其他实际相机名或 bev','steps':[6,7,8]}]},
          'visual_complete':{'action':'visual_complete','observations':['注明可见现象和step'],'candidates':[{'id':'A','reason':'一个候选行为原因','needs_evidence':'尚需核实什么'}]},
          'numbers':{'action':'numbers','purpose':'比较哪个候选','queries':[{'kind':'ego','steps':[3,4,5]},{'kind':'actors','steps':[3,4,5],'nearest':3}]},
          'final':{'action':'final','observations':['核验后现象'],'candidate_assessment':[{'id':'A','verdict':'supported/rejected/uncertain','evidence_ids':['E1'],'explanation':{'required_observation':'回答该候选needs_evidence，需要何种观测才能支持或直接否定','actual_observation':'实际证据中的相关观测；无直接反证不能rejected','missing_evidence':'尚缺什么，缺少关键证据则uncertain；确无缺失才写无'}}],
                   'selected_candidate_id':'支持最强的候选id；不能确定则null','failure_reason':'一句驾驶行为原因','reason_category':'简短可汇总的行为类别；不能确定则待确认','uncertainty':'仍不确定之处'}},
       'prefetched_numeric_evidence':prefetched,
       'image_budget':{'supplemental_images':6,'max_new_images_per_request':3},
       'instruction':'先用已有关键帧及数值。只为具体疑点补图；已看图片不重复请求。形成候选后，已有数值足够即可final。',
       'phase':'visual_review'}
    history=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(contract,ensure_ascii=False)}]
    for turn in range(max_turns):
        action=client.ask(history,f'turn_{turn:02d}')
        print('ACTION',turn,action.get('action'),flush=True)
        history.append({'role':'assistant','content':json.dumps(action,ensure_ascii=False)})
        try:
            kind=action.get('action')
            if kind=='images':
                requested=evidence.select_images(action.get('requests'))
                unseen={im['id']:im for im in requested if im['id'] not in image_ids}
                images=list(unseen.values())[:min(3,supplemental_remaining)]
                response={'image_reviews':inspect_images(client,evidence,images,action.get('purpose'),f'detail_{turn:02d}',workers) if images else [],
                    'already_viewed':[im['id'] for im in requested if im['id'] in image_ids],
                    'not_viewed':list(unseen)[len(images):]}
                image_ids.update(im['id'] for im in images)
                supplemental_remaining-=len(images)
                response['supplemental_images_remaining']=supplemental_remaining
            elif kind=='visual_complete':
                items=action.get('candidates')
                if not action.get('observations') or not isinstance(items,list) or not items or any(not c.get('id') or not c.get('reason') for c in items):
                    raise ValueError('Provide observations and candidates with unique ids and reasons')
                if len({c['id'] for c in items})!=len(items):raise ValueError('Duplicate candidate id')
                candidates={c['id']:c for c in items}
                response={'state':'numeric_verification','instruction':'用已有数值逐项核验候选。仅在证据不足时补充查询或图片；已有证据足够可以final。'}
            elif kind=='numbers':
                queries=action.get('queries')
                if not isinstance(queries,list) or not queries:raise ValueError('Provide nonempty queries')
                response={'evidence':numbers(queries)}
            elif kind=='final':
                if not candidates:raise ValueError('Provide visual observations and candidates before final')
                validate_final(action,candidates,numeric_ids,image_ids)
                result={**action,'token':evidence.row['token'],'status':'supported' if action['selected_candidate_id'] is not None else 'undetermined',
                        'analysis_level':'driving_behavior','source':'external_vlm','visual_review_reused':False}
                trace.append({'turn':turn,'action':action,'response':{'status':'completed'}})
                checkpoint('completed')
                return result
            else:raise ValueError('Unknown action')
        except (ValueError,KeyError,TypeError,AttributeError,IndexError) as exc:
            response={'tool_error':str(exc),'instruction':'修正请求或补充证据，不要跳过核验。'}
        trace.append({'turn':turn,'action':action,'response':response})
        checkpoint('running')
        history.append({'role':'user','content':json.dumps(response,ensure_ascii=False)})
    checkpoint('incomplete')
    raise RuntimeError('Turn budget exhausted before an evidence-backed final answer; analysis incomplete')
