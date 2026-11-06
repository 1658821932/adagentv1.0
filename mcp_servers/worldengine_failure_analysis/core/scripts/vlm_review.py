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
    def __init__(self,secret,out):
        self.secret,self.out=secret,out

    def ask(self,messages,name):
        payload={'model':self.secret['model'],'thinking':{'type':'disabled'},'max_tokens':1800,'messages':messages}
        encoded=json.dumps(payload,ensure_ascii=False).encode()
        save(self.out/(name+'_request.json'),{'sha256':hashlib.sha256(encoded).hexdigest(),'model':self.secret['model']})
        attempts=[]
        for attempt in range(2):
            started=time.monotonic()
            try:
                request=urllib.request.Request(self.secret['endpoint'],data=encoded,
                    headers={'Authorization':'Bearer '+self.secret['api_key'],'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=30) as response:raw=json.load(response)
                save(self.out/(name+'_raw.json'),raw)
                choice=raw['choices'][0]
                if choice.get('finish_reason')=='length':raise ValueError('Truncated model response')
                text=choice['message']['content'].strip()
                if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
                answer=json.loads(text)
                if not isinstance(answer,dict):raise ValueError('Expected JSON object')
                attempts.append({'state':'completed','seconds':round(time.monotonic()-started,2),'usage':raw.get('usage')})
                save(self.out/(name+'.json'),answer)
                print('VLM',name,attempts[-1]['seconds'],'seconds',flush=True)
                return answer
            except (TimeoutError,socket.timeout,urllib.error.URLError) as exc:
                attempts.append({'state':'failed','seconds':round(time.monotonic()-started,2),'error_type':type(exc).__name__})
                retryable=not isinstance(exc,urllib.error.HTTPError) or exc.code in (429,500,502,503,504)
                if attempt or not retryable:raise
            except Exception as exc:
                attempts.append({'state':'failed','seconds':round(time.monotonic()-started,2),'error_type':type(exc).__name__})
                raise
            finally:
                save(self.out/(name+'_status.json'),attempts)


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


def run(client,evidence,official,out,max_turns=12,workers=2):
    initial=[evidence.images[f'I_overview_{i:02d}'] for i in range(12)]
    overview=inspect_images(client,evidence,initial,'整体12个时刻，逐组观察', 'overview',workers)
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
       'phase':'visual_review'}
    save(out/'context.json',{'system_prompt':SYSTEM,**contract})
    history=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(contract,ensure_ascii=False)}]
    candidates={};numeric_ids=set();image_ids={im['id'] for im in initial};trace=[]
    for turn in range(max_turns):
        action=client.ask(history,f'turn_{turn:02d}')
        print('ACTION',turn,action.get('action'),flush=True)
        history.append({'role':'assistant','content':json.dumps(action,ensure_ascii=False)})
        try:
            kind=action.get('action')
            if kind=='images':
                images=evidence.select_images(action.get('requests'))
                response={'image_reviews':inspect_images(client,evidence,images,action.get('purpose'),f'detail_{turn:02d}',workers)}
                image_ids.update(im['id'] for im in images)
            elif kind=='visual_complete':
                items=action.get('candidates')
                if not action.get('observations') or not isinstance(items,list) or not items or any(not c.get('id') or not c.get('reason') for c in items):
                    raise ValueError('Provide observations and candidates with unique ids and reasons')
                if len({c['id'] for c in items})!=len(items):raise ValueError('Duplicate candidate id')
                candidates={c['id']:c for c in items};save(out/'visual_hypotheses.json',action)
                response={'state':'numeric_verification','instruction':'现在主动请求数值，逐项核验候选。必要时仍可补图。'}
            elif kind=='numbers':
                if not candidates:raise ValueError('Finish visual observations and candidates first')
                queries=action.get('queries')
                if not isinstance(queries,list) or not queries:raise ValueError('Provide nonempty queries')
                returned=[]
                for query in queries:
                    try:
                        value=evidence.query(query);identifier=f'E{len(numeric_ids)+1}'
                        value={'id':identifier,**value};numeric_ids.add(identifier)
                        save(out/(identifier+'.json'),value);returned.append(value)
                    except (ValueError,KeyError,TypeError) as exc:returned.append({'request':query,'tool_error':str(exc)})
                response={'evidence':returned}
            elif kind=='final':
                validate_final(action,candidates,numeric_ids,image_ids)
                result={**action,'token':evidence.row['token'],'status':'supported' if action['selected_candidate_id'] is not None else 'undetermined',
                        'analysis_level':'driving_behavior','source':'external_vlm','visual_review_reused':False}
                save(out/'result.json',result)
                trace.append({'action':action,'response':{'state':'completed'}});save(out/'trace.json',trace)
                return result
            else:raise ValueError('Unknown action')
        except (ValueError,KeyError,TypeError,AttributeError,IndexError) as exc:
            response={'tool_error':str(exc),'instruction':'修正请求或补充证据，不要跳过核验。'}
        trace.append({'action':action,'response':response});save(out/'trace.json',trace)
        history.append({'role':'user','content':json.dumps(response,ensure_ascii=False)})
    raise RuntimeError('Turn budget exhausted before an evidence-backed final answer; analysis incomplete')
