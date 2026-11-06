"""Summarize existing case results into scene families useful for scenario editing."""
import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evidence import save
from vlm_review import Client
from prompt_utils import load_prompt

PROJECT=Path(__file__).resolve().parents[1]
RULES = load_prompt("scene_groups.md")


def validate_description(g):
    fields=('name','ego_action','environment','wrong_behavior','expected_behavior')
    if any(not isinstance(g.get(k),str) or not g[k].strip() for k in fields):raise ValueError('Provide five nonempty scene description fields')
    if any(word in g['name'] for word in ('越界','追尾','不足','未减速','未停车','导致','错误')):raise ValueError('Title must contain only ego action and environment, not error/outcome')
    paragraph=''.join(g[k] for k in fields[1:])
    if any(word in paragraph for word in ('导致','控制不足','控制失效','感知失败','未识别')):raise ValueError('Describe observed motion and co-occurring outcomes; omit unsupported module/cause claims')
    if not g['expected_behavior'].startswith('自车应'):raise ValueError('Expected response starts with 自车应')


def ask(client,messages,name,validate=None):
    for attempt in range(3):
        call=name if attempt==0 else f'{name}_correction_{attempt}'
        path=client.out/(call+'.json')
        answer=json.loads(path.read_text()) if path.exists() else client.ask(messages,call)
        try:
            if validate is not None:validate(answer)
            return answer
        except (ValueError,KeyError,TypeError) as exc:
            if attempt==2:raise
            messages=messages+[{'role':'assistant','content':json.dumps(answer,ensure_ascii=False)},
                {'role':'user','content':'修正本次总结的格式或措辞，不改变成员、不补造证据：'+str(exc)}]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-config',type=Path,help='Standalone API profile JSON')
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-name',required=True)
    parser.add_argument('--groups-from',type=Path,help='Keep existing memberships and rewrite descriptions only')
    args=parser.parse_args()
    for name in (args.run_name,args.output_name):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):parser.error('Invalid name')
    root=args.run_dir.resolve();out=root/args.output_name;out.mkdir(exist_ok=True)
    cases=[]
    for i,token in enumerate(json.loads((root/'selected_tokens.json').read_text()),1):
        p=root/'scenes'/token;r=json.loads((p/'result.json').read_text())
        if json.loads((p/'status.json').read_text())['state']!='completed':raise ValueError('Finish all cases first')
        visual=json.loads((p/'visual_hypotheses.json').read_text())
        cases.append({'id':f'C{i:03d}','token':token,'reason':r['failure_reason'],
            'visual_observations':visual['observations'],
            'observations':r['observations'],'uncertainty':r.get('uncertainty',''),'reason_status':r['status']})
    snapshot={'cases':cases,'rules':RULES,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    existing=None
    if args.groups_from:
        source=args.groups_from if args.groups_from.is_absolute() else PROJECT/args.groups_from
        existing=json.loads(source.read_text())
        if Counter(c['token'] for g in existing['groups'] for c in g['cases'])!=Counter(c['token'] for c in cases):
            raise ValueError('Existing memberships must cover the same cases exactly once')
        snapshot['group_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot_path=out/'input.json'
    if snapshot_path.exists() and json.loads(snapshot_path.read_text())!=snapshot:raise ValueError('Inputs changed; use a new output-name')
    save(snapshot_path,snapshot)
    import sys
    sys.path.insert(0, str(PROJECT))
    from api_settings import vlm_settings
    client=Client(vlm_settings(args.api_config),out)
    if existing is not None:
        by_token={c['token']:c for c in cases}
        def rewrite(g):
            if g['id']=='U0':return {**g,'name':'关键场景信息不足','ego_action':'部分案例缺少明确的自车动作描述。','environment':'关键参与者或交互条件尚不明确。','wrong_behavior':'无法据此形成一致的错误行为描述。','expected_behavior':'暂不确定，不直接生成编辑任务。'}
            task={'task':'保持本组成员不变，按共同情境重新命名并填写描述字段。',
                  'previous_name':g['name'],'cases':[by_token[c['token']] for c in g['cases']],
                  'output':{'name':'动作+环境+场景，不写错误或结果','ego_action':'自车正在做什么。','environment':'环境如何。','wrong_behavior':'自车错误行为。','expected_behavior':'自车应如何响应。'}}
            description=ask(client,[{'role':'system','content':RULES},{'role':'user','content':json.dumps(task,ensure_ascii=False)}],'describe_'+g['id'],validate_description)
            return {**g,**{k:description[k] for k in ('name','ego_action','environment','wrong_behavior','expected_behavior')}}
        with ThreadPoolExecutor(max_workers=2) as pool:groups=list(pool.map(rewrite,existing['groups']))
        assignments=[{'case':by_token[c['token']]['id'],'group':g['id'],'basis':c['basis']} for g in groups for c in g['cases']]
        write_summary(out,cases,groups,assignments)
        return
    brief=[{**c,'visual_observations':c['visual_observations'][:3],'observations':c['observations'][:2]} for c in cases]
    task={'task':'阅读全部案例，提炼5至7个有案例支持的主要场景，另设U0。此阶段只输出类型定义，不输出逐例分组。',
          'cases':brief,'output':{'groups':[{'id':'G1','name':'自车动作+关键环境+场景',
          'ego_action':'自车正在进行的动作。','environment':'道路及其他参与者的位置和相对运动。',
          'wrong_behavior':'自车有证据支持的错误行为。','expected_behavior':'自车应采取的适当响应，作为测试目标。'}],
          'note':'U0也包含在groups中，明确缺少什么信息，不编造动作或响应。'}}
    ontology=ask(client,[{'role':'system','content':RULES},{'role':'user','content':json.dumps(task,ensure_ascii=False)}],'taxonomy',
                 lambda value:[validate_description(g) for g in value['groups'] if g['id']!='U0'])
    groups=ontology['groups'];group_ids={g['id'] for g in groups}
    if len(groups)!=len(group_ids) or 'U0' not in group_ids or not 6<=len(groups)<=8:raise ValueError('Expected 5..7 scene families and U0')
    if any(not isinstance(g.get(k),str) or not g[k].strip() for g in groups for k in ('name','ego_action','environment','wrong_behavior','expected_behavior')):
        raise ValueError('Each scene needs action, environment, wrong behavior and expected response')
    def classify(job):
        number,batch=job
        prompt={'task':'将每例归入一个符合定义的主场景组；缺少关键情境则U0。basis用不到40字说明明确依据。',
                'groups':groups,'cases':batch,'output':{'assignments':[{'case':'C001','group':'G1','basis':'归组所依据的明确场景条件，简短'}]}}
        answer=ask(client,[{'role':'system','content':RULES},{'role':'user','content':json.dumps(prompt,ensure_ascii=False)}],f'assign_{number:02d}')
        assignments=answer['assignments']
        if Counter(a['case'] for a in assignments)!=Counter(c['id'] for c in batch):raise ValueError('Each case must occur exactly once')
        if any(a['group'] not in group_ids or not a.get('basis') for a in assignments):raise ValueError('Invalid assignment')
        return assignments
    jobs=list(enumerate([cases[i:i+16] for i in range(0,len(cases),16)]))
    with ThreadPoolExecutor(max_workers=2) as pool:assignments=[a for batch in pool.map(classify,jobs) for a in batch]
    write_summary(out,cases,groups,assignments)


def write_summary(out,cases,groups,assignments):
    index={c['id']:c for c in cases};counts=Counter(a['group'] for a in assignments)
    result={'source':'external_kimi_summary_of_saved_results','scene_count':len(cases),'groups':[],
            'note':'场景情境归并，不是已验证的因果统计；不改写逐例原因。一个案例只计入一个主组。'}
    lines=['# 典型失败场景总结','',f'共{len(cases)}例，每例归入一个主场景组。以下由Kimi归并已有结果；应有响应是后续测试目标。','']
    for g in sorted(groups,key=lambda g:(g['id']=='U0',-counts[g['id']])):
        members=[{**a,'token':index[a['case']]['token'],'reason_status':index[a['case']]['reason_status']} for a in assignments if a['group']==g['id']]
        result['groups'].append({**g,'count':len(members),'cases':members})
        if not members:continue
        paragraph=''.join(g[k].strip().rstrip('。')+'。' for k in ('ego_action','environment','wrong_behavior','expected_behavior'))
        lines += [f"## {g['name']}（{len(members)}例）",'',paragraph,'']
    lines+=['逐例归组依据与案例编号另存groups.json；原始归因不变。应有响应不等于直接设定自车轨迹。']
    save(out/'groups.json',result);(out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'cases':len(cases),'counts':dict(counts),'summary':str(out/'summary.md')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
