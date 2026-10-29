"""Local, bilingual caption recall. Caption evidence is not annotation truth."""
from __future__ import annotations

from collections import Counter
import json
import math
import re

RECALL_VERSION = 'field-evidence-rrf-v2'
ALIASES = {
    'collision': ('碰撞', '撞车', '交通事故', 'collision', 'crash', 'accident'),
    'bus': ('公交车', '公共汽车', '巴士', 'bus', 'buses', 'coach'),
    'truck': ('卡车', '货车', 'truck', 'trucks', 'lorry', 'lorries'),
    'pedestrian': ('行人', '行走的人', 'pedestrian', 'pedestrians', 'people walking'),
    'cyclist': ('骑行者', '自行车', 'cyclist', 'cyclists', 'bicycle', 'bicycles', 'bike'),
    'motorcycle': ('摩托车', 'motorcycle', 'motorbike'),
    'intersection': ('路口', '十字路口', '交叉口', 'intersection', 'junction', 'crossroads'),
    'crosswalk': ('人行横道', '斑马线', 'crosswalk', 'zebra crossing', 'pedestrian crossing'),
    'traffic_light': ('交通信号灯', '红绿灯', '信号灯', 'traffic light', 'traffic lights', 'signalized'),
    'red_light': ('红灯', 'red light', 'red signal'),
    'green_light': ('绿灯', 'green light', 'green signal'),
    'rain': ('下雨', '雨天', '雨中', 'rain', 'rainy', 'raining'),
    'snow': ('下雪', '雪天', '积雪', 'snow', 'snowy', 'snowing'),
    'fog': ('雾天', '大雾', 'fog', 'foggy'),
    'night': ('夜间', '夜晚', '夜景', 'night', 'nighttime', 'after dark'),
    'day': ('白天', '日间', 'daytime', 'daylight'),
    'sunny': ('晴天', '晴朗', '阳光', 'sunny', 'clear skies', 'clear weather'),
    'wet_road': ('湿滑', '湿路面', '路面湿', 'wet road', 'wet pavement'),
    'highway': ('高速公路', '高速路', 'highway', 'freeway', 'motorway'),
    'roundabout': ('环岛', '环形交叉口', 'roundabout', 'traffic circle'),
    'left_turn': ('左转', '向左转', 'turn left', 'turning left', 'left turn'),
    'right_turn': ('右转', '向右转', 'turn right', 'turning right', 'right turn'),
    'lane_change': ('变道', '换道', '切入', 'lane change', 'changing lanes', 'cut-in', 'cut in'),
    'overtake': ('超车', '超越', 'overtake', 'overtaking', 'passing vehicle'),
    'following': ('跟车', '跟随前车', 'following', 'car following'),
    'stopped': ('停车等待', '等待红灯', '停等', 'stopped', 'waiting at'),
    'parked': ('路边停车', '停放', 'parked', 'parking'),
    'construction': ('施工', '道路施工', 'construction', 'roadworks', 'work zone'),
    'congestion': ('拥堵', '堵车', '密集车流', 'heavy traffic', 'congestion', 'traffic jam'),
    'sparse_traffic': ('车流较少', '车流稀少', '车辆较少', '车少', '交通稀疏', 'light traffic', 'sparse traffic'),
    'trees': ('树木', '树荫', '绿树', '林荫', 'tree', 'trees', 'tree-lined', 'foliage'),
    'buildings': ('建筑', '楼宇', '高楼', 'building', 'buildings', 'high-rise'),
}
FIELDS = {'summary_zh': 1.8, 'summary_en': 1.8, 'road_layout': 1.4,
          'ego_motion': 1.5, 'visible_actors': 2.4, 'observed_interactions': 2.2, 'tags': 1.0}
_SPLIT = re.compile(r'[，,。；;!?？\n]|\b(?:but|however)\b|但是|但', re.I)
_NEGATIVE = re.compile(r'(?:没有|不要|不含|不需要|未见|未发现|看不到|无|避免|排除)[^，,。；;!?？\n]{0,10}$|\b(?:no|not|without|exclude|excluding|avoid|absent)\b[^,;.!?]{0,28}$', re.I)
_UNCERTAIN = re.compile(r'不确定|无法确认|无法确定|是否|不可见|未确认|不清晰|未知|可能|\b(?:unknown|uncertain|unclear|unconfirmed|cannot confirm|not visible|not sure|whether|may|might|possibly|perhaps)\b', re.I)
_COMPILED = {name: re.compile('|'.join(
    sorted((re.escape(a) if re.search(r'[\u3400-\u9fff]', a) else r'(?<![a-z0-9])' + re.escape(a) + r'(?![a-z0-9])' for a in aliases), key=len, reverse=True)), re.I)
    for name, aliases in ALIASES.items()}


def tokens(text):
    result = []
    for word in re.findall(r'[a-z0-9_]+|[\u3400-\u9fff]+', text.lower()):
        if re.fullmatch(r'[\u3400-\u9fff]+', word):
            result.extend(word[i:i + size] for size in (1, 2) for i in range(len(word) - size + 1))
        else:
            result.append(word)
    return result


def _negative_before(prefix):
    match = _NEGATIVE.search(prefix)
    if match is None:
        return False
    scope = match.group()
    if re.match(r'not\s+only\b', scope, re.I):
        return False
    remainder = re.sub(r'^(?:没有|不要|不含|不需要|未见|未发现|看不到|无|避免|排除|no|not|without|exclude|excluding|avoid|absent)\s*', '', scope, flags=re.I)
    return not bool(re.search(r'(?<!没)有|\b(?:with|has|having|there is|there are)\b', remainder, re.I))


def evidence(text, *, query=False):
    positive, negative, uncertain, cleaned = set(), set(), set(), []
    for clause in _SPLIT.split(text):
        masked = list(clause)
        for name, pattern in _COMPILED.items():
            for match in pattern.finditer(clause):
                before = clause[:match.start()]
                # Unknown observations cannot establish presence or absence.
                if not query and _UNCERTAIN.search(clause):
                    uncertain.add(name)
                    masked[match.start():match.end()] = ' ' * (match.end() - match.start())
                elif _negative_before(before):
                    negative.add(name)
                    masked[match.start():match.end()] = ' ' * (match.end() - match.start())
                else:
                    positive.add(name)
        cleaned.append(''.join(masked))
    return {'positive': positive, 'negative': negative, 'uncertain': uncertain, 'text': ' '.join(cleaned)}


def split_conditions(requirement, limit=8):
    # Use the user's clauses verbatim; keyword expansion cannot add conditions.
    clauses = [x.strip() for x in re.split(r'[，,。；;\n]+', requirement) if x.strip()]
    return clauses[:limit - 1] + ['；'.join(clauses[limit - 1:])] if len(clauses) > limit else clauses


def bm25(query, documents):
    terms = set(tokens(query))
    docs = [Counter(tokens(text)) for text in documents]
    if not docs:
        return []
    lengths = [sum(doc.values()) for doc in docs]
    average = sum(lengths) / len(docs) or 1
    frequency = Counter(term for doc in docs for term in doc)
    return [sum(math.log(1 + (len(docs) - frequency[t] + .5) / (frequency[t] + .5)) * doc[t] * 2.5 /
                (doc[t] + 1.5 * (.25 + .75 * length / average)) for t in terms if doc[t])
            for doc, length in zip(docs, lengths)]


def _caption_evidence(caption):
    fields, positive, negative, uncertain = {}, set(), set(), set()
    for name in FIELDS:
        value = caption.get(name, '')
        parts = value if isinstance(value, list) else [value]
        observations = [evidence(part) for part in parts if isinstance(part, str)]
        fields[name] = ' '.join(item['text'] for item in observations)
        for item in observations:
            positive.update(item['positive']); negative.update(item['negative']); uncertain.update(item['uncertain'])
    return fields, positive, negative - positive, uncertain - positive


def hybrid_rank(requirement, rows, keywords=(), *, pool_size=24, candidate_count=4, diversity=.12):
    query = evidence(requirement, query=True)
    entries = [_caption_evidence(json.loads(row['caption'])) for row in rows]
    original = [0.] * len(rows)
    expanded = [0.] * len(rows)
    expansion = ' '.join(keywords) + ' ' + ' '.join(' '.join(ALIASES[c]) for c in sorted(query['positive']))
    # Expansion is a separate, lower-weight channel. Never concatenate it with
    # the original query, and remove terms for explicitly excluded concepts.
    for name in query['negative']:
        expansion = _COMPILED[name].sub(' ', expansion)
    for name, weight in FIELDS.items():
        documents = [entry[0][name] for entry in entries]
        for target, text in ((original, query['text']), (expanded, expansion)):
            for index, value in enumerate(bm25(text, documents)):
                target[index] += value * weight
    coverage, contradictions, absence = [], [], []
    for fields, positive, negative, uncertain in entries:
        coverage.append(len(query['positive'] & positive) / max(len(query['positive']), 1))
        contradictions.append(len(query['negative'] & positive) / max(len(query['negative']), 1))
        absence.append(len(query['negative'] & negative) / max(len(query['negative']), 1))
    channels = ((original, 2.5), (expanded, .75), (coverage, 1.5))
    fused = [0.] * len(rows)
    total_weight = sum(weight for values, weight in channels if any(v > 0 for v in values)) or 1
    for values, weight in channels:
        ordered = sorted((i for i, value in enumerate(values) if value > 0), key=lambda i: (-values[i], rows[i]['id']))
        for rank, index in enumerate(ordered, 1):
            fused[index] += weight / (60 + rank) * 61 / total_weight
    scored = []
    for i, row in enumerate(rows):
        if not any((original[i], expanded[i], coverage[i], absence[i])):
            continue
        lexical = original[i] / (original[i] + 8)
        score = (.5 * fused[i] + .2 * lexical + .3 * coverage[i] if query['positive'] else .75 * fused[i] + .25 * lexical)
        score = max(0., score - .45 * contradictions[i] + .12 * absence[i])
        caption_tokens = set(tokens(' '.join(entries[i][0].values())))
        scored.append({'row': row, 'score': score, 'token_set': caption_tokens,
                       'evidence': {'original_bm25': round(original[i], 6), 'expanded_bm25': round(expanded[i], 6),
                                    'concept_coverage': coverage[i], 'excluded_concept_conflicts': sorted(query['negative'] & entries[i][1]),
                                    'caption_supported_concepts': sorted(entries[i][1]),
                                    'caption_uncertain_concepts': sorted(entries[i][3])}})
    pool = sorted(scored, key=lambda entry: (-entry['score'], entry['row']['id']))[:pool_size]
    selected, used = [], set()
    while pool and len(selected) < candidate_count:
        def value(entry):
            redundancy = max((len(entry['token_set'] & other['token_set']) / max(len(entry['token_set'] | other['token_set']), 1) for other in selected), default=0.)
            return (1 - diversity) * entry['score'] - diversity * redundancy
        best = max(pool, key=value)
        pool.remove(best)
        metadata = json.loads(best['row']['metadata'])
        identity = metadata.get('scene_id') or best['row']['source_file']
        if identity in used or best['score'] <= 0:
            continue
        used.add(identity); selected.append(best)
    details = {'version': RECALL_VERSION, 'positive_concepts': sorted(query['positive']), 'excluded_concepts': sorted(query['negative']),
               'original_conditions': split_conditions(requirement), 'pool_size': min(pool_size, len(scored)),
               'candidate_count': len(selected), 'caption_uncertainties_used_as_positive_evidence': False}
    return selected, details
