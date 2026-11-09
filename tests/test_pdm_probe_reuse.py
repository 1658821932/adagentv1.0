"""PDM cache reuse binds complete scene semantics and runtime, not only filenames."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from mcp_servers.scene_editing.core.adversarial import pdm_binding,pdm_tool


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path,monkeypatch):
    scene=tmp_path/'scene';attack=tmp_path/'attack';maps=tmp_path/'maps'
    frames=[dict(timestamp=1,map_location='city',ego2global=[[1]],anns=dict(track_tokens=['other'],gt_boxes=[[1,2,3]]))]
    write(scene/'metadata/frames.json',frames);write(attack/'metadata/frames.json',frames)
    write(maps/'nuplan-maps-v1.0.json',dict(city=dict(version='v1')))
    (maps/'city').mkdir()
    settings=dict(python='/test/python',worldengine='/test/we',system_lib='/test/libs',map_root=str(maps),timeout_seconds=300)
    monkeypatch.setitem(pdm_tool.SETTINGS,'pdm',settings)
    args=dict(challenge_baseline_scene_path=str(scene),challenge_attack_scene_path=str(attack),
        challenge_history_s=3.,challenge_target_track='target',challenge_target_is_new=True)
    return args,settings


def make_probe(tmp_path,args,settings):
    source=tmp_path/'probe/pdm';source.mkdir(parents=True)
    data=dict(status='completed',contact_samples=0,road_violation_samples=0)
    for name in ('baseline','attack'):
        write(source/name/'summary.json',data)
        write(source/name/'observations.json',[dict(target_in_pdm=True,observation_timestamp_us=1,max_history_time_us=1)])
    write(source/'summary.json',dict(status='completed',results=dict(baseline=data,attack=data)))
    write(source/'binding.json',pdm_binding.binding(args,settings));write(source/'config.json',{})
    path=tmp_path/'probe/result.json';write(path,dict(status='completed',pdm_output=str(source)))
    return path,source


def test_complete_probe_reuses_without_subprocess(setup,tmp_path,monkeypatch):
    args,settings=setup;probe,source=make_probe(tmp_path,args,settings)
    def fail(*args,**kwargs):raise AssertionError('must not launch simulator twice')
    monkeypatch.setattr(pdm_tool.subprocess,'run',fail)
    args['challenge_probe_result_path']=str(probe)
    result=pdm_tool.run_pdm_challenge(args,tmp_path/'reuse',tmp_path/'reuse.log')
    assert result['pdm_defense_succeeded'] and result['summary']['probe_reused']
    assert (tmp_path/'reuse/attack/observations.json').is_file()


@pytest.mark.parametrize('change',['attack','background','history','target','runtime','traffic_lights'])
def test_different_candidate_or_context_never_reuses(setup,tmp_path,change):
    args,settings=setup;probe,source=make_probe(tmp_path,args,settings)
    if change in ('attack','background'):
        scene=args['challenge_attack_scene_path' if change=='attack' else 'challenge_baseline_scene_path']
        path=Path(scene)/'metadata/frames.json';frames=json.loads(path.read_text());frames[0]['anns']['gt_boxes'][0][0]+=.01;write(path,frames)
    elif change=='history':args['challenge_history_s']=4.
    elif change=='target':args['challenge_target_track']='other'
    elif change=='traffic_lights':args['challenge_ignore_traffic_lights']=True
    else:settings['worldengine']='/changed/we'
    args['challenge_probe_result_path']=str(probe)
    with pytest.raises(ValueError,match='differs'):pdm_tool.run_pdm_challenge(args,tmp_path/'reuse',tmp_path/'reuse.log')
    assert not (tmp_path/'reuse').exists()


def test_baseline_cache_ignores_attack_but_not_background_or_runtime(setup,tmp_path):
    args,settings=setup;probe,source=make_probe(tmp_path,args,settings)
    before=pdm_binding.binding(args,settings)
    assert pdm_binding.matching_baseline(source,before)
    path=Path(args['challenge_attack_scene_path'])/'metadata/frames.json'
    value=json.loads(path.read_text());value[0]['anns']['gt_boxes'][0][0]+=1;write(path,value)
    assert pdm_binding.matching_baseline(source,pdm_binding.binding(args,settings))
    changed=deepcopy(before);changed['baseline']['frames'][0]['anns']['gt_boxes'][0][0]+=1
    assert not pdm_binding.matching_baseline(source,changed)
    changed=deepcopy(before);changed['baseline']['runtime']['python']='/changed/python'
    assert not pdm_binding.matching_baseline(source,changed)


def test_baseline_cache_separates_ignored_traffic_light_mode(setup,tmp_path):
    args,settings=setup;_,source=make_probe(tmp_path,args,settings)
    args['challenge_ignore_traffic_lights']=True
    assert not pdm_binding.matching_baseline(source,pdm_binding.binding(args,settings))


def test_edited_only_result_needs_no_original_simulation(tmp_path):
    source=tmp_path/'pdm'
    write(source/'summary.json',dict(results=dict(attack=dict(contact_samples=1,road_violation_samples=0))))
    write(source/'attack/observations.json',[dict(target_in_pdm=True,observation_timestamp_us=3,max_history_time_us=3)])
    assert pdm_tool.read_result(source)['pdm_defense_succeeded'] is False
    assert not (source/'baseline').exists()


def test_unsafe_optional_original_diagnostic_never_gates_attack(tmp_path):
    source=tmp_path/'pdm'
    write(source/'summary.json',dict(results=dict(attack=dict(contact_samples=0,road_violation_samples=0),
                                                 baseline=dict(contact_samples=20,road_violation_samples=4))))
    write(source/'attack/observations.json',[dict(target_in_pdm=True,observation_timestamp_us=3,max_history_time_us=3)])
    assert pdm_tool.read_result(source)['pdm_defense_succeeded'] is True


def test_diagnostic_mode_is_bound_to_probe(setup,tmp_path):
    args,settings=setup;probe,_=make_probe(tmp_path,args,settings)
    args.update(challenge_run_baseline_diagnostic=True,challenge_probe_result_path=str(probe))
    with pytest.raises(ValueError,match='differs'):
        pdm_tool.run_pdm_challenge(args,tmp_path/'reuse',tmp_path/'reuse.log')
