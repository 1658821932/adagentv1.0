"""Completed preparation stages are reused without trusting stale artifacts."""
import json
import signal
import sys
from pathlib import Path
import pytest
from mcp_servers.scene_editing.core.adversarial import ego_stage_reuse as reuse

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))

@pytest.fixture
def stages(tmp_path, monkeypatch):
    implementation = tmp_path/"implementation"
    write(implementation/"backend_settings.json", {"budget": 2})
    monkeypatch.setattr(reuse, "ROOT", implementation)
    scene = tmp_path/"scene"
    images = tmp_path/"images"
    write(scene/"metadata/frames.json", [{"timestamp": 1}])
    write(images/"metadata/image_manifest.csv", {"timestamp": 1})
    (images/"original_images").mkdir()
    (images/"original_images/a.jpg").write_bytes(b"recorded image")
    args = dict(attack_result=dict(frozen_scene_path=str(scene), target_track="target",
        source_scene_ref=dict(sample_root=str(images))), target_effect="Stop safely",
        allow_safe_stop=True, use_foci=True)
    policy = {"profile": "test"}
    source = tmp_path/"source"
    write(source/"input/stage_request.json", dict(stage="ego_editing", arguments=args))
    output = source/"output"
    brief = output/"defense_understanding"
    write(brief/"summary.json", dict(status="completed"))
    write(brief/"config.json", dict(scene_dir=str(scene), target_track="target"))
    write(brief/"conflict-brief.json", dict(directions=[]))
    write(brief/"requests/old.json", {"request": "not copied"})
    reuse.save_completed("defense_understanding", args, policy, 24, 2, output,
        reuse.binding("defense_understanding", args, policy, 24, 2, output))
    pool = output/"defense_constructor"
    write(pool/"reference_screen/summary.json", dict(status="reference_screen_complete",
        candidates=[{"candidate_id":"candidate_001"}]))
    write(pool/"config.json", dict(pool_path=str(pool)))
    (pool/"trajectory.csv").write_text("time,x,y\n0,1,2\n")
    reuse.save_completed("defense_constructor", args, policy, 24, 2, output,
        reuse.binding("defense_constructor", args, policy, 24, 2, output))
    foci = output/"defense_foci"
    write(foci/"initial-stage.json", dict(status="completed", summary_path=str(foci/"summary.json")))
    write(foci/"summary.json", dict(candidates=[{"candidate_id":"foci_001"}]))
    write(foci/"config.json", dict(parent_run=str(pool)))
    reuse.save_completed("defense_foci", args, policy, 24, 2, output,
        reuse.binding("defense_foci", args, policy, 24, 2, output))
    return source, args, policy, scene, images, implementation

def restore(stage, fixture, dest):
    source, args, policy, *_ = fixture
    return reuse.restore(stage, source, args, policy, 24, 2, dest)

def test_all_stages_reused_without_probe_binding_and_paths_relocated(stages, tmp_path):
    source, args, policy, *_ = stages
    dest = tmp_path/"new/output"
    for stage in reuse.STAGES:
        assert restore(stage, stages, dest)
        reuse.validate_completed(stage, dest/stage)
    assert not (dest/"defense_understanding/requests").exists()
    assert not (source/"output/defense_constructor/probe-binding.json").exists()
    config = reuse.read(dest/"defense_foci/config.json")
    assert config["parent_run"] == str(dest/"defense_constructor")
    assert (source/"output/defense_understanding/requests/old.json").exists()

@pytest.mark.parametrize("change", ["task", "scene", "images", "code", "policy", "api"])
def test_changed_inputs_recompute_instead_of_reusing(stages, tmp_path, monkeypatch, change):
    source, args, policy, scene, images, implementation = stages
    if change == "task": args["target_effect"] = "Different objective"
    if change == "scene": write(scene/"metadata/frames.json", [{"timestamp":2}])
    if change == "images": (images/"original_images/a.jpg").write_bytes(b"different")
    if change == "code": write(implementation/"backend_settings.json", {"budget":3})
    if change == "policy": policy["profile"] = "changed"
    if change == "api": monkeypatch.setenv("ADVERSARIAL_API_MODEL", "different-model")
    dest = tmp_path/"new/output"
    assert not restore("defense_understanding", stages, dest)
    assert not (dest/"defense_understanding").exists()
    assert reuse.read(dest/"defense_understanding-reuse.json")["status"] == "recompute"

@pytest.mark.parametrize("damage", ["missing_checkpoint","missing_artifact","modified_artifact","incomplete"])
def test_incomplete_or_corrupt_stage_is_not_reused(stages, tmp_path, damage):
    source, *_ = stages
    brief = source/"output/defense_understanding"
    if damage == "missing_checkpoint": (brief/reuse.MANIFEST).unlink()
    if damage == "missing_artifact": (brief/"conflict-brief.json").unlink()
    if damage == "modified_artifact": write(brief/"conflict-brief.json", {"changed":True})
    if damage == "incomplete": write(brief/"summary.json", {"status":"running"})
    assert not restore("defense_understanding", stages, tmp_path/"new/output")

def test_completed_prefix_can_be_reused_when_later_stage_failed(stages, tmp_path):
    source, *_ = stages
    (source/"output/defense_foci"/reuse.MANIFEST).unlink()
    dest = tmp_path/"new/output"
    assert restore("defense_understanding", stages, dest)
    assert restore("defense_constructor", stages, dest)
    assert not restore("defense_foci", stages, dest)

def test_changed_upstream_constructor_invalidates_foci(stages, tmp_path):
    dest = tmp_path/"new/output"
    assert restore("defense_understanding", stages, dest)
    assert restore("defense_constructor", stages, dest)
    (dest/"defense_constructor/trajectory.csv").write_text("changed")
    assert not restore("defense_foci", stages, dest)

@pytest.mark.parametrize("status", ["no_parent","no_candidates"])
def test_empty_foci_search_is_still_a_completed_reusable_stage(stages, tmp_path, status):
    source,args,policy,*_ = stages
    output = source/"output"
    folder = output/"defense_foci"
    write(folder/"initial-stage.json", {"status":status})
    reuse.save_completed("defense_foci",args,policy,24,2,output,
        reuse.binding("defense_foci",args,policy,24,2,output))
    dest = tmp_path/"new/output"
    for stage in reuse.STAGES:
        assert restore(stage, stages, dest)

def test_inputs_changed_during_generation_are_not_checkpointed(stages):
    source,args,policy,scene,*_ = stages
    output = source/"output"
    before = reuse.binding("defense_understanding",args,policy,24,2,output)
    write(scene/"metadata/frames.json", [{"timestamp":3}])
    with pytest.raises(ValueError,match="changed during"):
        reuse.save_completed("defense_understanding",args,policy,24,2,output,before)


def test_preparation_deadline_interrupts_hung_stage_and_restores_timer():
    import time
    with pytest.raises(reuse.PreparationTimeout) as error:
        with reuse.preparation_deadline('defense_constructor',.02):
            time.sleep(.2)
    assert error.value.stage=='defense_constructor'
    assert signal.getsignal(signal.SIGALRM)==signal.SIG_DFL
    assert signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
    with reuse.preparation_deadline('defense_foci',1):
        pass



def test_constructor_progress_resumes_only_complete_matching_candidates(tmp_path):
    signature=reuse.constructor_signature({'allow_safe_stop':True},[('baseline',0.,None)],'target')
    candidate=tmp_path/'reference_screen/candidate_001';candidate.mkdir(parents=True)
    write(candidate/'trajectory.csv','time,x\n0,1\n')
    row=dict(candidate_id='candidate_001',status='reference_screen_passed')
    write(candidate/'screen.json',row)
    reuse.save_constructor_progress(tmp_path,signature,{1,2},[row],
        [dict(candidate_index=2,status='construction_rejected')])
    completed,ledger,failures=reuse.load_constructor_progress(tmp_path,signature)
    assert completed=={1,2} and ledger==[row] and failures[0]['candidate_index']==2
    with pytest.raises(ValueError,match='scene or policy'):
        reuse.load_constructor_progress(tmp_path,'different')
    (candidate/'trajectory.csv').unlink()
    with pytest.raises(ValueError,match='artifacts are incomplete'):
        reuse.load_constructor_progress(tmp_path,signature)
