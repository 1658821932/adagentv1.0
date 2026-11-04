import csv
import pickle
from pathlib import Path
import pytest
from mcp_servers.worldengine.adapter.evaluate_batch import evaluate_batch


def setup_case(tmp_path, score, nc=1):
    scenario=tmp_path/'scene.pkl'
    # Geometry overlaps intentionally; official scores must remain authoritative.
    tracks={k:{'state':{'position':[[0,0,0]],'length':[4],'width':[2]}} for k in ('ego','car')}
    with scenario.open('wb') as f: pickle.dump({'case':{'id':'case','object_track':tracks}},f)
    rollout=tmp_path/'result/WE_output/openscene_format';(rollout/'meta_datas').mkdir(parents=True)
    with (rollout/'meta_datas/case.pkl').open('wb') as f: pickle.dump([{'scene_token':'case','frame_idx':0}],f)
    path=rollout/'all_scenes_pdm_averages_NR.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,['token','score','no_at_fault_collisions']);w.writeheader();w.writerow({'token':'case','score':score,'no_at_fault_collisions':nc})
    return scenario,tmp_path/'result',path


def test_official_positive_score_overrides_overlapping_geometry(tmp_path):
    s,r,_=setup_case(tmp_path,.8149)
    out=evaluate_batch(s,r)
    assert not out['is_failure_case']
    assert out['scene_results'][0]['pdm_score']==.8149


def test_zero_pdm_selects_failure_even_without_collision(tmp_path):
    s,r,_=setup_case(tmp_path,0)
    out=evaluate_batch(s,r)
    assert out['is_failure_case']
    assert out['failure_cases'][0]['pdm_score']==0


@pytest.mark.parametrize('score',['nan','inf',-1,''])
def test_invalid_official_score_never_falls_back_to_geometry(tmp_path,score):
    s,r,_=setup_case(tmp_path,score)
    with pytest.raises(ValueError): evaluate_batch(s,r)


def test_duplicate_pdm_files_are_rejected(tmp_path):
    s,r,p=setup_case(tmp_path,.8)
    p.with_name('all_scenes_pdm_averages_R.csv').write_bytes(p.read_bytes())
    with pytest.raises(ValueError): evaluate_batch(s,r)


def test_missing_result_dir_cannot_fall_back_to_geometry(tmp_path):
    scenario, _, _ = setup_case(tmp_path, 0.8)
    with pytest.raises(ValueError, match="result_dir"):
        evaluate_batch(scenario)


def test_missing_pdm_csv_cannot_fall_back_to_geometry(tmp_path):
    scenario, result, score_file = setup_case(tmp_path, 0.8)
    score_file.unlink()
    with pytest.raises(FileNotFoundError, match="PDM"):
        evaluate_batch(scenario, result)


def test_duplicate_scene_rows_are_rejected(tmp_path):
    scenario, result, score_file = setup_case(tmp_path, 0.8)
    text = score_file.read_text()
    score_file.write_text(text + text.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="Duplicate official PDM scene"):
        evaluate_batch(scenario, result)


def test_missing_score_for_executed_scene_is_rejected(tmp_path):
    import json
    scenario, result, score_file = setup_case(tmp_path, 0.8)
    with scenario.open("rb") as stream:
        payload = pickle.load(stream)
    payload["second"] = {**payload["case"], "id": "second"}
    with scenario.open("wb") as stream:
        pickle.dump(payload, stream)
    (result / "runner_report_0.json").write_text(json.dumps([{"scenario_name":"case"}, {"scenario_name":"second"}]))
    with pytest.raises(ValueError, match="Missing official PDM scores"):
        evaluate_batch(scenario, result)


def test_public_tool_requires_result_dir_before_backend():
    from tools.registry import call_tool
    result = call_tool("failure_case_filter", {"scenario_pkl":"/unavailable.pkl"}, mode="real")
    assert result["error"]["type"] == "input_validation_error"


def test_only_total_score_is_read_and_no_collision_fields_returned(tmp_path):
    s, r, path = setup_case(tmp_path, .8)
    path.write_text("token,score,no_at_fault_collisions,comfort\ncase,0.8,not-read,not-read\n")
    out = evaluate_batch(s, r)
    assert not out["is_failure_case"]
    assert out["scene_results"] == [{"scene_id":"case", "pdm_score":.8}]
    assert set(out) == {"is_failure_case", "failure_cases", "scene_count", "scene_results", "pdm_scores_file"}
