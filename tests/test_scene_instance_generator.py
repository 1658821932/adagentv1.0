"""Offline tool02 tests: verified tool01 result in, three-path scene plan out."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from tools.tool02_scene_instance_generator import mock, real
from tools.tool02_scene_instance_generator.schemas import Input, Output, Response
from tools.registry import call_tool

TOOL01 = {
    "token": "2021.09.16.19.49.00_veh-42_00990_01609-4abb45d9ee1a5c5c",
    "simulation": "experiment/batch",
    "map_location": "us-pa-pittsburgh-hazelwood",
    "evidence_dir": "/logs/tool01/experiment/4abb45d9ee1a5c5c/run-1",
    "failure": {"outcome": "collision", "interaction": "crossing",
                "ego_cause": "在丁字路口右转时，没有对从右侧支路左转过来的车辆让行"},
    "attack": {"map_context": "t_junction", "ego_action": "turn_right",
               "other_origin": "right_arm", "other_action": "turn_left"},
    "environment": {"time": "2021-09-16T20:16:41+00:00", "weather": "cloudy", "extra": "路面干燥"},
    "verification": {"matched": True, "notes": []},
}
TOOL01_NONE = deepcopy(TOOL01)
TOOL01_NONE["attack"] = {"map_context": "curve", "ego_action": "straight",
                         "other_origin": "none", "other_action": "none"}
FORWARDED = {"failure", "attack", "environment", "verification"}
FIELDS = {"target_scene", "attack", "candidates"}
CANDIDATE_FIELDS = {"candidate_id", "base_video_requirement", "attack_edit_strategy",
                    "ego_input_edit_strategy", "ego_correct_trajectory_strategy"}


@pytest.fixture(autouse=True)
def records_in_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(real, "RECORDS_ROOT", tmp_path / "records")


class Client:
    model = "offline-test"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = []

    def chat(self, messages):
        self.calls += 1
        self.messages.append(deepcopy(messages))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


def upstream(**changes):
    value = deepcopy(TOOL01)
    value.update(changes)
    return value


def envelope(value=None):
    return {"status": "success", "tool": "failure_analyzer", "result": value or upstream()}


def response(attack=None):
    out = mock.run({"failure_analysis": upstream()})
    assert out["status"] == "success", out
    result = out["result"]
    if attack is not None:
        result["attack"] = deepcopy(attack)
    return result


def call(client, **kwargs):
    return real.run({"failure_analysis": deepcopy(TOOL01), "_llm_client": client, **kwargs})


def payload_of(client):
    return json.loads(client.messages[0][1]["content"])


def test_result_shape_and_recorded_source():
    before = deepcopy(TOOL01)
    client = Client(response())
    out = call(client)
    assert out["status"] == "success", out
    assert set(out["result"]) == FIELDS
    assert set(out["result"]) == set(Output.model_fields)
    path = Path(out["meta"]["record_dir"])
    saved = json.loads((path / "input.json").read_text())["failure_analysis"]
    assert set(saved) == FORWARDED | {"source"}
    assert saved["attack"] == before["attack"]
    assert saved["failure"] == before["failure"]
    assert "_llm_client" not in (path / "input.json").read_text()
    assert json.loads((path / "output.json").read_text()) == out


def test_model_payload_carries_the_new_structure_only():
    client = Client(response())
    assert call(client)["status"] == "success"
    payload = payload_of(client)
    assert set(payload) == {"failure_analysis"}
    forwarded = payload["failure_analysis"]
    assert set(forwarded) == FORWARDED
    assert forwarded["attack"] == TOOL01["attack"]
    assert forwarded["failure"]["ego_cause"] == TOOL01["failure"]["ego_cause"]
    assert forwarded["environment"] == TOOL01["environment"]
    assert forwarded["verification"] == TOOL01["verification"]
    text = json.dumps(payload, ensure_ascii=False)
    for trace_only in ("4abb45d9ee1a5c5c", "experiment/batch", "run-1", "sensor_blobs"):
        assert trace_only not in text


def test_mock_is_a_fixed_valid_example_without_failure_wording():
    first = mock.run({"failure_analysis": upstream()})
    second = mock.run({"failure_analysis": upstream()})
    assert first["status"] == "success"
    assert first["result"] == second["result"]
    Output.model_validate(first["result"])
    assert "冲突" not in first["result"]["target_scene"]
    assert "碰撞" not in first["result"]["target_scene"]
    assert [item["candidate_id"] for item in first["result"]["candidates"]] == ["cand_1", "cand_2", "cand_3"]


def test_candidates_carry_only_the_five_fields():
    for candidate in response()["candidates"]:
        assert set(candidate) == CANDIDATE_FIELDS
        assert candidate["ego_correct_trajectory_strategy"].strip()


def test_mock_through_the_public_registry_entry():
    out = call_tool("scene_instance_generator", {"failure_analysis": upstream()}, mode="mock")
    assert out["status"] == "success", out
    Output.model_validate(out["result"])
    assert set(out["result"]) == FIELDS
    assert "interaction" not in out["result"]["candidates"][0]


def test_attack_must_repeat_the_upstream_structure():
    invented = dict(TOOL01["attack"], other_origin="left_arm")
    bad = response(attack=invented)
    client = Client(bad, bad)
    out = call(client)
    assert out["status"] == "error", out
    assert out["error"]["type"] == "output_validation_error"
    assert client.calls == 2


def test_none_target_allows_ego_input_editing_without_an_attack_vehicle():
    item = none_response()
    item["candidates"][0]["ego_input_edit_strategy"] = "把自车接近路口的输入轨迹改为更快进入弯道"
    out = real.run({"failure_analysis": deepcopy(TOOL01_NONE),
                    "_llm_client": Client(item)})
    assert out["status"] == "success", out
    assert out["result"]["attack"] == TOOL01_NONE["attack"]
    assert all(candidate["attack_edit_strategy"] is None for candidate in out["result"]["candidates"])
    assert out["result"]["candidates"][0]["ego_input_edit_strategy"]


def none_response():
    item = response(attack=TOOL01_NONE["attack"])
    for candidate in item["candidates"]:
        candidate["attack_edit_strategy"] = None
    return item


def test_none_upstream_keeps_a_none_attack_and_invents_no_vehicle():
    out = real.run({"failure_analysis": deepcopy(TOOL01_NONE),
                    "_llm_client": Client(none_response())})
    assert out["status"] == "success", out
    assert out["result"]["attack"] == TOOL01_NONE["attack"]
    invented = response(attack=TOOL01["attack"])
    blocked = real.run({"failure_analysis": deepcopy(TOOL01_NONE), "_llm_client": Client(invented, invented)})
    assert blocked["status"] == "error"
    assert blocked["error"]["type"] == "output_validation_error"


def test_none_upstream_rejects_a_vehicle_edit_strategy():
    item = none_response()
    item["candidates"][1]["attack_edit_strategy"] = "新增一辆右侧支路车辆"
    out = real.run({"failure_analysis": deepcopy(TOOL01_NONE), "_llm_client": Client(item, item)})
    assert out["status"] == "error", out
    assert out["error"]["type"] == "output_validation_error"
    assert "attack_edit_strategy must be null" in json.dumps(out["error"], ensure_ascii=False)


@pytest.mark.parametrize("field", sorted(FORWARDED))
def test_upstream_needs_every_forwarded_field(field):
    value = upstream()
    value.pop(field)
    client = Client()
    out = real.run({"failure_analysis": value, "_llm_client": client})
    assert out["error"]["type"] == "input_validation_error"
    assert client.calls == 0


def test_unverified_or_failed_upstream_is_rejected_before_the_model():
    unverified = upstream(verification={"matched": False, "notes": ["画面不符"]})
    for value in (unverified, {"status": "error", "error": {"message": "failed"}}, {"failure_reason": "old"}):
        client = Client()
        out = real.run({"failure_analysis": value, "_llm_client": client})
        assert out["error"]["type"] == "input_validation_error", out
        assert client.calls == 0


def test_upstream_envelope_is_accepted():
    out = call_tool("scene_instance_generator",
                    {"failure_analysis": {"status": "success", "tool": "failure_analyzer",
                                          "result": upstream()}}, mode="mock")
    assert out["status"] == "success", out


def test_upstream_result_ref_is_loaded_instead_of_retyping_its_contents(tmp_path):
    reference = tmp_path / "tool01-result.json"
    reference.write_text(json.dumps(envelope(), ensure_ascii=False))
    client = Client(response())
    out = real.run({"failure_analysis": str(reference), "_llm_client": client})
    assert out["status"] == "success", out
    recorded = json.loads((Path(out["meta"]["record_dir"]) / "input.json").read_text())
    assert recorded["failure_analysis"]["source"]["token"] == TOOL01["token"]
    assert recorded["failure_analysis"]["attack"] == TOOL01["attack"]


def test_upstream_envelope_tool_identity_is_required():
    client = Client()
    wrong = dict(envelope(), tool="video_retriever")
    out = real.run({"failure_analysis": wrong, "_llm_client": client})
    assert out["error"]["type"] == "input_validation_error"
    assert client.calls == 0


def test_invalid_attack_in_upstream_is_rejected():
    for attack in (dict(TOOL01["attack"], map_context="roundabout"),
                   {"map_context": "curve", "ego_action": "straight", "other_origin": "none"},
                   dict(TOOL01["attack"], extra="nope")):
        client = Client()
        out = real.run({"failure_analysis": upstream(attack=attack), "_llm_client": client})
        assert out["error"]["type"] == "input_validation_error", out
        assert client.calls == 0


def test_legacy_input_fields_are_rejected():
    for extra in ({"parameter_assignment": {"lighting": "dusk"}},
                  {"variation_hints": {"lighting": "dusk"}},
                  {"existing_instances": ["old target scene"]}):
        with pytest.raises(ValueError):
            Input.model_validate({"failure_analysis": upstream(), **extra})


def test_identical_candidates_are_rejected_after_retry():
    item = response()
    item["candidates"] = [deepcopy(item["candidates"][0]) for _ in range(3)]
    for position, candidate in enumerate(item["candidates"], start=1):
        candidate["candidate_id"] = f"cand_{position}"
    client = Client(item, item)
    out = call(client)
    assert out["status"] == "error"
    assert out["error"]["type"] == "output_validation_error"
    assert client.calls == 2


def test_deduplication_ignores_the_correct_trajectory_wording():
    base = response()
    template = deepcopy(base["candidates"][0])
    item = deepcopy(base)
    item["candidates"] = []
    for position, wording in enumerate(("减速通过", "保持车距通过", "稍晚转向通过"), start=1):
        candidate = deepcopy(template)
        candidate["candidate_id"] = f"cand_{position}"
        candidate["ego_correct_trajectory_strategy"] = wording
        item["candidates"].append(candidate)
    out = call(Client(item, item))
    assert out["status"] == "error", out
    assert out["error"]["type"] == "output_validation_error"
    assert "distinct" in json.dumps(out["error"], ensure_ascii=False)


def test_source_fields_are_recorded_but_never_sent_to_the_model():
    client = Client(response())
    out = call(client)
    assert out["status"] == "success", out
    recorded = json.loads((Path(out["meta"]["record_dir"]) / "input.json").read_text())
    assert recorded["failure_analysis"]["source"] == {
        "token": TOOL01["token"], "simulation": TOOL01["simulation"],
        "map_location": TOOL01["map_location"], "evidence_dir": TOOL01["evidence_dir"],
    }
    payload_text = client.messages[0][1]["content"]
    assert "source" not in json.loads(payload_text)["failure_analysis"]
    for trace_only in (TOOL01["token"], TOOL01["simulation"], TOOL01["evidence_dir"]):
        assert trace_only not in payload_text


def test_registry_entry_keeps_the_source_in_the_record():
    out = call_tool("scene_instance_generator", {"failure_analysis": upstream()}, mode="mock")
    assert out["status"] == "success", out
    recorded = json.loads((Path(out["meta"]["record_dir"]) / "input.json").read_text())
    assert recorded["failure_analysis"]["source"]["token"] == TOOL01["token"]
    assert recorded["failure_analysis"]["source"]["evidence_dir"] == TOOL01["evidence_dir"]


def test_result_without_source_fields_stays_valid():
    minimal = {key: TOOL01[key] for key in FORWARDED}
    out = real.run({"failure_analysis": minimal, "_llm_client": Client(response())})
    assert out["status"] == "success", out
    recorded = json.loads((Path(out["meta"]["record_dir"]) / "input.json").read_text())
    assert "source" not in recorded["failure_analysis"]


def test_non_text_source_field_is_rejected():
    client = Client()
    out = real.run({"failure_analysis": upstream(token=5), "_llm_client": client})
    assert out["error"]["type"] == "input_validation_error", out
    assert client.calls == 0


def test_output_schema_rejections():
    base = response()
    for legacy in ("scene_requirement", "target_effect"):
        item = deepcopy(base)
        item[legacy] = "legacy"
        with pytest.raises(ValidationError):
            Response.model_validate(item)
    for legacy in ("edit_strategy", "interaction"):
        item = deepcopy(base)
        item["candidates"][0][legacy] = "legacy"
        with pytest.raises(ValidationError):
            Response.model_validate(item)
    item = deepcopy(base)
    item["candidates"][1]["candidate_id"] = "cand_1"
    with pytest.raises(ValidationError):
        Response.model_validate(item)
    item = deepcopy(base)
    item["candidates"][0]["ego_correct_trajectory_strategy"] = None
    with pytest.raises(ValidationError):
        Response.model_validate(item)
    item = deepcopy(base)
    item["attack"]["ego_action"] = "reverse"
    with pytest.raises(ValidationError):
        Response.model_validate(item)
    item = deepcopy(base)
    item["attack"] = {"map_context": "curve", "ego_action": "straight",
                      "other_origin": "none", "other_action": "turn_left"}
    with pytest.raises(ValidationError):
        Response.model_validate(item)
    item = deepcopy(none_response())
    item["candidates"][1]["attack_edit_strategy"] = "新增一辆车"
    with pytest.raises(ValidationError, match="attack_edit_strategy must be null"):
        Response.model_validate(item)
