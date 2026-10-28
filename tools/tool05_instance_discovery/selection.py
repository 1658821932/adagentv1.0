"""Request-conditioned visual target selection and auditable ID binding."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_selection(value: dict, candidates: dict) -> dict:
    if value.get("decision") not in {"selected", "ambiguous", "not_found"}:
        raise ValueError(
            "target selection requires selected, ambiguous, or not_found decision"
        )
    available = {item["instance_id"]: item for item in candidates["instances"]}
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("target selection requires a reason")
    ranking = value.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("target selection requires candidate ranking")
    seen = set()
    for item in ranking:
        if not isinstance(item, dict):
            raise ValueError("ranked candidates must be objects")
        ident = item.get("instance_id")
        if type(ident) is not int or ident not in available or ident in seen:
            raise ValueError("ranking contains invalid or duplicate candidate ID")
        seen.add(ident)
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("each ranked candidate requires a reason")
    if seen != set(available):
        raise ValueError("ranking must explain every candidate")
    ident = value.get("selected_instance_id")
    if value["decision"] == "selected":
        if type(ident) is not int or ident not in available:
            raise ValueError("selected ID is not a detected candidate")
        if not ranking or ranking[0]["instance_id"] != ident:
            raise ValueError("selected ID must match the first ranked candidate")
        evidence = value.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("selected target requires frame evidence")
        visible = {r["frame_index"] for r in available[ident]["frames"]}
        for item in evidence:
            if (
                not isinstance(item, dict)
                or type(item.get("frame_index")) is not int
                or item["frame_index"] not in visible
                or not isinstance(item.get("observation"), str)
                or not item["observation"].strip()
            ):
                raise ValueError(
                    "target evidence must reference a visible frame and observation"
                )
    elif ident is not None:
        raise ValueError("unresolved selection must not bind an instance")
    confidence = value.get("confidence")
    if (
        type(confidence) not in (float, int)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence must be finite and within [0,1]")
    return value


def load_bound_instances(path: str | Path) -> tuple[dict, dict]:
    """Verify the persisted decision and candidate snapshot before editing."""
    selection_path = Path(path).expanduser().resolve()
    binding = json.loads(selection_path.read_text())
    candidate_path = Path(binding["candidate_info_path"])
    if file_hash(candidate_path) != binding["candidate_info_sha256"]:
        raise ValueError("candidate snapshot changed after target selection")
    candidates = json.loads(candidate_path.read_text())
    validate_selection(binding, candidates)
    if binding["decision"] != "selected":
        raise ValueError("target selection is unresolved; editing is blocked")
    selected = next(
        i
        for i in candidates["instances"]
        if i["instance_id"] == binding["selected_instance_id"]
    )
    return binding, selected


def check_edit_config(config: dict, config_path: str | Path) -> None:
    """Apply selection guards even when scene_editor is called directly."""
    instance_path = config.get("instance_info_path")
    if not instance_path:
        return
    path = Path(instance_path).expanduser()
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    data = json.loads(path.read_text())
    if data.get("selection_required"):
        raise ValueError(
            "unselected candidate file cannot be edited; run target_selector"
        )
    if not data.get("target_selection_path"):
        return
    binding, selected = load_bound_instances(data["target_selection_path"])
    if Path(
        binding["selected_instance_info_path"]
    ).resolve() != path.resolve() or data.get("instances") != [selected]:
        raise ValueError("instance file differs from selected target binding")
    for op, params in config.get("Nodes", {}).get("RigidNodes", {}).items():
        if op not in {"trajectory_edit", "rotate", "scale", "remove", "replace"}:
            continue
        ids = params.get("target_id" if op == "replace" else "instance_id", [])
        if not ids or any(
            type(i) is not int or i != selected["instance_id"] for i in ids
        ):
            raise ValueError("edit config changed the selected target")


def legacy_run(input_data: dict[str, Any]) -> dict:
    from tools.tool10_scene_validator.real import _dashscope_chat, _extract_json

    request = input_data.get("edit_request")
    if not isinstance(request, str) or not request.strip():
        raise ValueError("target_selector requires edit_request")
    source = Path(input_data["instance_info_path"]).expanduser().resolve()
    digest = file_hash(source)
    candidates = json.loads(source.read_text())
    if not candidates.get("instances"):
        raise ValueError("target_selector requires detected candidates")
    output = Path(input_data["output_path"]).expanduser().resolve()
    if output == source:
        raise ValueError("selection output must not overwrite candidates")
    compact = [
        {
            "instance_id": i["instance_id"],
            "track_id": i.get("track_id"),
            "summary": i.get("summary"),
            "frames": i["frames"],
        }
        for i in candidates["instances"]
    ]
    prompt = """Select ONE existing car for the editing request from numbered candidates C0, C1, etc.
Use the annotated multi-frame scene AND crops, cross-check frame indices and track summaries.
Distinguish explicit object references (e.g. left white car) from choosing any suitable car.
Never silently substitute a more editable car for an explicitly requested object.
For an unspecified target, rank semantic suitability and track visibility/continuity together.
The requested FUTURE action is an edit objective, not something the source car must already do.
Do not equate screen center with ego lane or bbox size with metric distance; no metric/lane claims without evidence.
Track IDs can switch; report suspected identity changes, occlusions, and limitations.
If references cannot be resolved confidently return ambiguous; if absent return not_found.
Return ONLY JSON with decision (selected/ambiguous/not_found), selected_instance_id (integer or null),
confidence (0..1, self-reported not calibrated), reason (Chinese),
ranking (ALL candidates, best first, each {instance_id,reason}),
evidence ([{frame_index,observation}] for the chosen car), limitations ([strings]).
Do not select by bbox area alone. Candidate IDs must come from the supplied list.
"""
    content = [
        {
            "type": "text",
            "text": prompt
            + "\n"
            + json.dumps(
                {
                    "edit_request": request,
                    "source": candidates.get("source"),
                    "candidates": compact,
                },
                ensure_ascii=False,
            ),
        }
    ]
    for image in candidates.get("evidence_images", []):
        path = Path(image)
        if file_hash(path) != candidates.get("evidence_sha256", {}).get(image):
            raise ValueError("visual evidence changed since candidate detection")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(path.read_bytes()).decode()
                },
            }
        )
    if len(content) == 1:
        raise ValueError("target_selector requires annotated visual evidence")
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("target_selector requires DASHSCOPE_API_KEY")
    model = os.environ.get("TARGET_SELECTOR_MODEL", "qwen3-vl-plus")
    required_ids = [item["instance_id"] for item in candidates["instances"]]
    content[0]["text"] += (
        f"\nranking MUST contain exactly {len(required_ids)} entries, each ID exactly once: {required_ids}."
    )
    messages = [{"role": "user", "content": content}]
    responses = []
    response_path = output.with_name(output.stem + ".model_responses.json")
    if response_path == source:
        raise ValueError("model response log must not overwrite candidates")
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        response = _dashscope_chat(
            os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            api_key,
            {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 4096,
                "response_format": {"type": "json_object"},
            },
            180,
        )
        responses.append({"attempt": attempt + 1, "response": response})
        raw = response["choices"][0]["message"]["content"]
        try:
            decision = validate_selection(_extract_json(raw), candidates)
        except (ValueError, TypeError, KeyError) as exc:
            responses[-1]["validation_error"] = str(exc)
            response_path.write_text(
                json.dumps(responses, ensure_ascii=False, indent=2) + "\n"
            )
            if attempt == 1:
                raise ValueError(
                    f"target selection failed after one schema correction: {exc}"
                ) from exc
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"Correct your JSON. Validation error: {exc}. Return the complete schema, including ranking with exactly these IDs once each: {required_ids}. Keep evidence grounded in supplied frames.",
                    },
                ]
            )
        else:
            response_path.write_text(
                json.dumps(responses, ensure_ascii=False, indent=2) + "\n"
            )
            break
    minimum = input_data.get("min_confidence", 0.7)
    if (
        type(minimum) not in (int, float)
        or not math.isfinite(minimum)
        or not 0 <= minimum <= 1
    ):
        raise ValueError("min_confidence must be finite and within [0,1]")
    if decision["decision"] == "selected" and decision["confidence"] < minimum:
        decision = {
            **decision,
            "decision": "ambiguous",
            "selected_instance_id": None,
            "reason": f"Selection confidence below configured threshold {minimum}: "
            + decision["reason"],
        }
    if file_hash(source) != digest:
        raise ValueError("candidate snapshot changed during selection")
    result = {
        **decision,
        "edit_request": request,
        "candidate_info_path": str(source),
        "candidate_info_sha256": digest,
        "model": model,
        "min_confidence": minimum,
        "evidence_images": candidates["evidence_images"],
        "model_response_path": str(response_path),
        "model_attempts": len(responses),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Unresolved results have no executable instance file.
    if decision["decision"] == "selected":
        selected = next(
            i
            for i in candidates["instances"]
            if i["instance_id"] == decision["selected_instance_id"]
        )
        selected_path = output.with_name(output.stem + ".instances.json")
        if selected_path == source:
            raise ValueError("selected instance file must not overwrite candidates")
        selected_path.write_text(
            json.dumps(
                {
                    "instances": [selected],
                    "source": candidates.get("source"),
                    "target_selection_path": str(output),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        result["selected_instance_info_path"] = str(selected_path)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {
        "status": "success",
        "tool": "target_selector",
        "result": {**result, "selection_path": str(output)},
    }


def run(input_data: dict[str, Any]) -> dict:
    strategy = input_data.get("strategy", "grounded_v2")
    if strategy == "legacy_v1":
        return legacy_run(input_data)
    if strategy != "grounded_v2":
        raise ValueError("unknown target selection strategy")
    from .grounded import run as grounded_run

    return grounded_run(input_data)


def mock_run(input_data: dict) -> dict:
    return {
        "status": "success",
        "tool": "target_selector",
        "result": {
            "decision": "ambiguous",
            "selected_instance_id": None,
            "reason": "Mock mode does not perform visual selection.",
        },
    }
