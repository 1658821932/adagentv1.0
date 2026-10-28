"""Request-independent object profiles, symbolic matching, visual verification.

No expected labels, benchmark case IDs, or hand-picked instance IDs are used here.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import inspect
import json
import math
import os
import re
import statistics
import urllib.error
from pathlib import Path

from .selection import file_hash, validate_selection

STRATEGY_VERSION = "grounded_v2.0"
TYPES = {"car", "sedan", "suv", "van", "truck", "bus", "motorcycle", "unknown"}
COLORS = {
    "white",
    "black",
    "gray",
    "silver",
    "blue",
    "red",
    "green",
    "yellow",
    "orange",
    "brown",
    "unknown",
}


def component_hash(*functions):
    """Cache only unchanged stage logic, with model/provider identity attached."""
    content = "\n".join(inspect.getsource(f) for f in functions)
    content += json.dumps(
        {
            "types": sorted(TYPES),
            "colors": sorted(COLORS),
            "model": os.environ.get("TARGET_SELECTOR_MODEL", "qwen3-vl-plus"),
            "endpoint": os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()


def image_content(card):
    path = Path(card["path"])
    if file_hash(path) != card["sha256"]:
        raise ValueError("object card changed after detection")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,"
            + base64.b64encode(path.read_bytes()).decode(),
            "detail": "high",
        },
    }


def anchored_frame_card(source, instance, frame, output):
    """Show the exact requested frame when the standard object card skipped it."""
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError:
        # Keep imaging dependencies in ufoex; the lightweight Agent environment
        # only makes online selection requests and verifies evidence hashes.
        candidates = json.loads(source.read_text())
        if candidates.get('source', {}).get('backend') != 'ufo':
            raise
        import subprocess
        from mcp_servers.ufo.runtime import PROJECT, python_path, offline_env
        payload = {'source': str(source), 'instance': instance, 'frame': frame, 'output': str(output)}
        process = subprocess.run([str(python_path()), '-m', 'mcp_servers.ufo.evidence'],
                                 input=json.dumps(payload), text=True, capture_output=True,
                                 cwd=PROJECT, env=offline_env(), check=True)
        return json.loads(process.stdout)

    record = next(r for r in instance["frames"] if r["frame_index"] == frame)
    original_path = source.parent / "evidence" / "frames" / f"{frame:03d}.jpg"
    original = Image.open(original_path).convert("RGB")
    x1, y1, x2, y2 = record["bbox"]
    crop = ImageOps.contain(original.crop((x1, y1, x2, y2)), (330, 240))
    context = ImageOps.contain(original, (850, 300))
    sx, sy = context.width / original.width, context.height / original.height
    ImageDraw.Draw(context).rectangle(
        (x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline="#00ffff", width=4
    )
    canvas = Image.new("RGB", (1200, 350), "#111827")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), f"C{instance['instance_id']} exact frame {frame}: crop | cyan target in context", fill="white")
    canvas.paste(crop, (15, 60))
    canvas.paste(context, (350, 40))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=96)
    return {"path": str(output), "sha256": file_hash(output), "frame_indices": [frame]}


def call_json(prompt, images, log_path, validator):
    from tools.tool10_scene_validator.real import _dashscope_chat, _extract_json

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("target_selector requires DASHSCOPE_API_KEY")
    content = [
        {"type": "text", "text": prompt},
        *[image_content(card) for card in images],
    ]
    messages = [{"role": "user", "content": content}]
    records = []
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            response = _dashscope_chat(
                os.environ.get(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                key,
                {
                    "model": os.environ.get("TARGET_SELECTOR_MODEL", "qwen3-vl-plus"),
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 2500,
                    "response_format": {"type": "json_object"},
                },
                180,
            )
        except (TimeoutError, urllib.error.URLError) as exc:
            records.append({"attempt": attempt + 1, "error": str(exc)})
            log_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
            if attempt:
                raise
            continue
        record = {
            "attempt": attempt + 1,
            "strategy": STRATEGY_VERSION,
            "prompt": prompt,
            "image_hashes": [i["sha256"] for i in images],
            "response": response,
        }
        records.append(record)
        try:
            value = _extract_json(response["choices"][0]["message"]["content"])
            validator(value)
        except (ValueError, TypeError, KeyError) as exc:
            record["validation_error"] = str(exc)
            log_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n"
            )
            if attempt:
                raise ValueError(f"Invalid grounded selector output: {exc}") from exc
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": response["choices"][0]["message"]["content"],
                    },
                    {
                        "role": "user",
                        "content": f"Correct schema only: {exc}. Do not invent visual evidence.",
                    },
                ]
            )
        else:
            log_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n"
            )
            return value


def validate_profile(value):
    if value.get("vehicle_type") not in TYPES or value.get("color") not in COLORS:
        raise ValueError("profile requires enumerated vehicle_type and color")
    if type(value.get("identity_consistent")) is not bool:
        raise ValueError("profile identity_consistent must be boolean")
    if value.get("view") not in {"rear", "front", "side", "unknown"}:
        raise ValueError("profile requires rear/front/side/unknown view")
    if value.get("emergency") is not None and type(value["emergency"]) is not bool:
        raise ValueError("profile emergency must be boolean or null")
    if (
        not isinstance(value.get("description"), str)
        or not value["description"].strip()
    ):
        raise ValueError("profile requires visual description")


def profile_scene(candidates, source_hash, directory):
    directory = Path(directory)
    model = os.environ.get("TARGET_SELECTOR_MODEL", "qwen3-vl-plus")
    key = hashlib.sha256(
        (
            STRATEGY_VERSION
            + component_hash(profile_scene, validate_profile)
            + source_hash
            + model
        ).encode()
    ).hexdigest()[:20]
    root = directory / "profiles" / key
    root.mkdir(parents=True, exist_ok=True)
    cards = candidates.get("object_cards", {})
    if any(str(i["instance_id"]) not in cards for i in candidates["instances"]):
        raise ValueError(
            "grounded strategy needs per-object cards; rerun instance_discovery"
        )

    def describe(instance):
        ident = instance["instance_id"]
        card = cards[str(ident)]
        image_content(card)  # Verify cache inputs too.
        path = root / f"C{ident}.json"
        if path.is_file():
            cached = json.loads(path.read_text())
            validate_profile(cached)
            return ident, cached
        prompt = f"""Describe only TARGET C{ident} in this object card, independently of any editing request.
Each row: LEFT is a tight crop of exactly the target. RIGHT is full-scene context with the target outlined CYAN.
Do NOT describe an unboxed nearby truck as the target. Compare all rows for physical identity.
Detector prior: {instance.get("class", "vehicle")} (fallible; inspect the pixels).
Return ONLY JSON: {{"vehicle_type":"sedan|suv|van|truck|bus|motorcycle|car|unknown", "color":"white|black|gray|silver|blue|red|green|yellow|orange|brown|unknown", "view":"rear|front|side|unknown", "emergency":false, "identity_consistent":true, "description":"Chinese visual appearance grounded in the tight crop", "limitations":[]}}.
view describes the visible vehicle end: rear shows tail lamps/license/doors, front shows headlights/grille. It is appearance, not a claim of measured velocity.
Use unknown/null when not visible. A large bbox does not imply truck. White cargo vans have a tall box body; SUVs have a passenger body. Do not guess from size or ID.
"""
        value = call_json(
            prompt, [card], root / f"C{ident}.responses.json", validate_profile
        )
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        return ident, value

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        profiles = dict(pool.map(describe, candidates["instances"]))
    path = root / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "strategy": STRATEGY_VERSION,
                "candidate_sha256": source_hash,
                "profiles": profiles,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return profiles, str(path)


def validate_spec(spec):
    if type(spec.get("reference_resolved")) is not bool:
        raise ValueError("reference_resolved must be boolean")
    if spec.get("choice") not in {"explicit", "any_suitable"}:
        raise ValueError("choice must be explicit or any_suitable")
    if spec.get("relation") not in {"none", "left_of", "right_of"}:
        raise ValueError("relation must be none, left_of, right_of")
    for field in ("target", "reference"):
        value = spec.get(field)
        if value is None and field == "reference":
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be an attribute object")
        if not isinstance(value.get("vehicle_types"), list) or any(
            t not in TYPES | {"any"} for t in value["vehicle_types"]
        ):
            raise ValueError("invalid requested vehicle_types")
        if not isinstance(value.get("colors"), list) or any(
            c not in COLORS | {"dark", "light"} for c in value["colors"]
        ):
            raise ValueError("invalid requested colors")
        if value.get("screen_position", "any") not in {
            "any",
            "left",
            "right",
            "center",
        }:
            raise ValueError("invalid screen_position")
        if value.get("ordinal") not in {
            None,
            "leftmost",
            "rightmost",
            "largest",
            "nearest_camera",
            "nearest_to_reference",
        }:
            raise ValueError("invalid ordinal")
        frame = value.get("visible_at_frame")
        if frame is not None and (type(frame) is not int or frame < 0):
            raise ValueError("visible_at_frame must be a nonnegative integer or null")
        if value.get("view", "any") not in {"any", "rear", "front", "side"}:
            raise ValueError("requested view must be any/rear/front/side")
    if spec["relation"] != "none" and spec.get("reference") is None:
        raise ValueError("relative relation requires reference attributes")
    if spec["relation"] == "none" and spec.get("reference") is not None:
        raise ValueError("reference requires an explicit supported relative relation")


def parse_request(request, directory):
    root = Path(directory) / "queries"
    path = root / (
        hashlib.sha256(
            (
                STRATEGY_VERSION
                + component_hash(parse_request, validate_spec)
                + request
            ).encode()
        ).hexdigest()
        + ".json"
    )
    if path.is_file():
        spec = json.loads(path.read_text())
        validate_spec(spec)
        return spec, str(path)
    prompt = (
        """Parse the ORIGINAL referent of an edit request, NOT a desired future action. Do not choose an instance ID.
Separate target attributes from reference-object attributes, and separate object location from action direction.
Example '货车右侧的白色厢式车向左切入': target van/white, reference truck, relation right_of. '向左切入' does not mean target screen_position=left.
Screen left/right means image coordinates, x grows rightward. '货车左侧' is left_of truck, never 'truck is left of target'.
Tight adjacency ('紧邻/紧挨') means target ordinal nearest_to_reference. Do not invent ordinal if not requested.
Without a reference object, '近处/近旁镜头' means nearest_camera; do not use nearest_to_reference. Apparent area is only a depth cue among otherwise matching vehicles.
'车' without a subtype is any; '轿车' is sedan; 'SUV' is suv; '厢式车/面包车' is van; '货车' is truck.
If '刚才那辆' has no supplied history, reference_resolved=false. If no object is specified and the user permits choosing a suitable car, choice=any_suitable; otherwise explicit.
Return ONLY JSON: {"reference_resolved":true,"choice":"explicit","target":{"vehicle_types":[],"colors":[],"screen_position":"any","ordinal":null,"emergency":null,"visible_at_frame":null,"view":"any"},"reference":null,"relation":"none","reason":"Chinese explanation"}.
Allowed vehicle_types: any, car, sedan, suv, van, truck, bus, motorcycle. Empty lists impose no attribute restriction.
Allowed colors: white, black, gray, silver, blue, red, green, yellow, orange, brown, dark, light.
When reference is present it has the same attribute schema as target. Ordinal: null,leftmost,rightmost,largest,nearest_camera,nearest_to_reference.
For '初始画面/开始时/第0帧' set visible_at_frame=0 for target and reference; explicit '第N帧' sets N. Otherwise null. Do not confuse frame of the original target with the future edit's start time.
For '同向/车尾' require view=rear, '对向/车头' view=front. Future turns do not constrain current view.
'最靠左/最靠右' are ordinal leftmost/rightmost over candidates, not an absolute image half; use screen_position=any unless the request separately says '左半边/右半边/左侧/右侧'. '正前方/画面中央' is screen_position=center.
"""
        + "\nREQUEST: "
        + request
    )
    spec = call_json(prompt, [], path.with_suffix(".responses.json"), validate_spec)
    # A missing reference object is not missing conversational history. Only
    # demonstratives that actually refer to earlier context require it.
    if spec["reference"] is None and not re.search(
        r"刚才那辆|之前那辆|上一辆|前面提到的|刚才说的|那辆车|那个目标",
        request,
    ):
        spec["reference_resolved"] = True
    if spec["reference"] is None and spec["target"].get("ordinal") == "nearest_to_reference":
        spec["target"]["ordinal"] = "nearest_camera" if "近处" in request else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    return spec, str(path)


def position(instance, width, frame=None):
    records = (
        [r for r in instance["frames"] if r["frame_index"] == frame]
        if frame is not None
        else instance["frames"][:5]
    )
    if not records:
        raise ValueError("position requested outside visible frames")
    return statistics.median((r["bbox"][0] + r["bbox"][2]) / 2 / width for r in records)


def attribute_match(instance, profile, attributes, width):
    reasons = []
    if not profile.get("identity_consistent", False):
        reasons.append("跨帧身份不一致")
    requested = set(attributes.get("vehicle_types", []))
    accepted = set(requested)
    if "car" in accepted:
        accepted |= {"sedan", "suv", "van"}
    if requested and "any" not in requested and profile["vehicle_type"] not in accepted:
        reasons.append("车型不符")
    colors = set(attributes.get("colors", []))
    if "dark" in colors:
        colors |= {"black", "gray", "blue", "brown"}
    if "light" in colors:
        colors |= {"white", "silver", "gray"}
    if colors and profile["color"] not in colors:
        reasons.append("颜色不符")
    if (
        attributes.get("emergency") is not None
        and profile.get("emergency") is not attributes["emergency"]
    ):
        reasons.append("应急车辆属性不符")
    if (
        attributes.get("view", "any") != "any"
        and profile.get("view", "unknown") != attributes["view"]
    ):
        reasons.append("车头车尾朝向不符")
    frame = attributes.get("visible_at_frame")
    if frame is not None and not any(
        r["frame_index"] == frame for r in instance["frames"]
    ):
        reasons.append("指定帧中不可见")
        return reasons
    x = position(instance, width, frame)
    side = attributes.get("screen_position", "any")
    if (
        (side == "left" and x >= 0.5)
        or (side == "right" and x <= 0.5)
        or (side == "center" and not 0.35 <= x <= 0.65)
    ):
        reasons.append("画面位置不符")
    return reasons


def relative_geometry(target, reference, relation, width, frame=None):
    targets = {r["frame_index"]: r["bbox"] for r in target["frames"]}
    refs = {r["frame_index"]: r["bbox"] for r in reference["frames"]}
    common = sorted(set(targets) & set(refs))[:5]
    if frame is not None:
        common = [frame] if frame in targets and frame in refs else []
    if not common:
        return {"passes": False, "reason": "没有同帧观测", "frames": []}
    deltas = [
        ((targets[i][0] + targets[i][2]) - (refs[i][0] + refs[i][2])) / 2 / width
        for i in common
    ]
    valid = [d < -0.005 if relation == "left_of" else d > 0.005 for d in deltas]
    return {
        "passes": sum(valid) / len(valid) >= 0.8,
        "frames": common,
        "target_minus_reference_x": deltas,
        "distance": statistics.median(abs(d) for d in deltas),
    }


def apply_ordinal(ids, attributes, instances, width):
    ordinal = attributes.get("ordinal")
    if not ids or ordinal not in {"leftmost", "rightmost", "largest", "nearest_camera"}:
        return ids
    if ordinal in {"largest", "nearest_camera"}:
        frame = attributes.get("visible_at_frame")
        scores = {
            i: statistics.median(
                (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])
                for r in (
                    [r for r in instances[i]["frames"] if r["frame_index"] == frame]
                    if frame is not None
                    else instances[i]["frames"][:5]
                )
            )
            for i in ids
        }
        best = max(scores.values())
    else:
        scores = {
            i: position(instances[i], width, attributes.get("visible_at_frame"))
            for i in ids
        }
        best = (min if ordinal == "leftmost" else max)(scores.values())
    return [i for i in ids if math.isclose(scores[i], best, rel_tol=1e-6, abs_tol=1e-6)]


def solve(candidates, profiles, spec):
    validate_spec(spec)
    instances = {i["instance_id"]: i for i in candidates["instances"]}
    width = candidates["source"]["frame_size"][0]
    failures = {
        i: attribute_match(obj, profiles[i], spec["target"], width)
        for i, obj in instances.items()
    }
    ids = [i for i in instances if not failures[i]]
    references = []
    geometry = {}
    chosen_ref = None
    reference_ambiguous = False
    if not spec["reference_resolved"]:
        ids = []
        status = "ambiguous"
        reason = "需求指代缺少上下文，不能绑定对象"
    else:
        if spec.get("reference") is not None:
            references = [
                i
                for i, obj in instances.items()
                if not attribute_match(obj, profiles[i], spec["reference"], width)
            ]
            references = apply_ordinal(references, spec["reference"], instances, width)
            pairs = []
            for ident in ids:
                for ref in references:
                    if ident == ref:
                        continue
                    check = relative_geometry(
                        instances[ident],
                        instances[ref],
                        spec["relation"],
                        width,
                        spec["target"].get("visible_at_frame"),
                    )
                    geometry[f"{ident}:{ref}"] = check
                    if check["passes"]:
                        pairs.append((ident, ref, check["distance"]))
            if spec["target"].get("ordinal") == "nearest_to_reference" and pairs:
                best = min(p[2] for p in pairs)
                pairs = [p for p in pairs if math.isclose(p[2], best, abs_tol=0.005)]
            ids = sorted({p[0] for p in pairs})
            if len(ids) == 1:
                refs = {p[1] for p in pairs if p[0] == ids[0]}
                if len(refs) == 1:
                    chosen_ref = next(iter(refs))
                else:
                    ids = []
                    reference_ambiguous = True
        ids = apply_ordinal(ids, spec["target"], instances, width)
        if spec["choice"] == "any_suitable" and ids:
            ids = [
                max(
                    ids,
                    key=lambda i: (
                        instances[i].get("summary", {}).get("coverage", 0),
                        -i,
                    ),
                )
            ]
        if reference_ambiguous:
            status = "ambiguous"
            reason = "多个参照物无法唯一对应，不能绑定目标"
        elif len(ids) == 1:
            status = "selected"
            reason = "唯一候选满足对象外观、位置与参照关系约束"
        elif len(ids) > 1:
            status = "ambiguous"
            reason = "多个候选满足约束，不能唯一确定目标"
        elif spec.get("reference") is not None and not references:
            status = "not_found"
            reason = "候选中未找到可确认的参照物"
        else:
            status = "not_found"
            reason = "候选中没有同时满足对象属性与空间约束的目标"
    ident = ids[0] if status == "selected" else None
    ranking = [
        {
            "instance_id": i,
            "reason": "；".join(failures[i])
            or ("满足外观约束；需结合空间约束" if i != ident else reason),
        }
        for i in sorted(instances, key=lambda i: (i != ident, bool(failures[i]), i))
    ]
    evidence = []
    if ident is not None:
        card = candidates.get("object_cards", {}).get(str(ident), {})
        frames = card.get(
            "frame_indices", [instances[ident]["frames"][0]["frame_index"]]
        )
        evidence = [
            {"frame_index": k, "observation": profiles[ident]["description"]}
            for k in frames
        ]
    result = {
        "decision": status,
        "selected_instance_id": ident,
        "reason": reason,
        "confidence": 1.0 if ident is not None else 0.0,
        "ranking": ranking,
        "evidence": evidence,
        "reference_instance_id": chosen_ref,
        "geometry_checks": geometry,
        "parsed_request": spec,
        "limitations": [],
    }
    validate_selection(result, candidates)
    return result


def validate_verification(value):
    if value.get("target_matches") not in {"yes", "no", "uncertain"}:
        raise ValueError("verification requires target_matches yes/no/uncertain")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("verification requires visual reason")


def run(input_data):
    request = input_data.get("edit_request")
    if not isinstance(request, str) or not request.strip():
        raise ValueError("target_selector requires edit_request")
    source = Path(input_data["instance_info_path"]).expanduser().resolve()
    output = Path(input_data["output_path"]).expanduser().resolve()
    if source == output:
        raise ValueError("selection output must not overwrite candidates")
    digest = file_hash(source)
    candidates = json.loads(source.read_text())
    root = Path(input_data.get("cache_dir", source.parent / "grounded_cache"))
    profiles, catalog_path = profile_scene(candidates, digest, root)
    spec, spec_path = parse_request(request, root)
    decision = solve(candidates, profiles, spec)
    if decision["decision"] == "selected":
        ident = decision["selected_instance_id"]
        cards = [candidates["object_cards"][str(ident)]]
        anchored = spec["target"].get("visible_at_frame")
        if anchored is not None:
            instance = next(i for i in candidates["instances"] if i["instance_id"] == ident)
            cards.append(
                anchored_frame_card(
                    source,
                    instance,
                    anchored,
                    output.with_name(output.stem + f".frame_{anchored:03d}.jpg"),
                )
            )
        ref = decision["reference_instance_id"]
        if ref is not None:
            cards.append(candidates["object_cards"][str(ref)])
        appearance = {
            key: spec["target"].get(key)
            for key in ("vehicle_types", "colors", "view", "emergency")
        }
        prompt = f"""Independently verify visual appearance and track identity for proposed target C{ident}. The first image card has tight target crops on the LEFT and full-scene context with the target boxed CYAN on the RIGHT. The crop and cyan box are generated from the same tracked bbox. Other unboxed vehicles are not the proposed target.
{"The second image shows the exact requested frame with C" + str(ident) + " boxed in cyan." if anchored is not None else ""}
{"The last image is reference C" + str(ref) + "." if ref is not None else ""}
ORIGINAL REQUEST: {request}
TARGET APPEARANCE TO VERIFY: {json.dumps(appearance, ensure_ascii=False)}
REFERENCE APPEARANCE TO VERIFY: {json.dumps(spec.get("reference"), ensure_ascii=False)}
The computer has already checked frame visibility, image position, ordinal area/rank, and target-reference geometry across all detected candidates. These cards do not show every candidate, so do not recompute those constraints or compare the target with vehicles from other frames. Verify that the boxed target is a consistent vehicle and meets the requested visual type, color, orientation, and emergency attributes. Check the reference appearance if provided. Future edit actions are irrelevant to original appearance.
Return ONLY JSON {{"target_matches":"yes|no|uncertain","reason":"Chinese visual evidence or contradiction"}}.
"""
        verification = call_json(
            prompt,
            cards,
            output.with_name(output.stem + ".verification.responses.json"),
            validate_verification,
        )
        decision["verification"] = verification
        if verification["target_matches"] != "yes":
            decision.update(
                {
                    "decision": "ambiguous",
                    "selected_instance_id": None,
                    "confidence": 0.0,
                    "reason": "独立视觉核验未通过：" + verification["reason"],
                }
            )
    if file_hash(source) != digest:
        raise ValueError("candidate snapshot changed during selection")
    result = {
        **decision,
        "strategy": STRATEGY_VERSION,
        "edit_request": request,
        "candidate_info_path": str(source),
        "candidate_info_sha256": digest,
        "catalog_path": catalog_path,
        "request_spec_path": spec_path,
        "model": os.environ.get("TARGET_SELECTOR_MODEL", "qwen3-vl-plus"),
        "evidence_images": [c["path"] for c in candidates["object_cards"].values()],
    }
    validate_selection(result, candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    if result["decision"] == "selected":
        selected = next(
            i
            for i in candidates["instances"]
            if i["instance_id"] == result["selected_instance_id"]
        )
        path = output.with_name(output.stem + ".instances.json")
        if path == source:
            raise ValueError("selected instance file must not overwrite candidates")
        path.write_text(
            json.dumps(
                {
                    "instances": [selected],
                    "source": candidates["source"],
                    "target_selection_path": str(output),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        result["selected_instance_info_path"] = str(path)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {
        "status": "success",
        "tool": "target_selector",
        "result": {**result, "selection_path": str(output)},
    }
