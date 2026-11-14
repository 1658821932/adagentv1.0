"""Content-bound checkpoints for completed ego preparation stages."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import sys

ROOT = Path(__file__).resolve().parent
VERSION = "ego_preparation.v1"
STAGES = ("defense_understanding", "defense_constructor", "defense_foci")
IGNORED = {"requests", "timing", "status-history", "tool-exchanges", "__pycache__"}
MANIFEST = "completed-stage.json"


class PreparationTimeout(RuntimeError):
    def __init__(self, stage, seconds):
        self.stage, self.seconds = stage, seconds
        super().__init__(f"{stage} exceeded {seconds}s")


@contextmanager
def preparation_deadline(stage, seconds):
    if type(seconds) not in (int, float) or seconds <= 0:
        raise ValueError("Preparation timeout must be positive")
    def expire(signum, frame):
        raise PreparationTimeout(stage, seconds)
    previous_handler = signal.signal(signal.SIGALRM, expire)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def constructor_signature(effective, decisions, target_track):
    return digest(dict(policy=effective, decisions=decisions, target_track=target_track))


def save_constructor_progress(run, signature, completed, ledger, failures, status='in_progress'):
    write(Path(run)/'constructor-progress.json', dict(version='constructor_progress.v1',
        status=status, signature=signature, completed_indices=sorted(completed),
        ledger=ledger, construction_failures=failures))


def load_constructor_progress(run, signature):
    path=Path(run)/'constructor-progress.json'
    if not path.is_file():
        return set(), [], []
    progress=read(path)
    if (progress.get('version')!='constructor_progress.v1'
            or progress.get('status') not in {'in_progress','completed'}
            or progress.get('signature')!=signature):
        raise ValueError('Constructor progress does not match the current scene or policy')
    completed=set(progress.get('completed_indices',[]))
    ledger=progress.get('ledger',[]);failures=progress.get('construction_failures',[])
    if (not isinstance(completed,set) or any(type(index) is not int or index<1 for index in completed)
            or not isinstance(ledger,list) or not isinstance(failures,list)):
        raise ValueError('Constructor progress is malformed')
    failed={row.get('candidate_index') for row in failures}
    if failed-completed or {row.get('candidate_id') for row in ledger} & failed:
        raise ValueError('Constructor progress has inconsistent rejected candidates')
    for row in ledger:
        cid=row.get('candidate_id');index=int(cid.rsplit('_',1)[-1])
        candidate=Path(run)/'reference_screen'/cid
        if (index not in completed or not (candidate/'trajectory.csv').is_file()
                or read(candidate/'screen.json').get('candidate_id')!=cid):
            raise ValueError('Constructor candidate artifacts are incomplete')
    return completed, ledger, failures


def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()

def inventory(folder):
    """Normalize the task workspace so relocation preserves artifact identity."""
    folder = Path(folder)
    output = str(folder.parent.resolve()).encode()
    rows = {}
    for parent, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED)
        for name in sorted(files):
            if name in {MANIFEST, "reuse-provenance.json"} or name.endswith((".log", ".pyc")):
                continue
            path = Path(parent) / name
            content = path.read_bytes().replace(output, b"$OUTPUT")
            rows[str(path.relative_to(folder))] = hashlib.sha256(content).hexdigest()
    return rows

def binding(stage, arguments, acceptance, count, foci_count, output):
    attack = arguments["attack_result"]
    scene = Path(attack["frozen_scene_path"])
    image_scene = Path(attack["source_scene_ref"]["sample_root"])
    data = {}
    for base, prefix in [(scene, "scene"), (image_scene, "image_scene")]:
        for name in ("frames.json", "cameras.json", "map_features.json", "map_reference.json",
                     "map_drivable_official.json", "no_cross_lines.json", "image_manifest.csv"):
            path = base / "metadata" / name
            data[prefix + "/" + name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    # Bind the actual recorded images too, not just their filenames.
    images = {}
    if stage == "defense_understanding":
        for path in sorted((image_scene / "original_images").rglob("*")):
            if path.is_file():
                images[str(path.relative_to(image_scene / "original_images"))] = hashlib.sha256(path.read_bytes()).hexdigest()
    implementation = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.iterdir()) if p.is_file() and p.suffix in {".py", ".json"}}
    request = {key: arguments.get(key) if key == "require_route_progress" else arguments[key]
        for key in ("attack_result", "target_effect", "allow_safe_stop", "require_route_progress", "use_foci")}
    upstream = {}
    if stage in ("defense_constructor", "defense_foci"):
        upstream["defense_understanding"] = inventory(Path(output) / "defense_understanding")
    if stage == "defense_foci":
        upstream["defense_constructor"] = inventory(Path(output) / "defense_constructor")
    return digest(dict(version=VERSION, stage=stage, request=request, data=data, images=images,
        acceptance=acceptance, count=count, foci_count=foci_count if stage=="defense_foci" else None,
        implementation=implementation, python=sys.version, executable=sys.executable,
        api={key:os.environ.get(key) for key in ("ADVERSARIAL_API_BASE_URL",
            "ADVERSARIAL_API_MODEL", "ADVERSARIAL_API_MAX_TOKENS")},
        upstream=upstream))

def validate_completed(stage, folder):
    folder = Path(folder)
    if stage == "defense_understanding":
        if read(folder / "summary.json").get("status") != "completed":
            raise ValueError("Understanding stage is not completed")
        read(folder / "config.json")
        read(folder / "conflict-brief.json")
    elif stage == "defense_constructor":
        summary = read(folder / "reference_screen/summary.json")
        if summary.get("status") != "reference_screen_complete":
            raise ValueError("Constructor stage is not completed")
        if not isinstance(summary.get("candidates"), list):
            raise ValueError("Constructor candidate list is missing")
    elif stage == "defense_foci":
        status = read(folder / "initial-stage.json").get("status")
        if status not in {"completed", "no_candidates", "no_parent"}:
            raise ValueError("FOCI stage is not completed")
        if status != "no_parent":
            read(folder / "summary.json")
    else:
        raise ValueError("Unknown preparation stage")

def save_completed(stage, arguments, acceptance, count, foci_count, output, started_binding):
    folder = Path(output) / stage
    validate_completed(stage, folder)
    if started_binding != binding(stage, arguments, acceptance, count, foci_count, output):
        raise ValueError("Preparation inputs changed during execution")
    write(folder / MANIFEST, dict(version=VERSION, stage=stage, binding=started_binding,
        artifacts=inventory(folder)))

def restore(stage, source_run, arguments, acceptance, count, foci_count, output):
    """Return False on missing, incomplete or stale stages; never bless old results."""
    if not source_run:
        return False
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    source = source_run / "output" / stage
    destination = output / stage
    reason = None
    try:
        payload = read(source_run / "input/stage_request.json")
        if payload.get("stage") != "ego_editing":
            raise ValueError("Source is not ego editing")
        previous = payload["arguments"]
        if any(previous.get(k) != arguments.get(k) for k in
               ("attack_result", "target_effect", "allow_safe_stop", "require_route_progress", "use_foci")):
            raise ValueError("Task or attack changed")
        checkpoint = read(source / MANIFEST)
        if checkpoint.get("version") != VERSION or checkpoint.get("stage") != stage:
            raise ValueError("Unsupported checkpoint")
        if checkpoint["binding"] != binding(stage, arguments, acceptance, count, foci_count, output):
            raise ValueError("Scene, configuration, implementation or upstream stage changed")
        validate_completed(stage, source)
        if checkpoint["artifacts"] != inventory(source):
            raise ValueError("Completed artifacts changed or are missing")
    except (OSError, ValueError, KeyError, TypeError) as error:
        reason = str(error)
    if reason is not None:
        write(output / (stage + "-reuse.json"), dict(status="recompute", reason=reason,
            source_run=str(source_run)))
        return False
    if destination.exists():
        raise ValueError("Reuse destination must be a fresh stage directory")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        *IGNORED, MANIFEST, "reuse-provenance.json", "*.log", "*.pyc"))
    # Rebase workspace-local references; original dataset/attack paths stay intact.
    for path in destination.rglob("*.json"):
        text = path.read_text()
        path.write_text(text.replace(str(source_run / "output"), str(output)))
    write(destination / "reuse-provenance.json", dict(status="reused", source_run=str(source_run),
        stage=stage, new_model_request=False, new_search=False))
    if inventory(destination) != checkpoint["artifacts"]:
        shutil.rmtree(destination)
        raise ValueError("Relocated stage artifacts differ")
    write(destination / MANIFEST, checkpoint)
    print("Reused completed preparation stage: " + stage, flush=True)
    return True
