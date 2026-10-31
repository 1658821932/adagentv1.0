"""Parameter forwarding tested without importing GPU models."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from mcp_servers.worldengine.adapter.launch import build_commands
from mcp_servers.worldengine.adapter.training import train_policy

def args():
    return argparse.Namespace(config="/cfg.py",checkpoint="/model.pth",model_name="test",
        data_type="demo",react_type="NR",asset_name="demo",added_name="test")

def test_absolute_paths_and_all_effective_parameters(tmp_path):
    env={"WORLDENGINE_ROOT":str(tmp_path/"engine"),"WORLDENGINE_TEST_PATH":str(tmp_path/"out"),
         "SCENARIO_DATA_FOLDER_PATH":str(tmp_path/"external scenes"),
         "NUM_HISTORY":"4","NUM_FUTURE":"8","MAX_FRAME_PER_SCENE":"12",
         "NUM_SCENARIOS":"1","SAMPLING_POSES":"3","WITH_DENSE_REWARD_MANAGER":"true",
         "SIMENGINE_PYTHON":"/custom/sim/python","ALGENGINE_PYTHON":"/custom/alg/python"}
    sim,alg,_,_,_,_,effective=build_commands(args(),env)
    assert sim[0]=="/custom/sim/python" and alg[0]=="/custom/alg/python"
    assert effective["scenario_file"]==str(tmp_path/"external scenes/all_scenarios.pkl")
    assert effective["simulator"]["num_history"]==4
    assert effective["simulator"]["num_future"]==8
    assert effective["simulator"]["+num_scenarios"]==1
    assert effective["simulator"]["sampling_poses"]==3
    assert "with_dense_reward_manager=true" in sim
    assert "sim.max_frame_per_scene=12" in alg
    assert "sim.maximum_step=12" in alg
    assert "queue_length=4" in alg

def test_defaults_derive_future_instead_of_retaining_eight(tmp_path):
    env={"WORLDENGINE_ROOT":str(tmp_path),"WORLDENGINE_TEST_PATH":str(tmp_path/"out")}
    *_,effective=build_commands(args(),env)
    assert effective["num_history"]+effective["num_future"]==effective["max_frame_per_scene"]==40

def test_default_training_config_is_usable_without_explicit_override(tmp_path, stub_training_preparation):
    import shutil
    core=tmp_path/"core"
    config=core/"experiments/posttrain/e2e_vadv2_50pct_waymo_ilft.py"
    config.parent.mkdir(parents=True);config.write_text("# valid default")
    script=core/"projects/AlgEngine/scripts/train.py";script.parent.mkdir(parents=True)
    script.write_text('import sys,json\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index("--work-dir")+1]);(p/"epoch_1.pth").write_text("fresh")\n(p/"training_audit.json").write_text(json.dumps({"finite_losses":True,"changed_parameter_count":1,"losses":{"loss":0.1}}))\nprint("loss.imi: 0.1")\n')
    files=[tmp_path/name for name in ("ann.pkl","tokens.yaml","base.pth")]
    for f in files:f.touch()
    result=train_policy({"worldengine_root":str(core),"python":sys.executable,"base_checkpoint":str(files[2]),
        "training_data":{"ann_file":str(files[0]),"token_file":str(files[1])}})
    assert Path(result["checkpoint"]).read_text()=="fresh"
    assert result["final_loss"]==0.1

def test_launcher_runs_both_commands_and_records_effective_config(tmp_path,monkeypatch):
    from mcp_servers.worldengine.adapter import launch
    root=tmp_path/"engine"
    sim=root/"projects/SimEngine";alg=root/"projects/AlgEngine"
    (sim/"worldengine/runner").mkdir(parents=True);(alg/"closed_loop").mkdir(parents=True)
    (sim/"worldengine/runner/run_simulation.py").write_text('import time\ntime.sleep(0.05)\n')
    (alg/"closed_loop/sim_test.py").write_text('import time\ntime.sleep(0.2)\n')
    out=tmp_path/"out"
    monkeypatch.setenv("WORLDENGINE_ROOT",str(root))
    monkeypatch.setenv("WORLDENGINE_TEST_PATH",str(out))
    monkeypatch.setenv("SIMENGINE_PYTHON",sys.executable)
    monkeypatch.setenv("ALGENGINE_PYTHON",sys.executable)
    monkeypatch.setattr(sys,"argv",["launch","/cfg","/ckpt","test","demo","NR","demo","x"])
    launch.main()
    assert json.loads((out/"effective_config.json").read_text())["max_frame_per_scene"]==40


def test_planner_can_exit_after_completion_flag_before_simulator_cleanup(tmp_path,monkeypatch):
    from mcp_servers.worldengine.adapter import launch
    root=tmp_path/"engine"
    sim=root/"projects/SimEngine";alg=root/"projects/AlgEngine"
    (sim/"worldengine/runner").mkdir(parents=True);(alg/"closed_loop").mkdir(parents=True)
    code = 'import sys,time\nfrom pathlib import Path\narg=next(x for x in sys.argv if x.startswith("output_dir=")); p=Path(arg.split("=",1)[1]);p.mkdir(parents=True,exist_ok=True);(p/"simulation_completed.flag").touch();time.sleep(0.4)\n'
    (sim/"worldengine/runner/run_simulation.py").write_text(code)
    (alg/"closed_loop/sim_test.py").write_text('import time\ntime.sleep(0.15)\n')
    monkeypatch.setenv("WORLDENGINE_ROOT",str(root))
    monkeypatch.setenv("WORLDENGINE_TEST_PATH",str(tmp_path/"out"))
    monkeypatch.setenv("SIMENGINE_PYTHON",sys.executable)
    monkeypatch.setenv("ALGENGINE_PYTHON",sys.executable)
    monkeypatch.setattr(sys,"argv",["launch","/cfg","/ckpt","test","demo","NR","demo","x"])
    launch.main()

def test_native_runner_flushes_last_scene_before_completion_flag(tmp_path):
    import ast
    from types import SimpleNamespace
    source=Path(__file__).resolve().parents[2]/"core/projects/SimEngine/worldengine/runner/executor.py"
    tree=ast.parse(source.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_simulation")
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),fn],type_ignores=[])
    ast.fix_missing_locations(module)
    flag=tmp_path/"simulation_completed.flag"
    persisted=[]
    def save():
        assert not flag.exists()
        persisted.append(True)
    env=SimpleNamespace(config=SimpleNamespace(output_dir=str(tmp_path)),
        engine=SimpleNamespace(managers={"data_manager":SimpleNamespace(save_data=save)}),
        run=lambda:["finished"])
    namespace={"Path":Path,"logger":SimpleNamespace(info=lambda *a:None)}
    exec(compile(module,str(source),"exec"),namespace)
    assert namespace["run_simulation"](env)==["finished"]
    assert persisted and flag.exists()

def test_relative_interpreter_overrides_are_project_relative(tmp_path):
    env={"WORLDENGINE_ROOT":str(tmp_path),"WORLDENGINE_TEST_PATH":str(tmp_path/"out"),
         "SIMENGINE_PYTHON":"environments/custom-sim/bin/python",
         "ALGENGINE_PYTHON":"environments/custom-alg/bin/python"}
    sim,alg,*_=build_commands(args(),env)
    project=Path(__file__).resolve().parents[4]
    assert sim[0]==str(project/"environments/custom-sim/bin/python")
    assert alg[0]==str(project/"environments/custom-alg/bin/python")
