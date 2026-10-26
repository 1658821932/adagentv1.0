import shutil
import subprocess
import argparse
from tempfile import TemporaryDirectory
from pathlib import Path
import os
file_dir = Path(__file__).parent

MEGASAM_CHECKPOINT = file_dir / "checkpoints" / "megasam_final.pth"
RAFT_CHECKPOINT = file_dir / "cvd_opt" / "raft-things.pth"
DAV1_CHECKPOINT = file_dir / "Depth-Anything" / "checkpoints" / "depth_anything_vitl14.pth"
DAV2_CHECKPOINT = file_dir / "Depth-Anything-V2" / "checkpoints" / "depth_anything_v2_vitl.pth"
VIDEODEPTHANYTHING_CHECKPOINT = file_dir / "Video-Depth-Anything" / "checkpoints" / "video_depth_anything_vitl.pth"
DAV1_SCRIPT = file_dir / "Depth-Anything" / "run_videos.py"
DAV2_SCRIPT = file_dir / "Depth-Anything-V2" / "run_videos.py"
VIDEODEPTHANYTHING_SCRIPT = file_dir / "Video-Depth-Anything" / "run_videos.py"
MOGE_SCRIPT = file_dir / "MoGe" / "run_videos.py"
UNIDEPTH_SCRIPT = file_dir / "UniDepth" / "scripts" / "demo_mega-sam.py"
CAMERA_TRACKING_SCRIPT = file_dir / "camera_tracking_scripts" / "test_demo.py"
PREPROCESS_FLOW_SCRIPT = file_dir / "cvd_opt" / "preprocess_flow.py"
CVD_OPT_SCRIPT = file_dir / "cvd_opt" / "cvd_opt.py"
DEPTHANYTHING_V1_CMD = (
    f"""
      python {DAV1_SCRIPT.resolve()} \
          --encoder vitl \
          --load-from {DAV1_CHECKPOINT.resolve()} \
          --img-path 'images/test' \
          --outdir 'Depth-Anything/video_visualization/test'
    """
)
DEPTHANYTHING_V2_CMD = (
    f"""
      python {DAV2_SCRIPT.resolve()} \
          --encoder vitl \
          --load-from {DAV2_CHECKPOINT.resolve()} \
          --img-path 'images/test' \
          --outdir 'Depth-Anything-V2/video_visualization/test'
    """
)
VIDEODEPTHANYTHING_CMD = (
    f"""
      python {VIDEODEPTHANYTHING_SCRIPT.resolve()} \
          --encoder vitl \
          --load-from {VIDEODEPTHANYTHING_CHECKPOINT.resolve()} \
          --img-path 'images/test' \
          --outdir 'Video-Depth-Anything/video_visualization/test'
    """
)
MOGE_CMD = f"python {MOGE_SCRIPT.resolve()} --img-path 'images/test' --outdir 'MoGe/video_visualization/test' --fov_x '{{fov_x}}'"
UNIDEPTH_CMD = (
    f"""
      python {UNIDEPTH_SCRIPT.resolve()} \
          --scene-name test \
          --img-path 'images/test' \
          --outdir 'UniDepth/outputs'
    """
)
CAMERA_TRACKING_CMD = (
    f"""
    python {CAMERA_TRACKING_SCRIPT.resolve()} \
        --datapath=images/test \
        --weights={MEGASAM_CHECKPOINT.resolve()} \
        --scene_name test \
        --mono_depth_path Depth-Anything/video_visualization \
        --metric_depth_path UniDepth/outputs \
        --disable_vis \
        --fov_x '{{fov_x}}' \
        --depth_for_cvd '{{depth_for_cvd}}' \
        --resolution '{{resolution}}'
    """
)
PREPROCESS_FLOW_CMD = (
    f"""
    python {PREPROCESS_FLOW_SCRIPT.resolve()} \
        --datapath=images/test \
        --model={RAFT_CHECKPOINT.resolve()} \
        --scene_name test --mixed_precision \
        --resolution '{{resolution}}'
    """
)
CVD_OPT_CMD = (
    f"""  
    python {CVD_OPT_SCRIPT.resolve()} \
        --scene_name test \
        --w_grad 2.0 --w_normal 5.0
    """
)

CVD_OPT_MOGE_CMD = (
    f"""  
    python {CVD_OPT_SCRIPT.resolve()} \
        --scene_name test \
        --w_grad 2.0 --w_normal 5.0 \
        --freeze_shift
    """
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--fov_x", type=str, default="")
    parser.add_argument("--depth_model", type=str, default="dav1", choices=["dav1", "dav2", "videoda", "moge"])
    parser.add_argument("--resolution", type=int, default=384 * 512, help='resolution (pixels)')
    args = parser.parse_args()

    assert Path(args.input_dir).exists(), f"Input directory {args.input_dir} does not exist"

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        (temp_dir / "images").mkdir(parents=True, exist_ok=True)
        shutil.copytree(args.input_dir, temp_dir / "images" / "test")

        env = os.environ.copy()
        # dav1 & moge for camera tracking
        env["PYTHONPATH"] = (file_dir / "Depth-Anything").resolve()
        subprocess.run(DEPTHANYTHING_V1_CMD, shell=True, cwd=temp_dir, env=env)
        env["PYTHONPATH"] = str ((file_dir / "UniDepth").resolve())
        subprocess.run(UNIDEPTH_CMD, shell=True, cwd=temp_dir, env=env)

        if args.depth_model != "dav1":
            if args.depth_model == "dav2":
                depth_cmd = DEPTHANYTHING_V2_CMD
                env["PYTHONPATH"] = (file_dir / "Depth-Anything-V2").resolve()
            elif args.depth_model == "videoda":
                depth_cmd = VIDEODEPTHANYTHING_CMD
                env["PYTHONPATH"] = (file_dir / "Video-Depth-Anything").resolve()
            elif args.depth_model == "moge":
                depth_cmd = MOGE_CMD.format(fov_x=args.fov_x)
                env["PYTHONPATH"] = (file_dir / "MoGe").resolve()
            else:
                raise ValueError(f"Invalid depth model: {args.depth_model}")
            subprocess.run(depth_cmd, shell=True, cwd=temp_dir, env=env)

        env["PYTHONPATH"] = str (file_dir.resolve())
        if args.depth_model == "dav1":
            depth_for_cvd = "Depth-Anything/video_visualization"
        elif args.depth_model == "dav2":
            depth_for_cvd = "Depth-Anything-V2/video_visualization"
        elif args.depth_model == "videoda":
            depth_for_cvd = "Video-Depth-Anything/video_visualization"
        elif args.depth_model == "moge":
            depth_for_cvd = "MoGe/video_visualization"
        else:
            raise ValueError(f"Invalid depth model: {args.depth_model}")
        subprocess.run(CAMERA_TRACKING_CMD.format(fov_x=args.fov_x, depth_for_cvd=depth_for_cvd, resolution=args.resolution), shell=True, cwd=temp_dir, env=env)
        subprocess.run(PREPROCESS_FLOW_CMD.format(resolution=args.resolution), shell=True, cwd=temp_dir, env=env)
        if args.depth_model != "moge":
            subprocess.run(CVD_OPT_CMD, shell=True, cwd=temp_dir, env=env)
        else:
            subprocess.run(CVD_OPT_MOGE_CMD, shell=True, cwd=temp_dir, env=env)

        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(temp_dir / "outputs_cvd" / "test_sgd_cvd_hr.npz", args.output_path)
