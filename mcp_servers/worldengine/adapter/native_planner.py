"""Forward native AlgEngine execution with the actual scene frame budget."""
import argparse
from pathlib import Path
import runpy
import sys
try:
    from .startup_timing import get_timing
except ImportError:
    from startup_timing import get_timing

_STARTUP = get_timing('planner')
_STARTUP.mark('wrapper_entered')


def configure_monitor(frame_count):
    if frame_count < 5:
        raise ValueError('Native planner requires at least five scene frames')
    import closed_loop.monitor as monitor
    original = monitor.AsyncFileMonitor

    class ConfiguredMonitor(original):
        def __init__(self, folder_path, check_interval=1.0, max_history=4,
                     max_frame_per_scene=12, **options):
            super().__init__(folder_path=folder_path,check_interval=check_interval,
                max_history=max_history,max_frame_per_scene=frame_count,**options)

    monitor.AsyncFileMonitor = ConfiguredMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-frames',type=int,required=True)
    args, forwarded = parser.parse_known_args()
    if not forwarded:
        parser.error('Missing native AlgEngine entry point')
    with _STARTUP.span('monitor_import_and_configuration'):
        configure_monitor(args.scene_frames)
    entry = Path(forwarded[0]).resolve()
    sys.argv = [str(entry),*forwarded[1:]]
    with _STARTUP.span('native_entry_total'):
        runpy.run_path(str(entry),run_name='__main__')


if __name__ == '__main__':
    main()
