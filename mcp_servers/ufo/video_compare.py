"""Write synchronized before/after videos with PyAV, without CLI ffmpeg."""
from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import zip_longest
import json
import os
from pathlib import Path
import tempfile


def compare_videos(before: Path, after: Path, output: Path) -> dict:
    import av
    import numpy as np

    before, after, output = (Path(path).resolve() for path in (before, after, output))
    if output in (before, after):
        raise ValueError('Comparison output must differ from both input videos')
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=output.stem + '.', suffix='.mp4', dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    requested_times = []
    try:
        with av.open(str(before)) as left, av.open(str(after)) as right, av.open(str(temporary), 'w') as encoded:
            stream = None
            origins = None
            for index, pair in enumerate(zip_longest(left.decode(video=0), right.decode(video=0))):
                a, b = pair
                if a is None or b is None:
                    raise ValueError('Before/after videos have different frame counts')
                if a.pts is None or b.pts is None:
                    raise ValueError('Before/after frames must contain timestamps')
                stamps = [Fraction(frame.pts) * frame.time_base for frame in (a, b)]
                if origins is None:
                    origins = stamps
                times = [value - origin for value, origin in zip(stamps, origins)]
                if abs(times[0] - times[1]) > Fraction(1, 100000):
                    raise ValueError(f'Before/after timestamps differ at frame {index}')
                images = [frame.to_ndarray(format='rgb24') for frame in (a, b)]
                if images[0].shape != images[1].shape:
                    raise ValueError('Before/after frame dimensions differ')
                image = np.concatenate(images, axis=1)
                if stream is None:
                    stream = encoded.add_stream('libx264', rate=left.streams.video[0].average_rate or 10)
                    stream.width, stream.height = image.shape[1], image.shape[0]
                    stream.pix_fmt = 'yuv420p'
                    stream.options = {'crf': '18', 'preset': 'fast'}
                    stream.time_base = stream.codec_context.time_base = Fraction(1, 1000000)
                elif (stream.width, stream.height) != (image.shape[1], image.shape[0]):
                    raise ValueError('Video dimensions change during comparison')
                frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                frame.pts = round(times[0] * 1000000)
                frame.time_base = Fraction(1, 1000000)
                requested_times.append(Fraction(frame.pts, 1000000))
                for packet in stream.encode(frame):
                    encoded.mux(packet)
            if stream is None:
                raise ValueError('Cannot compare empty videos')
            for packet in stream.encode():
                encoded.mux(packet)
        with av.open(str(temporary)) as check:
            actual_times = [Fraction(frame.pts) * frame.time_base for frame in check.decode(video=0)]
        if len(actual_times) != len(requested_times) or any(
            abs(actual - wanted) > Fraction(1, 1000000) for actual, wanted in zip(actual_times, requested_times)
        ):
            raise ValueError('Comparison encoding changed frame count or timestamps')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'comparison_video': str(output), 'frames': len(requested_times), 'timestamps_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_videos(args.before, args.after, args.out)))


if __name__ == '__main__':
    main()
