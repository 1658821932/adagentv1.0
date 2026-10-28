from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

try:
    import av
    import numpy as np
except ImportError:
    av = np = None


@unittest.skipIf(av is None or np is None, 'Video integration fixture needs PyAV and numpy')
class VideoCompareTests(unittest.TestCase):
    def video(self, path, color, timestamps):
        with av.open(str(path), 'w') as output:
            stream = output.add_stream('libx264', rate=10)
            stream.width = stream.height = 16
            stream.pix_fmt = 'yuv420p'
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000000)
            image = np.empty((16, 16, 3), dtype=np.uint8)
            image[:] = color
            for stamp in timestamps:
                frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                frame.pts, frame.time_base = stamp, Fraction(1, 1000000)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)

    def test_preserves_variable_timestamps_and_panel_colors(self):
        from mcp_servers.ufo.video_compare import compare_videos
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            timestamps = [0, 100000, 250000, 400000]
            self.video(root / 'before.mp4', [255, 0, 0], timestamps)
            self.video(root / 'after.mp4', [0, 0, 255], timestamps)
            result = compare_videos(root / 'before.mp4', root / 'after.mp4', root / 'comparison.mp4')
            self.assertEqual(result['frames'], 4)
            with av.open(result['comparison_video']) as check:
                frames = list(check.decode(video=0))
                self.assertEqual([round(frame.pts * frame.time_base * 1000000) for frame in frames], timestamps)
                image = frames[0].to_ndarray(format='rgb24')
                self.assertEqual(image.shape, (16, 32, 3))
                self.assertGreater(int(image[8, 8, 0]), 200)
                self.assertLess(int(image[8, 8, 2]), 30)
                self.assertGreater(int(image[8, 24, 2]), 200)
                self.assertLess(int(image[8, 24, 0]), 30)

    def test_rejects_mismatched_frames_without_replacing_existing_output(self):
        from mcp_servers.ufo.video_compare import compare_videos
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            self.video(root / 'before.mp4', [255, 0, 0], [0, 100000])
            self.video(root / 'after.mp4', [0, 0, 255], [0])
            output = root / 'comparison.mp4'
            output.write_bytes(b'preserve prior output')
            with self.assertRaisesRegex(ValueError, 'frame counts'):
                compare_videos(root / 'before.mp4', root / 'after.mp4', output)
            self.assertEqual(output.read_bytes(), b'preserve prior output')
            self.assertEqual(list(root.glob('comparison.*.mp4')), [])


if __name__ == '__main__':
    unittest.main()
