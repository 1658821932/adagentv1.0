import unittest
import json
import tempfile
from pathlib import Path
from fractions import Fraction

from mcp_servers.ufo.metric import fit_metres_per_native_unit, metric_offset_to_native_world
from mcp_servers.ufo.metric_edit import _load_offsets, _video_rate


class UFOMetricTests(unittest.TestCase):
    def test_video_rate_does_not_overflow_after_float_rounding(self):
        times = [i * (1001 / 30000) for i in range(40)]
        old_rate = Fraction(str((len(times) - 1) / (times[-1] - times[0])))
        self.assertGreater(old_rate.numerator, 2147483647)
        rate = _video_rate(times)
        self.assertEqual(rate, Fraction(30000, 1001))
        self.assertLessEqual(rate.numerator, 2147483647)
        self.assertLessEqual(rate.denominator, 2147483647)

    def test_video_rate_preserves_explicit_nominal_fraction(self):
        times = [i * 1001 / 30000 for i in range(40)]
        original = list(times)
        self.assertEqual(_video_rate(times, '30000/1001'), Fraction(30000, 1001))
        self.assertEqual(_video_rate(times, '6000/301'), Fraction(6000, 301))
        self.assertEqual(times, original)
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            _video_rate([0, 0])
        with self.assertRaisesRegex(ValueError, 'frame rate'):
            _video_rate(times, '0')

    def test_recovers_scale_despite_outliers(self):
        pairs = [(float(i), float(i) * 2.5) for i in range(1, 201)]
        pairs += [(10.0, 1000.0)] * 10
        fitted = fit_metres_per_native_unit(pairs)
        self.assertAlmostEqual(fitted["metres_per_native_unit"], 2.5)
        self.assertLess(fitted["median_relative_error"], 0.01)

    def test_rejects_inconsistent_depths(self):
        pairs = [(1.0, float(i)) for i in range(1, 201)]
        with self.assertRaisesRegex(ValueError, "residual"):
            fit_metres_per_native_unit(pairs)

    def test_right_three_metres_uses_calibrated_world_units(self):
        camera_to_world = [
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ]
        self.assertEqual(
            metric_offset_to_native_world((0, -3, 0), camera_to_world, 2.5),
            (0.0, -1.2, 0.0),
        )

    def test_offsets_require_explicit_metric_axes_and_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offsets.json"
            payload = {"units": "metres", "coordinate_system": "camera_local_forward_left_up",
                       "offsets": [[0, -3, 0]]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_load_offsets(path, 3), [(0.0, -3.0, 0.0)] * 3)
            payload["units"] = "native"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicitly use metres"):
                _load_offsets(path, 3)
            payload["units"] = "metres"
            payload["offsets"] = [[0, float("nan"), 0]]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite"):
                _load_offsets(path, 3)


if __name__ == "__main__":
    unittest.main()
