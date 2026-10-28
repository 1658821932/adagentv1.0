import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Depth diagnostics need numpy')
class MetricDiagnosticTests(unittest.TestCase):
    def test_spatial_scale_mismatch_is_reported_without_creating_calibration(self):
        from mcp_servers.ufo.metric_diagnostics import depth_report, write_failure_report
        native = np.full((20, 20), 10.)
        metric = np.full((20, 20), 8.)
        metric[:, 10:] = 20
        classes = np.zeros_like(native, dtype=int)
        classes[:, 10:] = 8
        result = depth_report(native, metric, np.ones_like(native, dtype=bool), classes)
        self.assertAlmostEqual(result['metric_depth_bands']['0-10m']['meters_per_native_unit'], .8)
        self.assertAlmostEqual(result['metric_depth_bands']['20-40m']['meters_per_native_unit'], 2)
        self.assertGreater(result['all_static']['median_relative_error'], .15)
        with tempfile.TemporaryDirectory() as folder:
            path = write_failure_report(Path(folder), 0, result, ValueError('residual too large'))
            value = json.loads(path.read_text())
            self.assertEqual(value['status'], 'rejected')
            self.assertNotIn('calibration_path', value)
            self.assertEqual(value['required_median_relative_error_max'], .15)

    def test_invalid_and_empty_samples_do_not_produce_nonfinite_json(self):
        from mcp_servers.ufo.metric_diagnostics import depth_report
        result = depth_report(np.array([np.nan, -1, 0]), np.ones(3), np.ones(3, dtype=bool))
        self.assertEqual(result['all_static']['valid_samples'], 0)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
