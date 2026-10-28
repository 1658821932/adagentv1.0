"""Explain failed scale fits without relaxing metric acceptance criteria."""
from __future__ import annotations

import json
from pathlib import Path


def depth_report(native_depth, metric_depth, valid, classes=None) -> dict:
    import numpy as np
    native, metric = np.asarray(native_depth), np.asarray(metric_depth)
    valid = np.asarray(valid, dtype=bool).copy()
    if native.shape != metric.shape or native.shape != valid.shape:
        raise ValueError('Diagnostic depths and validity mask must have the same pixel grid')
    valid &= np.isfinite(native) & np.isfinite(metric) & (native > 0) & (metric > 0)

    def summarize(mask):
        a, b = native[mask], metric[mask]
        if not len(a):
            return {'valid_samples': 0}
        ratio = b/a
        scale = float(np.median(ratio))
        errors = np.abs(a*scale-b)/b
        return {'valid_samples': int(len(a)), 'meters_per_native_unit': scale,
                'median_relative_error': float(np.median(errors)),
                'p90_relative_error': float(np.sort(errors)[min(len(errors)-1, int(np.ceil(len(errors)*.9))-1)]),
                'scale_ratio_p10_p50_p90': np.quantile(ratio, [.1, .5, .9]).tolist()}

    result = {'all_static': summarize(valid), 'metric_depth_bands': {}, 'semantic_classes': {},
              'interpretation': 'Diagnostics only. Group fits are not accepted calibrations and must not be used as per-object/per-frame scales.'}
    for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 80), (80, 400)]:
        result['metric_depth_bands'][f'{lo}-{hi}m'] = summarize(valid & (metric >= lo) & (metric < hi))
    if classes is not None:
        classes = np.asarray(classes)
        if classes.shape != native.shape:
            raise ValueError('Diagnostic semantics must match the depth grid')
        for ident in np.unique(classes[valid]):
            result['semantic_classes'][str(int(ident))] = summarize(valid & (classes == ident))
    return result


def write_failure_report(directory: Path, frame: int, report: dict, error: Exception) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f'frame_{frame:06d}.json'
    value = {'schema_version': 1, 'status': 'rejected', 'frame': frame, 'error': str(error),
             'required_median_relative_error_max': .15, 'required_p90_relative_error_max': .25,
             'diagnostics': report,
             'next_step': 'Check camera/depth correspondence and reconstruction geometry. A single global scale is unverified; do not continue metric editing or silently increase thresholds.'}
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    return output
