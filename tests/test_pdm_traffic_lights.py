"""Signal-agnostic experiments require an explicit choice and preserve source data."""
from copy import deepcopy
import pytest
from mcp_servers.scene_editing.core.adversarial.pdm_traffic_lights import traffic_light_policy


def test_signal_scenes_still_rejected_by_default():
    with pytest.raises(ValueError,match='conversion required'):
        traffic_light_policy({'baseline':[{'traffic_lights':[[1,True]]}]})


def test_explicit_ignore_preserves_annotations_and_reports_scope():
    frames={'baseline':[{'traffic_lights':[[1,True],[2,False]]}],
            'attack':[{'traffic_lights':[[1,True],[2,False]]},{'traffic_lights':[]}]}
    before=deepcopy(frames)
    result=traffic_light_policy(frames,ignore=True)
    assert frames==before
    assert result['mode']=='ignored'
    assert result['source_annotation_counts']['attack']==dict(frames_with_signals=1,signal_entries=2)
    assert result['traffic_light_compliance']=='not_evaluated'
    assert not result['signals_passed_to_planner'] and not result['source_annotations_modified']


def test_empty_signal_scene_retains_default_mode():
    result=traffic_light_policy({'baseline':[{'traffic_lights':[]}]})
    assert result['mode']=='require_empty'
