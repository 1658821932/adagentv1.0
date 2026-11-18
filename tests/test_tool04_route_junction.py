"""Route order fixes the target before its suitability is assessed."""
import pytest
pytest.importorskip('shapely')
from shapely.geometry import box
from tools.tool04_edit_planner.route_junction import first_route_junction


def test_start_inside_real_junction_wins_over_later_target():
    selected = first_route_junction([(0, 0), (10, 0)],
                                    [(1, box(-2, -2, 2, 2), False),
                                     (2, box(5, -2, 8, 2), False)])
    assert selected['intersection_fid'] == 1
    assert selected['entry_distance_m'] == 0


def test_order_is_along_route_not_euclidean_distance_or_input_order():
    route = [(0, 0), (0, 10), (10, 10), (10, 1), (1, 1)]
    selected = first_route_junction(route, [(2, box(.5, .5, 1.5, 1.5), False),
                                            (1, box(-1, 4, 1, 6), False)])
    assert selected['intersection_fid'] == 1
    assert selected['entry_distance_m'] == 4


def test_first_crossed_junction_is_found_between_sparse_frames():
    selected = first_route_junction([(0, 0), (20, 0)],
                                    [(7, box(4, -1, 6, 1), False),
                                     (8, box(12, -1, 14, 1), False)])
    assert selected['intersection_fid'] == 7
    assert selected['entry_frame_index'] == 0
    assert selected['review_xy'] == [5, 0]


def test_mini_intersections_are_skipped():
    selected = first_route_junction([(0, 0), (20, 0)],
                                    [(1, box(2, -1, 4, 1), True),
                                     (2, box(6, -1, 8, 1), False)])
    assert selected['intersection_fid'] == 2


def test_nearby_unvisited_junction_is_not_substituted():
    assert first_route_junction([(0, 0), (10, 0)], [(1, box(3, 1, 5, 3), False)]) is None


def test_boundary_only_touch_does_not_count_as_entering():
    assert first_route_junction([(0, 0), (10, 0)], [(1, box(3, 0, 5, 2), False)]) is None


def test_stationary_start_inside_junction():
    selected = first_route_junction([(0, 0), (0, 0)], [(1, box(-1, -1, 1, 1), False)])
    assert selected['intersection_fid'] == 1


def test_missing_route_does_not_select_a_junction():
    assert first_route_junction([], [(1, box(-1, -1, 1, 1), False)]) is None
