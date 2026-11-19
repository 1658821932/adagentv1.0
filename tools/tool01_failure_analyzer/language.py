"""Deterministic failure wording shared by extraction and visual verification."""

BODY_DISTANCE_LIMIT_M = 5.0


def ego_cause(interaction_kind, attack, ego_action, ego_lane_keeping="unknown", outcome=None):
    place = {
        "straight_road": "直路上",
        "t_junction": "丁字路口",
        "cross_intersection": "十字路口",
        "curve": "弯道上",
        "merge": "汇流处",
    }[attack["map_context"]]
    motion = {"straight": "直行", "turn_left": "左转", "turn_right": "右转"}[ego_action]
    lane = {
        "left_offset": "并向左偏离本车道",
        "right_offset": "并向右偏离本车道",
        "keep": "并保持在本车道内",
        "unknown": "",
    }[ego_lane_keeping]
    if attack["other_origin"] == "none":
        failure = {
            "collision": "发生单车碰撞",
            "road_departure": "驶出可行驶道路",
            "collision_and_departure": "发生碰撞并驶出可行驶道路",
            None: "发生单车失败",
        }.get(outcome, "发生单车失败")
        return f"在{place}{motion}{lane}时{failure}，首次违规时刻{BODY_DISTANCE_LIMIT_M:g}米内无目标车辆"
    origin = {
        "ahead_same_lane": "同车道前方",
        "behind_same_lane": "同车道后方",
        "left_adjacent": "左相邻车道",
        "right_adjacent": "右相邻车道",
        "oncoming": "对向车道",
        "left_arm": "左侧支路",
        "right_arm": "右侧支路",
    }[attack["other_origin"]]
    action = {
        "straight": "直行",
        "turn_left": "左转",
        "turn_right": "右转",
    }[attack["other_action"]]
    if interaction_kind == "near_miss":
        observed = {
            "collision": "发生碰撞",
            "road_departure": "驶出可行驶道路",
            "collision_and_departure": "发生碰撞并驶出可行驶道路",
        }.get(outcome, "发生失败")
        return (f"在{place}{motion}{lane}时{observed}，"
                f"首次违规时刻{BODY_DISTANCE_LIMIT_M:g}米内最近车辆位于{origin}并{action}")
    if interaction_kind == "rear_end":
        return f"在{place}{motion}{lane}时，没有对{origin}{action}的车辆及时避让"
    if interaction_kind == "opposite_direction":
        return f"在{place}{motion}{lane}时，没有避让{origin}{action}的车辆"
    if interaction_kind == "cut_in":
        return f"在{place}{motion}{lane}时，没有避让从{origin}{action}进入自车车道的车辆"
    return f"在{place}{motion}{lane}时，没有对从{origin}{action}过来的车辆让行"
