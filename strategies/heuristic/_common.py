import numpy as np

from pyquaticus.base_policies.utils import (
    get_avoid_vect,
    global_rect_to_abs_bearing,
    rel_bearing_to_local_unit_rect,
)
from pyquaticus.utils.utils import angle180

_CARRY_GOAL_WEIGHT = 1.5
_CARRY_AVOID_THRESH = 55.0

def nearest_capture_corner(capture_targets: np.ndarray, my_pos: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(capture_targets - my_pos, axis=1)

    return capture_targets[np.argmin(dists)]

def untagged_enemy_obstacles(
    opponent_ids,
    opp_team_pos,
    global_state,
    only_tagging_capable: bool = False,
):
    obstacles = []

    for enemy_id, pos in zip(opponent_ids, opp_team_pos):
        if global_state[(enemy_id, "is_tagged")]:
            continue

        if only_tagging_capable and not global_state[(enemy_id, "on_side")]:
            continue

        obstacles.append((float(pos[0]), float(pos[1])))

    return obstacles

def carry_home_vector(
    agent,
    global_state,
    capture_targets: np.ndarray,
    threat_obstacles,
) -> np.ndarray:
    agent_id = agent.id
    my_pos = global_state[(agent_id, "pos")]
    my_heading = global_state[(agent_id, "heading")]
    on_side = global_state[(agent_id, "on_side")]

    nearest = nearest_capture_corner(capture_targets, my_pos)
    bearing = angle180(global_rect_to_abs_bearing(nearest - my_pos) - my_heading)
    goal = rel_bearing_to_local_unit_rect(bearing)

    if on_side:
        return 2.0 * goal

    avoid = get_avoid_vect(threat_obstacles, avoid_threshold=_CARRY_AVOID_THRESH)

    return _CARRY_GOAL_WEIGHT * goal + avoid

def count_invaders(opponent_ids, global_state) -> int:
    count = 0

    for enemy_id in opponent_ids:
        if global_state[(enemy_id, "is_tagged")]:
            continue

        if not global_state[(enemy_id, "on_side")]:
            count += 1

    return count
