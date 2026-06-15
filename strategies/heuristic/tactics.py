import numpy as np

from pyquaticus.base_policies.utils import (
    get_avoid_vect,
    global_rect_to_abs_bearing,
    rel_bearing_to_local_unit_rect,
)
from pyquaticus.utils.utils import angle180

CARRY_GOAL_WEIGHT = 1.5
CARRY_AVOID_THRESHOLD = 55.0
SAFE_CARRY_GOAL_WEIGHT = 2.0

def nearest_capture_corner(capture_corners, agent_position):
    distances = np.linalg.norm(capture_corners - agent_position, axis=1)
    return capture_corners[np.argmin(distances)]

def untagged_enemy_obstacles(
    opponent_ids, opponent_positions, global_state, only_tagging_capable=False
):
    obstacles = []

    for opponent_id, position in zip(opponent_ids, opponent_positions):
        if global_state[(opponent_id, "is_tagged")]:
            continue
        if only_tagging_capable and not global_state[(opponent_id, "on_side")]:
            continue

        obstacles.append((float(position[0]), float(position[1])))

    return obstacles

def carry_home_vector(agent, global_state, capture_corners, threat_obstacles):
    agent_position = global_state[(agent.id, "pos")]
    agent_heading = global_state[(agent.id, "heading")]
    is_on_own_side = global_state[(agent.id, "on_side")]

    corner = nearest_capture_corner(capture_corners, agent_position)
    bearing_to_corner = angle180(
        global_rect_to_abs_bearing(corner - agent_position) - agent_heading
    )
    goal_vector = rel_bearing_to_local_unit_rect(bearing_to_corner)

    if is_on_own_side:
        return SAFE_CARRY_GOAL_WEIGHT * goal_vector

    avoidance_vector = get_avoid_vect(threat_obstacles, avoid_threshold=CARRY_AVOID_THRESHOLD)

    return CARRY_GOAL_WEIGHT * goal_vector + avoidance_vector

def count_invaders(opponent_ids, global_state):
    invaders = 0

    for opponent_id in opponent_ids:
        if global_state[(opponent_id, "is_tagged")]:
            continue
        if not global_state[(opponent_id, "on_side")]:
            invaders += 1

    return invaders
