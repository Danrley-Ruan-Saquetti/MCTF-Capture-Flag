from typing import Union

import numpy as np

from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.utils import (
    dist_rel_bearing_to_local_rect,
    get_avoid_vect,
    rel_bearing_to_local_unit_rect,
)
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge

from .tactics import carry_home_vector, untagged_enemy_obstacles

class Attacker(BaseAttacker):

    COMMIT_RADIUS = 24.0
    WALL_PROXIMITY = 10.0
    APPROACH_AVOID_THRESHOLD = 28.0
    APPROACH_GOAL_WEIGHT = 1.5

    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        continuous: bool = False,
        mode: str = "hard",
    ):
        super().__init__(agent_id, env, continuous=continuous, mode=mode)
        self.capture_corners = self._read_capture_corners(env)

    def _read_capture_corners(self, env):
        if self.team == Team.BLUE_TEAM:
            return np.array(env.blue_untag_coords, dtype=float)

        return np.array(env.red_untag_coords, dtype=float)

    def compute_action(self, obs, info):
        self.update_state(obs, info)

        if self.mode != "hard":
            return super().compute_action(obs, info)

        global_state = self._read_global_state(info)

        if self.has_flag:
            return self._carry_action(global_state)

        return self._approach_action(global_state)

    def _read_global_state(self, info):
        global_state = info[self.id]["global_state"]

        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        return global_state

    def _carry_action(self, global_state):
        threats = untagged_enemy_obstacles(
            self.opponent_ids, self.opp_team_pos, global_state, only_tagging_capable=True
        )
        movement_vector = carry_home_vector(self, global_state, self.capture_corners, threats)

        wall_obstacles = self._wall_obstacles()
        if wall_obstacles:
            movement_vector = movement_vector + get_avoid_vect(wall_obstacles)

        return self.action_from_vector(movement_vector, 1)

    def _approach_action(self, global_state):
        goal_vector = rel_bearing_to_local_unit_rect(self.opp_flag_bearing)

        if self.opp_flag_distance < self.COMMIT_RADIUS:
            return self.action_from_vector(goal_vector, 1)

        taggers = untagged_enemy_obstacles(
            self.opponent_ids, self.opp_team_pos, global_state, only_tagging_capable=True
        )
        obstacles = taggers + self._wall_obstacles()
        avoidance_vector = get_avoid_vect(obstacles, avoid_threshold=self.APPROACH_AVOID_THRESHOLD)

        if self._vectors_cancel_out(goal_vector, avoidance_vector):
            return self.action_from_vector(self._boundary_escape(), 1)

        return self.action_from_vector(self.APPROACH_GOAL_WEIGHT * goal_vector + avoidance_vector, 1)

    def _wall_obstacles(self):
        obstacles = []

        for near_index, far_index in [(0, 2), (1, 3)]:
            for wall_index in (near_index, far_index):
                is_close = self.wall_distances[wall_index] < self.WALL_PROXIMITY
                is_ahead = -90 < self.wall_bearings[wall_index] < 90

                if is_close and is_ahead:
                    obstacles.append((self.wall_distances[wall_index], self.wall_bearings[wall_index]))
                    break

        return obstacles

    def _vectors_cancel_out(self, goal_vector, avoidance_vector):
        if not np.any(goal_vector + avoidance_vector):
            return True

        return np.allclose(
            np.abs(np.abs(goal_vector) - np.abs(avoidance_vector)),
            np.zeros(np.shape(goal_vector)),
            atol=1e-01,
            rtol=1e-02,
        )

    def _boundary_escape(self):
        top_distance = self.wall_distances[0]
        bottom_distance = self.wall_distances[2]

        if top_distance > 1.25 * bottom_distance:
            return dist_rel_bearing_to_local_rect(top_distance, self.wall_bearings[0])

        return dist_rel_bearing_to_local_rect(bottom_distance, self.wall_bearings[2])
