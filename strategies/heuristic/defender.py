from typing import Union

import numpy as np

from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.utils import (
    dist_rel_bearing_to_local_rect,
    get_avoid_vect,
    rel_bearing_to_local_unit_rect,
    unit_vect_between_points,
)
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.utils import dist

_DEFENSE_PERIMETER_IN_METERS = 25.0
_WALL_PROXIMITY = 7.0

class Defender(BaseDefender):

    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        continuous: bool = False,
        mode: str = "hard",
    ):
        super().__init__(agent_id, env, continuous=continuous, mode=mode)

    def compute_action(self, obs, info):
        self.update_state(obs, info)

        if self.mode != "hard":
            return super().compute_action(obs, info)

        global_state = info[self.id]["global_state"]

        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        wall_obstacles = self._wall_obstacles()

        if self.opp_team_has_flag:
            ag_vect = rel_bearing_to_local_unit_rect(self.my_flag_bearing)
        else:
            ag_vect = self._guard_vector(global_state)

        if wall_obstacles:
            ag_vect = ag_vect + get_avoid_vect(wall_obstacles)

        return self.action_from_vector(ag_vect, 1)

    def _guard_vector(self, global_state):
        enemy_id, enemy_loc, on_our_side = self._closest_threat(global_state)

        if enemy_id is None:
            return rel_bearing_to_local_unit_rect(self.my_flag_bearing)

        if on_our_side:
            return enemy_loc

        enemy_dist_to_flag = dist(np.array(self.my_flag_loc), enemy_loc)

        if enemy_dist_to_flag < _DEFENSE_PERIMETER_IN_METERS * 0.5:
            return enemy_loc

        unit = unit_vect_between_points(np.array(self.my_flag_loc), enemy_loc)
        intercept_dist = min(enemy_dist_to_flag * 0.45, _DEFENSE_PERIMETER_IN_METERS)

        return np.array(self.my_flag_loc) + intercept_dist * unit

    def _closest_threat(self, global_state):
        best_id = None
        best_dist = float("inf")
        best_loc = np.zeros(2)
        best_on_our_side = False

        for enemy_id, pos in self.opp_team_pos_dict.items():
            if global_state.get((enemy_id, "is_tagged"), True):
                continue

            enemy_loc = dist_rel_bearing_to_local_rect(pos[0], pos[1])
            d = dist(np.array(self.my_flag_loc), enemy_loc)

            if d < best_dist:
                best_dist = d
                best_id = enemy_id
                best_loc = enemy_loc
                best_on_our_side = not global_state.get((enemy_id, "on_side"), True)

        if best_id is None and self.opp_team_pos_dict:
            best_id = min(self.opp_team_pos_dict, key=lambda k: self.opp_team_pos_dict[k][0])
            pos = self.opp_team_pos_dict[best_id]
            best_loc = dist_rel_bearing_to_local_rect(pos[0], pos[1])
            best_on_our_side = False

        return best_id, best_loc, best_on_our_side

    def _wall_obstacles(self):
        obstacles = []
        pairs = [(0, 2), (1, 3)]

        for a, b in pairs:
            if self.wall_distances[a] < _WALL_PROXIMITY and -90 < self.wall_bearings[a] < 90:
                obstacles.append((self.wall_distances[a], self.wall_bearings[a]))
            elif self.wall_distances[b] < _WALL_PROXIMITY and -90 < self.wall_bearings[b] < 90:
                obstacles.append((self.wall_distances[b], self.wall_bearings[b]))

        return obstacles
