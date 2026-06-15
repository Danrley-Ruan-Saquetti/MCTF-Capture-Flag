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

DEFENSE_PERIMETER = 25.0
WALL_PROXIMITY = 7.0

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

        global_state = self._read_global_state(info)

        if self.opp_team_has_flag:
            movement_vector = rel_bearing_to_local_unit_rect(self.my_flag_bearing)
        else:
            movement_vector = self._guard_vector(global_state)

        wall_obstacles = self._wall_obstacles()

        if wall_obstacles:
            movement_vector = movement_vector + get_avoid_vect(wall_obstacles)

        return self.action_from_vector(movement_vector, 1)

    def _read_global_state(self, info):
        """Estado global desnormalizado do ambiente."""
        global_state = info[self.id]["global_state"]

        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        return global_state

    def _guard_vector(self, global_state):
        threat_id, threat_location, threat_on_our_side = self._closest_threat(global_state)

        if threat_id is None:
            return rel_bearing_to_local_unit_rect(self.my_flag_bearing)

        if threat_on_our_side:
            return threat_location

        flag_location = np.array(self.my_flag_loc)
        enemy_distance_to_flag = dist(flag_location, threat_location)

        if enemy_distance_to_flag < DEFENSE_PERIMETER * 0.5:
            return threat_location

        direction_to_enemy = unit_vect_between_points(flag_location, threat_location)
        intercept_distance = min(enemy_distance_to_flag * 0.45, DEFENSE_PERIMETER)

        return flag_location + intercept_distance * direction_to_enemy

    def _closest_threat(self, global_state):

        closest_id = None
        closest_distance = float("inf")
        closest_location = np.zeros(2)
        closest_on_our_side = False
        flag_location = np.array(self.my_flag_loc)

        for enemy_id, polar_position in self.opp_team_pos_dict.items():
            if global_state.get((enemy_id, "is_tagged"), True):
                continue

            enemy_location = dist_rel_bearing_to_local_rect(polar_position[0], polar_position[1])
            distance_to_flag = dist(flag_location, enemy_location)

            if distance_to_flag < closest_distance:
                closest_distance = distance_to_flag
                closest_id = enemy_id
                closest_location = enemy_location
                closest_on_our_side = not global_state.get((enemy_id, "on_side"), True)

        if closest_id is None and self.opp_team_pos_dict:
            closest_id = min(
                self.opp_team_pos_dict, key=lambda enemy: self.opp_team_pos_dict[enemy][0]
            )
            polar_position = self.opp_team_pos_dict[closest_id]
            closest_location = dist_rel_bearing_to_local_rect(polar_position[0], polar_position[1])
            closest_on_our_side = False

        return closest_id, closest_location, closest_on_our_side

    def _wall_obstacles(self):
        obstacles = []

        for near_index, far_index in [(0, 2), (1, 3)]:
            for wall_index in (near_index, far_index):
                is_close = self.wall_distances[wall_index] < WALL_PROXIMITY
                is_ahead = -90 < self.wall_bearings[wall_index] < 90

                if is_close and is_ahead:
                    obstacles.append((self.wall_distances[wall_index], self.wall_bearings[wall_index]))
                    break

        return obstacles
