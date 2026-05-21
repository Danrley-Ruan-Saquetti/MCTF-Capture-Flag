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
    """
    Agente dedicado à defesa da bandeira.

    Comportamentos prioritários (modo hard):
    1. Inimigo tem nossa bandeira → intercepta na rota de fuga.
    2. Inimigo está perto da bandeira → corre direto para ele.
    3. Inimigo está no perímetro defensivo → posiciona-se entre o inimigo e a bandeira.
    4. Sem ameaça imediata → patrulha entre a bandeira e o meio-campo.
    """

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

        closest_enemy, enemy_loc = self._closest_threat(global_state)

        if self.opp_team_has_flag:
            ag_vect = rel_bearing_to_local_unit_rect(self.my_flag_bearing)

        elif closest_enemy is not None:
            enemy_dist_to_flag = dist(np.array(self.my_flag_loc), enemy_loc)

            if enemy_dist_to_flag < _DEFENSE_PERIMETER_IN_METERS * 0.5:
                ag_vect = enemy_loc
            else:
                unit = unit_vect_between_points(np.array(self.my_flag_loc), enemy_loc)
                intercept_dist = min(enemy_dist_to_flag * 0.45, _DEFENSE_PERIMETER_IN_METERS)
                ag_vect = np.array(self.my_flag_loc) + intercept_dist * unit

        else:
            ag_vect = rel_bearing_to_local_unit_rect(self.my_flag_bearing)

        if wall_obstacles:
            ag_vect = ag_vect + get_avoid_vect(wall_obstacles)

        return self.action_from_vector(ag_vect, 1)

    def _closest_threat(self, global_state):
        """Retorna (agent_id, local_rect_pos) do inimigo não-tagueado mais próximo da bandeira."""
        best_id = None
        best_dist = float("inf")
        best_loc = np.zeros(2)

        for enemy_id, pos in self.opp_team_pos_dict.items():
            if global_state.get((enemy_id, "is_tagged"), True):
                continue

            enemy_loc = dist_rel_bearing_to_local_rect(pos[0], pos[1])
            d = dist(np.array(self.my_flag_loc), enemy_loc)

            if d < best_dist:
                best_dist = d
                best_id = enemy_id
                best_loc = enemy_loc

        if best_id is None and self.opp_team_pos_dict:
            best_id = min(self.opp_team_pos_dict, key=lambda k: self.opp_team_pos_dict[k][0])
            pos = self.opp_team_pos_dict[best_id]
            best_loc = dist_rel_bearing_to_local_rect(pos[0], pos[1])

        return best_id, best_loc

    def _wall_obstacles(self):
        obstacles = []
        pairs = [(0, 2), (1, 3)]

        for a, b in pairs:
            if self.wall_distances[a] < _WALL_PROXIMITY and -90 < self.wall_bearings[a] < 90:
                obstacles.append((self.wall_distances[a], self.wall_bearings[a]))
            elif self.wall_distances[b] < _WALL_PROXIMITY and -90 < self.wall_bearings[b] < 90:
                obstacles.append((self.wall_distances[b], self.wall_bearings[b]))

        return obstacles
