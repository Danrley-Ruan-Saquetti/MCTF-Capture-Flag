from typing import Union

import numpy as np

from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.utils import (
    dist_rel_bearing_to_local_rect,
    get_avoid_vect,
    rel_bearing_to_local_unit_rect,
)
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge

class Attacker(BaseAttacker):

    _CARRY_AVOID_THRESH = 45.0
    _ATTACK_AVOID_THRESH = 35.0
    _WALL_PROXIMITY = 10.0

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

        wall_obstacles = self._wall_obstacles()
        obstacles = list(self.opp_team_pos) + wall_obstacles

        if self.has_flag:
            goal = 1.5 * rel_bearing_to_local_unit_rect(self.home_bearing)
            avoid = get_avoid_vect(obstacles, avoid_threshold=self._CARRY_AVOID_THRESH)

            return self.action_from_vector(goal + avoid, 1)

        if self.my_team_has_flag:
            goal = rel_bearing_to_local_unit_rect(self.opp_flag_bearing)
            avoid = get_avoid_vect(obstacles, avoid_threshold=self._ATTACK_AVOID_THRESH)

            return self.action_from_vector(1.25 * goal + avoid, 1)

        goal = rel_bearing_to_local_unit_rect(self.opp_flag_bearing)
        avoid = get_avoid_vect(obstacles, avoid_threshold=self._ATTACK_AVOID_THRESH)
        combined = goal + avoid

        if not np.any(combined) or self._nearly_cancelled(goal, avoid):
            combined = self._boundary_escape()

        return self.action_from_vector(1.25 * goal + avoid, 1)

    def _wall_obstacles(self):
        obstacles = []
        pairs = [(0, 2), (1, 3)]

        for a, b in pairs:
            if self.wall_distances[a] < self._WALL_PROXIMITY and -90 < self.wall_bearings[a] < 90:
                obstacles.append((self.wall_distances[a], self.wall_bearings[a]))
            elif self.wall_distances[b] < self._WALL_PROXIMITY and -90 < self.wall_bearings[b] < 90:
                obstacles.append((self.wall_distances[b], self.wall_bearings[b]))

        return obstacles

    def _nearly_cancelled(self, goal, avoid):
        return np.allclose(
            np.abs(np.abs(goal) - np.abs(avoid)),
            np.zeros(np.array(goal).shape),
            atol=1e-01,
            rtol=1e-02,
        )

    def _boundary_escape(self):
        top = self.wall_distances[0]
        bot = self.wall_distances[2]

        if top > 1.25 * bot:
            return dist_rel_bearing_to_local_rect(top, self.wall_bearings[0])

        return dist_rel_bearing_to_local_rect(bot, self.wall_bearings[2])
