from typing import Union

import numpy as np

from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.utils import (
    dist_rel_bearing_to_local_rect,
    get_avoid_vect,
    global_rect_to_abs_bearing,
    rel_bearing_to_local_unit_rect,
)
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.utils import angle180


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

        if self.team == Team.BLUE_TEAM:
            self._capture_targets = np.array(env.blue_untag_coords, dtype=float)
        else:
            self._capture_targets = np.array(env.red_untag_coords, dtype=float)

    def compute_action(self, obs, info):
        self.update_state(obs, info)

        if self.mode != "hard":
            return super().compute_action(obs, info)

        wall_obstacles = self._wall_obstacles()
        obstacles = list(self.opp_team_pos) + wall_obstacles

        if self.has_flag:
            return self._carry_to_capture_zone(info, obstacles)

        if self.my_team_has_flag:
            goal = rel_bearing_to_local_unit_rect(self.opp_flag_bearing)
            avoid = get_avoid_vect(obstacles, avoid_threshold=self._ATTACK_AVOID_THRESH)
            return self.action_from_vector(1.25 * goal + avoid, 1)

        goal = rel_bearing_to_local_unit_rect(self.opp_flag_bearing)
        avoid = get_avoid_vect(obstacles, avoid_threshold=self._ATTACK_AVOID_THRESH)

        if not np.any(goal + avoid) or self._nearly_cancelled(goal, avoid):
            return self.action_from_vector(self._boundary_escape(), 1)

        return self.action_from_vector(1.25 * goal + avoid, 1)

    def _carry_to_capture_zone(self, info, obstacles):
        """Navega para o ponto de captura mais próximo em vez de flag_home."""
        global_state = info[self.id]["global_state"]
        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        my_pos = global_state[(self.id, "pos")]
        my_heading = global_state[(self.id, "heading")]

        dists = np.linalg.norm(self._capture_targets - my_pos, axis=1)
        nearest = self._capture_targets[np.argmin(dists)]

        bearing = angle180(global_rect_to_abs_bearing(nearest - my_pos) - my_heading)

        goal = 1.5 * rel_bearing_to_local_unit_rect(bearing)
        avoid = get_avoid_vect(obstacles, avoid_threshold=self._CARRY_AVOID_THRESH)
        return self.action_from_vector(goal + avoid, 1)

    def _wall_obstacles(self):
        obstacles = []
        for a, b in [(0, 2), (1, 3)]:
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
