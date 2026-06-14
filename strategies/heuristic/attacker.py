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

from ._common import carry_home_vector, untagged_enemy_obstacles

class Attacker(BaseAttacker):

    _ATTACK_AVOID_THRESH = 28.0
    _WALL_PROXIMITY = 10.0
    _COMMIT_RADIUS = 24.0
    _APPROACH_GOAL_WEIGHT = 1.5

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

        global_state = info[self.id]["global_state"]
        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        if self.has_flag:
            carry_threats = untagged_enemy_obstacles(
                self.opponent_ids, self.opp_team_pos, global_state
            )
            vector = carry_home_vector(
                self, global_state, self._capture_targets, carry_threats
            )

            return self.action_from_vector(vector, 1)

        threats = untagged_enemy_obstacles(
            self.opponent_ids,
            self.opp_team_pos,
            global_state,
            only_tagging_capable=True,
        )

        goal = rel_bearing_to_local_unit_rect(self.opp_flag_bearing)

        if self.opp_flag_distance < self._COMMIT_RADIUS:
            return self.action_from_vector(goal, 1)

        wall_obstacles = self._wall_obstacles()
        obstacles = threats + wall_obstacles

        avoid = get_avoid_vect(obstacles, avoid_threshold=self._ATTACK_AVOID_THRESH)

        if not np.any(goal + avoid) or self._nearly_cancelled(goal, avoid):
            return self.action_from_vector(self._boundary_escape(), 1)

        return self.action_from_vector(self._APPROACH_GOAL_WEIGHT * goal + avoid, 1)

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
