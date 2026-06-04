from typing import Union

import numpy as np

from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.utils import (
    get_avoid_vect,
    global_rect_to_abs_bearing,
    rel_bearing_to_local_unit_rect,
)
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.utils import angle180


_DEFEND_THREAT_THRESHOLD = 2
_CARRY_AVOID_THRESH = 45.0

class HybridAgent(Heuristic_CTF_Agent):

    def __init__(
        self,
        agent_id: str,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        continuous: bool = False,
        mode: str = "hard",
        defensiveness: float = 20.0,
    ):
        super().__init__(
            agent_id,
            env,
            continuous=continuous,
            mode=mode,
            defensiveness=defensiveness,
        )

        if self.team == Team.BLUE_TEAM:
            self._capture_targets = np.array(env.blue_untag_coords, dtype=float)
        else:
            self._capture_targets = np.array(env.red_untag_coords, dtype=float)

    def compute_action(self, obs, info):
        self.update_state(obs, info)

        if self.mode in ("nothing", "easy"):
            return super().compute_action(obs, info)

        global_state = info[self.id]["global_state"]

        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        if self.has_flag:
            return self._carry_to_capture_zone(global_state)

        role = self._choose_role(global_state)

        if role == "attack":
            return self.base_attacker.compute_action(obs, info)
        else:
            return self.base_defender.compute_action(obs, info)

    def _carry_to_capture_zone(self, global_state):
        my_pos = global_state[(self.id, "pos")]
        my_heading = global_state[(self.id, "heading")]

        dists = np.linalg.norm(self._capture_targets - my_pos, axis=1)
        nearest = self._capture_targets[np.argmin(dists)]

        bearing = angle180(global_rect_to_abs_bearing(nearest - my_pos) - my_heading)

        goal = 1.5 * rel_bearing_to_local_unit_rect(bearing)
        avoid = get_avoid_vect(self.opp_team_pos, avoid_threshold=_CARRY_AVOID_THRESH)
        return self.action_from_vector(goal + avoid, 1)

    def _choose_role(self, global_state) -> str:
        team_has_flag = any(
            global_state.get((tid, "has_flag"), False)
            for tid in self.teammate_ids
            if tid != self.id
        )
        if team_has_flag:
            return "attack"

        threats = self._count_threats(global_state)

        if threats >= _DEFEND_THREAT_THRESHOLD:
            return "defend"

        if self.opp_team_has_flag and threats >= 1:
            return "defend"

        return "attack"

    def _count_threats(self, global_state) -> int:
        count = 0

        for enemy_id in self.opponent_ids:
            tagged = global_state.get((enemy_id, "is_tagged"), True)
            on_own_side = global_state.get((enemy_id, "on_side"), 1)

            if not tagged and not on_own_side:
                count += 1

        return count
