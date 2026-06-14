from typing import Union

import numpy as np

from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge

from ._common import carry_home_vector, count_invaders, untagged_enemy_obstacles
from .attacker import Attacker

_INVADER_DEFEND_THRESHOLD = 2

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

        self._offense = Attacker(agent_id, env, continuous=continuous, mode=mode)

    def compute_action(self, obs, info):
        self.update_state(obs, info)

        if self.mode in ("nothing", "easy"):
            return super().compute_action(obs, info)

        global_state = info[self.id]["global_state"]
        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        if self.has_flag:
            threats = untagged_enemy_obstacles(
                self.opponent_ids, self.opp_team_pos, global_state
            )
            vector = carry_home_vector(
                self, global_state, self._capture_targets, threats
            )

            return self.action_from_vector(vector, 1)

        if self._should_defend(global_state):
            return self.base_defender.compute_action(obs, info)

        return self._offense.compute_action(obs, info)

    def _should_defend(self, global_state) -> bool:
        if self.opp_team_has_flag:
            return True

        if count_invaders(self.opponent_ids, global_state) >= _INVADER_DEFEND_THRESHOLD:
            return True

        return False
