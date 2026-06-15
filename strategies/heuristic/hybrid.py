from typing import Union

from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge

from .attacker import Attacker
from .tactics import count_invaders

INVADERS_TO_DEFEND = 2

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
            agent_id, env, continuous=continuous, mode=mode, defensiveness=defensiveness
        )
        self.offense = Attacker(agent_id, env, continuous=continuous, mode=mode)

    def compute_action(self, obs, info):
        self.update_state(obs, info)

        if self.mode in ("nothing", "easy"):
            return super().compute_action(obs, info)

        global_state = self._read_global_state(info)

        if not self.has_flag and self._should_defend(global_state):
            return self.base_defender.compute_action(obs, info)

        return self.offense.compute_action(obs, info)

    def _read_global_state(self, info):
        global_state = info[self.id]["global_state"]

        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        return global_state

    def _should_defend(self, global_state):
        if self.opp_team_has_flag:
            return True

        return count_invaders(self.opponent_ids, global_state) >= INVADERS_TO_DEFEND
