from typing import Union

from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge

_DEFEND_THREAT_THRESHOLD = 2

class HybridAgent(Heuristic_CTF_Agent):
    """
    Agente híbrido com papéis dinâmicos.

    A cada passo, avalia o estado global para decidir entre atacar e defender:

    - Carregando a bandeira → sempre ataca (volta para casa).
    - Companheiro de equipe carrega a bandeira → ataca (pressão adicional).
    - ≥2 inimigos não-tagueados no nosso lado → defende.
    - Inimigo carrega nossa bandeira E há ameaça no nosso lado → defende.
    - Caso contrário → ataca.

    O papel é calculado a cada step; não há memória de papel entre steps.
    """

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

    def compute_action(self, obs, info):
        self.update_state(obs, info)

        if self.mode in ("nothing", "easy"):
            return super().compute_action(obs, info)

        global_state = info[self.id]["global_state"]

        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        role = self._choose_role(global_state)

        if role == "attack":
            return self.base_attacker.compute_action(obs, info)
        else:
            return self.base_defender.compute_action(obs, info)

    def _choose_role(self, global_state) -> str:
        if self.has_flag:
            return "attack"

        team_has_flag = any(
            global_state.get((tid, "has_flag"), False)
            for tid in self.teammate_ids
            if tid != self.id
        )

        if team_has_flag:
            return "attack"

        threats_on_our_side = self._count_threats(global_state)

        if threats_on_our_side >= _DEFEND_THREAT_THRESHOLD:
            return "defend"

        if self.opp_team_has_flag and threats_on_our_side >= 1:
            return "defend"

        return "attack"

    def _count_threats(self, global_state) -> int:
        """Conta inimigos não-tagueados que cruzaram para o nosso lado do campo."""
        count = 0

        for enemy_id in self.opponent_ids:
            tagged = global_state.get((enemy_id, "is_tagged"), True)
            on_own_side = global_state.get((enemy_id, "on_side"), 1)

            if not tagged and not on_own_side:
                count += 1

        return count
