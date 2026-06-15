"""
Wrapper que anexa um one-hot do "papel" (índice do agente dentro do seu time) ao
vetor de observação de cada agente.

Por quê: usamos UMA política compartilhada para os três agentes azuis (eficiência
amostral 3x e coordenação emergente). Mas, com pesos compartilhados, o agente
precisa de um sinal para diferenciar seu papel (atacante/defensor/etc.) — caso
contrário a política fica simétrica e não especializa. O one-hot de 3 dims
(agent_0/3 -> [1,0,0], agent_1/4 -> [0,1,0], agent_2/5 -> [0,0,1]) resolve isso.

O índice é RELATIVO AO TIME, então um snapshot azul consegue controlar os
vermelhos no self-play vendo os mesmos índices de papel (o ambiente é simétrico).

Só o vetor `obs` é alterado; o `info` (com `global_state`/`unnorm_obs`) é
preservado intacto — as políticas heurísticas oponentes leem o `info`, não o obs.
"""

import numpy as np
from gymnasium import spaces

TEAM_SIZE = 3

class AgentIDObsWrapper:
    """Delega tudo ao env PettingZoo base, anexando o one-hot de papel ao obs."""

    def __init__(self, env):
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = list(env.possible_agents)

        # one-hot por agente, fixo (índice = sufixo numérico % TEAM_SIZE).
        self._onehot = {}
        for aid in self.possible_agents:
            role = int(str(aid).split("_")[-1]) % TEAM_SIZE
            vec = np.zeros(TEAM_SIZE, dtype=np.float32)
            vec[role] = 1.0
            self._onehot[aid] = vec

        # espaço de observação estendido (+TEAM_SIZE dims em [0,1]).
        self._obs_spaces = {
            aid: self._extend_space(env.observation_space(aid))
            for aid in self.possible_agents
        }

    # -- API PettingZoo usada pelo ParallelPettingZooEnv do RLlib --

    @property
    def agents(self):
        return self.env.agents

    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        return self._augment(obs), rew, term, trunc, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    # NB: não definir um método `state` aqui — `CompPyquaticusEnv.state` é um
    # ATRIBUTO (dict), acessado via __getattr__. Um método sombrearia o dict.

    # -- internos --

    def _extend_space(self, space: spaces.Box) -> spaces.Box:
        low = np.concatenate([space.low, np.zeros(TEAM_SIZE, dtype=space.dtype)])
        high = np.concatenate([space.high, np.ones(TEAM_SIZE, dtype=space.dtype)])
        return spaces.Box(low=low, high=high, dtype=space.dtype)

    def _augment(self, obs_dict):
        out = {}
        for aid, o in obs_dict.items():
            out[aid] = np.concatenate(
                [np.asarray(o, dtype=np.float32), self._onehot[aid]]
            )
        return out

    def __getattr__(self, name):
        # Delega atributos não definidos (agent_obs_normalizer, players,
        # agent_ids_of_team, _walls, etc.) para o env base.
        return getattr(self.env, name)
