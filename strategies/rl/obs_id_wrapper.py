import numpy as np
from gymnasium import spaces

TEAM_SIZE = 3

class AgentIDObsWrapper:

    def __init__(self, env):
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.possible_agents = list(env.possible_agents)

        self._onehot = {}
        for aid in self.possible_agents:
            role = int(str(aid).split("_")[-1]) % TEAM_SIZE
            vec = np.zeros(TEAM_SIZE, dtype=np.float32)
            vec[role] = 1.0
            self._onehot[aid] = vec

        self._obs_spaces = {
            aid: self._extend_space(env.observation_space(aid))
            for aid in self.possible_agents
        }

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
        return getattr(self.env, name)
