import random
from typing import Dict

import numpy as np
from ray.rllib.policy.policy import Policy

from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_policy_wrappers import NoOp
from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv
from pyquaticus.mctf26_config import get_std_config

_SEED = 42
_PHASE_DIST: Dict[str, float] = {"easy": 1.0}
_ACTIVE_SNAPS = []
_EPISODE_CHOICE: Dict[object, object] = {}
_FALLBACK_COUNTER = [0]

def set_league_state(phase_dist: Dict[str, float], active_snaps):
    global _PHASE_DIST, _ACTIVE_SNAPS

    _PHASE_DIST = dict(phase_dist)
    _ACTIVE_SNAPS = list(active_snaps)
    _EPISODE_CHOICE.clear()

def _team_suffix(agent_id: str) -> int:
    return int(str(agent_id).split("_")[-1])

def _sample_kind(rng: random.Random) -> str:
    kinds, probs = zip(*_PHASE_DIST.items())

    if "snap" in kinds and not _ACTIVE_SNAPS:
        pairs = [(k, p) for k, p in zip(kinds, probs) if k != "snap"]

        if not pairs:
            return "easy"

        kinds, probs = zip(*pairs)

    total = float(sum(probs))
    r = rng.random() * total
    acc = 0.0

    for k, p in zip(kinds, probs):
        acc += p
        if r <= acc:
            return k

    return kinds[-1]

def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    suffix = _team_suffix(agent_id)

    if suffix < 3:
        return _BLUE_IDS[suffix] if len(_BLUE_IDS) == 3 else _BLUE_IDS[0]

    ep_id = getattr(episode, "episode_id", None)
    if ep_id is None:
        ep_id = ("fallback", _FALLBACK_COUNTER[0])

    if ep_id not in _EPISODE_CHOICE:
        rng = random.Random(hash(ep_id) ^ _SEED)
        kind = _sample_kind(rng)
        snap_id = rng.choice(_ACTIVE_SNAPS) if (kind == "snap" and _ACTIVE_SNAPS) else None
        _EPISODE_CHOICE[ep_id] = (kind, snap_id)

        if isinstance(ep_id, tuple):
            _FALLBACK_COUNTER[0] += 1

    kind, snap_id = _EPISODE_CHOICE[ep_id]

    if kind == "noop":
        return "opp_noop"
    if kind == "snap" and snap_id is not None:
        return snap_id
    return f"opp_easy_{suffix}"

_BLUE_IDS = ["blue-shared", "blue-shared", "blue-shared"]

def set_blue_ids(blue_ids):
    global _BLUE_IDS

    if len(blue_ids) == 1:
        _BLUE_IDS = [blue_ids[0]] * 3
    else:
        _BLUE_IDS = list(blue_ids)

_BASE_ENV_CACHE = [None]

def _get_base_env():
    if _BASE_ENV_CACHE[0] is None:
        _BASE_ENV_CACHE[0] = CompPyquaticusEnv(
            config_dict=get_std_config(), render_mode=None, reward_config={}
        )
    return _BASE_ENV_CACHE[0]

class HeuristicOpponentPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.agent_id = config["agent_id"]
        role = config.get("role", "combined")
        mode = config.get("mode", "competition_easy")
        env = _get_base_env()
        if role == "attack":
            self.policy = BaseAttacker(self.agent_id, env, mode=mode)
        elif role == "defend":
            self.policy = BaseDefender(self.agent_id, env, mode=mode)
        else:
            self.policy = Heuristic_CTF_Agent(self.agent_id, env, mode=mode)

    def compute_actions(
        self, obs_batch, state_batches=None, prev_action_batch=None,
        prev_reward_batch=None, info_batch=None, episodes=None,
        explore=None, timestep=None, **kwargs,
    ):
        if not info_batch:
            # Sem info (ex.: inicialização do RLlib): fica parado.
            return [16 for _ in obs_batch], [], {}
        actions = []
        for i in range(len(obs_batch)):
            info_i = {k: v[i] for k, v in info_batch.items()}
            actions.append(self.policy.compute_action(obs_batch[i], info_i))
        return actions, [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass

class OpponentLeague:
    def __init__(self, exp_config, obs_space, act_space):
        self.cfg = exp_config
        self.obs_space = obs_space
        self.act_space = act_space
        self.n_slots = exp_config.league.n_snapshot_slots
        self.snap_ids = [f"snap_{i}" for i in range(self.n_slots)]
        self.active_snaps = []
        self._next_slot = 0

        if exp_config.shared_policy:
            self.blue_ids = [exp_config.blue_shared_id]
        else:
            self.blue_ids = list(exp_config.blue_independent_ids)
        set_blue_ids(self.blue_ids)

    def build_policies(self):
        o, a = self.obs_space, self.act_space
        policies = {}

        for bid in self.blue_ids:
            policies[bid] = (None, o, a, {})

        policies["opp_noop"] = (NoOp, o, a, {"no_checkpoint": True})
        role_map = {3: "attack", 4: "defend", 5: "attack"}
        for suffix in (3, 4, 5):
            policies[f"opp_easy_{suffix}"] = (
                HeuristicOpponentPolicy, o, a,
                {
                    "no_checkpoint": True,
                    "agent_id": f"agent_{suffix}",
                    "role": role_map[suffix],
                    "mode": self.cfg.league.heuristic_mode,
                },
            )

        for sid in self.snap_ids:
            policies[sid] = (None, o, a, {"no_checkpoint": True})

        return policies

    @property
    def policies_to_train(self):
        return list(self.blue_ids)

    def snapshot_into_next_slot(self, algo):
        slot_id = self.snap_ids[self._next_slot]
        self._next_slot = (self._next_slot + 1) % self.n_slots

        blue_weights = algo.get_policy(self.blue_ids[0]).get_weights()

        def _apply(runner):
            pol = runner.get_policy(slot_id)
            if pol is not None:
                pol.set_weights(blue_weights)

        try:
            algo.get_policy(slot_id).set_weights(blue_weights)
        except Exception:
            pass

        _broadcast(algo, _apply)

        if slot_id not in self.active_snaps:
            self.active_snaps.append(slot_id)
        return slot_id

    def set_phase(self, algo, phase):
        dist = dict(phase.opponent_dist)
        active = list(self.active_snaps)

        def _apply(_runner):
            set_league_state(dist, active)

        set_league_state(dist, active)
        _broadcast(algo, _apply)

    def sync(self, algo):
        dist = dict(_PHASE_DIST)
        active = list(self.active_snaps)

        def _apply(_runner):
            set_league_state(dist, active)

        set_league_state(dist, active)
        _broadcast(algo, _apply)

def _broadcast(algo, fn):
    group = getattr(algo, "env_runner_group", None) or getattr(algo, "workers", None)
    if group is None:
        return
    try:
        group.foreach_env_runner(fn)
    except AttributeError:
        group.foreach_worker(fn)
