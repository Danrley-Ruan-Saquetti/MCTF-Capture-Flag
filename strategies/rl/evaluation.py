"""
Avaliação e inferência das políticas RL treinadas.

Fornece:
* atores azuis que transformam observações em ações, a partir de um algoritmo
  RLlib vivo (durante o treino) ou de um checkpoint salvo (uso offline);
* ``evaluate``, que mede o desempenho do time azul contra oponentes heurísticos,
  um snapshot treinado ou um time parado, em episódios determinísticos.

Um "ator" aqui é simplesmente uma função ``observações -> {agente: ação}``.
"""

import os

import numpy as np

from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv
from pyquaticus.mctf26_config import get_std_config

from strategies.rl.obs_id_wrapper import AgentIDObsWrapper

BLUE_AGENT_IDS = ("agent_0", "agent_1", "agent_2")
RED_AGENT_IDS = ("agent_3", "agent_4", "agent_5")
NOOP_ACTION = 16  # índice de "parado" no espaço de ações Discrete(17)
RED_ROLES = {"agent_3": "attack", "agent_4": "defend", "agent_5": "attack"}

def build_evaluation_env(experiment_config):
    """Cria o ambiente de avaliação (com one-hot de papel, se o treino usou)."""
    base_env = CompPyquaticusEnv(
        config_dict=get_std_config(), render_mode=None, reward_config={}
    )

    if experiment_config.append_agent_id:
        return AgentIDObsWrapper(base_env)

    return base_env

def make_blue_actor_from_algo(algorithm, experiment_config, blue_policy_id=None):
    """Ator azul a partir de um algoritmo RLlib vivo (usado no gating do treino)."""
    policies = {
        agent_id: algorithm.get_policy(_blue_policy_id(experiment_config, team_index))
        for team_index, agent_id in enumerate(BLUE_AGENT_IDS)
    }

    return _greedy_actor(policies)

def make_blue_actor_from_checkpoint(checkpoint_path, experiment_config):
    """Ator azul a partir de um checkpoint salvo (uso offline, ex.: main.py)."""
    load = _checkpoint_policy_loader(checkpoint_path)
    policies = {
        agent_id: load(_blue_policy_id(experiment_config, team_index))
        for team_index, agent_id in enumerate(BLUE_AGENT_IDS)
    }

    return _greedy_actor(policies)

def make_red_actor(opponent, env, experiment_config, opponent_checkpoint=None):
    """Cria o ator do time vermelho conforme o oponente escolhido."""
    if opponent == "noop":
        return lambda observations, info: {agent: NOOP_ACTION for agent in RED_AGENT_IDS}

    if opponent in ("checkpoint", "snapshot", "self"):
        return _checkpoint_red_actor(opponent_checkpoint, experiment_config)

    return _heuristic_red_actor(env)

def evaluate(blue_actor, opponent, episodes, seed, exp, opp_checkpoint=None):
    """Roda ``episodes`` partidas e agrega métricas do time azul.

    Retorna win-rate, diferencial médio de capturas, colisões e OOB médios.
    """
    env = build_evaluation_env(exp)
    red_actor = make_red_actor(opponent, env, exp, opp_checkpoint)
    blue_indices = [env.players[agent_id].idx for agent_id in BLUE_AGENT_IDS]

    wins = draws = losses = 0
    capture_differences, blue_collisions, blue_out_of_bounds = [], [], []

    for episode in range(episodes):
        observations, info = env.reset(seed=seed + episode)

        while True:
            actions = {**blue_actor(observations), **red_actor(observations, info)}
            observations, _, terminated, truncated, info = env.step(actions)
            any_agent = next(iter(terminated))

            if terminated[any_agent] or truncated[any_agent]:
                break

        state = env.state
        blue_captures = int(state["captures"][0])
        red_captures = int(state["captures"][1])
        capture_differences.append(blue_captures - red_captures)
        collisions = np.asarray(state["agent_collisions"])
        blue_collisions.append(float(collisions[blue_indices].sum()))
        blue_out_of_bounds.append(float(np.asarray(state["agent_oob"])[blue_indices].sum()))

        if blue_captures > red_captures:
            wins += 1
        elif red_captures > blue_captures:
            losses += 1
        else:
            draws += 1

    env.close()
    episode_count = max(episodes, 1)

    return {
        "episodes": episodes,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / episode_count,
        "cap_diff_mean": float(np.mean(capture_differences)) if capture_differences else 0.0,
        "collisions_mean": float(np.mean(blue_collisions)) if blue_collisions else 0.0,
        "oob_mean": float(np.mean(blue_out_of_bounds)) if blue_out_of_bounds else 0.0,
    }

# --- helpers internos ---------------------------------------------------------

def _blue_policy_id(experiment_config, team_index):
    """Id da policy azul para um índice de papel (compartilhada ou independente)."""
    if experiment_config.shared_policy:
        return experiment_config.blue_shared_id

    return experiment_config.blue_independent_ids[team_index]

def _greedy_actor(policies_by_agent):
    """Ator que escolhe a ação de maior valor de cada policy (sem exploração)."""
    def actor(observations):
        return {
            agent_id: int(policy.compute_single_action(observations[agent_id], explore=False)[0])
            for agent_id, policy in policies_by_agent.items()
        }

    return actor

def _checkpoint_policy_loader(checkpoint_path):
    """Carregador de policies do checkpoint, com cache por id."""
    from ray.rllib.policy.policy import Policy

    cache = {}

    def load(policy_id):
        if policy_id not in cache:
            # pyarrow exige caminho ABSOLUTO (path relativo falha: "empty scheme").
            policy_dir = os.path.abspath(os.path.join(checkpoint_path, "policies", policy_id))
            cache[policy_id] = Policy.from_checkpoint(policy_dir)

        return cache[policy_id]

    return load

def _heuristic_red_actor(env):
    """Time vermelho heurístico competition_easy (2 atacantes + 1 defensor)."""
    base_policies = {}

    for agent_id, role in RED_ROLES.items():
        if role == "defend":
            base_policies[agent_id] = BaseDefender(agent_id, env, mode="competition_easy")
        else:
            base_policies[agent_id] = BaseAttacker(agent_id, env, mode="competition_easy")

    def actor(observations, info):
        return {
            agent_id: policy.compute_action(observations[agent_id], info)
            for agent_id, policy in base_policies.items()
        }

    return actor

def _checkpoint_red_actor(checkpoint_path, experiment_config):
    """Time vermelho controlado por uma política treinada (self-play)."""
    load = _checkpoint_policy_loader(checkpoint_path)
    policy = load(experiment_config.blue_shared_id)

    def actor(observations, info):
        return {
            agent_id: int(policy.compute_single_action(observations[agent_id], explore=False)[0])
            for agent_id in RED_AGENT_IDS
        }

    return actor
