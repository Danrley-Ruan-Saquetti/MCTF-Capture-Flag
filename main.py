import argparse
import glob
import os
import sys

import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pyquaticus"))

from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv
from pyquaticus.mctf26_config import get_std_config

from strategies.heuristic import Attacker, Defender, HybridAgent

from strategies.rl.config_mctf import get_experiment_config
from strategies.rl.evaluation import make_blue_actor_from_checkpoint
from strategies.rl.obs_id_wrapper import AgentIDObsWrapper

BLUE_AGENT_IDS = ("agent_0", "agent_1", "agent_2")
RED_AGENT_IDS = ("agent_3", "agent_4", "agent_5")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulação MCTF 3v3")
    parser.add_argument("--agents", choices=["heuristic", "rl"], default="heuristic", help="tipo de agente do time azul")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint RL (default: o mais recente em ray_tests/)")
    parser.add_argument("--render", action="store_true", help="renderiza o jogo pygame")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--speedup", type=int, default=None, help="velocidade da simulação")
    parser.add_argument("--opponent", choices=["easy", "medium", "hard"], default="hard", help="dificuldade oponente")
    return parser.parse_args()

def make_environment(arguments):
    config = get_std_config()

    if arguments.speedup is not None:
        config["sim_speedup_factor"] = arguments.speedup

    config["render_agent_ids"] = True

    return CompPyquaticusEnv(
        config_dict=config,
        render_mode="human" if arguments.render else None,
        reward_config={},
    )

def make_red_team(env, difficulty):
    return {
        agent_id: Heuristic_CTF_Agent(agent_id, env, continuous=True, mode=difficulty)
        for agent_id in RED_AGENT_IDS
    }

def make_heuristic_blue_team(env):
    return {
        "agent_0": Attacker("agent_0", env, continuous=True, mode="hard"),
        "agent_1": HybridAgent("agent_1", env, continuous=True, mode="hard"),
        "agent_2": Defender("agent_2", env, continuous=True, mode="hard"),
    }

def latest_checkpoint(directory="ray_tests"):
    checkpoints = glob.glob(os.path.join(directory, "iter_*"))

    if not checkpoints:
        return None

    return max(checkpoints, key=lambda path: int(path.rsplit("_", 1)[-1]))

def build_heuristic_simulation(arguments):
    env = make_environment(arguments)
    heuristic_agents = {**make_heuristic_blue_team(env), **make_red_team(env, arguments.opponent)}

    return env, None, heuristic_agents

def build_rl_simulation(arguments):
    checkpoint = arguments.checkpoint or latest_checkpoint()

    if checkpoint is None or not os.path.isdir(checkpoint):
        sys.exit("Nenhum checkpoint RL encontrado. Treine antes ou passe --checkpoint.")

    experiment_config = get_experiment_config()
    base_env = make_environment(arguments)

    env = AgentIDObsWrapper(base_env) if experiment_config.append_agent_id else base_env

    print(f"Carregando agentes RL de: {checkpoint}")
    blue_actor = make_blue_actor_from_checkpoint(checkpoint, experiment_config)

    return env, blue_actor, make_red_team(env, arguments.opponent)

def handle_quit_events():
    if not pygame.display.get_init():
        return

    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            sys.exit()

def run_episode(env, blue_actor, heuristic_agents, render):
    observations, info = env.reset()

    while True:
        if render:
            handle_quit_events()

        actions = {}

        if blue_actor is not None:
            actions.update(blue_actor(observations))
        for agent_id, agent in heuristic_agents.items():
            actions[agent_id] = agent.compute_action(observations, info)

        observations, _, terminated, truncated, info = env.step(actions)
        any_agent = next(iter(terminated))

        if terminated[any_agent] or truncated[any_agent]:
            break

    return env.state

def print_episode_result(episode, state):
    captures, grabs, tags = state["captures"], state["grabs"], state["tags"]

    print(f"[Ep {episode:02d}] Placar: Azul {int(captures[0])} x {int(captures[1])} Vermelho | "
          f"Grabs A{int(grabs[0])} V{int(grabs[1])} | Tags A{int(tags[0])} V{int(tags[1])}")

def print_final_result(episodes, blue_total, red_total):
    print(f"\n{'=' * 55}")
    print(f"  Resultado final ({episodes} ep.): Azul {blue_total} x {red_total} Vermelho")

    if blue_total > red_total:
        print("  Resultado: AZUL VENCEU")
    elif red_total > blue_total:
        print("  Resultado: VERMELHO VENCEU")
    else:
        print("  Resultado: EMPATE")

    print(f"{'=' * 55}\n")

def main():
    arguments = parse_arguments()
    build_simulation = build_rl_simulation if arguments.agents == "rl" else build_heuristic_simulation
    env, blue_actor, heuristic_agents = build_simulation(arguments)

    print(f"\n=== MCTF 3v3 | azul: {arguments.agents} | vermelho: {arguments.opponent} | "
          f"{arguments.episodes} episódio(s) ===\n")

    blue_total = red_total = 0

    for episode in range(1, arguments.episodes + 1):
        state = run_episode(env, blue_actor, heuristic_agents, arguments.render)
        blue_total += int(state["captures"][0])
        red_total += int(state["captures"][1])

        print_episode_result(episode, state)

    env.close()
    print_final_result(arguments.episodes, blue_total, red_total)

if __name__ == "__main__":
    main()
