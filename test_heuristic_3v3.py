import argparse
import sys
import os
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pyquaticus"))

from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv
from pyquaticus.mctf26_config import get_std_config

from strategies.heuristic import Attacker, Defender, HybridAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Teste heurístico 3v3 MCTF")
    parser.add_argument("--render", action="store_true", help="Ativa visualização pygame")
    parser.add_argument("--episodes", type=int, default=1, help="Número de episódios")
    parser.add_argument("--speedup", type=int, default=None,
                        help="Fator de aceleração da simulação (substitui o padrão do config)")
    parser.add_argument("--opponent", choices=["easy", "medium", "hard"], default="hard",
                        help="Dificuldade do time oponente heurístico")

    return parser.parse_args()

def make_env(render: bool, speedup: int | None) -> CompPyquaticusEnv:
    config = get_std_config()

    if speedup is not None:
        config["sim_speedup_factor"] = speedup

    config["render_agent_ids"] = True
    render_mode = "human" if render else None

    return CompPyquaticusEnv(config_dict=config, render_mode=render_mode, reward_config={})

def make_blue_team(env: CompPyquaticusEnv):
    return {
        "agent_0": Attacker("agent_0", env, continuous=False, mode="hard"),
        "agent_1": HybridAgent("agent_1", env, continuous=False, mode="hard"),
        "agent_2": Defender("agent_2", env, continuous=False, mode="hard"),
    }

def make_red_team(env: CompPyquaticusEnv, mode: str):
    return {
        "agent_3": Heuristic_CTF_Agent("agent_3", env, continuous=False, mode=mode),
        "agent_4": Heuristic_CTF_Agent("agent_4", env, continuous=False, mode=mode),
        "agent_5": Heuristic_CTF_Agent("agent_5", env, continuous=False, mode=mode),
    }

def print_results(episode: int, state: dict, blue_score: int, red_score: int):
    captures = state["captures"]
    tags = state["tags"]
    grabs = state["grabs"]
    print(
        f"[Ep {episode:02d}] "
        f"Placar: Azul {captures[0]} x {captures[1]} Vermelho | "
        f"Grabs: A={grabs[0]} V={grabs[1]} | "
        f"Tags: A={tags[0]} V={tags[1]}"
    )

def run_episode(
    env: CompPyquaticusEnv,
    blue_agents: dict,
    red_agents: dict,
    episode: int,
) -> tuple[int, int]:
    obs, info = env.reset()

    all_agents = {**blue_agents, **red_agents}
    terminated = {agent_id: False for agent_id in all_agents}
    truncated = {agent_id: False for agent_id in all_agents}

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (
                event.type == KEYDOWN and event.key == K_ESCAPE
            ):
                env.close()
                sys.exit()

        actions = {}

        for agent_id, agent in all_agents.items():
            actions[agent_id] = agent.compute_action(obs, info)

        obs, _, terminated, truncated, info = env.step(actions)

        first_agent = next(iter(terminated))

        if terminated[first_agent] or truncated[first_agent]:
            break

    captures = env.state["captures"]
    print_results(episode, env.state, captures[0], captures[1])

    return int(captures[0]), int(captures[1])

def main():
    args = parse_args()

    env = make_env(args.render, args.speedup)

    env.reset()

    blue_agents = make_blue_team(env)
    red_agents = make_red_team(env, args.opponent)

    total_blue, total_red = 0, 0

    print(f"\n=== MCTF Heuristic 3v3 | {args.episodes} episódio(s) | oponente: {args.opponent} ===\n")

    for ep in range(1, args.episodes + 1):
        b, r = run_episode(env, blue_agents, red_agents, ep)
        total_blue += b
        total_red += r

    env.close()

    print(f"\n{'='*55}")
    print(f"  Resultado final ({args.episodes} ep.): Azul {total_blue} x {total_red} Vermelho")

    if total_blue > total_red:
        print("  Resultado: AZUL VENCEU")
    elif total_red > total_blue:
        print("  Resultado: VERMELHO VENCEU")
    else:
        print("  Resultado: EMPATE")

    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
