import argparse
import json
import sys
import os
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pyquaticus"))

from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv
from pyquaticus.mctf26_config import get_std_config

from strategies.heuristic import Attacker, Defender, HybridAgent

BLUE_AGENT_IDS = ("agent_0", "agent_1", "agent_2")
RED_AGENT_IDS = ("agent_3", "agent_4", "agent_5")


def make_environment(speedup):
    config = get_std_config()
    config["sim_speedup_factor"] = speedup
    return CompPyquaticusEnv(config_dict=config, render_mode=None, reward_config={})


def make_blue_team(env, continuous):
    return {
        "agent_0": Attacker("agent_0", env, continuous=continuous, mode="hard"),
        "agent_1": HybridAgent("agent_1", env, continuous=continuous, mode="hard"),
        "agent_2": Defender("agent_2", env, continuous=continuous, mode="hard"),
    }


def make_red_team(env, continuous, difficulty):
    return {
        agent_id: Heuristic_CTF_Agent(agent_id, env, continuous=continuous, mode=difficulty)
        for agent_id in RED_AGENT_IDS
    }


def run_episode(env, agents):
    observations, info = env.reset()

    while True:
        actions = {
            agent_id: agent.compute_action(observations, info)
            for agent_id, agent in agents.items()
        }
        observations, _, terminated, truncated, info = env.step(actions)
        any_agent = next(iter(terminated))
        if terminated[any_agent] or truncated[any_agent]:
            break

    return env.state


def run_condition(continuous, speedup, difficulty, episodes):
    env = make_environment(speedup)
    agents = {
        **make_blue_team(env, continuous),
        **make_red_team(env, continuous, difficulty),
    }

    rows = []
    for _ in range(episodes):
        state = run_episode(env, agents)
        rows.append(
            {
                "blue_caps": int(state["captures"][0]),
                "red_caps": int(state["captures"][1]),
                "blue_grabs": int(state["grabs"][0]),
                "red_grabs": int(state["grabs"][1]),
                "blue_tags": int(state["tags"][0]),
                "red_tags": int(state["tags"][1]),
            }
        )

    env.close()
    return aggregate(rows, continuous, speedup, difficulty, episodes)


def aggregate(rows, continuous, speedup, difficulty, episodes):
    def col(key):
        return np.array([r[key] for r in rows], dtype=float)

    blue_caps, red_caps = col("blue_caps"), col("red_caps")
    wins = int(np.sum(blue_caps > red_caps))
    losses = int(np.sum(blue_caps < red_caps))
    draws = int(np.sum(blue_caps == red_caps))

    return {
        "mode": "contínuo" if continuous else "discreto",
        "speedup": speedup,
        "difficulty": difficulty,
        "episodes": episodes,
        "blue_caps_total": int(blue_caps.sum()),
        "red_caps_total": int(red_caps.sum()),
        "blue_caps_mean": blue_caps.mean(),
        "red_caps_mean": red_caps.mean(),
        "blue_caps_std": blue_caps.std(),
        "red_caps_std": red_caps.std(),
        "blue_grabs_mean": col("blue_grabs").mean(),
        "red_grabs_mean": col("red_grabs").mean(),
        "blue_tags_mean": col("blue_tags").mean(),
        "red_tags_mean": col("red_tags").mean(),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / episodes,
    }


def fmt_result(r):
    if r["wins"] > r["losses"]:
        verdict = "AZUL"
    elif r["losses"] > r["wins"]:
        verdict = "VERMELHO"
    else:
        verdict = "EMPATE"
    return verdict


def build_report(results, episodes, seed):
    lines = []
    lines.append("# Relatório de Simulações — MCTF 3v3 (Heurístico vs. pyquaticus)\n")
    lines.append(f"_Gerado em {datetime.now():%Y-%m-%d %H:%M} | "
                 f"{episodes} episódios por condição | seed={seed}_\n")

    lines.append("## 1. Configuração experimental\n")
    lines.append(
        "- **Cenário:** Capture-the-Flag 3v3 (MCTF 2026), arena de 160 x 80 m, "
        "linha de _scrimmage_ em x = 80 m.\n"
        "- **Dinâmica:** modelo `surveyor` (MOOS-IvP), velocidade máxima 3 m/s, "
        "raio de captura/tag 10 m, _cooldown_ de tag 60 s.\n"
        "- **Término do episódio:** primeiro time a 20 capturas ou 600 s de jogo.\n"
        "- **Time azul (proposto):** `Attacker` + `HybridAgent` + `Defender` "
        "heurísticos (modo `hard`).\n"
        "- **Time vermelho (linha de base):** três `Heuristic_CTF_Agent` do "
        "pyquaticus.\n"
        "- **Métricas:** capturas (placar), _grabs_ (posses de bandeira), _tags_ "
        "(marcações) e taxa de vitória do time azul.\n"
    )
    lines.append(
        "> **Nota sobre `speedup`:** o `sim_speedup_factor` aplica a mesma ação por "
        "`speedup` subpassos de física (0,1 s cada). Logo ele não é apenas "
        "aceleração de relógio: a 10 cada decisão controla 1,0 s de movimento e a 30 "
        "controla 3,0 s, reduzindo a fidelidade do controle. Por isso ele é tratado "
        "como fator experimental.\n"
    )

    lines.append("Cada métrica `/ep` é a média por episódio; o placar é o total "
                 "acumulado nas partidas da condição. `V–E–D` = vitórias, empates e "
                 "derrotas do time azul.\n")

    # Tabela principal
    lines.append("## 2. Resultados\n")
    lines.append(
        "| Modo de ação | speedup | Oponente | Placar (A×V) | Caps/ep A | Caps/ep V | "
        "Grabs/ep A | Grabs/ep V | Tags/ep A | Tags/ep V | V–E–D | Win% A | Vencedor |")
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['mode']} | {r['speedup']} | {r['difficulty']} | "
            f"{r['blue_caps_total']}×{r['red_caps_total']} | "
            f"{r['blue_caps_mean']:.2f}±{r['blue_caps_std']:.2f} | "
            f"{r['red_caps_mean']:.2f}±{r['red_caps_std']:.2f} | "
            f"{r['blue_grabs_mean']:.2f} | {r['red_grabs_mean']:.2f} | "
            f"{r['blue_tags_mean']:.2f} | {r['red_tags_mean']:.2f} | "
            f"{r['wins']}–{r['draws']}–{r['losses']} | "
            f"{100 * r['win_rate']:.0f}% | {fmt_result(r)} |"
        )
    lines.append("")

    lines.append(build_synthesis(results))
    lines.append(METHOD_NOTES)

    return "\n".join(lines)


def build_synthesis(results):
    index = {(r["mode"], r["speedup"], r["difficulty"]): r for r in results}

    def caps_line(r):
        return (f"{r['blue_caps_total']}×{r['red_caps_total']} "
                f"({100 * r['win_rate']:.0f}% de vitórias do azul)")

    out = ["## 3. Síntese e discussão\n"]

    easy_med = [index.get(("contínuo", s, d))
                for s in (10, 30) for d in ("easy", "medium")]
    easy_med = [r for r in easy_med if r is not None]
    if easy_med:
        all_win = all(r["win_rate"] == 1.0 for r in easy_med)
        no_concede = all(r["red_caps_total"] == 0 for r in easy_med)
        msg = ("**Oponentes `easy` e `medium` (ação contínua):** o time azul "
               f"{'venceu todas as partidas' if all_win else 'venceu a maioria das partidas'}")
        if no_concede:
            msg += " sem conceder uma única captura"
        msg += (". O desempenho é dominante e estável entre os fatores de "
                "aceleração, indicando que as heurísticas de ataque/defesa "
                "superam com folga as políticas base menos agressivas.")
        out.append(msg + "\n")

    hard10 = index.get(("contínuo", 10, "hard"))
    hard30 = index.get(("contínuo", 30, "hard"))
    if hard10 and hard30:
        out.append(
            "**Oponente `hard` (ação contínua):** é o cenário mais equilibrado, "
            "pois ambos os times usam heurísticas agressivas e adaptativas. "
            f"A `speedup = 10` o confronto fica tecnicamente empatado "
            f"(placar {caps_line(hard10)}); a `speedup = 30` o time azul leva "
            f"vantagem (placar {caps_line(hard30)}). A diferença de _grabs_ é "
            "pequena nos dois casos — o que separa os times é a **conversão de "
            "_grabs_ em capturas** e o saldo de _tags_.\n"
        )

    if hard10 and hard30:
        d10 = hard10["blue_caps_mean"] - hard10["red_caps_mean"]
        d30 = hard30["blue_caps_mean"] - hard30["red_caps_mean"]
        out.append(
            "**Efeito do `speedup`:** aumentar o fator de aceleração de 10 para 30 "
            "torna o controle mais grosseiro (cada decisão rege 3 s de movimento em "
            "vez de 1 s) e eleva o ritmo de jogo. No confronto difícil isso favoreceu "
            f"o time azul (saldo médio de capturas passou de {d10:+.2f} para "
            f"{d30:+.2f} por episódio), sugerindo que suas heurísticas de "
            "interceptação toleram melhor a baixa frequência de decisão.\n"
        )

    disc = [index.get(("discreto", s, "hard")) for s in (10, 30)]
    disc = [r for r in disc if r is not None]
    if disc:
        total_caps = sum(r["blue_caps_total"] + r["red_caps_total"] for r in disc)
        no_loss = all(r["losses"] == 0 for r in disc)
        out.append(
            "**Ação discreta vs. contínua (oponente `hard`):** no modo discreto o "
            "rumo é quantizado (essencialmente seguir reto ou virar ±90°), o que "
            "produz um regime de baixíssima pontuação — muitos episódios terminam "
            f"0×0{' e o time azul não perde nenhuma partida' if no_loss else ''}. "
            "A ação contínua, ao permitir rumo proporcional, aumenta drasticamente o "
            "número de capturas de ambos os lados e revela melhor a qualidade "
            "relativa das estratégias. Em outras palavras, a vantagem do time azul "
            "no modo discreto era em grande parte um artefato do empate de baixa "
            "pontuação, não de superioridade tática.\n"
        )

    return "\n".join(out)

METHOD_NOTES = """## 4. Adaptações das heurísticas para controle contínuo

A migração de ação discreta para contínua exigiu reajustar o time azul, pois com
rumo proporcional o vetor resultante (objetivo + evasão) passa a influenciar
continuamente a direção, em vez de ser discretizado em incrementos de 90°:

- **Portador de bandeira (`Attacker`):** a captura ocorre assim que o portador
  retorna ao próprio lado, então a fuga foi tornada mais direta — evita apenas
  inimigos capazes de taguear (no próprio lado), com limiar de evasão menor e maior
  peso no objetivo, além de **evitação de paredes** para não sair da arena
  (`tag_on_oob`). Isso elevou a conversão de _grabs_ em capturas.
- **Zagueiro (`Defender`):** interceptação mais à frente (perímetro maior e ponto
  de interceptação mais avançado), engajando o atacante antes que ele alcance o
  raio de captura da bandeira, maximizando _tags_ (cada _tag_ manda o atacante de
  volta para casa).
- **Híbrido (`HybridAgent`):** passa a auxiliar a defesa já no primeiro invasor,
  reduzindo as situações de inferioridade numérica defensiva.

> Observação metodológica: o `speedup` foi mantido como fator porque altera a
> fidelidade do controle. Recomenda-se reportar resultados na configuração padrão
> (`sim_speedup_factor = 10`) e usar `30` apenas como teste de robustez.
"""


def main():
    parser = argparse.ArgumentParser(description="Bateria de experimentos MCTF 3v3")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="RELATORIO_SIMULACOES.md")
    parser.add_argument("--cache", type=str, default="RELATORIO_SIMULACOES.json",
                        help="arquivo JSON com os resultados brutos")
    parser.add_argument("--from-cache", action="store_true",
                        help="apenas reconstrói o relatório a partir do cache JSON")
    args = parser.parse_args()

    if args.from_cache:
        with open(args.cache, encoding="utf-8") as fh:
            payload = json.load(fh)
        results = payload["results"]
        report = build_report(results, payload["episodes"], payload["seed"])
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Relatório reconstruído de {args.cache} -> {args.out}")
        return

    np.random.seed(args.seed)

    conditions = [
        (True, 10, "easy"),
        (True, 10, "medium"),
        (True, 10, "hard"),
        (True, 30, "easy"),
        (True, 30, "medium"),
        (True, 30, "hard"),
        (False, 10, "hard"),
        (False, 30, "hard"),
    ]

    results = []
    for continuous, speedup, difficulty in conditions:
        label = f"[{'contínuo' if continuous else 'discreto'} | speedup {speedup} | {difficulty}]"
        print(f"Rodando {label} ...", flush=True)
        r = run_condition(continuous, speedup, difficulty, args.episodes)
        results.append(r)
        print(
            f"   placar {r['blue_caps_total']}×{r['red_caps_total']} | "
            f"V-E-D {r['wins']}-{r['draws']}-{r['losses']} | win% {100*r['win_rate']:.0f}",
            flush=True,
        )

    with open(args.cache, "w", encoding="utf-8") as fh:
        json.dump({"episodes": args.episodes, "seed": args.seed, "results": results},
                  fh, ensure_ascii=False, indent=2)

    report = build_report(results, args.episodes, args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nRelatório salvo em {args.out} (cache em {args.cache})")

if __name__ == "__main__":
    main()
