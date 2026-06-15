"""
Funções de recompensa para o treino RL do MCTF.

Inclui:
  - `caps_and_grabs`  : versão correta da recompensa esparsa de referência
                        (idêntica à de pyquaticus/utils/rewards.py — os termos
                        `prev_` vêm de `prev_state`, NÃO de `state`). O "bug"
                        citado no TRAIN_AGENT.md existe só no exemplo do doc; o
                        código instalado já está correto. Vendorizada aqui para
                        o treino não depender da doc.
  - `mctf_team_reward`: recompensa principal — esparso dominante e majoritariamente
                        de time, com pequeno shaping potential-based (PBRS) que
                        guia o ciclo grab -> carregar -> capturar (o gargalo
                        identificado na análise das heurísticas) sem alterar a
                        política ótima (Ng et al., 1999).

NOTA DE SERIALIZAÇÃO (Ray): o env serializa `reward_config` para os workers via
cloudpickle. Por isso os PESOS são constantes de módulo (picláveis por
referência) e não closures/config. Para recalibrar, edite WEIGHTS abaixo.

NOTA PBRS: GAMMA aqui DEVE casar com `ppo.gamma` em config_mctf.py.
"""

import numpy as np

from pyquaticus.structs import Team

# Desconto usado no shaping PBRS — manter igual ao gamma do PPO.
GAMMA = 0.99

# Pesos da recompensa (todos editáveis). Termos esparsos de time dominam.
WEIGHTS = {
    "capture": 1.0,       # capturar a bandeira inimiga (evento de time)
    "capture_against": 1.0,
    "grab": 0.25,         # pegar a bandeira inimiga (evento de time)
    "grab_against": 0.25,
    "tag_carrier": 0.5,   # marcar um inimigo que carregava nossa bandeira
    "tag_generic": 0.1,   # marcar um inimigo qualquer
    "got_tagged": 0.1,    # ser marcado
    "got_tagged_carrying": 0.5,  # ser marcado carregando a bandeira
    "oob": 1.0,           # sair dos limites (auto-tag)
    "collision": 0.02,    # por nova colisão (<4 m) — penaliza desempate
    "shaping": 0.05,      # peso do termo PBRS (pequeno: só orienta)
}

def _idx(agents, agent_id):
    return agents.index(agent_id)

def caps_and_grabs(
    agent_id, team, agents, agent_inds_of_team, state, prev_state,
    env_size, agent_radius, catch_radius, scrimmage_coords, max_speeds, tagging_cooldown,
):
    """Recompensa esparsa de referência (correta). Útil como baseline/ablação."""
    reward = 0.0
    i = _idx(agents, agent_id)

    if state["agent_oob"][i] > prev_state["agent_oob"][i]:
        reward += -1.0

    # Perdeu a bandeira
    if prev_state["agent_has_flag"][i] > state["agent_has_flag"][i]:
        reward += -0.25

    for t in range(len(state["grabs"])):
        if state["grabs"][t] > prev_state["grabs"][t]:
            reward += 0.25 if t == int(team) else -0.25
        if state["captures"][t] > prev_state["captures"][t]:
            reward += 1.0 if t == int(team) else -1.0

    return reward

def _potential(state, idx, team_idx, opp_idx, diag):
    """
    Potencial Φ(s) para PBRS (em [-1, 0]):
      - sem bandeira: progride rumo à bandeira inimiga (alvo = posição da bandeira inimiga);
      - com bandeira: progride rumo à própria base (alvo = casa da própria bandeira).
    Quanto mais perto do alvo, maior (menos negativo) o potencial.
    """
    pos = np.asarray(state["agent_position"][idx], dtype=float)
    if state["agent_has_flag"][idx]:
        target = np.asarray(state["flag_home"][team_idx], dtype=float)
    else:
        target = np.asarray(state["flag_position"][opp_idx], dtype=float)
    return -float(np.linalg.norm(pos - target)) / diag

def mctf_team_reward(
    agent_id, team, agents, agent_inds_of_team, state, prev_state,
    env_size, agent_radius, catch_radius, scrimmage_coords, max_speeds, tagging_cooldown,
):
    """
    Recompensa principal do MCTF (esparso dominante + PBRS leve).

    Termos de TIME (todos os azuis recebem -> incentiva cooperação, não egoísmo):
        captura ±1.0, grab ±0.25.
    Termos INDIVIDUAIS:
        marcar carregador +0.5 / marcar geral +0.1; ser marcado -0.5 (c/ bandeira)
        ou -0.1; OOB -1.0; colisão -0.02 por nova ocorrência (<4 m).
    Shaping PBRS: WEIGHTS['shaping'] * (γ·Φ(s') − Φ(s)).
    """
    w = WEIGHTS
    i = _idx(agents, agent_id)
    team_idx = int(team)
    opp_idx = 1 - team_idx
    reward = 0.0

    # --- Termos esparsos de time: capturas e grabs (deltas dos contadores) ---
    for t in range(len(state["grabs"])):
        if state["grabs"][t] > prev_state["grabs"][t]:
            reward += w["grab"] if t == team_idx else -w["grab_against"]
        if state["captures"][t] > prev_state["captures"][t]:
            reward += w["capture"] if t == team_idx else -w["capture_against"]

    # --- OOB (auto-tag) ---
    if state["agent_oob"][i] > prev_state["agent_oob"][i]:
        reward -= w["oob"]

    # --- Ser marcado (transição não-marcado -> marcado) ---
    if state["agent_is_tagged"][i] and not prev_state["agent_is_tagged"][i]:
        # Carregava a bandeira no passo anterior? Punição maior (a bandeira volta).
        if prev_state["agent_has_flag"][i]:
            reward -= w["got_tagged_carrying"]
        else:
            reward -= w["got_tagged"]

    # --- Marcar um inimigo (eu apliquei um tag neste passo) ---
    tagged_other = state["agent_made_tag"][i]
    if tagged_other is not None and prev_state["agent_made_tag"][i] is None:
        # O inimigo marcado carregava nossa bandeira? Bônus maior.
        if prev_state["agent_has_flag"][int(tagged_other)]:
            reward += w["tag_carrier"]
        else:
            reward += w["tag_generic"]

    # --- Colisões (<4 m): contador cumulativo por agente, penaliza o delta ---
    new_collisions = state["agent_collisions"][i] - prev_state["agent_collisions"][i]
    if new_collisions > 0:
        reward -= w["collision"] * float(new_collisions)

    # --- Shaping PBRS: F = γ·Φ(s') − Φ(s) ---
    diag = float(np.linalg.norm(np.asarray(env_size, dtype=float)))
    phi_next = _potential(state, i, team_idx, opp_idx, diag)
    phi_prev = _potential(prev_state, i, team_idx, opp_idx, diag)
    reward += w["shaping"] * (GAMMA * phi_next - phi_prev)

    return reward
