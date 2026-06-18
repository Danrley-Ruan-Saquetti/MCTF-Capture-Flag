import numpy as np

from pyquaticus.structs import Team

GAMMA = 0.99

WEIGHTS = {
    "capture": 1.0,
    "capture_against": 1.0,
    "grab": 0.25,
    "grab_against": 0.25,
    "tag_carrier": 0.5,
    "tag_generic": 0.1,
    "got_tagged": 0.1,
    "got_tagged_carrying": 0.5,
    "oob": 1.0,
    "collision": 0.02,
    "shaping": 0.05,
}

def _idx(agents, agent_id):
    return agents.index(agent_id)

def caps_and_grabs(
    agent_id, team, agents, agent_inds_of_team, state, prev_state,
    env_size, agent_radius, catch_radius, scrimmage_coords, max_speeds, tagging_cooldown,
):
    reward = 0.0
    i = _idx(agents, agent_id)

    if state["agent_oob"][i] > prev_state["agent_oob"][i]:
        reward += -1.0

    if prev_state["agent_has_flag"][i] > state["agent_has_flag"][i]:
        reward += -0.25

    for t in range(len(state["grabs"])):
        if state["grabs"][t] > prev_state["grabs"][t]:
            reward += 0.25 if t == int(team) else -0.25
        if state["captures"][t] > prev_state["captures"][t]:
            reward += 1.0 if t == int(team) else -1.0

    return reward

def _potential(state, idx, team_idx, opp_idx, diag):
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
    w = WEIGHTS
    i = _idx(agents, agent_id)
    team_idx = int(team)
    opp_idx = 1 - team_idx
    reward = 0.0

    for t in range(len(state["grabs"])):
        if state["grabs"][t] > prev_state["grabs"][t]:
            reward += w["grab"] if t == team_idx else -w["grab_against"]
        if state["captures"][t] > prev_state["captures"][t]:
            reward += w["capture"] if t == team_idx else -w["capture_against"]

    if state["agent_oob"][i] > prev_state["agent_oob"][i]:
        reward -= w["oob"]

    if state["agent_is_tagged"][i] and not prev_state["agent_is_tagged"][i]:
        if prev_state["agent_has_flag"][i]:
            reward -= w["got_tagged_carrying"]
        else:
            reward -= w["got_tagged"]

    tagged_other = state["agent_made_tag"][i]
    if tagged_other is not None and prev_state["agent_made_tag"][i] is None:
        if prev_state["agent_has_flag"][int(tagged_other)]:
            reward += w["tag_carrier"]
        else:
            reward += w["tag_generic"]

    new_collisions = state["agent_collisions"][i] - prev_state["agent_collisions"][i]
    if new_collisions > 0:
        reward -= w["collision"] * float(new_collisions)

    diag = float(np.linalg.norm(np.asarray(env_size, dtype=float)))
    phi_next = _potential(state, i, team_idx, opp_idx, diag)
    phi_prev = _potential(prev_state, i, team_idx, opp_idx, diag)
    reward += w["shaping"] * (GAMMA * phi_next - phi_prev)

    return reward
