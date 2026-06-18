from dataclasses import dataclass, field
from typing import Dict, List

import torch

SEED = 42

@dataclass
class Resources:
    num_env_runners: int = 6
    num_cpus_per_env_runner: int = 1
    num_cpus_for_main_process: int = 1

    num_gpus: float = 0.0
    allow_gpu: bool = False

    def resolved_num_gpus(self) -> float:
        if self.allow_gpu and torch.cuda.is_available():
            return self.num_gpus if self.num_gpus > 0 else 0.5

        return 0.0

@dataclass
class PPOHyperparams:
    fcnet_hiddens: tuple = (256, 256)
    fcnet_activation: str = "tanh"

    train_batch_size: int = 6000
    sgd_minibatch_size: int = 1024
    num_sgd_iter: int = 10
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_param: float = 0.2
    lr: float = 3.0e-4
    entropy_coeff: float = 0.01
    kl_coeff: float = 0.2
    grad_clip: float = 1.0
    vf_clip_param: float = 10.0

@dataclass
class LeagueConfig:
    n_snapshot_slots: int = 5
    snapshot_every: int = 50
    heuristic_mode: str = "competition_easy"

@dataclass
class Phase:
    name: str
    opponent_dist: Dict[str, float]
    win_rate_advance: float
    cap_diff_advance: float
    min_iters: int

@dataclass
class Curriculum:
    phases: List[Phase] = field(
        default_factory=lambda: [
            Phase(
                name="A_warmup",
                opponent_dist={"noop": 0.5, "easy": 0.5},
                win_rate_advance=0.60,
                cap_diff_advance=0.3,
                min_iters=30,
            ),
            Phase(
                name="B_competition_easy",
                opponent_dist={"easy": 1.0},
                win_rate_advance=0.55,
                cap_diff_advance=0.3,
                min_iters=60,
            ),
            Phase(
                name="C_self_play",
                opponent_dist={"easy": 0.3, "snap": 0.7},
                win_rate_advance=2.0,
                cap_diff_advance=99.0,
                min_iters=10**9,
            ),
        ]
    )

@dataclass
class EvalConfig:
    eval_every: int = 25
    eval_episodes: int = 20
    checkpoint_every: int = 25
    checkpoint_dir: str = "./ray_tests"
    total_iters: int = 4000

@dataclass
class ExperimentConfig:
    shared_policy: bool = True
    append_agent_id: bool = True
    resources: Resources = field(default_factory=Resources)
    ppo: PPOHyperparams = field(default_factory=PPOHyperparams)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    curriculum: Curriculum = field(default_factory=Curriculum)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = SEED

    blue_shared_id: str = "blue-shared"
    blue_independent_ids: tuple = ("agent-0-policy", "agent-1-policy", "agent-2-policy")

def get_experiment_config() -> ExperimentConfig:
    return ExperimentConfig()
