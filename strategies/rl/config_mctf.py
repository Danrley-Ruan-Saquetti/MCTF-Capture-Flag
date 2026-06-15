"""
Configuração central do treino RL para MCTF (PPO old API stack, RLlib 2.41).

Tudo que governa um experimento fica aqui: recursos de hardware, hiperparâmetros
do PPO, fases do currículo, parâmetros da liga de self-play e seeds. Os pesos da
recompensa ficam em `rewards_mctf.py` (constantes de módulo) por causa da
serialização do Ray para os workers — ver nota lá.

Alvo de hardware (decisão do projeto): WSL2, i5 8 núcleos, 40 GB RAM,
GTX 1650 4 GB. ATENÇÃO: o torch instalado é CPU-only (`torch==2.6.0+cpu`,
`cuda.is_available()==False`), então o treino roda 100% em CPU e `num_gpus=0`.
Para usar a GPU seria preciso reinstalar o torch com CUDA (ver README).
"""

from dataclasses import dataclass, field
from typing import Dict, List

import torch

# Seed global (reprodutibilidade). Fixada em todo o pipeline.
SEED = 42

# ---------------------------------------------------------------------------
# Recursos de hardware (caminho CPU-only — ver nota no topo)
# ---------------------------------------------------------------------------

@dataclass
class Resources:
    # 8 núcleos: 6 env_runners + 1 driver + 1 de folga para o SO/avaliação.
    num_env_runners: int = 6
    num_cpus_per_env_runner: int = 1
    num_cpus_for_main_process: int = 1
    # CPU-only por padrão. Só vira >0 se o torch tiver CUDA E `allow_gpu=True`.
    num_gpus: float = 0.0
    allow_gpu: bool = False

    def resolved_num_gpus(self) -> float:
        """num_gpus efetivo: 0 a menos que haja CUDA e uso de GPU autorizado."""
        if self.allow_gpu and torch.cuda.is_available():
            # MLP pequena: 0.5 GPU é suficiente e mantém VRAM bem abaixo de 4 GB.
            return self.num_gpus if self.num_gpus > 0 else 0.5
        return 0.0

# ---------------------------------------------------------------------------
# Hiperparâmetros do PPO (conservadores — cabem com folga em 40 GB de RAM)
# ---------------------------------------------------------------------------

@dataclass
class PPOHyperparams:
    fcnet_hiddens: tuple = (256, 256)
    fcnet_activation: str = "tanh"
    # ~10 episódios de 600 passos por iteração (batch menor = iterações mais
    # rápidas e mais updates; o env é lento em CPU, então isso melhora o
    # feedback sem perder estabilidade no PPO).
    train_batch_size: int = 6000
    sgd_minibatch_size: int = 1024
    num_sgd_iter: int = 10
    gamma: float = 0.99  # DEVE casar com GAMMA em rewards_mctf (shaping PBRS).
    lambda_: float = 0.95
    clip_param: float = 0.2
    lr: float = 3.0e-4
    entropy_coeff: float = 0.01
    kl_coeff: float = 0.2
    grad_clip: float = 1.0
    vf_clip_param: float = 10.0

# ---------------------------------------------------------------------------
# Liga / self-play
# ---------------------------------------------------------------------------

@dataclass
class LeagueConfig:
    # Slots fixos pré-registrados para snapshots congelados da própria política.
    # Usar slots fixos (em vez de add_policy dinâmico) é robusto no old API stack.
    n_snapshot_slots: int = 5
    # A cada quantas iterações de treino congelar um novo snapshot (round-robin).
    snapshot_every: int = 50
    # Perfil dos oponentes heurísticos de treino.
    heuristic_mode: str = "competition_easy"

# ---------------------------------------------------------------------------
# Currículo: fases com distribuição de oponentes e critério de avanço
# ---------------------------------------------------------------------------
#
# "kind" de oponente que o policy_mapping_fn pode amostrar:
#   - "noop"  : todos os vermelhos parados (NoOp).
#   - "easy"  : heurística competition_easy (combined) por agente vermelho.
#   - "snap"  : um snapshot congelado da própria política (self-play).
#
# Avanço de fase é decidido por avaliação periódica (eval_mctf): exige
# win_rate >= win_rate_advance E diferencial médio de capturas >= cap_diff_advance.

@dataclass
class Phase:
    name: str
    opponent_dist: Dict[str, float]
    win_rate_advance: float
    cap_diff_advance: float
    # Piso de iterações na fase antes de poder avançar (estabilidade).
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
                # Mantém um pouco de heurística para não esquecer e enfrenta
                # majoritariamente snapshots da própria política.
                opponent_dist={"easy": 0.3, "snap": 0.7},
                # Fase final: não avança (fica refinando via self-play).
                win_rate_advance=2.0,  # inalcançável => permanece na fase
                cap_diff_advance=99.0,
                min_iters=10**9,
            ),
        ]
    )

# ---------------------------------------------------------------------------
# Avaliação / checkpoints
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    eval_every: int = 25          # iterações entre avaliações de gating
    eval_episodes: int = 20       # episódios por avaliação
    checkpoint_every: int = 25    # iterações entre checkpoints
    checkpoint_dir: str = "./ray_tests"
    total_iters: int = 4000

@dataclass
class ExperimentConfig:
    shared_policy: bool = True     # política compartilhada (True) vs 3 independentes
    append_agent_id: bool = True   # anexa one-hot de papel ao obs (exige shared)
    resources: Resources = field(default_factory=Resources)
    ppo: PPOHyperparams = field(default_factory=PPOHyperparams)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    curriculum: Curriculum = field(default_factory=Curriculum)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = SEED

    # Ids das policies (mantidos como constantes para uso no mapping/liga).
    blue_shared_id: str = "blue-shared"
    blue_independent_ids: tuple = ("agent-0-policy", "agent-1-policy", "agent-2-policy")

def get_experiment_config() -> ExperimentConfig:
    """Config padrão do experimento (edite aqui ou via flags do train_mctf)."""
    return ExperimentConfig()
