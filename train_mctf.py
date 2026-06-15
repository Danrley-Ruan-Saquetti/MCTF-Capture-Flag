import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pyquaticus"))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus.mctf26_config import get_std_config

from strategies.rl import opponent_league as league_module
from strategies.rl.config_mctf import get_experiment_config
from strategies.rl.evaluation import evaluate, make_blue_actor_from_algo
from strategies.rl.obs_id_wrapper import AgentIDObsWrapper
from strategies.rl.opponent_league import OpponentLeague
from strategies.rl.rewards_mctf import mctf_team_reward

BLUE_AGENT_IDS = ("agent_0", "agent_1", "agent_2")
RED_AGENT_IDS = ("agent_3", "agent_4", "agent_5")

def build_reward_config():
    reward_config = {agent_id: mctf_team_reward for agent_id in BLUE_AGENT_IDS}
    reward_config.update({agent_id: None for agent_id in RED_AGENT_IDS})

    return reward_config

def make_env_creator(append_agent_id):
    def env_creator(_env_config):
        base_env = CompPyquaticusEnv(
            config_dict=get_std_config(),
            render_mode=None,
            reward_config=build_reward_config(),
        )

        if append_agent_id:
            base_env = AgentIDObsWrapper(base_env)

        return ParallelPettingZooWrapper(base_env)

    return env_creator

def build_algorithm(experiment_config):
    env_creator = make_env_creator(experiment_config.append_agent_id)
    register_env("pyquaticus", env_creator)

    temporary_env = env_creator({})
    observation_space = temporary_env.observation_space["agent_0"]
    action_space = temporary_env.action_space["agent_0"]
    temporary_env.close()

    league = OpponentLeague(experiment_config, observation_space, action_space)
    ppo_config = _build_ppo_config(experiment_config, league)
    algorithm = ppo_config.build_algo()

    return algorithm, league

def _build_ppo_config(experiment_config, league):
    resources = experiment_config.resources
    hyperparameters = experiment_config.ppo

    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env="pyquaticus")
        .framework("torch")
        .env_runners(
            num_env_runners=resources.num_env_runners,
            num_cpus_per_env_runner=resources.num_cpus_per_env_runner,
            rollout_fragment_length="auto",
            sample_timeout_s=600.0,
        )
        .resources(num_gpus=resources.resolved_num_gpus())
        .training(
            train_batch_size=hyperparameters.train_batch_size,
            minibatch_size=hyperparameters.sgd_minibatch_size,
            num_epochs=hyperparameters.num_sgd_iter,
            gamma=hyperparameters.gamma,
            lambda_=hyperparameters.lambda_,
            clip_param=hyperparameters.clip_param,
            vf_clip_param=hyperparameters.vf_clip_param,
            entropy_coeff=hyperparameters.entropy_coeff,
            kl_coeff=hyperparameters.kl_coeff,
            grad_clip=hyperparameters.grad_clip,
            lr=hyperparameters.lr,
            model={
                "fcnet_hiddens": list(hyperparameters.fcnet_hiddens),
                "fcnet_activation": hyperparameters.fcnet_activation,
            },
        )
        .multi_agent(
            policies=league.build_policies(),
            policy_mapping_fn=league_module.policy_mapping_fn,
            policies_to_train=league.policies_to_train,
        )
        .debugging(seed=experiment_config.seed)
    )

def run_training(algorithm, league, experiment_config):
    phases = experiment_config.curriculum.phases
    phase_index = 0
    iterations_in_phase = 0

    league.set_phase(algorithm, phases[phase_index])
    print(f"[fase] iniciando em {phases[phase_index].name}")

    for iteration in range(experiment_config.eval.total_iters):
        result = algorithm.train()
        iterations_in_phase += 1

        print(f"iter {iteration:4d} | fase {phases[phase_index].name} | "
              f"reward_mean {_episode_reward_mean(result)}")

        if iteration % experiment_config.eval.checkpoint_every == 0:
            _save_checkpoint(algorithm, experiment_config)

        if iteration > 0 and iteration % experiment_config.league.snapshot_every == 0:
            snapshot_id = league.snapshot_into_next_slot(algorithm)
            league.sync(algorithm)

            print(f"[snapshot] política congelada em {snapshot_id}; ativos={league.active_snaps}")

        if iteration > 0 and iteration % experiment_config.eval.eval_every == 0:
            advanced = _evaluate_and_maybe_advance(
                algorithm, league, experiment_config, phase_index, iterations_in_phase
            )

            if advanced:
                phase_index += 1
                iterations_in_phase = 0

    final_path = os.path.join(experiment_config.eval.checkpoint_dir, "final")
    algorithm.save(final_path)

    print(f"[final] checkpoint: {final_path}")

def _episode_reward_mean(result):
    return result.get("env_runners", {}).get(
        "episode_reward_mean", result.get("episode_reward_mean", float("nan"))
    )

def _save_checkpoint(algorithm, experiment_config):
    global_iteration = int(algorithm.iteration)
    checkpoint_path = os.path.join(experiment_config.eval.checkpoint_dir, f"iter_{global_iteration}")
    algorithm.save(checkpoint_path)
    print(f"[ckpt] salvo: {checkpoint_path}")

def _evaluate_and_maybe_advance(algorithm, league, experiment_config, phase_index, iterations_in_phase):
    phases = experiment_config.curriculum.phases
    phase = phases[phase_index]
    opponent = "easy" if "easy" in phase.opponent_dist else "noop"

    blue_actor = make_blue_actor_from_algo(algorithm, experiment_config, league.blue_ids[0])
    metrics = evaluate(
        blue_actor, opponent=opponent, episodes=experiment_config.eval.eval_episodes,
        seed=experiment_config.seed, exp=experiment_config,
    )

    print(f"[eval vs {opponent}] win_rate={metrics['win_rate']:.2f} "
          f"cap_diff={metrics['cap_diff_mean']:.2f} col={metrics['collisions_mean']:.1f}")

    can_advance = (
        phase_index < len(phases) - 1
        and iterations_in_phase >= phase.min_iters
        and metrics["win_rate"] >= phase.win_rate_advance
        and metrics["cap_diff_mean"] >= phase.cap_diff_advance
    )

    if can_advance:
        league.set_phase(algorithm, phases[phase_index + 1])
        print(f"[fase] AVANÇOU para {phases[phase_index + 1].name}")

    return can_advance


def parse_arguments():
    parser = argparse.ArgumentParser(description="Treino RL 3v3 MCTF")
    parser.add_argument("--independent", action="store_true", help="três políticas independentes em vez de uma compartilhada")
    parser.add_argument("--no-agent-id", action="store_true", help="não anexar o one-hot de papel ao obs")
    parser.add_argument("--total-iters", type=int, default=None)
    parser.add_argument("--num-env-runners", type=int, default=None, help="sobrescreve o nº de env_runners (0 = rollout local, p/ debug)")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--snapshot-every", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="caminho de checkpoint para retomar")
    parser.add_argument("--allow-gpu", action="store_true", help="usar GPU se o torch tiver CUDA (default: CPU-only)")

    return parser.parse_args()

def apply_arguments(experiment_config, arguments):
    if arguments.independent:
        experiment_config.shared_policy = False
        experiment_config.append_agent_id = False
    if arguments.no_agent_id:
        experiment_config.append_agent_id = False
    if arguments.total_iters is not None:
        experiment_config.eval.total_iters = arguments.total_iters
    if arguments.num_env_runners is not None:
        experiment_config.resources.num_env_runners = arguments.num_env_runners
    if arguments.eval_every is not None:
        experiment_config.eval.eval_every = arguments.eval_every
    if arguments.eval_episodes is not None:
        experiment_config.eval.eval_episodes = arguments.eval_episodes
    if arguments.checkpoint_every is not None:
        experiment_config.eval.checkpoint_every = arguments.checkpoint_every
    if arguments.snapshot_every is not None:
        experiment_config.league.snapshot_every = arguments.snapshot_every
    if arguments.allow_gpu:
        experiment_config.resources.allow_gpu = True

def main():
    arguments = parse_arguments()
    experiment_config = get_experiment_config()
    apply_arguments(experiment_config, arguments)

    experiment_config.eval.checkpoint_dir = os.path.abspath(experiment_config.eval.checkpoint_dir)
    os.makedirs(experiment_config.eval.checkpoint_dir, exist_ok=True)

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    try:
        algorithm, league = build_algorithm(experiment_config)

        if arguments.resume:
            resume_path = os.path.abspath(arguments.resume)
            print(f"[resume] restaurando de {resume_path}")
            algorithm.restore(resume_path)

        run_training(algorithm, league, experiment_config)
        algorithm.stop()
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
