"""Train PPO with any Atari2600 env.

Author: Jinwoo Park
Email: www.jwpark.co.kr@gmail.com
"""

import argparse
from typing import Any, Dict

import gym
import ray
from ray.rllib.agents import ppo
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from ray.tune.logger import pretty_print


def train(train_config: Dict[str, Any], n_iters: int) -> None:
    """Train the agent with n_iters iterations."""
    agent = ppo.PPOTrainer(config=train_config, env=train_config["env"])
    for _ in range(n_iters):
        result = agent.train()
        print(pretty_print(result))
        checkpoint_path = agent.save()
        print(f"Checkpoint saved in {checkpoint_path}")


def evaluate(eval_config: Dict[str, Any], checkpoint_path: str, render: bool) -> None:
    """Evaluate the agent with a single iteration."""
    agent = ppo.PPOTrainer(config=eval_config, env=eval_config["env"])
    agent.restore(checkpoint_path)
    print(f"Checkpoint loaded from {checkpoint_path}")

    env = gym.make(eval_config["env"], render_mode="human" if render else None)
    env = wrap_deepmind(env)
    obs = env.reset()
    print(f"Created env for {eval_config['env']}")

    done = False
    score = 0.0
    while not done:
        action = agent.compute_single_action(obs)
        obs, reward, done, _ = env.step(action)
        score += reward

    env.close()
    print(f"Evaluation score: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="Breakout-v0",
        help="Atari-2600 env name (Default: Breakout-v0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path for inference",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=10,
        help="Training iteration number (Default: 10)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of workers for sampling (Default: 4)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU (Default: False)")
    parser.add_argument("--render", action="store_true", help="Render env during eval")
    args = parser.parse_args()

    # initialize ray
    ray.init()

    # set configrations
    # you can choose any RLlib algorithm that supports Descrete Actions:
    # https://docs.ray.io/en/releases-1.9.0/rllib-algorithms.html
    config = {
        "env": args.env,
        "num_gpus": int(args.gpu),
        "lr": 1e-3,
        "num_workers": args.n_workers,
        "framework": "torch",
    }
    train_config = ppo.DEFAULT_CONFIG.copy()
    train_config.update(config)

    # evaluate
    if args.checkpoint:
        print("Start evaluation.")
        eval_config = train_config
        eval_config["evaluation_interval"] = None
        eval_config["num_workers"] = 0
        eval_config["explore"] = False
        evaluate(eval_config, args.checkpoint, args.render)
    # train
    else:
        print("Start training.")
        train(train_config, n_iters=args.n_iters)
