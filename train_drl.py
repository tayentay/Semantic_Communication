from __future__ import annotations

"""
PPO trainer for semantic communication, adapted from vwxyzjn/ppo-implementation-details.

Usage (CPU-friendly defaults):
    python train_drl.py --config config/default.yaml --total-timesteps 20000
"""

import argparse
import os
import random
import time
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from semantic_comm.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO for SemanticCom (vwxyzjn/ppo-implementation-details style)")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--exp-name", type=str, default="semanticcom-ppo", help="experiment name prefix")
    parser.add_argument("--total-timesteps", type=int, default=20000, help="total environment steps")
    parser.add_argument("--num-envs", type=int, default=2, help="number of parallel envs")
    parser.add_argument("--num-steps", type=int, default=256, help="rollout length per env")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="optimizer LR")
    parser.add_argument("--anneal-lr", action="store_true", default=True, help="anneal learning rate")
    parser.add_argument("--gae", action="store_true", default=True, help="use GAE")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--num-minibatches", type=int, default=4, help="PPO minibatches")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO epochs")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="surrogate clip coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="entropy bonus")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="value loss weight")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="grad clip")
    parser.add_argument("--target-kl", type=float, default=None, help="optional early stop KL")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--cuda", action="store_true", default=False, help="force CUDA")
    parser.add_argument("--episode-length", type=int, default=32, help="env episode length (optimizer steps)")
    parser.add_argument("--reward-scale", type=float, default=10.0, help="scale energy to reward")
    parser.add_argument("--latency-penalty", type=float, default=25.0, help="penalty for latency over budget")
    parser.add_argument("--coverage-penalty", type=float, default=5.0, help="penalty per uncovered GT")
    parser.add_argument("--track-interval", type=int, default=10, help="log every N updates")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(action_shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)


def main():
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    os.makedirs("runs", exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_fns = [
        make_env(
            cfg_path=args.config,
            episode_length=args.episode_length,
            reward_scale=args.reward_scale,
            latency_penalty=args.latency_penalty,
            coverage_penalty=args.coverage_penalty,
        )
        for _ in range(args.num_envs)
    ]
    envs = gym.vector.SyncVectorEnv(env_fns)
    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

    agent = Agent(obs_shape, action_shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            clipped_action = torch.clamp(action, -1.0, 1.0)
            next_obs_np, reward, term, trunc, infos = envs.step(clipped_action.cpu().numpy())
            done = np.logical_or(term, trunc)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
            next_done = torch.tensor(done, dtype=torch.float32).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if mb_advantages.numel() > 0:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                if approx_kl > args.target_kl:
                    break

        if update % args.track_interval == 0 or update == num_updates:
            explained_var = (
                torch.var(b_returns - b_values).item()
                if torch.var(b_returns).item() > 1e-8
                else float("nan")
            )
            avg_reward = rewards.mean().item()
            print(
                f"[{run_name}] update {update}/{num_updates} "
                f"avg_reward={avg_reward:.3f} "
                f"last_return={b_returns[-1].item():.3f} "
                f"explained_var={explained_var:.3f}"
            )

    envs.close()
    torch.save(agent.state_dict(), os.path.join("runs", f"{run_name}.pt"))
    print(f"Training complete. Policy saved to runs/{run_name}.pt")


if __name__ == "__main__":
    main()
