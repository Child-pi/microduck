#!/usr/bin/env python3
"""Microduck Reinforcement Learning Training with Stable-Baselines3 (PPO).

Optimized for Apple Silicon / multi-core macOS.
Supports parallel vectorized environments, TensorBoard logging, and model checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import envs  # Registers Microduck-v0


def parse_args():
    parser = argparse.ArgumentParser(description="Train Microduck RL Locomotion Policy (PPO)")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps (default: 1,000,000)")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per environment per PPO update (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size (default: 128)")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (default: 0.95)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range (default: 0.2)")
    parser.add_argument("--ent-coef", type=float, default=0.005, help="Entropy coefficient (default: 0.005)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'mps')")
    parser.add_argument("--save-freq", type=int, default=50_000, help="Checkpoint frequency in timesteps (default: 50,000)")
    parser.add_argument("--eval-freq", type=int, default=20_000, help="Evaluation frequency in timesteps (default: 20,000)")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint zip to resume from")
    parser.add_argument("--duck-mimic", action="store_true", default=True, help="Enable biomimetic duck waddle, CPG, and head bobbing (default: True)")
    parser.add_argument("--f-duck", type=float, default=1.8, help="Duck stride frequency in Hz (default: 1.8)")
    parser.add_argument("--waddle-amp", type=float, default=0.075, help="Duck waddle roll amplitude in rad (default: 0.075)")
    parser.add_argument("--recipe", type=str, default=None, choices=["duck_waddle", "from_scratch"], help="Preset recipe ('duck_waddle' fine-tunes from best checkpoint with biomimetic parameters)")
    parser.add_argument("--output-name", type=str, default="microduck_final", help="Final model filename without extension (default: microduck_final)")
    return parser.parse_args()


def make_env(rank: int, seed: int = 0) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make("Microduck-v0")
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    args = parse_args()

    # Apply recipe presets if selected
    if args.recipe == "duck_waddle":
        if args.resume_from is None and os.path.exists(os.path.join(args.checkpoint_dir, "best", "best_model.zip")):
            args.resume_from = os.path.join(args.checkpoint_dir, "best", "best_model.zip")
        args.lr = 5e-5
        args.ent_coef = 0.0005
        args.gamma = 0.9985
        args.output_name = "microduck_duck_walk_champion"
        print("🦆 Applied 'duck_waddle' fine-tuning recipe (LR=5e-5, ent_coef=0.0005, gamma=0.9985)")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print("🦆 Starting Microduck RL Locomotion Training (PPO)")
    print("=" * 70)
    print(f"Total Timesteps  : {args.timesteps:,}")
    print(f"Parallel Envs    : {args.num_envs}")
    print(f"Learning Rate    : {args.lr}")
    print(f"PPO n_steps      : {args.n_steps} (Total buffer: {args.n_steps * args.num_envs:,})")
    print(f"Minibatch Size   : {args.batch_size}")
    print(f"Epochs per Update: {args.n_epochs}")
    print(f"Entropy Coef     : {args.ent_coef}")
    print(f"Duck Mimic Mode  : {args.duck_mimic} (f={args.f_duck} Hz, roll_amp={args.waddle_amp} rad)")
    print(f"Checkpoints Dir  : {args.checkpoint_dir}")
    print(f"TensorBoard Logs : {args.log_dir}")
    print(f"Device           : {args.device}")
    print("=" * 70)

    env_kwargs = {
        "duck_mimic_mode": args.duck_mimic,
        "duck_f_gait": args.f_duck,
        "duck_waddle_amp": args.waddle_amp,
    }

    # Vectorized training environments
    vec_env_cls = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    train_env = make_vec_env(
        "Microduck-v0",
        n_envs=args.num_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
        env_kwargs=env_kwargs,
    )

    # Vectorized evaluation environment
    eval_env = make_vec_env(
        "Microduck-v0",
        n_envs=1,
        seed=args.seed + 1000,
        vec_env_cls=DummyVecEnv,
        env_kwargs=env_kwargs,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.num_envs),
        save_path=args.checkpoint_dir,
        name_prefix="microduck_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.checkpoint_dir, "best"),
        log_path=os.path.join(args.log_dir, "eval"),
        eval_freq=max(1, args.eval_freq // args.num_envs),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Policy Network Architecture (MLP 256 -> 128 -> 64 with ELU activations)
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
    )

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\n📂 Resuming training from checkpoint: {args.resume_from}")
        model = PPO.load(
            args.resume_from,
            env=train_env,
            device=args.device,
            tensorboard_log=args.log_dir,
        )
        model.ent_coef = args.ent_coef
        model.learning_rate = args.lr
        model.gamma = args.gamma
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=args.log_dir,
        )

    print("\n🚀 Training in progress... Open TensorBoard in another terminal to monitor:")
    print(f"   tensorboard --logdir {args.log_dir}")
    print("-" * 70)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, args.output_name)
    model.save(final_path)
    print(f"\n✅ Training finished! Final model saved to: {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
