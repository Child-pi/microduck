#!/usr/bin/env python3
"""Sanity test script for Microduck Gymnasium environment."""

import time
import numpy as np
import envs  # Registers Microduck-v0
from envs.microduck_env import MicroduckEnv

def main():
    print("=" * 60)
    print("Testing Microduck Gymnasium Environment...")
    print("=" * 60)

    env = MicroduckEnv()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    assert env.action_space.shape == (14,), f"Expected 14 actions, got {env.action_space.shape}"
    assert env.observation_space.shape == (61,), f"Expected 61 obs, got {env.observation_space.shape}"

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\nReset successful!")
    print(f"Observation shape: {obs.shape}")
    print(f"Initial trunk height: {env.data.qpos[2]:.4f} m")
    print(f"Initial commands (vx, vy, omega): {info['commands'][:3]}")
    print(f"Gyro trunk: {obs[0:3]}")
    print(f"Projected gravity: {obs[3:6]}")
    print(f"Joint pos offset from home: max={np.max(np.abs(obs[6:20])):.4f} rad")

    # Step test
    print("\nStepping environment for 200 steps with random actions...")
    start_time = time.time()
    total_reward = 0.0

    for step in range(200):
        action = env.action_space.sample() * 0.1  # small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if (step + 1) % 50 == 0 or step == 0:
            print(
                f"Step {step+1:3d} | Reward: {reward:+.3f} | "
                f"Trunk z: {info['trunk_height']:.3f}m | "
                f"Rew lin_vel: {info['rew_lin_vel']:.3f} | "
                f"Rew upright: {info['rew_upright']:.3f}"
            )

        if terminated or truncated:
            print(f"Episode finished early at step {step+1} (terminated={terminated}, truncated={truncated})")
            obs, info = env.reset()

    elapsed = time.time() - start_time
    fps = 200 / elapsed
    sim_fps = 200 * env.frame_skip / elapsed
    print("\n" + "=" * 60)
    print(f"Test completed successfully!")
    print(f"Elapsed: {elapsed:.3f}s | Policy steps/sec: {fps:.1f} Hz | MuJoCo physics steps/sec: {sim_fps:.1f} Hz")
    print("=" * 60)
    env.close()

if __name__ == "__main__":
    main()
