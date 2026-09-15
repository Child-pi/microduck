#!/usr/bin/env python3
"""Interactive Keyboard Controller for Microduck Robot.

Allows user to drive the Microduck biped robot interactively using WASD keys!
Features:
  - 'W' / 'S': Increase / Decrease forward velocity (vx)
  - 'A' / 'D': Turn left / Turn right (omega yaw)
  - 'X' or 'SPACE': Emergency brake / Stand still
  - 'R': Reset robot to initial stand pose
  - 'Q' or ESC: Quit controller
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO

import envs
from envs.microduck_env import MicroduckEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Microduck Interactive Keyboard Controller")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best/best_model.zip",
        help="Path to trained PPO model zip",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = None
    if os.path.exists(args.model_path):
        print(f"Loading trained policy from: {args.model_path}")
        model = PPO.load(args.model_path, device="cpu")
    else:
        print(f"⚠️ Model checkpoint not found at '{args.model_path}'. Running with baseline.")

    env = MicroduckEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Current user commands
    vx = 0.0
    vy = 0.0
    omega = 0.0

    window_name = "🦆 Microduck Keyboard Controller (WASD to Drive)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "=" * 65)
    print("🎮 Microduck 鍵盤即時操控模式已啟動！")
    print("=" * 65)
    print("操作按鍵：")
    print("  [W] : 加速前進 (+0.05 m/s)")
    print("  [S] : 減速 / 後退 (-0.05 m/s)")
    print("  [A] : 左轉向 (+0.25 rad/s)")
    print("  [D] : 右轉向 (-0.25 rad/s)")
    print("  [空白鍵] / [X] : 煞車 / 原地站立平衡")
    print("  [R] : 重置機器人位置")
    print("  [Q] / [ESC] : 離開操控視窗")
    print("=" * 65)

    last_time = time.time()

    while True:
        # Override commands
        env.commands[0] = vx
        env.commands[1] = vy
        env.commands[2] = omega

        # Policy inference
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(14, dtype=np.float32)

        obs, reward, terminated, truncated, step_info = env.step(action)

        if terminated or truncated:
            print("⚠️ 機器人傾倒或重置，自動復位站立...")
            obs, info = env.reset()

        # Render 3D frame
        frame = env.render()
        if frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw HUD Overlay
            h, w, _ = bgr.shape
            overlay = bgr.copy()
            cv2.rectangle(overlay, (10, 10), (320, 140), (10, 15, 25), -1)
            cv2.addWeighted(overlay, 0.75, bgr, 0.25, 0, bgr)
            cv2.rectangle(bgr, (10, 10), (320, 140), (80, 120, 180), 1)

            # Actual trunk velocity
            trunk_mat = env.data.xmat[env.trunk_body_id].reshape(3, 3)
            real_vx = (trunk_mat.T @ env.data.qvel[:3])[0]
            real_omega = (trunk_mat.T @ env.data.qvel[3:6])[2]

            cv2.putText(bgr, "MICRODUCK TELEOP", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 40), 2)
            cv2.putText(bgr, f"Cmd Vx: {vx:+.2f} m/s | Actual: {real_vx:+.2f}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
            cv2.putText(bgr, f"Cmd Turn: {omega:+.2f} rad/s | Actual: {real_omega:+.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
            cv2.putText(bgr, f"Trunk Height: {env.data.qpos[2]:.3f} m", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 230, 180), 1)
            cv2.putText(bgr, "[W/A/S/D]: Drive | [Space]: Brake | [Q]: Quit", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

            cv2.imshow(window_name, bgr)

        # Keyboard input handling (20ms wait = 50 FPS)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break
        elif key in (ord('w'), ord('W')):
            vx = float(np.clip(vx + 0.05, -0.4, 0.5))
            print(f"🎮 前進速度增加: vx={vx:.2f} m/s")
        elif key in (ord('s'), ord('S')):
            vx = float(np.clip(vx - 0.05, -0.4, 0.5))
            print(f"🎮 後退 / 減速: vx={vx:.2f} m/s")
        elif key in (ord('a'), ord('A')):
            omega = float(np.clip(omega + 0.25, -1.2, 1.2))
            print(f"🎮 向左轉向: omega={omega:.2f} rad/s")
        elif key in (ord('d'), ord('D')):
            omega = float(np.clip(omega - 0.25, -1.2, 1.2))
            print(f"🎮 向右轉向: omega={omega:.2f} rad/s")
        elif key in (ord(' '), ord('x'), ord('X')):
            vx = 0.0
            vy = 0.0
            omega = 0.0
            print("🛑 煞車停止，原地站立平衡")
        elif key in (ord('r'), ord('R')):
            obs, info = env.reset()
            vx = 0.0
            omega = 0.0
            print("🔄 機器人狀態已重置")

    cv2.destroyAllWindows()
    env.close()
    print("👋 互動操控已結束。")


if __name__ == "__main__":
    main()
