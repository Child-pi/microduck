#!/usr/bin/env python3
"""Microduck 8-Clone Simultaneous Demonstration & Evaluation.

Renders all 8 duck clones in a real-time 2x4 grid, each executing
diverse locomotion gaits and commands using the trained PPO policy!
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import List

import cv2
import numpy as np
from stable_baselines3 import PPO

import envs
from envs.microduck_env import MicroduckEnv

# Distinct theme colors for 8 clones (BGR format)
CLONE_COLORS = [
    (240, 180, 50),   # #32b4f0 Cyan
    (50, 140, 255),   # #ff8c32 Coral Orange
    (80, 220, 120),   # #78dc50 Emerald Green
    (220, 100, 200),  # #c864dc Neon Violet
    (50, 220, 255),   # #ffd232 Amber Gold
    (180, 80, 255),   # #ff50b4 Magenta Pink
    (255, 200, 80),   # #50c8ff Azure Blue
    (100, 240, 240),  # #f0f064 Bright Lime
]

CLONE_NAMES = [
    "分身 #1 直行標準步態",
    "分身 #2 弧形左轉行走",
    "分身 #3 弧形右轉行走",
    "分身 #4 衝刺快速步伐",
    "分身 #5 慢速穩健步伐",
    "分身 #6 點頭探索步態",
    "分身 #7 轉頭環顧步態",
    "分身 #8 自由隨機應變",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Microduck 8 Clones Simultaneous Evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best/best_model.zip",
        help="Path to trained PPO model zip",
    )
    parser.add_argument("--steps", type=int, default=1500, help="Number of evaluation steps")
    parser.add_argument("--record", type=str, default=None, help="Save MP4 recording to this path")
    parser.add_argument("--fps", type=int, default=30, help="Display FPS")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        if os.path.exists("checkpoints/microduck_final.zip"):
            model_path = "checkpoints/microduck_final.zip"
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    print("=" * 70)
    print("🦆 Microduck 8 影分身全景同屏展示 (8 Clones Simultaneous Showcase)")
    print("=" * 70)
    print(f"載入模型: {model_path}")
    print(f"展示分身: 8 個平行小鴨環境 (2x4 分割畫面)")
    print("即將彈出實時 8 分身畫面...")
    print("=" * 70)

    # Load policy
    model = PPO.load(model_path, device="cpu")

    rw, rh = 320, 240
    envs_list = [
        MicroduckEnv(
            render_mode="rgb_array",
            render_width=rw,
            render_height=rh,
        )
        for _ in range(8)
    ]

    observations = [env.reset()[0] for env in envs_list]
    rewards_acc = [0.0] * 8
    ep_count = [1] * 8

    window_name = "🦆 Microduck 8 影分身同步展示 (8 Clones Live Showcase)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    video_writer = None
    grid_w = rw * 4
    grid_h = rh * 2 + 64

    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.record, fourcc, args.fps, (grid_w, grid_h))
        print(f"🎥 正在錄製 8 分身影片至: {args.record}")

    last_time = time.time()

    for step in range(args.steps):
        t = step * 0.02

        # Assign unique gaits/commands to each clone
        # Clone 1: Straight normal
        envs_list[0].commands[0] = 0.30
        envs_list[0].commands[1] = 0.0
        envs_list[0].commands[2] = 0.0

        # Clone 2: Left turn
        envs_list[1].commands[0] = 0.25
        envs_list[1].commands[1] = 0.0
        envs_list[1].commands[2] = 0.35

        # Clone 3: Right turn
        envs_list[2].commands[0] = 0.25
        envs_list[2].commands[1] = 0.0
        envs_list[2].commands[2] = -0.35

        # Clone 4: Fast sprint
        envs_list[3].commands[0] = 0.42
        envs_list[3].commands[1] = 0.0
        envs_list[3].commands[2] = 0.0

        # Clone 5: Slow trot
        envs_list[4].commands[0] = 0.16
        envs_list[4].commands[1] = 0.0
        envs_list[4].commands[2] = 0.0

        # Clone 6: Nodding head
        envs_list[5].commands[0] = 0.28
        envs_list[5].commands[3] = 0.25 * math.sin(t * 3.0)  # neck pitch
        envs_list[5].commands[4] = 0.20 * math.sin(t * 3.0)  # head pitch

        # Clone 7: Looking around
        envs_list[6].commands[0] = 0.28
        envs_list[6].commands[5] = 0.40 * math.sin(t * 2.5)  # head yaw

        # Clone 8: S-curve wander
        envs_list[7].commands[0] = 0.30 + 0.10 * math.sin(t * 2.0)
        envs_list[7].commands[2] = 0.40 * math.cos(t * 1.5)

        # Predict actions for all 8 clones
        obs_array = np.array(observations)
        actions, _ = model.predict(obs_array, deterministic=True)

        clone_frames: List[np.ndarray] = []

        for i, env in enumerate(envs_list):
            obs, rew, terminated, truncated, _ = env.step(actions[i])
            observations[i] = obs
            rewards_acc[i] += rew

            if terminated or truncated:
                observations[i], _ = env.reset()
                ep_count[i] += 1
                rewards_acc[i] = 0.0

            # Render
            f = env.render()
            if f is None:
                f = np.zeros((rh, rw, 3), dtype=np.uint8)
            else:
                f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)

            # Clone HUD
            color = CLONE_COLORS[i]
            cv2.rectangle(f, (0, 0), (rw - 1, rh - 1), color, 2)

            # Top overlay
            overlay = f.copy()
            cv2.rectangle(overlay, (0, 0), (rw, 36), (15, 20, 30), -1)
            cv2.rectangle(overlay, (0, rh - 26), (rw, rh), (15, 20, 30), -1)
            cv2.addWeighted(overlay, 0.70, f, 0.30, 0, f)

            # Label
            cv2.circle(f, (14, 18), 5, (60, 230, 90), -1)
            cv2.putText(f, f"Clone #{i + 1}", (26, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
            cv2.putText(
                f,
                f"vx:{env.commands[0]:+.2f}",
                (rw - 85, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (220, 220, 220),
                1,
            )

            # Bottom info
            cv2.putText(
                f,
                CLONE_NAMES[i],
                (8, rh - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (240, 240, 240),
                1,
            )

            clone_frames.append(f)

        # 2x4 layout
        row1 = np.hstack(clone_frames[0:4])
        row2 = np.hstack(clone_frames[4:8])
        grid = np.vstack([row1, row2])

        # Header bar
        header = np.full((64, grid_w, 3), (12, 18, 28), dtype=np.uint8)
        cv2.putText(
            header,
            "MICRODUCK 8 影分身全景同屏實時展示 (8 CLONES SIMULTANEOUS SHOWCASE)",
            (20, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (46, 196, 182),
            2,
        )

        mean_rew = float(np.mean(rewards_acc))
        sub_str = (
            f"Step: {step:04d}/{args.steps} | 8 Clones Active | Mean Rew: {mean_rew:+.1f} | "
            f"50 Hz Real-time Control | Press [Q] or [ESC] to Exit"
        )
        cv2.putText(
            header,
            sub_str,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1,
        )

        full_frame = np.vstack([header, grid])

        cv2.imshow(window_name, full_frame)
        if video_writer:
            video_writer.write(full_frame)

        # Pacing
        elapsed = time.time() - last_time
        delay = max(1, int((1.0 / args.fps - elapsed) * 1000))
        last_time = time.time()

        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("\n使用者手動關閉展示視窗。")
            break

    for env in envs_list:
        env.close()

    if video_writer:
        video_writer.release()
        print(f"✅ 8 分身展示影片已儲存: {args.record}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
