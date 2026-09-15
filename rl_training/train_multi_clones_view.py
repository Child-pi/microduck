#!/usr/bin/env python3
"""Microduck 8-Clone Parallel RL Training Visualizer.

Runs 8 parallel reinforcement learning environments simultaneously,
rendering all 8 duck clones in a real-time 2x4 split-screen grid on screen!
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import envs
from envs.microduck_env import MicroduckEnv


# Distinct theme colors for 8 clones (BGR format)
CLONE_COLORS = [
    (240, 180, 50),   # #32b4f0 Cyan / Sky
    (50, 140, 255),   # #ff8c32 Coral Orange
    (80, 220, 120),   # #78dc50 Emerald Green
    (220, 100, 200),  # #c864dc Neon Violet
    (50, 220, 255),   # #ffd232 Amber Gold
    (180, 80, 255),   # #ff50b4 Magenta Pink
    (255, 200, 80),   # #50c8ff Azure Blue
    (100, 240, 240),  # #f0f064 Bright Lime
]


class MultiClonesRenderCallback(BaseCallback):
    """Callback rendering all 8 clone environments in a 2x4 grid."""

    def __init__(
        self,
        num_clones: int = 8,
        render_width: int = 320,
        render_height: int = 240,
        fps_limit: int = 30,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.num_clones = num_clones
        self.rw = render_width
        self.rh = render_height
        self.fps_limit = fps_limit
        self.window_name = "🦆 Microduck 8 影分身同步強化學習 (8 Clones Live RL Training)"

        self.clone_ep_rewards = [0.0] * num_clones
        self.clone_ep_counts = [1] * num_clones
        self.clone_falls = [0] * num_clones
        self.last_frame_time = time.time()
        self.step_counter = 0

    def _on_training_start(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    def _on_step(self) -> bool:
        self.step_counter += 1
        rewards = self.locals.get("rewards", [0.0] * self.num_clones)
        dones = self.locals.get("dones", [False] * self.num_clones)

        for i in range(self.num_clones):
            self.clone_ep_rewards[i] += float(rewards[i])
            if dones[i]:
                self.clone_ep_counts[i] += 1
                self.clone_falls[i] += 1
                self.clone_ep_rewards[i] = 0.0

        # Periodic TensorBoard logging for each clone / training object
        if self.step_counter % 25 == 0 and self.logger is not None:
            for i in range(self.num_clones):
                self.logger.record(f"clones/clone_{i + 1}_reward", self.clone_ep_rewards[i])
                self.logger.record(f"clones/clone_{i + 1}_episodes", self.clone_ep_counts[i])
            self.logger.record("clones/mean_reward", float(np.mean(self.clone_ep_rewards)))
            self.logger.dump(step=self.num_timesteps)

        # Render all 8 clones
        clone_frames: List[np.ndarray] = []
        for i, sub_env in enumerate(self.training_env.envs):
            raw_env = sub_env.unwrapped
            frame = raw_env.render()
            if frame is None:
                frame = np.zeros((self.rh, self.rw, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if frame.shape[0] != self.rh or frame.shape[1] != self.rw:
                    frame = cv2.resize(frame, (self.rw, self.rh), interpolation=cv2.INTER_AREA)

            # Clone HUD
            color = CLONE_COLORS[i % len(CLONE_COLORS)]
            trunk_z = float(raw_env.data.qpos[2])
            cmd_vx = float(raw_env.commands[0])
            is_alive = trunk_z >= 0.06

            # Clone header card
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.rw, 46), (15, 20, 28), -1)
            cv2.rectangle(overlay, (0, self.rh - 24), (self.rw, self.rh), (15, 20, 28), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Border with clone color
            cv2.rectangle(frame, (0, 0), (self.rw - 1, self.rh - 1), color, 2)

            # Status dot
            dot_color = (60, 230, 90) if is_alive else (50, 50, 240)
            cv2.circle(frame, (16, 22), 6, dot_color, -1)

            # Label
            cv2.putText(
                frame,
                f"分身 #{i + 1} (Clone {i + 1})",
                (30, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
            )

            # Stats line
            status_text = f"z:{trunk_z:.2f}m | vx:{cmd_vx:+.2f}"
            cv2.putText(
                frame,
                status_text,
                (self.rw - 145, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (220, 220, 220),
                1,
            )

            # Bottom info: Reward & Episode
            bot_text = f"Ep:{self.clone_ep_counts[i]}  Rew:{self.clone_ep_rewards[i]:+.1f}"
            cv2.putText(
                frame,
                bot_text,
                (10, self.rh - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 250, 200) if self.clone_ep_rewards[i] >= 0 else (180, 180, 255),
                1,
            )

            clone_frames.append(frame)

        # Build 2x4 grid (4 columns, 2 rows)
        if len(clone_frames) >= 8:
            row1 = np.hstack(clone_frames[0:4])
            row2 = np.hstack(clone_frames[4:8])
            grid = np.vstack([row1, row2])
        elif len(clone_frames) == 4:
            grid = np.hstack(clone_frames)
        else:
            grid = clone_frames[0]

        # Top Master Control Header Bar
        grid_w = grid.shape[1]
        header = np.full((64, grid_w, 3), (12, 18, 28), dtype=np.uint8)

        # Title
        cv2.putText(
            header,
            "MICRODUCK 8 影分身同步強化學習 (8 CLONES MULTI-AGENT LIVE TRAINING)",
            (18, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (46, 196, 182),
            2,
        )

        # System stats
        tot_steps = self.num_timesteps
        mean_rew = float(np.mean(self.clone_ep_rewards))
        stats_str = (
            f"Step: {tot_steps:,} | 8-Env Mean Rew: {mean_rew:+.1f} | "
            f"FPS Target: {self.fps_limit} | Apple M4 Pro 12-Core | Press [Q] to Save & Exit"
        )
        cv2.putText(
            header,
            stats_str,
            (18, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1,
        )

        # Combine Header + Grid
        full_window = np.vstack([header, grid])

        cv2.imshow(self.window_name, full_window)

        # Frame pacing
        if self.fps_limit > 0:
            elapsed = time.time() - self.last_frame_time
            delay = max(1, int((1.0 / self.fps_limit - elapsed) * 1000))
        else:
            delay = 1

        key = cv2.waitKey(delay) & 0xFF
        self.last_frame_time = time.time()

        if key in (ord("q"), ord("Q"), 27):
            print("\n⚠️ 使用者手動退出 8 分身即時訓練視窗。")
            return False

        return True

    def _on_training_end(self) -> None:
        print("\n訓練結束！即將關閉 8 分身即時視窗。")
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Microduck 8 Clones Live Visual RL Training")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training steps (default: 100,000)")
    parser.add_argument("--num-clones", type=int, default=8, help="Number of clones (default: 8)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--fps", type=int, default=25, help="Visual playback FPS target (default: 25)")
    parser.add_argument("--resume-from", type=str, default="checkpoints/best/best_model.zip", help="Checkpoint to resume")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f"🦆 啟動 Microduck {args.num_clones} 影分身同步強化學習訓練 (8 Clones Live View)")
    print("=" * 70)
    print(f"分身數量: {args.num_clones} 個 Microduck 分身")
    print(f"即時視窗: 2x4 分割畫面 (即時顯示 8 隻小鴨探索學習)")
    print(f"接續模型: {args.resume_from}")
    print("即將在你的螢幕上彈出 8 分身實時畫面...")
    print("=" * 70)

    # 300x225 or 320x240 for crisp 2x4 grid (1280x480 resolution)
    def make_env_fn(clone_id: int):
        def _thunk():
            env = MicroduckEnv(
                render_mode="rgb_array",
                render_width=320,
                render_height=240,
            )
            return Monitor(env)
        return _thunk

    train_env = DummyVecEnv([make_env_fn(i) for i in range(args.num_clones)])

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
    )

    resume_path = args.resume_from
    if resume_path and not os.path.exists(resume_path):
        # Fallback to microduck_final.zip if best_model doesn't exist
        alt = "checkpoints/microduck_final.zip"
        if os.path.exists(alt):
            resume_path = alt
        else:
            resume_path = None

    if resume_path and os.path.exists(resume_path):
        print(f"📂 載入接續權重: {resume_path} (8 分身將立即展示高水準步伐訓練)")
        model = PPO.load(resume_path, env=train_env, device="cpu", tensorboard_log="logs")
    else:
        print("🌱 從隨機權重開始 8 分身平行訓練")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cpu",
            tensorboard_log="logs",
        )

    callback = MultiClonesRenderCallback(
        num_clones=args.num_clones,
        render_width=320,
        render_height=240,
        fps_limit=args.fps,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
        )
    except KeyboardInterrupt:
        print("\n使用者手動中斷訓練。")
    finally:
        os.makedirs("checkpoints", exist_ok=True)
        save_path = "checkpoints/microduck_8clones_trained.zip"
        model.save(save_path)
        print(f"\n✅ 8 分身訓練進度已儲存至: {save_path}")
        train_env.close()


if __name__ == "__main__":
    main()
