#!/usr/bin/env python3
"""Evaluate a trained Microduck PPO policy in MuJoCo.

Supports on-screen 3D live rendering via OpenCV window (fully compatible
with standard python on macOS Apple Silicon, no mjpython required)
and high-definition MP4 video recording.
"""

from __future__ import annotations

import argparse
import os
import time
import cv2
import numpy as np
from stable_baselines3 import PPO

import envs
from envs.microduck_env import MicroduckEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Microduck Locomotion Policy")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best/best_model.zip",
        help="Path to trained PPO model zip",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max simulation steps per episode (default: 1000, 20s)",
    )
    parser.add_argument(
        "--vx",
        type=float,
        default=None,
        help="Override linear velocity x command (forward m/s)",
    )
    parser.add_argument(
        "--vy",
        type=float,
        default=None,
        help="Override linear velocity y command (lateral m/s)",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="Override yaw angular velocity command (turning rad/s)",
    )
    parser.add_argument(
        "--record-video",
        type=str,
        default=None,
        help="Path to save MP4 video (e.g. 'eval_video.mp4')",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable on-screen GUI rendering",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = None
    if os.path.exists(args.model_path):
        print(f"Loading trained policy from: {args.model_path}")
        model = PPO.load(args.model_path, device="cpu")
    else:
        alt = "checkpoints/microduck_final.zip"
        if os.path.exists(alt):
            print(f"Loading fallback policy from: {alt}")
            model = PPO.load(alt, device="cpu")
        else:
            print(f"⚠️ Model checkpoint not found at '{args.model_path}'. Running with zero/random policy for demonstration.")

    # Always use rgb_array to ensure 100% native macOS compatibility without mjpython
    show_window = not args.no_render
    render_mode = "rgb_array" if (show_window or args.record_video) else None

    env = MicroduckEnv(
        max_episode_steps=args.max_steps,
        render_mode=render_mode,
        render_width=640,
        render_height=480,
    )

    window_name = "🦆 Microduck 步態評估視窗 (Locomotion Evaluation)"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    # Video writer setup
    video_writer = None
    if args.record_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.record_video)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.record_video, fourcc, 50.0, (640, 480))
        print(f"🎥 Recording video to: {args.record_video} (50 FPS)")

    print("=" * 65)
    print("🦆 Starting Microduck Locomotion Evaluation")
    print("=" * 65)

    user_exit = False
    for ep in range(1, args.episodes + 1):
        if user_exit:
            break

        obs, info = env.reset(seed=ep * 100)

        # Apply velocity command overrides if requested
        if args.vx is not None:
            env.commands[0] = args.vx
        if args.vy is not None:
            env.commands[1] = args.vy
        if args.omega is not None:
            env.commands[2] = args.omega

        cmd_vx, cmd_vy, cmd_omega = env.commands[:3]
        print(f"\n[Episode {ep}/{args.episodes}] Commanded: vx={cmd_vx:+.2f} m/s, vy={cmd_vy:+.2f} m/s, omega={cmd_omega:+.2f} rad/s")

        ep_reward = 0.0
        step = 0
        last_time = time.time()

        while step < args.max_steps:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(14, dtype=np.float32)

            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            step += 1

            if render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    if show_window:
                        # Draw aesthetic HUD overlay
                        overlay = bgr_frame.copy()
                        cv2.rectangle(overlay, (10, 10), (380, 130), (12, 18, 30), -1)
                        cv2.addWeighted(overlay, 0.75, bgr_frame, 0.25, 0, bgr_frame)
                        cv2.rectangle(bgr_frame, (10, 10), (380, 130), (46, 196, 182), 1)

                        trunk_z = float(env.data.qpos[2])
                        cv2.putText(bgr_frame, f"EVALUATION: Episode {ep}/{args.episodes}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 159, 28), 2)
                        cv2.putText(bgr_frame, f"Step: {step:04d} | Rew: {ep_reward:+.1f}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        cv2.putText(bgr_frame, f"Command: vx={cmd_vx:+.2f}m/s | turn={cmd_omega:+.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 240, 160), 1)
                        cv2.putText(bgr_frame, f"Trunk Height: {trunk_z:.3f} m", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                        cv2.putText(bgr_frame, "Press [Q] or [ESC] to Exit", (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

                        cv2.imshow(window_name, bgr_frame)

                        # Pacing ~50 FPS
                        elapsed = time.time() - last_time
                        delay = max(1, int((1.0 / 50.0 - elapsed) * 1000))
                        last_time = time.time()

                        key = cv2.waitKey(delay) & 0xFF
                        if key in (ord('q'), ord('Q'), 27):
                            print("\n使用者手動退出評估視窗。")
                            user_exit = True
                            break

                    if video_writer is not None:
                        video_writer.write(bgr_frame)

            if (step % 100 == 0) or terminated or truncated:
                print(
                    f"  Step {step:4d} | Rew: {reward:+.3f} | Trunk Z: {step_info['trunk_height']:.3f}m | "
                    f"Vel track: {step_info['rew_lin_vel']:.2f} | Upright: {step_info['rew_upright']:.2f}"
                )

            if terminated or truncated:
                status = "Fall / Tipped" if terminated else "Completed"
                print(f"  Episode {ep} finished: {status} at step {step}. Total reward: {ep_reward:.2f}")
                break

    if video_writer is not None:
        video_writer.release()
        print(f"\n✅ Video successfully saved to: {args.record_video}")

    env.close()
    if show_window:
        cv2.destroyAllWindows()

    print("=" * 65)
    print("Evaluation finished!")


if __name__ == "__main__":
    main()
