#!/usr/bin/env python3
"""Automated Video Recording & Gait Analyzer for Microduck Locomotion.

Renders high-definition simulation with an informative HUD:
- Current Step & Elapsed Time (s)
- Forward Velocity vx vs Target Command
- Trunk Height z
- Pitch gx (Forward/Back) & Roll gy (Lateral tilt)
- Left & Right Foot Contact Stance Indicators
Saves MP4 video and animated GIF for visual inspection.
"""

from __future__ import annotations

import argparse
import os
import cv2
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

import envs
from envs.microduck_env import MicroduckEnv


def evaluate_and_record(
    model_path: str,
    output_video: str = "walk_analysis.mp4",
    output_gif: str = "walk_analysis.gif",
    output_snapshot: str = "walk_snapshot.png",
    target_vx: float = 0.08,
    max_steps: int = 600,
    seed: int = 3,
):
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    env = MicroduckEnv(
        max_episode_steps=max_steps,
        render_mode="rgb_array",
        render_width=640,
        render_height=480,
    )

    env.resample_commands = False
    obs, _ = env.reset(seed=seed)
    env.commands[0] = target_vx
    env.commands[1] = 0.0
    env.commands[2] = 0.0
    obs = env._get_obs()

    frames = []
    telemetry = []

    print(f"Starting simulation for up to {max_steps} steps ({max_steps*0.02:.1f}s)...")

    for s in range(max_steps):
        rgb = env.render()

        # Telemetry
        z = float(env.data.qpos[2])
        mat = env.data.site_xmat[env.imu_site_id].reshape(3, 3)
        grav = mat.T @ np.array([0.0, 0.0, -1.0])
        linvel = mat.T @ env.data.qvel[:3]
        dist_x = float(env.data.qpos[0])

        lc, rc = False, False
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == env.floor_geom_id and g2 == env.left_foot_geom_id) or (g2 == env.floor_geom_id and g1 == env.left_foot_geom_id):
                lc = True
            if (g1 == env.floor_geom_id and g2 == env.right_foot_geom_id) or (g2 == env.floor_geom_id and g1 == env.right_foot_geom_id):
                rc = True

        # Draw HUD on frame
        hud = rgb.copy()
        time_sec = s * 0.02
        cv2.putText(hud, f"Time: {time_sec:.2f}s  Step: {s}/{max_steps}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(hud, f"Dist X: {dist_x:+.3f}m  Vel: {linvel[0]:+.2f}m/s (cmd {target_vx:.2f})", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(hud, f"Height Z: {z:.3f}m  Pitch gx: {grav[0]:+.2f}  Roll gy: {grav[1]:+.2f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Foot indicators
        l_color = (0, 255, 0) if lc else (100, 100, 100)
        r_color = (0, 255, 0) if rc else (100, 100, 100)
        cv2.rectangle(hud, (20, 115), (90, 145), l_color, -1)
        cv2.putText(hud, "L FOOT", (25, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.rectangle(hud, (105, 115), (175, 145), r_color, -1)
        cv2.putText(hud, "R FOOT", (110, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        frames.append(hud)
        telemetry.append((s, time_sec, z, dist_x, linvel[0], grav[0], grav[1], lc, rc))

        action, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(action)
        env.commands[0] = target_vx
        env.commands[1] = 0.0
        env.commands[2] = 0.0
        obs[48:51] = [target_vx, 0.0, 0.0]

        if term or trunc:
            print(f"Terminated at step {s} ({time_sec:.2f}s)! Final height: {z:.3f}m, dist: {dist_x:.3f}m")
            break

    total_steps = len(frames)
    duration_s = total_steps * 0.02
    print(f"\nSimulation completed: {total_steps} steps ({duration_s:.2f} seconds)")

    # Save MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(output_video, fourcc, 50.0, (640, 480))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"Saved video to: {output_video}")

    # Save GIF (downsample to 25 fps, frame stride 2)
    pil_frames = [Image.fromarray(f) for f in frames[::2]]
    if pil_frames:
        pil_frames[0].save(output_gif, save_all=True, append_images=pil_frames[1:], duration=40, loop=0)
        print(f"Saved GIF to: {output_gif}")
        pil_frames[len(pil_frames) // 2].save(output_snapshot)
        print(f"Saved snapshot to: {output_snapshot}")

    return duration_s, total_steps, telemetry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/microduck_endurance_10s_champion.zip")
    parser.add_argument("--output-video", type=str, default="walk_analysis.mp4")
    parser.add_argument("--output-gif", type=str, default="walk_analysis.gif")
    parser.add_argument("--output-snapshot", type=str, default="walk_snapshot.png")
    parser.add_argument("--vx", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    evaluate_and_record(
        model_path=args.model_path,
        output_video=args.output_video,
        output_gif=args.output_gif,
        output_snapshot=args.output_snapshot,
        target_vx=args.vx,
        max_steps=args.max_steps,
        seed=args.seed,
    )
