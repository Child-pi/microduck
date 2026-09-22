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
import sys
import subprocess

# Auto-configure headless MuJoCo rendering on Linux / Google Colab
if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    has_gpu = False
    try:
        has_gpu = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        pass
    if has_gpu:
        icd_path = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        if not os.path.exists(icd_path):
            try:
                os.makedirs(os.path.dirname(icd_path), exist_ok=True)
                with open(icd_path, "w") as f:
                    f.write('{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n')
            except Exception:
                pass
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    else:
        os.environ["MUJOCO_GL"] = "osmesa"

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

        # Foot positions & neck/head pitch
        lf_z = float(env.data.site_xpos[env.left_foot_site_id][2])
        rf_z = float(env.data.site_xpos[env.right_foot_site_id][2])
        joint_pos = env.data.qpos[7:21]
        head_pitch_deg = float(np.rad2deg(joint_pos[5] + joint_pos[6]))
        phase = float(getattr(env.unwrapped, "gait_phase", 0.0))
        waddle_target = float(getattr(env.unwrapped, "duck_waddle_amp", 0.075) * np.sin(phase))

        # Draw HUD on frame
        hud = rgb.copy()
        time_sec = s * 0.02
        cv2.putText(hud, f"Time: {time_sec:.2f}s  Step: {s}/{max_steps}  [DUCK GAIT]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
        cv2.putText(hud, f"Dist X: {dist_x:+.3f}m  Vel: {linvel[0]:+.2f}m/s (cmd {target_vx:.2f})", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(hud, f"Height: {z:.3f}m  Roll: {grav[1]:+.2f} (waddle tgt: {waddle_target:+.2f})", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(hud, f"Head Bob: {head_pitch_deg:+.1f} deg  Phase: {phase:.2f} rad", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)

        # Foot indicators with clearance in mm
        l_color = (0, 255, 0) if lc else (60, 60, 200)
        r_color = (0, 255, 0) if rc else (60, 60, 200)
        cv2.rectangle(hud, (20, 125), (130, 155), l_color, -1)
        cv2.putText(hud, f"L:{lf_z*1000:.0f}mm", (25, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(hud, (145, 125), (255, 155), r_color, -1)
        cv2.putText(hud, f"R:{rf_z*1000:.0f}mm", (150, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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

    # Save MP4 (try H.264 via imageio first for HTML5 browser playback, fallback to OpenCV)
    saved_h264 = False
    try:
        import imageio
        writer = imageio.get_writer(output_video, fps=50, codec="libx264", pixelformat="yuv420p")
        for f in frames:
            writer.append_data(f)
        writer.close()
        saved_h264 = True
        print(f"Saved web-compatible H.264 MP4 via imageio: {output_video}")
    except Exception:
        pass

    if not saved_h264:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(output_video, fourcc, 50.0, (640, 480))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        # If ffmpeg is available, ensure H.264 web compatibility
        try:
            import shutil
            ffmpeg_bin = shutil.which("ffmpeg") or ("/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg") else "ffmpeg")
            temp_raw = output_video + ".temp.mp4"
            os.replace(output_video, temp_raw)
            ret = subprocess.run(
                [ffmpeg_bin, "-y", "-i", temp_raw, "-vcodec", "libx264", "-pix_fmt", "yuv420p", output_video],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ret.returncode == 0 and os.path.exists(output_video):
                os.remove(temp_raw)
            else:
                os.replace(temp_raw, output_video)
        except Exception:
            pass
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
