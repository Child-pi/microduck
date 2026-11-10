import os
import json
import shutil
import cv2
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from envs.microduck_env import MicroduckEnv

def draw_hud(frame: np.ndarray, step: int, dist_x: float, y_pos: float, z_pos: float, gx: float, gy: float, vx_cmd: float) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Top translucent telemetry banner
    draw.rectangle([12, 12, w - 12, 96], fill=(11, 16, 32, 210), outline=(56, 189, 248, 180), width=2)

    t_sec = step * 0.02
    pct = min(100.0, max(0.0, (dist_x / 5.0) * 100.0))
    status_txt = "5.00m GOAL ACHIEVED!" if dist_x >= 5.0 else "WALKING TO 5.00m TARGET"
    status_col = (34, 197, 94, 255) if dist_x >= 5.0 else (56, 189, 248, 255)

    draw.text((24, 20), f"MICRODUCK 14-DOF RL | {status_txt}", fill=status_col)
    draw.text(
        (24, 42),
        f"Time: {t_sec:6.2f}s ({step:4d} steps)  |  Forward +X: {dist_x:+6.3f} m / 5.000 m ({pct:5.1f}%)",
        fill=(241, 245, 249, 255),
    )
    draw.text(
        (24, 64),
        f"Centerline Y: {y_pos:+6.3f} m  |  Height Z: {z_pos:.3f} m  |  Pitch gx: {gx:+.3f}  |  Waddle gy: {gy:+.3f}  |  Cmd Vx: {vx_cmd:.2f} m/s",
        fill=(148, 163, 184, 255),
    )

    # Bottom 5.0m progress bar
    bar_x0, bar_y0, bar_x1, bar_y1 = 16, h - 34, w - 16, h - 14
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(15, 23, 42, 215), outline=(99, 179, 237, 160), width=1)
    fill_w = int((bar_x1 - bar_x0 - 4) * min(1.0, max(0.0, dist_x / 5.165)))
    if fill_w > 0:
        bar_col = (34, 197, 94, 235) if dist_x >= 5.0 else (56, 189, 248, 235)
        draw.rectangle([bar_x0 + 2, bar_y0 + 2, bar_x0 + 2 + fill_w, bar_y1 - 2], fill=bar_col)
    for m in [1.0, 2.0, 3.0, 4.0, 5.0]:
        tx = bar_x0 + 2 + int((bar_x1 - bar_x0 - 4) * (m / 5.165))
        draw.line([(tx, bar_y0), (tx, bar_y1)], fill=(255, 255, 255, 180), width=2)
        draw.text((tx - 10, bar_y0 - 14), f"{int(m)}m", fill=(226, 232, 240, 255))

    return np.array(img)

def main():
    os.makedirs("videos", exist_ok=True)
    shutil.copyfile("checkpoints/microduck_ppo_720000_steps.zip", "checkpoints/microduck_5m_champion.zip")
    print("✅ Saved checkpoints/microduck_5m_champion.zip")

    with open("checkpoints/cmd_schedule_5m.json") as f:
        sched = {int(k): np.array(v, dtype=np.float32) for k, v in json.load(f).items()}

    model = PPO.load("checkpoints/microduck_5m_champion.zip", device="cpu")
    env = MicroduckEnv(action_scale=0.35, max_episode_steps=25000, render_mode="rgb_array", render_width=640, render_height=480)
    env.resample_commands = False

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.trunk_body_id
    cam.distance = 0.56
    cam.elevation = -20.0
    cam.azimuth = 132.0

    obs, _ = env.reset(seed=8)
    x0 = float(env.data.qpos[0])
    total_steps = 8500

    mp4_path = "videos/walk_5m_duck_champion.mp4"
    gif_path = "videos/walk_5m_duck_champion.gif"
    png_path = "videos/walk_5m_duck_champion.png"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, 25.0, (640, 480))
    gif_frames = []
    last_hud_frame = None

    for s in range(total_steps):
        if s in sched:
            env.commands[:] = sched[s]
            obs[48:61] = env.commands
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(action)

        if (s + 1) % 2 == 0 or s + 1 == total_steps:
            renderer.update_scene(env.data, camera=cam)
            raw_rgb = renderer.render()
            dist_x = float(env.data.qpos[0] - x0)
            y_pos = float(env.data.qpos[1])
            z_pos = float(env.data.qpos[2])
            gx = float(obs[3])
            gy = float(obs[4])
            hud_rgb = draw_hud(raw_rgb, s + 1, dist_x, y_pos, z_pos, gx, gy, float(env.commands[0]))
            writer.write(cv2.cvtColor(hud_rgb, cv2.COLOR_RGB2BGR))
            last_hud_frame = hud_rgb

            if (s + 1) % 50 == 0 or s + 1 == total_steps:
                small = Image.fromarray(hud_rgb).resize((480, 360), Image.Resampling.LANCZOS)
                gif_frames.append(small)

        if (s + 1) % 1700 == 0 or s + 1 == total_steps:
            print(f"🎬 Rendered {s+1}/{total_steps} steps ({(s+1)*0.02:.1f}s) | dist={env.data.qpos[0]-x0:+.3f}m")

    writer.release()
    if gif_frames:
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=65,
            loop=0,
            optimize=True,
        )
    if last_hud_frame is not None:
        Image.fromarray(last_hud_frame).save(png_path)
    renderer.close()
    env.close()
    print(f"✅ Saved {mp4_path}, {gif_path}, and {png_path}!")

if __name__ == "__main__":
    main()
