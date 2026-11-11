import os
import cv2
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from envs.microduck_env import MicroduckEnv

def draw_hud(frame: np.ndarray, step: int, dist_x: float, y_pos: float, z_pos: float, gx: float, gy: float, vx_cmd: float, waddle_rew: float, head_rew: float) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Top banner
    draw.rectangle([12, 12, w - 12, 102], fill=(11, 16, 32, 215), outline=(56, 189, 248, 190), width=2)
    t_sec = step * 0.02
    draw.text((24, 18), "🦆 MICRODUCK BIOMIMETIC DUCK WADDLE & BOBBING", fill=(34, 197, 94, 255))
    draw.text(
        (24, 38),
        f"Time: {t_sec:5.2f}s ({step:3d} steps)  |  Forward +X: {dist_x:+5.3f} m  |  Centerline Y: {y_pos:+5.3f} m",
        fill=(241, 245, 249, 255),
    )
    draw.text(
        (24, 58),
        f"Waddle Roll gy: {gy:+.3f} rad (~{np.degrees(abs(gy)):.1f} deg)  |  Pitch gx: {gx:+.3f}  |  Height Z: {z_pos:.3f} m",
        fill=(56, 189, 248, 255),
    )
    draw.text(
        (24, 78),
        f"Waddle Tracking Score: {waddle_rew:.2f} / 1.0  |  Head-Bobbing Score: {head_rew:.2f} / 1.0",
        fill=(251, 191, 36, 255),
    )
    return np.array(img)

def main():
    os.makedirs("videos", exist_ok=True)
    model = PPO.load("checkpoints/microduck_5m_duck_waddle_champion.zip", device="cpu")
    env = MicroduckEnv(action_scale=0.35, max_episode_steps=1200, render_mode="rgb_array", render_width=640, render_height=480)
    env.resample_commands = False

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.trunk_body_id
    cam.distance = 0.52
    cam.elevation = -18.0
    cam.azimuth = 135.0

    obs, _ = env.reset(seed=33)
    env.commands[0] = 0.16
    obs[48:61] = env.commands
    x0 = float(env.data.qpos[0])

    mp4_path = "videos/duck_waddle_walk_champion.mp4"
    gif_path = "videos/duck_waddle_walk_champion.gif"
    png_path = "videos/duck_waddle_walk_champion.png"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, 25.0, (640, 480))
    gif_frames = []
    last_hud_frame = None

    for s in range(1200):
        a, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(a)

        if (s + 1) % 2 == 0 or term or trunc:
            renderer.update_scene(env.data, camera=cam)
            raw_rgb = renderer.render()
            dist_x = float(env.data.qpos[0] - x0)
            y_pos = float(env.data.qpos[1])
            z_pos = float(env.data.qpos[2])
            gx = float(obs[3])
            gy = float(obs[4])
            w_score = float(info.get("rew_duck_waddle", 0.0))
            h_score = float(info.get("rew_duck_head_bob", 0.0))
            hud_rgb = draw_hud(raw_rgb, s + 1, dist_x, y_pos, z_pos, gx, gy, float(env.commands[0]), w_score, h_score)
            writer.write(cv2.cvtColor(hud_rgb, cv2.COLOR_RGB2BGR))
            last_hud_frame = hud_rgb

            if (s + 1) % 4 == 0:
                small = Image.fromarray(hud_rgb).resize((480, 360), Image.Resampling.LANCZOS)
                gif_frames.append(small)

        if term or trunc:
            print(f"Episode finished at step {s+1} ({(s+1)*0.02:.2f}s), dist: {env.data.qpos[0]-x0:+.3f}m")
            break

    writer.release()
    if gif_frames:
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=80,
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
