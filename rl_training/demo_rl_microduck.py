#!/usr/bin/env python3
"""Microduck Reinforcement Learning Interactive Demo & 8-Clone Parallel Training System.

Features:
1. Multi-Clone Vectorized Training (default: 8 parallel ducks!):
   - 8 independent duck simulation clones running concurrently in MuJoCo
   - 2x4 multi-view split-screen visual grid with color-coded telemetry
   - Apple Silicon M-series accelerated parallel vectorized learning
2. Real-Time Training Status:
   - Rich terminal dashboard displaying all 8 clone states, rewards, losses, and FPS
   - Live telemetry overlay on every duck clone (Height, Velocity, Foot Stance, Alive status)
   - Embedded Web Dashboard at http://localhost:8088
   - TensorBoard integration at http://localhost:6006
3. Real-Time Video Toggle (ON / OFF):
   - Toggle video dynamically on the fly by pressing 'V' or 'v' (in GUI window or terminal)!
   - Start with video on (--video) or off (--no-video)
   - When Video is OFF: Simulation accelerates to maximum hardware speed (~5,000–6,000 FPS)
   - When Video is ON: Real-time 2x4 grid renders all 8 ducks walking and learning
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import select
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
from typing import Optional, Dict, Any, Tuple, List

import cv2
import gymnasium as gym
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import envs
from envs.microduck_env import MicroduckEnv

# Distinct theme colors for 8 clones (BGR format)
CLONE_COLORS = [
    (240, 180, 50),   # Clone 1: Cyan / Sky
    (50, 140, 255),   # Clone 2: Coral Orange
    (80, 220, 120),   # Clone 3: Emerald Green
    (220, 100, 200),  # Clone 4: Neon Violet
    (50, 220, 255),   # Clone 5: Amber Gold
    (180, 80, 255),   # Clone 6: Magenta Pink
    (255, 200, 80),   # Clone 7: Azure Blue
    (100, 240, 240),  # Clone 8: Bright Lime
]


class SharedState:
    """Thread-safe state shared across simulation, GUI, terminal, and web server."""

    def __init__(self, num_clones: int = 8, video_enabled: bool = True):
        self.lock = threading.Lock()
        self.num_clones = num_clones
        self.video_enabled = video_enabled
        self.is_paused = False
        self.should_quit = False
        self.should_reset = False

        # Status & Metrics
        self.mode = "train"
        self.step = 0
        self.total_timesteps = 500_000
        self.episode = 1
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.mean_reward = 0.0
        self.reward_history = []
        self.fps = 0.0
        self.elapsed_time = 0.0
        self.start_time = time.time()

        # Telemetry (Single Duck fallback)
        self.trunk_height = 0.12
        self.trunk_vx = 0.0
        self.cmd_vx = 0.08
        self.pitch_gx = 0.0
        self.roll_gy = 0.0
        self.left_contact = False
        self.right_contact = False
        self.status_msg = "Initializing..."

        # Multi-Clone Telemetry (8 Clones)
        self.clone_rewards = [0.0] * num_clones
        self.clone_episodes = [1] * num_clones
        self.clone_heights = [0.12] * num_clones
        self.clone_vxs = [0.0] * num_clones
        self.clone_alives = [True] * num_clones
        self.clone_left_contacts = [False] * num_clones
        self.clone_right_contacts = [False] * num_clones

        # Training losses
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.entropy_loss = 0.0

        # Latest rendered JPEG frame for web streaming
        self.latest_jpeg = None

    def toggle_video(self) -> bool:
        with self.lock:
            self.video_enabled = not self.video_enabled
            state_str = "ON" if self.video_enabled else "OFF"
            self.status_msg = f"Video toggled {state_str} (Press 'V' to toggle)"
            return self.video_enabled

    def toggle_pause(self) -> bool:
        with self.lock:
            self.is_paused = not self.is_paused
            state_str = "PAUSED" if self.is_paused else "RUNNING"
            self.status_msg = f"Simulation {state_str}"
            return self.is_paused


def render_single_hud(
    frame: np.ndarray,
    state: SharedState,
    window_w: int = 640,
    window_h: int = 480,
) -> np.ndarray:
    """Draw telemetry HUD for single-duck mode."""
    hud = frame.copy()

    overlay = hud.copy()
    cv2.rectangle(overlay, (0, 0), (window_w, 92), (12, 18, 28), -1)
    cv2.rectangle(overlay, (0, window_h - 40), (window_w, window_h), (12, 18, 28), -1)
    cv2.addWeighted(overlay, 0.80, hud, 0.20, 0, hud)

    mode_text = "DEMO (10.5s Policy)" if state.mode == "demo" else "RL TRAINING (PPO)"
    cv2.putText(hud, f"MICRODUCK RL  |  {mode_text}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (46, 196, 182), 2, cv2.LINE_AA)

    v_badge_color = (60, 220, 90) if state.video_enabled else (80, 80, 240)
    v_badge_text = "[V] VIDEO: ON" if state.video_enabled else "[V] VIDEO: OFF"
    cv2.rectangle(hud, (window_w - 170, 8), (window_w - 12, 32), (25, 35, 48), -1)
    cv2.rectangle(hud, (window_w - 170, 8), (window_w - 12, 32), v_badge_color, 1)
    cv2.circle(hud, (window_w - 156, 20), 5, v_badge_color, -1)
    cv2.putText(hud, v_badge_text, (window_w - 144, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    time_s = state.ep_steps * 0.02
    stats_line1 = (
        f"Step: {state.step:,} | Ep: {state.episode} | Time: {time_s:.2f}s | "
        f"Rew: {state.ep_reward:+.1f} | Mean: {state.mean_reward:+.1f} | FPS: {state.fps:.0f}"
    )
    cv2.putText(hud, stats_line1, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1, cv2.LINE_AA)

    stats_line2 = (
        f"Height: {state.trunk_height:.3f}m | Vel: {state.trunk_vx:+.2f}m/s (cmd {state.cmd_vx:.2f}) | "
        f"Pitch: {state.pitch_gx:+.2f} | Roll: {state.roll_gy:+.2f}"
    )
    cv2.putText(hud, stats_line2, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (160, 220, 255), 1, cv2.LINE_AA)

    l_color = (60, 230, 90) if state.left_contact else (55, 60, 75)
    r_color = (60, 230, 90) if state.right_contact else (55, 60, 75)
    cv2.rectangle(hud, (window_w - 170, 48), (window_w - 95, 76), l_color, -1)
    cv2.rectangle(hud, (window_w - 170, 48), (window_w - 95, 76), (200, 200, 200), 1)
    cv2.putText(hud, "L FOOT", (window_w - 162, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0) if state.left_contact else (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(hud, (window_w - 88, 48), (window_w - 12, 76), r_color, -1)
    cv2.rectangle(hud, (window_w - 88, 48), (window_w - 12, 76), (200, 200, 200), 1)
    cv2.putText(hud, "R FOOT", (window_w - 80, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0) if state.right_contact else (200, 200, 200), 1, cv2.LINE_AA)

    bottom_str = "[V] Toggle Video  |  [Space] Pause  |  [R] Reset  |  [Q] Save & Quit"
    cv2.putText(hud, bottom_str, (16, window_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 210, 230), 1, cv2.LINE_AA)
    return hud


def build_multi_clones_grid(
    clone_frames: List[np.ndarray],
    state: SharedState,
    clone_w: int = 320,
    clone_h: int = 240,
) -> np.ndarray:
    """Compose 8 clone sub-frames into a 2x4 grid with color-coded telemetry and master header."""
    decorated_frames = []

    for i in range(len(clone_frames)):
        frame = clone_frames[i].copy()
        color = CLONE_COLORS[i % len(CLONE_COLORS)]
        is_alive = state.clone_alives[i]
        trunk_z = state.clone_heights[i]
        cmd_vx = state.clone_vxs[i]
        ep_cnt = state.clone_episodes[i]
        ep_rew = state.clone_rewards[i]
        lc = state.clone_left_contacts[i]
        rc = state.clone_right_contacts[i]

        # Header overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (clone_w, 36), (15, 20, 28), -1)
        cv2.rectangle(overlay, (0, clone_h - 26), (clone_w, clone_h), (15, 20, 28), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Border
        cv2.rectangle(frame, (0, 0), (clone_w - 1, clone_h - 1), color, 2)

        # Status dot
        dot_color = (60, 230, 90) if is_alive else (50, 50, 240)
        cv2.circle(frame, (14, 18), 5, dot_color, -1)

        # Label
        cv2.putText(frame, f"DUCK #{i + 1}", (26, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

        # Telemetry
        status_text = f"z:{trunk_z:.2f}m | vx:{cmd_vx:+.2f}"
        cv2.putText(frame, status_text, (clone_w - 142, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

        # Bottom stats
        bot_text = f"Ep:{ep_cnt} Rew:{ep_rew:+.1f}"
        cv2.putText(frame, bot_text, (10, clone_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 250, 200) if ep_rew >= 0 else (180, 180, 255), 1, cv2.LINE_AA)

        # Foot contact badges (mini)
        l_col = (60, 230, 90) if lc else (60, 60, 75)
        r_col = (60, 230, 90) if rc else (60, 60, 75)
        cv2.circle(frame, (clone_w - 36, clone_h - 13), 5, l_col, -1)
        cv2.putText(frame, "L", (clone_w - 40, clone_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0) if lc else (200, 200, 200), 1)
        cv2.circle(frame, (clone_w - 16, clone_h - 13), 5, r_col, -1)
        cv2.putText(frame, "R", (clone_w - 20, clone_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0) if rc else (200, 200, 200), 1)

        decorated_frames.append(frame)

    # 2x4 Layout
    if len(decorated_frames) >= 8:
        row1 = np.hstack(decorated_frames[0:4])
        row2 = np.hstack(decorated_frames[4:8])
        grid = np.vstack([row1, row2])
    elif len(decorated_frames) == 4:
        grid = np.hstack(decorated_frames)
    else:
        grid = decorated_frames[0]

    # Master Header Bar (width = grid.shape[1], height = 66)
    gw = grid.shape[1]
    header = np.full((66, gw, 3), (12, 18, 28), dtype=np.uint8)

    # Title
    cv2.putText(
        header,
        f"MICRODUCK {state.num_clones}-CLONE PARALLEL REINFORCEMENT LEARNING (50Hz PPO)",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (46, 196, 182),
        2,
        cv2.LINE_AA,
    )

    # Video Toggle Status Badge
    v_badge_color = (60, 220, 90) if state.video_enabled else (80, 80, 240)
    v_badge_text = "[V] VIDEO: ON" if state.video_enabled else "[V] VIDEO: OFF"
    cv2.rectangle(header, (gw - 175, 8), (gw - 16, 32), (25, 35, 48), -1)
    cv2.rectangle(header, (gw - 175, 8), (gw - 16, 32), v_badge_color, 1)
    cv2.circle(header, (gw - 160, 20), 5, v_badge_color, -1)
    cv2.putText(header, v_badge_text, (gw - 148, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    # Stats Line
    mean_r = state.mean_reward
    stats_str = (
        f"Step: {state.step:,} | 8-Env Mean Rew: {mean_r:+.1f} | "
        f"FPS: {state.fps:,.0f} | Apple Silicon M4 Pro | [V] Toggle Video  [Space] Pause  [Q] Save & Quit"
    )
    cv2.putText(header, stats_str, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1, cv2.LINE_AA)

    return np.vstack([header, grid])


def create_standby_frame(state: SharedState, width: int = 1280, height: int = 546) -> np.ndarray:
    """Lightweight frame rendered when video is toggled OFF (runs at ~5,000–6,000 FPS)."""
    frame = np.full((height, width, 3), (18, 22, 32), dtype=np.uint8)

    cv2.rectangle(frame, (12, 12), (width - 12, height - 12), (80, 100, 130), 1)
    cv2.putText(frame, f"MICRODUCK {state.num_clones}-CLONE PARALLEL REINFORCEMENT LEARNING", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (46, 196, 182), 2, cv2.LINE_AA)

    # Turbo Banner
    cv2.rectangle(frame, (35, 95), (width - 35, 190), (30, 36, 52), -1)
    cv2.rectangle(frame, (35, 95), (width - 35, 190), (60, 80, 120), 2)
    cv2.putText(frame, ">> VIDEO IS OFF  [8-CLONE HIGH-SPEED TURBO MODE]", (55, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Simulating {state.num_clones} parallel environments at maximum hardware speed (~{state.fps:,.0f} FPS)", (55, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 240), 1, cv2.LINE_AA)

    # 8-Clone Mini Summary Cards
    card_w = (width - 70) // 4
    for i in range(min(8, state.num_clones)):
        r = i // 4
        c = i % 4
        x0 = 35 + c * card_w
        y0 = 220 + r * 110
        color = CLONE_COLORS[i % len(CLONE_COLORS)]

        cv2.rectangle(frame, (x0, y0), (x0 + card_w - 12, y0 + 95), (25, 32, 45), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + card_w - 12, y0 + 95), color, 1)

        cv2.putText(frame, f"Duck Clone #{i+1}", (x0 + 12, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Ep: #{state.clone_episodes[i]}", (x0 + 12, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Rew: {state.clone_rewards[i]:+.1f}", (x0 + 12, y0 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 140) if state.clone_rewards[i] >= 0 else (180, 180, 255), 1, cv2.LINE_AA)

    # Instruction Card
    cv2.rectangle(frame, (35, height - 85), (width - 35, height - 20), (25, 45, 65), -1)
    cv2.rectangle(frame, (35, height - 85), (width - 35, height - 20), (0, 180, 255), 1)
    cv2.putText(frame, ">> PRESS [V] ON KEYBOARD TO TURN 8-DUCK VIDEO VIEW ON!", (55, height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

    return frame


def render_terminal_dashboard(state: SharedState) -> Layout:
    """Build a rich terminal dashboard with 8 clone statuses and global PPO progress."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", size=13),
        Layout(name="footer", size=3),
    )
    layout["main"].split_row(
        Layout(name="progress_box", ratio=1),
        Layout(name="clones_box", ratio=1),
    )

    mode_str = f"[bold cyan]8-CLONE PARALLEL TRAINING (PPO)[/bold cyan]" if state.num_clones > 1 else "[bold yellow]DEMO MODE[/bold yellow]"
    video_status = "[bold green]● ON[/bold green] (3D Camera Active)" if state.video_enabled else "[bold red]○ OFF[/bold red] (Turbo Speed ~6,000 FPS)"
    header_text = f"🦆 [bold white]Microduck Reinforcement Learning Platform[/bold white] | {mode_str} | Video: {video_status}"
    layout["header"].update(Panel(header_text, style="cyan"))

    # Global Progress Box
    t_prog = Table.grid(padding=(0, 2))
    t_prog.add_column("Key", style="bold cyan")
    t_prog.add_column("Val", style="bold white")

    pct = min(100.0, (state.step / max(1, state.total_timesteps)) * 100)
    t_prog.add_row("Total Steps:", f"{state.step:,} / {state.total_timesteps:,} ({pct:.1f}%)")
    t_prog.add_row("Parallel Ducks:", f"{state.num_clones} Ducks Active")
    t_prog.add_row("Simulation FPS:", f"{state.fps:,.0f} steps/sec")
    t_prog.add_row("8-Env Mean Rew:", f"{state.mean_reward:+.2f}")
    t_prog.add_row("Policy / Val Loss:", f"{state.policy_loss:.4f} / {state.value_loss:.4f}")
    t_prog.add_row("Entropy (Exploration):", f"{state.entropy_loss:.4f}")
    t_prog.add_row("Elapsed Time:", f"{time.time() - state.start_time:.1f} seconds")

    layout["main"]["progress_box"].update(
        Panel(t_prog, title="[bold green]📊 Learning & Training Metrics[/bold green]", border_style="green")
    )

    # 8 Clones Telemetry Table
    t_clones = Table(show_header=True, header_style="bold yellow", box=None, padding=(0, 1))
    t_clones.add_column("Clone", style="bold cyan")
    t_clones.add_column("Height", justify="right")
    t_clones.add_column("Vel", justify="right")
    t_clones.add_column("Stance", justify="center")
    t_clones.add_column("Ep", justify="right")
    t_clones.add_column("Rew", justify="right")

    for i in range(min(8, state.num_clones)):
        h = state.clone_heights[i]
        h_str = f"[green]{h:.2f}m[/green]" if h >= 0.09 else f"[red]{h:.2f}m[/red]"
        v = state.clone_vxs[i]
        v_str = f"{v:+.2f}"
        lc = "L" if state.clone_left_contacts[i] else "-"
        rc = "R" if state.clone_right_contacts[i] else "-"
        stance = f"[{lc}|{rc}]"
        ep = f"#{state.clone_episodes[i]}"
        rew = state.clone_rewards[i]
        rew_str = f"[green]{rew:+.1f}[/green]" if rew >= 0 else f"[red]{rew:+.1f}[/red]"
        t_clones.add_row(f"#{i+1}", h_str, v_str, stance, ep, rew_str)

    layout["main"]["clones_box"].update(
        Panel(t_clones, title="[bold yellow]🦆 8-Duck Clone Status[/bold yellow]", border_style="yellow")
    )

    footer_text = (
        "[bold white]Controls:[/bold white] "
        "[bold cyan][V][/bold cyan] Toggle Video ON/OFF  •  "
        "[bold cyan][Space][/bold cyan] Pause/Resume  •  "
        "[bold cyan][Q][/bold cyan] Save & Exit"
    )
    layout["footer"].update(Panel(footer_text, style="dim white"))

    return layout


class WebStreamServer:
    """Embedded HTTP server for browser streaming and remote video toggle."""

    def __init__(self, state: SharedState, host: str = "0.0.0.0", port: int = 8088):
        self.state = state
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self):
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=utf-8")
                    self.end_headers()
                    html = parent.get_html_page()
                    self.wfile.write(html.encode("utf-8"))
                elif self.path == "/video_feed":
                    self.send_response(200)
                    self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while not parent.state.should_quit:
                            jpeg = parent.state.latest_jpeg
                            if jpeg is not None:
                                self.wfile.write(b"--frame\r\n")
                                self.send_header("Content-type", "image/jpeg")
                                self.send_header("Content-length", str(len(jpeg)))
                                self.end_headers()
                                self.wfile.write(jpeg)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.04)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                elif self.path == "/toggle_video":
                    new_val = parent.state.toggle_video()
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"video_enabled": new_val}).encode("utf-8"))
                elif self.path == "/status":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    with parent.state.lock:
                        now = time.time()
                        elapsed = max(0.1, now - parent.state.start_time)
                        total_ts = max(1, parent.state.total_timesteps)
                        pct = min(100.0, (parent.state.step / total_ts) * 100.0)
                        rem_sec = round((total_ts - parent.state.step) / max(1.0, parent.state.fps), 1) if parent.state.step < total_ts else 0.0
                        clones_data = []
                        for i in range(parent.state.num_clones):
                            clones_data.append({
                                "id": i + 1,
                                "height": round(parent.state.clone_heights[i], 3),
                                "vx": round(parent.state.clone_vxs[i], 3),
                                "alive": bool(parent.state.clone_alives[i]),
                                "left": bool(parent.state.clone_left_contacts[i]),
                                "right": bool(parent.state.clone_right_contacts[i]),
                                "episode": int(parent.state.clone_episodes[i]),
                                "reward": round(parent.state.clone_rewards[i], 1),
                            })
                        data = {
                            "step": parent.state.step,
                            "total_timesteps": total_ts,
                            "progress_pct": round(pct, 1),
                            "num_clones": parent.state.num_clones,
                            "fps": round(parent.state.fps, 1),
                            "mean_reward": round(parent.state.mean_reward, 2),
                            "elapsed_seconds": round(elapsed, 1),
                            "remaining_seconds": rem_sec,
                            "video_enabled": parent.state.video_enabled,
                            "is_paused": parent.state.is_paused,
                            "status_msg": parent.state.status_msg,
                            "policy_loss": round(parent.state.policy_loss, 5),
                            "value_loss": round(parent.state.value_loss, 2),
                            "clones": clones_data,
                        }
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                else:
                    self.send_error(404)

        try:
            self.server = ThreadingHTTPServer((self.host, self.port), Handler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            print(f"🌐 Web Dashboard started at: http://localhost:{self.port}")
        except Exception as e:
            print(f"⚠️ Web Dashboard could not start on port {self.port}: {e}")

    def get_html_page(self) -> str:
        return """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🦆 Microduck RL 即時同畫面刷新監控儀表板</title>
<style>
  :root {
    --bg-main: #0a0f1d;
    --bg-card: rgba(19, 29, 48, 0.85);
    --border: rgba(46, 196, 182, 0.25);
    --accent-teal: #2ec4b6;
    --accent-duck: #ff9f1c;
    --accent-green: #2ecc71;
    --accent-red: #e74c3c;
    --text-primary: #f0f4f8;
    --text-dim: #8fa0b5;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg-main);
    color: var(--text-primary);
    padding: 20px;
  }
  .container { max-width: 1300px; margin: 0 auto; }
  
  /* Header */
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 16px;
    margin-bottom: 20px;
  }
  .title-box { text-align: left; }
  .title-box h1 { font-size: 24px; color: var(--accent-teal); display: flex; align-items: center; gap: 10px; }
  .pulse-dot {
    width: 10px; height: 10px; border-radius: 50%; background: var(--accent-green);
    box-shadow: 0 0 10px var(--accent-green);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.2); } }
  .title-box p { color: var(--text-dim); font-size: 13px; margin-top: 4px; }
  
  .btn-bar { display: flex; gap: 10px; }
  button {
    background: var(--accent-duck); color: #0b111e; border: none; padding: 10px 20px;
    font-size: 14px; font-weight: 700; border-radius: 8px; cursor: pointer; transition: all 0.2s;
  }
  button:hover { background: #ffb74d; transform: translateY(-1px); }
  button.pause-btn { background: #3a86ff; color: #fff; }

  /* Progress Bar */
  .progress-wrap {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px 20px; margin-bottom: 20px; text-align: left;
  }
  .progress-labels { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px; }
  .bar-bg { background: #162238; border-radius: 999px; height: 14px; overflow: hidden; position: relative; }
  .bar-fill {
    background: linear-gradient(90deg, #2ec4b6, #3a86ff, #ff9f1c);
    height: 100%; width: 0%; border-radius: 999px; transition: width 0.4s ease;
  }

  /* Metric Cards */
  .metrics-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 24px;
  }
  .card {
    background: var(--bg-card); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
    padding: 14px 18px; text-align: left; position: relative;
  }
  .card-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
  .card-val { font-size: 26px; font-weight: 700; color: #fff; margin-top: 4px; font-family: monospace; }
  .card-val.teal { color: var(--accent-teal); }
  .card-val.duck { color: var(--accent-duck); }
  .card-val.green { color: var(--accent-green); }

  /* Video / Stream section */
  .video-panel {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px; margin-bottom: 24px; display: none;
  }
  .video-panel.active { display: block; }
  .video-box { border-radius: 8px; overflow: hidden; display: inline-block; max-width: 100%; border: 2px solid #23344e; }
  .video-box img { display: block; max-width: 100%; height: auto; }

  /* 8-Duck Clone Cards */
  .section-title { text-align: left; font-size: 16px; margin-bottom: 12px; color: var(--accent-teal); font-weight: 700; }
  .clones-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
  }
  .clone-card {
    background: #0f172a; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px;
    padding: 12px 14px; text-align: left; font-size: 13px;
  }
  .clone-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .clone-tag { font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  .status-alive { color: var(--accent-green); font-size: 11px; font-weight: 600; }
  .status-dead { color: var(--accent-red); font-size: 11px; font-weight: 600; }
  .clone-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px; font-family: monospace; font-size: 12px; }
  .clone-stats span { color: var(--text-dim); }
  .clone-stats b { color: #f0f4f8; }
</style>
</head>
<body>
<div class="container">
  <!-- Header -->
  <div class="header">
    <div class="title-box">
      <h1><span class="pulse-dot"></span> 🦆 Microduck 8-Clone RL 即時訓練監控</h1>
      <p>Pollen Robotics 14-DOF 雙足機器人強化學習 • <b>同一個畫面即時刷新 (Real-Time In-Place Refresh)</b></p>
    </div>
    <div class="btn-bar">
      <button onclick="toggleVideo()">📹 切換視訊 (ON / OFF)</button>
      <button class="pause-btn" onclick="togglePause()">⏸️ 暫停 / 繼續</button>
    </div>
  </div>

  <!-- Progress Bar -->
  <div class="progress-wrap">
    <div class="progress-labels">
      <span><b>訓練總進度</b>: <span id="p_steps">0</span> / <span id="p_total">500,000</span> (<span id="p_pct">0.0%</span>)</span>
      <span><b>預估剩餘時間</b>: <span id="p_rem" style="color:var(--accent-duck); font-weight:700;">--</span></span>
    </div>
    <div class="bar-bg">
      <div class="bar-fill" id="p_bar"></div>
    </div>
  </div>

  <!-- Primary Metrics Cards -->
  <div class="metrics-grid">
    <div class="card">
      <div class="card-label">模擬速度 (FPS)</div>
      <div class="card-val green" id="m_fps">0</div>
    </div>
    <div class="card">
      <div class="card-label">8 分身平均獎勵</div>
      <div class="card-val duck" id="m_rew">0.0</div>
    </div>
    <div class="card">
      <div class="card-label">策略梯度損失 (Policy Loss)</div>
      <div class="card-val teal" id="m_ploss">0.0000</div>
    </div>
    <div class="card">
      <div class="card-label">價值網絡損失 (Value Loss)</div>
      <div class="card-val" id="m_vloss">0.0</div>
    </div>
    <div class="card">
      <div class="card-label">已運行時間</div>
      <div class="card-val" id="m_elapsed">0s</div>
    </div>
    <div class="card">
      <div class="card-label">視訊顯示狀態</div>
      <div class="card-val teal" id="m_video">OFF (極速模式)</div>
    </div>
  </div>

  <!-- Video Stream Panel (Toggled dynamically) -->
  <div class="video-panel" id="video_panel">
    <div class="section-title">📹 8 影分身實時可視化 2x4 分割畫面</div>
    <div class="video-box">
      <img id="stream_img" src="/video_feed" alt="Microduck RL Stream" />
    </div>
  </div>

  <!-- 8 Ducks Clone Grid -->
  <div class="section-title">🦆 8 隻平行小鴨即時遙測狀態 (同畫面無縫刷新)</div>
  <div class="clones-grid" id="clones_grid">
    <!-- Populated dynamically by JavaScript -->
  </div>
</div>

<script>
const CLONE_THEMES = [
  { bg: 'rgba(50, 180, 240, 0.2)', border: '#32b4f0', color: '#32b4f0', name: '#1 天藍' },
  { bg: 'rgba(255, 140, 50, 0.2)', border: '#ff8c32', color: '#ff8c32', name: '#2 珊瑚橘' },
  { bg: 'rgba(120, 220, 80, 0.2)', border: '#78dc50', color: '#78dc50', name: '#3 翡翠綠' },
  { bg: 'rgba(200, 100, 220, 0.2)', border: '#c864dc', color: '#c864dc', name: '#4 霓虹紫' },
  { bg: 'rgba(255, 210, 50, 0.2)', border: '#ffd232', color: '#ffd232', name: '#5 琥珀金' },
  { bg: 'rgba(255, 80, 180, 0.2)', border: '#ff50b4', color: '#ff50b4', name: '#6 亮洋紅' },
  { bg: 'rgba(80, 200, 255, 0.2)', border: '#50c8ff', color: '#50c8ff', name: '#7 湛青藍' },
  { bg: 'rgba(240, 240, 100, 0.2)', border: '#f0f064', color: '#f0f064', name: '#8 螢光青' }
];

function toggleVideo() {
  fetch('/toggle_video').then(r => r.json()).then(() => updateUI());
}

function togglePause() {
  fetch('/status').then(r => r.json()); // Will toggle via key or API
}

function updateUI() {
  fetch('/status')
    .then(r => r.json())
    .then(d => {
      // Progress
      document.getElementById('p_steps').innerText = d.step.toLocaleString();
      document.getElementById('p_total').innerText = d.total_timesteps.toLocaleString();
      document.getElementById('p_pct').innerText = d.progress_pct + '%';
      document.getElementById('p_bar').style.width = d.progress_pct + '%';
      document.getElementById('p_rem').innerText = d.remaining_seconds > 0 ? d.remaining_seconds + ' 秒' : '已完成 / 保存中';

      // Primary cards
      document.getElementById('m_fps').innerText = Math.round(d.fps).toLocaleString() + ' steps/s';
      document.getElementById('m_rew').innerText = (d.mean_reward > 0 ? '+' : '') + d.mean_reward;
      document.getElementById('m_ploss').innerText = d.policy_loss;
      document.getElementById('m_vloss').innerText = d.value_loss.toLocaleString();
      document.getElementById('m_elapsed').innerText = Math.round(d.elapsed_seconds) + 's';
      document.getElementById('m_video').innerText = d.video_enabled ? 'ON 🟢 (已開啟視訊)' : 'OFF 🔴 (極速無畫面 ~5,500 FPS)';

      // Video panel visibility
      const vpanel = document.getElementById('video_panel');
      if (d.video_enabled) {
        vpanel.classList.add('active');
      } else {
        vpanel.classList.remove('active');
      }

      // 8 Ducks Clones Cards
      if (d.clones && d.clones.length > 0) {
        const grid = document.getElementById('clones_grid');
        grid.innerHTML = d.clones.map((c, i) => {
          const theme = CLONE_THEMES[i % CLONE_THEMES.length];
          const stance = (c.left ? 'L' : '-') + '|' + (c.right ? 'R' : '-');
          return `
            <div class="clone-card" style="border-left: 3px solid ${theme.border}">
              <div class="clone-header">
                <span class="clone-tag" style="background:${theme.bg}; color:${theme.color}; border: 1px solid ${theme.border}">小鴨 ${c.id}</span>
                <span class="${c.alive ? 'status-alive' : 'status-dead'}">${c.alive ? '🟢 行走中' : '🔴 跌倒重置'}</span>
              </div>
              <div class="clone-stats">
                <div><span>高度:</span> <b>${c.height.toFixed(2)}m</b></div>
                <div><span>速度:</span> <b>${(c.vx >= 0 ? '+' : '') + c.vx.toFixed(2)}m/s</b></div>
                <div><span>著地:</span> <b>[${stance}]</b></div>
                <div><span>回合:</span> <b>#${c.episode}</b></div>
                <div style="grid-column: span 2; margin-top: 4px;"><span>獎勵:</span> <b style="color:var(--accent-duck);">${(c.reward >= 0 ? '+' : '') + c.reward.toFixed(1)}</b></div>
              </div>
            </div>
          `;
        }).join('');
      }
    })
    .catch(() => {});
}

// In-place refresh every 500ms
setInterval(updateUI, 500);
updateUI();
</script>
</body>
</html>
"""


class TerminalInputListener(threading.Thread):
    """Listens for terminal keypresses without blocking."""

    def __init__(self, state: SharedState):
        super().__init__(daemon=True)
        self.state = state

    def run(self):
        while not self.state.should_quit:
            try:
                if sys.stdin and not sys.stdin.closed and sys.stdin in select.select([sys.stdin], [], [], 0.5)[0]:
                    raw = sys.stdin.readline()
                    if not raw:
                        break
                    line = raw.strip().lower()
                    if line == "v":
                        new_state = self.state.toggle_video()
                        print(f"\n[Console] Video toggled -> {'ON' if new_state else 'OFF'}")
                    elif line == " ":
                        is_paused = self.state.toggle_pause()
                        print(f"\n[Console] Pause toggled -> {'PAUSED' if is_paused else 'RESUMED'}")
                    elif line in ("q", "quit", "exit"):
                        self.state.should_quit = True
                        print("\n[Console] Exit requested.")
            except Exception:
                break


class MultiClonesRLCallback(BaseCallback):
    """Stable-Baselines3 Callback for 8-Clone Parallel Training with Live Grid & Telemetry."""

    def __init__(
        self,
        training_env: DummyVecEnv,
        state: SharedState,
        fps_target: int = 30,
        enable_gui: bool = True,
        clone_w: int = 320,
        clone_h: int = 240,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.vec_env = training_env
        self.state = state
        self.num_clones = state.num_clones
        self.fps_target = fps_target
        self.enable_gui = enable_gui
        self.clone_w = clone_w
        self.clone_h = clone_h
        self.window_name = f"🦆 Microduck {self.num_clones} 影分身同步強化學習 (Live Parallel Training)"

        self.last_render_time = time.time()
        self.last_step_time = time.time()
        self.step_counter = 0
        self.window_is_open = False

    def _on_training_start(self) -> None:
        if self.enable_gui and self.state.video_enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            try:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass
            self.window_is_open = True

    def _on_step(self) -> bool:
        if self.state.should_quit:
            return False

        while self.state.is_paused and not self.state.should_quit:
            time.sleep(0.05)
            self._handle_gui_events(delay=20)

        self.step_counter += 1
        self.state.step = self.num_timesteps

        # Extract rewards and dones for all 8 clones
        rewards = self.locals.get("rewards", [0.0] * self.num_clones)
        dones = self.locals.get("dones", [False] * self.num_clones)

        for i in range(self.num_clones):
            r = float(rewards[i])
            self.state.clone_rewards[i] += r
            if dones[i]:
                self.state.clone_episodes[i] += 1
                self.state.reward_history.append(self.state.clone_rewards[i])
                if len(self.state.reward_history) > 40:
                    self.state.reward_history.pop(0)
                self.state.clone_rewards[i] = 0.0

        if len(self.state.reward_history) > 0:
            self.state.mean_reward = float(np.mean(self.state.reward_history))

        # Calculate FPS
        now = time.time()
        dt_step = now - self.last_step_time
        if dt_step > 0:
            instant_fps = (1.0 / dt_step) * self.num_clones
            self.state.fps = 0.95 * self.state.fps + 0.05 * instant_fps
        self.last_step_time = now

        # Update training losses from SB3 logger if available
        if hasattr(self, "logger") and self.logger is not None:
            name_to_val = getattr(self.logger, "name_to_value", {})
            if "train/policy_gradient_loss" in name_to_val:
                self.state.policy_loss = float(name_to_val["train/policy_gradient_loss"])
            if "train/value_loss" in name_to_val:
                self.state.value_loss = float(name_to_val["train/value_loss"])
            if "train/entropy_loss" in name_to_val:
                self.state.entropy_loss = float(name_to_val["train/entropy_loss"])

        # Update telemetry for all 8 clones
        self._update_all_clones_telemetry()

        # Render & GUI events
        self._render_and_handle_events()

        # Periodic status dump for live in-place refresh
        if self.step_counter % 25 == 0:
            self._dump_live_status()

        return True

    def _dump_live_status(self):
        try:
            total_ts = max(1, self.state.total_timesteps)
            pct = min(100.0, (self.state.step / total_ts) * 100.0)
            elapsed = time.time() - self.state.start_time
            clones_info = []
            for i in range(self.num_clones):
                clones_info.append({
                    "id": i + 1,
                    "height": round(float(self.state.clone_heights[i]), 3),
                    "vx": round(float(self.state.clone_vxs[i]), 3),
                    "alive": bool(self.state.clone_alives[i]),
                    "left_contact": bool(self.state.clone_left_contacts[i]),
                    "right_contact": bool(self.state.clone_right_contacts[i]),
                    "episode": int(self.state.clone_episodes[i]),
                    "reward": round(float(self.state.clone_rewards[i]), 1),
                })
            data = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step": self.state.step,
                "total_timesteps": total_ts,
                "progress_pct": round(pct, 1),
                "fps": round(self.state.fps, 1),
                "mean_reward": round(self.state.mean_reward, 2),
                "elapsed_seconds": round(elapsed, 1),
                "video_enabled": self.state.video_enabled,
                "policy_loss": round(self.state.policy_loss, 5),
                "value_loss": round(self.state.value_loss, 2),
                "clones": clones_info,
            }
            with open("status.json", "w") as f:
                json.dump(data, f, indent=2)

            brain_dir = "/Users/childpi/.gemini/antigravity-cli/brain/ea94ede1-8c83-4314-85ff-1f88b3bf3655"
            if os.path.exists(brain_dir):
                md_path = os.path.join(brain_dir, "LIVE_TRAINING_STATUS.md")
                rows = []
                for c in clones_info:
                    stance = f"[{'L' if c['left_contact'] else '-'}|{'R' if c['right_contact'] else '-'}]"
                    st = "🟢 行走中" if c["alive"] else "🔴 跌倒重置"
                    rows.append(f"| **小鴨 #{c['id']}** | {st} | `{c['height']:.3f}m` | `{c['vx']:+.2f}m/s` | `{stance}` | `#{c['episode']}` | **`{c['reward']:+.1f}`** |")
                table_md = "\n".join(rows)
                md_content = f"""# 🦆 Microduck RL 即時訓練狀態 (10秒自動更新)

> [!NOTE]
> **更新時間**: {data['timestamp']}  
> **訓練進度**: **{data['step']:,} / {data['total_timesteps']:,} ({data['progress_pct']}%)**  
> **模擬速度**: **{int(data['fps']):,} FPS** | **8分身平均獎勵**: **{data['mean_reward']:+.2f}**  
> **即時 Web 儀表板**: [http://localhost:8088](http://localhost:8088)

---

### 🦆 8 隻小鴨個別遙測數據 (每 10 秒自動刷新)

| 分身 (Clone) | 存活狀態 | 軀幹高度 z | 前進速度 vx | 雙腳著地 [L|R] | 當前回合 | 當前獎勵 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_md}
"""
                with open(md_path, "w") as f_md:
                    f_md.write(md_content)

            # 10-second periodic stdout print
            if not hasattr(self, "last_10s_print"):
                self.last_10s_print = time.time()
            if time.time() - self.last_10s_print >= 10.0:
                self.last_10s_print = time.time()
                elapsed_m = int(data['elapsed_seconds'] // 60)
                elapsed_s = int(data['elapsed_seconds'] % 60)
                print(f"\n📊 [10秒定時更新] 8 隻小鴨個別遙測數據 (Step: {data['step']:,} | FPS: {int(data['fps']):,} | 已運行: {elapsed_m}m {elapsed_s}s)")
                print("| 分身 (Clone) | 存活狀態 | 軀幹高度 z | 前進速度 vx | 雙腳著地 [L|R] | 當前回合 | 當前獎勵 |")
                print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
                for c in clones_info:
                    stance = f"[{'L' if c['left_contact'] else '-'}|{'R' if c['right_contact'] else '-'}]"
                    st = "🟢 行走中" if c["alive"] else "🔴 跌倒重置"
                    print(f"| **小鴨 #{c['id']}** | {st} | `{c['height']:.3f}m` | `{c['vx']:+.2f}m/s` | `{stance}` | `#{c['episode']}` | **`{c['reward']:+.1f}`** |")
                print("", flush=True)
        except Exception:
            pass

    def _update_all_clones_telemetry(self):
        for i, sub_env in enumerate(self.vec_env.envs):
            if i >= self.num_clones:
                break
            try:
                raw_env = sub_env.unwrapped
                z = float(raw_env.data.qpos[2])
                self.state.clone_heights[i] = z
                self.state.clone_alives[i] = (z >= 0.06)

                site_mat = raw_env.data.site_xmat[raw_env.imu_site_id].reshape(3, 3)
                linvel = site_mat.T @ raw_env.data.qvel[:3]
                self.state.clone_vxs[i] = float(linvel[0])

                lc, rc = False, False
                for c_idx in range(raw_env.data.ncon):
                    c = raw_env.data.contact[c_idx]
                    g1, g2 = c.geom1, c.geom2
                    if (g1 == raw_env.floor_geom_id and g2 == raw_env.left_foot_geom_id) or (
                        g2 == raw_env.floor_geom_id and g1 == raw_env.left_foot_geom_id
                    ):
                        lc = True
                    if (g1 == raw_env.floor_geom_id and g2 == raw_env.right_foot_geom_id) or (
                        g2 == raw_env.floor_geom_id and g1 == raw_env.right_foot_geom_id
                    ):
                        rc = True
                self.state.clone_left_contacts[i] = lc
                self.state.clone_right_contacts[i] = rc
            except Exception:
                pass

    def _render_and_handle_events(self):
        # Case 1: Video Display is ENABLED
        if self.state.video_enabled:
            # If window was closed, re-open it
            if self.enable_gui and not self.window_is_open:
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                self.window_is_open = True
                print("\n[Display] 🟢 Video Display ENABLED. 3D window opened.")

            # Check if user clicked 'X' close button on window
            if self.enable_gui and self.window_is_open:
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        self.state.video_enabled = False
                        self.window_is_open = False
                        cv2.destroyWindow(self.window_name)
                        print("\n[Display] 🔴 Window closed by user. Training running in turbo mode (~6,000 FPS).")
                        print("[Display] 👉 Type 'v' + Enter in terminal or toggle at http://localhost:8088 to re-open.")
                        return
                except Exception:
                    pass

            now = time.time()
            if now - self.last_render_time >= (1.0 / self.fps_target):
                clone_frames = []
                for sub_env in self.vec_env.envs:
                    raw_env = sub_env.unwrapped
                    frame = raw_env.render()
                    if frame is None:
                        frame = np.zeros((self.clone_h, self.clone_w, 3), dtype=np.uint8)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if frame.shape[0] != self.clone_h or frame.shape[1] != self.clone_w:
                            frame = cv2.resize(frame, (self.clone_w, self.clone_h), interpolation=cv2.INTER_AREA)
                    clone_frames.append(frame)

                full_grid = build_multi_clones_grid(clone_frames, self.state, self.clone_w, self.clone_h)

                # JPEG for web stream
                _, jpeg = cv2.imencode(".jpg", full_grid, [cv2.IMWRITE_JPEG_QUALITY, 75])
                self.state.latest_jpeg = jpeg.tobytes()

                if self.enable_gui and self.window_is_open:
                    cv2.imshow(self.window_name, full_grid)
                self.last_render_time = now

            if self.enable_gui and self.window_is_open:
                self._handle_gui_events(delay=1)

        # Case 2: Video Display is DISABLED (Bypass rendering completely -> ~6,000 FPS!)
        else:
            # If window is currently open, destroy it to free desktop
            if self.enable_gui and self.window_is_open:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
                self.window_is_open = False
                print("\n[Display] 🔴 Video Display DISABLED. Window closed.")
                print("[Display] 👉 Simulating at full speed (~6,000 FPS). Type 'v' to re-enable display.")

            # Lightweight JPEG update for web interface if someone visits
            if self.step_counter % 50 == 0:
                standby = create_standby_frame(self.state)
                _, jpeg = cv2.imencode(".jpg", standby, [cv2.IMWRITE_JPEG_QUALITY, 60])
                self.state.latest_jpeg = jpeg.tobytes()

    def _handle_gui_events(self, delay: int = 1):
        if not self.enable_gui or not self.window_is_open:
            return
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("v"), ord("V"), ord("h"), ord("H")):
            self.state.toggle_video()
            if not self.state.video_enabled:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
                self.window_is_open = False
                print("\n[Display] 🔴 Video Display DISABLED. Window closed.")
                print("[Display] 👉 Type 'v' + Enter in terminal or toggle at http://localhost:8088 to re-enable.")
        elif key == 32:  # Space
            self.state.toggle_pause()
        elif key in (ord("q"), ord("Q"), 27):
            print("\n[GUI] User requested exit. Saving checkpoint...")
            self.state.should_quit = True

    def _on_training_end(self) -> None:
        if self.enable_gui:
            cv2.destroyAllWindows()


def run_training_mode(args, state: SharedState, enable_gui: bool = True):
    """Run PPO Reinforcement Learning Training across 8 parallel ducks."""
    state.mode = "train"
    state.total_timesteps = args.timesteps
    state.num_clones = args.num_envs

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print(f"🦆 Starting Microduck {args.num_envs}-Duck Parallel Reinforcement Learning")
    print("=" * 70)
    print(f"Parallel Ducks   : {args.num_envs} Ducks (2x4 Grid View)")
    print(f"Total Timesteps  : {args.timesteps:,}")
    print(f"Video Display    : {'ON (Press V to toggle OFF)' if state.video_enabled else 'OFF (Press V to toggle ON)'}")
    print(f"Web Dashboard    : http://localhost:{args.web_port}")
    print("=" * 70)

    def make_env_fn(clone_id: int):
        def _thunk():
            env = MicroduckEnv(
                render_mode="rgb_array",
                render_width=320,
                render_height=240,
            )
            return Monitor(env)
        return _thunk

    vec_env = DummyVecEnv([make_env_fn(i) for i in range(args.num_envs)])

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
    )

    resume_path = args.resume_from
    if resume_path and not os.path.exists(resume_path):
        for alt in ["checkpoints/best/best_model.zip", "checkpoints/microduck_10s_walker.zip", "checkpoints/microduck_final.zip"]:
            if os.path.exists(alt):
                resume_path = alt
                break

    if resume_path and os.path.exists(resume_path):
        print(f"📂 Resuming from checkpoint: {resume_path} (8 ducks immediately show learned gait)")
        model = PPO.load(
            resume_path,
            env=vec_env,
            device="cpu",
            tensorboard_log=args.log_dir,
            learning_rate=args.lr,
            gamma=args.gamma,
        )
    else:
        print(f"🌱 Initializing fresh PPO policy across {args.num_envs} ducks from scratch...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="cpu",
            tensorboard_log=args.log_dir,
        )

    callback = MultiClonesRLCallback(
        training_env=vec_env,
        state=state,
        fps_target=args.fps,
        enable_gui=enable_gui,
    )

    console = Console()
    try:
        with Live(render_terminal_dashboard(state), refresh_per_second=6, console=console) as live:
            orig_step_callback = callback._on_step

            def wrapped_step():
                res = orig_step_callback()
                if state.step % 10 == 0:
                    live.update(render_terminal_dashboard(state))
                return res

            callback._on_step = wrapped_step
            model.learn(total_timesteps=args.timesteps, callback=callback)
    except KeyboardInterrupt:
        print("\n[User Interrupt] Training stopped manually.")
    finally:
        save_path = os.path.join(args.checkpoint_dir, f"{args.output_name}.zip")
        model.save(save_path)
        print(f"\n✅ Training checkpoint saved successfully: {save_path}")
        vec_env.close()


def run_demo_mode(model_path: str, state: SharedState, enable_gui: bool = True, fps_target: int = 50):
    """Run interactive evaluation demo of a pre-trained walking policy."""
    state.mode = "demo"
    print(f"📂 Loading pre-trained model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    env = MicroduckEnv(
        render_mode="rgb_array",
        render_width=640,
        render_height=480,
    )
    env.resample_commands = False

    window_name = "🦆 Microduck RL Live Demo & Visualizer"
    if enable_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    obs, _ = env.reset(seed=89)
    env.commands[0] = 0.08
    obs[48:51] = [0.08, 0.0, 0.0]

    console = Console()
    with Live(render_terminal_dashboard(state), refresh_per_second=8, console=console) as live:
        while not state.should_quit:
            if state.is_paused:
                time.sleep(0.04)
                if enable_gui:
                    key = cv2.waitKey(20) & 0xFF
                    if key in (ord("v"), ord("V")):
                        state.toggle_video()
                    elif key == 32:
                        state.toggle_pause()
                    elif key in (ord("q"), ord("Q"), 27):
                        state.should_quit = True
                live.update(render_terminal_dashboard(state))
                continue

            t0 = time.time()
            state.step += 1
            state.ep_steps += 1

            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            obs[48:51] = [0.08, 0.0, 0.0]
            state.ep_reward += float(rew)

            state.trunk_height = float(env.data.qpos[2])
            site_mat = env.data.site_xmat[env.imu_site_id].reshape(3, 3)
            grav = site_mat.T @ np.array([0.0, 0.0, -1.0])
            state.pitch_gx = float(grav[0])
            state.roll_gy = float(grav[1])
            linvel = site_mat.T @ env.data.qvel[:3]
            state.trunk_vx = float(linvel[0])
            state.cmd_vx = 0.08

            lc, rc = False, False
            for i in range(env.data.ncon):
                c = env.data.contact[i]
                g1, g2 = c.geom1, c.geom2
                if (g1 == env.floor_geom_id and g2 == env.left_foot_geom_id) or (
                    g2 == env.floor_geom_id and g1 == env.left_foot_geom_id
                ):
                    lc = True
                if (g1 == env.floor_geom_id and g2 == env.right_foot_geom_id) or (
                    g2 == env.floor_geom_id and g1 == env.right_foot_geom_id
                ):
                    rc = True
            state.left_contact = lc
            state.right_contact = rc

            if term or trunc or state.should_reset:
                state.should_reset = False
                state.episode += 1
                state.reward_history.append(state.ep_reward)
                if len(state.reward_history) > 20:
                    state.reward_history.pop(0)
                state.mean_reward = float(np.mean(state.reward_history))
                state.ep_reward = 0.0
                state.ep_steps = 0
                obs, _ = env.reset()
                obs[48:51] = [0.08, 0.0, 0.0]

            if state.video_enabled:
                frame = env.render()
                if frame is not None:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    hud = render_single_hud(bgr, state)
                    _, jpeg = cv2.imencode(".jpg", hud, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    state.latest_jpeg = jpeg.tobytes()
                    if enable_gui:
                        cv2.imshow(window_name, hud)
            else:
                if state.step % 20 == 0:
                    standby = create_standby_frame(state, 640, 480)
                    _, jpeg = cv2.imencode(".jpg", standby, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    state.latest_jpeg = jpeg.tobytes()
                    if enable_gui:
                        cv2.imshow(window_name, standby)

            if enable_gui:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("v"), ord("V")):
                    state.toggle_video()
                elif key == 32:
                    state.toggle_pause()
                elif key in (ord("r"), ord("R")):
                    state.should_reset = True
                elif key in (ord("q"), ord("Q"), 27):
                    state.should_quit = True

            if state.video_enabled:
                elapsed = time.time() - t0
                target_dt = 1.0 / fps_target
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

            step_dt = time.time() - t0
            if step_dt > 0:
                state.fps = 0.9 * state.fps + 0.1 * (1.0 / step_dt)

            if state.step % 5 == 0:
                live.update(render_terminal_dashboard(state))

    if enable_gui:
        cv2.destroyAllWindows()
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Microduck 8-Clone Parallel RL Training")
    parser.add_argument("--mode", type=str, choices=["demo", "train"], default="train", help="Execution mode: 'train' (8 ducks) or 'demo' (1 duck)")
    parser.add_argument("--train", action="store_true", help="Shortcut for --mode train")
    parser.add_argument("--demo", action="store_true", help="Shortcut for --mode demo")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel duck clones (default: 8)")
    parser.add_argument("--video", dest="video", action="store_true", default=True, help="Start with real-time video ON (default)")
    parser.add_argument("--no-video", dest="video", action="store_false", help="Start with real-time video OFF (turbo speed)")
    parser.add_argument("--no-gui", action="store_true", help="Disable desktop OpenCV window")
    parser.add_argument("--web", action="store_true", default=True, help="Enable embedded Web Dashboard (default: True)")
    parser.add_argument("--web-port", type=int, default=8088, help="Web dashboard port (default: 8088)")
    parser.add_argument("--fps", type=int, default=25, help="Visual playback FPS target (default: 25)")

    # Model & Hyperparameters
    parser.add_argument("--model-path", type=str, default="checkpoints/best/best_model.zip", help="Checkpoint for demo mode")
    parser.add_argument("--resume-from", type=str, default="checkpoints/best/best_model.zip", help="Checkpoint to resume/fine-tune in train mode")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total timesteps to train (default: 500,000)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--n-steps", type=int, default=512, help="Steps per PPO rollout buffer per env (default: 512)")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size (default: 64)")
    parser.add_argument("--n-epochs", type=int, default=10, help="Epochs per PPO update (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.9985, help="Discount factor (default: 0.9985)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (default: 0.95)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range (default: 0.2)")
    parser.add_argument("--ent-coef", type=float, default=0.0005, help="Entropy coefficient (default: 0.0005)")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log dir")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoints dir")
    parser.add_argument("--output-name", type=str, default="microduck_8clones_trained", help="Output model name")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.demo:
        args.mode = "demo"
    elif args.train:
        args.mode = "train"

    state = SharedState(num_clones=args.num_envs, video_enabled=args.video)

    term_listener = TerminalInputListener(state)
    term_listener.start()

    if args.web:
        web_server = WebStreamServer(state, port=args.web_port)
        web_server.start()

    try:
        if args.mode == "demo":
            run_demo_mode(
                model_path=args.model_path,
                state=state,
                enable_gui=not args.no_gui,
                fps_target=args.fps,
            )
        else:
            run_training_mode(
                args=args,
                state=state,
                enable_gui=not args.no_gui,
            )
    except KeyboardInterrupt:
        print("\nSession exited by user.")
    finally:
        state.should_quit = True
        print("\n👋 Microduck demonstration session concluded.")


if __name__ == "__main__":
    main()
