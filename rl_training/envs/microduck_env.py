"""Microduck Gymnasium Reinforcement Learning Environment.

Based on Pollen Robotics' Microduck robot model and microduck_rl specification.
Optimized for local training on Apple Silicon / macOS using native MuJoCo 3.
"""

from __future__ import annotations

import os
import sys
import subprocess
from typing import Any, Dict, Optional, Tuple

# Auto-configure headless MuJoCo rendering backend on Linux (Google Colab / Cloud)
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

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

# 14 Actuated joints in policy order (skipping mouth at index 9)
JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

# Nominal standing pose (STAND2 from scene_walk.xml / microduck_constants.py)
DEFAULT_JOINT_POS = np.array(
    [
        0.0,  # left_hip_yaw
        -0.08726646,  # left_hip_roll (~ -5 deg)
        -0.457924,  # left_hip_pitch (~ -26.24 deg)
        -0.004940,  # left_knee (~ -0.28 deg)
        0.452984,  # left_ankle (~ +25.95 deg)
        0.34906585,  # neck_pitch (~ +20 deg, matching STAND2 keyframe)
        0.34906585,  # head_pitch (~ +20 deg, matching STAND2 keyframe)
        0.0,  # head_yaw
        0.0,  # head_roll
        0.0,  # right_hip_yaw
        0.08726646,  # right_hip_roll (~ +5 deg)
        0.457924,  # right_hip_pitch (~ +26.24 deg)
        0.004940,  # right_knee (~ +0.28 deg)
        -0.452984,  # right_ankle (~ -25.95 deg)
    ],
    dtype=np.float32,
)

NOMINAL_ROOT_POS = np.array([0.0, 0.0, 0.12], dtype=np.float32)
NOMINAL_ROOT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class MicroduckEnv(gym.Env):
    """Microduck Biped Locomotion Environment for Gymnasium.

    Observation layout (61-D):
      0..3   (3): Base angular velocity (gyro) in trunk frame [rad/s]
      3..6   (3): Projected gravity vector in trunk frame
      6..20 (14): Joint positions minus default standing pose [rad]
     20..34 (14): Joint velocities [rad/s]
     34..48 (14): Previous action vector
     48..61 (13): Command vector:
                  48..51 (3): [v_x, v_y, omega_yaw]
                  51..55 (4): [neck_pitch, head_pitch, head_yaw, head_roll]
                  55..57 (2): [sin(phi), cos(phi)] CPG cyclic phase clock (or body x, y)
                  57     (1): body z (0)
                  58     (1): body roll (0)
                  59     (1): body pitch (0)
                  60     (1): body yaw (0)

    Action space (14-D):
      Target joint position offsets in [-1.0, 1.0].
      target_qpos = DEFAULT_JOINT_POS + action_scale * action
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        frame_skip: int = 10,  # 10 * 0.002s = 0.02s (50 Hz control)
        action_scale: float = 0.35,  # Sim2Real calibrated 0.35 rad unscale for long-horizon stability
        action_ema_alpha: float = 0.50,  # Exponential low-pass filter alpha (prevents 50Hz chatter & CoM drift)
        max_episode_steps: int = 4500,  # 4500 steps * 0.02s = 90.0s (enables 5.0m+ continuous duck walking!)
        render_mode: Optional[str] = None,
        render_width: int = 640,
        render_height: int = 480,
        command_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        duck_mimic_mode: bool = True,   # Biomimetic duck waddle, CPG, and head-bobbing motion
        duck_f_gait: float = 1.8,       # Duck stride frequency in Hz (1.8 Hz = ~0.55s per stride)
        duck_waddle_amp: float = 0.075, # Waddle roll amplitude in rad (~4.3 deg)
        duck_head_bob_amp: float = 0.07,# Head bobbing amplitude in rad (~4.0 deg)
        domain_randomization: bool = False, # Official non-accumulating DR (CoM <= 15mm, friction, encoder bias)
        **kwargs,                       # Gracefully absorb extra kwargs passed by gym.make wrappers
    ):
        super().__init__()

        if model_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(curr_dir, "..", "models", "scene_walk.xml")

        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MuJoCo model not found at {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * self.frame_skip
        self.action_scale = action_scale
        self.action_ema_alpha = action_ema_alpha
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height
        self.duck_mimic_mode = duck_mimic_mode
        self.duck_f_gait = duck_f_gait
        self.duck_waddle_amp = duck_waddle_amp
        self.duck_head_bob_amp = duck_head_bob_amp
        self.domain_randomization = domain_randomization
        self.gait_phase = 0.0

        # Cache IDs
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.left_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.head_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.left_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
        self.right_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")

        # Cache nominal physics arrays for non-accumulating Domain Randomization (pollen-robotics/microduck_rl)
        self.nominal_body_ipos = self.model.body_ipos.copy()
        self.nominal_geom_friction = self.model.geom_friction.copy()
        self.joint_qpos_bias = np.zeros(14, dtype=np.float32)

        # Official microduck_rl v8 leg-only pose kernel indices & stds (tight hip_roll=0.05 stops leg splay!)
        self.leg_joint_indices = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13], dtype=np.int32)
        self.leg_std_walking = np.array(
            [0.30, 0.05, 0.40, 0.40, 0.25, 0.30, 0.05, 0.40, 0.40, 0.25], dtype=np.float32
        )
        self.leg_std_standing = np.array(
            [0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.05, 0.10, 0.10, 0.10], dtype=np.float32
        )

        # Actuators
        self.nu = self.model.nu
        assert self.nu == 14, f"Expected 14 actuators, found {self.nu}"

        # Action & Observation Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(14,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(61,), dtype=np.float32
        )

        # Command ranges focused on brisk, sustained 5.0m+ forward duck locomotion
        self.command_ranges = {
            "lin_vel_x": (0.12, 0.22),
            "lin_vel_y": (-0.02, 0.02),
            "ang_vel_yaw": (-0.08, 0.08),
        }
        if command_ranges is not None:
            self.command_ranges.update(command_ranges)

        # State tracking
        self.current_step = 0
        self.last_action = np.zeros(14, dtype=np.float32)
        self.smoothed_action = np.zeros(14, dtype=np.float32)
        self.commands = np.zeros(13, dtype=np.float32)  # 3 twist + 4 head + 6 body
        self.resample_commands = True
        self.resample_interval = 12.0  # 12 seconds per command phase (600 steps)

        # Upstream Pollen RL tracking variables (head EMA droop filter & air-time window)
        self.head_bias_ema = np.zeros(4, dtype=np.float32)
        self.left_air_time = 0.0
        self.right_air_time = 0.0
        self.prev_left_contact = True
        self.prev_right_contact = True

        # Renderer
        self._viewer = None
        self._renderer = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.gait_phase = 0.0
        self.last_action[:] = 0.0
        self.smoothed_action[:] = 0.0
        self.head_bias_ema[:] = 0.0
        self.left_air_time = 0.0
        self.right_air_time = 0.0
        self.prev_left_contact = True
        self.prev_right_contact = True

        # Restore nominal physics parameters before applying non-accumulating DR
        self.model.body_ipos[:] = self.nominal_body_ipos
        self.model.geom_friction[:] = self.nominal_geom_friction
        self.joint_qpos_bias[:] = 0.0

        if self.domain_randomization:
            # Official non-accumulating CoM randomization (capped well within ±15mm trunk / ±10mm head)
            if self.trunk_body_id != -1:
                self.model.body_ipos[self.trunk_body_id] = (
                    self.nominal_body_ipos[self.trunk_body_id]
                    + self.np_random.uniform(-0.008, 0.008, size=3)
                )
            if self.head_body_id != -1:
                self.model.body_ipos[self.head_body_id] = (
                    self.nominal_body_ipos[self.head_body_id]
                    + self.np_random.uniform(-0.005, 0.005, size=3)
                )
            # Foot friction randomization in (0.75, 1.25)
            fric_scale = float(self.np_random.uniform(0.75, 1.25))
            for gid in (self.left_foot_geom_id, self.right_foot_geom_id, self.floor_geom_id):
                if gid != -1:
                    self.model.geom_friction[gid, 0] = self.nominal_geom_friction[gid, 0] * fric_scale
            # Joint encoder calibration bias (±0.008 rad)
            self.joint_qpos_bias[:] = self.np_random.uniform(-0.008, 0.008, size=14).astype(np.float32)

        if self.duck_mimic_mode:
            self.commands[7] = 0.0  # sin(0) = 0.0
            self.commands[8] = 1.0  # cos(0) = 1.0
        else:
            self.commands[7:9] = 0.0

        # Reset MuJoCo physics to nominal standing keyframe
        if self.stand_key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:3] = NOMINAL_ROOT_POS
            self.data.qpos[3:7] = NOMINAL_ROOT_QUAT
            self.data.qpos[7:21] = DEFAULT_JOINT_POS

        # Add small random perturbation to initial joint positions & root height
        qpos_noise = self.np_random.uniform(-0.015, 0.015, size=14)
        self.data.qpos[7:21] += qpos_noise
        self.data.qvel[:] = self.np_random.uniform(-0.02, 0.02, size=self.model.nv)

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Sample initial commands
        self._sample_commands()

        obs = self._get_obs()
        info = {
            "commands": self.commands.copy(),
            "trunk_height": float(self.data.qpos[2]),
            "gait_phase": 0.0,
        }
        return obs, info

    def _sample_commands(self):
        """Sample velocity and posture commands for steady sustained forward walking."""
        vx = self.np_random.uniform(*self.command_ranges["lin_vel_x"])
        vy = self.np_random.uniform(*self.command_ranges["lin_vel_y"])
        omega = self.np_random.uniform(*self.command_ranges["ang_vel_yaw"])

        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = omega

        # Head commands (small offsets around home pose)
        self.commands[3] = self.np_random.uniform(-0.08, 0.08)  # neck pitch
        self.commands[4] = self.np_random.uniform(-0.08, 0.08)  # head pitch
        self.commands[5] = self.np_random.uniform(-0.10, 0.10)  # head yaw
        self.commands[6] = self.np_random.uniform(-0.03, 0.03)  # head roll

        # Body commands: [x, y, z, roll, pitch, yaw]
        # In duck_mimic_mode, commands[7:9] carry [sin(phi), cos(phi)] CPG cyclic clock
        if self.duck_mimic_mode:
            self.commands[7] = float(np.sin(self.gait_phase))
            self.commands[8] = float(np.cos(self.gait_phase))
        else:
            self.commands[7:9] = 0.0
        self.commands[9] = self.np_random.uniform(-0.005, 0.005)  # z
        self.commands[10] = self.np_random.uniform(-0.03, 0.03)   # roll
        self.commands[11] = self.np_random.uniform(-0.03, 0.03)   # pitch
        self.commands[12] = 0.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raw_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.smoothed_action = (
            self.action_ema_alpha * raw_action
            + (1.0 - self.action_ema_alpha) * self.smoothed_action
        )
        action = self.smoothed_action

        # Target joint position PD control
        target_qpos = DEFAULT_JOINT_POS + self.action_scale * action
        self.data.ctrl[:] = target_qpos

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # Advance biomimetic duck CPG gait phase and cyclic clock in observation
        cmd_vx = self.commands[0]
        if abs(cmd_vx) > 0.02 and self.duck_mimic_mode:
            self.gait_phase = (2.0 * np.pi * self.duck_f_gait * self.current_step * self.dt) % (2.0 * np.pi)
            self.commands[7] = float(np.sin(self.gait_phase))
            self.commands[8] = float(np.cos(self.gait_phase))
        else:
            self.gait_phase = 0.0
            if self.duck_mimic_mode:
                self.commands[7] = 0.0
                self.commands[8] = 1.0
            else:
                self.commands[7:9] = 0.0

        # Resample commands periodically if enabled
        if self.resample_commands and (self.current_step % int(self.resample_interval / self.dt) == 0):
            self._sample_commands()

        # Compute observation ONCE per step (high-throughput fast path)
        obs = self._get_obs()

        # Compute rewards and termination (reusing precomputed obs)
        terminated = self._is_terminated(obs)
        reward, reward_info = self._compute_reward(action, obs, terminated)
        truncated = self.current_step >= self.max_episode_steps

        # Store last action
        self.last_action = action.copy()

        info = {
            **reward_info,
            "commands": self.commands.copy(),
            "trunk_height": float(self.data.qpos[2]),
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct the 61-D observation vector."""
        obs = np.zeros(61, dtype=np.float32)

        # 1 & 2. Base angular velocity (gyro) and Projected gravity in trunk frame [6]
        if self.imu_site_id != -1:
            site_mat = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
            world_angvel = self.data.qvel[3:6]
            obs[0:3] = site_mat.T @ world_angvel
            obs[3:6] = site_mat.T @ np.array([0.0, 0.0, -1.0])
        else:
            obs[0:3] = self.data.qvel[3:6]
            obs[3:6] = np.array([0.0, 0.0, -1.0])

        # 3. Joint positions relative to default standing pose [14] (+ optional encoder bias)
        current_joint_pos = self.data.qpos[7:21]
        obs[6:20] = (current_joint_pos - DEFAULT_JOINT_POS) + self.joint_qpos_bias

        # 4. Joint velocities [14]
        current_joint_vel = self.data.qvel[6:20]
        obs[20:34] = current_joint_vel

        # 5. Previous action [14]
        obs[34:48] = self.last_action

        # 6. Commands [13]
        obs[48:61] = self.commands

        return obs

    def _compute_reward(
        self,
        action: np.ndarray,
        obs: Optional[np.ndarray] = None,
        terminated: Optional[bool] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute composite RL locomotion reward synced with pollen-robotics/microduck_rl v8."""
        if obs is None:
            obs = self._get_obs()
        if terminated is None:
            terminated = self._is_terminated(obs)

        # Current base linear and angular velocities in base frame
        trunk_mat = self.data.xmat[self.trunk_body_id].reshape(3, 3)
        lin_vel_world = self.data.qvel[:3]
        lin_vel_trunk = trunk_mat.T @ lin_vel_world
        ang_vel_world = self.data.qvel[3:6]
        ang_vel_trunk = trunk_mat.T @ ang_vel_world

        cmd_vx = self.commands[0]
        cmd_vy = self.commands[1]
        cmd_omega = self.commands[2]

        # 1. Forward velocity tracking reward (sharp Gaussian centered at command)
        lin_vel_err = np.sum(np.square(lin_vel_trunk[:2] - np.array([cmd_vx, cmd_vy])))
        rew_lin_vel = float(np.exp(-lin_vel_err / 0.03))

        # 2. Direct forward progress reward and moderate backward motion penalty
        cost_backward_motion = 0.0
        cost_stand_still = 0.0
        rew_progress = 0.0
        if cmd_vx > 0.03:
            if lin_vel_trunk[0] < 0.0:
                cost_backward_motion = -16.0 * float(abs(lin_vel_trunk[0])) - 30.0 * float(lin_vel_trunk[0]**2)
            elif lin_vel_trunk[0] < 0.015:
                cost_stand_still = -2.0
            else:
                rew_progress = float(np.clip(lin_vel_trunk[0] / max(cmd_vx, 0.05), 0.0, 1.25))
        elif cmd_vx < -0.03:
            rew_progress = float(np.clip(lin_vel_trunk[0] / min(cmd_vx, -0.05), 0.0, 1.25))

        # Unwanted excessive lateral drift penalty (permits natural duck waddle sway <= 0.06 m/s)
        lateral_err = abs(lin_vel_trunk[1] - cmd_vy)
        cost_lateral_vel = 0.0
        if lateral_err > 0.06:
            cost_lateral_vel = -8.0 * float(lateral_err - 0.06)**2

        # 3. Official microduck_rl v8 Angular Velocity + Fine Yaw Tracking (eliminates yaw dead-zone!)
        ang_vel_err = float(np.square(ang_vel_trunk[2] - cmd_omega))
        rew_ang_vel = float(np.exp(-ang_vel_err / 0.50))       # Official std^2 = 0.50
        rew_yaw_fine = float(np.exp(-ang_vel_err / 0.1225))    # Official v8 std = 0.35 (std^2 = 0.1225)

        # 4. Foot positions, contacts, and Biomimetic Duck Gait CPG
        proj_grav_x = float(obs[3])  # Pitch (forward/back tilt)
        proj_grav_y = float(obs[4])  # Roll (lateral tilt)
        proj_grav_z = float(obs[5])  # Vertical axis (~ -1.0 when upright)
        joint_pos = self.data.qpos[7:21]
        is_upright = (proj_grav_z < -0.766)  # Official upright gate: tilt < 40 deg

        lf_z = float(self.data.site_xpos[self.left_foot_site_id][2])
        rf_z = float(self.data.site_xpos[self.right_foot_site_id][2])

        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == self.floor_geom_id and g2 == self.left_foot_geom_id) or (g2 == self.floor_geom_id and g1 == self.left_foot_geom_id):
                left_contact = True
            if (g1 == self.floor_geom_id and g2 == self.right_foot_geom_id) or (g2 == self.floor_geom_id and g1 == self.right_foot_geom_id):
                right_contact = True

        rew_pitch = float(np.exp(-np.square(proj_grav_x) / 0.012))

        # --- Biomimetic Duck Gait Motion Components (normalized in [0, 1]) ---
        rew_duck_waddle = 0.0
        rew_duck_head_bob = 0.0
        rew_duck_crouch = 0.0
        rew_toe_out = 0.0
        cost_head_wobble = 0.0
        cost_roll_lean = 0.0
        cost_roll_rate = 0.0
        waddle_target = 0.0

        if self.duck_mimic_mode and abs(cmd_vx) > 0.02:
            phi = self.gait_phase
            # 5a. CPG Harmonic Duck Waddle:
            waddle_target = self.duck_waddle_amp * float(np.sin(phi))
            waddle_vel_target = self.duck_waddle_amp * 2.0 * np.pi * self.duck_f_gait * float(np.cos(phi))
            rew_duck_waddle = float(np.exp(-np.square(proj_grav_y - waddle_target) / 0.0045))

            # Roll rate tracking matching CPG cadence
            cost_roll_rate = -0.08 * float(np.square(ang_vel_trunk[0] - waddle_vel_target))

            # Allow natural waddle sway (|gy| <= 0.15 rad ~ 8.6 deg) without harsh penalty
            if abs(proj_grav_y) > 0.15:
                excess_roll = abs(proj_grav_y) - 0.15
                cost_roll_lean = -15.0 * float(excess_roll) - 30.0 * float(excess_roll**2)
            rew_roll = float(np.exp(-np.square(proj_grav_y - waddle_target) / 0.010))

            # 5b. Webbed-Foot Stepping & Ground Clearance (target 0.02 m = 2 cm per official recipe)
            if phi < np.pi:  # Right Swing, Left Stance
                swing_clearance = float(np.clip(rf_z / 0.020, 0.0, 1.0)) if not right_contact else 0.15
                stance_solid = 1.0 if left_contact else 0.0
            else:            # Left Swing, Right Stance
                swing_clearance = float(np.clip(lf_z / 0.020, 0.0, 1.0)) if not left_contact else 0.15
                stance_solid = 1.0 if right_contact else 0.0
            rew_stepping = 0.6 * swing_clearance + 0.4 * stance_solid

            # 5c. Rhythmic Head Bobbing (dual-frequency 2*phi: nods once per footstep)
            # Centered around nominal walking head posture (~0.58 rad = 33 deg)
            head_bob_target = 0.58 + self.duck_head_bob_amp * float(np.cos(2.0 * phi))
            actual_head_pitch = float(joint_pos[5] + joint_pos[6])
            rew_duck_head_bob = float(np.exp(-np.square(actual_head_pitch - head_bob_target) / 0.025))
            cost_head_wobble = -0.3 * float(joint_pos[7]**2 + joint_pos[8]**2)

            # 5d. Duck Crouch: low Center of Mass and knee flexion
            trunk_z = float(self.data.qpos[2])
            height_err = np.square(trunk_z - 0.110)
            rew_height = float(np.exp(-height_err / 0.005))
            cost_low_height = 0.0
            if trunk_z < 0.082:
                cost_low_height = -10.0 * float(0.082 - trunk_z)
            elif trunk_z > 0.125:
                cost_low_height = -5.0 * float(trunk_z - 0.125)

            # Knee flexion & ankle synergy reward (maintains flat foot and upright CoM while crouched)
            left_pitch_err = (joint_pos[3] - DEFAULT_JOINT_POS[3]) + (joint_pos[4] - DEFAULT_JOINT_POS[4])
            right_pitch_err = (joint_pos[12] - DEFAULT_JOINT_POS[12]) + (joint_pos[13] - DEFAULT_JOINT_POS[13])
            rew_foot_flat = float(np.exp(-np.square(left_pitch_err) / 0.015) * np.exp(-np.square(right_pitch_err) / 0.015))
            # Symmetrical knee crouch target (+0.10 rad left, -0.10 rad right relative to default)
            rew_duck_crouch = 0.5 * float(np.exp(-np.square(joint_pos[3] - 0.10) / 0.03) * np.exp(-np.square(joint_pos[12] + 0.10) / 0.03)) + 0.5 * rew_foot_flat

            # 5e. Outward Webbed-Foot Stance
            rew_toe_out = float(np.exp(-np.square(joint_pos[0] - 0.035) / 0.02) * np.exp(-np.square(joint_pos[9] + 0.035) / 0.02))
        else:
            rew_roll = float(np.exp(-np.square(proj_grav_y) / 0.012))
            if abs(proj_grav_y) > 0.08:
                cost_roll_lean = -6.0 * float(abs(proj_grav_y) - 0.08)
            cost_roll_rate = -0.08 * float(np.square(ang_vel_trunk[0]))
            rew_stepping = 0.5 if (left_contact and right_contact) else 0.0
            trunk_z = float(self.data.qpos[2])
            height_err = np.square(trunk_z - 0.116)  # Official nominal_z = 0.116m
            rew_height = float(np.exp(-height_err / 0.005))
            cost_low_height = -10.0 * float(0.105 - trunk_z) if trunk_z < 0.105 else 0.0

        # --- Official Upstream Pollen RL v8 Components (Fixed: No Double-Multiplication!) ---
        # 1. Official Air-Time Window [0.125, 0.300]s (normalized base 1.0, gated on upright tilt < 40 deg)
        rew_air_time = 0.0
        if not left_contact:
            self.left_air_time += self.dt
        else:
            if not self.prev_left_contact:  # Touchdown event
                if 0.125 <= self.left_air_time <= 0.300 and abs(cmd_vx) > 0.02 and is_upright:
                    rew_air_time += 1.0
                self.left_air_time = 0.0

        if not right_contact:
            self.right_air_time += self.dt
        else:
            if not self.prev_right_contact:  # Touchdown event
                if 0.125 <= self.right_air_time <= 0.300 and abs(cmd_vx) > 0.02 and is_upright:
                    rew_air_time += 1.0
                self.right_air_time = 0.0

        self.prev_left_contact = left_contact
        self.prev_right_contact = right_contact

        # 2. Official 1.0s EMA Head-Pose Bias Penalty (gated on upright & trunk_z > 0.09m)
        neck_head_indices = [5, 6, 7, 8]  # neck_pitch, head_pitch, head_yaw, head_roll
        neck_head_pos = joint_pos[neck_head_indices]
        neck_head_default = DEFAULT_JOINT_POS[neck_head_indices]
        neck_head_cmd = self.commands[3:7]
        head_err = (neck_head_pos - neck_head_default) - neck_head_cmd
        alpha_ema = min(1.0, float(self.dt) / 1.0)
        self.head_bias_ema = (1.0 - alpha_ema) * self.head_bias_ema + alpha_ema * head_err
        cost_head_bias = -0.6 * float(np.mean(np.abs(self.head_bias_ema))) if (is_upright and trunk_z > 0.09) else 0.0

        # 3. Official Foot Clearance Target (0.02 m / 2 cm) & Foot Slip (-0.1)
        cost_foot_clearance = 0.0
        if not left_contact and lf_z < 0.020 and abs(cmd_vx) > 0.02:
            cost_foot_clearance -= 0.5 * float(0.020 - lf_z) / 0.020
        if not right_contact and rf_z < 0.020 and abs(cmd_vx) > 0.02:
            cost_foot_clearance -= 0.5 * float(0.020 - rf_z) / 0.020

        # 4. Official Upright Stance Reward (normalized base in [0, 1], std^2 = 0.05)
        tilt_sq = float(np.square(proj_grav_x) + np.square(proj_grav_y))
        rew_upright = float(np.exp(-tilt_sq / 0.05))

        # 5. Official Exponential Leg-Only Pose Reward (tight hip_roll std = 0.05 stops leg splay!)
        leg_stds = self.leg_std_walking if abs(cmd_vx) > 0.02 else self.leg_std_standing
        leg_q_err = (joint_pos[self.leg_joint_indices] - DEFAULT_JOINT_POS[self.leg_joint_indices]) / leg_stds
        rew_pose = float(np.exp(-np.mean(np.square(leg_q_err))))

        # Direct penalty for excessive forward/backward leaning
        # Note: In MuJoCo trunk frame, forward pitch (nose down toward +X) has proj_grav_x > 0,
        # while backward pitch (tail down) has proj_grav_x < 0.
        cost_forward_lean = 0.0
        if proj_grav_x > 0.16:   # Forward tilt > ~9.2 deg (prevents forward nose-dive)
            excess_f = proj_grav_x - 0.16
            cost_forward_lean = -15.0 * float(excess_f) - 35.0 * float(excess_f**2)

        cost_backward_lean = 0.0
        if proj_grav_x < -0.18:  # Backward tilt > ~10.4 deg (prevents backward flip)
            excess_b = -proj_grav_x - 0.18
            cost_backward_lean = -15.0 * float(excess_b) - 35.0 * float(excess_b**2)

        # Overspeed quadratic speed corridor (permits brisk 5m walking up to cmd_vx + 0.16 m/s)
        cost_overspeed = 0.0
        if cmd_vx > 0.03 and lin_vel_trunk[0] > cmd_vx + 0.16:
            excess_v = lin_vel_trunk[0] - (cmd_vx + 0.16)
            cost_overspeed = -15.0 * float(excess_v) - 40.0 * float(excess_v**2)

        # 6. Active walking endurance bonus (normalized in [0, 1] for steady controlled strides)
        rew_endurance = 0.0
        if (0.04 <= lin_vel_trunk[0] <= cmd_vx + 0.16) and abs(proj_grav_x) < 0.16 and abs(proj_grav_y) < 0.18 and trunk_z > 0.092:
            rew_endurance = 1.0

        # 6b. World-frame Straight-Line (+X) Velocity & 5-Meter Distance Milestone Ladder
        vx_world = float(self.data.qvel[0])
        dist_x = float(self.data.qpos[0])
        y_pos = float(self.data.qpos[1])
        w_q, x_q, y_q, z_q = self.data.qpos[3:7]
        yaw_world = float(np.arctan2(2.0 * (w_q * z_q + x_q * y_q), 1.0 - 2.0 * (y_q * y_q + z_q * z_q)))
        rew_straight_line = float(np.exp(-np.square(yaw_world) / 0.06) * np.exp(-np.square(y_pos) / 0.08))
        rew_world_vx = float(np.clip(vx_world / 0.16, -0.5, 2.0))
        rew_distance_5m = 0.0
        if is_upright and vx_world > 0.03 and trunk_z > 0.088:
            rew_distance_5m = 0.8 * float(np.clip(dist_x, 0.0, 6.0))
            if dist_x >= 5.0:
                rew_distance_5m += 10.0
            elif dist_x >= 3.5:
                rew_distance_5m += 6.0
            elif dist_x >= 2.0:
                rew_distance_5m += 4.0
            elif dist_x >= 1.0:
                rew_distance_5m += 2.0
            elif dist_x >= 0.5:
                rew_distance_5m += 1.0

        # 7. Bounded Survival Milestone Bonus (steep incentive gradient for 20s milestone)
        if (trunk_z > 0.090 and abs(proj_grav_x) < 0.14 and abs(proj_grav_y) < 0.17 and 0.02 <= lin_vel_trunk[0] <= cmd_vx + 0.10):
            if self.current_step >= 1000:    # 20.0+ 秒突破！
                rew_survival = 5.0
            elif self.current_step >= 750:   # 15.0 ~ 20.0 秒
                rew_survival = 3.5
            elif self.current_step >= 500:   # 10.0 ~ 15.0 秒
                rew_survival = 2.5
            elif self.current_step >= 250:   # 5.0 ~ 10.0 秒
                rew_survival = 1.5
            else:                            # 0.0 ~ 5.0 秒
                rew_survival = 0.8
        elif trunk_z > 0.085:
            rew_survival = 0.2
        else:
            rew_survival = 0.0

        # 8. Smoothness & energy penalties (scaled by 0.1 when fallen per official microduck_rl)
        action_diff = float(np.sum(np.square(action - self.last_action)))
        action_rate_scale = 1.0 if is_upright else 0.1
        cost_action_rate = -0.08 * action_rate_scale * action_diff

        torques = self.data.actuator_force[:14]
        cost_energy = -0.0001 * float(np.sum(np.square(torques)))

        jnt_ranges = self.model.jnt_range[1:]
        viol_low = np.maximum(0.0, jnt_ranges[:, 0] - joint_pos)
        viol_high = np.maximum(0.0, joint_pos - jnt_ranges[:, 1])
        cost_joint_limits = -2.0 * float(np.sum(viol_low + viol_high))

        # Continuous healthy alive bonus & catastrophic termination penalty for 20s endurance
        rew_alive = 2.5 if (not terminated and trunk_z > 0.080 and is_upright) else 0.0
        cost_termination = -25.0 if terminated else 0.0
        cost_lin_vel_z = -1.5 * float(np.square(lin_vel_trunk[2]))
        cost_ang_vel_xy = -0.05 * float(np.square(ang_vel_trunk[0]) + np.square(ang_vel_trunk[1]))

        # Composite Total Reward (all official v8 + biomimetic terms properly connected & bounded!)
        total_reward = (
            rew_alive                   # Continuous alive bonus (+2.5/step, +2500 for 20s!)
            + 2.0 * rew_lin_vel         # Official linear velocity tracking (w=2.0)
            + 1.5 * rew_progress        # Forward progress incentive
            + 2.0 * rew_upright         # Official upright stance (w=2.0, fixed single multiply!)
            + 1.5 * rew_pose            # Official leg-only anti-splay pose kernel (hip_roll std=0.05)
            + 1.5 * rew_ang_vel         # Coarse angular velocity tracking
            + 2.0 * rew_yaw_fine        # Official v8 fine yaw tracking (std=0.35, eliminates yaw drift!)
            + 3.0 * rew_air_time        # Official air-time [0.125, 0.300]s (w=3.0, fixed single multiply!)
            + 2.0 * rew_stepping        # Phase contact & swing clearance
            + 2.2 * rew_duck_waddle     # Biomimetic CPG lateral waddle (boosted for realistic duck sway)
            + 1.6 * rew_duck_head_bob   # Biomimetic rhythmic head bobbing (boosted for step-synchronized nods)
            + 1.2 * rew_duck_crouch     # Low CoM crouch & flat sole (boosted for bipedal duck stance)
            + 1.2 * rew_endurance       # Steady speed endurance
            + 1.0 * rew_survival        # Bounded survival milestone ladder
            + 2.5 * rew_world_vx        # World +X forward velocity push for 5m goal
            + 2.0 * rew_straight_line   # World +X straight-line yaw & centerline corridor
            + 1.0 * rew_distance_5m     # 5-meter cumulative distance milestone ladder
            + cost_foot_clearance       # 2cm swing foot clearance penalty
            + cost_head_bias            # Official 1.0s EMA head droop penalty
            + cost_forward_lean         # Anti-tipping forward pitch guard
            + cost_backward_lean        # Anti-tipping backward pitch guard
            + cost_roll_lean            # Excessive roll guard
            + cost_overspeed            # Anti-surging speed guard
            + cost_backward_motion      # Anti-backward slip
            + cost_stand_still          # Anti-freezing
            + cost_lateral_vel          # Lateral drift guard
            + cost_action_rate          # Smoothness (gated when fallen)
            + cost_ang_vel_xy           # Roll/pitch rate damping
            + cost_lin_vel_z            # Vertical bounce damping
            + cost_joint_limits         # Soft joint limit penalty
            + cost_energy               # Actuator torque efficiency
            + cost_termination          # Fall termination penalty
        )

        reward_info = {
            "rew_alive": float(rew_alive),
            "rew_lin_vel": float(rew_lin_vel),
            "rew_progress": float(rew_progress),
            "rew_pose": float(rew_pose),
            "rew_yaw_fine": float(rew_yaw_fine),
            "rew_stepping": float(rew_stepping),
            "rew_air_time": float(rew_air_time),
            "rew_endurance": float(rew_endurance),
            "rew_survival": float(rew_survival),
            "rew_duck_waddle": float(rew_duck_waddle),
            "rew_duck_head_bob": float(rew_duck_head_bob),
            "rew_duck_crouch": float(rew_duck_crouch),
            "rew_toe_out": float(rew_toe_out),
            "cost_head_bias": float(cost_head_bias),
            "cost_foot_clearance": float(cost_foot_clearance),
            "rew_upright": float(rew_upright),
            "cost_forward_lean": float(cost_forward_lean),
            "cost_roll_lean": float(cost_roll_lean),
            "rew_height": float(rew_height),
            "waddle_target": float(waddle_target),
            "gait_phase": float(self.gait_phase),
            "total_reward": float(total_reward),
        }

        return float(total_reward), reward_info

    def _is_terminated(self, obs: Optional[np.ndarray] = None) -> bool:
        """Check fall, NaN physics instability, or termination condition."""
        # Official microduck_rl NaN/Inf physics state guard
        if (
            np.isnan(self.data.qpos).any()
            or np.isnan(self.data.qvel).any()
            or np.isinf(self.data.qpos).any()
            or np.isinf(self.data.qvel).any()
        ):
            return True

        trunk_z = float(self.data.qpos[2])
        # Ground impact termination: trunk drops below 0.058 m (nominal is ~0.11-0.12 m)
        if trunk_z < 0.058:
            return True

        if obs is not None:
            proj_gravity = obs[3:6]
        elif self.imu_site_id != -1:
            site_mat = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
            proj_gravity = site_mat.T @ np.array([0.0, 0.0, -1.0])
        else:
            return False

        # Calibrated tilt limits: permits dynamic stepping pitch up to ~42 deg
        # and roll waddle sway up to ~38 deg without false-positive cutoffs
        if proj_gravity[0] < -0.67:  # Extreme forward pitch (> 42.0 deg)
            return True
        if proj_gravity[0] > 0.62:   # Extreme backward pitch (> 38.3 deg)
            return True
        if abs(proj_gravity[1]) > 0.62:  # Extreme lateral roll (> 38.3 deg)
            return True
        if proj_gravity[2] > -0.60:  # Overturned / upside down (> 53.1 deg total tilt)
            return True

        return False

    def render(self):
        """Render frame."""
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
            cam_names = [self.model.camera(i).name for i in range(self.model.ncam)]
            cam = "tracking" if "tracking" in cam_names else ("head_camera" if "head_camera" in cam_names else -1)
            self._renderer.update_scene(self.data, camera=cam)
            return self._renderer.render()
        elif self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            del self._renderer
            self._renderer = None
