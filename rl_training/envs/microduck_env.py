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
        0.08726646,  # neck_pitch (~ +5 deg, upright CoM)
        0.08726646,  # head_pitch (~ +5 deg, upright CoM)
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
                  55..57 (2): body [x, y] (0, 0)
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
        action_scale: float = 0.5,
        max_episode_steps: int = 1500,  # 1500 steps * 0.02s = 30.0s (enables 20s+ breakthrough!)
        render_mode: Optional[str] = None,
        render_width: int = 640,
        render_height: int = 480,
        command_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        duck_mimic_mode: bool = True,   # Biomimetic duck waddle, CPG, and head-bobbing motion
        duck_f_gait: float = 1.8,       # Duck stride frequency in Hz (1.8 Hz = ~0.55s per stride)
        duck_waddle_amp: float = 0.075, # Waddle roll amplitude in rad (~4.3 deg)
        duck_head_bob_amp: float = 0.07,# Head bobbing amplitude in rad (~4.0 deg)
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
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height
        self.duck_mimic_mode = duck_mimic_mode
        self.duck_f_gait = duck_f_gait
        self.duck_waddle_amp = duck_waddle_amp
        self.duck_head_bob_amp = duck_head_bob_amp
        self.gait_phase = 0.0

        # Cache IDs
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.left_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.left_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
        self.right_foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")

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

        # Command ranges focused on stable, sustained forward locomotion
        self.command_ranges = {
            "lin_vel_x": (0.05, 0.12),
            "lin_vel_y": (-0.02, 0.02),
            "ang_vel_yaw": (-0.15, 0.15),
        }
        if command_ranges is not None:
            self.command_ranges.update(command_ranges)

        # State tracking
        self.current_step = 0
        self.last_action = np.zeros(14, dtype=np.float32)
        self.commands = np.zeros(13, dtype=np.float32)  # 3 twist + 4 head + 6 body
        self.resample_commands = True
        self.resample_interval = 12.0  # 12 seconds per command phase (600 steps)

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
        """Sample velocity and posture commands."""
        # 10% probability of standing still
        if self.np_random.random() < 0.10:
            vx = 0.0
            vy = 0.0
            omega = 0.0
        else:
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
        # x, y, yaw are 0; z, roll, pitch small offsets
        self.commands[7:9] = 0.0
        self.commands[9] = self.np_random.uniform(-0.005, 0.005)  # z
        self.commands[10] = self.np_random.uniform(-0.03, 0.03)   # roll
        self.commands[11] = self.np_random.uniform(-0.03, 0.03)   # pitch
        self.commands[12] = 0.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Target joint position PD control
        target_qpos = DEFAULT_JOINT_POS + self.action_scale * action
        self.data.ctrl[:] = target_qpos

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # Advance biomimetic duck CPG gait phase
        cmd_vx = self.commands[0]
        if abs(cmd_vx) > 0.02:
            self.gait_phase = (2.0 * np.pi * self.duck_f_gait * self.current_step * self.dt) % (2.0 * np.pi)
        else:
            self.gait_phase = 0.0

        # Resample commands periodically if enabled
        if self.resample_commands and (self.current_step % int(self.resample_interval / self.dt) == 0):
            self._sample_commands()

        # Compute observation
        obs = self._get_obs()

        # Compute rewards and termination
        reward, reward_info = self._compute_reward(action)
        terminated = self._is_terminated()
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

        # 1. Base angular velocity (gyro) in trunk frame [3]
        if self.imu_site_id != -1:
            site_mat = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
            # Sensor angular velocity or body angvel rotated to site frame
            world_angvel = self.data.qvel[3:6]
            gyro_trunk = site_mat.T @ world_angvel
            obs[0:3] = gyro_trunk
        else:
            obs[0:3] = self.data.qvel[3:6]

        # 2. Projected gravity in trunk frame [3]
        if self.imu_site_id != -1:
            site_mat = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
            proj_gravity = site_mat.T @ np.array([0.0, 0.0, -1.0])
            obs[3:6] = proj_gravity
        else:
            obs[3:6] = np.array([0.0, 0.0, -1.0])

        # 3. Joint positions relative to default standing pose [14]
        current_joint_pos = self.data.qpos[7:21]
        obs[6:20] = current_joint_pos - DEFAULT_JOINT_POS

        # 4. Joint velocities [14]
        current_joint_vel = self.data.qvel[6:20]
        obs[20:34] = current_joint_vel

        # 5. Previous action [14]
        obs[34:48] = self.last_action

        # 6. Commands [13]
        obs[48:61] = self.commands

        return obs

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute composite RL locomotion reward."""
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
        rew_lin_vel = np.exp(-lin_vel_err / 0.03)

        # 2. Direct forward progress reward and strict backward motion penalty
        cost_backward_motion = 0.0
        cost_stand_still = 0.0
        rew_progress = 0.0
        if cmd_vx > 0.03:
            if lin_vel_trunk[0] < 0.0:
                cost_backward_motion = -16.0 * float(abs(lin_vel_trunk[0]))  # Heavy penalty for moving backwards
            elif lin_vel_trunk[0] < 0.02:
                cost_stand_still = -3.0  # Penalize stalling
            else:
                rew_progress = 3.5 * float(np.clip(lin_vel_trunk[0], 0.0, cmd_vx * 1.2))
        elif cmd_vx < -0.03:
            rew_progress = -3.0 * float(np.clip(lin_vel_trunk[0], cmd_vx * 1.2, 0.0))

        # Unwanted excessive lateral drift penalty (permits natural duck waddle sway <= 0.08 m/s)
        lateral_err = abs(lin_vel_trunk[1] - cmd_vy)
        cost_lateral_vel = 0.0
        if lateral_err > 0.08:
            cost_lateral_vel = -8.0 * float(lateral_err - 0.08)**2

        # 3. Angular velocity (yaw) tracking reward
        ang_vel_err = np.square(ang_vel_trunk[2] - cmd_omega)
        rew_ang_vel = np.exp(-ang_vel_err / 0.20)

        # 4. Foot positions, contacts, and Biomimetic Duck Gait CPG
        obs = self._get_obs()
        proj_grav_x = obs[3]  # Pitch (forward/back tilt)
        proj_grav_y = obs[4]  # Roll (lateral tilt)
        proj_grav_z = obs[5]  # Vertical axis
        joint_pos = self.data.qpos[7:21]

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

        # --- Biomimetic Duck Gait Motion Components ---
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
            # phi in [0, pi): left foot stance, right foot swing -> body rolls left (+proj_grav_y ~ +0.075)
            # phi in [pi, 2pi): right foot stance, left foot swing -> body rolls right (-proj_grav_y ~ -0.075)
            waddle_target = self.duck_waddle_amp * float(np.sin(phi))
            waddle_vel_target = self.duck_waddle_amp * 2.0 * np.pi * self.duck_f_gait * float(np.cos(phi))
            rew_duck_waddle = 2.5 * float(np.exp(-np.square(proj_grav_y - waddle_target) / 0.0035))

            # Roll rate tracking matching CPG cadence (allows waddle velocity without flat penalty)
            cost_roll_rate = -0.04 * float(np.square(ang_vel_trunk[0] - waddle_vel_target))

            # Allow natural waddle sway (|gy| <= 0.16 rad ~ 9.2 deg) without harsh penalty
            if abs(proj_grav_y) > 0.16:
                cost_roll_lean = -10.0 * float(abs(proj_grav_y) - 0.16)
            rew_roll = float(np.exp(-np.square(proj_grav_y - waddle_target) / 0.010))

            # 5b. Webbed-Foot Stepping & Ground Clearance
            if phi < np.pi:  # Right Swing, Left Stance
                swing_clearance = float(np.clip(rf_z / 0.016, 0.0, 1.2)) if not right_contact else 0.2
                stance_solid = 1.0 if left_contact else 0.0
            else:            # Left Swing, Right Stance
                swing_clearance = float(np.clip(lf_z / 0.016, 0.0, 1.2)) if not left_contact else 0.2
                stance_solid = 1.0 if right_contact else 0.0
            rew_stepping = 1.5 * swing_clearance + 1.0 * stance_solid

            # 5c. Rhythmic Head Bobbing (dual-frequency 2*phi: nods once per footstep)
            head_bob_target = 0.18 + self.duck_head_bob_amp * float(np.cos(2.0 * phi))
            actual_head_pitch = float(joint_pos[5] + joint_pos[6])
            rew_duck_head_bob = 1.8 * float(np.exp(-np.square(actual_head_pitch - head_bob_target) / 0.010))
            cost_head_wobble = -0.6 * float(joint_pos[7]**2 + joint_pos[8]**2)

            # 5d. Duck Crouch: low Center of Mass and knee flexion
            trunk_z = float(self.data.qpos[2])
            height_err = np.square(trunk_z - 0.110)
            rew_height = 1.5 * float(np.exp(-height_err / 0.005))
            cost_low_height = 0.0
            if trunk_z < 0.082:
                cost_low_height = -15.0 * float(0.082 - trunk_z)
            elif trunk_z > 0.122:
                cost_low_height = -8.0 * float(trunk_z - 0.122)

            # Knee flexion & ankle synergy reward (maintains flat foot and upright CoM while crouched)
            left_pitch_err = (joint_pos[3] - DEFAULT_JOINT_POS[3]) + (joint_pos[4] - DEFAULT_JOINT_POS[4])
            right_pitch_err = (joint_pos[12] - DEFAULT_JOINT_POS[12]) + (joint_pos[13] - DEFAULT_JOINT_POS[13])
            rew_foot_flat = 1.0 * float(np.exp(-np.square(left_pitch_err) / 0.015) * np.exp(-np.square(right_pitch_err) / 0.015))

            # Knee flexion reward (crouch walking stance)
            rew_duck_crouch = 0.8 * float(np.exp(-np.square(joint_pos[3] - 0.12) / 0.04) * np.exp(-np.square(joint_pos[12] + 0.12) / 0.04)) + rew_foot_flat

            # 5e. Outward Webbed-Foot Stance (Toe-out / 外八字)
            rew_toe_out = 0.5 * float(np.exp(-np.square(joint_pos[0] - 0.035) / 0.02) * np.exp(-np.square(joint_pos[9] + 0.035) / 0.02))
        else:
            rew_roll = float(np.exp(-np.square(proj_grav_y) / 0.012))
            if abs(proj_grav_y) > 0.08:
                cost_roll_lean = -8.0 * float(abs(proj_grav_y) - 0.08)
            cost_roll_rate = -0.10 * float(np.square(ang_vel_trunk[0]))
            rew_stepping = 0.5 if (left_contact and right_contact) else 0.0
            trunk_z = float(self.data.qpos[2])
            height_err = np.square(trunk_z - 0.12)
            rew_height = float(np.exp(-height_err / 0.005))
            cost_low_height = -15.0 * float(0.112 - trunk_z) if trunk_z < 0.112 else 0.0

        rew_upright = 0.5 * rew_pitch + 0.5 * rew_roll

        # Direct penalty for excessive forward/backward leaning (sharper to prevent pitch tipping)
        cost_forward_lean = 0.0
        if proj_grav_x < -0.07:  # Forward tilt > ~4.0 deg
            cost_forward_lean = -14.0 * float(-proj_grav_x - 0.07)

        cost_backward_lean = 0.0
        if proj_grav_x > 0.07:   # Backward tilt > ~4.0 deg
            cost_backward_lean = -14.0 * float(proj_grav_x - 0.07)

        # Overspeed penalty (prevents runaway acceleration sprinting)
        cost_overspeed = 0.0
        if cmd_vx > 0.03 and lin_vel_trunk[0] > cmd_vx + 0.08:
            cost_overspeed = -12.0 * float(lin_vel_trunk[0] - (cmd_vx + 0.08))

        # 6. Active walking endurance bonus (rewards steady, controlled continuous strides)
        trunk_z = float(self.data.qpos[2])
        rew_endurance = 0.0
        if (0.03 <= lin_vel_trunk[0] <= cmd_vx + 0.08) and abs(proj_grav_x) < 0.12 and abs(proj_grav_y) < 0.16 and trunk_z > 0.095:
            rew_endurance = 3.5  # Steady controlled forward walking

        # 7. Joint pose regularization (relaxed on knees, ankles, neck to allow duck waddle & bobbing)
        joint_weights = np.array([
            0.3, 0.4, 0.5, 0.4, 0.6,   # left: yaw, roll, pitch, knee, ankle (relaxed!)
            0.1, 0.1, 0.3, 0.3,         # neck & head (pitch relaxed for bobbing, yaw/roll stabilized)
            0.3, 0.4, 0.5, 0.4, 0.6    # right: yaw, roll, pitch, knee, ankle
        ], dtype=np.float32)
        joint_err = np.square(joint_pos - DEFAULT_JOINT_POS)
        cost_joint_drift = -0.05 * float(np.sum(joint_weights * joint_err))

        # 8. 20-Second Continuous Survival Milestone Ladder
        if (trunk_z > 0.090 and abs(proj_grav_x) < 0.14 and abs(proj_grav_y) < 0.18 and 0.02 <= lin_vel_trunk[0] <= cmd_vx + 0.10):
            if self.current_step >= 1000:    # 20.0+ 秒突破！
                rew_survival = 25.0
            elif self.current_step >= 750:   # 15.0 ~ 20.0 秒
                rew_survival = 15.0
            elif self.current_step >= 500:   # 10.0 ~ 15.0 秒
                rew_survival = 8.0
            else:                            # 0.0 ~ 10.0 秒
                rew_survival = 5.0
        elif trunk_z > 0.085:
            rew_survival = 1.0
        else:
            rew_survival = 0.0

        # 9. Smoothness & energy penalties
        action_diff = np.sum(np.square(action - self.last_action))
        cost_action_rate = -0.05 * action_diff

        torques = self.data.actuator_force[:14]
        cost_energy = -0.0003 * np.sum(np.square(torques))

        jnt_ranges = self.model.jnt_range[1:]
        viol_low = np.maximum(0.0, jnt_ranges[:, 0] - joint_pos)
        viol_high = np.maximum(0.0, joint_pos - jnt_ranges[:, 1])
        cost_joint_limits = -5.0 * np.sum(viol_low + viol_high)

        # Pitch angular velocity damping to prevent oscillatory pitching
        cost_pitch_rate = -0.25 * float(np.square(ang_vel_trunk[1]))
        cost_body_rates = -0.03 * float(np.square(ang_vel_trunk[2])) + cost_pitch_rate + cost_roll_rate

        # Total composite reward
        total_reward = (
            1.5 * rew_lin_vel
            + rew_progress
            + rew_stepping
            + rew_endurance
            + rew_survival
            + rew_duck_waddle
            + rew_duck_head_bob
            + rew_duck_crouch
            + rew_toe_out
            + cost_head_wobble
            + cost_stand_still
            + cost_backward_motion
            + cost_overspeed
            + cost_lateral_vel
            + 1.0 * rew_ang_vel
            + 1.5 * rew_upright
            + cost_forward_lean
            + cost_backward_lean
            + cost_roll_lean
            + 0.8 * rew_height
            + cost_low_height
            + cost_joint_drift
            + cost_action_rate
            + cost_energy
            + cost_joint_limits
            + cost_body_rates
        )

        reward_info = {
            "rew_lin_vel": float(rew_lin_vel),
            "rew_progress": float(rew_progress),
            "rew_stepping": float(rew_stepping),
            "rew_endurance": float(rew_endurance),
            "rew_survival": float(rew_survival),
            "rew_duck_waddle": float(rew_duck_waddle),
            "rew_duck_head_bob": float(rew_duck_head_bob),
            "rew_duck_crouch": float(rew_duck_crouch),
            "rew_toe_out": float(rew_toe_out),
            "rew_upright": float(rew_upright),
            "rew_pitch": float(rew_pitch),
            "cost_forward_lean": float(cost_forward_lean),
            "cost_roll_lean": float(cost_roll_lean),
            "rew_height": float(rew_height),
            "waddle_target": float(waddle_target),
            "gait_phase": float(self.gait_phase),
            "total_reward": float(total_reward),
        }

        return float(total_reward), reward_info

    def _is_terminated(self) -> bool:
        """Check fall or termination condition."""
        trunk_z = self.data.qpos[2]
        # Duck crouch permits lower height before calling fall (down to 0.068 m)
        if trunk_z < 0.068:
            return True

        if self.imu_site_id != -1:
            site_mat = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
            proj_gravity = site_mat.T @ np.array([0.0, 0.0, -1.0])
            # Forward tilt limit: max ~28 deg pitch (gx < -0.47)
            if proj_gravity[0] < -0.47:
                return True
            # Backward tilt limit: max ~26 deg pitch (gx > 0.44)
            if proj_gravity[0] > 0.44:
                return True
            # Lateral tilt limit: max ~28 deg roll (|gy| > 0.46)
            if abs(proj_gravity[1]) > 0.46:
                return True
            # General tilt limit: max ~39 deg (gz > -0.77)
            if proj_gravity[2] > -0.77:
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
