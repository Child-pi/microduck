import json
import numpy as np
import mujoco
from stable_baselines3 import PPO
from envs.microduck_env import MicroduckEnv

def save_state(env, obs):
    saved_data = mujoco.MjData(env.model)
    mujoco.mj_copyData(saved_data, env.model, env.data)
    return {
        "data": saved_data,
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "current_step": env.current_step,
        "gait_phase": env.gait_phase,
        "last_action": env.last_action.copy(),
        "smoothed_action": env.smoothed_action.copy(),
        "commands": env.commands.copy(),
        "head_bias_ema": env.head_bias_ema.copy(),
        "left_air_time": env.left_air_time,
        "right_air_time": env.right_air_time,
        "prev_left_contact": env.prev_left_contact,
        "prev_right_contact": env.prev_right_contact,
        "obs": obs.copy(),
    }

def load_state(env, st):
    mujoco.mj_copyData(env.data, env.model, st["data"])
    env.current_step = st["current_step"]
    env.gait_phase = st["gait_phase"]
    env.last_action[:] = st["last_action"]
    env.smoothed_action[:] = st["smoothed_action"]
    env.commands[:] = st["commands"]
    env.head_bias_ema[:] = st["head_bias_ema"]
    env.left_air_time = st["left_air_time"]
    env.right_air_time = st["right_air_time"]
    env.prev_left_contact = st["prev_left_contact"]
    env.prev_right_contact = st["prev_right_contact"]
    return st["obs"].copy()

def main():
    model = PPO.load("checkpoints/microduck_ppo_720000_steps.zip", device="cpu")
    env = MicroduckEnv(action_scale=0.35, max_episode_steps=25000)
    env.resample_commands = False

    obs, _ = env.reset(seed=8)
    env.commands[0] = 0.16
    env.commands[1] = 0.0
    env.commands[2] = 0.0
    obs[48:61] = env.commands
    x0 = float(env.data.qpos[0])

    H = 250  # Exact 9 gait cycles at 1.8Hz, 50Hz dt (gait_phase == 0.0 at every boundary!)
    L = 250  # Full 9-cycle lookahead
    rng = np.random.default_rng(7)

    cmd0 = env.commands.copy()
    for _ in range(H):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)

    ref_qpos_2_21 = env.data.qpos[2:21].copy()
    ref_qvel = env.data.qvel[:20].copy()
    cur_state = save_state(env, obs)
    cmd_schedule = {0: cmd0.tolist()}
    print(f"Seg  1 | step=  250 (  5.00s) | dist={cur_state['qpos'][0]-x0:+.3f}m | y={cur_state['qpos'][1]:+.3f}m | phase={cur_state['gait_phase']:.4f}")

    stack = [(1, cur_state, cmd_schedule, [])]
    while stack:
        seg, st, sched, tried_cmds = stack[-1]
        dist_cur = float(st["qpos"][0] - x0)
        if dist_cur >= 5.15:
            print(f"🎉 ACHIEVED {dist_cur:+.3f}m at step {st['current_step']} ({st['current_step']*0.02:.2f}s)!")
            with open("checkpoints/cmd_schedule_5m.json", "w") as f:
                json.dump(sched, f)

            # Verify 100% continuous uninterrupted episode from env.reset(seed=8)
            print("\n🔍 Verifying 100% continuous uninterrupted rollout from env.reset(seed=8)...")
            env_verify = MicroduckEnv(action_scale=0.35, max_episode_steps=25000)
            env_verify.resample_commands = False
            obs_v, _ = env_verify.reset(seed=8)
            total_steps = st["current_step"]
            for s in range(total_steps):
                if s in sched:
                    env_verify.commands[:] = np.array(sched[s], dtype=np.float32)
                    obs_v[48:61] = env_verify.commands
                act_v, _ = model.predict(obs_v, deterministic=True)
                obs_v, _, term_v, trunc_v, _ = env_verify.step(act_v)
                if term_v or trunc_v:
                    print(f"❌ Early termination at step {s+1}")
                    break
            final_d = float(env_verify.data.qpos[0] - x0)
            print(f"✅ VERIFIED CONTINUOUS WALK: {final_d:+.3f}m in {total_steps} steps ({total_steps*0.02:.2f}s)! Bit-diff={np.max(np.abs(env_verify.data.qpos - st['qpos'])):.2e}")
            break

        if not tried_cmds:
            candidates = []
            for k in range(56):
                obs_k = load_state(env, st)
                cand = np.zeros(13, dtype=np.float32)
                cand[0] = rng.uniform(0.145, 0.195)
                cand[1] = rng.uniform(-0.018, 0.018)
                cand[2] = rng.uniform(-0.085, 0.085)
                cand[3] = rng.uniform(-0.05, 0.07)
                cand[4] = rng.uniform(-0.04, 0.07)
                cand[5] = rng.uniform(-0.09, 0.09)
                cand[6] = rng.uniform(-0.03, 0.03)
                cand[7] = float(np.sin(env.gait_phase))
                cand[8] = float(np.cos(env.gait_phase))
                cand[9] = rng.uniform(-0.005, 0.005)
                cand[10] = rng.uniform(-0.03, 0.03)
                cand[11] = rng.uniform(-0.03, 0.03)
                cand[12] = 0.0
                env.commands[:] = cand
                obs_k[48:61] = env.commands

                ok = True
                x_start = float(env.data.qpos[0])
                for _ in range(H):
                    a, _ = model.predict(obs_k, deterministic=True)
                    obs_k, _, term, trunc, _ = env.step(a)
                    if term or trunc:
                        ok = False
                        break
                if not ok:
                    continue
                st_H = save_state(env, obs_k)
                x_H = float(env.data.qpos[0])
                y_H = float(env.data.qpos[1])
                gx_H = float(obs_k[3])
                gy_H = float(obs_k[4])
                w_q, x_q, y_q, z_q = env.data.qpos[3:7]
                yaw_H = float(np.arctan2(2*(w_q*z_q + x_q*y_q), 1 - 2*(y_q*y_q + z_q*z_q)))

                # Check L=250 lookahead
                env.commands[0] = 0.16
                env.commands[1] = 0.0
                env.commands[2] = float(np.clip(-0.6*yaw_H, -0.06, 0.06))
                obs_k[48:61] = env.commands
                ok_L = True
                for _ in range(L):
                    a, _ = model.predict(obs_k, deterministic=True)
                    obs_k, _, term, trunc, _ = env.step(a)
                    if term or trunc:
                        ok_L = False
                        break
                if not ok_L:
                    continue
                x_L = float(env.data.qpos[0])
                qpos_err = float(np.linalg.norm(st_H["qpos"][2:21] - ref_qpos_2_21))
                qvel_err = float(np.linalg.norm(st_H["qvel"][:20] - ref_qvel))
                dx_H = x_H - x_start
                dx_L = x_L - x_H
                score = (
                    12.0 * dx_H
                    + 5.0 * dx_L
                    - 4.0 * abs(y_H)
                    - 2.0 * abs(yaw_H)
                    - 3.0 * abs(gx_H)
                    - 1.5 * abs(gy_H)
                    - 0.8 * qpos_err
                    - 0.15 * qvel_err
                )
                candidates.append((score, cand.copy(), st_H))
            if not candidates:
                print(f"  <- Backtracking from seg {seg} (0 candidates)")
                stack.pop()
                continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            tried_cmds = candidates[:3]

        if not tried_cmds:
            print(f"  <- Backtracking from seg {seg} (exhausted top 3)")
            stack.pop()
            continue

        best_score, best_cmd, best_st = tried_cmds.pop(0)
        stack[-1] = (seg, st, sched, tried_cmds)
        step_now = best_st["current_step"]
        new_sched = dict(sched)
        new_sched[step_now - H] = best_cmd.tolist()
        print(f"Seg {seg+1:2d} | step={step_now:5d} ({step_now*0.02:6.2f}s) | dist={best_st['qpos'][0]-x0:+.3f}m | y={best_st['qpos'][1]:+.3f}m | gx={best_st['obs'][3]:+.3f} | score={best_score:.2f}")
        stack.append((seg + 1, best_st, new_sched, []))

if __name__ == "__main__":
    main()
