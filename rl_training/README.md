# 🦆 Microduck Reinforcement Learning: 20s+ Biomimetic Duck Locomotion

[![Open In Colab (Quick Training)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Child-pi/microduck/blob/rl-locomotion/Microduck_Colab_Training.ipynb)
[![Open In Colab (Git Workflow)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Child-pi/microduck/blob/rl-locomotion/Microduck_RL_Colab.ipynb)

This directory contains the reinforcement learning (PPO) training environment, simulation assets, pre-trained policy checkpoints, evaluation tools, and Google Colab integration for the **Pollen Robotics Microduck** 14-DOF bipedal robot.

---

## 🌟 Highlights & Duck Mimicry Features

- **🦆 Biomimetic Duck Motion (模仿真實鴨子動作)**:
  - **Duck Waddle (左右搖擺步態)**: Rhythmic alternating lateral pelvic sway ($g_y$) synchronized with alternating foot contacts.
  - **Duck Head Bobbing (前後節奏點頭)**: Rhythmic forward/backward neck & head pitch nodding in cadence with the gait, keeping the beak looking forward.
  - **Low CoM Webbed-Foot Stance (穩固微蹲步姿)**: Compliant ankle and knee stance preventing forward tipping or falling backward.
- **🔬 Upstream Official Pollen Robotics Insights (學習官方高階經驗)**:
  - **Head Mass Constraint & 1.0s EMA Filter**: Microduck 的頭部重達 280g（佔全身 800g 的 38%），在行走時必然會劇烈晃動。若強加瞬間俯仰角懲罰會導致機器人停止行走；官方採用 **1 秒指數移動平均 (EMA) 濾波器** (`head_pose_bias`)，僅懲罰重力造成的 DC 垂頭（~15°），而不懲罰動態晃動與鴨子點頭。
  - **Air-Time Window**: 設定精準單腳騰空時間窗口 $[0.125, 0.300]\text{ s}$（50 Hz 下 6~15 控制步），步態頻率達到約 $5.0\text{ steps/s}$，徹底消除拖腳與小碎步。
  - **Foot Clearance**: 設定騰空腳抬起目標高度 $0.02\text{ m}$ (20 mm)，大幅減少腳尖卡地跌倒。
  - **Upright Tolerance**: 將軀幹直立常態標準差校準為 $\sigma = \sqrt{0.05} \approx 0.224\text{ rad}$，抑制持續前傾加速。
- **⏱️ Continuous Training Loop (直到跨越 20 秒)**:
  - Automated iterative training and evaluation loop until achieving continuous standing/walking $\ge 20.0$ seconds ($1,000+$ steps).
  - Exponential 20-second survival milestone reward ladder (+25.0 per step above 1,000 steps).
  - Planning horizon $\gamma = 0.9991$ ($1,111$ steps = $22.2$ seconds).
- **🎥 20s HUD Video Player**: In-notebook HTML5 video playback with time, waddle roll, head-bob pitch, height, velocity, and foot contact indicators.
- **Sim2Real ONNX Exporter**: Complies with Pollen Robotics `robotd` interface (`exported/microduck_policy.onnx`).
- **One-Click GitHub Push-Back**: Automatically commits and pushes 20s breakthrough checkpoints and video back to `Child-pi/microduck-private` on branch `rl-locomotion`.

---

## ☁️ Google Colab Cloud Training & Automatic GitHub Sync

Open either notebook directly in Google Colab:
- 🚀 **[Microduck_Colab_Training.ipynb](Microduck_Colab_Training.ipynb)** (1-Click Standalone Training & Telemetry Video, Auto-downloads package)
- 🔄 **[Microduck_RL_Colab.ipynb](Microduck_RL_Colab.ipynb)** (Full Repo Clone, Continuous 20s+ Training & Git Push-Back)

### Automated Workflow in Colab:
1. **Auto-Download from GitHub**: Clones `https://github.com/Child-pi/microduck` (`rl-locomotion` branch) directly into Colab.
2. **Headless Environment Setup**: Configures GPU hardware rendering (EGL) or CPU software rendering (OSMesa).
3. **Continuous Biomimetic Training**: Iteratively trains 8 duck clones until crossing **20.0 seconds** ($\ge 1,000$ steps).
4. **20s HUD Video Playback**: Evaluates the breakthrough model and plays the 20-second walking video directly in the notebook cell.
5. **Sim2Real ONNX Export**: Exports the 20-second policy to `exported/microduck_20s_champion.onnx`.
6. **Push Back to GitHub**: Automatically commits and pushes `microduck_20s_duck_champion.zip`, ONNX model, and telemetry video back to GitHub!

---

## 🚀 Quick Start (Local macOS / Linux)

### 1. Requirements
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_colab.txt
```

### 2. Run Pre-Trained Walking Demonstration
```bash
./run_demo.sh
```

### 3. Run 8-Duck Parallel RL Training
```bash
./run_training.sh --num-envs 8 --timesteps 500000
```
- Press **`V`** in the window or terminal to toggle real-time video rendering ON/OFF.
- When OFF, training accelerates to **~5,000+ FPS**.
