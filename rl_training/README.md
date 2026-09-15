# 🦆 Microduck Reinforcement Learning: 20s+ Biomimetic Duck Locomotion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Child-pi/microduck/blob/rl-locomotion/Microduck_RL_Colab.ipynb)

This directory contains the reinforcement learning (PPO) training environment, simulation assets, pre-trained policy checkpoints, evaluation tools, and Google Colab integration for the **Pollen Robotics Microduck** 14-DOF bipedal robot.

---

## 🌟 Highlights & Duck Mimicry Features

- **🦆 Biomimetic Duck Motion (模仿真實鴨子動作)**:
  - **Duck Waddle (左右搖擺步態)**: Rhythmic alternating lateral pelvic sway ($g_y$) synchronized with alternating foot contacts.
  - **Duck Head Bobbing (前後節奏點頭)**: Rhythmic forward/backward neck & head pitch nodding in cadence with the gait, keeping the beak looking forward.
  - **Low CoM Webbed-Foot Stance (穩固微蹲步姿)**: Compliant ankle and knee stance preventing forward tipping or falling backward.
- **⏱️ Continuous Training Loop (直到跨越 20 秒)**:
  - Automated iterative training and evaluation loop until achieving continuous standing/walking $\ge 20.0$ seconds ($1,000+$ steps).
  - Exponential 20-second survival milestone reward ladder (+25.0 per step above 1,000 steps).
  - Planning horizon $\gamma = 0.9991$ ($1,111$ steps = $22.2$ seconds).
- **🎥 20s HUD Video Player**: In-notebook HTML5 video playback with time, waddle roll, head-bob pitch, height, velocity, and foot contact indicators.
- **Sim2Real ONNX Exporter**: Complies with Pollen Robotics `robotd` interface (`exported/microduck_20s_champion.onnx`).
- **One-Click GitHub Push-Back**: Automatically commits and pushes 20s breakthrough checkpoints and video back to `Child-pi/microduck` on branch `rl-locomotion`.

---

## ☁️ Google Colab Cloud Training & Automatic GitHub Sync

Open the notebook directly in Google Colab:
👉 **[Microduck_RL_Colab.ipynb](Microduck_RL_Colab.ipynb)**

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
