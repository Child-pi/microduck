# 🦆 Microduck Reinforcement Learning & Bipedal Locomotion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Child-pi/microduck/blob/rl-locomotion/Microduck_RL_Colab.ipynb)

This directory contains the reinforcement learning (PPO) training environment, simulation assets, pre-trained policy checkpoints, evaluation tools, and Google Colab integration for the **Pollen Robotics Microduck** 14-DOF bipedal robot.

---

## 🌟 Highlights

- **MuJoCo 3.0 Physics & Gymnasium**: Vectorized biped simulation with realistic ground contact dynamics and domain randomization.
- **PPO Locomotion Policy**: 61-D observation vector mapped to 14-D joint target positions at a 50 Hz control rate.
- **10s+ Champion Checkpoints**: Pre-trained walking policies (`checkpoints/best/best_model.zip`, `checkpoints/microduck_8clones_trained.zip`).
- **Real-Time Dynamic Visualizer & Web HUD**: Native 3D viewer + Web dashboard on port 8088 with on-the-fly video ON/OFF toggling.
- **Sim2Real ONNX Exporter**: Complies with Pollen Robotics `robotd` interface (`exported/microduck_10s_champion.onnx`).
- **Google Colab Cloud Training & GitHub Sync**: Fully automated cloud training workflow with one-click GitHub push-back!

---

## ☁️ Google Colab Cloud Training & Automatic GitHub Sync

Open the notebook directly in Google Colab:
👉 **[Microduck_RL_Colab.ipynb](Microduck_RL_Colab.ipynb)**

### Automated 6-Step Workflow in Colab:
1. **Auto-Download from GitHub**: Clones `https://github.com/Child-pi/microduck` (`rl-locomotion` branch) directly into Colab.
2. **Headless Environment Setup**: Configures GPU hardware rendering (EGL) or CPU software rendering (OSMesa).
3. **8-Clone Parallel Training**: Trains 8 duck clones concurrently with PPO (100k ~ 1M+ steps).
4. **Evaluation & In-Notebook HUD Video**: Evaluates the newly trained model and plays the walking video with real-time gait telemetry right inside the notebook cell.
5. **Sim2Real ONNX Export**: Exports the policy for physical robot deployment.
6. **Push Back to GitHub**: Automatically commits and pushes the new checkpoint (`checkpoints/colab_trained_model.zip`), ONNX model, and telemetry status back to `Child-pi/microduck` on branch `rl-locomotion`!

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
