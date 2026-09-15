# 🦆 Microduck Reinforcement Learning & Bipedal Locomotion

This directory contains the reinforcement learning (PPO) training environment, simulation assets, pre-trained policy checkpoints, evaluation tools, and Google Colab integration for the **Pollen Robotics Microduck** 14-DOF bipedal robot.

---

## 🌟 Highlights

- **MuJoCo 3.0 Physics & Gymnasium**: Fast, vectorized simulation with accurate ground contact dynamics and domain randomization.
- **PPO Locomotion Policy**: 61-D observation vector to 14-D joint position target actions (50 Hz control rate).
- **10s+ Champion Checkpoints**: Includes pre-trained walking policies (`checkpoints/best/best_model.zip`, `checkpoints/microduck_8clones_trained.zip`).
- **Real-Time Dynamic Visualizer & Web HUD**: Native 3D viewer + Web dashboard on port 8088 with on-the-fly video ON/OFF toggling.
- **Sim2Real ONNX Exporter**: Complies with Pollen Robotics `robotd` interface (`exported/microduck_10s_champion.onnx`).
- **Google Colab & Drive Integration**: One-click cloud training notebook (`Microduck_RL_Colab.ipynb`).

---

## 🚀 Quick Start (Local)

### 1. Requirements
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mujoco gymnasium stable-baselines3 opencv-python rich torch onnx
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

---

## ☁️ Running on Google Colab

1. Open [Microduck_RL_Colab.ipynb](Microduck_RL_Colab.ipynb) directly in Google Colab.
2. The notebook automatically syncs code from this branch (`rl-locomotion`) directly into your Google Drive (`/content/drive/MyDrive/microduck_training`).
3. Train with GPU/CPU acceleration, evaluate gait, and watch HUD videos right inside the notebook!
