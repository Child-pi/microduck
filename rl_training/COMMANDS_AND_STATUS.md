# Microduck 訓練狀態與指令保存手冊 (Commands and Status)

**更新時間 (Local Time)**: 2026-09-13T23:16:00+08:00  
**專案路徑**: `/Users/childpi/working/microduck_training`  
**虛擬環境**: `/Users/childpi/working/microduck_training/.venv`  

---

## 🏆 目前訓練狀態 (Current Training Status)

| 項目 (Item) | 數值 / 狀態 (Value / Status) | 備註 (Notes) |
| :--- | :--- | :--- |
| **最長連續行走時間** | **10.52 秒 (526 steps)** | 實測種子 Seed 89，超越 10 秒目標 |
| **直線前進淨位移** | **+0.480 公尺** | 勻速穩定前進，無暴衝摔覆 |
| **控制頻率** | **50 Hz** (每步 0.02s) | 物理步長 0.002s，Frame Skip 10 |
| **姿態平衡度** | 俯仰 $g_x \in [-0.06, +0.04]$，側傾 $g_y \in [-0.08, +0.02]$ | 全程直立在 $\pm 5^\circ$ 安全椎內 |
| **最佳模型權重** | `checkpoints/best/best_model.zip` | 10.52 秒冠軍模型 |
| **接續微調權重** | `checkpoints/microduck_10s_infinite.zip` | 速度走廊收斂版本 |
| **實機部署 ONNX** | `exported/microduck_10s_champion.onnx` | 符合 Pollen Robotics `robotd` 規範 |
| **高清評估影片** | `walk_10s_victory.mp4` / `walk_10s_victory.gif` | 含步態 HUD 遙測數據 |
| **TensorBoard 監控** | 運行於 `http://localhost:6006` | 背景常駐 Task-674 |

---

## 📋 常用核心指令全集 (Saved Commands)

### 1. 虛擬環境啟動 (Environment Setup)
```bash
cd /Users/childpi/working/microduck_training
source .venv/bin/activate
```

---

### 2. 評估 10 秒冠軍策略並錄製 HUD 影片 (Evaluation & Video Recording)
載入最佳權重，運行 600 步（12 秒）並輸出包含步數、時間、瞬時速度、軀幹高度、俯仰角、側傾角與雙腳著地指示的 640x480 MP4 與動態 GIF：
```bash
python analyze_and_record.py \
  --model-path checkpoints/best/best_model.zip \
  --output-video walk_10s_victory.mp4 \
  --output-gif walk_10s_victory.gif \
  --output-snapshot walk_10s_victory_snapshot.png \
  --vx 0.06 \
  --seed 89 \
  --max-steps 600
```
*或直接執行一鍵腳本：*
```bash
./.agents/skills/train-microduck/scripts/eval_10s.sh
```

---

### 3. 多種子耐力快速巡檢 (Multi-Seed Scan)
快速測試 30 個隨機種子的連續邁步時間與位移，無渲染負擔：
```bash
python -c '
import numpy as np
from stable_baselines3 import PPO
from envs.microduck_env import MicroduckEnv

model = PPO.load("checkpoints/best/best_model.zip", device="cpu")
env = MicroduckEnv(max_episode_steps=800)
env.resample_commands = False

for seed in range(30):
    obs, _ = env.reset(seed=seed)
    env.commands[0] = 0.06
    obs[48:51] = [0.06, 0.0, 0.0]
    steps = 0
    for s in range(800):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, _ = env.step(action)
        steps = s + 1
        if term or trunc:
            break
    print(f"Seed {seed:2d}: {steps*0.02:5.2f}s ({steps:3d} steps), x={env.data.qpos[0]:+.3f}m")
'
```

---

### 4. 鍵盤互動遙控小鴨 (Interactive Keyboard Teleop)
使用方向鍵即時控制小鴨行走前進、後退、轉彎與頭部姿態：
```bash
python interactive_control.py --model-path checkpoints/best/best_model.zip
```
- **W / S**: 前進 / 後退
- **A / D**: 左右轉彎 (Yaw)
- **Q / E**: 左右平移 (Strafe)
- **I / K**: 頭頸抬頭 / 低頭 (Pitch)
- **J / L**: 頭部左右擺頭 (Yaw)
- **Space**: 原地立正平衡 (Stop)

---

### 5. 8 影分身實時可視化訓練 (Live 8-Clone Multi-View Training)
在 2x4 分割畫面中即時觀看 8 隻小鴨同時邁步探索與訓練：
```bash
python train_with_live_view.py \
  --num-envs 8 \
  --resume-from checkpoints/best/best_model.zip \
  --fps 30 \
  --timesteps 500000
```

---

### 6. 接續強化學習微調訓練 (PPO Fine-Tuning)
利用 Apple Silicon M4 Pro 8 核心平行向量化環境進行接續訓練：
```bash
python train.py \
  --timesteps 1500000 \
  --num-envs 8 \
  --lr 5e-5 \
  --ent-coef 0.0001 \
  --gamma 0.9985 \
  --resume-from checkpoints/best/best_model.zip \
  --output-name microduck_10s_v2
```

---

### 7. 匯出實機 ONNX 模型 (Sim2Real Export)
將訓練好的神經網絡匯出為符合 Pollen Robotics 官方規範之 ONNX 檔案：
```bash
python export_onnx.py \
  --model-path checkpoints/best/best_model.zip \
  --output-path exported/microduck_10s_champion.onnx
```
- **輸入**: `obs` [batch_size, 61] (float32)
- **輸出**: `actions` [batch_size, 14] (float32)

---

### 8. TensorBoard 即時圖表監控
```bash
tensorboard --logdir logs --port 6006 --bind_all
# 瀏覽器訪問: http://localhost:6006
```

---

## 📁 相關說明文件位置
- **完整使用手冊**: [how_to_use_training/index.html](file:///Users/childpi/working/microduck_training/how_to_use_training/index.html)
- **4秒至10秒突破指南**: [how_to_training_from_4sec_to_10sec/index.html](file:///Users/childpi/working/microduck_training/how_to_training_from_4sec_to_10sec/index.html)
- **Antigravity Skill 定義**: [.agents/skills/train-microduck/SKILL.md](file:///Users/childpi/working/.agents/skills/train-microduck/SKILL.md)
