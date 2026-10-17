# 🦆 Microduck 專案跨工作階段狀態與指令備忘錄 (Session Handoff & Status)

> **建立時間**: `2026-10-17 11:00:00`  
> **當前分支**: `rl-locomotion` @ `Child-pi/microduck-private` (Commit `f70bbbe`)  
> **當前最高成就**: **25.00 秒 (1,157 物理步)** 仿生站立行走 (超越 20 秒門檻，官方 v8 策略)  
> **累計算力花費**: **~1.07 CU** (Google Colab Tesla T4 GPU，全週期 9 輪共 2.35M 步)

---

## 📌 當前資產與檔案盤點 (Project Assets & Checkpoints)

| 類型 | 檔案路徑 | 說明 |
| :--- | :--- | :--- |
| 🏆 **20s+ 冠軍模型** | `rl_training/checkpoints/microduck_20s_duck_champion.zip` | 實測 25.00 秒 (1157 步) 頂級 PPO 權重 |
| ⭐ **最佳耐力權重** | `rl_training/checkpoints/best_duration_model.zip` | 官方 v8 策略接續訓練起點 |
| 📦 **實機 ONNX 模型** | `rl_training/exported/microduck_20s_champion.onnx` | 14-DOF CPU 1~2ms 邊緣推理模型 |
| 📊 **全週期遙測資料集** | `rl_training/telemetry.jsonl` | 9 輪訓練歷程資料 (步數、獎勵、搖擺、CU) |
| 📋 **最新狀態記錄檔** | `rl_training/status.json` | 包含 25.00 秒各項關鍵指標與硬體規格 |
| 📑 **GitHub 原生指南** | `docs/MICRODUCK_COMPLETE_WALKTHROUGH_AND_GUIDE.md` | GitHub 原生排版與四大 Mermaid 互動架構圖 |
| 🌐 **本地炫彩 HTML 指南** | `docs/MICRODUCK_COMPLETE_WALKTHROUGH_AND_GUIDE.html` | 現代化 Glassmorphic 介面全手冊 |
| 📓 **一鍵 Colab 筆記本** | `Microduck_FastTrack_Colab.ipynb` | 內建 Base64 核心自解壓與資產備援 |

---

## 💻 下一階段常用指令清單 (Command Cheatsheet for Next Session)

### 1. 本地模型評估與生成 HUD 步態影片
```bash
cd /Users/childpi/working/microduck
python3 -c "
from stable_baselines3 import PPO
from rl_training.envs.microduck_env import MicroduckEnv
import numpy as np

model = PPO.load(\"rl_training/checkpoints/microduck_20s_duck_champion.zip\", device=\"cpu\")
env = MicroduckEnv(render_mode=\"rgb_array\", max_episode_steps=1500, duck_mimic_mode=True)
obs, _ = env.reset(seed=42)
steps = 0
for s in range(1500):
    action, _ = model.predict(obs, deterministic=True)
    obs, rew, term, trunc, info = env.step(action)
    steps += 1
    if term or trunc:
        break
print(f\"Evaluation result: {steps*0.02:.2f}s ({steps} steps)\\")
"
```

### 2. 本地接續強化學習訓練 (以 25 秒冠軍為 Warm-Start 衝刺 30 秒+)
```bash
cd /Users/childpi/working/microduck/rl_training
python3 train.py \
  --num-clones 16 \
  --warm-start checkpoints/microduck_20s_duck_champion.zip \
  --total-timesteps 2000000 \
  --target-kl 0.015 \
  --headless
```

### 3. 實機 ONNX 50Hz (20ms) 閉環推論腳本啟動
```bash
cd /Users/childpi/working/microduck
# 執行包含前處理 (IMU+時鐘) 與後處理 (變化率截斷+指數平滑) 的標準推論：
python3 -c "
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession(\"rl_training/exported/microduck_20s_champion.onnx\", providers=[\"CPUExecutionProvider\"])
dummy_obs = np.random.randn(1, 61).astype(np.float32)
out_action = session.run(None, {session.get_inputs()[0].name: dummy_obs})[0]
print(\"ONNX 14-DOF Action output shape:\", out_action.shape)
"
```

### 4. 啟動 Chrome CDP 遠端自動化除錯埠 (若需再次由 Agent 遙控 Colab)
```bash
# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir="/tmp/chrome-debug" &
```

### 5. 啟動即時遙測看板背景守護進程 (Dashboard Daemon)
```bash
python3 /Users/childpi/.gemini/antigravity-cli/brain/6240d44c-3026-466b-878b-8bfa9c4c36a7/scratch/dashboard/dashboard_daemon.py &
```

### 6. Git 提交與同步狀態
```bash
cd /Users/childpi/working/microduck
git status
git pull private rl-locomotion
git push private rl-locomotion
```

---

## 🎯 下一階段建議任務 (Recommended Next Goals)
1. **衝刺 30.0 秒極限**：以當前 25.00 秒模型為起點，開啟地形高度小幅度擾動（Curriculum Learning）。
2. **實機關節零位校準**：在真實 Microduck 機器人上進行實體舵機歸零偏移量標定。
3. **Sim-to-Real 上機行走**：將 `microduck_20s_champion.onnx` 與前後處理模組部署至板載樹莓派進行桌面首走。
