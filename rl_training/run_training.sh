#!/usr/bin/env bash
# Run Microduck Reinforcement Learning Training with Real-Time Video Toggle & Telemetry
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🦆 Launching Microduck RL Training Session..."
"$SCRIPT_DIR/.venv/bin/python" demo_rl_microduck.py --train "$@"
