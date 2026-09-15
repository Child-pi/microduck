#!/usr/bin/env bash
# Run Microduck Walking Evaluation Demo with Live Video & Status Dashboard
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🦆 Launching Microduck RL Walking Demonstration..."
"$SCRIPT_DIR/.venv/bin/python" demo_rl_microduck.py --demo "$@"
