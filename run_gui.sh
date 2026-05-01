#!/usr/bin/env bash
# Launch GUI on the Pi when connected via SSH (uses local desktop session :0).
cd "$(dirname "$0")"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
exec ./.venv/bin/python ./poly_monitor.py "$@"
