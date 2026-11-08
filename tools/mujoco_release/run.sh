#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=python3
if command -v python3.11 >/dev/null 2>&1; then PYTHON=python3.11; fi
if [ ! -f .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e "./CausalForge[mujoco]" "imageio>=2.34" pillow
fi
exec .venv/bin/python CausalForge/tools/mujoco_lab.py "$@"
