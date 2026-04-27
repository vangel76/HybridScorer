#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv312"
PYTHON_BIN="$VENV_DIR/bin/python"
SCRIPT_PATH="$SCRIPT_DIR/Hybrid-Scorer.py"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "venv312 was not found."
  echo "Run ./setup_update-linux.sh first."
  exit 1
fi

source "$VENV_DIR/bin/activate"
"$PYTHON_BIN" "$SCRIPT_PATH"
