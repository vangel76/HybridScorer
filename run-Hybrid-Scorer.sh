#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv312"
PYTHON_BIN="$VENV_DIR/bin/python"
SCRIPT_PATH="$SCRIPT_DIR/Hybrid-Scorer.py"

echo "=== Hybrid-Scorer ==="

if [ ! -x "$PYTHON_BIN" ]; then
  echo "venv312 was not found."
  echo "Run ./setup-venv312.sh first."
  exit 1
fi

echo "Activating venv: $VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Running: $SCRIPT_PATH"
echo
"$PYTHON_BIN" "$SCRIPT_PATH"

echo
echo "=== Finished ==="
