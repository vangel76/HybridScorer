#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv312"
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python 3.12 or python3 was not found." >&2
  exit 1
fi

echo "Using Python: $PYTHON_BIN"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Reusing existing virtual environment at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing CUDA-enabled PyTorch from:"
echo "  $PYTORCH_CUDA_INDEX_URL"
python -m pip install torch torchvision torchaudio --index-url "$PYTORCH_CUDA_INDEX_URL"

python -m pip install -r "$SCRIPT_DIR/requirements.txt"

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit(
        "CUDA is mandatory for this project, but torch.cuda.is_available() is False.\n"
        "Install a matching NVIDIA driver and CUDA-enabled PyTorch build, then try again."
    )

print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
PY

echo
echo "venv312 is ready."
echo "Activate later with:"
echo "  source venv312/bin/activate"
