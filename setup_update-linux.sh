#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv312"
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTORCH_TORCH_VERSION="${PYTORCH_TORCH_VERSION:-2.9.1}"
PYTORCH_TORCHVISION_VERSION="${PYTORCH_TORCHVISION_VERSION:-0.24.1}"

usage() {
  echo "Usage: ./setup_update-linux.sh" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --update)
      echo "Note: --update is no longer required; setup_update-linux.sh already checks for a safe git refresh."
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
  shift
done

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python 3.12 or python3 was not found." >&2
  exit 1
fi

echo "Using Python: $PYTHON_BIN"

maybe_update_git_checkout() {
  if ! command -v git >/dev/null 2>&1; then
    echo "git was not found in PATH. Skipping automatic git refresh."
    return
  fi

  cd "$SCRIPT_DIR"

  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "This folder is not a git working tree. Skipping automatic git refresh."
    return
  fi

  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Tracked local changes were detected. Skipping automatic git refresh."
    return
  fi

  if ! git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    echo "No upstream branch is configured. Skipping automatic git refresh."
    return
  fi

  local current_branch
  current_branch="$(git branch --show-current 2>/dev/null || true)"
  if [ -n "$current_branch" ]; then
    echo "Checking for git updates on branch: $current_branch"
  fi

  if git pull --ff-only; then
    echo "Git checkout is up to date."
  else
    echo "Automatic git refresh failed. Continuing with venv setup."
  fi
  echo
}

validate_existing_venv() {
  local venv_python="$VENV_DIR/bin/python"
  local pyvenv_cfg="$VENV_DIR/pyvenv.cfg"

  if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
    echo "Existing venv312 is not healthy." >&2
    echo "python -m pip failed inside $VENV_DIR." >&2
    echo "Delete venv312 and run ./setup_update-linux.sh again." >&2
    exit 1
  fi

  if ! grep -Fq "$VENV_DIR" "$pyvenv_cfg" 2>/dev/null; then
    echo "Existing venv312 appears to have been copied or moved from another path." >&2
    echo "Expected to find this project path in $pyvenv_cfg:" >&2
    echo "  $VENV_DIR" >&2
    echo "Delete venv312 and run ./setup_update-linux.sh again." >&2
    exit 1
  fi
}

# Returns 0 if a Python snippet exits cleanly, 1 otherwise.
py_check() {
  python -c "$1" 2>/dev/null
}

echo "Checking whether a safe git refresh is needed..."
maybe_update_git_checkout

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Reusing existing virtual environment at $VENV_DIR"
  validate_existing_venv
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# ── PyTorch ───────────────────────────────────────────────────────────────────
if py_check "
import torch, torchvision
assert torch.__version__ == '$PYTORCH_TORCH_VERSION', torch.__version__
assert torchvision.__version__ == '$PYTORCH_TORCHVISION_VERSION', torchvision.__version__
"; then
  echo "PyTorch $PYTORCH_TORCH_VERSION + torchvision $PYTORCH_TORCHVISION_VERSION already installed, skipping."
else
  echo
  echo "Installing CUDA-enabled PyTorch from:"
  echo "  $PYTORCH_CUDA_INDEX_URL"
  echo "Pinned package versions:"
  echo "  torch==$PYTORCH_TORCH_VERSION"
  echo "  torchvision==$PYTORCH_TORCHVISION_VERSION"
  python -m pip install \
    "torch==$PYTORCH_TORCH_VERSION" \
    "torchvision==$PYTORCH_TORCHVISION_VERSION" \
    --index-url "$PYTORCH_CUDA_INDEX_URL"
fi

# ── requirements.txt ──────────────────────────────────────────────────────────
python -m pip install -r "$SCRIPT_DIR/requirements.txt"

# ── onnxruntime-gpu ───────────────────────────────────────────────────────────
if py_check "
import onnxruntime as ort
assert 'CUDAExecutionProvider' in ort.get_available_providers()
"; then
  echo "onnxruntime-gpu (CUDA) already installed, skipping."
else
  python -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  python -m pip install onnxruntime-gpu
fi

# ── image-reward ──────────────────────────────────────────────────────────────
if py_check "from importlib.metadata import version; assert version('image-reward') == '1.5'"; then
  echo "image-reward 1.5 already installed, skipping."
else
  python -m pip install --no-deps image-reward==1.5
fi

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
