#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv312"
UPDATE_GIT=0
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTORCH_TORCH_VERSION="${PYTORCH_TORCH_VERSION:-2.9.1}"
PYTORCH_TORCHVISION_VERSION="${PYTORCH_TORCHVISION_VERSION:-0.24.1}"

usage() {
  echo "Usage: ./setup-hybridscorer-linux.sh [--update]" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --update)
      UPDATE_GIT=1
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

update_git_checkout() {
  if ! command -v git >/dev/null 2>&1; then
    echo "git was not found in PATH." >&2
    exit 1
  fi

  cd "$SCRIPT_DIR"

  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "This folder is not a git working tree:" >&2
    echo "  $SCRIPT_DIR" >&2
    exit 1
  fi

  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Tracked local changes were detected." >&2
    echo "Commit or stash them before running ./setup-hybridscorer-linux.sh --update." >&2
    exit 1
  fi

  if ! git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    echo "No upstream branch is configured for this checkout." >&2
    echo "Set a tracking branch first, then rerun ./setup-hybridscorer-linux.sh --update." >&2
    exit 1
  fi

  local current_branch
  current_branch="$(git branch --show-current 2>/dev/null || true)"
  if [ -n "$current_branch" ]; then
    echo "Updating branch: $current_branch"
  fi

  git pull --ff-only
  echo
}

validate_existing_venv() {
  local venv_python="$VENV_DIR/bin/python"
  local pyvenv_cfg="$VENV_DIR/pyvenv.cfg"

  if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
    echo "Existing venv312 is not healthy." >&2
    echo "python -m pip failed inside $VENV_DIR." >&2
    echo "Delete venv312 and run ./setup-hybridscorer-linux.sh again." >&2
    exit 1
  fi

  if [ -f "$pyvenv_cfg" ] && ! grep -Fq "$VENV_DIR" "$pyvenv_cfg"; then
    echo "Existing venv312 appears to have been copied or moved from another path." >&2
    echo "Expected to find this project path in $pyvenv_cfg:" >&2
    echo "  $VENV_DIR" >&2
    echo "Delete venv312 and run ./setup-hybridscorer-linux.sh again." >&2
    exit 1
  fi
}

if [ "$UPDATE_GIT" = "1" ]; then
  echo "Updating git checkout before refreshing venv312..."
  update_git_checkout
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Reusing existing virtual environment at $VENV_DIR"
  validate_existing_venv
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing CUDA-enabled PyTorch from:"
echo "  $PYTORCH_CUDA_INDEX_URL"
echo "Pinned package versions:"
echo "  torch==$PYTORCH_TORCH_VERSION"
echo "  torchvision==$PYTORCH_TORCHVISION_VERSION"
echo "If you override PYTORCH_CUDA_INDEX_URL, make sure these pinned versions exist on that index."
python -m pip install \
  "torch==$PYTORCH_TORCH_VERSION" \
  "torchvision==$PYTORCH_TORCHVISION_VERSION" \
  --index-url "$PYTORCH_CUDA_INDEX_URL"

python -m pip install -r "$SCRIPT_DIR/requirements.txt"
python -m pip install --no-deps image-reward==1.5

if [ "${INSTALL_JOYCAPTION_GGUF:-0}" = "1" ]; then
  echo
  echo "Installing optional JoyCaption GGUF runtime from requirements-gguf.txt"
  python -m pip install -r "$SCRIPT_DIR/requirements-gguf.txt"
else
  echo
  echo "Skipping optional JoyCaption GGUF runtime."
  echo "To enable it later, rerun:"
  echo "  INSTALL_JOYCAPTION_GGUF=1 ./setup-hybridscorer-linux.sh"
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
