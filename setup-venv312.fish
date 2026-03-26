#!/usr/bin/env fish

set SCRIPT_DIR (cd (dirname (status --current-filename)); and pwd)
set VENV_DIR "$SCRIPT_DIR/venv312"

if set -q PYTORCH_CUDA_INDEX_URL
    set TORCH_INDEX_URL "$PYTORCH_CUDA_INDEX_URL"
else
    set TORCH_INDEX_URL "https://download.pytorch.org/whl/cu126"
end

if command -q python3.12
    set PYTHON_BIN python3.12
else if command -q python3
    set PYTHON_BIN python3
else
    echo "Python 3.12 or python3 was not found." >&2
    exit 1
end

echo "Using Python: $PYTHON_BIN"

if not test -x "$VENV_DIR/bin/python"
    echo "Creating virtual environment at $VENV_DIR"
    $PYTHON_BIN -m venv "$VENV_DIR"
else
    echo "Reusing existing virtual environment at $VENV_DIR"
end

source "$VENV_DIR/bin/activate.fish"

python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing CUDA-enabled PyTorch from:"
echo "  $TORCH_INDEX_URL"
python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

python -m pip install -r "$SCRIPT_DIR/requirements.txt"

python -c "import sys, torch; ok=torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)}' if ok else 'CUDA missing'); sys.exit(0 if ok else 1)"
or begin
    echo "CUDA is mandatory for this project, but torch.cuda.is_available() is False." >&2
    echo "Install a matching NVIDIA driver and CUDA-enabled PyTorch build, then try again." >&2
    exit 1
end

echo
echo "venv312 is ready."
echo "Activate later with:"
echo "  source venv312/bin/activate.fish"
