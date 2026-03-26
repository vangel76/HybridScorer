@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=%CD%\venv312"
if "%PYTORCH_CUDA_INDEX_URL%"=="" set "PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu126"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher "py" was not found.
    echo Install Python 3.12 for Windows from python.org and enable the launcher.
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment at "%VENV_DIR%"...
    py -3.12 -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
) else (
    echo Reusing existing virtual environment at "%VENV_DIR%".
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

echo.
echo Installing CUDA-enabled PyTorch from:
echo   %PYTORCH_CUDA_INDEX_URL%
python -m pip install torch torchvision torchaudio --index-url %PYTORCH_CUDA_INDEX_URL%
if errorlevel 1 exit /b 1

python -m pip install -r "%CD%\requirements.txt"
if errorlevel 1 exit /b 1

python -c "import sys, torch; ok=torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)}' if ok else 'CUDA missing'); sys.exit(0 if ok else 1)"
if errorlevel 1 (
    echo CUDA is mandatory for this project, but torch.cuda.is_available() is False.
    echo Install a matching NVIDIA driver and CUDA-enabled PyTorch build, then try again.
    exit /b 1
)

echo.
echo venv312 is ready.
echo Activate later with:
echo   venv312\Scripts\activate.bat
