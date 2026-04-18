@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="--update" (
    echo Note: --update is no longer required; setup_update-windows.bat already checks for a safe git refresh.
    shift
)
if not "%~1"=="" (
    echo Usage: setup_update-windows.bat
    exit /b 1
)

set "VENV_DIR=%CD%\venv312"
if "%PYTORCH_CUDA_INDEX_URL%"=="" set "PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128"
if "%PYTORCH_TORCH_VERSION%"=="" set "PYTORCH_TORCH_VERSION=2.9.1"
if "%PYTORCH_TORCHVISION_VERSION%"=="" set "PYTORCH_TORCHVISION_VERSION=0.24.1"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher "py" was not found.
    where winget >nul 2>nul
    if errorlevel 1 (
        echo winget was not found.
        echo Install Python 3.12 for Windows from python.org and enable the launcher.
        exit /b 1
    )

    echo Attempting to install Python 3.12 with winget...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo Automatic Python install failed.
        echo Install Python 3.12 for Windows from python.org and enable the launcher.
        exit /b 1
    )

    where py >nul 2>nul
    if errorlevel 1 (
        echo Python 3.12 installation finished, but the py launcher is still not available in this shell.
        echo Close this window, open a new one, and run setup_update-windows.bat again.
        exit /b 1
    )
)

where git >nul 2>nul
if errorlevel 1 (
    echo Git was not found in PATH.
    where winget >nul 2>nul
    if errorlevel 1 (
        echo winget was not found.
        echo Install Git for Windows first, then run this setup script again.
        exit /b 1
    )

    echo Attempting to install Git with winget...
    winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo Automatic Git install failed.
        echo Install Git for Windows first, then run this setup script again.
        exit /b 1
    )

    where git >nul 2>nul
    if errorlevel 1 (
        echo Git installation finished, but git is still not available in this shell.
        echo Close this window, open a new one, and run setup_update-windows.bat again.
        exit /b 1
    )
)

echo Checking whether a safe git refresh is needed...

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
    echo This folder is not a git working tree. Skipping automatic git refresh.
    goto after_git_refresh
)

git diff --quiet
if errorlevel 1 (
    echo Tracked local changes were detected. Skipping automatic git refresh.
    goto after_git_refresh
)

git diff --cached --quiet
if errorlevel 1 (
    echo Tracked local changes were detected. Skipping automatic git refresh.
    goto after_git_refresh
)

git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>nul
if errorlevel 1 (
    echo No upstream branch is configured. Skipping automatic git refresh.
    goto after_git_refresh
)

set "CURRENT_BRANCH="
for /f "delims=" %%I in ('git branch --show-current 2^>nul') do set "CURRENT_BRANCH=%%I"
if not "%CURRENT_BRANCH%"=="" (
    echo Checking for git updates on branch: %CURRENT_BRANCH%
)

git pull --ff-only
if errorlevel 1 (
    echo Automatic git refresh failed. Continuing with venv setup.
) else (
    echo Git checkout is up to date.
)

echo.
:after_git_refresh

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment at "%VENV_DIR%"...
    py -3.12 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Python 3.12 was found, but creating the virtual environment failed.
        echo Make sure Python 3.12 is installed correctly, then run this script again.
        exit /b 1
    )
) else (
    echo Reusing existing virtual environment at "%VENV_DIR%".
    "%VENV_DIR%\Scripts\python.exe" -m pip --version >nul 2>nul
    if errorlevel 1 (
        echo Existing venv312 is not healthy.
        echo python -m pip failed inside "%VENV_DIR%".
        echo Delete venv312 and run setup_update-windows.bat again.
        exit /b 1
    )

    if exist "%VENV_DIR%\pyvenv.cfg" (
        findstr /C:"%VENV_DIR%" "%VENV_DIR%\pyvenv.cfg" >nul 2>nul
        if errorlevel 1 (
            echo Existing venv312 appears to have been copied or moved from another path.
            echo Expected to find this project path in "%VENV_DIR%\pyvenv.cfg":
            echo   %VENV_DIR%
            echo Delete venv312 and run setup_update-windows.bat again.
            exit /b 1
        )
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

echo.
echo Installing CUDA-enabled PyTorch from:
echo   %PYTORCH_CUDA_INDEX_URL%
echo Pinned package versions:
echo   torch==%PYTORCH_TORCH_VERSION%
echo   torchvision==%PYTORCH_TORCHVISION_VERSION%
echo If you override PYTORCH_CUDA_INDEX_URL, make sure these pinned versions exist on that index.
python -m pip install torch==%PYTORCH_TORCH_VERSION% torchvision==%PYTORCH_TORCHVISION_VERSION% --index-url %PYTORCH_CUDA_INDEX_URL%
if errorlevel 1 exit /b 1

python -m pip install -r "%CD%\requirements.txt"
if errorlevel 1 exit /b 1

python -m pip uninstall -y onnxruntime onnxruntime-gpu >nul 2>nul
python -m pip install onnxruntime-gpu
if errorlevel 1 exit /b 1

python -m pip install --no-deps image-reward==1.5
if errorlevel 1 exit /b 1

echo.
echo Installing JoyCaption GGUF runtime with CUDA-enabled llama-cpp-python
set "CMAKE_ARGS=-DGGML_CUDA=on"
set "FORCE_CMAKE=1"
python -m pip install --upgrade --force-reinstall --no-cache-dir "llama-cpp-python>=0.3.7"
if errorlevel 1 (
    echo Standard CUDA llama-cpp-python build failed.
    echo Trying Visual Studio developer shell plus Ninja fallback...

    set "VSDEVCMD="
    for %%F in (
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
    ) do (
        if exist "%%~fF" if not defined VSDEVCMD set "VSDEVCMD=%%~fF"
    )

    where ninja >nul 2>nul
    if errorlevel 1 (
        echo Ninja was not found in PATH, so the fallback build cannot run.
        exit /b 1
    )
    if not defined VSDEVCMD (
        echo Visual Studio developer shell was not found, so the fallback build cannot run.
        exit /b 1
    )

    call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul
    if errorlevel 1 (
        echo Failed to initialize the Visual Studio developer shell.
        exit /b 1
    )
    if not defined VCToolsInstallDir (
        echo VCToolsInstallDir was not set by the developer shell.
        exit /b 1
    )

    set "CC=%VCToolsInstallDir%bin\Hostx64\x64\cl.exe"
    set "CXX=%CC%"
    set "CMAKE_ARGS=-G Ninja -DCMAKE_MAKE_PROGRAM=C:/Windows/ninja.exe -DGGML_CUDA=on"
    python -m pip install --upgrade --force-reinstall --no-cache-dir "llama-cpp-python>=0.3.7"
    if errorlevel 1 exit /b 1
)
set "CMAKE_ARGS="
set "FORCE_CMAKE="
set "CC="
set "CXX="

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
