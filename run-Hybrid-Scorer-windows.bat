@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%CD%\venv312\Scripts\python.exe"
set "APP_PY=%CD%\Hybrid-Scorer.py"

if not exist "%VENV_PY%" (
    echo venv312 was not found.
    echo Run setup_update-windows.bat first.
    exit /b 1
)

if not exist "%APP_PY%" (
    echo Hybrid-Scorer.py was not found in:
    echo   %CD%
    exit /b 1
)

"%VENV_PY%" "%APP_PY%"
