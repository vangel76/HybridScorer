@echo off
setlocal
cd /d "%~dp0"

if not exist "venv312\Scripts\python.exe" (
    echo venv312 was not found.
    echo Run setup-venv312-windows.bat first.
    exit /b 1
)

call "venv312\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python promptmatch_windows.py
