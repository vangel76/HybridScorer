#!/usr/bin/env fish

set SCRIPT_DIR /home/vangel/apps/RATEImagesCLIP
set VENV $SCRIPT_DIR/venv312
set PYTHON $VENV/bin/python
set SCRIPT $SCRIPT_DIR/promptmatch.py

echo "=== Prompt-Match Image Sorter ==="
echo "Activating venv: $VENV"
source $VENV/bin/activate.fish

echo "Running: $SCRIPT"
echo ""
$PYTHON $SCRIPT

echo ""
echo "=== Finished ==="
