# RATEImagesCLIP

This repository contains one interactive Gradio application for rating and sorting images with GPU-accelerated AI models.

## What This Is

`RATEImagesCLIP` is built for quick human-in-the-loop image triage.

- `hybrid_selector.py` combines PromptMatch and ImageReward in one UI.
- CUDA is required so scoring stays fast enough to be practical on large folders.
- In the end, the app losslessly copies the original image files into two output folders: `selected` and `rejected`.
- The source images are not recompressed or edited.

## Main App

| App | Best for | How it scores | Output buckets |
| --- | --- | --- | --- |
| `hybrid_selector.py` | Switching between content matching and aesthetic ranking in one place | PromptMatch with CLIP-family models or ImageReward with optional penalty prompt | `selected` / `rejected` |

## Install With Setup Scripts

Set up the Python virtual environment first. You need to do this before trying to run the app.

### Linux Setup Script

Use [setup-venv312.sh](setup-venv312.sh) to create `venv312`, install CUDA-enabled PyTorch, install `requirements.txt`, and verify that CUDA is available:

```bash
./setup-venv312.sh
```

### Windows Setup Script

Use [setup-venv312-windows.bat](setup-venv312-windows.bat) to create `venv312`, install CUDA-enabled PyTorch, install `requirements.txt`, and verify that CUDA is available:

```bat
setup-venv312-windows.bat
```

## Run

After the virtual environment is set up, just run the launcher script. The run scripts activate `venv312` automatically.

### Linux

```bash
./run-hybrid-selector.sh
```

Open:

- `http://localhost:7862` for HybridSelector

### Windows

```bat
run-hybrid-selector-windows.bat
```

## Manual Install

If you do not want to use the setup scripts, you can set up the environment manually.

### Linux Manual Install

```bash
python3.12 -m venv venv312
source venv312/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

### Windows Manual Install

1. Install Python 3.12.
2. Open `cmd.exe` in the project folder.
3. Create the virtual environment:
   ```bat
   py -3.12 -m venv venv312
   ```
4. Activate it:
   ```bat
   venv312\Scripts\activate.bat
   ```
5. Upgrade packaging tools:
   ```bat
   python -m pip install --upgrade pip setuptools wheel
   ```
6. Install CUDA-enabled PyTorch:
   ```bat
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
7. Install the app dependencies:
   ```bat
   python -m pip install -r requirements.txt
   ```

## CUDA Requirement

CUDA is mandatory for this project. The app is meant to rate and sort many images quickly, and that speed depends on GPU inference.

- If PyTorch cannot see a CUDA device, the app exits immediately.
- The setup scripts install CUDA 12.6 PyTorch wheels by default.
- If your machine needs a different supported PyTorch CUDA wheel, set `PYTORCH_CUDA_INDEX_URL` before setup.

## Screenshots

`hybrid_selector.py` lets you switch between PromptMatch for content matching and ImageReward for aesthetic scoring inside one shared workflow.

![HybridSelector UI](screenshots/HybridScorer.jpg)

## Architecture

The repository centers on one Python app:

- **`hybrid_selector.py`**: a combined interface that lets you switch between semantic prompt matching and aesthetic scoring in one place.

The app is built with [Gradio](https://www.gradio.app/) and uses PyTorch for model inference. Set up a local Python 3.12 virtual environment named `venv312` before running it.

### `hybrid_selector.py` - Combined Selector

This app combines PromptMatch and ImageReward into one UI and lets you switch scoring methods without leaving the page.

**Key Features:**

*   **Method Selector**: Switch between PromptMatch and ImageReward inside one app.
*   **Shared Gallery Workflow**: One folder input, shared galleries, manual overrides, export, and threshold controls.
*   **PromptMatch Mode**: Supports CLIP-family model selection plus positive and negative prompts.
*   **ImageReward Mode**: Supports a positive aesthetic prompt and an experimental penalty prompt.
*   **Histogram Threshold Selection**: Click the histogram to set thresholds directly.
*   **Thumbnail Control**: Resize both galleries together with one top-bar slider.

## Files Included

Windows scripts:

*   `setup-venv312-windows.bat`
*   `run-hybrid-selector-windows.bat`

Linux scripts:

*   `setup-venv312.sh`
*   `run-hybrid-selector.sh`

Main cross-platform app:

*   `hybrid_selector.py`

There are no separate Windows-only Python app files. Both operating systems use the same `hybrid_selector.py`.

Dependency notes:

- `requirements.txt` contains the shared application dependencies.
- Because `requirements.txt` includes OpenAI CLIP from GitHub, `git` must be installed and available in `PATH` during setup.
- Model weights are not stored in this repository. ImageReward, OpenCLIP, SigLIP, and OpenAI CLIP weights are downloaded on first use by their libraries.

Place your images in a folder named `images` in the root of the repository to have them loaded at startup. You can also load images from any other folder using the UI.
