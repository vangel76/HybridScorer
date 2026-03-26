# RATEImagesCLIP

This repository contains two interactive Gradio applications for rating and sorting images using different AI models.

## Screenshots

### PromptMatch
Finds images that match a subject or concept you describe in text, with optional negative prompting to exclude unwanted content.

![PromptMatch UI](screenshots/Promptmatch.jpg)

### ImageReward
Ranks images by overall aesthetic fit to a style prompt, so it is better for taste, mood, and visual quality than literal content matching.

![ImageReward UI](screenshots/Imagereward.jpg)

## Architecture

The repository consists of two main Python scripts, each providing a web-based UI for image evaluation:

1.  **`imagereward.py`**: An interactive tool for **aesthetic scoring** of images.
2.  **`promptmatch.py`**: An interactive tool for **semantic content matching** using CLIP models.

Both applications are built with [Gradio](https://www.gradio.app/) and use PyTorch for model inference. They are designed to be run as standalone scripts. A Python 3.12 virtual environment (`venv312`) is included with the necessary dependencies.

---

### `imagereward.py` - Aesthetic Scorer

This tool uses the [ImageReward](https://github.com/THUDM/ImageReward) model (`ImageReward-v1.0`) to score images based on their aesthetic quality, guided by a text prompt.

**Key Features:**

*   **Aesthetic Scoring**: Ranks images based on how well they match a desired aesthetic (e.g., "cinematic," "high fashion").
*   **Best/Normal Galleries**: The UI is split into "BEST" and "NORMAL" galleries.
*   **Score Threshold**: A slider allows you to dynamically set the score threshold to move images between the two galleries.
*   **Manual Override**: Manually move images between galleries if the model's score is not to your liking.
*   **Re-scoring**: Change the prompt to re-evaluate all images based on a new aesthetic.
*   **Export**: Save the sorted images into `best` and `normal` subdirectories.

---

### `promptmatch.py` - Semantic Sorter

This tool uses various CLIP-style models to find images that match a specific textual description (semantic content).

**Key Features:**

*   **Flexible Model Backend**: Supports multiple CLIP models through a unified `ModelBackend` class:
    *   **OpenAI CLIP**: The original models (e.g., `ViT-L/14`).
    *   **OpenCLIP**: High-performance open-source models (e.g., `ViT-bigG-14`).
    *   **Google SigLIP**: State-of-the-art models from Google.
*   **Positive & Negative Prompts**: Sort images based on a **positive prompt** (what you want to find) and an optional **negative prompt** (what you want to exclude).
*   **Found/Not Found Galleries**: The UI is split into "FOUND" and "NOT FOUND" galleries.
*   **Dual Thresholds**: Independent sliders for positive and negative similarity scores provide fine-grained control.
*   **On-the-Fly Model Switching**: Load and switch between different CLIP models directly from the UI.
*   **Export**: Save the sorted images into `found` and `notfound` subdirectories.

---

## How to Run

The repository includes Fish shell scripts for running the applications.

1.  Activate the Python virtual environment:
    ```bash
    source venv312/bin/activate.fish
    ```

2.  To run the aesthetic scorer:
    ```bash
    ./run-imagereward.fish
    ```
    The UI will be available at `http://localhost:7860`.

3.  To run the semantic sorter:
    ```bash
    ./run-promptmatch.fish
    ```
    The UI will be available at `http://localhost:7861`.

### Windows Setup

Windows-ready entrypoints are included:

*   `imagereward_windows.py`
*   `promptmatch_windows.py`
*   `setup-venv312-windows.bat`
*   `run-imagereward-windows.bat`
*   `run-promptmatch-windows.bat`

Recommended setup on Windows:

1.  Install Python 3.12.
2.  Open `cmd.exe` in the project folder.
3.  Create the virtual environment:
    ```bat
    py -3.12 -m venv venv312
    ```
4.  Activate it:
    ```bat
    venv312\Scripts\activate.bat
    ```
5.  Upgrade packaging tools:
    ```bat
    python -m pip install --upgrade pip setuptools wheel
    ```
6.  Install PyTorch:
    ```bat
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```
7.  Install the application packages:
    ```bat
    python -m pip install -r requirements.txt
    ```
8.  Start either app with:
    ```bat
    run-imagereward-windows.bat
    ```
    or
    ```bat
    run-promptmatch-windows.bat
    ```

You can also let the helper script do the setup for you:

```bat
setup-venv312-windows.bat
```

The Windows scripts use `venv312\Scripts\activate.bat` and accept Windows image paths like `C:\images`.

Dependency files:

*   `requirements.txt` contains the application dependencies used by both tools, including OpenAI CLIP support for PromptMatch.

Notes:

*   CUDA is mandatory for this project. The app scripts now exit immediately if PyTorch cannot see a CUDA device.
*   The setup scripts install the CUDA 12.6 PyTorch wheels by default from `https://download.pytorch.org/whl/cu126`.
*   If your machine needs a different supported CUDA wheel, set `PYTORCH_CUDA_INDEX_URL` before running the setup script.
*   Because `requirements.txt` now includes OpenAI CLIP from GitHub, `git` must be installed and available in `PATH` during setup.
*   The model weights themselves are not stored in this repository. ImageReward, OpenCLIP, SigLIP, and OpenAI CLIP weights are downloaded on first use by their respective libraries.

Place your images in a folder named `images` in the root of the repository to have them loaded at startup. You can also load images from any other folder using the UI.
