# HybridScorer

`HybridScorer` is a local, GPU-first image triage app for sorting large folders of images with AI scoring plus manual review.

Current version: `2.1.0`

## What The App Does

Point the app at a folder of images, run one of the built-in scoring/search modes, review the `SELECTED` and `REJECTED` buckets, fix edge cases manually, then export the result.

Working modes in the current app:

- `PromptMatch` → CLIP-family text-image similarity
- `ImageReward` → prompt-conditioned aesthetic/preference scoring
- `LLM Search` → PromptMatch shortlist plus local vision-language rerank
- `Similarity` → preview-driven image-image search
- `SamePerson` → preview-driven face-identity search

Shared workflow features:

- manual bucket overrides that persist across rescoring in the same folder
- prompt generation from the current preview image
- export by copy or optional move
- histogram-based threshold review
- cached proxies and cached model results where appropriate

## Screenshot

![HybridScorer UI](screenshots/HybridScorer.jpg)

## Install

Clone the repo:

```bash
git clone https://github.com/vangel76/HybridScorer.git
cd HybridScorer
```

The supported setup path is a local Python `3.12` virtual environment named `venv312`.

### Linux

```bash
./setup_update-linux.sh
```

### Windows

```bat
setup_update-windows.bat
```

What the setup scripts do now:

- create or refresh `venv312`
- install CUDA-enabled PyTorch
- install the main app requirements
- install `onnxruntime-gpu` for face search
- install the JoyCaption GGUF runtime into `venv312`
- verify that CUDA is available before finishing

The setup scripts also try a safe `git pull --ff-only` first when the checkout is clean and has an upstream branch.

## Run

### Linux

```bash
./run-Hybrid-Scorer.sh
```

### Windows

```bat
run-Hybrid-Scorer-windows.bat
```

Default local URL:

- `http://localhost:7862`

The app also honors `HYBRIDSELECTOR_PORT` if you need a different port.

## Current Runtime Model

CUDA is mandatory for the app.

The app is designed around local GPU inference:

- PromptMatch models run on CUDA
- ImageReward runs on CUDA
- SamePerson uses InsightFace with CUDA-enabled ONNX Runtime
- JoyCaption GGUF expects a CUDA-enabled `llama-cpp-python` build

If `torch.cuda.is_available()` is false, the setup scripts fail intentionally.

## Prompt Generation

Prompt generation works from the currently previewed image.

Current prompt generators:

- `Florence-2`
- `JoyCaption Beta One`
- `JoyCaption Beta One GGUF (Q4_K_M)`

Behavior:

- generated text is stored separately until you insert it
- you can insert it into `PromptMatch`, `ImageReward`, or `LLM Search`
- prompt detail has 3 levels
- backend instances are cached in memory once loaded

## LLM Search

`LLM Search` is a scored mode in the current app, not just a helper action.

Current behavior:

1. PromptMatch creates a shortlist from your text query
2. a local vision-language backend reranks only the shortlisted images
3. non-shortlisted images get a deterministic reject-floor score

Current backend options:

- `Florence-2`
- `JoyCaption Beta One`
- `JoyCaption Beta One GGUF (Q4_K_M)`

Current default LLM Search backend:

- `JoyCaption Beta One GGUF (Q4_K_M)`

## Cache Behavior

Cache defaults are OS-sensitive:

- Windows default: project-local caches under `models/` and `cache/`
- Linux default: system caches under `~/.cache/...` and proxy cache under the temp directory

Override with:

- `HYBRIDSCORER_CACHE_MODE=project`
- `HYBRIDSCORER_CACHE_MODE=system`

Project-mode cache directories used by the app:

- `models/huggingface`
- `models/clip`
- `models/ImageReward`
- `models/insightface`
- `cache/`

## Main Models And Downloads

Models are downloaded on first use for the mode/backend you actually select.

Highlights:

- default PromptMatch model: `SigLIP so400m-patch14-384`
- ImageReward model: `ImageReward-v1.0`
- face search model pack: `InsightFace buffalo_l`
- Florence prompt generation: `florence-community/Florence-2-base`
- JoyCaption HF: `fancyfeast/llama-joycaption-beta-one-hf-llava`
- JoyCaption GGUF: `cinnabrad/llama-joycaption-beta-one-hf-llava-mmproj-gguf`

## PromptMatch Models

The current PromptMatch dropdown is intentionally trimmed to a practical set instead of every possible variant.

Use cases:

- smaller VRAM / lighter default: `SigLIP`
- stronger NSFW-oriented matching: the OpenCLIP LAION variants
- heavier high-end option: `OpenCLIP ViT-bigG-14 laion2b`

`ImageReward` is separate from PromptMatch and does not use that model dropdown.

## What The App Is Not

- not a cloud service
- not a hosted tagging system
- not a DAM or catalog product
- not a database-style auto-indexer

It is a local human-in-the-loop sorting tool for image folders.

## Manual Setup

If you do not use the setup scripts, the expected environment is still:

```bash
python3.12 -m venv venv312
source venv312/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install onnxruntime-gpu
pip install --no-deps image-reward==1.5
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall --no-cache-dir -r requirements-gguf.txt
```

You also need a CUDA-enabled PyTorch install that matches your system and GPU.

## Repo Notes

- main app entry: `Hybrid-Scorer.py`
- launcher scripts use `venv312`
- the app is intentionally a large single-file Gradio app
- architecture and behavior details live in:
  - [docs/architecture.md](docs/architecture.md)
  - [docs/behavior-notes.md](docs/behavior-notes.md)
