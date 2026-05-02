# HybridScorer

`HybridScorer` is a 100% local app for sorting large folders of images with AI scoring.

Current version: `2.9.1`

## What The App Does

HybridScorer cuts large image sets down to size fast. Point it at a folder, let the AI score everything, then review a clean `SELECTED` / `REJECTED` split and export what you want to keep. No cloud, no uploads — all processing runs on your local GPU.

Working modes in the current app:

- `PromptMatch` — describe what you want in plain text and score the whole folder against it; a per-segment mode scores each comma-separated phrase independently and shows match strength per phrase on hover
- `TagMatch` — for NSFW searches and for hunting down AI anatomy errors: wrong hands, extra limbs, duplicate faces, bad proportions; ships with a ready-to-use artifact tag set, suggests valid booru-style tags live while you type, and shows per-tag confidence as color-coded pills when you hover a thumbnail
- `ImageReward` — rank images by how well they match a prompt, weighted against unwanted content
- `LM Search` — use a local vision-language model to deeply understand each image, not just match keywords
- `Similarity` — pick one image you like and find everything in the folder that looks like it
- `SamePerson` — pick a face and pull every image of that person from the folder automatically
- `ObjectSearch` — drop or paste any image of a specific object and find every folder image that contains it, even when the object is small or partially visible; uses DINOv2 dense patch features with GPU-accelerated nearest-neighbor search

Everything in the app is built around the same fast loop:

- generate a prompt from any preview image or temporary external query image and feed it straight into scoring
- export your kept images by copy or by move
- the histogram shows you exactly where the threshold cuts, so you keep control
- re-scoring reuses cached results — switching prompts or tweaking the threshold is near-instant
- manually move any image between buckets — overrides AI re-scoring allways when needed

## Screenshots

![HybridScorer UI](screenshots/HybridScorer.jpg)


The image sorting updates live as you move the thresholds. The red/green split shows exactly which images cross the cut — hover over any thumbnail and its score appears as a marker on the graph. You can drag the slider or click anywhere directly on the histogram to set the cut point instantly. Or hit **Fit thresh** to let the app position it automatically based on your current manual picks.

## Install

Clone the repo:

```bash
git clone https://github.com/vangel76/HybridScorer.git
cd HybridScorer
```

then run:

### Windows

```bat
setup_update-windows.bat
```

### Linux

```bash
./setup_update-linux.sh
```


What the setup scripts do now:

- create or refresh the folder `venv312` containing all packages needed
- install CUDA-enabled PyTorch (skipped if already at the pinned version)
- install the main app requirements including `faiss-cpu` for ObjectSearch
- install `onnxruntime-gpu` for face search (skipped if CUDA provider already present)
- install `image-reward 1.5` (skipped if already installed)
- install JoyCaption HF/NF4 runtime dependencies through `requirements.txt`
- verify that CUDA is available before finishing

The setup scripts also try a safe `git pull --ff-only` first when the checkout is clean and has an upstream branch.

## Run

### Windows

```bat
run-Hybrid-Scorer-windows.bat
```

### Linux

```bash
./run-Hybrid-Scorer.sh
```

The FastAPI + Tabler UI appears at local URL:

- `http://localhost:7862`


## Current Runtime Model

CUDA is mandatory for the app.

The app is designed around local GPU inference:

- PromptMatch models run on CUDA
- ImageReward runs on CUDA and includes in-app compatibility shims for newer Transformers 5.x builds
- SamePerson uses InsightFace with CUDA-enabled ONNX Runtime
- JoyCaption HF/NF4 uses Transformers on CUDA; the NF4 backend requires `accelerate` and `bitsandbytes`
- JoyCaption no longer uses the GGUF / llama.cpp runtime path

If `torch.cuda.is_available()` is false, the setup scripts fail intentionally.

## Prompt Generation

Prompt generation works from the active query image:
- external query image first if you dropped, pasted, or uploaded one in the sidebar
- otherwise the currently previewed gallery image

Current prompt generators:

- `Florence-2`
- `JoyCaption Beta One`
- `JoyCaption Beta One NF4`
- `Huihui Gemma 4 E4B`

Behavior:

- generated text is stored separately until you insert it
- you can insert it into `PromptMatch`, `ImageReward`, or `LM Search`
- prompt detail has 3 levels
- backend instances are cached in memory once loaded
- the dropdown shows cached backends in green and first-download backends in amber
- JoyCaption NF4 uses the Hugging Face model cache and may need a larger first download before it appears as cached
- the `Huihui Gemma 4 E4B` option is a less-filtered abliterated model and may produce less-filtered text
- Huihui Gemma 4 also requires a Transformers build that includes Gemma 4 runtime classes; rerun setup after dependency updates if the backend reports missing Gemma 4 processor/model classes

## LM Search

`LM Search` is a scored mode in the current app, not just a helper action.

Current behavior:

1. PromptMatch creates a shortlist from your text query
2. a local vision-language backend reranks only the shortlisted images
3. non-shortlisted images get a deterministic reject-floor score

Current backend options:

- `Florence-2`
- `JoyCaption Beta One`
- `JoyCaption Beta One NF4`
- `Huihui Gemma 4 E4B`

The LM Search backend dropdown also shows cached backends in green and first-download backends in amber.

Current default LLM Search backend:

- `JoyCaption Beta One NF4`
- `Huihui Gemma 4 E4B` is optional only; defaults are unchanged and its outputs may be less filtered

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
- ObjectSearch model: `facebook/dinov2-base`
- Florence prompt generation: `florence-community/Florence-2-base`
- JoyCaption HF: `fancyfeast/llama-joycaption-beta-one-hf-llava`
- JoyCaption NF4: `John6666/llama-joycaption-beta-one-hf-llava-nf4`
- Huihui Gemma 4: `huihui-ai/Huihui-gemma-4-E4B-it-abliterated`

The downloadable model selectors in the UI show cached items in green and items that still need a first download in amber.

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
```

You also need a CUDA-enabled PyTorch install that matches your system and GPU.

## Repo Notes

- main app entry: `Hybrid-Scorer.py`
- launcher scripts use `venv312`
- the app uses FastAPI for local HTTP/WebSocket routes and a server-rendered Tabler UI
- model, scoring, cache, state, prompt, and export logic live under `lib/`
- architecture and behavior details live in:
  - [docs/architecture.md](docs/architecture.md)
  - [docs/behavior-notes.md](docs/behavior-notes.md)
