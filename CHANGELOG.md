# HybridScorer User-Friendly Updates

## Version 2.5.0 - ObjectSearch & Faster Setup

**New ObjectSearch Mode**
- New search mode: point at any image (or paste/upload a query) and find every image in the folder that contains that specific object, even as a small part of a larger scene
- Uses DINOv2 patch-level features — 256 dense patches per image — so a small object in the corner of a photo still registers as a match
- Gallery patch features are uploaded to GPU memory for fast torch-matmul nearest-neighbor search; falls back to CPU FAISS when no GPU is available
- Results use the same threshold slider and top-N controls as Similarity and SamePerson
- Feature cache is reused across queries against the same folder, so a second search is near-instant
- Model: `facebook/dinov2-base` (downloaded once on first use)

**Faster Setup / Update**
- `setup_update-linux.sh` now skips each install step when the package is already present and working
- PyTorch is skipped if the pinned version is already installed (avoids hitting the CUDA index every time)
- `onnxruntime-gpu` is skipped if `CUDAExecutionProvider` is already available
- `image-reward 1.5` is skipped if it is already at that exact version
- `llama-cpp-python` CUDA rebuild is skipped when the installed build already reports GPU offload support — the slow from-source compile only runs when actually needed

**Parallel worker count for SamePerson halved**
- InsightFace face-embedding worker count is now capped at half the previous value to reduce VRAM pressure when running alongside other models

## Version 2.4.0 - Modular Codebase

**Code split into modules**
- `Hybrid-Scorer.py` reduced from ~9,700 lines to ~838 lines of pure Gradio wiring
- Logic extracted into `lib/` subfolder: `config`, `utils`, `backend`, `scoring`, `helpers`, `state`, `state_helpers`, `loaders`, `view`, and `callbacks/` (scoring, prompts, ui)
- CSS moved to `static/style.css`, JS to `static/app.js`
- No user-facing changes; all behavior preserved

## Version 2.3.8 - Cleaner Sidebar & Threshold Panel

**Smarter Sidebar**
- Opening any sidebar section now automatically closes all others — no more hunting through several open panels at once
- Only **Setup** is open when the app first loads; everything else starts collapsed

**Thresholds Always Visible**
- The Thresholds panel is no longer a collapsible accordion — it is permanently pinned to the bottom of the sidebar so your sliders and histogram are always one glance away
- Export is now section 4 (previously 5)

**Changelog Overlay**
- Click the version number in the top-right corner to open a changelog popup
- Close it with ✕, click outside, or press Escape

**Cleaner Controls**
- Removed the 50% reset buttons next to the sliders; sliders now take the full width of the panel
- Positive prompt box is now twice as tall so longer prompts are easier to read and edit
- Removed the export result text field — it was redundant
- Hidden the prompt-generation status message that was left overlaid on the image section

## Version 2.3.3 - Easier Windows Setup & Clearer Startup

**Faster, Clearer App Startup**
- The app now prints a more helpful startup summary in the console before the Gradio URL appears
- Startup logs now explain the project folder, model storage location, default backends, and that the default PromptMatch model is loaded only when first needed
- PromptMatch no longer downloads or loads its default SigLIP model during app launch; it now waits until the first PromptMatch or LM Search use

**Safer Windows Python Setup**
- `setup_update-windows.bat` now tries much harder to get `Python 3.12` automatically for the user
- If `py -3.12` is missing, the script first tries `winget`, then falls back to downloading the official `python.org` installer
- The script now uses a dedicated `Python 3.12` executable for `venv312` creation instead of trying to make `3.12` the user's normal default Python
- Windows setup now tells the user clearly that this project-specific `Python 3.12` runtime is only for `venv312` and will not replace their usual Python setup

## Version 2.3.0 - Huihui, Better Model Pickers & ImageReward Compatibility

**New Huihui Gemma 4 Option**
- `Huihui Gemma 4 E4B` can now be used in both `Prompt from image` and `LM Search`
- Prompt generation keeps the same 3 output styles: short tags, compact prompt, and fuller prose
- `LM Search` can use Huihui as the deeper rerank backend after PromptMatch builds the shortlist

**Clearer Downloadable Model Pickers**
- All downloadable model dropdowns now show whether a model is already cached
- Green means the model is already on disk
- Orange means the model will need a first download before use

**LM Search Polish**
- Direct numeric LM backends like Huihui and JoyCaption now use a proper `0-100` threshold scale
- PromptMatch shortlist stability was improved for newer Hugging Face output formats

## Version 2.2.5 - TagMatch Polish & Stability

**TagMatch Typing Help**
- The TagMatch tag box now suggests matching booru-style tags in real time while you type
- Suggestions come from the WD tagger vocabulary itself, so it is much easier to discover valid tags without guessing exact spellings
- You can click a suggestion or use arrow keys plus `Enter` / `Tab` to insert it quickly


## Version 2.2.0 - TagMatch & Per-Phrase PromptMatch

**TagMatch — Find AI Human Errors**
- New scoring mode that catches typical AI anatomy mistakes: wrong hands, extra limbs, deformed faces, bad proportions, and more
- Enter comma-separated artifact tags; every image is scored by how strongly the local WD tagger sees them — ships with a ready-to-use default set
- Hover a thumbnail to see each tag's confidence as a color-coded pill (yellow → green) right on the tag textbox
- Re-running with different tags is near-instant — image inference is cached, only the tag lookup changes

**PromptMatch Per-Segment Mode**
- New checkbox splits your prompt at commas and scores each phrase independently
- Hover a thumbnail to see per-phrase match strength as pills on both the positive (yellow → green) and negative (yellow → red) prompt textboxes
- Near-instant — image features are already cached

## Version 2.1.0 - Better Memory & GPU Use

**Smarter GPU Memory Management**
- Switching between scoring modes (e.g. LM Search → ImageReward) now automatically frees the previous mode's models from VRAM before loading the new ones
- Fixes a VRAM leak where JoyCaption and other large models would stay resident after switching modes, causing out-of-memory errors on cards with less than 32 GB VRAM

**Faster LM Search on HF JoyCaption**
- When using the `JoyCaption Beta One` (HF) backend for LM Search reranking, images are now scored in batches instead of one at a time
- Improves GPU utilisation and throughput on the rerank pass
- The GGUF backend is unaffected (sequential as before)

**Proxy Cache in RAM on Linux**
- On Linux, resized proxy thumbnails are now written to `/dev/shm` (a RAM-backed filesystem) instead of the system temp directory, eliminating disk I/O for proxy files
- Falls back silently to the previous temp-directory behaviour if `/dev/shm` is not available or not writable (e.g. some containers)
- Windows and project-mode behaviour are unchanged

## Version 2.0.0 - Faster & Smarter

**Faster Image Scoring**
- Large folders now process much faster
- Better performance monitoring to see if your computer is slow
- Repeat runs are faster because PromptMatch, ImageReward, face search, and LM Search reuse more cached work when possible

**Smarter Face Recognition**
- Find same person searches work better and faster
- Handles images without faces more gracefully

**Better Search Tools**
- `LM Search` , Language Model Search is now part of the main scored workflow
- It first shortlists images with PromptMatch, then reranks the shortlist with a local vision-language model
- JoyCaption GGUF is available as the default local LM Search backend

**Better Prompt Generation**
- Prompt generation now supports `Florence-2`, `JoyCaption Beta One`, and `JoyCaption Beta One GGUF (Q4_K_M)`
- Generated prompts stay in their own editable field until you insert them into the active scoring method

**Better Setup & Caching**
- Setup scripts now standardize on `venv312`
- Linux and Windows setup now install the JoyCaption GGUF runtime automatically
- Setup scripts try a safe git update automatically before refreshing the environment
- Cache defaults are smarter by operating system: Windows keeps model/proxy caches in the project, while Linux keeps the normal system-cache behavior

**Better Interface**
- Left sidebar scrolls independently for easier access
- More intuitive layout
- Thresholds are remembered per mode, so switching methods no longer steals the last slider value from another mode

## Version 1.9.0 - New Features

**New Search Tools**
- Find similar images (see which images look alike)
- Find same person (find all images of the same person)

**Better Controls**
- Smoother threshold sliders
- Settings remember what you last used
- Clearer visual feedback

## Version 1.8.0 - Easier Setup

**Simpler Installation**
- Works better on Windows and Linux
- Dependencies install automatically

**Better File Management**
- Models, and proxy files stored in your project folder
- No more cluttering your system

## Version 1.7.0 - Better Organization

**Easier Image Management**
- Export controls are now above each image group
- Can move instead of just copy images
- Visual indicators show selected/rejected images

**Better Layout**
- Tile size control is more accessible
- More efficient use of screen space

## Version 1.6.0 - Smoother Experience

**Faster Processing**
- More efficient image processing
- Caching helps avoid slow steps

**Better Visuals**
- Clearer histogram feedback
- Hover markers help understand scores

## Version 1.5.0 - Better Setup

**Easier Setup**
- Automatic updates and better setup scripts
- Clearer instructions

**Smarter Scoring**
- Better VRAM guidance
- Better handling of different image models

## Version 1.4.0 - Cleaner Interface

**Simpler Layout**
- Cleaner, more organized interface
- Better spacing and controls

**Improved Gallery**
- Larger thumbnails
- Drag-and-drop between groups
- Better tile size controls

## Version 1.3.0 - Better Prompts

**New Captioning**
- Generate prompts from images
- Three output styles: short tags, compact prompts, detailed descriptions
- Support for lower-end hardware

## Version 1.2.0 - New Tools

**New Features**
- "Fit threshold" tool
- "Prompt from preview image" tool
- More intuitive layout and buttons

## Version 1.1.0 - Unified App

**Combined Features**
- Merged PromptMatch and ImageReward into one app
- Better gallery UI with zoom and hover
- More model choices and faster scoring

## Version 1.0.0 - First Release

**New Tool**
- Combined old separate tools
- Added sorting, review controls, and export folders

## Pre-1.0 - Early Development

**Initial Features**
- Basic setup scripts
- Foundation for sorting and exporting
