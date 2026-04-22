# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Syntax check (fast, no GPU needed)
python -m py_compile Hybrid-Scorer.py

# Run the app
./run-Hybrid-Scorer.sh        # Linux
run-Hybrid-Scorer-windows.bat # Windows

# Setup / update environment
./setup_update-linux.sh       # Linux
setup_update-windows.bat      # Windows
```

Default app URL: `http://localhost:7862`

## Architecture

**Modular Gradio app.** `Hybrid-Scorer.py` (~838 lines) is pure Gradio wiring; all logic lives in `lib/`.

```
lib/
  config.py          # constants, method labels, model IDs
  utils.py           # pure utility functions
  backend.py         # ModelBackend class
  scoring.py         # score_all, encode_all_promptmatch_images
  helpers.py         # UI text helpers, score helpers, prompt helpers
  state.py           # get_state_defaults(), init_state()
  state_helpers.py   # state-management helpers (accept state param)
  loaders.py         # ensure_*_model, ensure_*_backend, feature caches
  view.py            # gallery, histogram, current_view, render_view_with_controls
  callbacks/
    scoring.py       # score_folder, find_similar_images, find_same_person_images
    prompts.py       # generate_prompt_from_preview, run_*_prompt_variant
    ui.py            # handle_thumb_action, export_files, threshold callbacks, etc.
static/
  style.css          # all app CSS
  app.js             # all app JS (tooltip dict injected at runtime)
```

`create_app()` is the center of gravity:
- owns the shared mutable `state` dict
- binds extracted callbacks with `functools.partial(func, state)` or `partial(func, state, device)`
- builds the UI and injects JS/CSS
- creates initial backend/model objects

**Note:** `.select()` event handlers that use `gr.SelectData` must be defined as thin wrapper closures in `create_app()` rather than bound with `partial` — Gradio cannot see the `SelectData` annotation through a partial.

UI behavior is callback-driven. Some behavior lives in injected JS, not Python. Always inspect both before changing UI logic.

### Six scoring modes

| Mode | Backend | Cache used |
|---|---|---|
| **PromptMatch** | CLIP-family (open_clip, SigLIP, etc.) | per-folder image embeddings |
| **ImageReward** | ImageReward-v1.0 | per-folder base + penalty scores |
| **Similarity** | reuses PromptMatch embeddings | same as PromptMatch |
| **SamePerson** | InsightFace buffalo_l (ONNX) | per-folder face embeddings |
| **LLM Search** | PromptMatch shortlist → vision-language rerank | PromptMatch embeddings + LLM captions |
| **ObjectSearch** | DINOv2 ViT-B/14 patch features + FAISS (CPU) / GPU matmul | per-folder patch embeddings (256 patches/image, in-memory) |

### LLM Search flow
1. PromptMatch builds a shortlist from the text query
2. A local vision-language backend reranks the shortlist (Florence-2, JoyCaption HF, or JoyCaption GGUF)
3. Non-shortlisted images get a deterministic reject-floor score
4. HF JoyCaption uses batched inference (`score_candidates_batch`, `LLMSEARCH_JOYCAPTION_HF_BATCH_SIZE=4`); GGUF is sequential

### ObjectSearch flow
1. User sets a query image (upload, paste, or gallery preview) in accordion 3
2. `ensure_objectsearch_feature_cache` extracts DINOv2 patch tokens for the whole folder, builds a `faiss.IndexFlatIP` CPU index and (if CUDA available) a GPU tensor copy of all patches
3. `encode_single_objectsearch_query` extracts patch tokens from the query image
4. `score_objectsearch_cached_features` runs GPU `torch.mm` (or FAISS CPU fallback), aggregates best-match-per-query-patch per gallery image, returns mean score
5. Scores feed into the standard threshold/split/export flow — uses `uses_similarity_topn` (top-N slider) same as Similarity and SamePerson
- State keys: `os_cached_*`, `dinov2_backend`, `objectsearch_query_fname/source`
- `release_inactive_gpu_models` clears `dinov2_backend`, `os_cached_faiss_index`, and `os_cached_patch_gpu_tensor`

### VRAM management
`release_inactive_gpu_models(target_method)` is called at the top of `score_folder`, `find_similar_images`, `find_same_person_images`, and `find_objectsearch_images` — before any model loads. It frees models not needed for the incoming method. The PromptMatch CLIP `state["backend"]` is never released because it is always needed for shortlist embeddings.

### Cache and proxy system
- Cache mode defaults: Windows → project-local (`models/`, `cache/`); Linux → system (`~/.cache/...`)
- Override: `HYBRIDSCORER_CACHE_MODE=project` or `HYBRIDSCORER_CACHE_MODE=system`
- On Linux (system mode), proxy thumbnails go to `/dev/shm` (RAM-backed tmpfs) with fallback to `tempfile.gettempdir()`
- `get_cache_config()` is `@lru_cache(maxsize=1)` — runs once per session

### Sidebar layout
The sidebar has four collapsible accordion sections (mutually exclusive — opening one closes all others; only **1. Setup** is open on first load):
- `#hy-acc-setup` — 1. Setup
- `#hy-acc-scoring` — 2. Scoring & Method/Settings
- `#hy-acc-search-image` — 3. Search + Prompt from image
- `#hy-acc-export` — 4. Export

Below the accordion scroll area is a permanent **Thresholds panel** (`#hy-thresholds-panel`, `.thresholds-panel`) pinned to the bottom of the sidebar via `position:sticky; bottom:0`. It always stays visible and is never collapsible.

The accordion JS (`hookSidebarAccordionBehavior`) targets Gradio 6.x's button-based accordion structure (`button.label-wrap` with an `open` class when expanded). Gradio 6 does not use `<details>`/`<summary>` — do not revert to that approach.

### Changelog overlay
The version tag (`v2.4.0`) in the app header is a `<button id="hy-version-btn">` that opens a modal overlay (`#hy-changelog-overlay`). `CHANGELOG.md` is read at startup by `load_changelog()`, HTML-escaped into `APP_CHANGELOG_HTML`, and embedded in the page. The overlay is shown/hidden via the `.open` CSS class; `hookChangelogOverlay()` attaches the listeners (idempotent, guarded by `dataset.changelogHooked`).

The three 50% midpoint buttons (`main_mid_btn`, `aux_mid_btn`, `percentile_mid_btn`) are kept as hidden Gradio components (`visible=False`). Do not remove them — they appear in callback output lists and removing them would break output arities.

### Hidden JS bridge elements
The UI wires Python ↔ JS through hidden Gradio components with these ids:
- `hy-thumb-action`, `hy-shortcut-action`, `hy-mark-state`, `hy-model-status`, `hy-hist-width`

## Editing rules (from AGENTS.md)

- **Be conservative with refactors.** This is an intentionally large single-file app.
- **Do not change callback return tuple sizes** without verifying every connected output's arity.
- **Search both Python and injected JS** before changing UI behavior.
- **Preserve** manual override behavior, threshold behavior, export behavior, and cache behavior unless explicitly asked.
- `docs/architecture.md` is the source of truth for app structure. Update it when changing structure, state, caches, workflows, or major functions.
- `docs/behavior-notes.md` is the source of truth for UX invariants. Update it when changing UX behavior or fragile constraints.

## High-risk areas

- Gradio callback return signatures (must match output count exactly)
- Hidden `hy-*` JS bridge wiring
- Shared `state` dict keys
- Filename-based manual override logic (overrides survive rescoring in same folder)
- Proxy/cache synchronization
- LLM Search shortlist and rerank flow
- CPython refcount behavior for GGUF `Llama` objects (no `__del__` fires until loop vars are explicitly `del`-ed)

## Smoke tests (manual)

- Rescore same folder → pinned images stay pinned
- Change PromptMatch prompts → cached rerun works
- Change ImageReward penalty weight → cached pass reused
- Preview an image → run Similarity
- Preview a face image → run SamePerson
- Run LLM Search twice same settings → cache reused
- Generate a prompt → insert into PromptMatch, ImageReward, LLM Search
- Confirm Windows project-mode caches land in repo-local paths
- Confirm Linux system-mode caches stay in system locations
