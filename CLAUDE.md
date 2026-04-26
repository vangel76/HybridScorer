# CLAUDE.md

## Commands
```bash
python -m py_compile Hybrid-Scorer.py   # syntax check
./run-Hybrid-Scorer.sh                  # run (Linux)
run-Hybrid-Scorer-windows.bat           # run (Windows)
./setup_update-linux.sh                 # setup/update (Linux)
setup_update-windows.bat                # setup/update (Windows)
```
Default: `http://localhost:7862`

## Architecture

**FastAPI + Tabler app.** `Hybrid-Scorer.py` starts the local FastAPI server; state, DTOs, jobs, and route adapters live in `lib/web_context.py`; scoring/model/cache logic stays in `lib/`.

```
lib/
  config.py        # constants, method labels, model IDs
  utils.py         # pure utilities
  backend.py       # ModelBackend class
  scoring.py       # score_all, encode_all_promptmatch_images
  helpers.py       # UI text, score, prompt helpers
  state.py         # get_state_defaults(), init_state()
  state_helpers.py # state-management helpers
  loaders.py       # ensure_*_model, ensure_*_backend, feature caches
  view.py          # gallery, histogram, current_view, render_view_with_controls
  web_context.py   # FastAPI-facing context, DTO rendering, WebSocket jobs, media registry
  ui_compat.py     # small gr.update/gr.skip compatibility shim, no Gradio runtime
  callbacks/
    scoring.py     # score_folder, find_similar_images, find_same_person_images
    prompts.py     # generate_prompt_from_preview, run_*_prompt_variant
    ui.py          # handle_thumb_action, export_files, threshold callbacks
static/
  tabler-app.css / tabler-app.js / vendor/tabler/
templates/
  index.html / setup_required.html
```

`HybridScorerContext` owns shared `state`, starts serialized jobs for long GPU work, streams progress over WebSockets, and renders JSON view/control state. Always inspect both `lib/web_context.py` and `static/tabler-app.js` before changing UI logic.

### Scoring modes

| Mode | Backend | Cache |
|---|---|---|
| PromptMatch | CLIP-family (open_clip, SigLIP) | per-folder image embeddings |
| ImageReward | ImageReward-v1.0 | per-folder base + penalty scores |
| Similarity | reuses PromptMatch embeddings | same as PromptMatch |
| SamePerson | InsightFace buffalo_l (ONNX) | per-folder face embeddings |
| LLM Search | PromptMatch shortlist → VLM rerank | embeddings + LLM captions |
| ObjectSearch | DINOv2 ViT-B/14 patch + FAISS/GPU matmul | per-folder patch embeddings (256/image) |

**LLM Search:** PromptMatch shortlists → Florence-2/JoyCaption HF/GGUF reranks; non-shortlisted get reject-floor score. HF uses `score_candidates_batch` (batch=4); GGUF sequential.

**ObjectSearch:** query image set in accordion 3 → `ensure_objectsearch_feature_cache` builds `faiss.IndexFlatIP` + GPU tensor → `score_objectsearch_cached_features` runs `torch.mm` or FAISS fallback → mean best-match-per-patch score. State keys: `os_cached_*`, `dinov2_backend`, `objectsearch_query_fname/source`.

**VRAM:** `release_inactive_gpu_models(target_method)` called before each scoring entry point; frees unneeded models. PromptMatch CLIP never released (always needed for shortlisting).

**Cache:** Windows → project-local (`models/`, `cache/`); Linux → system (`~/.cache/`). Override: `HYBRIDSCORER_CACHE_MODE=project|system`. Linux proxy thumbnails → `/dev/shm` (tmpfs) with fallback. `get_cache_config()` is `@lru_cache(maxsize=1)`.

**Sidebar:** 4 mutually-exclusive Tabler accordion sections; only Setup open on load. Permanent thresholds panel is sticky at the bottom of the sidebar.

**Changelog overlay:** version button opens the Tabler modal; `CHANGELOG.md` is read at startup and embedded in initial state.

## Editing Rules
- Be conservative with refactors; don't change route/DTO wire shape without updating frontend and tests.
- Search both Python route/context code and frontend JS before changing UI behavior.
- Preserve manual override, threshold, export, and cache behavior unless explicitly asked.
- `docs/architecture.md` → source of truth for structure; `docs/behavior-notes.md` → UX invariants. Update both when relevant.

## High-Risk Areas
- FastAPI route contracts and DTO shape
- WebSocket job final-state/progress delivery
- Shared `state` dict keys
- Filename-based manual override logic (survives rescoring)
- Proxy/cache synchronization
- LLM Search shortlist and rerank flow
- CPython refcount for GGUF `Llama` objects (no `__del__` until loop vars explicitly `del`-ed)

## Smoke Tests
- Rescore same folder → pinned images stay pinned
- Change PromptMatch prompts → cached rerun works
- Change ImageReward penalty weight → cached pass reused
- Preview image → run Similarity; face image → run SamePerson
- LLM Search twice same settings → cache reused
- Generate prompt → insert into PromptMatch, ImageReward, LLM Search
- Windows project-mode caches → repo-local; Linux system-mode → system locations
