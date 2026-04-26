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

**Modular Gradio app.** `Hybrid-Scorer.py` (~838 lines) is pure Gradio wiring; all logic in `lib/`.

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
  callbacks/
    scoring.py     # score_folder, find_similar_images, find_same_person_images
    prompts.py     # generate_prompt_from_preview, run_*_prompt_variant
    ui.py          # handle_thumb_action, export_files, threshold callbacks
static/
  style.css / app.js
```

`create_app()` owns shared `state`, binds callbacks via `functools.partial`, builds UI, injects JS/CSS. `.select()` handlers using `gr.SelectData` must be thin wrapper closures — `partial` hides the annotation from Gradio. Always inspect both Python and injected JS before changing UI logic.

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

**Sidebar:** 4 mutually-exclusive accordions (`#hy-acc-setup/scoring/search-image/export`); only Setup open on load. Permanent thresholds panel (`#hy-thresholds-panel`) pinned via `position:sticky; bottom:0`. Accordion JS (`hookSidebarAccordionBehavior`) targets Gradio 6.x `button.label-wrap` with `open` class — do not revert to `<details>/<summary>`.

**Changelog overlay:** `#hy-version-btn` opens `#hy-changelog-overlay`; `CHANGELOG.md` read at startup → `APP_CHANGELOG_HTML`; shown/hidden via `.open` class; `hookChangelogOverlay()` is idempotent.

**Hidden JS bridge:** `hy-thumb-action`, `hy-shortcut-action`, `hy-mark-state`, `hy-model-status`, `hy-hist-width` (hidden Gradio components). The three 50% midpoint buttons (`main_mid_btn`, `aux_mid_btn`, `percentile_mid_btn`) are `visible=False` — do not remove, they're in callback output lists.

## Editing Rules
- Be conservative with refactors; don't change callback return tuple sizes without checking all output arities.
- Search both Python and injected JS before changing UI behavior.
- Preserve manual override, threshold, export, and cache behavior unless explicitly asked.
- `docs/architecture.md` → source of truth for structure; `docs/behavior-notes.md` → UX invariants. Update both when relevant.

## High-Risk Areas
- Gradio callback return signatures (must match output count exactly)
- Hidden `hy-*` JS bridge wiring
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
