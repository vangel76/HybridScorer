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
  scoring.py       # _run_promptmatch_batches (shared loop), score_all, encode_all_promptmatch_images
  helpers.py       # UI text, score, prompt helpers
  state.py         # get_state_defaults(), init_state()
  state_helpers.py # state-management helpers
  loaders.py       # _ensure_hf_vlm (shared), ensure_*_model, ensure_*_backend, feature caches
  view.py          # gallery, histogram (shared draw_chart), current_view, render_view_with_controls
  web_context.py   # FastAPI-facing context, DTO rendering, WebSocket jobs, media registry (LRU)
  ui_compat.py     # gr.update/gr.skip shim; Update uses __getattr__ — no double __dict__ storage
  callbacks/
    scoring.py     # score_folder, _run_preview_search (shared), find_similar/same_person/objectsearch
    prompts.py     # generate_prompt_from_preview, run_*_prompt_variant, _resolve_generate_params
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

**TagMatch:** `_prep_tagmatch_batch(paths, proxy_map)` pure helper preps one batch (load + pad + resize + BGR convert). `score_tagmatch_folder` runs a `ThreadPoolExecutor(max_workers=1)` prefetch loop: submits batch N+1 prep while ONNX runs batch N — overlaps I/O+CPU with GPU inference.

**LLM Search:** PromptMatch shortlists → Florence-2/JoyCaption HF/JoyCaption NF4/Huihui Gemma reranks; non-shortlisted get reject-floor score. HF JoyCaption backends use `score_candidates_batch` (batch=4). All four HF VLM loaders share `_ensure_hf_vlm(state, cache_key, load_fn)` — do not duplicate the 6-step load pattern.

**ObjectSearch:** query image set in accordion 3 → `ensure_objectsearch_feature_cache` builds `faiss.IndexFlatIP` + GPU tensor → `score_objectsearch_cached_features` runs `torch.mm` or FAISS fallback → mean best-match-per-patch score. Inner patch-matching loop uses `np.maximum.at` (vectorized) — do not revert to pure-Python inner loop. State keys: `os_cached_*`, `dinov2_backend`, `objectsearch_query_fname/source`.

**VRAM:** `release_inactive_gpu_models(target_method)` called before each scoring entry point; frees unneeded models. PromptMatch CLIP never released (always needed for shortlisting). VLM backend (`prompt_backend_cache`) is NOT wiped when `target_method == METHOD_LLMSEARCH` — the loaded VLM must stay resident for rerank.

**JoyCaption NF4:** requires `accelerate` + `bitsandbytes`. Loader uses `device_map={"": device}` and disables SigLIP `vision_tower.use_head`; LLaVA uses hidden states, and the pooled head can crash with pre-quantized NF4 packed weights (`mat1 and mat2 shapes cannot be multiplied`). HF sharded-cache checks must verify every shard in `model.safetensors.index.json`, not just the index file, or partial downloads get misdetected as disk-cache-ready.

**No llama.cpp/GGUF JoyCaption path:** JoyCaption prompt generation and LM Search now use Transformers HF/NF4 backends. Do not add `llama-cpp-python` setup or GGUF runtime assumptions back unless explicitly reintroducing that backend.

**Cache:** Windows → project-local (`models/`, `cache/`); Linux → system (`~/.cache/`). Override: `HYBRIDSCORER_CACHE_MODE=project|system`. Linux proxy thumbnails → `/dev/shm` (tmpfs) with fallback. `get_cache_config()` is `@lru_cache(maxsize=1)`.

**Gallery UI:** Square thumbnails (`aspect-ratio:1`), zero gap/padding, no border, no border-radius, no figcaption — pure image grid. Thumb-size slider in header (180–512 px, drives `--thumb-size` CSS var → `minmax(var(--thumb-size),1fr)` grid). Left gallery header green-tinted, right red-tinted. "Move here" buttons in each gallery card-header. Drag-drop between galleries pins image to target side (same as move-left/move-right). Gallery containers are drop zones; thumb-level drop still handled per-figure. **Multi-drag:** dragstart on a marked (shift-clicked) thumb packs all `.hy-thumb.marked` filenames from that gallery into `fnames[]` in the dataTransfer payload; unmarked drags send `fnames:[item.filename]`. Both drop targets (thumb and gallery container) use `src.fnames || [src.filename]`.

**Zoom overlay:** click thumb → `#hy-zoom-overlay` (position:absolute inside `.hy-main`). Backdrop or Escape closes. Does not cover sidebar.

**Segment/tag pills:** hover thumb → pos_prompt textarea and tagmatch_tags textarea get `#pm-segment-pills` / `#tm-segment-pills` overlays (position:absolute, pointer-events:none). Tag text transparent on textarea while overlay active. Colors: pale-transparent yellow (low) → bright green (high), min-max normalized per image. Data from `state.view.segment_score_lookup` (PromptMatch per-segment) and `state.view.tag_score_lookup` (TagMatch).

**Histogram:** positive/main chart always drawn flipped (green=high on left). `hist_geom` stores real `pos_lo`/`pos_hi` values + `pos_flipped:True` flag. Single-hist path stores `flipped:True`. JS `placeValueLine` and `showHistogramHover` use `1-frac` when flag set. `on_hist_click` uses `hi - frac*(hi-lo)` when flipped. `main_threshold` slider has `transform:scaleX(-1)` in CSS. `_slider_state` reads `pos_lo`/`pos_hi` as slider min/max (real values, not swapped — flip is display-only).

**Folder loading:** Two buttons side-by-side (`col-8` / `col-4`): "Load folder" → `POST /api/folder/load`; "+subfolders" → `POST /api/folder/load-recursive`. Both call `load_folder_job(recursive=True/False)` → `load_folder_for_browse(recursive=)`. Sets `state["folder_recursive"]` on success. `prepare_scored_run_context` and `normalize_preview_search_request` check `state.get("folder_recursive")` to pick `scan_image_paths_recursive` vs `scan_image_paths`. `scan_image_paths_recursive` uses `os.walk`.

**Browser auto-open:** `@app.on_event("startup")` fires `webbrowser.open(f"http://localhost:{port}")` after 0.1 s.

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
- JoyCaption NF4 loader compatibility with Transformers/bitsandbytes/SigLIP internals

## Smoke Tests
- Rescore same folder → pinned images stay pinned
- Change PromptMatch prompts → cached rerun works
- Change ImageReward penalty weight → cached pass reused
- Preview image → run Similarity; face image → run SamePerson
- LLM Search twice same settings → cache reused
- Generate prompt → insert into PromptMatch, ImageReward, LLM Search
- Generate prompt with JoyCaption Beta One NF4 → first load completes, prompt text appears, no WebSocket stall
- Windows project-mode caches → repo-local; Linux system-mode → system locations
- PromptMatch per-segment + hover thumb → pills appear on pos_prompt overlay, colors normalized
- TagMatch score + hover thumb → pills appear on tagmatch_tags overlay
- Drag thumb left→right → image moves to right bucket; drag right→left → moves to left
- Shift-click multiple thumbs → drag any marked one → all selected move together
- Load folder with +subfolders → images from nested folders appear; scoring runs on full recursive list
- +subfolders load → rescore → `folder_recursive` persists, no re-scan of only top-level folder
- Click histogram → threshold line and slider update to match click position (flipped chart: left=high)
- Drag main threshold slider → line moves correctly (slider visually flipped, line mirrors)
- App start → browser opens automatically
