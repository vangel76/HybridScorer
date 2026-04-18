# Changelog

## [2.3.3] - 2026-04-18

- Added a new lightweight startup log sequence before `app.launch(...)` so the console now reports:
  - app title
  - project directory
  - default input folder
  - cache mode / project model store path
  - default PromptMatch model, prompt generator, and LM Search backend
  - dependency-check and UI-build progress
- Changed the startup wording on Windows project mode from `Hugging Face cache` to `Project model store` so the console reflects the intended packaging model (`models/huggingface` under the repo) instead of sounding like an external user-managed cache.
- Removed eager construction of the default PromptMatch backend from `create_app()`. The app no longer instantiates `ModelBackend(... siglip-so400m-patch14-384 ...)` during UI startup; `state["backend"]` now starts as `None` and the default PromptMatch / LLM Search shortlist dropdowns use a new `DEFAULT_PROMPTMATCH_MODEL_LABEL` constant instead.
- Updated `label_for_backend()` to tolerate `None` and return the default PromptMatch label cleanly for UI display.
- Fixed the lazy-load regression in `ensure_promptmatch_backend_loaded(...)`: the first PromptMatch run on the default model previously skipped loading because the UI default label matched even while `state["backend"]` was still `None`. The loader now explicitly loads when no backend instance exists yet.
- Expanded `setup_update-windows.bat` so Python 3.12 bootstrapping is more automatic:
  - still handles the original `py` launcher missing case
  - now also handles the case where `py` exists but `py -3.12` is unavailable
  - first tries `winget install -e --id Python.Python.3.12`
  - falls back to downloading and silently running the official `python.org` installer when `winget` is unavailable or fails
- Added configurable Windows Python installer knobs in `setup_update-windows.bat`:
  - `PYTHON312_VERSION`
  - `PYTHON312_INSTALLER_URL`
  - `PYTHON312_INSTALL_ARGS`
- Changed the Windows Python install strategy to avoid taking over the user's normal interpreter setup:
  - installer arguments now use `PrependPath=0` and `Include_launcher=0`
  - the setup script resolves a dedicated Python 3.12 executable path and uses that exact binary for `venv312` creation
  - setup output now explicitly tells the user that this Python 3.12 runtime is only for the project's `venv312` and will not replace the normal default Python
- Updated the Windows venv-creation messaging so failures now say `Python 3.12 is available` instead of the older misleading wording.

## [2.3.0] - 2026-04-17

- Added `Huihui Gemma 4 E4B` (`huihui-ai/Huihui-gemma-4-E4B-it-abliterated`) as a new multimodal backend in both `Prompt from image` and `LM Search`.
- Extended prompt-generation backend registration so Huihui appears in `PROMPT_GENERATOR_ALL_CHOICES` and in the LM Search backend choices, while leaving the existing default backends unchanged.
- Added a dedicated torch-backed Huihui loader path built on Hugging Face Transformers multimodal APIs and cached under `state["prompt_backend_cache"]`. The loader:
  - prefers CUDA `bfloat16` when supported, otherwise `float16`
  - validates that the installed Transformers build actually exposes Gemma 4 runtime classes
  - tries the finetuned repo processor metadata first and falls back to the base Gemma 4 processor assets when needed
  - raises backend-specific setup errors instead of failing silently
- Added Huihui-specific prompt generation and LM Search rerank helpers:
  - prompt generation maps the existing detail levels to short tags / compact prompt / fuller prose
  - LM Search uses Huihui as a direct numeric reranker over the PromptMatch shortlist
  - current Huihui LM Search inference stays sequential by design (no batch pre-pass yet)
- Added Huihui output cleanup to strip echoed chat-template scaffolding such as `system`, `user`, and `model` lines before prompt insertion or numeric score parsing.
- Fixed LM Search threshold-range handling for direct numeric rerank backends (`JoyCaption Beta One`, `Huihui Gemma 4 E4B`). The LM Search threshold slider, midpoint helper, saved-threshold recall, and fit-threshold behavior now use the rerank backend's real score range (`0..100`) instead of snapping back to PromptMatch-style shortlist similarity values.
- Extended cached-vs-download UI markers to all downloadable model selectors, not just the main PromptMatch dropdown. The injected JS availability painter now covers `#hy-model`, `#hy-llm-model`, `#hy-llm-backend`, and `#hy-prompt-generator`.
- Added cache-status-aware dropdown choice builders for prompt-style backends and WD TagMatch, plus tooltip help text explaining the green cached / amber first-download marker scheme.
- Updated PromptMatch shortlist compatibility with newer Hugging Face output objects:
  - text-feature encoding now unwraps `BaseModelOutputWithPooling`-style returns before normalization
  - the single-image retry path now unpacks `prepare_promptmatch_loaded_batch(...)` correctly instead of failing with `too many values to unpack (expected 3)`
- Raised the Transformers floor in `requirements.txt` to `transformers>=5.5.0,<6` so Gemma 4-capable installs are the default after rerunning setup.
- Added a broad ImageReward-on-Transformers-5 compatibility layer inside `get_imagereward_utils()` rather than patching the installed `ImageReward` package:
  - restore legacy imports expected by ImageReward BLIP code (`apply_chunking_to_forward`, `prune_linear_layer`)
  - provide a compatibility implementation of `find_pruneable_heads_and_indices`, which is gone in Transformers 5
  - add a legacy `BertTokenizer.additional_special_tokens_ids` shim so BLIP tokenizer setup can still assign `enc_token_id`
  - add `PreTrainedModel.all_tied_weights_keys` compatibility for older BLIP `BertModel` subclasses running under newer `tie_weights()` logic
  - add legacy `get_head_mask()` / `_convert_head_mask_to_5d()` helpers required by old BLIP forward passes
  - reuse already-downloaded local `ImageReward.pt` and `med_config.json` files before calling back into `hf_hub_download()`
  - convert ImageReward import failures from `sys.exit(...)` to normal runtime exceptions so a bad import no longer tears down the Gradio server
- Updated user-facing docs (`README.md`, `docs/architecture.md`, `docs/behavior-notes.md`) and bumped `VERSION` to `2.3.0`.

## [2.2.5] - 2026-04-16

- Added live TagMatch tag suggestions backed by the WD tagger vocabulary file (`selected_tags.csv`). New state keys: `tagmatch_vocab_tags` caches the parsed tag list in Python, and `tagmatch_vocab_json` mirrors that list into the hidden Gradio bridge textbox `#hy-tagmatch-vocab` so frontend JS can filter suggestions locally without round-tripping through Python on every keystroke.
- Added `load_tagmatch_vocabulary()` to fetch and parse `selected_tags.csv` independently of ONNX session creation, and `refresh_tagmatch_vocab_state(method)` to lazily populate the hidden vocab bridge when the user switches into `METHOD_TAGMATCH`.
- Added a new frontend TagMatch autocomplete system inside the injected JS:
  - `readTagMatchVocabulary()` reads the hidden JSON payload
  - `getTagMatchTokenInfo()` isolates the current comma-delimited tag fragment at the textarea caret
  - `rankTagMatchSuggestions()` prefers prefix matches, then substring matches, and limits output to `TAGMATCH_AUTOCOMPLETE_MAX_SUGGESTIONS`
  - `applyTagMatchSuggestion()` replaces only the active tag fragment and preserves the rest of the comma-separated query
  - keyboard controls support `ArrowUp`, `ArrowDown`, `Enter`, `Tab`, and `Escape`
- The first implementation rendered the suggestions panel inside `#hy-tagmatch-tags`, which still allowed it to be visually buried behind neighboring Gradio controls when parent containers clipped overflow. Reworked it to a body-mounted floating panel:
  - `ensureTagMatchSuggestBox()` now creates a single `#hy-tagmatch-suggest-box` under `document.body`
  - `positionTagMatchSuggestBox()` positions it from the textarea's `getBoundingClientRect()`
  - the popup now uses `position: fixed` and a high `z-index`, avoiding local stacking/overflow traps in the sidebar layout
  - resize and scroll listeners resync popup placement while it is active
- Added TagMatch-specific tooltip help text describing the new typing workflow: live suggestions plus click or arrow-key insertion.
- Hardened TagMatch slider handoff when switching methods. Rare Gradio crashes were triggered by a stale PromptMatch negative slider value such as `-0.002` arriving after the UI had already switched TagMatch to a `0..100` range. Changes:
  - introduced `TAGMATCH_SLIDER_PREPROCESS_MIN = -0.01` as a tiny tolerance floor for Gradio preprocessing only
  - added `normalize_threshold_inputs(method, main_threshold, aux_threshold)` and applied it in both `prepare_scored_run_context()` and `current_view()` so TagMatch logic always clamps thresholds back into valid `0..100` bounds before using them
  - `configure_controls(METHOD_TAGMATCH)` now updates both sliders with the preprocess-safe minimum instead of a strict `0.0`
  - the TagMatch branch in `score_folder()` now clamps the expanded lower slider bound with `safe_lo = max(TAGMATCH_SLIDER_PREPROCESS_MIN, safe_lo)` so stale negative values cannot widen the real UI range below the tolerated floor
- Made the `method_dd.change(fn=configure_controls, ...)` callback `queue=False` so mode-switch slider-range updates happen immediately instead of lagging behind queued events, reducing the chance of stale slider payloads racing the new mode configuration.

## [2.2.0] - 2026-04-15

- Updated `TAGMATCH_DEFAULT_TAGS` to only include tags confirmed present in the WD eva02-large-tagger-v3 ONNX model output layer. Removed `missing_hand`, `extra_legs`, and `missing_foot` (not in vocabulary); replaced with `bad_hands`, `bad_feet`, `bad_proportions`, `extra_arms`, `extra_faces`, `extra_mouth`, `missing_limb`, `multiple_legs`, `multiple_heads`, `oversized_limbs`, `wrong_foot`, `artistic_error`.
- Fixed TagMatch histogram hover marker: `usesPositiveSimilarityChart` in injected JS was missing `"TagMatch"` and `"LM Search"`, so no hover line was drawn for those modes. Added both to the array so hovering a thumbnail shows that image's score on the TagMatch histogram.
- Added per-tag probability pill overlay for TagMatch. When hovering a thumbnail in TagMatch mode, an absolutely-positioned overlay appears on top of the `#hy-tagmatch-tags` textarea showing each queried tag as a colored pill (yellow → green by probability, label shows `tag  XX%`). The textarea fades to near-invisible (opacity 0.05) and the Gradio label is hidden via CSS `:has()` while the overlay is active.
- Added per-segment scoring mode to PromptMatch. A new checkbox "Per-segment scoring (hover shows per-phrase match)" inside the PromptMatch group splits the positive prompt by commas at score time, encodes each phrase independently via the cached CLIP backend, and aggregates the final score as the **sum** of per-segment similarities. Image features are already cached — only cheap text encodings are recomputed. State keys added: `pm_segment_mode` (bool), `pm_segment_sims` (dict).
- Extended per-segment scoring to the negative prompt as well. Each comma-separated negative phrase is encoded separately; the aggregate neg score is the sum of per-segment neg similarities.
- Added `segment_score_lookup` and `neg_segment_score_lookup` to `marked_state_json()`, serialized into the `hy-mark-state` JSON bridge so JS can read per-segment similarities per filename without a round-trip.
- Added per-segment pill overlays on the positive (`#hy-pos`) and negative (`#hy-neg`) prompt textboxes. On thumbnail hover in PromptMatch per-segment mode, pills appear over both textboxes. Positive pills use relative yellow→green coloring (hue 55→120); negative pills use relative yellow→red coloring (hue 55→0). Color is normalized within the image's own segment range so relative matching is always visible.
- Refactored injected JS overlay helpers into `ensureOverlay(elemId)` and `hideOverlay(elemId)` shared utilities. Extended `syncTagMatchPills()` to drive all three overlays (TagMatch tags, PromptMatch pos, PromptMatch neg) from one function.
- CSS: added `#hy-pos` and `#hy-neg` to the overlay positioning rules (`position: relative`, textarea fade, label hide-on-active).
- Added `pm_segment_cb` to `_score_folder_inputs`, `score_folder()` signature, `handle_shortcut_action()` signature, and the `shortcut_action.change()` inputs list.

## [2.1.0] - 2026-04-12

- `release_inactive_gpu_models(target_method)` was fully implemented but never called, leaving all loaded models (JoyCaption GGUF, Florence-2, ImageReward, InsightFace) permanently resident in VRAM across mode switches. Added the call at the entry of `score_folder`, `find_similar_images`, and `find_same_person_images`, before any model loads, so each scoring entry point evicts models that are not needed for the incoming method.
- Fixed a reference-counting bug in `release_inactive_gpu_models` where Python loop variables `backend_name` and `cached` held the last dict value alive after `state["prompt_backend_cache"] = {}` cleared the state reference. The llama_cpp `Llama` object has no `"model"` key so `_clear_torch_model` silently skipped it; the object's refcount never reached zero before `gc.collect()` and `torch.cuda.empty_cache()` ran. Added explicit `del backend_name, cached` after the loop so CPython's refcount drops to zero in time for the gc and CUDA empty-cache calls to be meaningful.
- Added `score_candidates_batch()` to `VisionLLMRerankBackend` for the HF JoyCaption backend (`LlavaForConditionalGeneration`). It builds one batched `processor()` call with `padding=True` and a single `model.generate()` over N images, then decodes each output individually. The GGUF backend falls back to sequential `score_candidate()` calls since `llama_cpp` does not support multi-image batched generation via `create_chat_completion`.
- Added a batch pre-pass in `score_llmsearch_candidates` that runs before the main per-image result loop. When `backend_id == PROMPT_GENERATOR_JOYCAPTION`, all uncached candidate paths are collected, loaded, and scored through `score_candidates_batch` in chunks of `LLMSEARCH_JOYCAPTION_HF_BATCH_SIZE` (default 4), with results written directly to `caption_cache`. The existing sequential loop then reads from cache for pre-scored images and falls back to single-image inference for any that the batch pass failed to produce.
- Added `_system_proxy_root()` which checks `os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)` and returns `/dev/shm` if both pass, otherwise `tempfile.gettempdir()`. The system-mode branch of `get_cache_config()` now uses this for `proxy_root` instead of hardcoding `tempfile.gettempdir()`, so proxy thumbnails on a standard Linux desktop land in the RAM-backed tmpfs rather than on disk. The Windows project-mode branch is unchanged. `get_cache_config()` is `@lru_cache(maxsize=1)` so the check runs once per session.
- Merged `requirements-gguf.txt` (single line: `llama-cpp-python>=0.3.7`) into the bottom of `requirements.txt` and deleted the separate file. Both setup scripts now reference the package spec inline in the `CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall --no-cache-dir` step instead of via `-r requirements-gguf.txt`, preserving the CUDA build behaviour without the extra file.

## [2.0.0] - 2026-04-11

- Greatly sped up large-folder PromptMatch scoring by parallelizing image decode/preprocess work, adding one-batch-ahead host prefetch, and tuning the PromptMatch autobatch logic with better GPU-tier heuristics.
- Greatly sped up large-folder ImageReward scoring with the same style of threaded host preprocessing, prefetching, and a dedicated ImageReward autobatch policy.
- Added per-batch timing logs for PromptMatch and ImageReward plus free-VRAM reporting, which makes it much easier to see whether a run is CPU-bound or GPU-bound.
- Improved same-person search throughput by allowing more InsightFace workers when VRAM headroom exists, while keeping safer defaults for smaller 8 GB / 12 GB class GPUs.
- Reduced same-person terminal noise by muting ONNX Runtime provider/session info spam and trimming the custom progress logging down to the useful parts.
- Added an end-to-end same-person embedding-pass timing line so large face-search runs can be measured directly in seconds.
- Fixed same-person follow-up regressions so the first `Find same person` run on a folder works correctly again and PromptMatch/ImageReward keep working normally afterward.
- Thresholds are now remembered per mode, so returning to PromptMatch, ImageReward, Similarity, or SamePerson restores that mode's own last threshold instead of inheriting the previous mode's slider value.
- Reworked the main desktop layout so the full left control sidebar scrolls independently while the page itself stays fixed, which makes the lower setup panels reachable without dragging the entire gallery area around.

## [1.9.0] - 2026-04-11

- Added `Find similar images` as a new preview action that reuses cached PromptMatch image embeddings to rank the current folder by image-image similarity from the selected preview image.
- Reworked the preview-actions panel so similarity and prompt generation each have their own clearer section, and renamed the prompt action to `Prompt from image`.
- Similarity search now has its own better-tuned threshold UX, including `Show the N most similar`, improved defaults, a cap that keeps the slider usable on huge folders, and fixes so the query image is not incorrectly counted against the similar-image total.
- Added smarter same-person helper-threshold behavior and smoother large-folder interaction by keeping histogram feedback live while deferring the heavy gallery rebuilds until slider release.
- Fixed the ImageReward histogram so it now colors the keep/reject sides red/green around the threshold like PromptMatch.
- Added `Find same person` as a new preview action, powered by a dedicated `InsightFace buffalo_l` face-recognition backend instead of PromptMatch image embeddings.
- Same-person search now reuses the existing split view, histogram, thresholding, helper slider, manual overrides, and export flow, while keeping the query image marked in the results.
- Images with no detectable face are now handled gracefully as unmatched instead of crashing the same-person search flow.
- Linux and Windows setup now install the required face-search dependencies by default, including the CUDA-enabled ONNX runtime needed for GPU-backed InsightFace use.
- Same-person embedding extraction is now parallelized across a small worker pool, which makes large-folder face searches noticeably faster than the earlier strictly sequential pass.

## [1.8.5] - 2026-04-11

- JoyCaption GGUF generation is now quieter and finally must faster because GPU powered.
- Updated the README and in-app setup hints to match the new always-installed GGUF workflow.

## [1.8.3] - 2026-04-10

- Trimmed startup noise by removing the Linux launcher banner lines and fixing the embedded JS/raw-string plus transformers warnings that were cluttering app launch.
- Made the inline gallery preview close again with a simple left-click, so zooming in and back out feels direct instead of getting stuck in preview mode.
- Fixed a Gradio slider-range handoff bug when switching into ImageReward, which could throw out-of-bounds threshold errors during method changes or penalty-score recalculation.

## [1.8.0] - 2026-04-10

- On Windows, model downloads and PromptMatch proxy caches are now kept locally inside the project folder under `models/` and `cache/` instead of filling the user's profile or temp drive.
- On Linux, the app keeps using the normal system-cache behavior by default.
- Added clear cached-vs-download model markers to the PromptMatch model dropdown and fixed the availability detection so cached OpenCLIP models report correctly.
- Expanded the README cache and model-guide documentation so ImageReward, OS-specific cache defaults, and cache locations are explained more clearly.

## [1.7.0] - 2026-04-10

- Moved export controls into the gallery headers so each bucket now has its own `Export` toggle and editable folder name directly above the gallery.
- Added an optional `Move instead of copy` export mode in the export section.
- Added subtle green/red gallery background tinting and kept the `Tile Size` control pinned to the far right of the top toolbar.
- Model downloads now go into project-local `models/` folders by default, and PromptMatch proxy images now go into a project-local `cache/` folder instead of filling the user's profile or temp drive.
- Added `HYBRIDSCORER_CACHE_MODE=system` for users who want to keep the old cache behavior under their normal library-managed locations.

## [1.6.23] - 2026-04-09

- Due to a bug in Gradio: Disabled image-to-image navigation inside the zoom dialog so it no longer acts like a carousel.
- Hid the zoom dialog's thumbnail strip, blocked left/right navigation keys there, and prevented clicks inside the zoomed media area from stepping to neighboring images.

## [1.6.13] - 2026-04-09

- Stopped thumbnail preview clicks from forcing a full gallery and histogram rerender just to remember the active preview image for prompt generation and fit-threshold actions.

## [1.6.12] - 2026-04-09

- Fixed histogram hover markers reading stale gallery-card data after rerenders, which could make the positive or negative marker jump to the wrong side or bounce while moving across consecutive images.

## [1.6.11] - 2026-04-09

- Moved the thumbnail-hover histogram marker fully into frontend overlay code so hovering no longer forces the graph to be regenerated through Gradio.
- Reduced the histogram's fixed left/right padding and switched tick placement to explicit left/center/right alignment so the chart uses more width and no longer carries that oversized left gutter.

## [1.6.10] - 2026-04-09

- Stopped the custom gallery repaint observer from reacting to Gradio preview-dialog mutations, which was causing the UI to visibly refresh when zooming into an image.

## [1.6.9] - 2026-04-09

- Added a faint hover marker line in the histogram so moving over gallery thumbnails shows that image's current score against the stronger threshold line.
- Made the histogram width follow the available threshold-panel width so browser resizing no longer leaves large empty side gutters around a fixed-size graph.

## [1.6.8] - 2026-04-09

- Fixed the PromptMatch histogram so both positive and negative charts now show the full score range instead of incorrectly clipping everything below `0`.

## [1.6.5] - 2026-04-09

- Manual PromptMatch/ImageReward pinning now survives rescoring the same folder and only clears automatically when an image leaves that folder or you switch folders.
- Added per-method threshold lock checkboxes so prompt reruns can keep the exact current thresholds, while folder/model changes still release them automatically.
- Reworked the threshold panel so PromptMatch and ImageReward only show the relevant keep-threshold checkbox for the active method.
- Clarified threshold wording, replaced the slider reset arrows with explicit `50%` actions, and made the threshold sliders follow the same min/max score ranges shown in the histograms.
- Added threshold-side tinting to the PromptMatch positive and negative histograms to make the keep/reject sides easier to read at a glance.

## [1.6.0] - 2026-04-09

- Added PromptMatch image-embedding caching per folder and model, so rerunning with changed positive or negative prompts can reuse the expensive image pass and feel much faster.
- Refined the PromptMatch model lineup for practical NSFW-oriented use and updated the VRAM guidance to match observed real-world usage more closely.
- Expanded the README with detailed PromptMatch model descriptions, NSFW notes, and GPU-tier recommendations.

## [1.5.0] - 2026-04-08

- The setup scripts now try a safe git refresh automatically on every run before updating `venv312`.
- Renamed the setup scripts to `setup_update-linux.sh` and `setup_update-windows.bat`.
- Updated launcher hints and README instructions to match the new setup script names and update workflow.
- Trimmed the PromptMatch model list to a smaller NSFW-focused set and added VRAM guidance directly to the model labels.
- Adjusted the PromptMatch VRAM guidance to match observed real-world usage more closely instead of the earlier conservative estimates.
- Tightened the SigLIP VRAM guidance again to reflect measured usage more accurately.
- Updated the `OpenCLIP ViT-L-14` VRAM guidance to reflect its lower observed usage as well.
- PromptMatch now caches image embeddings per folder and model so prompt-only reruns can reuse the expensive image pass.

## [1.4.2] - 2026-04-08

- Added PromptMatch weighted fragment syntax like `(blonde:1.2)` for both positive and negative prompts.
- Added PromptMatch keyboard shortcuts to wrap selected prompt text with weights and nudge weights up or down by `0.1`.
- Returning a weighted fragment to `1.0` now removes the wrapper and restores plain prompt text automatically.
- Added `Ctrl+Enter` scoring shortcuts from the PromptMatch and ImageReward prompt boxes.
- Updated the README and in-app help text to document the new prompt weighting and keyboard shortcut workflow.

## [1.4.0] - 2026-04-07

- Compacted the sidebar UI to remove wasted padding, borders, gutters, and empty space.
- Simplified the header so it shows `HybridScorer` with version and creator on one line.
- Moved `Prompt from preview image` into its own dedicated collapsible section under scoring.
- Improved button clarity with stronger color coding for prompt generation, insert, move, fit-threshold, and clear-status actions.
- Added distinct accordion header colors so sidebar sections are easier to scan.
- Enlarged and clarified the manual move button arrows.
- Made the galleries use more of the viewport height and reduced excess outer page margins.
- Added drag-and-drop moving between the `SELECTED` and `REJECTED` galleries.
- Dragging a marked thumbnail now moves the full marked batch between galleries, not just one image.
- Inverted the upper-right tile-size slider so dragging right makes thumbnails larger.
- Renamed the gallery zoom label to `Tile Size` and fixed the toolbar spacing so the full label is visible.

## [1.3.0] - 2026-04-03

- Added `Prompt from preview image` model selection:
  `Florence-2`, `JoyCaption Beta One`, and `JoyCaption Beta One GGUF (Q4_K_M)`.
- JoyCaption now offers 3 clearer output styles:
  short tags, compact prompt, and detailed prose.
- Improved generated prompt cleanup and reduced generic lead-ins.
- Added optional GGUF setup for lower-VRAM JoyCaption use.
- Fixed prompt-generator cache/source detection issues.

## [1.2.5] - 2026-04-03

- Added `Fit thresh` to move the threshold so selected images switch sides automatically.
- Added `Prompt from preview image` to generate editable prompts from the current image.
- Prompt generation now keeps its own scratch field and no longer overwrites your active scoring prompt.
- Improved sidebar layout and scoring button placement.
- Switching between PromptMatch and ImageReward now keeps the current galleries visible.

## [1.1.0] - 2026-04-01

- Merged PromptMatch and ImageReward into one app.
- Added better gallery UI, zoom, hover polish, and clearer selection cues.
- Added more PromptMatch model choices and faster proxy-based scoring.
- Improved setup for fresh Python 3.12 / CUDA installs.
- Added version tracking and release files.

## [1.0.0] - 2026-03-27

- First single-app HybridScorer release.
- Combined the old separate tools into one shared interface.
- Added threshold-based sorting, review controls, and export folders.

## [pre-1.0] - 2026-03-26

- Early two-app prototype phase.
- Built the first PromptMatch and ImageReward workflows.
- Added setup scripts, sorting, export, and initial UI tooling.
