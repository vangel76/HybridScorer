# Changelog

## [1.7.0] - 2026-04-10

- Moved export controls into the gallery headers so each bucket now has its own `Export` toggle and editable folder name directly above the gallery.
- Added an optional `Move instead of copy` export mode in the export section.
- Added subtle green/red gallery background tinting and kept the `Tile Size` control pinned to the far right of the top toolbar.

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
