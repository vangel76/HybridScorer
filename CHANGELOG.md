# Changelog

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

## [1.42.0] - 2026-04-08

- Added PromptMatch weighted fragment syntax like `(blonde:1.2)` for both positive and negative prompts.
- Added PromptMatch keyboard shortcuts to wrap selected prompt text with weights and nudge weights up or down by `0.1`.
- Returning a weighted fragment to `1.0` now removes the wrapper and restores plain prompt text automatically.
- Added `Ctrl+Enter` scoring shortcuts from the PromptMatch and ImageReward prompt boxes.
- Updated the README and in-app help text to document the new prompt weighting and keyboard shortcut workflow.

## [1.40.0] - 2026-04-07

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

## [1.30.0] - 2026-04-03

- Added `Prompt from preview image` model selection:
  `Florence-2`, `JoyCaption Beta One`, and `JoyCaption Beta One GGUF (Q4_K_M)`.
- JoyCaption now offers 3 clearer output styles:
  short tags, compact prompt, and detailed prose.
- Improved generated prompt cleanup and reduced generic lead-ins.
- Added optional GGUF setup for lower-VRAM JoyCaption use.
- Fixed prompt-generator cache/source detection issues.

## [1.25.0] - 2026-04-03

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
