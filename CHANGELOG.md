# Changelog

All notable changes to this project should be tracked here.

Entries before `1.1.0` were reconstructed from git history and repository snapshots.

## [1.25.0] - 2026-04-03

- Added a threshold-fit action that moves the cutoff just enough to pull the currently previewed or multi-selected images into the opposite bucket, instead of relying only on manual overrides.
- Added a Florence-2 powered `Prompt from preview image` utility that can generate an editable prompt from the current preview image and feed it back into either PromptMatch or ImageReward when requested.
- Expanded prompt generation with cached `Core facts`, `Balanced`, and `Full` detail variants, kept the generated prompt in a separate scratch box instead of overwriting the active scoring prompt, and added smarter cleanup for stock lead-ins such as `In this image we can see`.
- Updated dependencies and runtime compatibility so Florence prompt generation works alongside the existing app stack while keeping ImageReward functional under the newer `transformers` version.
- Reworked the sidebar flow so scoring stays grouped with the active method prompts, the Florence controls stay together as their own block, and both PromptMatch and ImageReward `Run scoring` buttons are wired correctly.
- Improved gallery/session behavior by keeping scored results visible while switching methods, preserving prompt-generation state per preview image, and making the multi-selection status text larger and easier to read.

## [1.1.0] - 2026-04-01

- Unified the separate PromptMatch and ImageReward tools into the single `Hybrid-Scorer.py` app, with updated run scripts, screenshots, and documentation.
- Expanded the hybrid UI with hover polish, zoom controls, green/red edge cues, and follow-up interaction fixes across the March 27-29 iteration cycle.
- Added ConvNeXt-based PromptMatch model options, a proxy-based scoring path for PromptMatch, performance improvements, and stronger negative-prompt and redundancy handling.
- Hardened fresh Python 3.12 setup with CUDA 12.8 defaults, dependency pinning, `protobuf` support for SigLIP, and an ImageReward install path that avoids broken source-build imports.
- Added a repo-level `VERSION` file and app-side version loading so the UI and browser title show the current release.
- Added GitHub-friendly release metadata files with this changelog and a documented tag workflow, and improved proxy logging so the terminal prints the proxy cache directory during runs.

## [1.0.0] - 2026-03-27

- Introduced the first single-app hybrid release by consolidating PromptMatch and ImageReward workflows into one Gradio interface.
- Removed the separate `promptmatch.py` and `imagereward.py` entrypoints in favor of the shared hybrid app, with updated launcher scripts and README guidance.
- Added the core hybrid-era UI behavior, including threshold-driven sorting, zoom controls, green/red edge cues, hover polish, and follow-up UI fixes that shaped the first unified app experience.
- Carried forward the project's CUDA-first image triage workflow, manual review controls, and export-based folder sorting inside the new one-app flow.

## [pre-1.0] - 2026-03-26

- Preserved the project's earlier experimentation in the `old/` folder, including many PromptMatch and ImageReward prototypes plus prompt-training and sorting utilities that informed the first release.
- Landed the first working repository structure with setup scripts, dependency lists, run helpers, and the original dual-app architecture built around `promptmatch.py` and `imagereward.py`.
- Shipped the two-app workflow with semantic matching, aesthetic ranking, sortable galleries, export folders, and multiple PromptMatch model backends across OpenAI CLIP, OpenCLIP, and SigLIP.
- Rounded out the untagged prototype phase with README cleanup, shell-script standardization from `fish` to `sh`, Windows launcher updates, issue templates, credits, and small UI/script fixes that fed directly into `1.0.0`.
