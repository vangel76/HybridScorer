# Changelog

All notable changes to this project should be tracked here.

## [1.1.0] - 2026-04-01

- Added a repo-level `VERSION` file and app-side version loading so the UI and browser title show the current release.
- Added GitHub-friendly release metadata files with this changelog and a documented tag workflow.
- Improved proxy logging so the terminal prints the proxy cache directory during PromptMatch and ImageReward runs.
- Hardened fresh Python 3.12 environment setup by installing `image-reward==1.5` separately and avoiding the broken `image-reward==1.0` source build path.
