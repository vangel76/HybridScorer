"""
HybridScorer — image triage

Single-file Gradio app (`Hybrid-Scorer.py`)

Modes:
- PromptMatch → CLIP-based text-image similarity
- ImageReward → aesthetic/preference scoring
- Similarity → image-image search from preview
- SamePerson → face search from preview
- LM Search → PromptMatch shortlist + local vision-language rerank

Core:
- manual bucket overrides
- preview-driven actions
- prompt generation from preview
- cached scoring + proxy images
- export to structured folders

See:
- README.md
- docs/architecture.md
- docs/behavior-notes.md

Important:
- callback return signatures must match exactly
- UI behavior may live in Python and injected JS
"""

import base64
import gc
import io
import json
import math
import os
import re
import shutil
import socket
import string
import sys
import tempfile
import threading
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, partial
from hashlib import sha256
from importlib import metadata
from importlib import import_module
from importlib.machinery import ModuleSpec

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

APP_NAME = "HybridScorer"
APP_DISPLAY_NAME = "HybridSelector"


def load_app_version():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    version_path = os.path.join(script_dir, "VERSION")
    try:
        with open(version_path, "r", encoding="utf-8") as handle:
            version = handle.read().strip()
            if version:
                return version
    except OSError:
        pass
    return "0.0.0"


APP_VERSION = load_app_version()
APP_GITHUB_TAG = f"v{APP_VERSION}"
APP_WINDOW_TITLE = f"{APP_DISPLAY_NAME} {APP_GITHUB_TAG}"

def load_changelog():
    try:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(_script_dir, "CHANGELOG.md"), "r", encoding="utf-8") as _f:
            return _f.read()
    except Exception:
        return "Changelog not available."

import html as _html
APP_CHANGELOG_HTML = _html.escape(load_changelog())

_script_dir = os.path.dirname(os.path.abspath(__file__))
_APP_CSS = open(os.path.join(_script_dir, "static", "style.css"), encoding="utf-8").read()
_APP_JS_BODY = open(os.path.join(_script_dir, "static", "app.js"), encoding="utf-8").read()

from lib.config import *
from lib.utils import *
from lib.backend import ModelBackend
from lib.scoring import score_all, encode_all_promptmatch_images, score_promptmatch_cached_features
from lib.helpers import *
from lib.state import get_state_defaults as _get_state_defaults, init_state as _init_state
import lib.state_helpers as _sh
import lib.loaders as _lo
import lib.view as _vw
import lib.callbacks.scoring as _sc
import lib.callbacks.prompts as _pr
import lib.callbacks.ui as _ui

def create_app():
    require_cuda()
    configure_torch_cpu_threads()
    device = "cuda"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, INPUT_FOLDER_NAME)
    if not os.path.isdir(source_dir):
        source_dir = script_dir

    # Start with the lighter recommended SigLIP model so first-run download size
    # and VRAM pressure are more reasonable than the much larger ViT-bigG-14.
    prompt_backend = ModelBackend(
        device,
        backend="siglip",
        siglip_model="google/siglip-so400m-patch14-384",
        clip_cache_dir=get_cache_config()["clip_dir"],
        huggingface_cache_dir=get_cache_config()["huggingface_dir"],
    )

    # Shared mutable state for the one-page app. Gradio callbacks update this in place.
    state = _init_state(source_dir, prompt_backend)

    # Bind state helpers so all existing call sites remain unchanged.
    is_browse_mode = partial(_sh.is_browse_mode, state)
    set_scored_mode = partial(_sh.set_scored_mode, state)
    set_browse_mode = partial(_sh.set_browse_mode, state)
    remove_file_quietly = _sh.remove_file_quietly
    clear_external_query_state = partial(_sh.clear_external_query_state, state)
    save_external_query_image = partial(_sh.save_external_query_image, state)
    save_external_query_image_bytes = partial(_sh.save_external_query_image_bytes, state)
    get_preview_image_path = partial(_sh.get_preview_image_path, state)
    active_query_image_context = partial(_sh.active_query_image_context, state)
    active_query_image_widget_update = partial(_sh.active_query_image_widget_update, state)
    clear_external_query_button_update = partial(_sh.clear_external_query_button_update, state)
    reset_selection_state = partial(_sh.reset_selection_state, state)
    clear_preview_search_context = partial(_sh.clear_preview_search_context, state)
    clear_active_scores = partial(_sh.clear_active_scores, state)
    reset_for_folder_change = partial(_sh.reset_for_folder_change, state)
    invalidate_folder_level_caches = partial(_sh.invalidate_folder_level_caches, state)
    preserve_overrides_for_image_paths = partial(_sh.preserve_overrides_for_image_paths, state)
    preserve_overrides_for_folder_key = partial(_sh.preserve_overrides_for_folder_key, state)
    sync_promptmatch_proxy_cache = partial(_sh.sync_promptmatch_proxy_cache, state)
    begin_scored_run = partial(_sh.begin_scored_run, state)
    set_browse_folder_state = partial(_sh.set_browse_folder_state, state)
    can_reuse_proxy_map = partial(_sh.can_reuse_proxy_map, state)
    remember_mode_thresholds = partial(_sh.remember_mode_thresholds, state)
    recalled_mode_thresholds = partial(_sh.recalled_mode_thresholds, state)

    # Bind model loaders
    ensure_imagereward_model = partial(_lo.ensure_imagereward_model, state)
    release_inactive_gpu_models = partial(_lo.release_inactive_gpu_models, state)
    ensure_florence_model = partial(_lo.ensure_florence_model, state, device)
    ensure_joycaption_model = partial(_lo.ensure_joycaption_model, state, device)
    ensure_huihui_gemma4_model = partial(_lo.ensure_huihui_gemma4_model, state, device)
    ensure_joycaption_gguf_model = partial(_lo.ensure_joycaption_gguf_model, state, device)
    tagmatch_prepare_image = _lo.tagmatch_prepare_image
    ensure_tagmatch_model = partial(_lo.ensure_tagmatch_model, state, device)
    load_tagmatch_vocabulary = partial(_lo.load_tagmatch_vocabulary, state)
    refresh_tagmatch_vocab_state = partial(_lo.refresh_tagmatch_vocab_state, state)
    ensure_promptmatch_backend_loaded = partial(_lo.ensure_promptmatch_backend_loaded, state, device)
    ensure_face_backend_loaded = partial(_lo.ensure_face_backend_loaded, state)
    ensure_promptmatch_feature_cache = partial(_lo.ensure_promptmatch_feature_cache, state, device)
    encode_single_promptmatch_image = partial(_lo.encode_single_promptmatch_image, state)
    choose_primary_face = _lo.choose_primary_face
    ensure_face_feature_cache = partial(_lo.ensure_face_feature_cache, state)
    encode_single_face_embedding = partial(_lo.encode_single_face_embedding, state)

    # Bind view helpers
    gallery_update = _vw.gallery_update
    gallery_display_items = partial(_vw.gallery_display_items, state)
    ui_visibility_updates = partial(_vw.ui_visibility_updates, state)
    selection_info = partial(_vw.selection_info, state)
    marked_state_json = partial(_vw.marked_state_json, state)
    active_targets = partial(_vw.active_targets, state)
    render_histogram = partial(_vw.render_histogram, state)
    current_view = partial(_vw.current_view, state)
    configure_controls = partial(_vw.configure_controls, state)
    build_scored_callback_result = partial(_vw.build_scored_callback_result, state)
    build_preview_search_result = _vw.build_preview_search_result
    status_with_current_view = partial(_vw.status_with_current_view, state)
    empty_result = partial(_vw.empty_result, state)
    render_view_with_controls = partial(_vw.render_view_with_controls, state)

    # Bind scoring callbacks
    score_tagmatch_folder = partial(_sc.score_tagmatch_folder, state, device)
    score_imagereward = partial(_sc.score_imagereward, state, device)
    recompute_imagereward_scores = partial(_sc.recompute_imagereward_scores, state)
    load_folder_for_browse = partial(_sc.load_folder_for_browse, state)
    refresh_promptmatch_model_dropdown = _sc.refresh_promptmatch_model_dropdown
    middle_threshold_values = partial(_sc.middle_threshold_values, state)
    score_similarity_cached_features = _sc.score_similarity_cached_features
    score_sameperson_cached_features = _sc.score_sameperson_cached_features
    prepare_scored_run_context = partial(_sc.prepare_scored_run_context, state)
    render_scored_mode_result = partial(_sc.render_scored_mode_result, state)
    normalize_preview_search_request = partial(_sc.normalize_preview_search_request, state)
    llmsearch_similarity = partial(_sc.llmsearch_similarity, state)
    score_folder = partial(_sc.score_folder, state, device)
    find_similar_images = partial(_sc.find_similar_images, state, device)
    find_same_person_images = partial(_sc.find_same_person_images, state)
    find_objectsearch_images = partial(_sc.find_objectsearch_images, state, device)
    handle_shortcut_action = partial(_sc.handle_shortcut_action, state, device)

    # Bind prompt callbacks
    generated_prompt_variants_for = partial(_pr.generated_prompt_variants_for, state)
    select_cached_generated_prompt = partial(_pr.select_cached_generated_prompt, state)
    run_florence_prompt_variant = partial(_pr.run_florence_prompt_variant, state, device)
    run_huihui_gemma4_prompt_variant = partial(_pr.run_huihui_gemma4_prompt_variant, state, device)
    run_joycaption_prompt_variant = partial(_pr.run_joycaption_prompt_variant, state, device)
    run_wd_tags_prompt_variant = partial(_pr.run_wd_tags_prompt_variant, state, device)
    generate_prompt_variant = partial(_pr.generate_prompt_variant, state, device)
    external_query_prompt_status = partial(_pr.external_query_prompt_status, state)
    set_external_query_image = partial(_pr.set_external_query_image, state)
    set_external_query_from_bridge = partial(_pr.set_external_query_from_bridge, state)
    clear_external_query_image = partial(_pr.clear_external_query_image, state)
    generate_prompt_from_preview = partial(_pr.generate_prompt_from_preview, state, device)
    insert_generated_prompt = partial(_pr.insert_generated_prompt, state)

    # Bind UI callbacks
    update_split = partial(_ui.update_split, state)
    update_histogram_only = partial(_ui.update_histogram_only, state)
    update_prompt_generator = partial(_ui.update_prompt_generator, state)
    update_generated_prompt_detail = partial(_ui.update_generated_prompt_detail, state)
    update_proxy_display = partial(_ui.update_proxy_display, state)
    update_imagereward_penalty_weight = partial(_ui.update_imagereward_penalty_weight, state)
    handle_thumb_action = partial(_ui.handle_thumb_action, state)
    handle_hist_width = partial(_ui.handle_hist_width, state)
    move_right = partial(_ui.move_right, state)
    move_left = partial(_ui.move_left, state)
    pin_selected = partial(_ui.pin_selected, state)
    clear_status = partial(_ui.clear_status, state)
    clear_all_status = partial(_ui.clear_all_status, state)
    fit_threshold_to_targets = partial(_ui.fit_threshold_to_targets, state)
    set_from_percentile = partial(_ui.set_from_percentile, state)
    update_histogram_from_percentile = partial(_ui.update_histogram_from_percentile, state)
    reset_main_threshold_to_middle = partial(_ui.reset_main_threshold_to_middle, state)
    reset_aux_threshold_to_middle = partial(_ui.reset_aux_threshold_to_middle, state)
    reset_percentile_to_middle = partial(_ui.reset_percentile_to_middle, state)
    update_zoom = partial(_ui.update_zoom, state)
    def on_hist_click(sel: gr.SelectData, main_threshold, aux_threshold):
        return _ui.on_hist_click(state, sel, main_threshold, aux_threshold)
    export_files = partial(_ui.export_files, state)

    def tooltip_head(pairs):
        mapping = json.dumps(pairs)
        js = _APP_JS_BODY.replace("__TOOLTIPS__", mapping)
        return f"<script>\n{js}\n</script>\n"


    tooltips = {
        "hy-method": "Choose whether to sort by PromptMatch or ImageReward.",
        "hy-folder": "Path to the image folder you want to score. You can paste a full folder path here.",
        "hy-external-query-image": "This area shows the active query image. It displays the selected folder preview by default, or a dropped, pasted, or uploaded custom image when one overrides the preview.",
        "hy-clear-external-query": "Clear the external query image and fall back to the gallery preview.",
        "hy-load-folder": "Load the current folder into unscored browse mode and prepare proxies for faster gallery display.",
        "hy-model": "Choose the PromptMatch model. Cached models are shown in green text, and models that still need a first download are shown in amber.",
        "hy-llm-model": "Choose the PromptMatch model used for the fast shortlist stage before the vision-LLM rerank pass. Cached models are shown in green text, and models that still need a first download are shown in amber.",
        "hy-llm-backend": "Choose the local vision-language backend used to rerank shortlisted images at a deeper semantic level. Cached backends are shown in green text, and backends that still need a first download are shown in amber. The Huihui Gemma 4 option is less filtered and may produce less-filtered text.",
        "hy-llm-prompt": "Natural-language search request for the hybrid PromptMatch plus vision-LLM rerank mode.",
        "hy-llm-shortlist": "How many top PromptMatch candidates should be sent into the slower vision-LLM rerank stage.",
        "hy-pos": "Describe what you want to find in the images. PromptMatch also supports fragment weights like beautiful (blonde:1.2) woman. Select text and press Ctrl +/- to wrap or adjust it by 0.1. Press Ctrl+Enter to run scoring.",
        "hy-neg": "Optional PromptMatch negative prompt that counts against a match. Weighted fragments like (text:1.3) also work here. Select text and press Ctrl +/- to wrap or adjust it by 0.1. Press Ctrl+Enter to run scoring.",
        "hy-ir-pos": "Describe the style or aesthetic you want ImageReward to favor. Press Ctrl+Enter to run scoring.",
        "hy-ir-neg": "Optional experimental penalty prompt. Its score is subtracted from the positive style score. Press Ctrl+Enter to run scoring.",
        "hy-ir-weight": "How strongly the penalty prompt should reduce the final ImageReward score.",
        "hy-tagmatch-tags": "Comma-separated booru-style tags for TagMatch. Start typing to get live vocabulary suggestions, then use arrow keys plus Enter/Tab or click a suggestion.",
        "hy-run-llm": "Run hybrid image search: PromptMatch first shortlists likely matches, then the local vision-language backend reranks the top candidates.",
        "hy-prompt-generator": "Choose which caption model should draft the prompt from the active query image. Cached backends are shown in green text, and backends that still need a first download are shown in amber. The Huihui Gemma 4 option is less filtered and may produce less-filtered text.",
        "hy-generate-prompt": "Use the active query image to draft an editable prompt with the selected caption backend.",
        "hy-find-similar": "Use the active query image and rank the current folder by visual similarity with the active PromptMatch model.",
        "hy-find-same-person": "Use the active query image and rank the current folder by face identity similarity with InsightFace.",
        "hy-generated-prompt": "Editable scratch prompt generated from the active query image. You can tweak it before scoring or reinsert it into the active method.",
        "hy-generated-prompt-detail": "Choose whether the caption backend should describe only the core facts, a balanced amount of detail, or the full detailed prompt.",
        "hy-insert-prompt": "Copy the editable generated prompt back into the active method's main prompt field.",
        "hy-promptgen-status": "Small status readout for prompt generation.",
        "hy-run-pm": "Score the current folder with the selected method and prompts. Ctrl+Enter from a PromptMatch prompt box does the same.",
        "hy-run-ir": "Score the current folder with the selected method and prompts. Ctrl+Enter from an ImageReward prompt box does the same.",
        "hy-main-slider": "Minimum score needed to stay in SELECTED. Raising it keeps fewer images. Click the histogram to set it visually.",
        "hy-aux-slider": "Maximum negative-prompt similarity allowed in PromptMatch. Lowering it rejects more images that match the negative prompt.",
        "hy-main-mid": "Set the main threshold to 50% of the current score range.",
        "hy-aux-mid": "Set the negative threshold to 50% of the current negative-score range.",
        "hy-keep-pm-thresholds": "Keep the exact PromptMatch thresholds when rerunning with changed prompts. They reset automatically if folder or PromptMatch model changes.",
        "hy-keep-ir-thresholds": "Keep the exact ImageReward threshold when rerunning with changed prompts. It resets automatically if the folder changes.",
        "hy-percentile": "Automatically set the main threshold to keep roughly the top N percent, or show the N most similar images in FSI mode.",
        "hy-percentile-mid": "Reset the helper control to its default value for the current mode.",
        "hy-zoom-ui": "Choose how many thumbnails appear per row in both galleries.",
        "hy-use-proxy-display": "Show gallery images from cached proxies for faster browsing on large folders.",
        "hy-hist": "Histogram of current scores. In PromptMatch, click the top chart for positive threshold or bottom chart for negative threshold.",
        "hy-export": "COPY the current split into two SELECTED / REJECTED output folders inside source folder.",
        "hy-left-gallery": "Images currently in the left bucket. Click one to select it, Shift+click to mark, or drag an image to the other gallery.",
        "hy-right-gallery": "Images currently in the right bucket. Click one to select it, Shift+click to mark, or drag an image to the other gallery.",
        "hy-export-left-enabled": "Include the left bucket in the next export run.",
        "hy-export-right-enabled": "Include the right bucket in the next export run.",
        "hy-export-move-enabled": "Move files into the export folders instead of copying them. Disabled by default.",
        "hy-export-left-name": "Editable export folder name for the left bucket. Export writes directly into source_folder/<name>.",
        "hy-export-right-name": "Editable export folder name for the right bucket. Export writes directly into source_folder/<name>.",
        "hy-move-right": "Move all marked SELECTED images into REJECTED as manual overrides.",
        "hy-move-left": "Move all marked REJECTED images into SELECTED as manual overrides.",
        "hy-pin-selected": "Pin the currently marked or previewed images to their current bucket as manual overrides without moving them.",
        "hy-fit-threshold": "Adjust the score threshold just enough so the marked images flip to the other bucket. Uses the previewed image if nothing is marked.",
        "hy-clear-status": "Remove manual override status from all marked images so they snap back to their scored bucket.",
        "hy-clear-all-status": "Remove manual override status from every pinned image in the current folder so everything snaps back to the scored buckets.",
    }
    css = _APP_CSS

    with gr.Blocks(title=APP_WINDOW_TITLE) as demo:
        gr.HTML(f"""
<div class='app-header'>
  <h1>{APP_NAME}</h1>
  <div class='header-meta'><button id='hy-version-btn' class='version-btn'>{APP_GITHUB_TAG}</button> &middot; created by vangel</div>
</div>
<div id='hy-changelog-overlay' class='changelog-overlay'>
  <div class='changelog-modal'>
    <div class='changelog-modal-header'><span>Changelog</span><button id='hy-changelog-close' class='changelog-close'>&#x2715;</button></div>
    <div class='changelog-modal-body'><pre class='changelog-content'>{APP_CHANGELOG_HTML}</pre></div>
  </div>
</div>
""")

        with gr.Row(equal_height=False, elem_classes=["app-shell"]):
            with gr.Column(scale=1, min_width=300, elem_classes=["sidebar-box"]):
                thumb_action = gr.Textbox(value="", visible="hidden", elem_id="hy-thumb-action")
                hist_width_tb = gr.Textbox(value="300", visible="hidden", elem_id="hy-hist-width")
                shortcut_action = gr.Textbox(value="", visible="hidden", elem_id="hy-shortcut-action")
                external_query_bridge = gr.Textbox(value="", visible="hidden", elem_id="hy-external-query-bridge")
                mark_state = gr.Textbox(value='{"left":[],"right":[]}', visible="hidden", elem_id="hy-mark-state")
                model_status_state = gr.Textbox(value=promptmatch_model_status_json(), visible="hidden", elem_id="hy-model-status")
                tagmatch_vocab_state = gr.Textbox(value="[]", visible="hidden", elem_id="hy-tagmatch-vocab")
                with gr.Group(elem_classes=["sidebar-scroll"]):
                    with gr.Accordion("1. Setup", open=True, elem_id="hy-acc-setup"):
                        method_dd = gr.Dropdown(
                            choices=[METHOD_PROMPTMATCH, METHOD_IMAGEREWARD, METHOD_LLMSEARCH, METHOD_TAGMATCH],
                            value=METHOD_PROMPTMATCH,
                            label="Method",
                            elem_id="hy-method",
                        )
                        method_note = gr.Markdown(
                            "PromptMatch sorts by text-image similarity. Use a positive prompt and optional negative prompt. Fragment weights like (blonde:1.2) are supported.",
                            elem_classes=["method-note"],
                        )
                        folder_input = gr.Textbox(
                            value=source_dir,
                            label="Image folder - paste a path here",
                            lines=1,
                            placeholder=folder_placeholder(),
                            elem_id="hy-folder",
                        )
                        load_folder_btn = gr.Button("Load folder", elem_id="hy-load-folder")

                    with gr.Accordion("2. SCORING & Method/Settings", open=False, elem_id="hy-acc-scoring"):
                        with gr.Group(visible=True) as promptmatch_group:
                            model_dd = gr.Dropdown(
                                choices=promptmatch_model_dropdown_choices(),
                                value=label_for_backend(prompt_backend),
                                label="PromptMatch model",
                                elem_id="hy-model",
                            )
                            pos_prompt_tb = gr.Textbox(value=SEARCH_PROMPT, label="Positive prompt", lines=2, elem_id="hy-pos")
                            neg_prompt_tb = gr.Textbox(value=NEGATIVE_PROMPT, label="Negative prompt", lines=1, elem_id="hy-neg")
                            pm_segment_cb = gr.Checkbox(label="Per-segment scoring (hover shows per-phrase match)", value=False, elem_id="hy-pm-segment")
                            promptmatch_run_btn = gr.Button("Run scoring", elem_id="hy-run-pm", variant="primary")

                        with gr.Group(visible=False) as imagereward_group:
                            ir_prompt_tb = gr.Textbox(value=IR_PROMPT, label="ImageReward positive prompt", lines=3, elem_id="hy-ir-pos")
                            ir_negative_prompt_tb = gr.Textbox(
                                value=DEFAULT_IR_NEGATIVE_PROMPT,
                                label="Experimental penalty prompt",
                                lines=2,
                                placeholder="Optional: undesirable style or mood to subtract",
                                elem_id="hy-ir-neg",
                            )
                            ir_penalty_weight_tb = gr.Slider(
                                value=DEFAULT_IR_PENALTY_WEIGHT,
                                label="Penalty weight",
                                minimum=0.0,
                                maximum=4.0,
                                step=0.1,
                                elem_id="hy-ir-weight",
                            )
                            imagereward_run_btn = gr.Button("Run scoring", elem_id="hy-run-ir", variant="primary")

                        with gr.Group(visible=False) as llmsearch_group:
                            llm_model_dd = gr.Dropdown(
                                choices=promptmatch_model_dropdown_choices(),
                                value=label_for_backend(prompt_backend),
                                label="PromptMatch shortlist model",
                                elem_id="hy-llm-model",
                            )
                            llm_backend_dd = gr.Dropdown(
                                choices=prompt_backend_dropdown_choices(llmsearch_backend_choices()),
                                value=DEFAULT_LLMSEARCH_BACKEND,
                                label="Vision LM backend",
                                elem_id="hy-llm-backend",
                            )
                            llm_prompt_tb = gr.Textbox(
                                value=LLMSEARCH_DEFAULT_PROMPT,
                                label="LM search prompt",
                                lines=2,
                                elem_id="hy-llm-prompt",
                            )
                            llm_shortlist_slider = gr.Slider(
                                minimum=LLMSEARCH_SHORTLIST_MIN,
                                maximum=LLMSEARCH_SHORTLIST_MAX,
                                value=LLMSEARCH_SHORTLIST_DEFAULT,
                                step=1,
                                label="Shortlist size",
                                elem_id="hy-llm-shortlist",
                            )
                            llmsearch_run_btn = gr.Button("Run scoring", elem_id="hy-run-llm", variant="primary")

                        with gr.Group(visible=False) as tagmatch_group:
                            tagmatch_tags_tb = gr.Textbox(
                                value=TAGMATCH_DEFAULT_TAGS,
                                label="Tags to detect (comma-separated booru-style tags)",
                                lines=3,
                                elem_id="hy-tagmatch-tags",
                            )
                            tagmatch_run_btn = gr.Button("Run scoring", elem_id="hy-run-tagmatch", variant="primary")

                    with gr.Accordion("3. Search + Prompt from image", open=False, elem_id="hy-acc-search-image"):
                        external_query_image = gr.Image(
                            value=None,
                            sources=["upload"],
                            type="filepath",
                            image_mode="RGB",
                            label="Query image",
                            show_label=False,
                            height=210,
                            interactive=True,
                            placeholder="Drop, paste, or upload an image here. If you preview an image from the folder, it will show here too.",
                            elem_id="hy-external-query-image",
                            elem_classes=["external-query-image"],
                        )
                        clear_external_query_btn = gr.Button(
                            "Clear external override",
                            elem_id="hy-clear-external-query",
                            interactive=False,
                        )
                        with gr.Tabs():
                            with gr.Tab("Search"):
                                with gr.Column(elem_classes=["preview-action-stack"]):
                                    find_same_person_btn = gr.Button("Find same person", elem_id="hy-find-same-person")
                                    find_similar_btn = gr.Button("Find similar images", elem_id="hy-find-similar")
                                    find_object_btn = gr.Button("Find object images", elem_id="hy-find-object")
                            with gr.Tab("Prompt"):
                                with gr.Group(elem_classes=["preview-prompt-group"]):
                                    prompt_generator_dd = gr.Dropdown(
                                        choices=prompt_backend_dropdown_choices(PROMPT_GENERATOR_ALL_CHOICES),
                                        value=state["prompt_generator"],
                                        label="Prompt generator",
                                        elem_id="hy-prompt-generator",
                                    )
                                    generate_prompt_btn = gr.Button("Prompt from image", elem_id="hy-generate-prompt")
                                    promptgen_status_md = gr.Markdown(
                                        state["generated_prompt_status"],
                                        elem_classes=["promptgen-status"],
                                        elem_id="hy-promptgen-status",
                                    )
                                    generated_prompt_tb = gr.Textbox(
                                        value=state["generated_prompt"],
                                        label="Generated prompt",
                                        lines=4,
                                        placeholder="Preview, drop, paste, or upload an image, then generate an editable prompt here.",
                                        elem_id="hy-generated-prompt",
                                    )
                                    generated_prompt_detail_slider = gr.Slider(
                                        minimum=1,
                                        maximum=3,
                                        value=DEFAULT_GENERATED_PROMPT_DETAIL,
                                        step=1,
                                        label="Prompt detail",
                                        elem_id="hy-generated-prompt-detail",
                                    )
                                    insert_prompt_btn = gr.Button("Insert into active prompt", elem_id="hy-insert-prompt")

                    with gr.Accordion("4. Export", open=False, elem_id="hy-acc-export") as export_acc:
                        with gr.Row(equal_height=False, elem_classes=["export-options-row"]):
                            move_export_cb = gr.Checkbox(
                                value=False,
                                label="Move instead of copy",
                                container=False,
                                scale=0,
                                min_width=150,
                                elem_id="hy-export-move-enabled",
                                elem_classes=["gallery-export-toggle", "export-move-toggle"],
                            )
                        export_btn = gr.Button("Export folders", elem_id="hy-export", variant="primary")

                with gr.Group(elem_id="hy-thresholds-panel", elem_classes=["thresholds-panel"]):
                    hist_plot = gr.Image(value=None, show_label=False, interactive=False, elem_classes=["hist-img"], elem_id="hy-hist")
                    main_slider = gr.Slider(
                        minimum=-1.0,
                        maximum=1.0,
                        value=0.14,
                        step=0.001,
                        label=threshold_labels(METHOD_PROMPTMATCH)[0],
                        elem_id="hy-main-slider",
                        buttons=[],
                    )
                    main_mid_btn = gr.Button("50%", elem_id="hy-main-mid", visible=False, elem_classes=["threshold-mid"])
                    aux_slider = gr.Slider(
                        minimum=-1.0,
                        maximum=1.0,
                        value=NEGATIVE_THRESHOLD,
                        step=0.001,
                        label=threshold_labels(METHOD_PROMPTMATCH)[1],
                        elem_id="hy-aux-slider",
                        buttons=[],
                    )
                    aux_mid_btn = gr.Button("50%", elem_id="hy-aux-mid", visible=False, elem_classes=["threshold-mid"])
                    keep_pm_thresholds_cb = gr.Checkbox(
                        value=True,
                        label="Keep PromptMatch thresholds on prompt reruns",
                        visible=True,
                        elem_id="hy-keep-pm-thresholds",
                    )
                    keep_ir_thresholds_cb = gr.Checkbox(
                        value=True,
                        label="Keep ImageReward threshold on prompt reruns",
                        visible=False,
                        elem_id="hy-keep-ir-thresholds",
                    )
                    percentile_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label=percentile_slider_label(METHOD_PROMPTMATCH),
                        elem_id="hy-percentile",
                        buttons=[],
                    )
                    percentile_mid_btn = gr.Button("50%", elem_id="hy-percentile-mid", visible=False, elem_classes=["threshold-mid"])
                    proxy_display_cb = gr.Checkbox(value=True, label="Use proxies for gallery display", elem_id="hy-use-proxy-display")

            with gr.Column(scale=5, elem_classes=["gallery-pane"]):
                with gr.Row(equal_height=False, elem_classes=["gallery-topbar"]):
                    with gr.Column(scale=1, elem_classes=["gallery-side", "gallery-header-slot"]):
                        with gr.Row(equal_height=False, elem_classes=["gallery-head-row"]):
                            left_export_cb = gr.Checkbox(
                                value=True,
                                label="Export",
                                container=False,
                                scale=0,
                                min_width=62,
                                elem_id="hy-export-left-enabled",
                                elem_classes=["gallery-export-toggle"],
                            )
                            gr.HTML("", elem_classes=["gallery-head-fill"])
                            left_export_name_tb = gr.Textbox(value="selected", show_label=False, container=False, lines=1, elem_id="hy-export-left-name", elem_classes=["gallery-export-name"])
                            left_head = gr.Markdown("**0 images**", elem_classes=["gallery-count"])
                    with gr.Column(scale=0, min_width=100, elem_classes=["gallery-header-spacer"]) as gallery_header_spacer_col:
                        gr.HTML("")
                    with gr.Column(scale=1, elem_classes=["gallery-side", "gallery-header-slot"]) as right_header_col:
                        with gr.Row(equal_height=False, elem_classes=["gallery-head-row"]):
                            right_export_cb = gr.Checkbox(
                                value=True,
                                label="Export",
                                container=False,
                                scale=0,
                                min_width=62,
                                elem_id="hy-export-right-enabled",
                                elem_classes=["gallery-export-toggle"],
                            )
                            gr.HTML("", elem_classes=["gallery-head-fill"])
                            right_export_name_tb = gr.Textbox(value="rejected", show_label=False, container=False, lines=1, elem_id="hy-export-right-name", elem_classes=["gallery-export-name"])
                            right_head = gr.Markdown("**0 images**", elem_classes=["gallery-count"])
                    with gr.Column(scale=0, elem_classes=["gallery-zoom-slot"]):
                        with gr.Row(equal_height=False, elem_classes=["zoom-inline-wrap"]):
                            gr.Markdown("Tile Size", elem_classes=["zoom-inline-label"])
                            zoom_slider = gr.Slider(minimum=2, maximum=10, value=7, step=1, label="Thumbnail count", show_label=False, container=False, elem_id="hy-zoom")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes=["gallery-side"]):
                        left_gallery = gr.Gallery(show_label=False, columns=5, height="calc(100vh - 130px)", object_fit="contain", preview=True, allow_preview=True, elem_classes=["grid-wrap"], elem_id="hy-left-gallery")
                    with gr.Column(scale=0, min_width=100, elem_classes=["move-col"]) as move_controls_col:
                        status_md = gr.Markdown("", elem_classes=["status-md"])
                        sel_info = gr.Markdown("Shift+click thumbnails to mark multiple images.", elem_classes=["sel-info"])
                        move_right_btn = gr.Button("Move >>", elem_id="hy-move-right")
                        fit_threshold_btn = gr.Button("Fit thresh to filter image", elem_id="hy-fit-threshold")
                        pin_selected_btn = gr.Button("Pin selected", elem_id="hy-pin-selected")
                        move_left_btn = gr.Button("<< Move", elem_id="hy-move-left")
                        clear_status_btn = gr.Button("Clear marked", elem_id="hy-clear-status")
                        clear_all_status_btn = gr.Button("Clear all", elem_id="hy-clear-all-status")
                    with gr.Column(scale=1, elem_classes=["gallery-side"]) as right_gallery_col:
                        right_gallery = gr.Gallery(show_label=False, columns=5, height="calc(100vh - 130px)", object_fit="contain", preview=True, allow_preview=True, elem_classes=["grid-wrap"], elem_id="hy-right-gallery")

        method_dd.change(
            fn=configure_controls,
            inputs=[method_dd],
            outputs=[promptmatch_group, imagereward_group, llmsearch_group, tagmatch_group, main_slider, aux_slider, aux_mid_btn, keep_pm_thresholds_cb, keep_ir_thresholds_cb, percentile_slider, percentile_mid_btn, method_note],
            queue=False,
        )
        method_dd.change(
            fn=refresh_tagmatch_vocab_state,
            inputs=[method_dd],
            outputs=[tagmatch_vocab_state],
            queue=False,
        )

        _score_folder_inputs = [method_dd, folder_input, model_dd, pos_prompt_tb, neg_prompt_tb, pm_segment_cb, ir_prompt_tb, ir_negative_prompt_tb, ir_penalty_weight_tb, llm_model_dd, llm_prompt_tb, llm_backend_dd, llm_shortlist_slider, tagmatch_tags_tb, main_slider, aux_slider, keep_pm_thresholds_cb, keep_ir_thresholds_cb]
        _score_folder_outputs = [left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, main_slider, aux_slider, percentile_slider, percentile_mid_btn, model_status_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col]
        assert len(_score_folder_outputs) == len(SCORE_FOLDER_OUTPUT_KEYS), "score-folder output wiring drifted"
        _preview_search_outputs = [promptgen_status_md, *_score_folder_outputs]
        assert len(_preview_search_outputs) == len(PREVIEW_SEARCH_OUTPUT_KEYS), "preview-search output wiring drifted"
        _view_with_controls_outputs = [left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col]
        assert len(_view_with_controls_outputs) == len(VIEW_WITH_CONTROLS_OUTPUT_KEYS), "view-with-controls output wiring drifted"

        promptmatch_run_btn.click(fn=score_folder, inputs=_score_folder_inputs, outputs=_score_folder_outputs)
        imagereward_run_btn.click(fn=score_folder, inputs=_score_folder_inputs, outputs=_score_folder_outputs)
        llmsearch_run_btn.click(fn=score_folder, inputs=_score_folder_inputs, outputs=_score_folder_outputs)
        tagmatch_run_btn.click(fn=score_folder, inputs=_score_folder_inputs, outputs=_score_folder_outputs)
        folder_input.submit(
            fn=load_folder_for_browse,
            inputs=[folder_input, main_slider, aux_slider],
            outputs=_view_with_controls_outputs,
        )
        load_folder_btn.click(
            fn=load_folder_for_browse,
            inputs=[folder_input, main_slider, aux_slider],
            outputs=_view_with_controls_outputs,
        )
        external_query_image.upload(
            fn=set_external_query_image,
            inputs=[external_query_image],
            outputs=[external_query_image, clear_external_query_btn, promptgen_status_md],
        )
        external_query_bridge.change(
            fn=set_external_query_from_bridge,
            inputs=[external_query_bridge],
            outputs=[external_query_image, clear_external_query_btn, promptgen_status_md, external_query_bridge],
        )
        clear_external_query_btn.click(
            fn=clear_external_query_image,
            outputs=[external_query_image, clear_external_query_btn, promptgen_status_md],
        )
        prompt_generator_dd.change(
            fn=update_prompt_generator,
            inputs=[prompt_generator_dd, generated_prompt_detail_slider, generated_prompt_tb],
            outputs=[promptgen_status_md, generated_prompt_tb],
        )
        generate_prompt_btn.click(
            fn=generate_prompt_from_preview,
            inputs=[prompt_generator_dd, generated_prompt_tb, generated_prompt_detail_slider],
            outputs=[promptgen_status_md, generated_prompt_tb, generated_prompt_detail_slider],
        )
        find_similar_btn.click(
            fn=find_similar_images,
            inputs=[folder_input, model_dd, main_slider, aux_slider],
            outputs=_preview_search_outputs,
        )
        find_same_person_btn.click(
            fn=find_same_person_images,
            inputs=[folder_input, main_slider, aux_slider],
            outputs=_preview_search_outputs,
        )
        find_object_btn.click(
            fn=find_objectsearch_images,
            inputs=[folder_input, main_slider, aux_slider],
            outputs=_preview_search_outputs,
        )
        generated_prompt_detail_slider.change(
            fn=update_generated_prompt_detail,
            inputs=[prompt_generator_dd, generated_prompt_detail_slider, generated_prompt_tb],
            outputs=[promptgen_status_md, generated_prompt_tb],
        )
        insert_prompt_btn.click(
            fn=insert_generated_prompt,
            inputs=[method_dd, generated_prompt_tb],
            outputs=[promptgen_status_md, pos_prompt_tb, ir_prompt_tb, llm_prompt_tb],
        )

        main_slider.input(fn=update_histogram_only, inputs=[main_slider, aux_slider], outputs=[hist_plot], queue=False)
        aux_slider.input(fn=update_histogram_only, inputs=[main_slider, aux_slider], outputs=[hist_plot], queue=False)
        main_slider.release(fn=update_split, inputs=[main_slider, aux_slider], outputs=_view_with_controls_outputs, queue=False)
        aux_slider.release(fn=update_split, inputs=[main_slider, aux_slider], outputs=_view_with_controls_outputs, queue=False)
        percentile_slider.input(
            fn=update_histogram_from_percentile,
            inputs=[percentile_slider, aux_slider],
            outputs=[hist_plot],
            queue=False,
        )
        percentile_slider.release(
            fn=set_from_percentile,
            inputs=[percentile_slider, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider],
            queue=False,
        )
        main_mid_btn.click(
            fn=reset_main_threshold_to_middle,
            inputs=[main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider],
        )
        aux_mid_btn.click(
            fn=reset_aux_threshold_to_middle,
            inputs=[main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider],
        )
        percentile_mid_btn.click(
            fn=reset_percentile_to_middle,
            inputs=[main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, percentile_slider],
        )
        zoom_slider.change(
            fn=update_zoom,
            inputs=[zoom_slider, main_slider, aux_slider],
            outputs=_view_with_controls_outputs,
        )
        proxy_display_cb.change(
            fn=update_proxy_display,
            inputs=[proxy_display_cb, main_slider, aux_slider],
            outputs=_view_with_controls_outputs,
        )
        ir_penalty_weight_tb.change(
            fn=update_imagereward_penalty_weight,
            inputs=[ir_penalty_weight_tb, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider],
        )
        shortcut_action.change(
            fn=handle_shortcut_action,
            inputs=[shortcut_action, method_dd, folder_input, model_dd, pos_prompt_tb, neg_prompt_tb, pm_segment_cb, ir_prompt_tb, ir_negative_prompt_tb, ir_penalty_weight_tb, llm_model_dd, llm_prompt_tb, llm_backend_dd, llm_shortlist_slider, tagmatch_tags_tb, main_slider, aux_slider, keep_pm_thresholds_cb, keep_ir_thresholds_cb],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, main_slider, aux_slider, percentile_slider, percentile_mid_btn, model_status_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col],
        )
        model_status_state.change(
            fn=refresh_promptmatch_model_dropdown,
            inputs=[model_dd],
            outputs=[model_dd],
        )
        model_status_state.change(
            fn=refresh_promptmatch_model_dropdown,
            inputs=[llm_model_dd],
            outputs=[llm_model_dd],
        )
        hist_width_tb.change(fn=handle_hist_width, inputs=[hist_width_tb, main_slider, aux_slider], outputs=[hist_plot])
        thumb_action.change(
            fn=handle_thumb_action,
            inputs=[thumb_action, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider],
        )
        move_right_btn.click(fn=move_right, inputs=[main_slider, aux_slider], outputs=_view_with_controls_outputs)
        fit_threshold_btn.click(fn=fit_threshold_to_targets, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider])
        pin_selected_btn.click(fn=pin_selected, inputs=[main_slider, aux_slider], outputs=_view_with_controls_outputs)
        move_left_btn.click(fn=move_left, inputs=[main_slider, aux_slider], outputs=_view_with_controls_outputs)
        clear_status_btn.click(fn=clear_status, inputs=[main_slider, aux_slider], outputs=_view_with_controls_outputs)
        clear_all_status_btn.click(fn=clear_all_status, inputs=[main_slider, aux_slider], outputs=_view_with_controls_outputs)
        hist_plot.select(fn=on_hist_click, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider])
        export_btn.click(
            fn=export_files,
            inputs=[main_slider, aux_slider, left_export_cb, right_export_cb, move_export_cb, left_export_name_tb, right_export_name_tb],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, external_query_image, clear_external_query_btn, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col],
        )

    return demo, css, tooltip_head(tooltips)


def create_setup_required_app(requirement_issues):
    issue_lines = "\n".join(f"    <li><code>{issue}</code></li>" for issue in requirement_issues) or "    <li><code>Unknown dependency mismatch</code></li>"
    css = """
    body, .gradio-container { background:#120d0d !important; color:#f4eaea !important; }
    .gradio-container { max-width:960px !important; margin:0 auto !important; padding:24px 16px 40px !important; }
    .setup-alert {
        border:2px solid #d86161;
        border-radius:14px;
        background:linear-gradient(180deg, rgba(62, 19, 19, 0.96), rgba(28, 11, 11, 0.98));
        box-shadow:0 0 0 2px rgba(255,255,255,0.03) inset, 0 18px 50px rgba(0,0,0,0.35);
        padding:22px 22px 18px;
    }
    .setup-alert h1 {
        margin:0 0 12px 0;
        color:#ffb3b3;
        font-size:2rem;
        line-height:1.1;
        font-family:monospace;
        text-transform:uppercase;
        letter-spacing:.05em;
    }
    .setup-alert p, .setup-alert li, .setup-alert code {
        font-family:monospace !important;
        font-size:1rem !important;
        line-height:1.55 !important;
        color:#f7eaea !important;
    }
    .setup-alert .cmd {
        margin:14px 0;
        padding:12px 14px;
        border-radius:10px;
        background:#0f0b0b;
        border:1px solid #7d3d3d;
        color:#fff1f1;
    }
    .setup-alert .soft {
        color:#e3c9c9 !important;
    }
    """
    body = f"""
<div class="setup-alert">
  <h1>Setup Update Required</h1>
  <p>The app code was updated, but the current <code>venv312</code> does not match the packages this version expects.</p>
  <p>Please rerun setup before using this version:</p>
  <div class="cmd"><code>{SETUP_SCRIPT_HINT}</code></div>
  <p>Dependency issues found:</p>
  <ul class="soft">
{issue_lines}
  </ul>
  <p class="soft">After setup finishes, restart the launcher and reload the page.</p>
</div>
"""
    with gr.Blocks(title=APP_WINDOW_TITLE) as demo:
        gr.HTML(body)
    return demo, css, ""


if __name__ == "__main__":
    requirement_issues = runtime_requirement_issues()
    if requirement_issues:
        print("[Startup check] Dependency mismatch detected. Please rerun setup.")
        for issue in requirement_issues:
            print(f"  - {issue}")
        app, css, head = create_setup_required_app(requirement_issues)
    else:
        app, css, head = create_app()
    port = resolve_server_port(7862, "HYBRIDSELECTOR_PORT")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, INPUT_FOLDER_NAME)
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=True,
        share=False,
        css=css,
        head=head,
        allowed_paths=get_allowed_paths(script_dir, source_dir),
    )
