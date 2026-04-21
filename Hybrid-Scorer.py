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

    def score_tagmatch_folder(image_paths, query_tags_str, progress):
        backend = ensure_tagmatch_model()
        session = backend["session"]
        tags = backend["tags"]
        input_name = backend["input_name"]

        image_signature = get_image_paths_signature(image_paths)
        can_reuse = (
            state.get("tagmatch_cached_signature") == image_signature
            and state.get("tagmatch_cached_tag_vectors") is not None
            and state.get("tagmatch_cached_feature_paths") is not None
        )

        if not can_reuse:
            batch_size = get_auto_batch_size(device, mode="tagmatch")
            print(f"[TagMatch] Running inference on {len(image_paths)} images "
                  f"(batch size {batch_size})")
            tag_vectors = {}
            total = len(image_paths)
            for batch_start in range(0, total, batch_size):
                batch_paths = image_paths[batch_start:batch_start + batch_size]
                batch_tensors = []
                batch_valid_paths = []
                for p in batch_paths:
                    disp = state.get("proxy_map", {}).get(p, p)
                    try:
                        with Image.open(disp) as src:
                            arr = tagmatch_prepare_image(src)
                        batch_tensors.append(arr)
                        batch_valid_paths.append(p)
                    except Exception:
                        tag_vectors[p] = {}
                if batch_tensors:
                    try:
                        batch_np = np.stack(batch_tensors, axis=0)
                        raw_out = session.run(None, {input_name: batch_np})[0]  # (B, num_tags), probabilities [0,1]
                        for i, p in enumerate(batch_valid_paths):
                            row = raw_out[i]
                            # numpy threshold: find indices above cutoff, then build a small dict
                            keep = np.where(row >= TAGMATCH_WD_MIN_CACHE_PROB)[0]
                            tag_vectors[p] = {tags[j]: float(row[j]) for j in keep}
                    except Exception as _e:
                        print(f"[TagMatch] Batch inference error: {_e}")
                        for p in batch_valid_paths:
                            tag_vectors[p] = {}
                done = min(batch_start + batch_size, total)
                progress(done / max(total, 1), desc=f"TagMatch inference {done}/{total}")

            state["tagmatch_cached_signature"] = image_signature
            state["tagmatch_cached_feature_paths"] = list(image_paths)
            state["tagmatch_cached_tag_vectors"] = tag_vectors
        else:
            print(f"[TagMatch] Reusing cached tag vectors for {len(image_paths)} images")
            tag_vectors = state["tagmatch_cached_tag_vectors"]

        # Score from cache using the current query tags (fast re-score on tag change)
        state["tagmatch_last_query_tags_str"] = query_tags_str or ""
        query_tags = [t.strip().lower() for t in (query_tags_str or "").split(",") if t.strip()]
        tag_set = set(tags)
        missing = [t for t in query_tags if t not in tag_set]
        if missing:
            print(f"[TagMatch] WARNING: these query tags are not in the model vocabulary and will score 0: {missing}")
        results = {}
        for p in image_paths:
            tv = tag_vectors.get(p, {})
            # Sum matching tag probabilities directly — no division by tag count.
            # Each matched tag contributes its full probability (0–1) × 100 to the score.
            # Having more tags in the query doesn't deflate individual matches.
            score = min(100.0, sum(tv.get(t, 0.0) for t in query_tags) * 100.0)
            results[os.path.basename(p)] = {
                "pos": score,
                "neg": None,
                "path": p,
                "failed": False,
            }
        return results


    def generated_prompt_variants_for(query_cache_key, generator_name, create=False):
        if not query_cache_key:
            return {}
        preview_bucket = state["generated_prompt_variants"].get(query_cache_key)
        if preview_bucket is None:
            if not create:
                return {}
            preview_bucket = {}
            state["generated_prompt_variants"][query_cache_key] = preview_bucket
        backend_bucket = preview_bucket.get(generator_name)
        if backend_bucket is None:
            if not create:
                return {}
            backend_bucket = {}
            preview_bucket[generator_name] = backend_bucket
        return backend_bucket

    def select_cached_generated_prompt(generator_name, detail_level, current_generated_prompt):
        detail_level, detail_label, _ = prompt_generator_detail_config(generator_name, detail_level)
        state["prompt_generator"] = generator_name
        state["generated_prompt_detail"] = detail_level
        query_ctx = active_query_image_context()
        query_key = query_ctx.get("cache_key")
        query_label = query_ctx.get("label") or state.get("generated_prompt_source")
        prompt_text = generated_prompt_variants_for(query_key, generator_name).get(detail_level)
        if prompt_text:
            state["generated_prompt"] = prompt_text
            state["generated_prompt_source"] = query_label
            state["generated_prompt_backend"] = generator_name
            state["generated_prompt_status"] = (
                f"Showing cached {detail_label.lower()} prompt for {query_label} via {generator_name}."
                f"{prompt_backend_warning_text(generator_name)}"
            )
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=prompt_text),
            )

        if query_label:
            state["generated_prompt_status"] = (
                f"{generator_name} {detail_label.lower()} prompt is not cached for {query_label}. Click generate to create it."
            )
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=current_generated_prompt),
            )

        state["generated_prompt_status"] = "Preview, drop, paste, or upload an image, then generate a prompt from it."
        return (
            gr.update(value=state["generated_prompt_status"]),
            gr.update(value=current_generated_prompt),
        )

    def run_florence_prompt_variant(image, task_prompt):
        model, processor = ensure_florence_model()
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=FLORENCE_MAX_NEW_TOKENS,
                num_beams=3,
            )

        raw_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        if florence_task_is_pure_text(task_prompt):
            parsed = raw_text
        else:
            parsed = processor.post_process_generation(
                raw_text,
                task=task_prompt,
                image_size=image.size,
            )
        return normalize_generated_prompt(
            extract_florence_caption(parsed, raw_text, task_prompt)
        )

    def prepare_huihui_gemma4_inputs(processor, image, system_prompt, user_prompt):
        attempts = []
        structured_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        placeholder_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        try:
            return processor.apply_chat_template(
                structured_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except Exception as exc:
            attempts.append(f"embedded-image chat template: {exc}")

        try:
            return processor.apply_chat_template(
                placeholder_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                images=[image],
            )
        except Exception as exc:
            attempts.append(f"placeholder chat template: {exc}")

        try:
            rendered = processor.apply_chat_template(
                placeholder_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, list):
                rendered = rendered[0] if rendered else ""
            return processor(text=rendered, images=[image], return_tensors="pt")
        except Exception as exc:
            attempts.append(f"rendered prompt fallback: {exc}")

        raise RuntimeError("Huihui Gemma 4 input preparation failed: " + " / ".join(attempts))

    def run_huihui_gemma4_prompt_variant(
        image,
        user_prompt,
        detail_level,
        system_prompt=None,
        normalizer=None,
        max_new_tokens_override=None,
        stop_sequences=None,
        temperature_override=None,
        top_p_override=None,
        top_k_override=None,
    ):
        system_prompt = system_prompt or (
            "You are a helpful multimodal prompt writer. "
            "Describe only concrete visible content. "
            "Follow the requested output format exactly and avoid refusals or moralizing."
        )
        if normalizer is None:
            normalizer = lambda text: normalize_generated_prompt(
                extract_huihui_gemma4_caption(text),
                keep_prose=(detail_level == 3),
            )
        max_new_tokens = max_new_tokens_override if max_new_tokens_override is not None else joycaption_max_new_tokens(detail_level)
        temperature = temperature_override if temperature_override is not None else 0.0
        top_p = top_p_override if top_p_override is not None else 1.0
        top_k = top_k_override if top_k_override is not None else 0
        do_sample = bool(temperature and temperature > 0.0)

        model, processor = ensure_huihui_gemma4_model()
        inputs = prepare_huihui_gemma4_inputs(processor, image, system_prompt, user_prompt)
        inputs = move_processor_batch_to_device(inputs, device, next(model.parameters()).dtype)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "use_cache": True,
        }

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generate_kwargs)

        prompt_len = 0
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        if input_ids is not None and hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
            prompt_len = int(input_ids.shape[1])
        generated_slice = generated_ids[0][prompt_len:] if prompt_len else generated_ids[0]
        text = processor.decode(generated_slice, skip_special_tokens=True)
        return normalizer(text)

    def run_joycaption_prompt_variant(generator_name, image, user_prompt, detail_level, system_prompt=None, normalizer=None, max_new_tokens_override=None, stop_sequences=None, temperature_override=None, top_p_override=None, top_k_override=None):
        system_prompt = system_prompt or (
            "You are a helpful image captioner. "
            "Describe only concrete visible content and write output that is useful as a text-to-image prompt. "
            "Follow the requested output style exactly, whether it asks for short tags, a compact prompt line, or natural prose. "
            "Do not begin with meta phrases like 'This image shows', 'In this image we can see', or 'You are looking at'."
        )
        if normalizer is None:
            normalizer = lambda text: normalize_generated_prompt(
                extract_joycaption_caption(text),
                keep_prose=(detail_level == 3),
            )
        max_new_tokens = max_new_tokens_override if max_new_tokens_override is not None else joycaption_max_new_tokens(detail_level)
        temperature = temperature_override if temperature_override is not None else 0.0
        top_p = top_p_override if top_p_override is not None else 1.0
        top_k = top_k_override if top_k_override is not None else 0
        do_sample = bool(temperature and temperature > 0.0)

        if generator_name == PROMPT_GENERATOR_JOYCAPTION:
            model, processor = ensure_joycaption_model()
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            convo_string = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=[convo_string], images=[image], return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    use_cache=True,
                )[0]

            prompt_len = inputs["input_ids"].shape[1]
            text = processor.tokenizer.decode(
                generated_ids[prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return normalizer(text)

        llm = ensure_joycaption_gguf_model()
        image = joycaption_gguf_prepare_image(image)
        data_url = image_to_data_url(image)
        try:
            from llama_cpp._utils import suppress_stdout_stderr
        except ImportError:
            suppress_stdout_stderr = None
        completion_params = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "stop": stop_sequences or ["</s>", "User:", "Assistant:"],
            "repeat_penalty": 1.1,
        }
        if top_k and top_k > 0:
            completion_params["top_k"] = top_k
        if suppress_stdout_stderr is None:
            response = llm.create_chat_completion(**completion_params)
        else:
            with suppress_stdout_stderr(disable=False):
                response = llm.create_chat_completion(**completion_params)
        try:
            text = response["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected JoyCaption GGUF response shape: {exc}") from exc
        return normalizer(text)

    def run_wd_tags_prompt_variant(image, top_n):
        """Run WD tagger on a single PIL image and return the top-N tags as a comma-separated string."""
        backend = ensure_tagmatch_model()
        session = backend["session"]
        tags = backend["tags"]
        input_name = backend["input_name"]
        arr = tagmatch_prepare_image(image)          # (448, 448, 3) float32
        batch_np = arr[np.newaxis, ...]              # (1, 448, 448, 3)
        raw_out = session.run(None, {input_name: batch_np})[0][0]  # (num_tags,)
        probs = 1.0 / (1.0 + np.exp(-raw_out))
        tag_prob_pairs = sorted(
            ((tags[j], float(probs[j])) for j in range(len(tags))),
            key=lambda x: x[1],
            reverse=True,
        )
        top_tags = [t for t, p in tag_prob_pairs[:top_n] if p >= 0.05]
        return ", ".join(top_tags)

    def generate_prompt_variant(generator_name, image, detail_level):
        _, _, detail_prompt = prompt_generator_detail_config(generator_name, detail_level)
        if generator_name == PROMPT_GENERATOR_FLORENCE:
            return run_florence_prompt_variant(image, detail_prompt)
        if generator_name == PROMPT_GENERATOR_WD_TAGS:
            return run_wd_tags_prompt_variant(image, detail_prompt)  # detail_prompt is top_n int
        if generator_name == PROMPT_GENERATOR_HUIHUI_GEMMA4:
            return run_huihui_gemma4_prompt_variant(image, detail_prompt, detail_level)
        return run_joycaption_prompt_variant(generator_name, image, detail_prompt, detail_level)

    class VisionLLMRerankBackend:
        def __init__(self, backend_id):
            self.backend_id = backend_id

        def describe_source(self):
            return describe_llmsearch_backend_source(self.backend_id)

        def is_available(self):
            return True

        def load(self, progress):
            if self.backend_id == PROMPT_GENERATOR_FLORENCE:
                progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
                ensure_florence_model()
            elif self.backend_id == PROMPT_GENERATOR_JOYCAPTION:
                progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
                ensure_joycaption_model()
            elif self.backend_id == PROMPT_GENERATOR_JOYCAPTION_GGUF:
                progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
                ensure_joycaption_gguf_model()
            elif self.backend_id == PROMPT_GENERATOR_HUIHUI_GEMMA4:
                progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
                ensure_huihui_gemma4_model()
            else:
                raise RuntimeError(f"Unknown LLM rerank backend: {self.backend_id}")

        def release(self):
            return None

        def uses_direct_numeric_score(self):
            return self.backend_id in {
                PROMPT_GENERATOR_JOYCAPTION,
                PROMPT_GENERATOR_JOYCAPTION_GGUF,
                PROMPT_GENERATOR_HUIHUI_GEMMA4,
            }

        def candidate_text(self, image, query_text):
            if self.backend_id == PROMPT_GENERATOR_FLORENCE:
                return run_florence_prompt_variant(image, "<MORE_DETAILED_CAPTION>")
            if self.backend_id == PROMPT_GENERATOR_HUIHUI_GEMMA4:
                return run_huihui_gemma4_prompt_variant(
                    image,
                    build_llmsearch_huihui_gemma4_user_prompt(query_text),
                    2,
                    system_prompt=llmsearch_joycaption_system_prompt(),
                    normalizer=lambda text: normalize_generated_prompt(
                        extract_huihui_gemma4_caption(text)
                    ),
                    max_new_tokens_override=joycaption_max_new_tokens(2),
                )

            user_prompt = build_llmsearch_joycaption_user_prompt(query_text)
            return run_joycaption_prompt_variant(
                self.backend_id,
                image,
                user_prompt,
                2,
                system_prompt=llmsearch_joycaption_system_prompt(),
                normalizer=normalize_llmsearch_candidate_text,
            )

        def score_candidate(self, image, query_text):
            if not self.uses_direct_numeric_score():
                caption_text = self.candidate_text(image, query_text)
                return llmsearch_similarity(state["backend"].encode_text((query_text or "").strip()), caption_text), caption_text

            if self.backend_id == PROMPT_GENERATOR_HUIHUI_GEMMA4:
                raw_score_text = run_huihui_gemma4_prompt_variant(
                    image,
                    build_llmsearch_huihui_gemma4_user_prompt(query_text),
                    1,
                    system_prompt=llmsearch_joycaption_system_prompt(),
                    normalizer=extract_huihui_gemma4_caption,
                    max_new_tokens_override=LLMSEARCH_HUIHUI_GEMMA4_MAX_NEW_TOKENS,
                    stop_sequences=[" ", "\n", "\t", ",", ".", "%", "</s>", "User:", "Assistant:"],
                    temperature_override=LLMSEARCH_HUIHUI_GEMMA4_TEMPERATURE,
                    top_p_override=LLMSEARCH_HUIHUI_GEMMA4_TOP_P,
                    top_k_override=LLMSEARCH_HUIHUI_GEMMA4_TOP_K,
                )
            else:
                raw_score_text = run_joycaption_prompt_variant(
                    self.backend_id,
                    image,
                    build_llmsearch_joycaption_user_prompt(query_text),
                    1,
                    system_prompt=llmsearch_joycaption_system_prompt(),
                    normalizer=extract_joycaption_caption,
                    max_new_tokens_override=LLMSEARCH_JOYCAPTION_MAX_NEW_TOKENS,
                    stop_sequences=[" ", "\n", "\t", ",", ".", "%", "</s>", "User:", "Assistant:"],
                    temperature_override=LLMSEARCH_JOYCAPTION_TEMPERATURE,
                    top_p_override=LLMSEARCH_JOYCAPTION_TOP_P,
                    top_k_override=LLMSEARCH_JOYCAPTION_TOP_K,
                )
            return float(extract_llmsearch_numeric_score(raw_score_text)), raw_score_text

        def score_candidates_batch(self, images, query_text):
            """Score a batch of PIL images with the HF JoyCaption backend.
            Returns a list of (score_float, raw_text) pairs, one per image.
            Falls back to sequential for non-HF backends."""
            if self.backend_id != PROMPT_GENERATOR_JOYCAPTION:
                return [self.score_candidate(img, query_text) for img in images]
            model, processor = ensure_joycaption_model()
            user_prompt = build_llmsearch_joycaption_user_prompt(query_text)
            conversation = [
                {"role": "system", "content": llmsearch_joycaption_system_prompt()},
                {"role": "user", "content": user_prompt},
            ]
            convo_string = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            n = len(images)
            inputs = processor(text=[convo_string] * n, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)
            do_sample = bool(LLMSEARCH_JOYCAPTION_TEMPERATURE and LLMSEARCH_JOYCAPTION_TEMPERATURE > 0.0)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=LLMSEARCH_JOYCAPTION_MAX_NEW_TOKENS,
                    do_sample=do_sample,
                    temperature=LLMSEARCH_JOYCAPTION_TEMPERATURE,
                    top_p=LLMSEARCH_JOYCAPTION_TOP_P,
                    top_k=LLMSEARCH_JOYCAPTION_TOP_K,
                    use_cache=True,
                )
            prompt_len = inputs["input_ids"].shape[1]
            results = []
            for i in range(n):
                text = processor.tokenizer.decode(
                    generated_ids[i][prompt_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                raw = extract_joycaption_caption(text)
                try:
                    score = float(extract_llmsearch_numeric_score(raw))
                except Exception:
                    score = 0.0
                results.append((score, raw))
            return results

    def get_llmsearch_backend(backend_id):
        return VisionLLMRerankBackend(backend_id)

    def llmsearch_caption_cache_key(backend_id, image_signature, query_text):
        return (
            str(backend_id),
            LLMSEARCH_SCORING_MODE_NUMERIC_V1,
            str(image_signature),
            normalize_prompt_text(query_text or ""),
        )

    def llmsearch_similarity(query_embedding, text):
        text = normalize_prompt_text(text)
        if not text:
            return -1.0
        text_embedding = state["backend"].encode_text(text)
        score = float((query_embedding @ text_embedding.T).squeeze().item())
        return round(score, 6)

    def score_llmsearch_candidates(candidate_paths, query_text, backend_id, image_signature, progress):
        backend = get_llmsearch_backend(backend_id)
        backend.load(progress)
        cache_key = llmsearch_caption_cache_key(backend_id, image_signature, query_text)
        caption_cache = state["llmsearch_cached_captions"].setdefault(cache_key, {})
        total = len(candidate_paths)
        results = {}
        # Batch pre-pass for HF JoyCaption: score all uncached images in chunks to improve GPU utilization.
        # The main loop below reads from cache; any image not pre-scored falls back to sequential inference.
        if backend_id == PROMPT_GENERATOR_JOYCAPTION:
            needs_inference = [
                p for p in candidate_paths
                if not (isinstance(caption_cache.get(p), dict) and caption_cache[p].get("score") is not None)
            ]
            if needs_inference:
                print(f"[LLM Search] Batch scoring {len(needs_inference)}/{total} uncached images "
                      f"(batch size {LLMSEARCH_JOYCAPTION_HF_BATCH_SIZE})")
            for batch_start in range(0, len(needs_inference), LLMSEARCH_JOYCAPTION_HF_BATCH_SIZE):
                batch_paths = needs_inference[batch_start:batch_start + LLMSEARCH_JOYCAPTION_HF_BATCH_SIZE]
                loaded = []
                for p in batch_paths:
                    disp = state.get("proxy_map", {}).get(p, p)
                    try:
                        with Image.open(disp) as src:
                            loaded.append((p, src.convert("RGB")))
                    except Exception:
                        pass  # Will be retried sequentially in main loop
                if loaded:
                    try:
                        batch_scored = backend.score_candidates_batch([img for _, img in loaded], query_text)
                        for (p, _), (sv, ct) in zip(loaded, batch_scored):
                            caption_cache[p] = {"score": float(sv), "text": ct}
                    except Exception:
                        pass  # Batch failed entirely; main loop will retry individually
                done_so_far = batch_start + len(batch_paths)
                progress(done_so_far / max(total, 1), desc=f"LLM reranking {done_so_far}/{total} via {backend_id}")
        for index, original_path in enumerate(candidate_paths, start=1):
            cached_value = caption_cache.get(original_path)
            failed_reason = None
            score_value = None
            caption_text = ""
            if isinstance(cached_value, dict):
                score_value = cached_value.get("score")
                caption_text = cached_value.get("text") or ""
            elif isinstance(cached_value, str):
                caption_text = cached_value

            if score_value is None and not caption_text and backend.uses_direct_numeric_score():
                display_path = state.get("proxy_map", {}).get(original_path, original_path)
                try:
                    with Image.open(display_path) as src_img:
                        image = src_img.convert("RGB")
                    score_value, caption_text = backend.score_candidate(image, query_text)
                    caption_cache[original_path] = {
                        "score": float(score_value),
                        "text": caption_text,
                    }
                except Exception as exc:
                    failed_reason = str(exc) or "LLM rerank backend failed."
                    score_value = 0.0
                    caption_text = ""
            elif not caption_text:
                display_path = state.get("proxy_map", {}).get(original_path, original_path)
                try:
                    with Image.open(display_path) as src_img:
                        image = src_img.convert("RGB")
                    caption_text = backend.candidate_text(image, query_text)
                    caption_cache[original_path] = caption_text
                except Exception as exc:
                    failed_reason = str(exc) or "LLM rerank backend failed."
                    caption_text = ""
            if failed_reason:
                score_value = 0.0 if backend.uses_direct_numeric_score() else -1.0
            elif score_value is None:
                try:
                    score_value = llmsearch_similarity(state["backend"].encode_text((query_text or "").strip()), caption_text)
                except Exception as exc:
                    failed_reason = str(exc) or "LLM rerank text scoring failed."
                    score_value = -1.0

            results[os.path.basename(original_path)] = {
                "pos": float(score_value),
                "neg": None,
                "path": original_path,
                "failed": bool(failed_reason),
                "caption": caption_text,
                "reason": failed_reason,
            }
            progress(index / max(total, 1), desc=f"LLM reranking {index}/{total} via {backend_id}")

        return results

    def score_imagereward(folder_paths, positive_prompt, negative_prompt, penalty_weight, progress):
        # Optional penalty prompt is implemented as a second pass whose score is subtracted.
        model = ensure_imagereward_model()
        positive_prompt = (positive_prompt or "").strip() or IR_PROMPT
        negative_prompt = (negative_prompt or "").strip()
        penalty_weight = float(penalty_weight)
        state["ir_penalty_weight"] = penalty_weight
        image_signature = get_image_paths_signature(folder_paths)
        proxy_map = {}
        cache_dir = state.get("proxy_cache_dir")
        scoring_paths = list(folder_paths)

        if cache_dir and can_reuse_proxy_map(folder_paths, image_signature):
            proxy_map = dict(state.get("proxy_map") or {})
            scoring_paths = [proxy_map.get(path, path) for path in folder_paths]
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Reusing cached ImageReward proxies for {len(folder_paths)} images")
        elif cache_dir:
            print(f"[ImageReward] Proxy cache dir: {cache_dir}")
            def _proxy_prep_cb(done, total, generated, reused):
                desc = f"Preparing ImageReward proxies {done}/{total}"
                if generated or reused:
                    desc += f" ({generated} new, {reused} reused)"
                progress(PROMPTMATCH_PROXY_PROGRESS_SHARE * (done / max(total, 1)), desc=desc)

            progress(0, desc=f"Preparing ImageReward proxies 0/{len(folder_paths)}")
            proxy_map, generated, reused = prepare_promptmatch_proxies(
                folder_paths,
                cache_dir,
                progress_cb=_proxy_prep_cb,
            )
            state["proxy_map"] = dict(proxy_map)
            state["proxy_signature"] = image_signature
            scoring_paths = [proxy_map.get(path, path) for path in folder_paths]
            print(f"[ImageReward] Proxy prep complete in {cache_dir}: {generated} new, {reused} reused")

        base_scores = {}
        can_reuse_base = (
            state.get("ir_cached_signature") == image_signature
            and state.get("ir_cached_positive_prompt") == positive_prompt
            and state.get("ir_cached_base_scores") is not None
        )
        if can_reuse_base:
            base_scores = dict(state["ir_cached_base_scores"])
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Reusing cached ImageReward positive pass for {len(folder_paths)} images")
        else:
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Scoring {len(folder_paths)} images with ImageReward...")
            for event in iter_imagereward_scores(scoring_paths, model, device, positive_prompt, source_paths=folder_paths):
                if event["type"] == "oom":
                    progress(
                        PROMPTMATCH_PROXY_PROGRESS_SHARE
                        + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                        desc=f"ImageReward OOM, retrying autobatch {event['batch_size']}",
                    )
                    continue
                progress(
                    PROMPTMATCH_PROXY_PROGRESS_SHARE
                    + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                    desc=f"ImageReward {event['done']}/{event['total']} (autobatch {event['batch_size']})",
                )
                base_scores = event["scores"]
            state["ir_cached_signature"] = image_signature
            state["ir_cached_positive_prompt"] = positive_prompt
            state["ir_cached_base_scores"] = dict(base_scores)

        penalty_scores = {}
        if negative_prompt:
            can_reuse_penalty = (
                state.get("ir_cached_signature") == image_signature
                and state.get("ir_cached_negative_prompt") == negative_prompt
                and state.get("ir_cached_penalty_scores") is not None
            )
            if can_reuse_penalty:
                penalty_scores = dict(state["ir_cached_penalty_scores"])
                progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Reusing cached penalty pass for {len(folder_paths)} images")
            else:
                progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Applying penalty prompt to {len(folder_paths)} images...")
                for event in iter_imagereward_scores(scoring_paths, model, device, negative_prompt, source_paths=folder_paths):
                    if event["type"] == "oom":
                        progress(
                            PROMPTMATCH_PROXY_PROGRESS_SHARE
                            + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                            desc=f"Penalty OOM, retrying autobatch {event['batch_size']}",
                        )
                        continue
                    progress(
                        PROMPTMATCH_PROXY_PROGRESS_SHARE
                        + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                        desc=f"Penalty prompt {event['done']}/{event['total']} (autobatch {event['batch_size']})",
                    )
                    penalty_scores = event["scores"]
                state["ir_cached_signature"] = image_signature
                state["ir_cached_negative_prompt"] = negative_prompt
                state["ir_cached_penalty_scores"] = dict(penalty_scores)
        else:
            state["ir_cached_negative_prompt"] = ""
            state["ir_cached_penalty_scores"] = {}

        penalty_offset = get_imagereward_penalty_offset(
            item["score"] for item in penalty_scores.values()
        )
        wrapped = {}
        for path in folder_paths:
            fname = os.path.basename(path)
            base_item = base_scores.get(fname, {"score": -float("inf"), "path": path})
            penalty_item = penalty_scores.get(fname)
            penalty_value = penalty_item["score"] if penalty_item is not None else None
            final_score = compute_imagereward_final_score(
                base_item["score"],
                penalty_value,
                penalty_weight,
                penalty_offset=penalty_offset,
            )
            wrapped[fname] = {
                "score": float(final_score),
                "base": float(base_item["score"]),
                "penalty": float(penalty_value) if penalty_value is not None else None,
                "path": path,
            }
        return wrapped

    def recompute_imagereward_scores(penalty_weight):
        penalty_weight = float(penalty_weight)
        state["ir_penalty_weight"] = penalty_weight
        if state["method"] != METHOD_IMAGEREWARD or not state["scores"]:
            return False

        changed = False
        penalty_offset = get_imagereward_penalty_offset(
            item.get("penalty") for item in state["scores"].values() if "base" in item
        )
        for item in state["scores"].values():
            if "base" not in item:
                continue
            item["score"] = compute_imagereward_final_score(
                item["base"],
                item.get("penalty"),
                penalty_weight,
                penalty_offset=penalty_offset,
            )
            changed = True
        return changed


    def load_folder_for_browse(folder, main_threshold, aux_threshold, progress=gr.Progress()):
        folder = (folder or "").strip()
        if not folder or not os.path.isdir(folder):
            set_browse_folder_state(folder, [], f"Invalid folder: {folder!r}")
            return (*render_view_with_controls(main_threshold, aux_threshold),)

        image_paths = scan_image_paths(folder)
        if not image_paths:
            set_browse_folder_state(folder, [], f"No images found in {folder}")
            return (*render_view_with_controls(main_threshold, aux_threshold),)

        sync_promptmatch_proxy_cache(folder)
        image_signature = get_image_paths_signature(image_paths)
        proxy_map = {}
        cache_dir = state.get("proxy_cache_dir")
        if cache_dir and can_reuse_proxy_map(image_paths, image_signature):
            proxy_map = dict(state.get("proxy_map") or {})
            progress(1.0, desc=f"Reusing cached proxies for {len(image_paths)} images")
        elif cache_dir:
            def _proxy_prep_cb(done, total, generated, reused):
                desc = f"Preparing browse proxies {done}/{total}"
                if generated or reused:
                    desc += f" ({generated} new, {reused} reused)"
                progress(done / max(total, 1), desc=desc)

            progress(0, desc=f"Preparing browse proxies 0/{len(image_paths)}")
            proxy_map, _, _ = prepare_promptmatch_proxies(
                image_paths,
                cache_dir,
                progress_cb=_proxy_prep_cb,
            )
            state["proxy_map"] = dict(proxy_map)
            state["proxy_signature"] = image_signature

        browse_items = [(path, os.path.basename(path)) for path in image_paths]
        set_browse_folder_state(
            folder,
            browse_items,
            f"Browse mode for {folder}. {len(image_paths)} images loaded. Preview a gallery image or use the external query image to search or generate a prompt.",
        )
        return (*render_view_with_controls(main_threshold, aux_threshold),)

    def refresh_promptmatch_model_dropdown(current_model_label):
        selected = current_model_label if current_model_label in MODEL_LABELS else MODEL_LABELS[0]
        return gr.update(choices=promptmatch_model_dropdown_choices(), value=selected)

    def middle_threshold_values(method):
        if method == METHOD_LLMSEARCH:
            lo, hi, mid = llmsearch_slider_range(state["scores"], state.get("llmsearch_backend"))
            return round(float(mid), 3), NEGATIVE_THRESHOLD, False
        if uses_pos_similarity_scores(method):
            if state["scores"]:
                _, _, pos_mid, _, _, neg_mid, has_neg = promptmatch_slider_range(state["scores"])
            else:
                pos_mid, neg_mid, has_neg = 0.14, NEGATIVE_THRESHOLD, True
            return round(float(pos_mid), 3), round(float(neg_mid), 3), bool(has_neg)
        if state["scores"]:
            _, _, main_mid = imagereward_slider_range(state["scores"])
        else:
            main_mid = IMAGEREWARD_THRESHOLD
        return round(float(main_mid), 3), NEGATIVE_THRESHOLD, False

    def score_similarity_cached_features(feature_paths, image_features, failed_paths, query_path, query_feature=None):
        if not feature_paths or image_features is None or image_features.numel() == 0:
            raise RuntimeError("No PromptMatch image embeddings are available for similarity search.")
        if query_feature is None:
            if query_path in failed_paths:
                raise RuntimeError(f"{os.path.basename(query_path)} could not be encoded for similarity search.")
            try:
                query_index = feature_paths.index(query_path)
            except ValueError as exc:
                raise RuntimeError(f"{os.path.basename(query_path)} is missing from the cached PromptMatch embeddings.") from exc
            query_feature = image_features[query_index:query_index + 1]
        else:
            query_feature = query_feature.detach().float().cpu()

        results = {}
        sims = (image_features @ query_feature.T).squeeze(1).tolist()
        for original_path, score in zip(feature_paths, sims):
            fname = os.path.basename(original_path)
            results[fname] = {
                "pos": float(score),
                "neg": None,
                "path": original_path,
                "failed": False,
                "query": original_path == query_path,
            }

        for original_path in failed_paths:
            fname = os.path.basename(original_path)
            results[fname] = {
                "pos": 0.0,
                "neg": None,
                "path": original_path,
                "failed": True,
                "query": original_path == query_path,
            }
        return results

    def score_sameperson_cached_features(feature_paths, face_embeddings, failures, query_path, query_embedding=None):
        if not feature_paths or face_embeddings is None or face_embeddings.numel() == 0:
            raise RuntimeError("No face embeddings are available for same-person search.")
        if query_embedding is None:
            if query_path in failures:
                raise RuntimeError(f"{os.path.basename(query_path)}: {failures[query_path]}")
            try:
                query_index = feature_paths.index(query_path)
            except ValueError as exc:
                raise RuntimeError(f"{os.path.basename(query_path)} is missing from the cached face embeddings.") from exc
            query_embedding = face_embeddings[query_index:query_index + 1]
        else:
            query_embedding = F.normalize(query_embedding.detach().float().cpu(), dim=1)

        results = {}
        sims = (face_embeddings @ query_embedding.T).squeeze(1).tolist()
        for original_path, score in zip(feature_paths, sims):
            fname = os.path.basename(original_path)
            results[fname] = {
                "pos": float(score),
                "neg": None,
                "path": original_path,
                "failed": False,
                "query": original_path == query_path,
            }

        for failed_path, reason in failures.items():
            fname = os.path.basename(failed_path)
            results[fname] = {
                "pos": 0.0,
                "neg": None,
                "path": failed_path,
                "failed": True,
                "query": failed_path == query_path,
                "reason": reason,
            }
        return results

    def prepare_scored_run_context(method, folder, main_threshold, aux_threshold, llm_backend_id=None):
        folder = (folder or "").strip()
        if not folder or not os.path.isdir(folder):
            return None, empty_result(f"Invalid folder: {folder!r}", method)

        image_paths = scan_image_paths(folder)
        if not image_paths:
            return None, empty_result(f"No images found in {folder}", method)

        folder_key = normalize_folder_identity(folder)
        ctx = {
            "folder": folder,
            "folder_key": folder_key,
            "image_paths": image_paths,
            "image_signature": get_image_paths_signature(image_paths),
            "main_label": threshold_labels(method)[0],
            "aux_label": threshold_labels(method)[1],
            "previous_method": state.get("method"),
            "requested_main": float(main_threshold),
            "requested_aux": float(aux_threshold),
        }
        ctx["requested_main"], ctx["requested_aux"] = normalize_threshold_inputs(
            method,
            ctx["requested_main"],
            ctx["requested_aux"],
            llm_backend_id,
        )
        recalled_main, recalled_aux, has_recalled = recalled_mode_thresholds(method, main_threshold, aux_threshold)
        ctx["has_recalled"] = has_recalled
        if ctx["previous_method"] != method and has_recalled:
            ctx["requested_main"], ctx["requested_aux"] = normalize_threshold_inputs(
                method,
                recalled_main,
                recalled_aux,
                llm_backend_id,
            )
        ctx["preserved_overrides"] = preserve_overrides_for_folder_key(
            folder_key,
            {os.path.basename(path) for path in image_paths},
        )
        return ctx, None

    def render_scored_mode_result(method, next_main, next_aux, main_upd, aux_upd, percentile_upd, percentile_mid_upd):
        return build_scored_callback_result(
            current_view(next_main, next_aux),
            main_upd,
            aux_upd,
            percentile_upd,
            percentile_mid_upd,
        )

    def normalize_preview_search_request(folder, preview_missing_message, preview_not_in_folder_template):
        folder = (folder or "").strip()
        if not folder or not os.path.isdir(folder):
            return None, f"Invalid folder: {folder!r}"

        image_paths = scan_image_paths(folder)
        if not image_paths:
            return None, f"No images found in {folder}"

        query_ctx = active_query_image_context()
        query_path = query_ctx.get("path")
        query_label = query_ctx.get("label")
        query_source = query_ctx.get("source_label")
        preview_fname = query_ctx.get("preview_fname")
        if not query_path:
            return None, preview_missing_message

        folder_name_map = {os.path.basename(path): path for path in image_paths}
        query_in_folder = query_ctx.get("source_kind") == "gallery"
        if query_in_folder:
            query_path = folder_name_map.get(preview_fname)
        if query_in_folder and not query_path:
            return None, preview_not_in_folder_template.format(preview_fname=preview_fname)

        folder_key = normalize_folder_identity(folder)
        return {
            "folder": folder,
            "folder_key": folder_key,
            "image_paths": image_paths,
            "query_path": query_path,
            "query_label": query_label or preview_fname,
            "query_source": query_source,
            "query_in_folder": query_in_folder,
            "preview_fname": preview_fname,
            "preserved_overrides": preserve_overrides_for_folder_key(folder_key, set(folder_name_map.keys())),
        }, None

    def score_folder(method, folder, model_label, pos_prompt, neg_prompt, pm_segment_mode, ir_prompt, ir_negative_prompt, ir_penalty_weight, llm_model_label, llm_prompt, llm_backend_id, llm_shortlist_size, tagmatch_tags, main_threshold, aux_threshold, keep_pm_thresholds, keep_ir_thresholds, progress=gr.Progress()):
        # Main entrypoint for "Run scoring"; both methods converge back into current_view().
        run_ctx, error_result = prepare_scored_run_context(method, folder, main_threshold, aux_threshold, llm_backend_id)
        if error_result is not None:
            return error_result
        folder = run_ctx["folder"]
        image_paths = run_ctx["image_paths"]
        image_signature = run_ctx["image_signature"]
        folder_key = run_ctx["folder_key"]
        main_label = run_ctx["main_label"]
        aux_label = run_ctx["aux_label"]
        previous_method = run_ctx["previous_method"]
        requested_main = run_ctx["requested_main"]
        requested_aux = run_ctx["requested_aux"]
        has_recalled = run_ctx["has_recalled"]
        preserved_overrides = run_ctx["preserved_overrides"]
        preserve_promptmatch_thresholds = (
            bool(keep_pm_thresholds)
            and method == METHOD_PROMPTMATCH
            and state.get("last_scored_method") == METHOD_PROMPTMATCH
            and state.get("last_scored_folder_key") == folder_key
            and state.get("last_promptmatch_model_label") == model_label
        )
        preserve_imagereward_threshold = (
            bool(keep_ir_thresholds)
            and method == METHOD_IMAGEREWARD
            and state.get("last_scored_method") == METHOD_IMAGEREWARD
            and state.get("last_scored_folder_key") == folder_key
        )

        begin_scored_run(method, folder, preserved_overrides)
        release_inactive_gpu_models(method)

        if method == METHOD_LLMSEARCH:
            llm_model_label = llm_model_label if llm_model_label in MODEL_LABELS else label_for_backend(prompt_backend)
            llm_backend_id = llm_backend_id if llm_backend_id in llmsearch_backend_choices() else DEFAULT_LLMSEARCH_BACKEND
            llm_prompt = (llm_prompt or "").strip() or LLMSEARCH_DEFAULT_PROMPT
            try:
                shortlist_size = int(float(llm_shortlist_size))
            except Exception:
                shortlist_size = LLMSEARCH_SHORTLIST_DEFAULT
            shortlist_size = max(LLMSEARCH_SHORTLIST_MIN, min(LLMSEARCH_SHORTLIST_MAX, shortlist_size))
            state["llmsearch_backend"] = llm_backend_id
            state["llmsearch_shortlist_size"] = shortlist_size

            can_reuse_llm_cache = (
                state.get("llmsearch_cached_signature") == image_signature
                and state.get("llmsearch_cached_prompt") == llm_prompt
                and state.get("llmsearch_cached_backend") == llm_backend_id
                and state.get("llmsearch_cached_scoring_mode") == LLMSEARCH_SCORING_MODE_NUMERIC_V1
                and state.get("llmsearch_cached_shortlist_size") == shortlist_size
                and state.get("llmsearch_cached_model_label") == llm_model_label
                and state.get("llmsearch_cached_scores") is not None
            )
            if can_reuse_llm_cache:
                state["scores"] = dict(state["llmsearch_cached_scores"])
            else:
                try:
                    ensure_promptmatch_backend_loaded(llm_model_label, progress)
                    _, feature_paths, image_features, failed_paths = ensure_promptmatch_feature_cache(
                        image_paths,
                        llm_model_label,
                        progress,
                        reuse_desc="Reusing cached LLM-search shortlist embeddings for {count} images",
                        encode_desc="Encoding LLM-search shortlist embeddings for {count} images...",
                        progress_label="LLM shortlist",
                    )
                except Exception as exc:
                    return empty_result(str(exc), method)

                shortlist_query_emb = state["backend"].encode_text(llm_prompt)
                shortlist_scores = score_promptmatch_cached_features(
                    feature_paths,
                    image_features,
                    failed_paths,
                    shortlist_query_emb,
                    None,
                )
                ranked_candidates = [
                    item for item in shortlist_scores.values()
                    if not item.get("failed", False) and item.get("pos") is not None
                ]
                ranked_candidates.sort(key=lambda item: -float(item["pos"]))
                candidate_paths = [item["path"] for item in ranked_candidates[:shortlist_size]]
                if not candidate_paths:
                    return empty_result("LM search could not shortlist any usable images.", method)

                try:
                    llm_candidate_scores = score_llmsearch_candidates(
                        candidate_paths,
                        llm_prompt,
                        llm_backend_id,
                        image_signature,
                        progress,
                    )
                except Exception as exc:
                    return empty_result(f"LLM rerank failed: {exc}", method)

                shortlist_floor_candidates = [
                    item["pos"] for item in llm_candidate_scores.values()
                    if not item.get("failed", False)
                ]
                shortlist_floor = min(shortlist_floor_candidates) if shortlist_floor_candidates else -0.2
                reject_floor = max(-1.0, float(shortlist_floor) - 0.05)
                wrapped_scores = {}
                shortlisted_names = set(llm_candidate_scores.keys())
                for path in image_paths:
                    fname = os.path.basename(path)
                    base_item = shortlist_scores.get(fname)
                    llm_item = llm_candidate_scores.get(fname)
                    if llm_item is not None:
                        wrapped_scores[fname] = {
                            **llm_item,
                            "base_pos": float(base_item["pos"]) if base_item and base_item.get("pos") is not None else None,
                        }
                        continue
                    wrapped_scores[fname] = {
                        "pos": float(reject_floor),
                        "neg": None,
                        "path": path,
                        "failed": False,
                        "base_pos": float(base_item["pos"]) if base_item and base_item.get("pos") is not None else None,
                        "shortlisted": False,
                    }

                state["scores"] = wrapped_scores
                state["llmsearch_cached_signature"] = image_signature
                state["llmsearch_cached_prompt"] = llm_prompt
                state["llmsearch_cached_backend"] = llm_backend_id
                state["llmsearch_cached_scoring_mode"] = LLMSEARCH_SCORING_MODE_NUMERIC_V1
                state["llmsearch_cached_shortlist_size"] = shortlist_size
                state["llmsearch_cached_model_label"] = llm_model_label
                state["llmsearch_cached_scores"] = dict(wrapped_scores)

            pos_min, pos_max, pos_mid = llmsearch_slider_range(state["scores"], llm_backend_id)
            default_main = pos_mid
            should_preserve_main = (previous_method == METHOD_LLMSEARCH) or has_recalled
            next_main = clamp_threshold(requested_main, pos_min, pos_max) if should_preserve_main else default_main
            safe_pos_min, safe_pos_max = expand_slider_bounds(pos_min, pos_max, next_main)
            aux_default = 50.0 if llmsearch_uses_numeric_scores(llm_backend_id) else NEGATIVE_THRESHOLD
            safe_neg_min, safe_neg_max = expand_slider_bounds(aux_default, aux_default, aux_default)
            return render_scored_mode_result(
                METHOD_LLMSEARCH,
                next_main,
                aux_default,
                gr.update(minimum=safe_pos_min, maximum=safe_pos_max, value=next_main, label=main_label),
                gr.update(minimum=safe_neg_min, maximum=safe_neg_max, value=aux_default, visible=False, interactive=False, label=aux_label),
                percentile_slider_update(METHOD_LLMSEARCH, state["scores"]),
                percentile_reset_button_update(METHOD_LLMSEARCH, state["scores"]),
            )

        if method == METHOD_PROMPTMATCH:
            try:
                ensure_promptmatch_backend_loaded(model_label, progress)
                _, feature_paths, image_features, failed_paths = ensure_promptmatch_feature_cache(
                    image_paths,
                    model_label,
                    progress,
                    reuse_desc="Reusing cached PromptMatch image embeddings for {count} images",
                    encode_desc="Encoding PromptMatch image embeddings for {count} images...",
                    progress_label="PromptMatch",
                )
            except Exception as exc:
                return empty_result(str(exc), method)

            pos_prompt = (pos_prompt or "").strip() or SEARCH_PROMPT
            neg_prompt = (neg_prompt or "").strip()
            if pm_segment_mode:
                segments = [s.strip() for s in pos_prompt.split(",") if s.strip()]
                if not segments:
                    segments = [pos_prompt]
                neg_segments = [s.strip() for s in neg_prompt.split(",") if s.strip()] if neg_prompt else []
                seg_embs = [state["backend"].encode_text(seg) for seg in segments]
                neg_seg_embs = [state["backend"].encode_text(seg) for seg in neg_segments] if neg_segments else []
                seg_results = {}
                seg_embs_cpu = [e.detach().float().cpu() for e in seg_embs]
                neg_seg_embs_cpu = [e.detach().float().cpu() for e in neg_seg_embs]
                if feature_paths and image_features is not None and image_features.numel():
                    img_feat_cpu = image_features.detach().float().cpu()
                    per_seg_sims = [(img_feat_cpu @ e.T).squeeze(1).tolist() for e in seg_embs_cpu]
                    per_neg_seg_sims = [(img_feat_cpu @ e.T).squeeze(1).tolist() for e in neg_seg_embs_cpu]
                    for i, orig_path in enumerate(feature_paths):
                        fname = os.path.basename(orig_path)
                        seg_scores = {seg: float(per_seg_sims[j][i]) for j, seg in enumerate(segments)}
                        neg_seg_scores = {seg: float(per_neg_seg_sims[j][i]) for j, seg in enumerate(neg_segments)} if neg_segments else {}
                        pos_agg = sum(per_seg_sims[j][i] for j in range(len(segments)))
                        neg_agg = sum(per_neg_seg_sims[j][i] for j in range(len(neg_segments))) if neg_segments else None
                        seg_results[fname] = {
                            "pos": float(pos_agg),
                            "neg": float(neg_agg) if neg_agg is not None else None,
                            "path": orig_path,
                            "failed": False,
                            "segment_scores": seg_scores,
                            "neg_segment_scores": neg_seg_scores,
                        }
                for orig_path in failed_paths:
                    seg_results[os.path.basename(orig_path)] = {"pos": 0.0, "neg": None, "path": orig_path, "failed": True}
                state["scores"] = seg_results
                state["pm_segment_mode"] = True
            else:
                pos_emb = state["backend"].encode_text(pos_prompt)
                neg_emb = state["backend"].encode_text(neg_prompt) if neg_prompt else None
                state["scores"] = score_promptmatch_cached_features(
                    feature_paths,
                    image_features,
                    failed_paths,
                    pos_emb,
                    neg_emb,
                )
                state["pm_segment_mode"] = False
                state["pm_segment_sims"] = {}
            pos_min, pos_max, pos_mid, neg_min, neg_max, neg_mid, has_neg = promptmatch_slider_range(state["scores"])
            if preserve_promptmatch_thresholds or (previous_method != METHOD_PROMPTMATCH and has_recalled):
                next_main = clamp_threshold(requested_main, pos_min, pos_max)
                next_aux = clamp_threshold(requested_aux, neg_min, neg_max)
            else:
                next_main = pos_mid
                next_aux = neg_mid
            safe_pos_min, safe_pos_max = expand_slider_bounds(pos_min, pos_max, requested_main, next_main)
            safe_neg_min, safe_neg_max = expand_slider_bounds(neg_min, neg_max, requested_aux, next_aux)
            state["last_scored_method"] = METHOD_PROMPTMATCH
            state["last_scored_folder_key"] = folder_key
            state["last_promptmatch_model_label"] = model_label
            return render_scored_mode_result(
                METHOD_PROMPTMATCH,
                next_main,
                next_aux,
                gr.update(minimum=safe_pos_min, maximum=safe_pos_max, value=next_main, label=main_label),
                gr.update(minimum=safe_neg_min, maximum=safe_neg_max, value=next_aux, visible=True, interactive=has_neg, label=aux_label),
                percentile_slider_update(METHOD_PROMPTMATCH, state["scores"]),
                percentile_reset_button_update(METHOD_PROMPTMATCH, state["scores"]),
            )

        if method == METHOD_TAGMATCH:
            try:
                state["scores"] = score_tagmatch_folder(image_paths, tagmatch_tags, progress)
            except Exception as exc:
                return empty_result(str(exc), method)
            pos_vals = [
                v["pos"] for v in state["scores"].values()
                if not v.get("failed", False)
            ]
            if pos_vals:
                tm_lo = min(pos_vals)
                tm_hi = max(pos_vals)
                tm_mid = (tm_lo + tm_hi) / 2.0
            else:
                tm_lo, tm_hi, tm_mid = TAGMATCH_SLIDER_MIN, TAGMATCH_SLIDER_MAX, TAGMATCH_DEFAULT_THRESHOLD
            if previous_method != METHOD_TAGMATCH and has_recalled:
                next_main = clamp_threshold(requested_main, tm_lo, tm_hi)
            else:
                next_main = tm_mid
            safe_lo, safe_hi = expand_slider_bounds(tm_lo, tm_hi, requested_main, next_main)
            safe_lo = max(TAGMATCH_SLIDER_PREPROCESS_MIN, safe_lo)
            return render_scored_mode_result(
                METHOD_TAGMATCH,
                next_main,
                NEGATIVE_THRESHOLD,
                gr.update(minimum=safe_lo, maximum=safe_hi, value=next_main, label=main_label),
                gr.update(value=NEGATIVE_THRESHOLD, visible=False, minimum=TAGMATCH_SLIDER_PREPROCESS_MIN, maximum=TAGMATCH_SLIDER_MAX),
                percentile_slider_update(METHOD_TAGMATCH, state["scores"]),
                percentile_reset_button_update(METHOD_TAGMATCH, state["scores"]),
            )

        if state["ir_model"] is None:
            progress(0, desc=f"Loading ImageReward model from {describe_imagereward_source()}: ImageReward-v1.0")
        else:
            progress(0, desc="Using loaded ImageReward model from memory: ImageReward-v1.0")

        state["scores"] = score_imagereward(
            image_paths,
            ir_prompt,
            ir_negative_prompt,
            ir_penalty_weight,
            progress,
        )
        lo, hi, mid = imagereward_slider_range(state["scores"])
        if preserve_imagereward_threshold or (previous_method != METHOD_IMAGEREWARD and has_recalled):
            next_main = clamp_threshold(requested_main, lo, hi)
        else:
            next_main = mid
        safe_lo, safe_hi = expand_slider_bounds(lo, hi, requested_main, next_main)
        state["last_scored_method"] = METHOD_IMAGEREWARD
        state["last_scored_folder_key"] = folder_key
        return render_scored_mode_result(
            METHOD_IMAGEREWARD,
            next_main,
            NEGATIVE_THRESHOLD,
            gr.update(minimum=safe_lo, maximum=safe_hi, value=next_main, label=main_label),
            gr.update(value=NEGATIVE_THRESHOLD, visible=False),
            percentile_slider_update(METHOD_IMAGEREWARD, state["scores"]),
            percentile_reset_button_update(METHOD_IMAGEREWARD, state["scores"]),
        )

    def find_similar_images(folder, model_label, main_threshold, aux_threshold, progress=gr.Progress()):
        recalled_main, _, has_recalled = recalled_mode_thresholds(METHOD_SIMILARITY, main_threshold, aux_threshold)
        request_ctx, request_error = normalize_preview_search_request(
            folder,
            "Select, drop, paste, or upload a query image first, then find similar images.",
            "{preview_fname} is not part of the current folder, so similarity search cannot run.",
        )
        if request_error:
            return build_preview_search_result(request_error, status_with_current_view(request_error, main_threshold, aux_threshold))
        folder = request_ctx["folder"]
        image_paths = request_ctx["image_paths"]
        query_path = request_ctx["query_path"]
        query_label = request_ctx["query_label"]
        query_source = request_ctx["query_source"] or "gallery preview"
        query_in_folder = request_ctx["query_in_folder"]
        preserved_overrides = request_ctx["preserved_overrides"]

        release_inactive_gpu_models(METHOD_SIMILARITY)
        try:
            sync_promptmatch_proxy_cache(folder)
            ensure_promptmatch_backend_loaded(model_label, progress)
            _, feature_paths, image_features, failed_paths = ensure_promptmatch_feature_cache(
                image_paths,
                model_label,
                progress,
                reuse_desc="Reusing cached similarity embeddings for {count} images",
                encode_desc="Encoding similarity embeddings for {count} images...",
                progress_label="Similarity",
            )
            similarity_scores = score_similarity_cached_features(
                feature_paths,
                image_features,
                failed_paths,
                query_path,
                query_feature=None if query_in_folder else encode_single_promptmatch_image(query_path),
            )
        except Exception as exc:
            message = f"Similarity search failed: {exc}"
            return build_preview_search_result(
                message,
                status_with_current_view(message, main_threshold, aux_threshold),
            )

        state["method"] = METHOD_SIMILARITY
        set_scored_mode()
        state["source_dir"] = folder
        state["scores"] = similarity_scores
        state["overrides"] = preserved_overrides
        reset_selection_state()
        if query_in_folder:
            state["preview_fname"] = request_ctx["preview_fname"]
        clear_preview_search_context()
        state["similarity_query_fname"] = query_label
        state["similarity_query_source"] = query_source
        state["similarity_model_label"] = model_label

        pos_min, pos_max, _, neg_min, neg_max, _, _ = promptmatch_slider_range(state["scores"])
        _, default_top_n = similarity_topn_defaults(state["scores"])
        default_main = threshold_for_percentile(METHOD_SIMILARITY, state["scores"], default_top_n)
        next_main = clamp_threshold(recalled_main, pos_min, pos_max) if has_recalled else default_main
        safe_pos_min, safe_pos_max = expand_slider_bounds(pos_min, pos_max, next_main)
        safe_neg_min, safe_neg_max = expand_slider_bounds(neg_min, neg_max, NEGATIVE_THRESHOLD)
        status_text = f"Similarity search using {model_label} from {query_source} ({query_label})."
        score_outputs = render_scored_mode_result(
            METHOD_SIMILARITY,
            next_main,
            NEGATIVE_THRESHOLD,
            gr.update(
                minimum=safe_pos_min,
                maximum=safe_pos_max,
                value=next_main,
                label=threshold_labels(METHOD_SIMILARITY)[0],
            ),
            gr.update(
                minimum=safe_neg_min,
                maximum=safe_neg_max,
                value=NEGATIVE_THRESHOLD,
                visible=False,
                interactive=False,
                label=threshold_labels(METHOD_SIMILARITY)[1],
            ),
            percentile_slider_update(METHOD_SIMILARITY, state["scores"]),
            percentile_reset_button_update(METHOD_SIMILARITY, state["scores"]),
        )
        return build_preview_search_result(status_text, score_outputs)

    def find_same_person_images(folder, main_threshold, aux_threshold, progress=gr.Progress()):
        recalled_main, _, has_recalled = recalled_mode_thresholds(METHOD_SAMEPERSON, main_threshold, aux_threshold)
        request_ctx, request_error = normalize_preview_search_request(
            folder,
            "Select, drop, paste, or upload a query image first, then find the same person.",
            "{preview_fname} is not part of the current folder, so same-person search cannot run.",
        )
        if request_error:
            return build_preview_search_result(request_error, status_with_current_view(request_error, main_threshold, aux_threshold))
        folder = request_ctx["folder"]
        image_paths = request_ctx["image_paths"]
        query_path = request_ctx["query_path"]
        query_label = request_ctx["query_label"]
        query_source = request_ctx["query_source"] or "gallery preview"
        query_in_folder = request_ctx["query_in_folder"]
        preserved_overrides = request_ctx["preserved_overrides"]

        release_inactive_gpu_models(METHOD_SAMEPERSON)
        try:
            sync_promptmatch_proxy_cache(folder)
            _, feature_paths, face_embeddings, failures = ensure_face_feature_cache(image_paths, progress)
            sameperson_scores = score_sameperson_cached_features(
                feature_paths,
                face_embeddings,
                failures,
                query_path,
                query_embedding=None if query_in_folder else encode_single_face_embedding(query_path, progress),
            )
        except Exception as exc:
            message = f"Same-person search failed: {exc}"
            return build_preview_search_result(
                message,
                status_with_current_view(message, main_threshold, aux_threshold),
            )

        state["method"] = METHOD_SAMEPERSON
        set_scored_mode()
        state["source_dir"] = folder
        state["scores"] = sameperson_scores
        state["overrides"] = preserved_overrides
        reset_selection_state()
        if query_in_folder:
            state["preview_fname"] = request_ctx["preview_fname"]
        clear_preview_search_context()
        state["sameperson_query_fname"] = query_label
        state["sameperson_query_source"] = query_source
        state["sameperson_model_label"] = FACE_MODEL_LABEL

        pos_min, pos_max, _, neg_min, neg_max, _, _ = promptmatch_slider_range(state["scores"])
        _, default_top_n = similarity_topn_defaults(state["scores"])
        default_main = threshold_for_percentile(METHOD_SAMEPERSON, state["scores"], default_top_n)
        next_main = clamp_threshold(recalled_main, pos_min, pos_max) if has_recalled else default_main
        safe_pos_min, safe_pos_max = expand_slider_bounds(pos_min, pos_max, next_main)
        safe_neg_min, safe_neg_max = expand_slider_bounds(neg_min, neg_max, NEGATIVE_THRESHOLD)
        status_text = f"Same-person search using {FACE_MODEL_LABEL} from {query_source} ({query_label})."
        score_outputs = render_scored_mode_result(
            METHOD_SAMEPERSON,
            next_main,
            NEGATIVE_THRESHOLD,
            gr.update(
                minimum=safe_pos_min,
                maximum=safe_pos_max,
                value=next_main,
                label=threshold_labels(METHOD_SAMEPERSON)[0],
            ),
            gr.update(
                minimum=safe_neg_min,
                maximum=safe_neg_max,
                value=NEGATIVE_THRESHOLD,
                visible=False,
                interactive=False,
                label=threshold_labels(METHOD_SAMEPERSON)[1],
            ),
            percentile_slider_update(METHOD_SAMEPERSON, state["scores"]),
            percentile_reset_button_update(METHOD_SAMEPERSON, state["scores"]),
        )
        return build_preview_search_result(status_text, score_outputs)

    def handle_shortcut_action(action, method, folder, model_label, pos_prompt, neg_prompt, pm_segment_mode, ir_prompt, ir_negative_prompt, ir_penalty_weight, llm_model_label, llm_prompt, llm_backend_id, llm_shortlist_size, tagmatch_tags, main_threshold, aux_threshold, keep_pm_thresholds, keep_ir_thresholds, progress=gr.Progress()):
        action = (action or "").strip()
        if not action.startswith("run:"):
            return empty_result("Shortcut action ignored.", method)

        prompt_id = action.split(":", 2)[1] if ":" in action else ""
        if prompt_id in ("hy-pos", "hy-neg"):
            method = METHOD_PROMPTMATCH
        elif prompt_id in ("hy-ir-pos", "hy-ir-neg"):
            method = METHOD_IMAGEREWARD

        return score_folder(
            method,
            folder,
            model_label,
            pos_prompt,
            neg_prompt,
            pm_segment_mode,
            ir_prompt,
            ir_negative_prompt,
            ir_penalty_weight,
            llm_model_label,
            llm_prompt,
            llm_backend_id,
            llm_shortlist_size,
            tagmatch_tags,
            main_threshold,
            aux_threshold,
            keep_pm_thresholds,
            keep_ir_thresholds,
            progress=progress,
        )

    def update_split(main_threshold, aux_threshold):
        return render_view_with_controls(main_threshold, aux_threshold)

    def update_histogram_only(main_threshold, aux_threshold):
        return render_histogram(state["method"], state["scores"], main_threshold, aux_threshold)

    def update_prompt_generator(generator_name, detail_level, current_generated_prompt):
        return select_cached_generated_prompt(generator_name, detail_level, current_generated_prompt)

    def update_generated_prompt_detail(generator_name, detail_level, current_generated_prompt):
        return select_cached_generated_prompt(generator_name, detail_level, current_generated_prompt)

    def external_query_prompt_status():
        query_ctx = active_query_image_context()
        if query_ctx["source_kind"] == "external":
            return (
                f"External query image ready: {query_ctx['label']}. "
                "Click Prompt from image, Find similar images, or Find same person."
            )
        if query_ctx["source_kind"] == "gallery" and query_ctx["label"]:
            return f"Using gallery preview {query_ctx['label']} for preview actions."
        return "Preview, drop, paste, or upload an image, then generate a prompt from it."

    def set_external_query_image(image_path):
        if not image_path:
            clear_external_query_state(delete_file=True)
            return (
                active_query_image_widget_update(),
                clear_external_query_button_update(),
                gr.update(value=external_query_prompt_status()),
            )
        try:
            save_external_query_image(image_path)
        except Exception as exc:
            return (
                gr.update(value=None),
                clear_external_query_button_update(),
                gr.update(value=f"External query image failed: {exc}"),
            )
        return (
            active_query_image_widget_update(),
            clear_external_query_button_update(),
            gr.update(value=external_query_prompt_status()),
        )

    def set_external_query_from_bridge(payload):
        payload = (payload or "").strip()
        if not payload:
            return (
                active_query_image_widget_update(),
                clear_external_query_button_update(),
                gr.update(value=external_query_prompt_status()),
                gr.update(value=""),
            )
        try:
            parsed = json.loads(payload)
            data_url = str(parsed.get("data_url") or "")
            label = str(parsed.get("label") or "clipboard-image.png")
            if not data_url.startswith("data:image/"):
                raise ValueError("Unsupported clipboard image payload.")
            _, encoded = data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded, validate=True)
            save_external_query_image_bytes(image_bytes, label)
        except Exception as exc:
            return (
                gr.update(value=None),
                clear_external_query_button_update(),
                gr.update(value=f"External query image failed: {exc}"),
                gr.update(value=""),
            )
        return (
            active_query_image_widget_update(),
            clear_external_query_button_update(),
            gr.update(value=external_query_prompt_status()),
            gr.update(value=""),
        )

    def clear_external_query_image():
        clear_external_query_state(delete_file=True)
        return (
            active_query_image_widget_update(),
            clear_external_query_button_update(),
            gr.update(value=external_query_prompt_status()),
        )

    def generate_prompt_from_preview(generator_name, current_generated_prompt, detail_level, progress=gr.Progress()):
        detail_level, detail_label, _ = prompt_generator_detail_config(generator_name, detail_level)
        state["prompt_generator"] = generator_name
        state["generated_prompt_detail"] = detail_level
        query_ctx = active_query_image_context()
        image_path = query_ctx.get("path")
        preview_fname = query_ctx.get("label")
        query_key = query_ctx.get("cache_key")
        if not image_path or not os.path.isfile(image_path):
            state["generated_prompt_status"] = "Select or drop a query image first, then generate a prompt."
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=current_generated_prompt),
                gr.update(),
            )

        variants = generated_prompt_variants_for(query_key, generator_name, create=True)
        if all(level in variants and variants[level] for level in (1, 2, 3)):
            cached_prompt = variants.get(detail_level)
            state["generated_prompt"] = cached_prompt
            state["generated_prompt_source"] = preview_fname
            state["generated_prompt_backend"] = generator_name
            state["generated_prompt_status"] = (
                f"Reused cached prompt set for {preview_fname} via {generator_name}. Showing {detail_label.lower()} detail."
                f"{prompt_backend_warning_text(generator_name)}"
            )
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=cached_prompt),
                gr.update(value=detail_level),
            )

        backend_cached = state["prompt_backend_cache"].get(generator_name)
        if not backend_cached:
            progress(0, desc=f"Loading {generator_name} from {describe_prompt_generator_source(generator_name)}")
        else:
            progress(0, desc=f"Using loaded {generator_name} backend from memory")

        try:
            with Image.open(image_path) as src_img:
                image = src_img.convert("RGB")

            for idx, level in enumerate((1, 2, 3), start=1):
                _, loop_label, _ = prompt_generator_detail_config(generator_name, level)
                progress(
                    0.2 + (0.7 * ((idx - 1) / 3.0)),
                    desc=f"Generating {loop_label.lower()} prompt from {preview_fname} via {generator_name}",
                )
                variants[level] = generate_prompt_variant(generator_name, image, level)
        except Exception as exc:
            state["generated_prompt_status"] = f"Prompt generation failed: {exc}"
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=current_generated_prompt),
                gr.update(),
            )
        finally:
            if device == "cuda" and prompt_generator_supports_torch_cleanup(generator_name):
                torch.cuda.empty_cache()

        prompt_text = variants.get(detail_level, "")
        if not prompt_text:
            state["generated_prompt_status"] = f"No usable prompt text was produced for {preview_fname}."
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=current_generated_prompt),
                gr.update(),
            )

        state["generated_prompt"] = prompt_text
        state["generated_prompt_source"] = preview_fname
        state["generated_prompt_backend"] = generator_name
        ready_count = sum(1 for level in (1, 2, 3) if variants.get(level))
        state["generated_prompt_status"] = (
            f"Generated {ready_count} prompt detail levels for {preview_fname} via {generator_name}. "
            f"Showing {detail_label.lower()} detail."
            f"{prompt_backend_warning_text(generator_name)}"
        )
        return (
            gr.update(value=state["generated_prompt_status"]),
            gr.update(value=prompt_text),
            gr.update(value=detail_level),
        )

    def insert_generated_prompt(method, prompt_text):
        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            state["generated_prompt_status"] = "Generated prompt is empty. Edit or generate a prompt first."
            return gr.update(value=state["generated_prompt_status"]), gr.update(), gr.update(), gr.update()

        state["generated_prompt"] = prompt_text
        if method == METHOD_PROMPTMATCH:
            target_label = "PromptMatch positive prompt"
        elif method == METHOD_LLMSEARCH:
            target_label = "LM search prompt"
        else:
            target_label = "ImageReward positive prompt"
        state["generated_prompt_status"] = f"Inserted generated prompt into {target_label}."
        if method == METHOD_PROMPTMATCH:
            return gr.update(value=state["generated_prompt_status"]), gr.update(value=prompt_text), gr.update(), gr.update()
        if method == METHOD_LLMSEARCH:
            return gr.update(value=state["generated_prompt_status"]), gr.update(), gr.update(), gr.update(value=prompt_text)
        return gr.update(value=state["generated_prompt_status"]), gr.update(), gr.update(value=prompt_text), gr.update()

    def update_proxy_display(use_proxy_display, main_threshold, aux_threshold):
        state["use_proxy_display"] = bool(use_proxy_display)
        return render_view_with_controls(main_threshold, aux_threshold)

    def update_imagereward_penalty_weight(penalty_weight, main_threshold, aux_threshold):
        recomputed = recompute_imagereward_scores(penalty_weight)
        if not recomputed:
            return (
                *render_view_with_controls(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
            )

        lo, hi, _ = imagereward_slider_range(state["scores"])
        clamped = clamp_threshold(main_threshold, lo, hi)
        safe_lo, safe_hi = expand_slider_bounds(lo, hi, main_threshold, clamped)
        main_label, _, _, _ = threshold_labels(METHOD_IMAGEREWARD)
        return (
            *render_view_with_controls(clamped, aux_threshold),
            gr.update(
                minimum=safe_lo,
                maximum=safe_hi,
                value=clamped,
                label=main_label,
            ),
            gr.update(value=NEGATIVE_THRESHOLD, visible=False),
        )

    def handle_thumb_action(action, main_threshold, aux_threshold):
        # Custom JS reports preview clicks, shift-click bulk marking, and drag-drop moves.
        def thumb_noop(query_source_update=None):
            outputs = [gr.skip()] * 19
            if query_source_update is not None:
                outputs[8] = query_source_update
            return tuple(outputs)

        with_slider_skips = lambda view: (*view, gr.skip(), gr.skip())
        if not action:
            return thumb_noop()
        if str(action).startswith("previewfname:"):
            try:
                _, fname, _ = str(action).split(":", 2)
            except Exception:
                return thumb_noop()
            visible_names = {os.path.basename(path) for path, _ in state.get("browse_items", [])}
            if fname and (fname in state.get("scores", {}) or fname in visible_names):
                state["preview_fname"] = fname
            return thumb_noop(active_query_image_widget_update())
        if str(action).startswith("dialogactionjson:"):
            try:
                payload = json.loads(str(action)[17:])
                action_id = str(payload.get("action", "") or "")
                fname = str(payload.get("fname", "") or "")
            except Exception:
                return thumb_noop()
            visible_names = {os.path.basename(path) for path, _ in state.get("browse_items", [])}
            if fname and (fname in state.get("scores", {}) or fname in visible_names):
                state["preview_fname"] = fname
            if action_id == "hy-move-right":
                return with_slider_skips(move_right(main_threshold, aux_threshold, preview_override=fname))
            if action_id == "hy-move-left":
                return with_slider_skips(move_left(main_threshold, aux_threshold, preview_override=fname))
            if action_id == "hy-fit-threshold":
                return fit_threshold_to_targets(main_threshold, aux_threshold, preview_override=fname)
            return thumb_noop(active_query_image_widget_update())
        if str(action).startswith("dropjson:"):
            try:
                payload = json.loads(str(action)[9:])
                side = payload["source_side"]
                index = int(payload["source_index"])
                target_side = payload["target_side"]
                drop_fnames = [str(name) for name in payload.get("fnames", []) if str(name)]
            except Exception:
                return thumb_noop()
            left_items, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)
            items = left_items if side == "left" else right_items
            if 0 <= index < len(items) and target_side in ("left", "right") and target_side != side:
                fallback_fname = os.path.basename(items[index][0])
                move_fnames = drop_fnames or [fallback_fname]
                available_names = {os.path.basename(path) for path, _ in items}
                move_fnames = [fname for fname in move_fnames if fname in available_names]
                if not move_fnames:
                    move_fnames = [fallback_fname]
                left_name, right_name, _, _ = method_labels(state["method"])
                target_label = left_name if target_side == "left" else right_name
                for fname in move_fnames:
                    state["overrides"][fname] = target_label
                state["left_marked"] = [name for name in state["left_marked"] if name not in move_fnames]
                state["right_marked"] = [name for name in state["right_marked"] if name not in move_fnames]
                state["preview_fname"] = move_fnames[0]
            return with_slider_skips(render_view_with_controls(main_threshold, aux_threshold))
        parts = str(action).split(":")
        verb = parts[0] if parts else ""
        if is_browse_mode():
            left_items = list(state.get("browse_items", []))
            right_items = []
        else:
            left_items, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)

        try:
            _, side, raw_index, _ = parts
            index = int(raw_index)
        except Exception:
            return thumb_noop()
        items = left_items if side == "left" else right_items
        if 0 <= index < len(items):
            fname = os.path.basename(items[index][0])
            if verb == "preview":
                state["preview_fname"] = fname
                return thumb_noop(active_query_image_widget_update())
            if is_browse_mode():
                state["preview_fname"] = fname
                return thumb_noop(active_query_image_widget_update())
            else:
                marked_key = "left_marked" if side == "left" else "right_marked"
                if fname in state[marked_key]:
                    state[marked_key] = [name for name in state[marked_key] if name != fname]
                else:
                    state[marked_key].append(fname)
        return with_slider_skips(render_view_with_controls(main_threshold, aux_threshold))

    def handle_hist_width(width_value, main_threshold, aux_threshold):
        try:
            next_width = max(220, int(float(width_value)))
        except Exception:
            return gr.skip()
        if abs(next_width - int(state.get("hist_width", 300))) < 2:
            return gr.skip()
        state["hist_width"] = next_width
        return render_histogram(state["method"], state["scores"], main_threshold, aux_threshold)

    def move_right(main_threshold, aux_threshold, preview_override=None):
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        left_name, right_name, _, _ = method_labels(state["method"])
        left_targets, _ = active_targets(main_threshold, aux_threshold, preview_override=preview_override)
        targets = left_targets or list(state.get("left_marked", []))
        for fname in targets:
            state["overrides"][fname] = right_name
        state["left_marked"] = []
        state["right_marked"] = []
        return render_view_with_controls(main_threshold, aux_threshold)

    def move_left(main_threshold, aux_threshold, preview_override=None):
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        left_name, right_name, _, _ = method_labels(state["method"])
        _, right_targets = active_targets(main_threshold, aux_threshold, preview_override=preview_override)
        targets = right_targets or list(state.get("right_marked", []))
        for fname in targets:
            state["overrides"][fname] = left_name
        state["left_marked"] = []
        state["right_marked"] = []
        return render_view_with_controls(main_threshold, aux_threshold)

    def pin_selected(main_threshold, aux_threshold, preview_override=None):
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        left_name, right_name, _, _ = method_labels(state["method"])
        left_targets, right_targets = active_targets(main_threshold, aux_threshold, preview_override=preview_override)
        for fname in left_targets:
            state["overrides"][fname] = left_name
        for fname in right_targets:
            state["overrides"][fname] = right_name
        state["left_marked"] = []
        state["right_marked"] = []
        return render_view_with_controls(main_threshold, aux_threshold)

    def clear_status(main_threshold, aux_threshold):
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        for fname in set(state["left_marked"] + state["right_marked"]):
            state["overrides"].pop(fname, None)
        state["left_marked"] = []
        state["right_marked"] = []
        return render_view_with_controls(main_threshold, aux_threshold)

    def clear_all_status(main_threshold, aux_threshold):
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        state["overrides"].clear()
        state["left_marked"] = []
        state["right_marked"] = []
        return render_view_with_controls(main_threshold, aux_threshold)

    def fit_threshold_to_targets(main_threshold, aux_threshold, preview_override=None):
        if is_browse_mode():
            return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())
        left_targets, right_targets = active_targets(main_threshold, aux_threshold, preview_override=preview_override)
        targets = left_targets or right_targets
        if not targets or (left_targets and right_targets) or not state["scores"]:
            return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())

        for fname in targets:
            state["overrides"].pop(fname, None)

        new_main = float(main_threshold)
        new_aux = float(aux_threshold)

        if state["method"] == METHOD_IMAGEREWARD:
            lo, hi, _ = imagereward_slider_range(state["scores"])
            valid_scores = [
                state["scores"][fname]["score"]
                for fname in targets
                if fname in state["scores"] and state["scores"][fname]["score"] > -1000
            ]
            if valid_scores:
                if right_targets:
                    new_main = min(new_main, slider_step_floor(min(valid_scores)))
                else:
                    new_main = max(new_main, slider_step_ceil_exclusive(max(valid_scores)))
                new_main = round(max(lo, min(hi, new_main)), 3)
        elif state["method"] == METHOD_LLMSEARCH:
            lo, hi, _ = llmsearch_slider_range(state["scores"], state.get("llmsearch_backend"))
            valid_items = [
                state["scores"][fname]
                for fname in targets
                if fname in state["scores"] and not state["scores"][fname].get("failed", False)
            ]
            if valid_items:
                if right_targets:
                    new_main = min(new_main, slider_step_floor(min(item["pos"] for item in valid_items)))
                else:
                    new_main = max(new_main, slider_step_ceil_exclusive(max(item["pos"] for item in valid_items)))
                new_main = round(max(lo, min(hi, new_main)), 3)
        else:
            pos_lo, pos_hi, _, neg_lo, neg_hi, _, has_neg = promptmatch_slider_range(state["scores"])
            valid_items = [
                state["scores"][fname]
                for fname in targets
                if fname in state["scores"] and not state["scores"][fname].get("failed", False)
            ]
            if valid_items:
                if right_targets:
                    min_pos = min(item["pos"] for item in valid_items)
                    new_main = min(new_main, slider_step_floor(min_pos))
                    if has_neg:
                        neg_scores = [item["neg"] for item in valid_items if item.get("neg") is not None]
                        if neg_scores:
                            new_aux = max(new_aux, slider_step_ceil_exclusive(max(neg_scores)))
                else:
                    main_candidate = max(new_main, slider_step_ceil_exclusive(max(item["pos"] for item in valid_items)))
                    main_delta = abs(main_candidate - float(main_threshold))

                    aux_candidate = None
                    aux_delta = None
                    if has_neg and all(item.get("neg") is not None for item in valid_items):
                        aux_candidate = min(new_aux, slider_step_floor(min(item["neg"] for item in valid_items)))
                        aux_delta = abs(aux_candidate - float(aux_threshold))

                    if aux_candidate is not None and aux_delta is not None and aux_delta < main_delta:
                        new_aux = aux_candidate
                    else:
                        new_main = main_candidate

                new_main = round(max(pos_lo, min(pos_hi, new_main)), 3)
                new_aux = round(max(neg_lo, min(neg_hi, new_aux)), 3)

        state["left_marked"] = []
        state["right_marked"] = []
        return (
            *render_view_with_controls(new_main, new_aux),
            gr.update(value=new_main),
            gr.update(value=new_aux),
        )

    def set_from_percentile(percentile, main_threshold, aux_threshold):
        new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
        return (*render_view_with_controls(new_threshold, aux_threshold), gr.update(value=new_threshold))

    def update_histogram_from_percentile(percentile, aux_threshold):
        new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
        return render_histogram(state["method"], state["scores"], new_threshold, aux_threshold)

    def reset_main_threshold_to_middle(main_threshold, aux_threshold):
        new_main, _, _ = middle_threshold_values(state["method"])
        return (
            *render_view_with_controls(new_main, aux_threshold),
            gr.update(value=new_main),
            gr.update(),
        )

    def reset_aux_threshold_to_middle(main_threshold, aux_threshold):
        if state["method"] != METHOD_PROMPTMATCH:
            return (
                *render_view_with_controls(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
            )
        _, new_aux, _ = middle_threshold_values(METHOD_PROMPTMATCH)
        return (
            *render_view_with_controls(main_threshold, new_aux),
            gr.update(),
            gr.update(value=new_aux),
        )

    def reset_percentile_to_middle(main_threshold, aux_threshold):
        if uses_similarity_topn(state["method"]):
            _, percentile = similarity_topn_defaults(state["scores"])
        else:
            percentile = 50
        if state["scores"]:
            new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
        else:
            new_threshold = float(main_threshold)
        return (
            *render_view_with_controls(new_threshold, aux_threshold),
            gr.update(value=new_threshold),
            gr.update(value=percentile),
        )

    def update_zoom(zoom_value, main_threshold, aux_threshold):
        # Invert the slider so dragging right makes thumbnails larger by reducing columns.
        try:
            slider_value = max(2, min(10, int(zoom_value)))
            state["zoom_columns"] = 12 - slider_value
        except Exception:
            state["zoom_columns"] = 5
        return render_view_with_controls(main_threshold, aux_threshold)

    def on_hist_click(sel: gr.SelectData, main_threshold, aux_threshold):
        geom = state.get("hist_geom")
        if not geom:
            return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())
        try:
            cx, cy = sel.index
            if uses_pos_similarity_scores(geom["method"]):
                W = geom["W"]
                PAD_L = geom["PAD_L"]
                PAD_R = geom["PAD_R"]
                PAD_TOP = geom["PAD_TOP"]
                CH = geom["CH"]
                GAP = geom["GAP"]
                cW = W - PAD_L - PAD_R
                y0pos = PAD_TOP
                y0neg = PAD_TOP + CH + GAP
                if y0pos <= cy <= y0pos + CH:
                    lo, hi = geom["pos_lo"], geom["pos_hi"]
                    val = lo + ((cx - PAD_L) / cW) * (hi - lo)
                    main_threshold = round(max(lo, min(hi, val)), 3)
                elif geom["has_neg"] and y0neg <= cy <= y0neg + CH:
                    lo, hi = geom["neg_lo"], geom["neg_hi"]
                    val = lo + ((cx - PAD_L) / cW) * (hi - lo)
                    aux_threshold = round(max(lo, min(hi, val)), 3)
                else:
                    return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())
            else:
                PAD_L = geom["PAD_L"]
                PAD_TOP = geom["PAD_TOP"]
                PAD_BOT = geom["PAD_BOT"]
                H = geom["H"]
                cW = geom["cW"]
                lo, hi = geom["lo"], geom["hi"]
                if PAD_TOP <= cy <= PAD_TOP + H - PAD_BOT:
                    val = lo + ((cx - PAD_L) / cW) * (hi - lo)
                    main_threshold = round(max(lo, min(hi, val)), 3)
                else:
                    return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())
        except Exception:
            return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())

        return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(value=main_threshold), gr.update(value=aux_threshold))

    def export_files(main_threshold, aux_threshold, export_left_enabled, export_right_enabled, export_move_enabled, export_left_name, export_right_name):
        # Export is a lossless copy, not a rewrite or recompression of the originals.
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        left_items, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)
        left_name, right_name, left_dirname, right_dirname = method_labels(state["method"])
        base = state["source_dir"]
        targets = []
        if export_left_enabled:
            targets.append((left_name, sanitize_export_name(export_left_name) or left_dirname, left_items))
        if export_right_enabled:
            targets.append((right_name, sanitize_export_name(export_right_name) or right_dirname, right_items))
        if not targets:
            return render_view_with_controls(main_threshold, aux_threshold)
        target_names = [folder_name for _, folder_name, _ in targets]
        if len(set(target_names)) != len(target_names):
            return render_view_with_controls(main_threshold, aux_threshold)

        lines = []
        moved_names = []
        for bucket_label, folder_name, items in targets:
            bucket_dir = os.path.join(base, folder_name)
            os.makedirs(bucket_dir, exist_ok=True)
            for path, _ in items:
                original_name = os.path.basename(path)
                dest_path = export_destination(bucket_dir, original_name)
                if export_move_enabled:
                    shutil.copy2(path, dest_path)
                    os.remove(path)
                    moved_names.append(original_name)
                else:
                    shutil.copy2(path, dest_path)
            verb = "Moved" if export_move_enabled else "Losslessly copied"
            lines.append(f"{verb} {len(items)} {bucket_label.lower()} files to {bucket_dir}")
        disabled = []
        if not export_left_enabled:
            disabled.append(left_name.lower())
        if not export_right_enabled:
            disabled.append(right_name.lower())
        if disabled:
            lines.append(f"Skipped bucket(s): {', '.join(disabled)}")
        if moved_names:
            moved_set = set(moved_names)
            for fname in moved_set:
                state["scores"].pop(fname, None)
                state["overrides"].pop(fname, None)
            state["left_marked"] = [name for name in state.get("left_marked", []) if name not in moved_set]
            state["right_marked"] = [name for name in state.get("right_marked", []) if name not in moved_set]
            if state.get("preview_fname") in moved_set:
                state["preview_fname"] = None
            if (
                state.get("similarity_query_fname") in moved_set
                or state.get("sameperson_query_fname") in moved_set
            ):
                clear_preview_search_context()
        return render_view_with_controls(main_threshold, aux_threshold)

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
