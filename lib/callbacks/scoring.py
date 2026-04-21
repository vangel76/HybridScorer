import os

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from ..config import (
    METHOD_PROMPTMATCH, METHOD_IMAGEREWARD, METHOD_SIMILARITY, METHOD_SAMEPERSON,
    METHOD_LLMSEARCH, METHOD_TAGMATCH,
    TAGMATCH_WD_MIN_CACHE_PROB,
    PROMPT_GENERATOR_FLORENCE, PROMPT_GENERATOR_JOYCAPTION,
    PROMPT_GENERATOR_JOYCAPTION_GGUF, PROMPT_GENERATOR_HUIHUI_GEMMA4,
    LLMSEARCH_SCORING_MODE_NUMERIC_V1,
    LLMSEARCH_DEFAULT_PROMPT, LLMSEARCH_SHORTLIST_DEFAULT,
    LLMSEARCH_SHORTLIST_MIN, LLMSEARCH_SHORTLIST_MAX,
    LLMSEARCH_JOYCAPTION_HF_BATCH_SIZE,
    LLMSEARCH_JOYCAPTION_TEMPERATURE, LLMSEARCH_JOYCAPTION_TOP_P,
    LLMSEARCH_JOYCAPTION_TOP_K, LLMSEARCH_JOYCAPTION_MAX_NEW_TOKENS,
    LLMSEARCH_HUIHUI_GEMMA4_MAX_NEW_TOKENS, LLMSEARCH_HUIHUI_GEMMA4_TEMPERATURE,
    LLMSEARCH_HUIHUI_GEMMA4_TOP_P, LLMSEARCH_HUIHUI_GEMMA4_TOP_K,
    IR_PROMPT, PROMPTMATCH_PROXY_PROGRESS_SHARE, NEGATIVE_THRESHOLD,
    TAGMATCH_SLIDER_MIN, TAGMATCH_SLIDER_MAX, TAGMATCH_DEFAULT_THRESHOLD,
    TAGMATCH_SLIDER_PREPROCESS_MIN, FACE_MODEL_LABEL, SEARCH_PROMPT,
    DEFAULT_LLMSEARCH_BACKEND, MODEL_LABELS, IMAGEREWARD_THRESHOLD,
)
from ..utils import (
    get_image_paths_signature, get_auto_batch_size, normalize_folder_identity,
    normalize_prompt_text, prepare_promptmatch_proxies,
    iter_imagereward_scores, get_imagereward_penalty_offset, compute_imagereward_final_score,
    describe_imagereward_source, describe_llmsearch_backend_source,
    llmsearch_backend_choices,
)
from ..helpers import (
    label_for_backend, scan_image_paths, method_labels, threshold_labels,
    promptmatch_slider_range, imagereward_slider_range, llmsearch_slider_range,
    llmsearch_uses_numeric_scores, clamp_threshold, expand_slider_bounds,
    normalize_threshold_inputs, percentile_slider_update, percentile_reset_button_update,
    similarity_topn_defaults, threshold_for_percentile, uses_pos_similarity_scores,
    promptmatch_model_dropdown_choices,
    normalize_generated_prompt, extract_joycaption_caption, extract_huihui_gemma4_caption,
    joycaption_max_new_tokens,
    llmsearch_joycaption_system_prompt, build_llmsearch_joycaption_user_prompt,
    build_llmsearch_huihui_gemma4_user_prompt,
    normalize_llmsearch_candidate_text, extract_llmsearch_numeric_score,
)
from ..scoring import score_promptmatch_cached_features
from ..state_helpers import (
    can_reuse_proxy_map, begin_scored_run, set_scored_mode, set_browse_folder_state,
    recalled_mode_thresholds, preserve_overrides_for_folder_key,
    active_query_image_context, clear_preview_search_context, reset_selection_state,
    sync_promptmatch_proxy_cache,
)
from .. import loaders as _lo
from .. import view as _vw
from . import prompts as _pr


def score_tagmatch_folder(state, device, image_paths, query_tags_str, progress):
    backend = _lo.ensure_tagmatch_model(state, device)
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
                        arr = _lo.tagmatch_prepare_image(src)
                    batch_tensors.append(arr)
                    batch_valid_paths.append(p)
                except Exception:
                    tag_vectors[p] = {}
            if batch_tensors:
                try:
                    batch_np = np.stack(batch_tensors, axis=0)
                    raw_out = session.run(None, {input_name: batch_np})[0]
                    for i, p in enumerate(batch_valid_paths):
                        row = raw_out[i]
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

    state["tagmatch_last_query_tags_str"] = query_tags_str or ""
    query_tags = [t.strip().lower() for t in (query_tags_str or "").split(",") if t.strip()]
    tag_set = set(tags)
    missing = [t for t in query_tags if t not in tag_set]
    if missing:
        print(f"[TagMatch] WARNING: these query tags are not in the model vocabulary and will score 0: {missing}")
    results = {}
    for p in image_paths:
        tv = tag_vectors.get(p, {})
        score = min(100.0, sum(tv.get(t, 0.0) for t in query_tags) * 100.0)
        results[os.path.basename(p)] = {
            "pos": score,
            "neg": None,
            "path": p,
            "failed": False,
        }
    return results


class VisionLLMRerankBackend:
    def __init__(self, state, device, backend_id):
        self._state = state
        self._device = device
        self.backend_id = backend_id

    def describe_source(self):
        return describe_llmsearch_backend_source(self.backend_id)

    def is_available(self):
        return True

    def load(self, progress):
        if self.backend_id == PROMPT_GENERATOR_FLORENCE:
            progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
            _lo.ensure_florence_model(self._state, self._device)
        elif self.backend_id == PROMPT_GENERATOR_JOYCAPTION:
            progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
            _lo.ensure_joycaption_model(self._state, self._device)
        elif self.backend_id == PROMPT_GENERATOR_JOYCAPTION_GGUF:
            progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
            _lo.ensure_joycaption_gguf_model(self._state, self._device)
        elif self.backend_id == PROMPT_GENERATOR_HUIHUI_GEMMA4:
            progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
            _lo.ensure_huihui_gemma4_model(self._state, self._device)
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
            return _pr.run_florence_prompt_variant(self._state, self._device, image, "<MORE_DETAILED_CAPTION>")
        if self.backend_id == PROMPT_GENERATOR_HUIHUI_GEMMA4:
            return _pr.run_huihui_gemma4_prompt_variant(
                self._state, self._device,
                image,
                build_llmsearch_huihui_gemma4_user_prompt(query_text),
                2,
                system_prompt=llmsearch_joycaption_system_prompt(),
                normalizer=lambda text: normalize_generated_prompt(extract_huihui_gemma4_caption(text)),
                max_new_tokens_override=joycaption_max_new_tokens(2),
            )

        user_prompt = build_llmsearch_joycaption_user_prompt(query_text)
        return _pr.run_joycaption_prompt_variant(
            self._state, self._device,
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
            return (
                llmsearch_similarity(self._state, self._state["backend"].encode_text((query_text or "").strip()), caption_text),
                caption_text,
            )

        if self.backend_id == PROMPT_GENERATOR_HUIHUI_GEMMA4:
            raw_score_text = _pr.run_huihui_gemma4_prompt_variant(
                self._state, self._device,
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
            raw_score_text = _pr.run_joycaption_prompt_variant(
                self._state, self._device,
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
        model, processor = _lo.ensure_joycaption_model(self._state, self._device)
        user_prompt = build_llmsearch_joycaption_user_prompt(query_text)
        conversation = [
            {"role": "system", "content": llmsearch_joycaption_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]
        convo_string = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        n = len(images)
        inputs = processor(text=[convo_string] * n, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
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


def llmsearch_similarity(state, query_embedding, text):
    text = normalize_prompt_text(text)
    if not text:
        return -1.0
    text_embedding = state["backend"].encode_text(text)
    score = float((query_embedding @ text_embedding.T).squeeze().item())
    return round(score, 6)


def score_llmsearch_candidates(state, device, candidate_paths, query_text, backend_id, image_signature, progress):
    backend = VisionLLMRerankBackend(state, device, backend_id)
    backend.load(progress)
    cache_key = (
        str(backend_id),
        LLMSEARCH_SCORING_MODE_NUMERIC_V1,
        str(image_signature),
        normalize_prompt_text(query_text or ""),
    )
    caption_cache = state["llmsearch_cached_captions"].setdefault(cache_key, {})
    total = len(candidate_paths)
    results = {}
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
                    pass
            if loaded:
                try:
                    batch_scored = backend.score_candidates_batch([img for _, img in loaded], query_text)
                    for (p, _), (sv, ct) in zip(loaded, batch_scored):
                        caption_cache[p] = {"score": float(sv), "text": ct}
                except Exception:
                    pass
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
                score_value = llmsearch_similarity(state, state["backend"].encode_text((query_text or "").strip()), caption_text)
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


def score_imagereward(state, device, folder_paths, positive_prompt, negative_prompt, penalty_weight, progress):
    model = _lo.ensure_imagereward_model(state)
    positive_prompt = (positive_prompt or "").strip() or IR_PROMPT
    negative_prompt = (negative_prompt or "").strip()
    penalty_weight = float(penalty_weight)
    state["ir_penalty_weight"] = penalty_weight
    image_signature = get_image_paths_signature(folder_paths)
    proxy_map = {}
    cache_dir = state.get("proxy_cache_dir")
    scoring_paths = list(folder_paths)

    if cache_dir and can_reuse_proxy_map(state, folder_paths, image_signature):
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


def recompute_imagereward_scores(state, penalty_weight):
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


def load_folder_for_browse(state, folder, main_threshold, aux_threshold, progress=gr.Progress()):
    folder = (folder or "").strip()
    if not folder or not os.path.isdir(folder):
        set_browse_folder_state(state, folder, [], f"Invalid folder: {folder!r}")
        return (*_vw.render_view_with_controls(state, main_threshold, aux_threshold),)

    image_paths = scan_image_paths(folder)
    if not image_paths:
        set_browse_folder_state(state, folder, [], f"No images found in {folder}")
        return (*_vw.render_view_with_controls(state, main_threshold, aux_threshold),)

    sync_promptmatch_proxy_cache(state, folder)
    image_signature = get_image_paths_signature(image_paths)
    cache_dir = state.get("proxy_cache_dir")
    if cache_dir and can_reuse_proxy_map(state, image_paths, image_signature):
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
        state,
        folder,
        browse_items,
        f"Browse mode for {folder}. {len(image_paths)} images loaded. Preview a gallery image or use the external query image to search or generate a prompt.",
    )
    return (*_vw.render_view_with_controls(state, main_threshold, aux_threshold),)


def refresh_promptmatch_model_dropdown(current_model_label):
    selected = current_model_label if current_model_label in MODEL_LABELS else MODEL_LABELS[0]
    return gr.update(choices=promptmatch_model_dropdown_choices(), value=selected)


def middle_threshold_values(state, method):
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


def prepare_scored_run_context(state, method, folder, main_threshold, aux_threshold, llm_backend_id=None):
    folder = (folder or "").strip()
    if not folder or not os.path.isdir(folder):
        return None, _vw.empty_result(state, f"Invalid folder: {folder!r}", method)

    image_paths = scan_image_paths(folder)
    if not image_paths:
        return None, _vw.empty_result(state, f"No images found in {folder}", method)

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
    recalled_main, recalled_aux, has_recalled = recalled_mode_thresholds(state, method, main_threshold, aux_threshold)
    ctx["has_recalled"] = has_recalled
    if ctx["previous_method"] != method and has_recalled:
        ctx["requested_main"], ctx["requested_aux"] = normalize_threshold_inputs(
            method,
            recalled_main,
            recalled_aux,
            llm_backend_id,
        )
    ctx["preserved_overrides"] = preserve_overrides_for_folder_key(
        state,
        folder_key,
        {os.path.basename(path) for path in image_paths},
    )
    return ctx, None


def render_scored_mode_result(state, method, next_main, next_aux, main_upd, aux_upd, percentile_upd, percentile_mid_upd):
    return _vw.build_scored_callback_result(
        state,
        _vw.current_view(state, next_main, next_aux),
        main_upd,
        aux_upd,
        percentile_upd,
        percentile_mid_upd,
    )


def normalize_preview_search_request(state, folder, preview_missing_message, preview_not_in_folder_template):
    folder = (folder or "").strip()
    if not folder or not os.path.isdir(folder):
        return None, f"Invalid folder: {folder!r}"

    image_paths = scan_image_paths(folder)
    if not image_paths:
        return None, f"No images found in {folder}"

    query_ctx = active_query_image_context(state)
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
        "preserved_overrides": preserve_overrides_for_folder_key(state, folder_key, set(folder_name_map.keys())),
    }, None


def score_folder(state, device, method, folder, model_label, pos_prompt, neg_prompt, pm_segment_mode, ir_prompt, ir_negative_prompt, ir_penalty_weight, llm_model_label, llm_prompt, llm_backend_id, llm_shortlist_size, tagmatch_tags, main_threshold, aux_threshold, keep_pm_thresholds, keep_ir_thresholds, progress=gr.Progress()):
    run_ctx, error_result = prepare_scored_run_context(state, method, folder, main_threshold, aux_threshold, llm_backend_id)
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

    begin_scored_run(state, method, folder, preserved_overrides)
    _lo.release_inactive_gpu_models(state, method)

    if method == METHOD_LLMSEARCH:
        llm_model_label = llm_model_label if llm_model_label in MODEL_LABELS else label_for_backend(state["backend"])
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
                _lo.ensure_promptmatch_backend_loaded(state, device, llm_model_label, progress)
                _, feature_paths, image_features, failed_paths = _lo.ensure_promptmatch_feature_cache(
                    state, device, image_paths,
                    llm_model_label,
                    progress,
                    reuse_desc="Reusing cached LLM-search shortlist embeddings for {count} images",
                    encode_desc="Encoding LLM-search shortlist embeddings for {count} images...",
                    progress_label="LLM shortlist",
                )
            except Exception as exc:
                return _vw.empty_result(state, str(exc), method)

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
                return _vw.empty_result(state, "LM search could not shortlist any usable images.", method)

            try:
                llm_candidate_scores = score_llmsearch_candidates(
                    state, device,
                    candidate_paths,
                    llm_prompt,
                    llm_backend_id,
                    image_signature,
                    progress,
                )
            except Exception as exc:
                return _vw.empty_result(state, f"LLM rerank failed: {exc}", method)

            shortlist_floor_candidates = [
                item["pos"] for item in llm_candidate_scores.values()
                if not item.get("failed", False)
            ]
            shortlist_floor = min(shortlist_floor_candidates) if shortlist_floor_candidates else -0.2
            reject_floor = max(-1.0, float(shortlist_floor) - 0.05)
            wrapped_scores = {}
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
            state,
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
            _lo.ensure_promptmatch_backend_loaded(state, device, model_label, progress)
            _, feature_paths, image_features, failed_paths = _lo.ensure_promptmatch_feature_cache(
                state, device, image_paths,
                model_label,
                progress,
                reuse_desc="Reusing cached PromptMatch image embeddings for {count} images",
                encode_desc="Encoding PromptMatch image embeddings for {count} images...",
                progress_label="PromptMatch",
            )
        except Exception as exc:
            return _vw.empty_result(state, str(exc), method)

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
            state,
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
            state["scores"] = score_tagmatch_folder(state, device, image_paths, tagmatch_tags, progress)
        except Exception as exc:
            return _vw.empty_result(state, str(exc), method)
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
            state,
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
        state, device,
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
        state,
        METHOD_IMAGEREWARD,
        next_main,
        NEGATIVE_THRESHOLD,
        gr.update(minimum=safe_lo, maximum=safe_hi, value=next_main, label=main_label),
        gr.update(value=NEGATIVE_THRESHOLD, visible=False),
        percentile_slider_update(METHOD_IMAGEREWARD, state["scores"]),
        percentile_reset_button_update(METHOD_IMAGEREWARD, state["scores"]),
    )


def find_similar_images(state, device, folder, model_label, main_threshold, aux_threshold, progress=gr.Progress()):
    recalled_main, _, has_recalled = recalled_mode_thresholds(state, METHOD_SIMILARITY, main_threshold, aux_threshold)
    request_ctx, request_error = normalize_preview_search_request(
        state,
        folder,
        "Select, drop, paste, or upload a query image first, then find similar images.",
        "{preview_fname} is not part of the current folder, so similarity search cannot run.",
    )
    if request_error:
        return _vw.build_preview_search_result(request_error, _vw.status_with_current_view(state, request_error, main_threshold, aux_threshold))
    folder = request_ctx["folder"]
    image_paths = request_ctx["image_paths"]
    query_path = request_ctx["query_path"]
    query_label = request_ctx["query_label"]
    query_source = request_ctx["query_source"] or "gallery preview"
    query_in_folder = request_ctx["query_in_folder"]
    preserved_overrides = request_ctx["preserved_overrides"]

    _lo.release_inactive_gpu_models(state, METHOD_SIMILARITY)
    try:
        sync_promptmatch_proxy_cache(state, folder)
        _lo.ensure_promptmatch_backend_loaded(state, device, model_label, progress)
        _, feature_paths, image_features, failed_paths = _lo.ensure_promptmatch_feature_cache(
            state, device, image_paths,
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
            query_feature=None if query_in_folder else _lo.encode_single_promptmatch_image(state, query_path),
        )
    except Exception as exc:
        message = f"Similarity search failed: {exc}"
        return _vw.build_preview_search_result(
            message,
            _vw.status_with_current_view(state, message, main_threshold, aux_threshold),
        )

    state["method"] = METHOD_SIMILARITY
    set_scored_mode(state)
    state["source_dir"] = folder
    state["scores"] = similarity_scores
    state["overrides"] = preserved_overrides
    reset_selection_state(state)
    if query_in_folder:
        state["preview_fname"] = request_ctx["preview_fname"]
    clear_preview_search_context(state)
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
        state,
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
    return _vw.build_preview_search_result(status_text, score_outputs)


def find_same_person_images(state, folder, main_threshold, aux_threshold, progress=gr.Progress()):
    recalled_main, _, has_recalled = recalled_mode_thresholds(state, METHOD_SAMEPERSON, main_threshold, aux_threshold)
    request_ctx, request_error = normalize_preview_search_request(
        state,
        folder,
        "Select, drop, paste, or upload a query image first, then find the same person.",
        "{preview_fname} is not part of the current folder, so same-person search cannot run.",
    )
    if request_error:
        return _vw.build_preview_search_result(request_error, _vw.status_with_current_view(state, request_error, main_threshold, aux_threshold))
    folder = request_ctx["folder"]
    image_paths = request_ctx["image_paths"]
    query_path = request_ctx["query_path"]
    query_label = request_ctx["query_label"]
    query_source = request_ctx["query_source"] or "gallery preview"
    query_in_folder = request_ctx["query_in_folder"]
    preserved_overrides = request_ctx["preserved_overrides"]

    _lo.release_inactive_gpu_models(state, METHOD_SAMEPERSON)
    try:
        sync_promptmatch_proxy_cache(state, folder)
        _, feature_paths, face_embeddings, failures = _lo.ensure_face_feature_cache(state, image_paths, progress)
        sameperson_scores = score_sameperson_cached_features(
            feature_paths,
            face_embeddings,
            failures,
            query_path,
            query_embedding=None if query_in_folder else _lo.encode_single_face_embedding(state, query_path, progress),
        )
    except Exception as exc:
        message = f"Same-person search failed: {exc}"
        return _vw.build_preview_search_result(
            message,
            _vw.status_with_current_view(state, message, main_threshold, aux_threshold),
        )

    state["method"] = METHOD_SAMEPERSON
    set_scored_mode(state)
    state["source_dir"] = folder
    state["scores"] = sameperson_scores
    state["overrides"] = preserved_overrides
    reset_selection_state(state)
    if query_in_folder:
        state["preview_fname"] = request_ctx["preview_fname"]
    clear_preview_search_context(state)
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
        state,
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
    return _vw.build_preview_search_result(status_text, score_outputs)


def handle_shortcut_action(state, device, action, method, folder, model_label, pos_prompt, neg_prompt, pm_segment_mode, ir_prompt, ir_negative_prompt, ir_penalty_weight, llm_model_label, llm_prompt, llm_backend_id, llm_shortlist_size, tagmatch_tags, main_threshold, aux_threshold, keep_pm_thresholds, keep_ir_thresholds, progress=gr.Progress()):
    action = (action or "").strip()
    if not action.startswith("run:"):
        return _vw.empty_result(state, "Shortcut action ignored.", method)

    prompt_id = action.split(":", 2)[1] if ":" in action else ""
    if prompt_id in ("hy-pos", "hy-neg"):
        method = METHOD_PROMPTMATCH
    elif prompt_id in ("hy-ir-pos", "hy-ir-neg"):
        method = METHOD_IMAGEREWARD

    return score_folder(
        state, device,
        method, folder, model_label, pos_prompt, neg_prompt, pm_segment_mode,
        ir_prompt, ir_negative_prompt, ir_penalty_weight,
        llm_model_label, llm_prompt, llm_backend_id, llm_shortlist_size,
        tagmatch_tags, main_threshold, aux_threshold,
        keep_pm_thresholds, keep_ir_thresholds,
        progress=progress,
    )
