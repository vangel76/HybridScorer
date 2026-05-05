import os
import json
import base64

from .. import ui_compat as gr
import torch
import numpy as np
from PIL import Image

from ..config import (
    METHOD_PROMPTMATCH, METHOD_LLMSEARCH, METHOD_TAGMATCH,
    PROMPT_GENERATOR_FLORENCE, PROMPT_GENERATOR_JOYCAPTION,
    PROMPT_GENERATOR_JOYCAPTION_NF4, PROMPT_GENERATOR_HUIHUI_GEMMA4,
    PROMPT_GENERATOR_WD_TAGS, FLORENCE_MAX_NEW_TOKENS,
)
from ..utils import (
    describe_prompt_generator_source, prompt_generator_supports_torch_cleanup,
    prompt_backend_warning_text,
)
from ..helpers import (
    prompt_generator_detail_config,
    extract_florence_caption, normalize_generated_prompt, florence_task_is_pure_text,
    extract_joycaption_caption, extract_huihui_gemma4_caption,
    joycaption_max_new_tokens,
    move_processor_batch_to_device,
)
from ..state_helpers import (
    active_query_image_context,
    clear_external_query_state, active_query_image_widget_update,
    clear_external_query_button_update,
    save_external_query_image, save_external_query_image_bytes,
)
from .. import loaders as _lo

_DETAIL_LEVELS = (1, 2, 3)


def run_florence_prompt_variant(state, device, image, task_prompt):
    model, processor = _lo.ensure_florence_model(state, device)
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


def _resolve_generate_params(system_prompt, default_system, normalizer, default_normalizer,
                              max_new_tokens_override, detail_level, temperature_override,
                              top_p_override, top_k_override):
    system_prompt = system_prompt or default_system
    if normalizer is None:
        normalizer = default_normalizer
    max_new_tokens = max_new_tokens_override if max_new_tokens_override is not None else joycaption_max_new_tokens(detail_level)
    temperature = temperature_override if temperature_override is not None else 0.0
    top_p = top_p_override if top_p_override is not None else 1.0
    top_k = top_k_override if top_k_override is not None else 0
    do_sample = bool(temperature and temperature > 0.0)
    return system_prompt, normalizer, max_new_tokens, temperature, top_p, top_k, do_sample


def run_huihui_gemma4_prompt_variant(
    state,
    device,
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
    system_prompt, normalizer, max_new_tokens, temperature, top_p, top_k, do_sample = _resolve_generate_params(
        system_prompt,
        (
            "You are a helpful multimodal prompt writer. "
            "Describe only concrete visible content. "
            "Follow the requested output format exactly and avoid refusals or moralizing."
        ),
        normalizer,
        lambda text: normalize_generated_prompt(extract_huihui_gemma4_caption(text), keep_prose=(detail_level == 3)),
        max_new_tokens_override, detail_level, temperature_override, top_p_override, top_k_override,
    )

    model, processor = _lo.ensure_huihui_gemma4_model(state, device)
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


def run_joycaption_prompt_variant(state, device, generator_name, image, user_prompt, detail_level, system_prompt=None, normalizer=None, max_new_tokens_override=None, stop_sequences=None, temperature_override=None, top_p_override=None, top_k_override=None):
    system_prompt, normalizer, max_new_tokens, temperature, top_p, top_k, do_sample = _resolve_generate_params(
        system_prompt,
        (
            "You are a helpful image captioner. "
            "Describe only concrete visible content and write output that is useful as a text-to-image prompt. "
            "Follow the requested output style exactly, whether it asks for short tags, a compact prompt line, or natural prose. "
            "Do not begin with meta phrases like 'This image shows', 'In this image we can see', or 'You are looking at'."
        ),
        normalizer,
        lambda text: normalize_generated_prompt(extract_joycaption_caption(text), keep_prose=(detail_level == 3)),
        max_new_tokens_override, detail_level, temperature_override, top_p_override, top_k_override,
    )

    if generator_name in (PROMPT_GENERATOR_JOYCAPTION, PROMPT_GENERATOR_JOYCAPTION_NF4):
        if generator_name == PROMPT_GENERATOR_JOYCAPTION_NF4:
            model, processor = _lo.ensure_joycaption_nf4_model(state, device)
        else:
            model, processor = _lo.ensure_joycaption_model(state, device)
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
            pv_dtype = torch.bfloat16 if generator_name == PROMPT_GENERATOR_JOYCAPTION_NF4 else next(model.parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(pv_dtype)

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

    raise RuntimeError(f"Unknown JoyCaption generator: {generator_name}")


def run_wd_tags_prompt_variant(state, device, image, top_n):
    backend = _lo.ensure_tagmatch_model(state, device)
    session = backend["session"]
    tags = backend["tags"]
    input_name = backend["input_name"]
    arr = _lo.tagmatch_prepare_image(image)
    batch_np = arr[np.newaxis, ...]
    raw_out = session.run(None, {input_name: batch_np})[0][0]
    probs = 1.0 / (1.0 + np.exp(-raw_out))
    tag_prob_pairs = sorted(
        ((tags[j], float(probs[j])) for j in range(len(tags))),
        key=lambda x: x[1],
        reverse=True,
    )
    top_tags = [t for t, p in tag_prob_pairs[:top_n] if p >= 0.05]
    return ", ".join(top_tags)


def generate_prompt_variant(state, device, generator_name, image, detail_level):
    _, _, detail_prompt = prompt_generator_detail_config(generator_name, detail_level)
    if generator_name == PROMPT_GENERATOR_FLORENCE:
        return run_florence_prompt_variant(state, device, image, detail_prompt)
    if generator_name == PROMPT_GENERATOR_WD_TAGS:
        return run_wd_tags_prompt_variant(state, device, image, detail_prompt)
    if generator_name == PROMPT_GENERATOR_HUIHUI_GEMMA4:
        return run_huihui_gemma4_prompt_variant(state, device, image, detail_prompt, detail_level)
    return run_joycaption_prompt_variant(state, device, generator_name, image, detail_prompt, detail_level)


def generated_prompt_variants_for(state, query_cache_key, generator_name, create=False):
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


def select_cached_generated_prompt(state, generator_name, detail_level, current_generated_prompt):
    detail_level, detail_label, _ = prompt_generator_detail_config(generator_name, detail_level)
    state["prompt_generator"] = generator_name
    state["generated_prompt_detail"] = detail_level
    query_ctx = active_query_image_context(state)
    query_key = query_ctx.get("cache_key")
    query_label = query_ctx.get("label") or state.get("generated_prompt_source")
    prompt_text = generated_prompt_variants_for(state, query_key, generator_name).get(detail_level)
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


def external_query_prompt_status(state):
    query_ctx = active_query_image_context(state)
    if query_ctx["source_kind"] == "external":
        return (
            f"External query image ready: {query_ctx['label']}. "
            "Click Prompt from image, Find similar images, or Find same person."
        )
    if query_ctx["source_kind"] == "gallery" and query_ctx["label"]:
        return f"Using gallery preview {query_ctx['label']} for preview actions."
    return "Preview, drop, paste, or upload an image, then generate a prompt from it."


def set_external_query_image(state, image_path):
    if not image_path:
        clear_external_query_state(state, delete_file=True)
        return (
            active_query_image_widget_update(state),
            clear_external_query_button_update(state),
            gr.update(value=external_query_prompt_status(state)),
        )
    try:
        save_external_query_image(state, image_path)
    except Exception as exc:
        return (
            gr.update(value=None),
            clear_external_query_button_update(state),
            gr.update(value=f"External query image failed: {exc}"),
        )
    return (
        active_query_image_widget_update(state),
        clear_external_query_button_update(state),
        gr.update(value=external_query_prompt_status(state)),
    )


def set_external_query_from_bridge(state, payload):
    payload = (payload or "").strip()
    if not payload:
        return (
            active_query_image_widget_update(state),
            clear_external_query_button_update(state),
            gr.update(value=external_query_prompt_status(state)),
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
        save_external_query_image_bytes(state, image_bytes, label)
    except Exception as exc:
        return (
            gr.update(value=None),
            clear_external_query_button_update(state),
            gr.update(value=f"External query image failed: {exc}"),
            gr.update(value=""),
        )
    return (
        active_query_image_widget_update(state),
        clear_external_query_button_update(state),
        gr.update(value=external_query_prompt_status(state)),
        gr.update(value=""),
    )


def clear_external_query_image(state):
    clear_external_query_state(state, delete_file=True)
    return (
        active_query_image_widget_update(state),
        clear_external_query_button_update(state),
        gr.update(value=external_query_prompt_status(state)),
    )


def generate_prompt_from_preview(state, device, generator_name, current_generated_prompt, detail_level, progress=gr.Progress()):
    detail_level, detail_label, _ = prompt_generator_detail_config(generator_name, detail_level)
    state["prompt_generator"] = generator_name
    state["generated_prompt_detail"] = detail_level
    query_ctx = active_query_image_context(state)
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

    variants = generated_prompt_variants_for(state, query_key, generator_name, create=True)
    if all(level in variants and variants[level] for level in _DETAIL_LEVELS):
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

        for idx, level in enumerate(_DETAIL_LEVELS, start=1):
            _, loop_label, _ = prompt_generator_detail_config(generator_name, level)
            progress(
                0.2 + (0.7 * ((idx - 1) / 3.0)),
                desc=f"Generating {loop_label.lower()} prompt from {preview_fname} via {generator_name}",
            )
            variants[level] = generate_prompt_variant(state, device, generator_name, image, level)
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
    ready_count = sum(1 for level in _DETAIL_LEVELS if variants.get(level))
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


def insert_generated_prompt(state, method, prompt_text):
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        state["generated_prompt_status"] = "Generated prompt is empty. Edit or generate a prompt first."
        return gr.update(value=state["generated_prompt_status"]), gr.update(), gr.update(), gr.update()

    state["generated_prompt"] = prompt_text
    if method == METHOD_PROMPTMATCH:
        target_label = "PromptMatch positive prompt"
    elif method == METHOD_LLMSEARCH:
        target_label = "LM search prompt"
    elif method == METHOD_TAGMATCH:
        target_label = "TagMatch tags"
    else:
        target_label = "ImageReward positive prompt"
    state["generated_prompt_status"] = f"Inserted generated prompt into {target_label}."
    skip = gr.update()
    status = gr.update(value=state["generated_prompt_status"])
    if method == METHOD_PROMPTMATCH:
        return status, gr.update(value=prompt_text), skip, skip, skip
    if method == METHOD_LLMSEARCH:
        return status, skip, skip, gr.update(value=prompt_text), skip
    if method == METHOD_TAGMATCH:
        return status, skip, skip, skip, gr.update(value=prompt_text)
    return status, skip, gr.update(value=prompt_text), skip, skip
