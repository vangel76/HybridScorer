import base64
import io
import json
import math
import os
import re
import string

from . import ui_compat as gr
import torch
from PIL import Image

from .config import (
    METHOD_PROMPTMATCH,
    METHOD_IMAGEREWARD,
    METHOD_LLMSEARCH,
    METHOD_SIMILARITY,
    METHOD_SAMEPERSON,
    METHOD_TAGMATCH,
    METHOD_OBJECTSEARCH,
    PROMPT_GENERATOR_FLORENCE,
    PROMPT_GENERATOR_JOYCAPTION,
    PROMPT_GENERATOR_JOYCAPTION_NF4,
    PROMPT_GENERATOR_HUIHUI_GEMMA4,
    PROMPT_GENERATOR_WD_TAGS,
    PROMPT_GENERATOR_ALL_CHOICES,
    DEFAULT_GENERATED_PROMPT_DETAIL,
    LLMSEARCH_JOYCAPTION_SYSTEM_PROMPT,
    NEGATIVE_THRESHOLD,
    IMAGEREWARD_THRESHOLD,
    TAGMATCH_SLIDER_MIN,
    TAGMATCH_SLIDER_MAX,
    IMAGEREWARD_SLIDER_MIN,
    IMAGEREWARD_SLIDER_MAX,
    PROMPTMATCH_SLIDER_MIN,
    PROMPTMATCH_SLIDER_MAX,
    SIMILARITY_TOPN_DEFAULT,
    SIMILARITY_TOPN_SLIDER_MAX,
    SIMILARITY_AUTO_TOPN_SCAN_LIMIT,
    SIMILARITY_AUTO_KNEE_MIN_DISTANCE,
    SIMILARITY_AUTO_TOPN_MIN,
    ALLOWED_EXTENSIONS,
    MODEL_CHOICES,
    MODEL_LABELS,
    MODEL_STATUS_CACHED_MARKER,
    MODEL_STATUS_DOWNLOAD_MARKER,
)
from .utils import (
    normalize_prompt_text,
    describe_openai_clip_source,
    describe_openclip_source,
    describe_siglip_source,
    describe_tagmatch_source,
    describe_prompt_generator_source,
)


def label_for_backend(backend):
    if backend is None:
        return MODEL_LABELS[0]
    for label, name, kwargs in MODEL_CHOICES:
        if name != backend.backend:
            continue
        if name == "openai" and kwargs.get("clip_model") == backend._clip_model:
            return label
        if (
            name == "openclip"
            and kwargs.get("openclip_model") == backend._openclip_model
            and kwargs.get("openclip_pretrained") == backend._openclip_pre
        ):
            return label
        if name == "siglip" and kwargs.get("siglip_model") == backend._siglip_model:
            return label
    return MODEL_LABELS[0]


def promptmatch_model_status_map():
    mapping = {}
    for label, name, kwargs in MODEL_CHOICES:
        if name == "openai":
            source = describe_openai_clip_source(kwargs.get("clip_model"))
        elif name == "openclip":
            source = describe_openclip_source(kwargs.get("openclip_model"), kwargs.get("openclip_pretrained"))
        elif name == "siglip":
            source = describe_siglip_source(kwargs.get("siglip_model"))
        else:
            source = "network or disk cache"
        mapping[label] = {
            "cached": source in {"disk cache", "disk file"},
            "source": source,
        }
    for label in PROMPT_GENERATOR_ALL_CHOICES:
        if label == PROMPT_GENERATOR_WD_TAGS:
            source = describe_tagmatch_source()
        else:
            source = describe_prompt_generator_source(label)
        mapping[label] = {
            "cached": source in {"disk cache", "disk file"},
            "source": source,
        }
    return mapping


def promptmatch_model_status_json():
    return json.dumps(promptmatch_model_status_map())


def promptmatch_model_dropdown_choices(status_map=None):
    if status_map is None:
        status_map = promptmatch_model_status_map()
    choices = []
    for label in MODEL_LABELS:
        entry = status_map.get(label) or {}
        marker = MODEL_STATUS_CACHED_MARKER if entry.get("cached") else MODEL_STATUS_DOWNLOAD_MARKER
        choices.append((f"{marker} {label}", label))
    return choices


def prompt_backend_dropdown_choices(labels, status_map=None):
    if status_map is None:
        status_map = promptmatch_model_status_map()
    choices = []
    for label in labels:
        entry = status_map.get(label) or {}
        marker = MODEL_STATUS_CACHED_MARKER if entry.get("cached") else MODEL_STATUS_DOWNLOAD_MARKER
        choices.append((f"{marker} {label}", label))
    return choices


def is_windows():
    return os.name == "nt"


def folder_placeholder():
    return r"C:\path\to\images" if is_windows() else "/path/to/images"


def get_allowed_paths(*extra_paths):
    if not is_windows():
        return ["/"]

    allowed = []
    for drive in string.ascii_uppercase:
        root = f"{drive}:\\"
        if os.path.exists(root):
            allowed.append(root)
    for path in extra_paths:
        if path:
            allowed.append(os.path.abspath(path))

    deduped = []
    seen = set()
    for path in allowed:
        norm = os.path.normcase(os.path.abspath(path))
        if norm not in seen:
            seen.add(norm)
            deduped.append(path)
    return deduped or [os.path.abspath(os.getcwd())]


def scan_image_paths(folder):
    if not folder or not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(ALLOWED_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, f))
    ]


def scan_image_paths_recursive(folder):
    if not folder or not os.path.isdir(folder):
        return []
    paths = []
    for root, _dirs, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(ALLOWED_EXTENSIONS):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def get_model_config(label):
    return next((cfg for cfg in MODEL_CHOICES if cfg[0] == label), None)


def method_labels(method):
    return "SELECTED", "REJECTED", "selected", "rejected"


def sanitize_export_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("._-")
    return cleaned


def uses_similarity_topn(method):
    return method in (METHOD_SIMILARITY, METHOD_SAMEPERSON, METHOD_OBJECTSEARCH)


def uses_pos_similarity_scores(method):
    return method in (METHOD_PROMPTMATCH, METHOD_LLMSEARCH, METHOD_SIMILARITY, METHOD_SAMEPERSON, METHOD_TAGMATCH, METHOD_OBJECTSEARCH)



_THRESHOLD_LABELS = {
    METHOD_LLMSEARCH: (
        "Minimum LLM rerank score to keep (higher = fewer kept)",
        "Negative score is unused for LLM rerank search",
        "Minimum LLM rerank score",
        "Negative score",
    ),
    METHOD_SIMILARITY: (
        "Minimum similarity to keep (higher = fewer kept)",
        "Negative similarity is unused for image similarity search",
        "Minimum similarity",
        "Negative similarity",
    ),
    METHOD_SAMEPERSON: (
        "Minimum face similarity to keep (higher = fewer kept)",
        "Negative face similarity is unused for same-person search",
        "Minimum face similarity",
        "Negative face similarity",
    ),
    METHOD_PROMPTMATCH: (
        "Minimum positive similarity to keep (higher = fewer kept)",
        "Maximum negative similarity allowed (lower = fewer kept)",
        "Min positive similarity",
        "Max negative similarity",
    ),
    METHOD_TAGMATCH: (
        "Minimum artifact score to keep (higher = fewer kept)",
        "Negative score is unused for TagMatch",
        "Min artifact score",
        "Negative score",
    ),
    METHOD_OBJECTSEARCH: (
        "Minimum object match score to keep (higher = fewer kept)",
        "Negative score is unused for object search",
        "Min object match score",
        "Negative score",
    ),
}
_THRESHOLD_LABELS_DEFAULT = (
    "Minimum score to keep (higher = fewer kept)",
    "Maximum negative similarity allowed (lower = fewer kept)",
    "Minimum keep score",
    "Max negative similarity",
)


def threshold_labels(method):
    return _THRESHOLD_LABELS.get(method, _THRESHOLD_LABELS_DEFAULT)


def promptmatch_slider_range(scores):
    pos_vals, neg_vals = [], []
    for v in scores.values():
        if v.get("failed", False):
            continue
        pos_vals.append(v["pos"])
        if v.get("neg") is not None:
            neg_vals.append(v["neg"])
    pos_min = round(min(pos_vals) - 0.01, 3) if pos_vals else -1.0
    pos_max = round(max(pos_vals) + 0.01, 3) if pos_vals else 1.0
    pos_mid = round((pos_min + pos_max) / 2.0, 3)
    neg_min = round(min(neg_vals) - 0.01, 3) if neg_vals else -1.0
    neg_max = round(max(neg_vals) + 0.01, 3) if neg_vals else 1.0
    neg_mid = round((neg_min + neg_max) / 2.0, 3) if neg_vals else NEGATIVE_THRESHOLD
    return pos_min, pos_max, pos_mid, neg_min, neg_max, neg_mid, bool(neg_vals)


def imagereward_slider_range(scores):
    vals = [v["score"] for v in scores.values() if v["score"] > -1000]
    if not vals:
        return -2.0, 2.5, IMAGEREWARD_THRESHOLD
    lo = round(min(vals) - 0.05, 3)
    hi = round(max(vals) + 0.05, 3)
    mid = round((lo + hi) / 2.0, 3)
    return lo, hi, mid


def llmsearch_uses_numeric_scores(backend_id):
    return backend_id in {
        PROMPT_GENERATOR_JOYCAPTION,
        PROMPT_GENERATOR_JOYCAPTION_NF4,
        PROMPT_GENERATOR_HUIHUI_GEMMA4,
    }


def llmsearch_slider_range(scores, backend_id=None):
    vals = [float(v["pos"]) for v in (scores or {}).values() if not v.get("failed", False) and v.get("pos") is not None]
    if llmsearch_uses_numeric_scores(backend_id):
        if not vals:
            return 0.0, 100.0, 50.0
        lo = max(0.0, round(min(vals) - 0.5, 3))
        hi = min(100.0, round(max(vals) + 0.5, 3))
        if lo == hi:
            lo = max(0.0, lo - 0.5)
            hi = min(100.0, hi + 0.5)
        mid = round((lo + hi) / 2.0, 3)
        return lo, hi, mid
    pos_lo, pos_hi, pos_mid, _, _, _, _ = promptmatch_slider_range(scores or {})
    return pos_lo, pos_hi, pos_mid


def threshold_for_percentile(method, scores, percentile):
    if uses_similarity_topn(method):
        valid_items = [
            item for item in (scores or {}).values()
            if not item.get("failed", False) and item.get("pos") is not None
        ]
        if not valid_items:
            return 0.0
        valid_items.sort(key=lambda item: -float(item["pos"]))
        query_offset = 1 if valid_items and valid_items[0].get("query") else 0
        vals = [float(item["pos"]) for item in valid_items[query_offset:]]
        if not vals:
            return round(float(valid_items[0]["pos"]), 3)
        try:
            top_n = int(round(float(percentile)))
        except Exception:
            top_n = SIMILARITY_TOPN_DEFAULT
        top_n = max(1, min(len(vals), top_n))
        return round(vals[top_n - 1], 3)
    if uses_pos_similarity_scores(method):
        vals = sorted([v["pos"] for v in scores.values() if not v.get("failed", False)], reverse=True)
    else:
        vals = sorted([v["score"] for v in scores.values() if v["score"] > -1000], reverse=True)
    if not vals:
        return 0.0
    if percentile <= 0:
        return max(vals) + 0.01
    if percentile >= 100:
        return min(vals) - 0.01
    idx = max(0, int(len(vals) * (percentile / 100.0)) - 1)
    return round(vals[idx], 3)


def percentile_slider_label(method):
    if method == METHOD_LLMSEARCH:
        return "Or keep top N%"
    if method == METHOD_SIMILARITY:
        return "Show the N most similar"
    if method == METHOD_SAMEPERSON:
        return "Show the N closest face matches"
    if method == METHOD_OBJECTSEARCH:
        return "Show the N best object matches"
    return "Or keep top N%"


def _sorted_similarity_scores(scores):
    """Return (sorted valid_items, query_offset) for similarity/topn calculations."""
    valid_items = [
        item for item in (scores or {}).values()
        if not item.get("failed", False) and item.get("pos") is not None
    ]
    valid_items.sort(key=lambda item: -float(item["pos"]))
    query_offset = 1 if valid_items and valid_items[0].get("query") else 0
    return valid_items, query_offset


def estimate_similarity_topn(scores=None, _sorted_items=None):
    if _sorted_items is not None:
        valid_items, query_offset = _sorted_items
    else:
        valid_items, query_offset = _sorted_similarity_scores(scores)

    if not valid_items:
        return 1

    ranked_scores = [float(item["pos"]) for item in valid_items[query_offset:query_offset + SIMILARITY_AUTO_TOPN_SCAN_LIMIT]]

    if len(ranked_scores) < 2:
        return max(1, min(len(ranked_scores), SIMILARITY_TOPN_SLIDER_MAX))

    first = ranked_scores[0]
    last = ranked_scores[-1]
    if abs(first - last) < 1e-9:
        fallback = min(len(ranked_scores), SIMILARITY_TOPN_DEFAULT)
        return max(1, fallback)

    best_idx = 0
    best_distance = -float("inf")
    span = max(1, len(ranked_scores) - 1)
    for idx, score in enumerate(ranked_scores):
        x = idx / span
        y = (score - last) / (first - last)
        distance = y - (1.0 - x)
        if distance > best_distance:
            best_distance = distance
            best_idx = idx

    if best_distance < SIMILARITY_AUTO_KNEE_MIN_DISTANCE:
        fallback = min(len(ranked_scores), SIMILARITY_TOPN_DEFAULT)
        return max(1, min(fallback, SIMILARITY_TOPN_SLIDER_MAX))

    auto_topn = best_idx + 1
    min_topn = min(len(ranked_scores), max(1, SIMILARITY_AUTO_TOPN_MIN))
    auto_topn = max(min_topn, auto_topn)
    auto_topn = min(len(ranked_scores), auto_topn, SIMILARITY_TOPN_SLIDER_MAX)
    return max(1, auto_topn)


def similarity_topn_defaults(scores=None):
    sorted_data = _sorted_similarity_scores(scores)
    valid_items, query_offset = sorted_data
    if not valid_items:
        return 1, SIMILARITY_TOPN_DEFAULT
    similar_count = len(valid_items) - query_offset
    if similar_count <= 0:
        return 1, 1
    slider_max = min(SIMILARITY_TOPN_SLIDER_MAX, similar_count)
    auto_default = estimate_similarity_topn(_sorted_items=sorted_data)
    return slider_max, min(auto_default, slider_max)


def percentile_slider_update(method, scores=None):
    if uses_similarity_topn(method):
        max_items, default_items = similarity_topn_defaults(scores)
        return gr.update(
            minimum=1,
            maximum=max_items,
            value=default_items,
            step=1,
            label=percentile_slider_label(method),
        )
    return gr.update(
        minimum=0,
        maximum=100,
        value=50,
        step=1,
        label=percentile_slider_label(method),
    )


def percentile_reset_button_update(method, scores=None):
    if uses_similarity_topn(method):
        _, default_items = similarity_topn_defaults(scores)
        return gr.update(value=str(default_items))
    return gr.update(value="50%")


def slider_step_floor(value, step=0.001):
    return round(math.floor(float(value) / step) * step, 3)


def slider_step_ceil_exclusive(value, step=0.001):
    return round((math.floor(float(value) / step) + 1) * step, 3)


def clamp_threshold(value, lo, hi):
    return round(max(float(lo), min(float(hi), float(value))), 3)


def expand_slider_bounds(lo, hi, *values):
    safe_values = [float(v) for v in values if v is not None]
    if safe_values:
        lo = min(float(lo), *safe_values)
        hi = max(float(hi), *safe_values)
    # round() can push lo above/hi below the true min/max, causing slider out-of-bounds errors.
    return math.floor(lo * 1000) / 1000, math.ceil(hi * 1000) / 1000


def normalize_threshold_inputs(method, main_threshold, aux_threshold, llm_backend_id=None):
    main_value = float(main_threshold)
    aux_value = float(aux_threshold)
    if method == METHOD_TAGMATCH:
        main_value = clamp_threshold(main_value, TAGMATCH_SLIDER_MIN, TAGMATCH_SLIDER_MAX)
        aux_value = clamp_threshold(aux_value, TAGMATCH_SLIDER_MIN, TAGMATCH_SLIDER_MAX)
    elif method == METHOD_LLMSEARCH and llmsearch_uses_numeric_scores(llm_backend_id):
        main_value = clamp_threshold(main_value, 0.0, 100.0)
        aux_value = clamp_threshold(aux_value, 0.0, 100.0)
    elif method == METHOD_IMAGEREWARD:
        main_value = clamp_threshold(main_value, IMAGEREWARD_SLIDER_MIN, IMAGEREWARD_SLIDER_MAX)
        aux_value = clamp_threshold(aux_value, PROMPTMATCH_SLIDER_MIN, PROMPTMATCH_SLIDER_MAX)
    else:
        main_value = clamp_threshold(main_value, PROMPTMATCH_SLIDER_MIN, PROMPTMATCH_SLIDER_MAX)
        aux_value = clamp_threshold(aux_value, PROMPTMATCH_SLIDER_MIN, PROMPTMATCH_SLIDER_MAX)
    return main_value, aux_value


def extract_florence_caption(parsed, raw_text, task_prompt):
    def _clean(text):
        text = re.sub(r"<[^>]+>", " ", text or "")
        return re.sub(r"\s+", " ", text).strip()

    if isinstance(parsed, dict):
        value = parsed.get(task_prompt)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return ", ".join(value)
    if isinstance(parsed, str):
        return _clean(parsed)

    return _clean(raw_text)


def normalize_generated_prompt(text, keep_prose=False):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""

    lowered = text.lower()
    prefixes = (
        "in this image, we can see ",
        "in this image we can see ",
        "in this image, we see ",
        "in this image we see ",
        "this image shows ",
        "this image depicts ",
        "the image shows ",
        "the image depicts ",
        "this photo shows ",
        "this photo depicts ",
        "you are looking at ",
        "a photo of ",
        "an image of ",
        "image of ",
        "photo of ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    if keep_prose:
        text = re.sub(r"\s*([,:;])\s*", r"\1 ", text)
        text = re.sub(r"\s*\.\s*", ". ", text)
        text = re.sub(r"\s+", " ", text).strip(" ,;:-")
        return text.strip()

    text = re.sub(r"\s*[:;]\s*", ", ", text)
    text = re.sub(r"\.\s*", ", ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    parts = [part.strip(" ,.;:-") for part in text.split(",")]
    parts = [part for part in parts if part]
    return ", ".join(parts)


def florence_task_is_pure_text(task_prompt):
    return task_prompt in {
        "<CAPTION>",
        "<DETAILED_CAPTION>",
        "<MORE_DETAILED_CAPTION>",
        "<REGION_TO_CATEGORY>",
        "<REGION_TO_DESCRIPTION>",
        "<REGION_TO_OCR>",
    }


def _clamp_detail_level(detail_level):
    try:
        detail_level = int(detail_level)
    except Exception:
        detail_level = DEFAULT_GENERATED_PROMPT_DETAIL
    return max(1, min(3, detail_level))


def florence_detail_config(detail_level):
    detail_level = _clamp_detail_level(detail_level)
    mapping = {
        1: ("Core facts", "<CAPTION>"),
        2: ("Balanced", "<DETAILED_CAPTION>"),
        3: ("Full", "<MORE_DETAILED_CAPTION>"),
    }
    return detail_level, *mapping[detail_level]


def joycaption_detail_config(detail_level):
    detail_level = _clamp_detail_level(detail_level)
    mapping = {
        1: (
            "Core facts",
            (
                "Write only a very short comma-separated tag list for this image. "
                "Use about 4 to 8 short tags total. "
                "Include only the main subject, medium or style, pose, setting, lighting, and one standout object if clearly visible. "
                "Do not write full sentences. "
                "Drop minor details completely and avoid repeated concepts. "
                "Avoid meta lead-ins like 'This image shows', 'In this image we can see', or 'You are looking at'. "
                "Write it so the result is useful as a text-to-image prompt."
            ),
        ),
        2: (
            "Balanced",
            (
                "Write one compact text-to-image prompt line for this image. "
                "Use about 18 to 35 words maximum. "
                "Prefer prompt fragments separated by commas, not full prose. "
                "Describe the visible subject, appearance, clothing, pose, background, lighting, framing, and the most notable objects. "
                "Keep only clearly useful details, avoid speculation, and keep it prompt-friendly. "
                "Avoid meta lead-ins like 'This image shows', 'In this image we can see', or 'You are looking at'. "
                "Write it so the result is useful as a text-to-image prompt."
            ),
        ),
        3: (
            "Full",
            (
                "Write one natural descriptive paragraph for this image. "
                "Use about 45 to 90 words maximum. "
                "Write in clear flowing prose, not a tag list. "
                "Include concrete visible details about subject, appearance, outfit, pose, composition, background, lighting, viewpoint, and notable objects. "
                "Use direct factual language that is still useful for a text-to-image prompt. "
                "Avoid meta lead-ins like 'This image shows', 'In this image we can see', or 'You are looking at'."
            ),
        ),
    }
    return detail_level, *mapping[detail_level]


def wd_tags_detail_config(detail_level):
    detail_level = _clamp_detail_level(detail_level)
    # detail_prompt field holds the top-N count used by run_wd_tags_prompt_variant
    mapping = {
        1: ("Top 12 tags",  12),
        2: ("Top 36 tags",  36),
        3: ("Top 96 tags",  96),
    }
    return detail_level, *mapping[detail_level]


def joycaption_max_new_tokens(detail_level):
    detail_level = _clamp_detail_level(detail_level)
    mapping = {
        1: 24,
        2: 56,
        3: 128,
    }
    return mapping[detail_level]


def prompt_generator_detail_config(generator_name, detail_level):
    if generator_name == PROMPT_GENERATOR_FLORENCE:
        return florence_detail_config(detail_level)
    if generator_name == PROMPT_GENERATOR_WD_TAGS:
        return wd_tags_detail_config(detail_level)
    return joycaption_detail_config(detail_level)


def extract_joycaption_caption(text):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    text = re.sub(r"^(assistant|caption)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_huihui_gemma4_caption(text):
    text = extract_joycaption_caption(text)
    if not text:
        return ""

    role_echo_match = re.search(r"\bsystem\b.*\buser\b.*\bmodel\b\s*(.+)$", text, flags=re.IGNORECASE)
    if role_echo_match:
        candidate = role_echo_match.group(1).strip()
        if candidate:
            text = candidate

    text = re.sub(r"^(model|assistant)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def move_processor_batch_to_device(batch, target_device, float_dtype=None):
    if hasattr(batch, "to"):
        try:
            batch = batch.to(target_device)
        except Exception:
            pass

    if not isinstance(batch, dict):
        return batch

    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            try:
                moved[key] = value.to(target_device)
            except Exception:
                moved[key] = value
        else:
            moved[key] = value

    if float_dtype is not None:
        pixel_values = moved.get("pixel_values")
        if hasattr(pixel_values, "to"):
            try:
                moved["pixel_values"] = pixel_values.to(float_dtype)
            except Exception:
                pass
    return moved



def build_llmsearch_joycaption_user_prompt(query_text):
    normalized_query = normalize_prompt_text(query_text or "")
    return (
        "You are an image evaluation model.\n\n"
        "Your task is to score how well an image matches a given user prompt.\n\n"
        "Scoring rules:\n"
        "- Return a single number from 0 to 100 (integer only).\n"
        "- 0 = the image does NOT match the described prompt at all.\n"
        "- 100 = the image clearly and strongly matches the described prompt.\n"
        "- Intermediate values reflect partial presence or uncertainty.\n\n"
        "Evaluation guidelines:\n"
        "- Focus ONLY on visible evidence in the image.\n"
        "- Do NOT hallucinate details.\n"
        "- Be strict: weak or partial matches = low score, strong obvious matches = high score.\n"
        "- If multiple traits, attributes, objects, actions, styles, or conditions are listed, evaluate their combined presence.\n"
        "- If nothing is visible, return 0.\n\n"
        "Output format:\n"
        "- Return ONLY the number.\n"
        "- No explanation, no text, no symbols, no percentage sign.\n\n"
        "Prompt:\n"
        f"{normalized_query or 'no traits provided'}"
    )


def build_llmsearch_huihui_gemma4_user_prompt(query_text):
    normalized_query = normalize_prompt_text(query_text or "")
    return (
        "Score how well the image matches the user prompt.\n\n"
        "Rules:\n"
        "- Return exactly one integer from 0 to 100.\n"
        "- 0 means no meaningful match.\n"
        "- 100 means the prompt is clearly and strongly present.\n"
        "- Be strict and rely only on visible evidence.\n"
        "- If the output contains anything except the integer, the answer is invalid.\n\n"
        "Prompt:\n"
        f"{normalized_query or 'no traits provided'}"
    )


def normalize_llmsearch_candidate_text(text):
    text = extract_joycaption_caption(text)
    text = re.sub(r"^(search prompt|user search prompt|match summary|caption)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[-*]\s*", ", ", text)
    return normalize_generated_prompt(text)


def extract_llmsearch_numeric_score(text):
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        raise ValueError("LLM rerank backend returned an empty score.")
    match = re.fullmatch(r"(100|[0-9]{1,2})", cleaned)
    if not match:
        raise ValueError(f"LLM rerank backend did not return a usable 0-100 integer: {cleaned!r}")
    score_value = int(match.group(1))
    return max(0, min(100, score_value))


def image_to_data_url(image, image_format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def status_line(method, left_items, right_items, scores, overrides):
    left_name, right_name, _, _ = method_labels(method)
    if not scores:
        failed = 0
    elif uses_pos_similarity_scores(method) and all("pos" in v for v in scores.values()):
        failed = sum(1 for v in scores.values() if v.get("failed", False))
    elif method == METHOD_IMAGEREWARD and all("score" in v for v in scores.values()):
        failed = sum(1 for v in scores.values() if v["score"] < -1000)
    else:
        failed = 0
    text = f"{len(left_items)} {left_name.lower()} / {len(right_items)} {right_name.lower()} ({len(scores)} total)"
    if overrides:
        text += f"  •  {len(overrides)} manual override"
    if failed:
        text += f"  •  {failed} failed"
    return text


def build_split(method, scores, overrides, main_threshold, aux_threshold):
    left, right = [], []
    left_name, right_name, _, _ = method_labels(method)
    if not scores:
        return left, right

    if uses_pos_similarity_scores(method) and not all("pos" in item for item in scores.values()):
        return left, right
    if method == METHOD_IMAGEREWARD and not all("score" in item for item in scores.values()):
        return left, right

    if uses_pos_similarity_scores(method):
        ordered = sorted(scores.items(), key=lambda x: -x[1]["pos"])
        for fname, item in ordered:
            side = overrides.get(fname)
            if side is None:
                if item.get("failed", False):
                    side = right_name
                else:
                    pos_ok = item["pos"] >= main_threshold
                    neg_ok = (item["neg"] is None) or (item["neg"] < aux_threshold)
                    side = left_name if (pos_ok and neg_ok) else right_name
            score_text = "FAILED" if item.get("failed", False) else f"{item['pos']:.3f}"
            query_suffix = " | QUERY" if item.get("query") else ""
            caption = f"{'✋ ' if fname in overrides else ''}{score_text}{query_suffix} | {fname}"
            if item.get("base_pos") is not None:
                caption += f"  (shortlist {item['base_pos']:.3f})"
            if method == METHOD_LLMSEARCH:
                raw_response = (item.get("caption") or "").strip()
                if raw_response:
                    caption += f"  [LLM: {raw_response}]"
                elif item.get("reason"):
                    caption += f"  [LLM error: {item['reason']}]"
            entry = (item["path"], caption)
            if side == left_name:
                left.append(entry)
            else:
                right.append(entry)
    else:
        ordered = sorted(scores.items(), key=lambda x: -x[1]["score"])
        for fname, item in ordered:
            if item["score"] < -1000:
                continue
            side = overrides.get(fname)
            if side is None:
                side = left_name if item["score"] >= main_threshold else right_name
            caption = f"{'✋ ' if fname in overrides else ''}{item['score']:.3f} | {fname}"
            if item.get("base") is not None and item.get("penalty") is not None:
                caption += f"  (base {item['base']:.3f}, penalty {item['penalty']:.3f})"
            entry = (item["path"], caption)
            if side == left_name:
                left.append(entry)
            else:
                right.append(entry)
    return left, right
