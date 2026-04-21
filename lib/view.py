import os
import json

import gradio as gr
from PIL import Image, ImageDraw

from .config import (
    METHOD_PROMPTMATCH,
    METHOD_LLMSEARCH,
    METHOD_SIMILARITY,
    METHOD_SAMEPERSON,
    METHOD_TAGMATCH,
    HIST_HEIGHT_SCALE,
    IMAGEREWARD_THRESHOLD,
    IMAGEREWARD_SLIDER_MIN,
    IMAGEREWARD_SLIDER_MAX,
    PROMPTMATCH_SLIDER_MIN,
    PROMPTMATCH_SLIDER_MAX,
    NEGATIVE_THRESHOLD,
    TAGMATCH_DEFAULT_THRESHOLD,
    TAGMATCH_SLIDER_PREPROCESS_MIN,
    TAGMATCH_SLIDER_MAX,
    FACE_MODEL_LABEL,
    SCORE_FOLDER_OUTPUT_KEYS,
    PREVIEW_SEARCH_OUTPUT_KEYS,
    VIEW_WITH_CONTROLS_OUTPUT_KEYS,
)
from .helpers import (
    threshold_labels,
    uses_pos_similarity_scores,
    build_split,
    status_line,
    normalize_threshold_inputs,
    percentile_slider_update,
    percentile_reset_button_update,
    promptmatch_model_status_json,
    llmsearch_uses_numeric_scores,
)
from .state_helpers import (
    is_browse_mode,
    active_query_image_widget_update,
    clear_external_query_button_update,
    remember_mode_thresholds,
    set_scored_mode,
)


def gallery_update(items, columns=None):
    update_kwargs = {"value": items, "selected_index": None}
    if columns is not None:
        update_kwargs["columns"] = columns
    return gr.update(**update_kwargs)


def gallery_display_items(state, items):
    if not state.get("use_proxy_display", True):
        return items

    proxy_map = state.get("proxy_map", {})
    displayed = []
    for original_path, caption in items:
        display_path = proxy_map.get(original_path, original_path)
        if not os.path.isfile(display_path):
            display_path = original_path
        displayed.append((display_path, caption))
    return displayed


def ui_visibility_updates(state):
    browse_mode = is_browse_mode(state)
    return (
        gr.update(visible=not browse_mode),
        gr.update(visible=not browse_mode),
        gr.update(visible=not browse_mode),
        gr.update(visible=not browse_mode),
        gr.update(visible=not browse_mode),
        gr.update(visible=not browse_mode),
        gr.update(visible=not browse_mode),
    )


def selection_info(state):
    left_count = len(state.get("left_marked", []))
    right_count = len(state.get("right_marked", []))
    if not left_count and not right_count:
        return "Shift+click to mark multiple, or drag & drop."
    return f"Marked: **{left_count}** in SELECTED, **{right_count}** in REJECTED"


def marked_state_json(state, visible_fnames=None):
    score_lookup = {}
    media_lookup = {}
    left_order = []
    right_order = []
    visible_names = set(visible_fnames or [])
    for original_path, _ in state.get("browse_items", []):
        original_base = os.path.basename(original_path)
        if visible_names and original_base not in visible_names:
            continue
        media_lookup[original_base] = original_base
        media_lookup[original_path] = original_base
        display_path = state.get("proxy_map", {}).get(original_path, original_path)
        display_base = os.path.basename(display_path)
        media_lookup[display_base] = original_base
        media_lookup[display_path] = original_base
    for fname, item in state.get("scores", {}).items():
        if visible_names and fname not in visible_names:
            continue
        original_path = item.get("path")
        if original_path:
            original_base = os.path.basename(original_path)
            media_lookup[original_base] = fname
            media_lookup[original_path] = fname
            display_path = state.get("proxy_map", {}).get(original_path, original_path)
            display_base = os.path.basename(display_path)
            media_lookup[display_base] = fname
            media_lookup[display_path] = fname
        if uses_pos_similarity_scores(state["method"]):
            if item.get("failed", False) or "pos" not in item:
                continue
            score_lookup[fname] = {
                "main": round(float(item["pos"]), 6),
                "neg": round(float(item["neg"]), 6) if item.get("neg") is not None else None,
            }
        else:
            if item.get("score", -1001) <= -1000:
                continue
            score_lookup[fname] = {
                "main": round(float(item["score"]), 6),
                "neg": None,
            }
    tag_score_lookup = {}
    if state.get("method") == METHOD_TAGMATCH:
        tag_vectors = state.get("tagmatch_cached_tag_vectors") or {}
        raw_str = state.get("tagmatch_last_query_tags_str") or ""
        q_tags = [t.strip().lower() for t in raw_str.split(",") if t.strip()]
        if q_tags and tag_vectors:
            for original_path, _ in state.get("browse_items", []):
                fname = os.path.basename(original_path)
                tv = tag_vectors.get(original_path, {})
                if tv:
                    tag_score_lookup[fname] = {t: round(tv.get(t, 0.0), 4) for t in q_tags}
    segment_score_lookup = {}
    neg_segment_score_lookup = {}
    if state.get("method") == METHOD_PROMPTMATCH and state.get("pm_segment_mode"):
        for fname, item in state.get("scores", {}).items():
            seg = item.get("segment_scores")
            if seg:
                segment_score_lookup[fname] = {k: round(v, 4) for k, v in seg.items()}
            neg_seg = item.get("neg_segment_scores")
            if neg_seg:
                neg_segment_score_lookup[fname] = {k: round(v, 4) for k, v in neg_seg.items()}
    return json.dumps({
        "left": state.get("left_marked", []),
        "right": state.get("right_marked", []),
        "held": list(state.get("overrides", {}).keys()),
        "preview": state.get("preview_fname"),
        "hist_geom": state.get("hist_geom"),
        "score_lookup": score_lookup,
        "tag_score_lookup": tag_score_lookup,
        "segment_score_lookup": segment_score_lookup,
        "neg_segment_score_lookup": neg_segment_score_lookup,
        "media_lookup": media_lookup,
        "left_order": left_order,
        "right_order": right_order,
    })


def active_targets(state, main_threshold, aux_threshold, preview_override=None):
    left_items, right_items = build_split(
        state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold
    )
    left_names = {os.path.basename(path) for path, _ in left_items}
    right_names = {os.path.basename(path) for path, _ in right_items}
    left_marked = [name for name in state.get("left_marked", []) if name in left_names]
    right_marked = [name for name in state.get("right_marked", []) if name in right_names]
    if left_marked or right_marked:
        return left_marked, right_marked

    preview_fname = preview_override or state.get("preview_fname")
    if preview_fname in left_names:
        return [preview_fname], []
    if preview_fname in right_names:
        return [], [preview_fname]
    return [], []


def render_histogram(state, method, scores, main_threshold, aux_threshold):
    # Draw a lightweight PIL histogram image instead of depending on a plotting library.
    if not scores:
        state["hist_geom"] = None
        return None
    _, _, main_hist_label, aux_hist_label = threshold_labels(method)
    try:
        hist_width = max(220, int(state.get("hist_width", 300)))
    except Exception:
        hist_width = 300

    def draw_axis_labels(draw, left_x, center_x, right_x, y, lo, hi):
        labels = (
            (left_x, f"{lo:.3f}", "left"),
            (center_x, f"{((lo + hi) / 2):.3f}", "center"),
            (right_x, f"{hi:.3f}", "right"),
        )
        for x, text, align in labels:
            bbox = draw.textbbox((0, 0), text)
            width = bbox[2] - bbox[0]
            if align == "center":
                tx = x - (width / 2)
            elif align == "right":
                tx = x - width
            else:
                tx = x
            draw.text((int(tx), y), text, fill="#667755")

    if uses_pos_similarity_scores(method):
        if not all("pos" in item for item in scores.values()):
            state["hist_geom"] = None
            return None
        pos_vals = [v["pos"] for v in scores.values() if not v.get("failed", False)]
        neg_vals = [v["neg"] for v in scores.values() if not v.get("failed", False) and v.get("neg") is not None]
        has_neg = bool(neg_vals)
        if not pos_vals:
            state["hist_geom"] = None
            return None

        def _bins(vals, n=32):
            if not vals:
                return [], 0.0, 1.0
            lo, hi = min(vals), max(vals)
            if lo == hi:
                lo -= 0.005
                hi += 0.005
            width = (hi - lo) / n
            counts = [0] * n
            for val in vals:
                counts[min(int((val - lo) / width), n - 1)] += 1
            return counts, lo, hi

        pos_counts, pos_lo, pos_hi = _bins(pos_vals)
        neg_counts, neg_lo, neg_hi = _bins(neg_vals)
        W, CH = hist_width, max(60, int(130 * HIST_HEIGHT_SCALE))
        PAD_L, PAD_R = 12, 12
        PAD_TOP, PAD_BOT = max(10, int(18 * HIST_HEIGHT_SCALE)), max(14, int(22 * HIST_HEIGHT_SCALE))
        GAP = max(16, int(28 * HIST_HEIGHT_SCALE))
        n_ch = 2 if has_neg else 1
        H = PAD_TOP + n_ch * CH + (n_ch - 1) * GAP + PAD_BOT
        img = Image.new("RGB", (W, H), "#0d0d11")
        draw = ImageDraw.Draw(img)

        def draw_chart(y0, counts, lo, hi, threshold, bar_rgb, line_rgb, label, left_tint=None, right_tint=None):
            cW = W - PAD_L - PAD_R
            max_c = max(counts) if counts else 1
            bw = cW / len(counts)
            draw.rectangle([PAD_L, y0, W - PAD_R, y0 + CH], fill="#0f0f16")
            tx = PAD_L + int(((threshold - lo) / (hi - lo)) * cW)
            tx = max(PAD_L, min(W - PAD_R, tx))
            if left_tint and tx > PAD_L:
                draw.rectangle([PAD_L, y0, tx, y0 + CH], fill=left_tint)
            if right_tint and tx < (W - PAD_R):
                draw.rectangle([tx, y0, W - PAD_R, y0 + CH], fill=right_tint)
            for i, count in enumerate(counts):
                if count == 0:
                    continue
                bh = max(1, int((count / max_c) * (CH - 2)))
                x0 = PAD_L + int(i * bw) + 1
                x1 = PAD_L + int((i + 1) * bw) - 1
                draw.rectangle([x0, y0 + CH - bh, x1, y0 + CH], fill=bar_rgb)
            for yy in range(y0, y0 + CH, 6):
                draw.line([(tx, yy), (tx, min(yy + 3, y0 + CH))], fill=line_rgb, width=2)
            draw_axis_labels(draw, PAD_L, PAD_L + (cW / 2), W - PAD_R, y0 + CH + 4, lo, hi)
            draw.text((PAD_L, y0 - 14), f"{label} threshold: {threshold:.3f}", fill="#99bb88")

        draw_chart(
            PAD_TOP,
            pos_counts,
            pos_lo,
            pos_hi,
            main_threshold,
            "#3a7a3a",
            "#aadd66",
            main_hist_label,
            left_tint="#241416",
            right_tint="#142418",
        )
        if has_neg:
            draw_chart(
                PAD_TOP + CH + GAP,
                neg_counts,
                neg_lo,
                neg_hi,
                aux_threshold,
                "#7a3a3a",
                "#dd6644",
                aux_hist_label,
                left_tint="#142418",
                right_tint="#241416",
            )

        state["hist_geom"] = {
            "method": method,
            "W": W,
            "H": H,
            "PAD_L": PAD_L,
            "PAD_R": PAD_R,
            "PAD_TOP": PAD_TOP,
            "CH": CH,
            "GAP": GAP,
            "has_neg": has_neg,
            "pos_lo": pos_lo,
            "pos_hi": pos_hi,
            "neg_lo": neg_lo,
            "neg_hi": neg_hi,
        }
        return img

    if not all("score" in item for item in scores.values()):
        state["hist_geom"] = None
        return None
    vals = [v["score"] for v in scores.values() if v["score"] > -1000]
    if not vals:
        state["hist_geom"] = None
        return None
    lo, hi = min(vals), max(vals)
    if lo == hi:
        lo -= 0.05
        hi += 0.05
    bins = 32
    width = (hi - lo) / bins
    counts = [0] * bins
    for val in vals:
        counts[min(int((val - lo) / width), bins - 1)] += 1
    W, H = hist_width, max(70, int(130 * HIST_HEIGHT_SCALE))
    PAD_L, PAD_R = 12, 12
    PAD_TOP, PAD_BOT = max(10, int(18 * HIST_HEIGHT_SCALE)), max(14, int(22 * HIST_HEIGHT_SCALE))
    cW = W - PAD_L - PAD_R
    img = Image.new("RGB", (W, H), "#0d0d11")
    draw = ImageDraw.Draw(img)
    draw.rectangle([PAD_L, PAD_TOP, W - PAD_R, PAD_TOP + H - PAD_BOT], fill="#0f0f16")
    tx = PAD_L + int(((main_threshold - lo) / (hi - lo)) * cW)
    tx = max(PAD_L, min(W - PAD_R, tx))
    if tx > PAD_L:
        draw.rectangle([PAD_L, PAD_TOP, tx, PAD_TOP + H - PAD_BOT], fill="#241416")
    if tx < (W - PAD_R):
        draw.rectangle([tx, PAD_TOP, W - PAD_R, PAD_TOP + H - PAD_BOT], fill="#142418")
    max_c = max(counts) if counts else 1
    bw = cW / bins
    for i, count in enumerate(counts):
        if count == 0:
            continue
        bh = max(1, int((count / max_c) * (H - PAD_BOT - PAD_TOP - 2)))
        x0 = PAD_L + int(i * bw) + 1
        x1 = PAD_L + int((i + 1) * bw) - 1
        draw.rectangle([x0, PAD_TOP + (H - PAD_BOT - bh), x1, PAD_TOP + H - PAD_BOT], fill="#3a7a3a")
    for yy in range(PAD_TOP, PAD_TOP + H - PAD_BOT, 6):
        draw.line([(tx, yy), (tx, min(yy + 3, PAD_TOP + H - PAD_BOT))], fill="#aadd66", width=2)
    draw_axis_labels(draw, PAD_L, PAD_L + (cW / 2), W - PAD_R, PAD_TOP + H - PAD_BOT + 4, lo, hi)
    draw.text((PAD_L, PAD_TOP - 14), f"{main_hist_label}: {main_threshold:.3f}", fill="#99bb88")
    state["hist_geom"] = {
        "method": method,
        "W": W,
        "H": H,
        "PAD_L": PAD_L,
        "PAD_R": PAD_R,
        "PAD_TOP": PAD_TOP,
        "PAD_BOT": PAD_BOT,
        "cW": cW,
        "lo": lo,
        "hi": hi,
    }
    return img


def current_view(state, main_threshold, aux_threshold):
    # Single place that rebuilds gallery contents, status text, histogram, and marked-state JSON.
    zoom_columns = int(state.get("zoom_columns", 5))
    if is_browse_mode(state):
        browse_items = list(state.get("browse_items", []))
        left_names = {os.path.basename(path) for path, _ in browse_items}
        left_order = [os.path.basename(path) for path, _ in browse_items]
        state["left_marked"] = []
        state["right_marked"] = []
        state["hist_geom"] = None
        status = state.get("browse_status") or "Unscored browse mode. Preview a gallery image or use the external query image to search or generate a prompt."
        return (
            f"**UNSCORED**  •  **{len(browse_items)} images**",
            gallery_update(gallery_display_items(state, browse_items), columns=zoom_columns),
            "",
            gallery_update([], columns=zoom_columns),
            status,
            None,
            "Unscored browse mode. Preview a gallery image or use the external query image to search or generate a prompt.",
            json.dumps({
                **json.loads(marked_state_json(state, left_names)),
                "left_order": left_order,
                "right_order": [],
            }),
            active_query_image_widget_update(state),
            clear_external_query_button_update(state),
        )

    main_threshold, aux_threshold = normalize_threshold_inputs(
        state["method"],
        main_threshold,
        aux_threshold,
        state.get("llmsearch_backend"),
    )
    remember_mode_thresholds(state, state["method"], main_threshold, aux_threshold)
    left_items, right_items = build_split(
        state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold
    )
    left_names = {os.path.basename(path) for path, _ in left_items}
    right_names = {os.path.basename(path) for path, _ in right_items}
    left_order = [os.path.basename(path) for path, _ in left_items]
    right_order = [os.path.basename(path) for path, _ in right_items]
    state["left_marked"] = [name for name in state.get("left_marked", []) if name in left_names]
    state["right_marked"] = [name for name in state.get("right_marked", []) if name in right_names]
    visible_names = left_names | right_names
    status = status_line(state["method"], left_items, right_items, state["scores"], state["overrides"])
    if state["method"] == METHOD_LLMSEARCH and state.get("llmsearch_backend"):
        backend_label = state.get("llmsearch_backend") or ""
        status = f"LLM rerank via {backend_label}  •  {status}"
    if state["method"] == METHOD_SIMILARITY and state.get("similarity_query_fname"):
        model_label = state.get("similarity_model_label") or "PromptMatch model"
        query_source = state.get("similarity_query_source") or "gallery preview"
        status = f"Similarity from {query_source} ({state['similarity_query_fname']}) via {model_label}  •  {status}"
    elif state["method"] == METHOD_SAMEPERSON and state.get("sameperson_query_fname"):
        model_label = state.get("sameperson_model_label") or FACE_MODEL_LABEL
        query_source = state.get("sameperson_query_source") or "gallery preview"
        status = f"Same person from {query_source} ({state['sameperson_query_fname']}) via {model_label}  •  {status}"
    return (
        f"**{len(left_items)} images**",
        gallery_update(gallery_display_items(state, left_items), columns=zoom_columns),
        f"**{len(right_items)} images**",
        gallery_update(gallery_display_items(state, right_items), columns=zoom_columns),
        status,
        render_histogram(state, state["method"], state["scores"], main_threshold, aux_threshold),
        selection_info(state),
        json.dumps({
            **json.loads(marked_state_json(state, visible_names)),
            "left_order": left_order,
            "right_order": right_order,
        }),
        active_query_image_widget_update(state),
        clear_external_query_button_update(state),
    )


def configure_controls(state, method):
    main_label, aux_label, _, _ = threshold_labels(method)
    if method == METHOD_PROMPTMATCH:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(label=main_label, value=0.14, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
            gr.update(label=aux_label, visible=True, value=NEGATIVE_THRESHOLD, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            percentile_slider_update(method),
            percentile_reset_button_update(method),
            gr.update(value="PromptMatch sorts by text-image similarity. Use a positive prompt and optional negative prompt. Fragment weights like (blonde:1.2) are supported."),
        )
    if method == METHOD_LLMSEARCH:
        llm_numeric = llmsearch_uses_numeric_scores(state.get("llmsearch_backend"))
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(
                label=main_label,
                value=50.0 if llm_numeric else 0.14,
                minimum=0.0 if llm_numeric else PROMPTMATCH_SLIDER_MIN,
                maximum=100.0 if llm_numeric else PROMPTMATCH_SLIDER_MAX,
            ),
            gr.update(
                visible=False,
                value=50.0 if llm_numeric else NEGATIVE_THRESHOLD,
                label=aux_label,
                minimum=0.0 if llm_numeric else PROMPTMATCH_SLIDER_MIN,
                maximum=100.0 if llm_numeric else PROMPTMATCH_SLIDER_MAX,
            ),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            percentile_slider_update(method),
            percentile_reset_button_update(method),
            gr.update(value="LM Search uses PromptMatch to shortlist likely matches, then reranks the top candidates with a local vision-language model."),
        )
    if method == METHOD_SIMILARITY:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(label=main_label, value=0.5, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
            gr.update(visible=False, value=NEGATIVE_THRESHOLD, label=aux_label, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            percentile_slider_update(method),
            percentile_reset_button_update(method),
            gr.update(value="Similarity search ranks the current folder by image-image similarity using the active PromptMatch model."),
        )
    if method == METHOD_SAMEPERSON:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(label=main_label, value=0.5, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
            gr.update(visible=False, value=NEGATIVE_THRESHOLD, label=aux_label, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            percentile_slider_update(method),
            percentile_reset_button_update(method),
            gr.update(value=f"Same-person search ranks the current folder by face identity similarity using {FACE_MODEL_LABEL}."),
        )
    if method == METHOD_TAGMATCH:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(label=main_label, value=TAGMATCH_DEFAULT_THRESHOLD, minimum=TAGMATCH_SLIDER_PREPROCESS_MIN, maximum=TAGMATCH_SLIDER_MAX),
            gr.update(visible=False, value=NEGATIVE_THRESHOLD, label=aux_label, minimum=TAGMATCH_SLIDER_PREPROCESS_MIN, maximum=TAGMATCH_SLIDER_MAX),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            percentile_slider_update(method),
            percentile_reset_button_update(method),
            gr.update(value="TagMatch scores images by WD tagger confidence for the specified tags. Use comma-separated booru-style tags from https://aibooru.online/wiki_pages"),
        )
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(label=main_label, value=IMAGEREWARD_THRESHOLD, minimum=IMAGEREWARD_SLIDER_MIN, maximum=IMAGEREWARD_SLIDER_MAX),
        gr.update(visible=False, value=NEGATIVE_THRESHOLD),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        percentile_slider_update(method),
        percentile_reset_button_update(method),
        gr.update(value="ImageReward sorts by aesthetic preference. Optional penalty prompt subtracts a second style score."),
    )


def build_scored_callback_result(state, view_outputs, main_upd, aux_upd, percentile_upd, percentile_mid_upd):
    result = (
        *view_outputs,
        main_upd,
        aux_upd,
        percentile_upd,
        percentile_mid_upd,
        promptmatch_model_status_json(),
        *ui_visibility_updates(state),
    )
    assert len(result) == len(SCORE_FOLDER_OUTPUT_KEYS), "score-folder callback output arity changed"
    return result


def build_preview_search_result(prompt_status, score_outputs):
    result = (gr.update(value=prompt_status), *score_outputs)
    assert len(result) == len(PREVIEW_SEARCH_OUTPUT_KEYS), "preview-search callback output arity changed"
    return result


def status_with_current_view(state, message, main_threshold, aux_threshold):
    view = current_view(state, main_threshold, aux_threshold)
    return build_scored_callback_result(
        state,
        (
            view[0],
            view[1],
            view[2],
            view[3],
            gr.update(value=message),
            view[5],
            view[6],
            view[7],
            view[8],
            view[9],
        ),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def empty_result(state, message, method):
    _, _, _, _, main_upd, aux_upd, _, _, _, percentile_upd, percentile_mid_upd, _ = configure_controls(state, method)
    set_scored_mode(state)
    state["browse_items"] = []
    state["browse_status"] = ""
    return build_scored_callback_result(
        state,
        (
            "### LEFT",
            gallery_update([]),
            "### RIGHT",
            gallery_update([]),
            message,
            None,
            selection_info(state),
            marked_state_json(state),
            active_query_image_widget_update(state),
            clear_external_query_button_update(state),
        ),
        main_upd,
        aux_upd,
        percentile_upd,
        percentile_mid_upd,
    )


def render_view_with_controls(state, main_threshold, aux_threshold):
    result = (*current_view(state, main_threshold, aux_threshold), *ui_visibility_updates(state))
    assert len(result) == len(VIEW_WITH_CONTROLS_OUTPUT_KEYS), "view-with-controls callback output arity changed"
    return result
