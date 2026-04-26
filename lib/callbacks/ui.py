import os
import json
import shutil

from .. import ui_compat as gr

from ..config import (
    METHOD_PROMPTMATCH, METHOD_IMAGEREWARD, METHOD_LLMSEARCH, NEGATIVE_THRESHOLD,
)
from ..helpers import (
    build_split, method_labels, uses_similarity_topn, similarity_topn_defaults,
    threshold_for_percentile, promptmatch_slider_range, imagereward_slider_range,
    llmsearch_slider_range, clamp_threshold, expand_slider_bounds,
    slider_step_floor, slider_step_ceil_exclusive, sanitize_export_name,
    export_destination,
)
from ..state_helpers import is_browse_mode, active_query_image_widget_update
from .. import view as _vw
from .scoring import middle_threshold_values, recompute_imagereward_scores
from .prompts import select_cached_generated_prompt


def update_split(state, main_threshold, aux_threshold):
    return _vw.render_view_with_controls(state, main_threshold, aux_threshold)


def update_histogram_only(state, main_threshold, aux_threshold):
    return _vw.render_histogram(state, state["method"], state["scores"], main_threshold, aux_threshold)


def update_prompt_generator(state, generator_name, detail_level, current_generated_prompt):
    return select_cached_generated_prompt(state, generator_name, detail_level, current_generated_prompt)


def update_generated_prompt_detail(state, generator_name, detail_level, current_generated_prompt):
    return select_cached_generated_prompt(state, generator_name, detail_level, current_generated_prompt)


def update_proxy_display(state, use_proxy_display, main_threshold, aux_threshold):
    state["use_proxy_display"] = bool(use_proxy_display)
    return _vw.render_view_with_controls(state, main_threshold, aux_threshold)


def update_imagereward_penalty_weight(state, penalty_weight, main_threshold, aux_threshold):
    recomputed = recompute_imagereward_scores(state, penalty_weight)
    if not recomputed:
        return (
            *_vw.render_view_with_controls(state, main_threshold, aux_threshold),
            gr.update(),
            gr.update(),
        )

    lo, hi, _ = imagereward_slider_range(state["scores"])
    clamped = clamp_threshold(main_threshold, lo, hi)
    safe_lo, safe_hi = expand_slider_bounds(lo, hi, main_threshold, clamped)
    from ..helpers import threshold_labels
    main_label = threshold_labels(METHOD_IMAGEREWARD)[0]
    return (
        *_vw.render_view_with_controls(state, clamped, aux_threshold),
        gr.update(
            minimum=safe_lo,
            maximum=safe_hi,
            value=clamped,
            label=main_label,
        ),
        gr.update(value=NEGATIVE_THRESHOLD, visible=False),
    )


def handle_thumb_action(state, action, main_threshold, aux_threshold):
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
        return thumb_noop(active_query_image_widget_update(state))
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
            return with_slider_skips(move_right(state, main_threshold, aux_threshold, preview_override=fname))
        if action_id == "hy-move-left":
            return with_slider_skips(move_left(state, main_threshold, aux_threshold, preview_override=fname))
        if action_id == "hy-fit-threshold":
            return fit_threshold_to_targets(state, main_threshold, aux_threshold, preview_override=fname)
        return thumb_noop(active_query_image_widget_update(state))
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
        return with_slider_skips(_vw.render_view_with_controls(state, main_threshold, aux_threshold))
    parts = str(action).split(":")
    verb = parts[0] if parts else ""
    if is_browse_mode(state):
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
            return thumb_noop(active_query_image_widget_update(state))
        if is_browse_mode(state):
            state["preview_fname"] = fname
            return thumb_noop(active_query_image_widget_update(state))
        else:
            marked_key = "left_marked" if side == "left" else "right_marked"
            if fname in state[marked_key]:
                state[marked_key] = [name for name in state[marked_key] if name != fname]
            else:
                state[marked_key].append(fname)
    return with_slider_skips(_vw.render_view_with_controls(state, main_threshold, aux_threshold))


def handle_hist_width(state, width_value, main_threshold, aux_threshold):
    try:
        next_width = max(220, int(float(width_value)))
    except Exception:
        return gr.skip()
    if abs(next_width - int(state.get("hist_width", 300))) < 2:
        return gr.skip()
    state["hist_width"] = next_width
    return _vw.render_histogram(state, state["method"], state["scores"], main_threshold, aux_threshold)


def move_right(state, main_threshold, aux_threshold, preview_override=None):
    if is_browse_mode(state):
        return _vw.render_view_with_controls(state, main_threshold, aux_threshold)
    left_name, right_name, _, _ = method_labels(state["method"])
    left_targets, _ = _vw.active_targets(state, main_threshold, aux_threshold, preview_override=preview_override)
    targets = left_targets or list(state.get("left_marked", []))
    for fname in targets:
        state["overrides"][fname] = right_name
    state["left_marked"] = []
    state["right_marked"] = []
    return _vw.render_view_with_controls(state, main_threshold, aux_threshold)


def move_left(state, main_threshold, aux_threshold, preview_override=None):
    if is_browse_mode(state):
        return _vw.render_view_with_controls(state, main_threshold, aux_threshold)
    left_name, right_name, _, _ = method_labels(state["method"])
    _, right_targets = _vw.active_targets(state, main_threshold, aux_threshold, preview_override=preview_override)
    targets = right_targets or list(state.get("right_marked", []))
    for fname in targets:
        state["overrides"][fname] = left_name
    state["left_marked"] = []
    state["right_marked"] = []
    return _vw.render_view_with_controls(state, main_threshold, aux_threshold)


def pin_selected(state, main_threshold, aux_threshold, preview_override=None):
    if is_browse_mode(state):
        return _vw.render_view_with_controls(state, main_threshold, aux_threshold)
    left_name, right_name, _, _ = method_labels(state["method"])
    left_targets, right_targets = _vw.active_targets(state, main_threshold, aux_threshold, preview_override=preview_override)
    for fname in left_targets:
        state["overrides"][fname] = left_name
    for fname in right_targets:
        state["overrides"][fname] = right_name
    state["left_marked"] = []
    state["right_marked"] = []
    return _vw.render_view_with_controls(state, main_threshold, aux_threshold)


def clear_status(state, main_threshold, aux_threshold):
    if is_browse_mode(state):
        return _vw.render_view_with_controls(state, main_threshold, aux_threshold)
    for fname in set(state["left_marked"] + state["right_marked"]):
        state["overrides"].pop(fname, None)
    state["left_marked"] = []
    state["right_marked"] = []
    return _vw.render_view_with_controls(state, main_threshold, aux_threshold)


def clear_all_status(state, main_threshold, aux_threshold):
    if is_browse_mode(state):
        return _vw.render_view_with_controls(state, main_threshold, aux_threshold)
    state["overrides"].clear()
    state["left_marked"] = []
    state["right_marked"] = []
    return _vw.render_view_with_controls(state, main_threshold, aux_threshold)


def fit_threshold_to_targets(state, main_threshold, aux_threshold, preview_override=None):
    if is_browse_mode(state):
        return (*_vw.render_view_with_controls(state, main_threshold, aux_threshold), gr.update(), gr.update())
    left_targets, right_targets = _vw.active_targets(state, main_threshold, aux_threshold, preview_override=preview_override)
    targets = left_targets or right_targets
    if not targets or (left_targets and right_targets) or not state["scores"]:
        return (*_vw.render_view_with_controls(state, main_threshold, aux_threshold), gr.update(), gr.update())

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
        *_vw.render_view_with_controls(state, new_main, new_aux),
        gr.update(value=new_main),
        gr.update(value=new_aux),
    )


def set_from_percentile(state, percentile, main_threshold, aux_threshold):
    new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
    return (*_vw.render_view_with_controls(state, new_threshold, aux_threshold), gr.update(value=new_threshold))


def update_histogram_from_percentile(state, percentile, aux_threshold):
    new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
    return _vw.render_histogram(state, state["method"], state["scores"], new_threshold, aux_threshold)


def reset_main_threshold_to_middle(state, main_threshold, aux_threshold):
    new_main, _, _ = middle_threshold_values(state, state["method"])
    return (
        *_vw.render_view_with_controls(state, new_main, aux_threshold),
        gr.update(value=new_main),
        gr.update(),
    )


def reset_aux_threshold_to_middle(state, main_threshold, aux_threshold):
    if state["method"] != METHOD_PROMPTMATCH:
        return (
            *_vw.render_view_with_controls(state, main_threshold, aux_threshold),
            gr.update(),
            gr.update(),
        )
    _, new_aux, _ = middle_threshold_values(state, METHOD_PROMPTMATCH)
    return (
        *_vw.render_view_with_controls(state, main_threshold, new_aux),
        gr.update(),
        gr.update(value=new_aux),
    )


def reset_percentile_to_middle(state, main_threshold, aux_threshold):
    if uses_similarity_topn(state["method"]):
        _, percentile = similarity_topn_defaults(state["scores"])
    else:
        percentile = 50
    if state["scores"]:
        new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
    else:
        new_threshold = float(main_threshold)
    return (
        *_vw.render_view_with_controls(state, new_threshold, aux_threshold),
        gr.update(value=new_threshold),
        gr.update(value=percentile),
    )


def update_zoom(state, zoom_value, main_threshold, aux_threshold):
    try:
        slider_value = max(2, min(10, int(zoom_value)))
        state["zoom_columns"] = 12 - slider_value
    except Exception:
        state["zoom_columns"] = 5
    return _vw.render_view_with_controls(state, main_threshold, aux_threshold)


def on_hist_click(state, sel: gr.SelectData, main_threshold, aux_threshold):
    geom = state.get("hist_geom")
    if not geom:
        return (*_vw.render_view_with_controls(state, main_threshold, aux_threshold), gr.update(), gr.update())
    try:
        cx, cy = sel.index
        from ..helpers import uses_pos_similarity_scores
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
                if geom.get("pos_flipped"):
                    val = hi - ((cx - PAD_L) / cW) * (hi - lo)
                else:
                    val = lo + ((cx - PAD_L) / cW) * (hi - lo)
                main_threshold = round(max(lo, min(hi, val)), 3)
            elif geom["has_neg"] and y0neg <= cy <= y0neg + CH:
                lo, hi = geom["neg_lo"], geom["neg_hi"]
                val = lo + ((cx - PAD_L) / cW) * (hi - lo)
                aux_threshold = round(max(lo, min(hi, val)), 3)
            else:
                return (*_vw.render_view_with_controls(state, main_threshold, aux_threshold), gr.update(), gr.update())
        else:
            PAD_L = geom["PAD_L"]
            PAD_TOP = geom["PAD_TOP"]
            PAD_BOT = geom["PAD_BOT"]
            H = geom["H"]
            cW = geom["cW"]
            lo, hi = geom["lo"], geom["hi"]
            if PAD_TOP <= cy <= PAD_TOP + H - PAD_BOT:
                if geom.get("flipped"):
                    val = hi - ((cx - PAD_L) / cW) * (hi - lo)
                else:
                    val = lo + ((cx - PAD_L) / cW) * (hi - lo)
                main_threshold = round(max(lo, min(hi, val)), 3)
            else:
                return (*_vw.render_view_with_controls(state, main_threshold, aux_threshold), gr.update(), gr.update())
    except Exception:
        return (*_vw.render_view_with_controls(state, main_threshold, aux_threshold), gr.update(), gr.update())

    return (*_vw.render_view_with_controls(state, main_threshold, aux_threshold), gr.update(value=main_threshold), gr.update(value=aux_threshold))


def export_files(state, main_threshold, aux_threshold, export_left_enabled, export_right_enabled, export_move_enabled, export_left_name, export_right_name):
    if is_browse_mode(state):
        return _vw.render_view_with_controls(state, main_threshold, aux_threshold)
    left_items, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)
    left_name, right_name, left_dirname, right_dirname = method_labels(state["method"])
    base = state["source_dir"]
    targets = []
    if export_left_enabled:
        targets.append((left_name, sanitize_export_name(export_left_name) or left_dirname, left_items))
    if export_right_enabled:
        targets.append((right_name, sanitize_export_name(export_right_name) or right_dirname, right_items))
    if not targets:
        return _vw.render_view_with_controls(state, main_threshold, aux_threshold)
    target_names = [folder_name for _, folder_name, _ in targets]
    if len(set(target_names)) != len(target_names):
        return _vw.render_view_with_controls(state, main_threshold, aux_threshold)

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
            from ..state_helpers import clear_preview_search_context
            clear_preview_search_context(state)
    return _vw.render_view_with_controls(state, main_threshold, aux_threshold)
