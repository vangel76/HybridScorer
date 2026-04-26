import os
import io
import tempfile
from hashlib import sha256

from PIL import Image
from . import ui_compat as gr

from .utils import (
    normalize_folder_identity,
    get_promptmatch_proxy_cache_dir,
    clear_promptmatch_proxy_cache,
)


def is_browse_mode(state):
    return state.get("view_mode") == "browse"


def set_scored_mode(state):
    state["view_mode"] = "scored"


def set_browse_mode(state, items=None, status_text=""):
    state["view_mode"] = "browse"
    state["browse_items"] = list(items or [])
    state["browse_status"] = status_text


def remove_file_quietly(path):
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def clear_external_query_state(state, delete_file=True):
    existing_path = state.get("external_query_path")
    state["external_query_path"] = None
    state["external_query_label"] = None
    state["external_query_active"] = False
    state["external_query_signature"] = None
    if delete_file:
        remove_file_quietly(existing_path)


def save_external_query_image(state, upload_path):
    if not upload_path or not os.path.isfile(upload_path):
        raise RuntimeError("No query image was received.")

    with Image.open(upload_path) as src_img:
        image = src_img.convert("RGB")
        image.load()

    label = os.path.basename(upload_path) or "query-image.png"
    return save_external_query_image_bytes(state, image, label)


def save_external_query_image_bytes(state, image_or_bytes, label):
    if isinstance(image_or_bytes, Image.Image):
        image = image_or_bytes.convert("RGB")
    else:
        with Image.open(io.BytesIO(image_or_bytes)) as src_img:
            image = src_img.convert("RGB")
            image.load()

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    signature = sha256(image_bytes).hexdigest()
    label = (label or "query-image.png").strip() or "query-image.png"

    temp_handle = tempfile.NamedTemporaryFile(
        prefix="hybridscorer-query-",
        suffix=".png",
        delete=False,
    )
    temp_path = temp_handle.name
    temp_handle.close()
    try:
        with open(temp_path, "wb") as handle:
            handle.write(image_bytes)
    except Exception:
        remove_file_quietly(temp_path)
        raise

    clear_external_query_state(state, delete_file=True)
    state["external_query_path"] = temp_path
    state["external_query_label"] = label
    state["external_query_active"] = True
    state["external_query_signature"] = signature
    return temp_path, label


def get_preview_image_path(state):
    preview_fname = state.get("preview_fname")
    if not preview_fname:
        return None, None
    if is_browse_mode(state):
        for path, _ in state.get("browse_items", []):
            if os.path.basename(path) == preview_fname:
                return path, preview_fname
        return None, preview_fname
    item = state.get("scores", {}).get(preview_fname)
    if not item:
        return None, preview_fname
    return item.get("path"), preview_fname


def active_query_image_context(state):
    if state.get("external_query_active"):
        external_path = state.get("external_query_path")
        if external_path and os.path.isfile(external_path):
            external_label = state.get("external_query_label") or os.path.basename(external_path)
            external_sig = state.get("external_query_signature") or external_label
            return {
                "path": external_path,
                "label": external_label,
                "source_kind": "external",
                "source_label": "external query image",
                "cache_key": f"external::{external_sig}",
                "preview_fname": None,
            }
        clear_external_query_state(state, delete_file=True)

    image_path, preview_fname = get_preview_image_path(state)
    if image_path and os.path.isfile(image_path):
        return {
            "path": image_path,
            "label": preview_fname,
            "source_kind": "gallery",
            "source_label": "gallery preview",
            "cache_key": preview_fname,
            "preview_fname": preview_fname,
        }
    return {
        "path": None,
        "label": preview_fname,
        "source_kind": None,
        "source_label": None,
        "cache_key": preview_fname,
        "preview_fname": preview_fname,
    }


def active_query_image_widget_update(state):
    query_ctx = active_query_image_context(state)
    image_path = query_ctx.get("path")
    if image_path and os.path.isfile(image_path):
        return gr.update(value=image_path)
    return gr.update(value=None)


def clear_external_query_button_update(state):
    query_ctx = active_query_image_context(state)
    external_active = query_ctx.get("source_kind") == "external"
    return gr.update(
        interactive=external_active,
        value="Clear external override" if external_active else "Clear external override",
    )


def reset_selection_state(state):
    state["left_marked"] = []
    state["right_marked"] = []


def clear_preview_search_context(state):
    state["similarity_query_fname"] = None
    state["similarity_query_source"] = None
    state["similarity_model_label"] = None
    state["sameperson_query_fname"] = None
    state["sameperson_query_source"] = None
    state["sameperson_model_label"] = None


def clear_active_scores(state):
    state["scores"] = {}
    state["overrides"] = {}
    state["preview_fname"] = None
    reset_selection_state(state)


def reset_for_folder_change(state, folder, clear_overrides=True):
    state["source_dir"] = folder
    state["scores"] = {}
    if clear_overrides:
        state["overrides"] = {}
    state["preview_fname"] = None
    reset_selection_state(state)
    clear_preview_search_context(state)


def invalidate_folder_level_caches(state):
    state["proxy_signature"] = None
    state["proxy_map"] = {}
    state["pm_cached_signature"] = None
    state["pm_cached_model_label"] = None
    state["pm_cached_feature_paths"] = None
    state["pm_cached_image_features"] = None
    state["pm_cached_failed_paths"] = None
    state["face_cached_signature"] = None
    state["face_cached_feature_paths"] = None
    state["face_cached_embeddings"] = None
    state["face_cached_failures"] = None
    state["ir_cached_signature"] = None
    state["ir_cached_positive_prompt"] = None
    state["ir_cached_negative_prompt"] = None
    state["ir_cached_base_scores"] = None
    state["ir_cached_penalty_scores"] = None
    state["llmsearch_cached_signature"] = None
    state["llmsearch_cached_prompt"] = None
    state["llmsearch_cached_backend"] = None
    state["llmsearch_cached_scoring_mode"] = None
    state["llmsearch_cached_shortlist_size"] = None
    state["llmsearch_cached_model_label"] = None
    state["llmsearch_cached_scores"] = None
    state["llmsearch_cached_captions"] = {}
    state["tagmatch_cached_signature"] = None
    state["tagmatch_cached_feature_paths"] = None
    state["tagmatch_cached_tag_vectors"] = None


def preserve_overrides_for_image_paths(state, image_paths):
    folder = state.get("source_dir")
    previous_folder_key = normalize_folder_identity(folder) if folder else None
    available_names = {os.path.basename(path) for path in image_paths}
    return preserve_overrides_for_folder_key(state, previous_folder_key, available_names)


def preserve_overrides_for_folder_key(state, folder_key, available_names):
    previous_folder = state.get("source_dir")
    previous_folder_key = normalize_folder_identity(previous_folder) if previous_folder else None
    if previous_folder_key != folder_key:
        return {}
    return {
        fname: side
        for fname, side in state.get("overrides", {}).items()
        if fname in available_names
    }


def sync_promptmatch_proxy_cache(state, folder):
    folder_key = normalize_folder_identity(folder)
    if state["proxy_folder_key"] != folder_key:
        clear_promptmatch_proxy_cache(state.get("proxy_cache_dir"))
        state["proxy_folder_key"] = folder_key
        state["proxy_cache_dir"] = get_promptmatch_proxy_cache_dir(folder)
        invalidate_folder_level_caches(state)
    return state["proxy_cache_dir"]


def begin_scored_run(state, method, folder, preserved_overrides):
    state["method"] = method
    set_scored_mode(state)
    sync_promptmatch_proxy_cache(state, folder)
    state["source_dir"] = folder
    state["overrides"] = dict(preserved_overrides)
    state["preview_fname"] = None
    reset_selection_state(state)
    clear_preview_search_context(state)


def set_browse_folder_state(state, folder, browse_items, status_text):
    reset_for_folder_change(state, folder, clear_overrides=True)
    set_browse_mode(state, browse_items, status_text)


def can_reuse_proxy_map(state, image_paths, image_signature):
    proxy_map = state.get("proxy_map") or {}
    if state.get("proxy_signature") != image_signature or not proxy_map:
        return False
    return all(path in proxy_map for path in image_paths)


def remember_mode_thresholds(state, method, main_threshold, aux_threshold):
    if method not in state["mode_thresholds"]:
        state["mode_thresholds"][method] = {}
    state["mode_thresholds"][method]["main"] = float(main_threshold)
    state["mode_thresholds"][method]["aux"] = float(aux_threshold)


def recalled_mode_thresholds(state, method, default_main, default_aux):
    entry = state.get("mode_thresholds", {}).get(method) or {}
    main_value = entry.get("main")
    aux_value = entry.get("aux")
    has_saved = (main_value is not None) or (aux_value is not None)
    if main_value is None:
        main_value = default_main
    if aux_value is None:
        aux_value = default_aux
    return float(main_value), float(aux_value), bool(has_saved)
