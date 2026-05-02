import base64
import io
import json
import mimetypes
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256

from fastapi import UploadFile
from fastapi.responses import FileResponse, Response

from . import ui_compat as gr
from .config import (
    DEFAULT_GENERATED_PROMPT_DETAIL,
    DEFAULT_IR_NEGATIVE_PROMPT,
    DEFAULT_IR_PENALTY_WEIGHT,
    DEFAULT_LLMSEARCH_BACKEND,
    PROMPT_GENERATOR_CHOICES,
    PROMPT_GENERATOR_ALL_CHOICES,
    DEFAULT_PROMPT_GENERATOR,
    INPUT_FOLDER_NAME,
    IR_PROMPT,
    LLMSEARCH_DEFAULT_PROMPT,
    LLMSEARCH_SHORTLIST_DEFAULT,
    METHOD_IMAGEREWARD,
    METHOD_LLMSEARCH,
    METHOD_OBJECTSEARCH,
    METHOD_PROMPTMATCH,
    METHOD_SAMEPERSON,
    METHOD_SIMILARITY,
    METHOD_TAGMATCH,
    NEGATIVE_PROMPT,
    NEGATIVE_THRESHOLD,
    SEARCH_PROMPT,
    TAGMATCH_DEFAULT_TAGS,
)
from .helpers import (
    get_allowed_paths,
    label_for_backend,
    prompt_backend_dropdown_choices,
    promptmatch_model_dropdown_choices,
    promptmatch_model_status_json,
)
from .state import get_state_defaults
from .utils import (
    configure_torch_cpu_threads,
    require_cuda,
    runtime_requirement_issues,
)
from . import view as _vw
from .callbacks import prompts as _pr
from .callbacks import scoring as _sc
from .callbacks import ui as _ui
from . import loaders as _lo


def _is_update(value):
    return isinstance(value, gr.Update)


def _merge_update(current, update):
    if update is gr.SKIP:
        return current
    if _is_update(update):
        if "value" in update:
            return update["value"]
        return current
    return update


def _plain_update(update):
    return dict(update) if _is_update(update) else {}


@dataclass
class JobState:
    id: str
    action: str
    status: str = "running"
    progress: float = 0.0
    message: str = ""
    result: dict | None = None
    error: str | None = None
    events: queue.Queue = field(default_factory=queue.Queue)

    def emit(self, payload):
        self.events.put(payload)


class ProgressReporter:
    def __init__(self, job):
        self.job = job

    def __call__(self, value=0.0, desc=None):
        try:
            fraction = float(value)
        except Exception:
            fraction = 0.0
        fraction = max(0.0, min(1.0, fraction))
        message = str(desc or self.job.message or "")
        self.job.progress = fraction
        self.job.message = message
        self.job.emit({
            "type": "progress",
            "job": {
                "id": self.job.id,
                "action": self.job.action,
                "status": self.job.status,
                "progress": self.job.progress,
                "message": self.job.message,
            },
        })


class MediaRegistry:
    def __init__(self):
        self._paths = {}
        self._bytes = {}
        self._lock = threading.Lock()

    def register_path(self, path):
        if not path or not os.path.isfile(path):
            return None
        real = os.path.realpath(path)
        try:
            stat = os.stat(real)
            key_src = f"path:{real}:{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            return None
        media_id = sha256(key_src.encode("utf-8")).hexdigest()[:24]
        with self._lock:
            self._paths[media_id] = real
        return f"/media/{media_id}"

    def register_pil(self, image):
        if image is None:
            return None
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        media_id = sha256(b"hist:" + data + str(time.time_ns()).encode("ascii")).hexdigest()[:24]
        with self._lock:
            self._bytes[media_id] = data
        return f"/media/{media_id}"

    def response(self, media_id):
        with self._lock:
            path = self._paths.get(media_id)
            data = self._bytes.get(media_id)
        if data is not None:
            return Response(data, media_type="image/png")
        if path and os.path.isfile(path):
            media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            return FileResponse(path, media_type=media_type)
        return None


class HybridScorerContext:
    def __init__(self, script_dir, app_name, app_version, changelog_html):
        require_cuda()
        configure_torch_cpu_threads()
        self.script_dir = script_dir
        self.app_name = app_name
        self.app_version = app_version
        self.changelog_html = changelog_html
        self.device = "cuda"
        source_dir = os.path.join(script_dir, INPUT_FOLDER_NAME)
        if not os.path.isdir(source_dir):
            source_dir = script_dir
        self.prompt_backend = None
        self.state = get_state_defaults(source_dir, self.prompt_backend)
        self.media = MediaRegistry()
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs = {}
        self.inputs = self._default_inputs(source_dir)

    def _default_inputs(self, source_dir):
        model_label = label_for_backend(self.prompt_backend)
        return {
            "method": METHOD_PROMPTMATCH,
            "folder": source_dir,
            "model_label": model_label,
            "pos_prompt": SEARCH_PROMPT,
            "neg_prompt": NEGATIVE_PROMPT,
            "pm_segment_mode": False,
            "ir_prompt": IR_PROMPT,
            "ir_negative_prompt": DEFAULT_IR_NEGATIVE_PROMPT,
            "ir_penalty_weight": DEFAULT_IR_PENALTY_WEIGHT,
            "llm_model_label": model_label,
            "llm_prompt": LLMSEARCH_DEFAULT_PROMPT,
            "llm_backend_id": DEFAULT_LLMSEARCH_BACKEND,
            "llm_shortlist_size": LLMSEARCH_SHORTLIST_DEFAULT,
            "tagmatch_tags": TAGMATCH_DEFAULT_TAGS,
            "main_threshold": 0.14,
            "aux_threshold": NEGATIVE_THRESHOLD,
            "percentile": 50,
            "keep_pm_thresholds": True,
            "keep_ir_thresholds": True,
            "prompt_generator": DEFAULT_PROMPT_GENERATOR,
            "generated_prompt": "",
            "generated_prompt_detail": DEFAULT_GENERATED_PROMPT_DETAIL,
            "proxy_display": True,
            "zoom": 7,
            "export_left_enabled": True,
            "export_right_enabled": True,
            "export_move_enabled": False,
            "export_left_name": "selected",
            "export_right_name": "rejected",
        }

    def allowed_paths(self):
        source_dir = os.path.join(self.script_dir, INPUT_FOLDER_NAME)
        return get_allowed_paths(self.script_dir, source_dir)

    def create_job(self, action, fn):
        job = JobState(id=uuid.uuid4().hex, action=action)
        self.jobs[job.id] = job

        def runner():
            try:
                with self.lock:
                    result = fn(ProgressReporter(job))
                    if result is not None:
                        self._apply_known_result(action, result)
                    job.result = self.to_payload()
                job.status = "done"
                job.progress = 1.0
                job.emit({"type": "done", "job": self.job_payload(job), "state": job.result})
            except Exception as exc:
                job.status = "error"
                job.error = str(exc)
                job.emit({"type": "error", "job": self.job_payload(job), "error": job.error})

        self.executor.submit(runner)
        return job

    def job_payload(self, job):
        return {
            "id": job.id,
            "action": job.action,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "error": job.error,
        }

    def job_status_payload(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            return None
        payload = {"job": self.job_payload(job)}
        if job.status == "done" and job.result is not None:
            payload["state"] = job.result
        if job.status == "error":
            payload["error"] = job.error or "Job failed"
        return payload

    _SCORE_KEYS = (
        "left_head", "left_gallery", "right_head", "right_gallery", "status_md",
        "hist_plot", "sel_info", "mark_state", "external_query_image",
        "clear_external_query_btn", "main_slider", "aux_slider",
        "percentile_slider", "percentile_mid_btn", "model_status_state",
        "left_export_cb", "left_export_name_tb", "export_acc",
        "move_controls_col", "gallery_header_spacer_col", "right_header_col",
        "right_gallery_col",
    )
    _VIEW_KEYS = (
        "left_head", "left_gallery", "right_head", "right_gallery", "status_md",
        "hist_plot", "sel_info", "mark_state", "external_query_image",
        "clear_external_query_btn", "left_export_cb", "left_export_name_tb",
        "export_acc", "move_controls_col", "gallery_header_spacer_col",
        "right_header_col", "right_gallery_col", "main_slider", "aux_slider",
    )

    def _apply_known_result(self, action, result):
        if action == "preview-search":
            self._apply_output_keys(result, ("promptgen_status_md",) + self._SCORE_KEYS)
        elif action in {"score", "shortcut"}:
            self._apply_output_keys(result, self._SCORE_KEYS)
        elif action in {"folder-load", "selection", "export", "threshold"}:
            self._apply_output_keys(result, self._VIEW_KEYS)
        elif action == "prompt-generate":
            self._apply_output_keys(result, ("promptgen_status_md", "generated_prompt_tb", "generated_prompt_detail_slider"))
        elif action == "prompt-insert":
            self._apply_output_keys(result, ("promptgen_status_md", "pos_prompt_tb", "ir_prompt_tb", "llm_prompt_tb", "tagmatch_tags_tb"))

    def _apply_output_keys(self, result, keys):
        if not isinstance(result, (list, tuple)):
            return
        for key, value in zip(keys, result):
            if value is gr.SKIP:
                continue
            if key == "main_slider":
                self.inputs["main_threshold"] = _merge_update(self.inputs["main_threshold"], value)
            elif key == "aux_slider":
                self.inputs["aux_threshold"] = _merge_update(self.inputs["aux_threshold"], value)
            elif key == "percentile_slider":
                self.inputs["percentile"] = _merge_update(self.inputs["percentile"], value)
            elif key == "generated_prompt_tb":
                self.inputs["generated_prompt"] = _merge_update(self.inputs["generated_prompt"], value)
            elif key == "generated_prompt_detail_slider":
                self.inputs["generated_prompt_detail"] = _merge_update(self.inputs["generated_prompt_detail"], value)
            elif key == "pos_prompt_tb":
                self.inputs["pos_prompt"] = _merge_update(self.inputs["pos_prompt"], value)
            elif key == "ir_prompt_tb":
                self.inputs["ir_prompt"] = _merge_update(self.inputs["ir_prompt"], value)
            elif key == "llm_prompt_tb":
                self.inputs["llm_prompt"] = _merge_update(self.inputs["llm_prompt"], value)
            elif key == "tagmatch_tags_tb":
                self.inputs["tagmatch_tags"] = _merge_update(self.inputs["tagmatch_tags"], value)
            elif key == "promptgen_status_md" and _is_update(value) and "value" in value:
                self.state["generated_prompt_status"] = value["value"]

    def update_inputs(self, payload):
        for key, value in (payload or {}).items():
            if key in self.inputs:
                self.inputs[key] = value
        if "method" in (payload or {}):
            self.state["method"] = payload["method"]
            if payload["method"] == METHOD_TAGMATCH:
                try:
                    _lo.refresh_tagmatch_vocab_state(self.state, METHOD_TAGMATCH)
                except Exception:
                    pass

    def load_folder_job(self, payload, recursive=False):
        self.update_inputs(payload)
        return self.create_job("folder-load", lambda progress: _sc.load_folder_for_browse(
            self.state,
            self.inputs["folder"],
            self.inputs["main_threshold"],
            self.inputs["aux_threshold"],
            progress=progress,
            recursive=recursive,
        ))

    def score_job(self, payload):
        self.update_inputs(payload)
        return self.create_job("score", lambda progress: _sc.score_folder(
            self.state,
            self.device,
            self.inputs["method"],
            self.inputs["folder"],
            self.inputs["model_label"],
            self.inputs["pos_prompt"],
            self.inputs["neg_prompt"],
            self.inputs["pm_segment_mode"],
            self.inputs["ir_prompt"],
            self.inputs["ir_negative_prompt"],
            self.inputs["ir_penalty_weight"],
            self.inputs["llm_model_label"],
            self.inputs["llm_prompt"],
            self.inputs["llm_backend_id"],
            self.inputs["llm_shortlist_size"],
            self.inputs["tagmatch_tags"],
            self.inputs["main_threshold"],
            self.inputs["aux_threshold"],
            self.inputs["keep_pm_thresholds"],
            self.inputs["keep_ir_thresholds"],
            progress=progress,
        ))

    def search_job(self, kind, payload):
        self.update_inputs(payload)
        if kind == "similar":
            return self.create_job("preview-search", lambda progress: _sc.find_similar_images(
                self.state,
                self.device,
                self.inputs["folder"],
                self.inputs["model_label"],
                self.inputs["main_threshold"],
                self.inputs["aux_threshold"],
                progress=progress,
            ))
        if kind == "same-person":
            return self.create_job("preview-search", lambda progress: _sc.find_same_person_images(
                self.state,
                self.inputs["folder"],
                self.inputs["main_threshold"],
                self.inputs["aux_threshold"],
                progress=progress,
            ))
        return self.create_job("preview-search", lambda progress: _sc.find_objectsearch_images(
            self.state,
            self.device,
            self.inputs["folder"],
            self.inputs["main_threshold"],
            self.inputs["aux_threshold"],
            progress=progress,
        ))

    def threshold_action(self, payload):
        self.update_inputs(payload)
        p = payload or {}
        action = p.get("action", "split")
        with self.lock:
            if action == "percentile":
                self.inputs["percentile"] = float(p.get("percentile", self.inputs["percentile"]))
                result = _ui.set_from_percentile(
                    self.state,
                    self.inputs["percentile"],
                    self.inputs["main_threshold"],
                    self.inputs["aux_threshold"],
                )
                self._apply_output_keys(result, self._VIEW_KEYS[:-1])
            elif action == "hist":
                index = [p.get("x", 0), p.get("y", 0)]
                result = _ui.on_hist_click(
                    self.state,
                    gr.SelectData(index=index),
                    self.inputs["main_threshold"],
                    self.inputs["aux_threshold"],
                )
                self._apply_known_result("threshold", result)
            else:
                result = _ui.update_split(self.state, self.inputs["main_threshold"], self.inputs["aux_threshold"])
                self._apply_known_result("folder-load", result)
            return self.to_payload()

    def selection_action(self, payload):
        self.update_inputs(payload)
        p = payload or {}
        action = p.get("action", "")
        side = p.get("side", "")
        index = int(p.get("index", -1))
        mt = self.inputs["main_threshold"]
        at = self.inputs["aux_threshold"]
        with self.lock:
            if action in {"preview", "mark"}:
                result = _ui.handle_thumb_action(
                    self.state,
                    f"{action}:{side}:{index}:{time.time_ns()}",
                    mt, at,
                )
            elif action == "drop":
                drop_payload = json.dumps({
                    "source_side": p.get("source_side", side),
                    "source_index": index,
                    "target_side": p.get("target_side", ""),
                    "fnames": p.get("fnames", []),
                })
                result = _ui.handle_thumb_action(self.state, f"dropjson:{drop_payload}", mt, at)
            elif action == "move-right":
                result = _ui.move_right(self.state, mt, at)
            elif action == "move-left":
                result = _ui.move_left(self.state, mt, at)
            elif action == "pin":
                result = _ui.pin_selected(self.state, mt, at)
            elif action == "fit-threshold":
                result = _ui.fit_threshold_to_targets(self.state, mt, at)
            elif action == "clear-marked":
                result = _ui.clear_status(self.state, mt, at)
            elif action == "clear-all":
                result = _ui.clear_all_status(self.state, mt, at)
            elif action == "zoom":
                self.inputs["zoom"] = int(p.get("zoom", self.inputs["zoom"]))
                result = _ui.update_zoom(self.state, self.inputs["zoom"], mt, at)
            else:
                result = _vw.render_view_with_controls(self.state, mt, at)
            self._apply_known_result("selection", result)
            return self.to_payload()

    async def set_query_image(self, upload: UploadFile | None = None, payload=None):
        with self.lock:
            if upload is not None:
                data = await upload.read()
                result = _pr.set_external_query_from_bridge(
                    self.state,
                    json.dumps({
                        "data_url": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
                        "label": upload.filename or "query-image.png",
                    }),
                )
            else:
                result = _pr.set_external_query_from_bridge(self.state, json.dumps(payload or {}))
            self._apply_known_result("query-image", result)
            return self.to_payload()

    def clear_query_image(self):
        with self.lock:
            _pr.clear_external_query_image(self.state)
            return self.to_payload()

    def prompt_generate_job(self, payload):
        self.update_inputs(payload)
        return self.create_job("prompt-generate", lambda progress: _pr.generate_prompt_from_preview(
            self.state,
            self.device,
            self.inputs["prompt_generator"],
            self.inputs["generated_prompt"],
            self.inputs["generated_prompt_detail"],
            progress=progress,
        ))

    def prompt_insert(self, payload):
        self.update_inputs(payload)
        with self.lock:
            result = _pr.insert_generated_prompt(
                self.state,
                self.inputs["method"],
                self.inputs["generated_prompt"],
            )
            self._apply_known_result("prompt-insert", result)
            return self.to_payload()

    def prompt_detail(self, payload):
        self.update_inputs(payload)
        with self.lock:
            result = _pr.select_cached_generated_prompt(
                self.state,
                self.inputs["prompt_generator"],
                self.inputs["generated_prompt_detail"],
                self.inputs["generated_prompt"],
            )
            self._apply_output_keys(result, ("promptgen_status_md", "generated_prompt_tb"))
            return self.to_payload()

    def export_job(self, payload):
        self.update_inputs(payload)
        return self.create_job("export", lambda progress: _ui.export_files(
            self.state,
            self.inputs["main_threshold"],
            self.inputs["aux_threshold"],
            self.inputs["export_left_enabled"],
            self.inputs["export_right_enabled"],
            self.inputs["export_move_enabled"],
            self.inputs["export_left_name"],
            self.inputs["export_right_name"],
        ))

    def to_payload(self):
        view = _vw.current_view(self.state, self.inputs["main_threshold"], self.inputs["aux_threshold"])
        left_gallery = _plain_update(view[1]).get("value", view[1] if isinstance(view[1], list) else [])
        right_gallery = _plain_update(view[3]).get("value", view[3] if isinstance(view[3], list) else [])
        hist_url = self.media.register_pil(view[5])
        try:
            mark = json.loads(view[7] or "{}")
        except Exception:
            mark = {}
        query_update = _plain_update(view[8])
        query_path = query_update.get("value")
        controls = self.control_state()
        return {
            "app": {
                "name": self.app_name,
                "version": self.app_version,
                "changelog_html": self.changelog_html,
            },
            "inputs": dict(self.inputs),
            "view": {
                "method": self.state.get("method"),
                "mode": self.state.get("view_mode"),
                "status": view[4],
                "selection_info": view[6],
                "left": {
                    "title": view[0],
                    "items": self._gallery_items(left_gallery, "left"),
                },
                "right": {
                    "title": view[2],
                    "items": self._gallery_items(right_gallery, "right"),
                },
                "histogram_url": hist_url,
                "hist_geom": self.state.get("hist_geom"),
                "marks": mark,
                "score_lookup": mark.get("score_lookup", {}),
                "tag_score_lookup": mark.get("tag_score_lookup", {}),
                "segment_score_lookup": mark.get("segment_score_lookup", {}),
                "neg_segment_score_lookup": mark.get("neg_segment_score_lookup", {}),
                "query_image_url": self.media.register_path(query_path) if query_path else None,
                "query_status": self.state.get("generated_prompt_status"),
            },
            "controls": controls,
        }

    def _gallery_items(self, items, side):
        try:
            media_lookup = json.loads(_vw.marked_state_json(self.state)).get("media_lookup", {})
        except Exception:
            media_lookup = {}
        out = []
        for index, item in enumerate(items or []):
            path, caption = item
            fname = os.path.basename(path)
            original = media_lookup.get(path) or media_lookup.get(fname) or fname
            out.append({
                "side": side,
                "index": index,
                "filename": original,
                "caption": caption,
                "url": self.media.register_path(path),
                "marked": original in self.state.get(f"{side}_marked", []),
                "preview": original == self.state.get("preview_fname"),
                "overridden": original in self.state.get("overrides", {}),
            })
        return out

    def control_state(self):
        controls_tuple = _vw.configure_controls(self.state, self.inputs["method"])
        keys = (
            "promptmatch_group", "imagereward_group", "llmsearch_group", "tagmatch_group",
            "main_slider", "aux_slider", "aux_mid_btn", "keep_pm_thresholds_cb",
            "keep_ir_thresholds_cb", "percentile_slider", "percentile_mid_btn",
            "method_note",
        )
        updates = {key: _plain_update(value) for key, value in zip(keys, controls_tuple)}
        return {
            "visible": {
                "promptmatch": updates["promptmatch_group"].get("visible", True),
                "imagereward": updates["imagereward_group"].get("visible", False),
                "llmsearch": updates["llmsearch_group"].get("visible", False),
                "tagmatch": updates["tagmatch_group"].get("visible", False),
                "aux_slider": updates["aux_slider"].get("visible", True),
                "keep_pm_thresholds": updates["keep_pm_thresholds_cb"].get("visible", True),
                "keep_ir_thresholds": updates["keep_ir_thresholds_cb"].get("visible", False),
            },
            "sliders": {
                "main": self._slider_state(updates["main_slider"], self.inputs["main_threshold"]),
                "aux": self._slider_state(updates["aux_slider"], self.inputs["aux_threshold"]),
                "percentile": self._slider_state(updates["percentile_slider"], self.inputs["percentile"]),
            },
            "method_note": updates["method_note"].get("value", ""),
            "choices": {
                "methods": [METHOD_PROMPTMATCH, METHOD_IMAGEREWARD, METHOD_LLMSEARCH, METHOD_TAGMATCH],
                "search_methods": [METHOD_SIMILARITY, METHOD_SAMEPERSON, METHOD_OBJECTSEARCH],
                "promptmatch_models": promptmatch_model_dropdown_choices(),
                "llm_backends": prompt_backend_dropdown_choices(PROMPT_GENERATOR_CHOICES),
                "prompt_generators": prompt_backend_dropdown_choices(PROMPT_GENERATOR_ALL_CHOICES),
            },
            "model_status": json.loads(promptmatch_model_status_json()),
            "tagmatch_vocab": json.loads(self.state.get("tagmatch_vocab_json") or "[]"),
        }

    def _slider_state(self, update, value):
        geom = self.state.get("hist_geom") or {}
        label = update.get("label", "")
        minimum = update.get("minimum", 0)
        maximum = update.get("maximum", 100)
        if "aux" in label.lower() or "negative" in label.lower():
            if geom.get("has_neg") and geom.get("neg_lo") is not None and geom.get("neg_hi") is not None:
                minimum = geom["neg_lo"]
                maximum = geom["neg_hi"]
        elif geom.get("pos_lo") is not None and geom.get("pos_hi") is not None:
            minimum = geom["pos_lo"]
            maximum = geom["pos_hi"]
        elif geom.get("lo") is not None and geom.get("hi") is not None:
            minimum = geom["lo"]
            maximum = geom["hi"]
        return {
            "label": label,
            "minimum": minimum,
            "maximum": maximum,
            "value": value,
            "visible": update.get("visible", True),
            "interactive": update.get("interactive", True),
        }


def dependency_issues():
    return runtime_requirement_issues()
