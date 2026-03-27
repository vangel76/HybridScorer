"""
HybridSelector — PromptMatch + ImageReward in one UI
----------------------------------------------------
• Switch scoring method inside one app
• PromptMatch: positive + optional negative prompt with CLIP-family models
• ImageReward: positive aesthetic prompt with optional penalty prompt
• Manual move between buckets
• Lossless export into method-specific output folders
• Created by vangel
"""

import json
import os
import shutil
import string
import sys

import gradio as gr
import ImageReward as RM

from imagereward import (
    ALLOWED_EXTENSIONS,
    INPUT_FOLDER_NAME,
    IR_PROMPT,
    THRESHOLD as IMAGEREWARD_THRESHOLD,
    iter_imagereward_scores,
    require_cuda,
    resolve_server_port,
)
from promptmatch import (
    MODEL_CHOICES,
    MODEL_LABELS,
    ModelBackend,
    NEGATIVE_PROMPT,
    NEGATIVE_THRESHOLD,
    SEARCH_PROMPT,
    label_for_backend,
    score_all,
)
from PIL import Image, ImageDraw


METHOD_PROMPTMATCH = "PromptMatch"
METHOD_IMAGEREWARD = "ImageReward"
DEFAULT_IR_NEGATIVE_PROMPT = ""
DEFAULT_IR_PENALTY_WEIGHT = 1.0
PROMPTMATCH_SLIDER_MIN = -1.0
PROMPTMATCH_SLIDER_MAX = 1.0
IMAGEREWARD_SLIDER_MIN = -5.0
IMAGEREWARD_SLIDER_MAX = 5.0


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


def get_model_config(label):
    return next((cfg for cfg in MODEL_CHOICES if cfg[0] == label), None)


def method_labels(method):
    if method == METHOD_PROMPTMATCH:
        return "FOUND", "NOT FOUND", "found", "notfound"
    return "BEST", "NORMAL", "best", "normal"


def promptmatch_slider_range(scores):
    pos_vals = [v["pos"] for v in scores.values() if v["pos"] >= 0]
    neg_vals = [v["neg"] for v in scores.values() if v.get("neg") is not None and v["neg"] >= 0]
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


def threshold_for_percentile(method, scores, percentile):
    if method == METHOD_PROMPTMATCH:
        vals = sorted([v["pos"] for v in scores.values() if v["pos"] >= 0], reverse=True)
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


def status_line(method, left_items, right_items, scores, overrides):
    left_name, right_name, _, _ = method_labels(method)
    if not scores:
        failed = 0
    elif method == METHOD_PROMPTMATCH and all("pos" in v for v in scores.values()):
        failed = sum(1 for v in scores.values() if v["pos"] < 0)
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

    if method == METHOD_PROMPTMATCH and not all("pos" in item for item in scores.values()):
        return left, right
    if method == METHOD_IMAGEREWARD and not all("score" in item for item in scores.values()):
        return left, right

    if method == METHOD_PROMPTMATCH:
        ordered = sorted(scores.items(), key=lambda x: -x[1]["pos"])
        for fname, item in ordered:
            if item["pos"] < 0:
                continue
            side = overrides.get(fname)
            if side is None:
                pos_ok = item["pos"] >= main_threshold
                neg_ok = (item["neg"] is None) or (item["neg"] < aux_threshold)
                side = left_name if (pos_ok and neg_ok) else right_name
            caption = f"{'✋ ' if fname in overrides else ''}{item['pos']:.3f} | {fname}"
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


def create_app():
    require_cuda()
    device = "cuda"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, INPUT_FOLDER_NAME)
    if not os.path.isdir(source_dir):
        source_dir = script_dir

    prompt_backend = ModelBackend(device)

    state = {
        "method": METHOD_PROMPTMATCH,
        "source_dir": source_dir,
        "scores": {},
        "overrides": {},
        "left_sel": None,
        "right_sel": None,
        "backend": prompt_backend,
        "ir_model": None,
        "hist_geom": None,
    }

    def tooltip_head(pairs):
        mapping = json.dumps(pairs)
        return f"""
<script>
(() => {{
  const tooltips = {mapping};
  const applyTooltips = () => {{
    for (const [id, text] of Object.entries(tooltips)) {{
      const root = document.getElementById(id);
      if (!root) continue;
      root.title = text;
      root.setAttribute("aria-label", text);
      const targets = root.querySelectorAll("button, input, textarea, select, img");
      for (const el of targets) {{
        el.title = text;
        el.setAttribute("aria-label", text);
      }}
    }}
  }};
  applyTooltips();
  new MutationObserver(applyTooltips).observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>
"""

    tooltips = {
        "hy-method": "Choose whether to sort by PromptMatch or ImageReward.",
        "hy-folder": "Path to the image folder you want to score.",
        "hy-model": "Choose the PromptMatch model family and size.",
        "hy-pos": "Describe what you want to find in the images.",
        "hy-neg": "Optional PromptMatch negative prompt that counts against a match.",
        "hy-ir-pos": "Describe the style or aesthetic you want ImageReward to favor.",
        "hy-ir-neg": "Optional experimental penalty prompt. Its score is subtracted from the positive style score.",
        "hy-ir-weight": "How strongly the penalty prompt should reduce the final ImageReward score.",
        "hy-run": "Score the current folder with the selected method and prompts.",
        "hy-main-slider": "Main classification threshold. Click the histogram to set it visually.",
        "hy-aux-slider": "PromptMatch negative threshold. Lower values pass the negative filter.",
        "hy-percentile": "Automatically set the main threshold to keep roughly the top N percent.",
        "hy-zoom": "Change thumbnail size in both galleries at once.",
        "hy-hist": "Histogram of current scores. In PromptMatch, click the top chart for positive threshold or bottom chart for negative threshold.",
        "hy-export": "Losslessly copy the current split into method-specific output folders.",
        "hy-left-gallery": "Images currently in the left bucket. Click one to select it.",
        "hy-right-gallery": "Images currently in the right bucket. Click one to select it.",
        "hy-move-right": "Move the selected left image into the right bucket as a manual override.",
        "hy-move-left": "Move the selected right image into the left bucket as a manual override.",
    }

    def ensure_imagereward_model():
        if state["ir_model"] is None:
            state["ir_model"] = RM.load("ImageReward-v1.0")
        return state["ir_model"]

    def gallery_update(items, columns=None):
        update_kwargs = {"value": items, "selected_index": None}
        if columns is not None:
            update_kwargs["columns"] = columns
        return gr.update(**update_kwargs)

    def render_histogram(method, scores, main_threshold, aux_threshold):
        if not scores:
            state["hist_geom"] = None
            return None

        if method == METHOD_PROMPTMATCH:
            if not all("pos" in item for item in scores.values()):
                state["hist_geom"] = None
                return None
            pos_vals = [v["pos"] for v in scores.values() if v["pos"] >= 0]
            neg_vals = [v["neg"] for v in scores.values() if v.get("neg") is not None and v["neg"] >= 0]
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
            W, CH = 300, 130
            PAD_L, PAD_R = 38, 8
            PAD_TOP, PAD_BOT = 18, 22
            GAP = 28
            n_ch = 2 if has_neg else 1
            H = PAD_TOP + n_ch * CH + (n_ch - 1) * GAP + PAD_BOT
            img = Image.new("RGB", (W, H), "#0d0d11")
            draw = ImageDraw.Draw(img)

            def draw_chart(y0, counts, lo, hi, threshold, bar_rgb, line_rgb, label):
                cW = W - PAD_L - PAD_R
                max_c = max(counts) if counts else 1
                bw = cW / len(counts)
                draw.rectangle([PAD_L, y0, W - PAD_R, y0 + CH], fill="#0f0f16")
                for i, count in enumerate(counts):
                    if count == 0:
                        continue
                    bh = max(1, int((count / max_c) * (CH - 2)))
                    x0 = PAD_L + int(i * bw) + 1
                    x1 = PAD_L + int((i + 1) * bw) - 1
                    draw.rectangle([x0, y0 + CH - bh, x1, y0 + CH], fill=bar_rgb)
                tx = PAD_L + int(((threshold - lo) / (hi - lo)) * cW)
                tx = max(PAD_L, min(W - PAD_R, tx))
                for yy in range(y0, y0 + CH, 6):
                    draw.line([(tx, yy), (tx, min(yy + 3, y0 + CH))], fill=line_rgb, width=2)
                for frac, val in [(0.0, lo), (0.5, (lo + hi) / 2), (1.0, hi)]:
                    lx = PAD_L + int(frac * cW)
                    draw.text((lx, y0 + CH + 4), f"{val:.3f}", fill="#667755", anchor="mt")
                draw.text((PAD_L, y0 - 14), f"{label} threshold: {threshold:.3f}", fill="#99bb88")

            draw_chart(PAD_TOP, pos_counts, pos_lo, pos_hi, main_threshold, "#3a7a3a", "#aadd66", "Positive")
            if has_neg:
                draw_chart(PAD_TOP + CH + GAP, neg_counts, neg_lo, neg_hi, aux_threshold, "#7a3a3a", "#dd6644", "Negative")

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
        W, H = 300, 130
        PAD_L, PAD_R = 38, 8
        PAD_TOP, PAD_BOT = 18, 22
        cW = W - PAD_L - PAD_R
        img = Image.new("RGB", (W, H), "#0d0d11")
        draw = ImageDraw.Draw(img)
        draw.rectangle([PAD_L, PAD_TOP, W - PAD_R, PAD_TOP + H - PAD_BOT], fill="#0f0f16")
        max_c = max(counts) if counts else 1
        bw = cW / bins
        for i, count in enumerate(counts):
            if count == 0:
                continue
            bh = max(1, int((count / max_c) * (H - PAD_BOT - PAD_TOP - 2)))
            x0 = PAD_L + int(i * bw) + 1
            x1 = PAD_L + int((i + 1) * bw) - 1
            draw.rectangle([x0, PAD_TOP + (H - PAD_BOT - bh), x1, PAD_TOP + H - PAD_BOT], fill="#3a7a3a")
        tx = PAD_L + int(((main_threshold - lo) / (hi - lo)) * cW)
        tx = max(PAD_L, min(W - PAD_R, tx))
        for yy in range(PAD_TOP, PAD_TOP + H - PAD_BOT, 6):
            draw.line([(tx, yy), (tx, min(yy + 3, PAD_TOP + H - PAD_BOT))], fill="#aadd66", width=2)
        for frac, val in [(0.0, lo), (0.5, (lo + hi) / 2), (1.0, hi)]:
            lx = PAD_L + int(frac * cW)
            draw.text((lx, PAD_TOP + H - PAD_BOT + 4), f"{val:.3f}", fill="#667755", anchor="mt")
        draw.text((PAD_L, PAD_TOP - 14), f"ImageReward threshold: {main_threshold:.3f}", fill="#99bb88")
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

    def current_view(main_threshold, aux_threshold):
        left_name, right_name, _, _ = method_labels(state["method"])
        zoom_columns = int(state.get("zoom_columns", 5))
        left_items, right_items = build_split(
            state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold
        )
        return (
            f"### {left_name} — {len(left_items)} images",
            gallery_update(left_items, columns=zoom_columns),
            f"### {right_name} — {len(right_items)} images",
            gallery_update(right_items, columns=zoom_columns),
            status_line(state["method"], left_items, right_items, state["scores"], state["overrides"]),
            render_histogram(state["method"], state["scores"], main_threshold, aux_threshold),
        )

    def score_imagereward(folder_paths, positive_prompt, negative_prompt, penalty_weight, progress):
        model = ensure_imagereward_model()
        positive_prompt = (positive_prompt or "").strip() or IR_PROMPT
        negative_prompt = (negative_prompt or "").strip()
        penalty_weight = float(penalty_weight)

        base_scores = {}
        progress(0, desc=f"Scoring {len(folder_paths)} images with ImageReward...")
        for event in iter_imagereward_scores(folder_paths, model, device, positive_prompt):
            if event["type"] == "oom":
                progress(event["done"] / max(event["total"], 1), desc=f"ImageReward OOM, retrying batch {event['batch_size']}")
                continue
            progress(event["done"] / max(event["total"], 1), desc=f"ImageReward {event['done']}/{event['total']}")
            base_scores = event["scores"]

        penalty_scores = {}
        if negative_prompt:
            progress(0, desc=f"Applying penalty prompt to {len(folder_paths)} images...")
            for event in iter_imagereward_scores(folder_paths, model, device, negative_prompt):
                if event["type"] == "oom":
                    progress(event["done"] / max(event["total"], 1), desc=f"Penalty OOM, retrying batch {event['batch_size']}")
                    continue
                progress(event["done"] / max(event["total"], 1), desc=f"Penalty prompt {event['done']}/{event['total']}")
                penalty_scores = event["scores"]

        wrapped = {}
        for path in folder_paths:
            fname = os.path.basename(path)
            base_item = base_scores.get(fname, {"score": -float("inf"), "path": path})
            penalty_item = penalty_scores.get(fname)
            penalty_value = penalty_item["score"] if penalty_item is not None else None
            final_score = base_item["score"]
            if penalty_value is not None:
                final_score = final_score - (penalty_weight * penalty_value)
            wrapped[fname] = {
                "score": float(final_score),
                "base": float(base_item["score"]),
                "penalty": float(penalty_value) if penalty_value is not None else None,
                "path": path,
            }
        return wrapped

    def configure_controls(method):
        state["method"] = method
        state["overrides"] = {}
        state["left_sel"] = None
        state["right_sel"] = None
        if method == METHOD_PROMPTMATCH:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(label="Primary threshold (>= -> FOUND)", value=0.14, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(label="Negative threshold (< -> passes)", visible=True, value=NEGATIVE_THRESHOLD, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(value="PromptMatch sorts by text-image similarity. Use a positive prompt and optional negative prompt."),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(label="Primary threshold (>= -> BEST)", value=IMAGEREWARD_THRESHOLD, minimum=IMAGEREWARD_SLIDER_MIN, maximum=IMAGEREWARD_SLIDER_MAX),
            gr.update(visible=False, value=NEGATIVE_THRESHOLD),
            gr.update(value="ImageReward sorts by aesthetic preference. Optional penalty prompt subtracts a second style score."),
        )

    def empty_result(message, method):
        _, _, main_upd, aux_upd, _ = configure_controls(method)
        return (
            "### LEFT",
            gallery_update([]),
            "### RIGHT",
            gallery_update([]),
            message,
            None,
            main_upd,
            aux_upd,
        )

    def score_folder(method, folder, model_label, pos_prompt, neg_prompt, ir_prompt, ir_negative_prompt, ir_penalty_weight, progress=gr.Progress()):
        folder = (folder or "").strip()
        if not folder or not os.path.isdir(folder):
            return empty_result(f"Invalid folder: {folder!r}", method)

        image_paths = scan_image_paths(folder)
        if not image_paths:
            return empty_result(f"No images found in {folder}", method)

        state["method"] = method
        state["source_dir"] = folder
        state["overrides"] = {}
        state["left_sel"] = None
        state["right_sel"] = None

        if method == METHOD_PROMPTMATCH:
            cfg = get_model_config(model_label)
            if cfg is None:
                return empty_result(f"Unknown PromptMatch model: {model_label}", method)
            _, backend_name, kwargs = cfg
            if label_for_backend(state["backend"]) != model_label:
                state["backend"] = ModelBackend(device, backend=backend_name, **kwargs)

            pos_prompt = (pos_prompt or "").strip() or SEARCH_PROMPT
            neg_prompt = (neg_prompt or "").strip()
            pos_emb = state["backend"].encode_text(pos_prompt)
            neg_emb = state["backend"].encode_text(neg_prompt) if neg_prompt else None
            progress(0, desc=f"Scoring {len(image_paths)} images with PromptMatch...")

            def _cb(done, total, batch_size, oom_retry):
                label = f"PromptMatch {done}/{total} (batch {batch_size})"
                if oom_retry:
                    label = f"PromptMatch OOM, retrying batch {batch_size}"
                progress(done / max(total, 1), desc=label)

            state["scores"] = score_all(image_paths, state["backend"], pos_emb, neg_emb, progress_cb=_cb)
            pos_min, pos_max, pos_mid, neg_min, neg_max, neg_mid, has_neg = promptmatch_slider_range(state["scores"])
            left_head, left_gallery, right_head, right_gallery, status, hist = current_view(pos_mid, neg_mid)
            return (
                left_head,
                left_gallery,
                right_head,
                right_gallery,
                status,
                hist,
                gr.update(minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX, value=pos_mid, label="Primary threshold (>= -> FOUND)"),
                gr.update(minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX, value=neg_mid, visible=True, interactive=has_neg, label="Negative threshold (< -> passes)"),
            )

        state["scores"] = score_imagereward(
            image_paths,
            ir_prompt,
            ir_negative_prompt,
            ir_penalty_weight,
            progress,
        )
        lo, hi, mid = imagereward_slider_range(state["scores"])
        left_head, left_gallery, right_head, right_gallery, status, hist = current_view(mid, NEGATIVE_THRESHOLD)
        return (
            left_head,
            left_gallery,
            right_head,
            right_gallery,
            status,
            hist,
            gr.update(minimum=IMAGEREWARD_SLIDER_MIN, maximum=IMAGEREWARD_SLIDER_MAX, value=mid, label="Primary threshold (>= -> BEST)"),
            gr.update(value=NEGATIVE_THRESHOLD, visible=False),
        )

    def update_split(main_threshold, aux_threshold):
        return current_view(main_threshold, aux_threshold)

    def select_left(evt: gr.SelectData, main_threshold, aux_threshold):
        left_items, _ = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)
        state["left_sel"] = os.path.basename(left_items[evt.index][0]) if evt.index < len(left_items) else None
        state["right_sel"] = None
        left_name, _, _, _ = method_labels(state["method"])
        return f"Selected in {left_name}: **{state['left_sel'] or '?'}**"

    def select_right(evt: gr.SelectData, main_threshold, aux_threshold):
        _, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)
        state["right_sel"] = os.path.basename(right_items[evt.index][0]) if evt.index < len(right_items) else None
        state["left_sel"] = None
        _, right_name, _, _ = method_labels(state["method"])
        return f"Selected in {right_name}: **{state['right_sel'] or '?'}**"

    def move_right(main_threshold, aux_threshold):
        left_name, right_name, _, _ = method_labels(state["method"])
        if state["left_sel"]:
            state["overrides"][state["left_sel"]] = right_name
        state["left_sel"] = None
        state["right_sel"] = None
        return (*current_view(main_threshold, aux_threshold), "")

    def move_left(main_threshold, aux_threshold):
        left_name, right_name, _, _ = method_labels(state["method"])
        if state["right_sel"]:
            state["overrides"][state["right_sel"]] = left_name
        state["left_sel"] = None
        state["right_sel"] = None
        return (*current_view(main_threshold, aux_threshold), "")

    def set_from_percentile(percentile, main_threshold, aux_threshold):
        new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
        return (*current_view(new_threshold, aux_threshold), gr.update(value=new_threshold))

    def update_zoom(zoom_value, main_threshold, aux_threshold):
        try:
            state["zoom_columns"] = max(2, min(10, int(zoom_value)))
        except Exception:
            state["zoom_columns"] = 5
        left_head, left_gallery, right_head, right_gallery, status, hist = current_view(main_threshold, aux_threshold)
        return left_head, left_gallery, right_head, right_gallery, status, hist

    def on_hist_click(sel: gr.SelectData, main_threshold, aux_threshold):
        geom = state.get("hist_geom")
        if not geom:
            return (*current_view(main_threshold, aux_threshold), gr.update(), gr.update())
        try:
            cx, cy = sel.index
            if geom["method"] == METHOD_PROMPTMATCH:
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
                    return (*current_view(main_threshold, aux_threshold), gr.update(), gr.update())
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
                    return (*current_view(main_threshold, aux_threshold), gr.update(), gr.update())
        except Exception:
            return (*current_view(main_threshold, aux_threshold), gr.update(), gr.update())

        return (*current_view(main_threshold, aux_threshold), gr.update(value=main_threshold), gr.update(value=aux_threshold))

    def export_files(main_threshold, aux_threshold):
        left_items, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)
        left_name, right_name, left_dirname, right_dirname = method_labels(state["method"])
        base = state["source_dir"]
        left_dir = os.path.join(base, left_dirname)
        right_dir = os.path.join(base, right_dirname)
        for folder in (left_dir, right_dir):
            os.makedirs(folder, exist_ok=True)
            for name in os.listdir(folder):
                fp = os.path.join(folder, name)
                if os.path.isfile(fp):
                    os.remove(fp)
        for path, _ in left_items:
            shutil.copy2(path, os.path.join(left_dir, os.path.basename(path)))
        for path, _ in right_items:
            shutil.copy2(path, os.path.join(right_dir, os.path.basename(path)))
        return (
            f"Losslessly copied {len(left_items)} files to {left_dir}\n"
            f"Losslessly copied {len(right_items)} files to {right_dir}"
        )

    css = """
    body, .gradio-container { background:#0d0d11 !important; color:#ddd8cc !important; }
    .gr-block,.gr-box,.panel { background:#14141c !important; border-color:#252530 !important; }
    .gradio-container { max-width: 100% !important; padding: 8px 12px !important; }
    .main { max-width: 100% !important; }
    footer { display: none !important; }
    h1 { font-family:'Courier New',monospace; letter-spacing:.18em; color:#aadd66; text-transform:uppercase; margin:0; font-size:1.4rem; }
    .subhead { color:#667755; font-family:monospace; font-size:.78em; margin-top:3px; }
    .sidebar-box { background:#171722; border:1px solid #2c2c39; border-radius:10px; padding:12px; }
    .method-note { font-family:monospace; font-size:.78em; color:#94a57f; background:#11111a; border-radius:8px; padding:8px 10px; }
    .status-md p { font-family:monospace !important; color:#9fc27c !important; }
    .hist-img img { cursor:crosshair !important; border-radius:6px; }
    .grid-wrap img { object-fit: contain !important; background: #0a0a12; }
    .grid-wrap img[alt^="✋"] { outline: 3px solid #dd3322 !important; outline-offset: -3px; }
    .grid-wrap .caption-label span, .grid-wrap [class*="caption"] { font-family:monospace !important; font-size:.72em !important; color:#8899aa !important; }
    .move-col { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px; padding:10px 6px; background:#0f0f16; border-radius:8px; border:1px solid #252535; }
    .move-col button { width:100%; }
    .sel-info p { font-family:monospace !important; font-size:.72em !important; color:#aabb88 !important; text-align:center; word-break:break-all; }
    """

    with gr.Blocks(title="HybridSelector") as demo:
        gr.HTML("""
<h1>⬡ HybridSelector</h1>
<div class='subhead'>PromptMatch + ImageReward in one UI &middot; quick image triage &middot; created by vangel</div>
""")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=330, elem_classes=["sidebar-box"]):
                method_dd = gr.Dropdown(
                    choices=[METHOD_PROMPTMATCH, METHOD_IMAGEREWARD],
                    value=METHOD_PROMPTMATCH,
                    label="Method",
                    elem_id="hy-method",
                )
                method_note = gr.Markdown(
                    "PromptMatch sorts by text-image similarity. Use a positive prompt and optional negative prompt.",
                    elem_classes=["method-note"],
                )
                folder_input = gr.Textbox(value=source_dir, label="Image folder", lines=1, placeholder=folder_placeholder(), elem_id="hy-folder")

                with gr.Group(visible=True) as promptmatch_group:
                    model_dd = gr.Dropdown(choices=MODEL_LABELS, value=label_for_backend(prompt_backend), label="PromptMatch model", elem_id="hy-model")
                    pos_prompt_tb = gr.Textbox(value=SEARCH_PROMPT, label="Positive prompt", lines=1, elem_id="hy-pos")
                    neg_prompt_tb = gr.Textbox(value=NEGATIVE_PROMPT, label="Negative prompt", lines=1, elem_id="hy-neg")

                with gr.Group(visible=False) as imagereward_group:
                    ir_prompt_tb = gr.Textbox(value=IR_PROMPT, label="ImageReward positive prompt", lines=3, elem_id="hy-ir-pos")
                    ir_negative_prompt_tb = gr.Textbox(
                        value=DEFAULT_IR_NEGATIVE_PROMPT,
                        label="Experimental penalty prompt",
                        lines=2,
                        placeholder="Optional: undesirable style or mood to subtract",
                        elem_id="hy-ir-neg",
                    )
                    ir_penalty_weight_tb = gr.Number(
                        value=DEFAULT_IR_PENALTY_WEIGHT,
                        label="Penalty weight",
                        minimum=0.0,
                        maximum=3.0,
                        step=0.1,
                        elem_id="hy-ir-weight",
                    )

                run_btn = gr.Button("Run scoring", elem_id="hy-run")
                hist_plot = gr.Image(value=None, show_label=False, interactive=False, elem_classes=["hist-img"], elem_id="hy-hist")
                main_slider = gr.Slider(minimum=-1.0, maximum=1.0, value=0.14, step=0.001, label="Primary threshold (>= -> FOUND)", elem_id="hy-main-slider")
                aux_slider = gr.Slider(minimum=-1.0, maximum=1.0, value=NEGATIVE_THRESHOLD, step=0.001, label="Negative threshold (< -> passes)", elem_id="hy-aux-slider")
                percentile_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Or keep top N%", elem_id="hy-percentile")
                zoom_slider = gr.Slider(minimum=2, maximum=10, value=5, step=1, label="Thumbnail zoom", elem_id="hy-zoom")
                status_md = gr.Markdown("", elem_classes=["status-md"])
                export_btn = gr.Button("Export folders", elem_id="hy-export")
                export_tb = gr.Textbox(label="Export result", lines=3, interactive=False)

            with gr.Column(scale=5):
                with gr.Row():
                    left_head = gr.Markdown("### FOUND")
                    right_head = gr.Markdown("### NOT FOUND")
                with gr.Row(equal_height=True):
                    left_gallery = gr.Gallery(show_label=False, columns=5, height="80vh", object_fit="contain", preview=True, allow_preview=True, elem_classes=["grid-wrap"], elem_id="hy-left-gallery")
                    with gr.Column(scale=0, min_width=100, elem_classes=["move-col"]):
                        sel_info = gr.Markdown("", elem_classes=["sel-info"])
                        move_right_btn = gr.Button("Move →", elem_id="hy-move-right")
                        move_left_btn = gr.Button("← Move", elem_id="hy-move-left")
                    right_gallery = gr.Gallery(show_label=False, columns=5, height="80vh", object_fit="contain", preview=True, allow_preview=True, elem_classes=["grid-wrap"], elem_id="hy-right-gallery")

        method_dd.change(
            fn=configure_controls,
            inputs=[method_dd],
            outputs=[promptmatch_group, imagereward_group, main_slider, aux_slider, method_note],
        )

        run_btn.click(
            fn=score_folder,
            inputs=[method_dd, folder_input, model_dd, pos_prompt_tb, neg_prompt_tb, ir_prompt_tb, ir_negative_prompt_tb, ir_penalty_weight_tb],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, main_slider, aux_slider],
        )

        main_slider.change(fn=update_split, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot])
        aux_slider.change(fn=update_split, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot])
        percentile_slider.change(
            fn=set_from_percentile,
            inputs=[percentile_slider, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, main_slider],
        )
        zoom_slider.change(
            fn=update_zoom,
            inputs=[zoom_slider, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot],
        )
        left_gallery.select(fn=select_left, inputs=[main_slider, aux_slider], outputs=[sel_info])
        right_gallery.select(fn=select_right, inputs=[main_slider, aux_slider], outputs=[sel_info])
        move_right_btn.click(fn=move_right, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info])
        move_left_btn.click(fn=move_left, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info])
        hist_plot.select(fn=on_hist_click, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, main_slider, aux_slider])
        export_btn.click(fn=export_files, inputs=[main_slider, aux_slider], outputs=[export_tb])

    return demo, css, tooltip_head(tooltips)


if __name__ == "__main__":
    app, css, head = create_app()
    port = resolve_server_port(7862, "HYBRIDSELECTOR_PORT")
    print(f"Launching HybridSelector on http://localhost:{port} ...")
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
