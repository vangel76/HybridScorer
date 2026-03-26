"""
ImageReward Aesthetic Scorer — Interactive Gallery Sorter
-----------------------------------------------------------
• ImageReward v1.5 scoring for aesthetic preference
• Native Gradio galleries — proven to work
• object_fit="contain" — full image always visible, never cropped
• Threshold slider → live re-sort
• Manual move: select an image in a gallery, then click ← Move or Move →
  Manual overrides persist through slider changes, re-scoring, and folder changes
"""

import os, sys, json
import torch
from PIL import Image
from tqdm import tqdm
import ImageReward as RM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IR_PROMPT = (
    "masterpiece, best quality, ultra-detailed, cinematic, high fashion photography, "
    "extremely beautiful woman, 25-35 years old, sharp elegant features, "
    "flawless skin, dramatic lighting, rim light, chiaroscuro, "
    "moody atmosphere, volumetric god rays, professional studio portrait, vogue style, "
    "highly detailed face and eyes, 8k, award-winning"
)

THRESHOLD = 0.5  # ImageReward typically ranges around -1 to +2
INPUT_FOLDER_NAME = "images"
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.bmp', '.avif')
DEFAULT_BATCH_SIZE = 32
MAX_BATCH_SIZE = 128
DEFAULT_SCORE_RANGE = (-2.0, 2.5)


def get_auto_batch_size(device, reference_vram_gb=32):
    """
    Pick a conservative batch size from currently free CUDA memory.

    The heuristic uses free VRAM after the model is loaded, so it adapts to
    both the GPU size and the model's existing memory footprint.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return DEFAULT_BATCH_SIZE

    try:
        free_bytes, _ = torch.cuda.mem_get_info()
    except RuntimeError:
        return DEFAULT_BATCH_SIZE

    free_gb = free_bytes / (1024 ** 3)
    scaled = max(1, int(DEFAULT_BATCH_SIZE * (free_gb / float(reference_vram_gb))))

    if scaled >= 16:
        scaled = max(16, (scaled // 8) * 8)
    elif scaled >= 8:
        scaled = max(8, (scaled // 4) * 4)

    return max(1, min(MAX_BATCH_SIZE, scaled))


def is_cuda_oom_error(exc):
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def render_progress(done=0, total=0, label="Idle"):
    if total > 0:
        pct = max(0, min(100, int((done / total) * 100)))
        detail = f"{done}/{total} ({pct}%)"
    else:
        pct = 0
        detail = label
    return f"""
<div class="progress-wrap">
  <div class="progress-label">{label}</div>
  <div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>
  <div class="progress-meta">{detail}</div>
</div>
"""


def iter_imagereward_scores(image_paths, model, device, prompt):
    scores = {}
    total = len(image_paths)
    done = 0
    batch_size = get_auto_batch_size(device)
    model = model.to(device).eval()
    print(f"[ImageReward] Using batch size {batch_size}")

    while done < total:
        current_size = min(batch_size, total - done)
        batch_paths = image_paths[done:done + current_size]
        batch_filenames = [os.path.basename(p) for p in batch_paths]

        try:
            with torch.no_grad():
                _, batch_rewards = model.inference_rank(prompt, batch_paths)
            for filename, path, reward in zip(batch_filenames, batch_paths, batch_rewards):
                scores[filename] = {"score": float(reward), "path": path}
            done += len(batch_paths)
            yield {"type": "progress", "done": done, "total": total,
                   "batch_size": batch_size, "scores": dict(scores)}
        except Exception as e:
            if device == "cuda" and is_cuda_oom_error(e) and current_size > 1:
                batch_size = max(1, current_size // 2)
                print(f"[ImageReward] CUDA OOM, retrying with batch size {batch_size}")
                torch.cuda.empty_cache()
                yield {"type": "oom", "done": done, "total": total,
                       "batch_size": batch_size, "scores": dict(scores)}
                continue

            print(f"Batch error at {done}: {e}", file=sys.stderr)
            for filename, path in zip(batch_filenames, batch_paths):
                scores[filename] = {"score": -float('inf'), "path": path}
            done += len(batch_paths)
            yield {"type": "progress", "done": done, "total": total,
                   "batch_size": batch_size, "scores": dict(scores)}
        finally:
            if device == 'cuda':
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# ImageReward Scoring
# ---------------------------------------------------------------------------

def calculate_imagereward_scores(image_paths, model, device):
    """
    Batched scoring using ImageReward inference_rank.
    Returns dict of {filename: score_value}.
    """
    scores = {}
    progress = tqdm(total=len(image_paths), desc="Scoring with ImageReward")
    for event in iter_imagereward_scores(image_paths, model, device, IR_PROMPT):
        if event["type"] == "oom":
            continue
        scores = {fname: item["score"] for fname, item in event["scores"].items()}
        progress.update(event["done"] - progress.n)
    progress.close()
    return scores


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def launch_ui(initial_scores, image_paths, model, device, script_dir, source_dir):
    try:
        import gradio as gr
    except ImportError:
        sys.exit("Gradio not installed.\nRun: pip install gradio")

    import shutil, statistics

    # ── shared mutable state ────────────────────────────────────────────────
    def scan_image_paths():
        """Re-scan state["source_dir"] — always reads current folder from state."""
        d = state.get("source_dir") or source_dir
        if not d or not os.path.isdir(d):
            print(f"  [WARN] scan_image_paths: not a directory: {d!r}")
            return []
        all_files = os.listdir(d)
        found = [
            os.path.join(d, f)
            for f in sorted(all_files)
            if f.lower().endswith(ALLOWED_EXTENSIONS)
            and os.path.isfile(os.path.join(d, f))
        ]
        print(f"  [INFO] scan_image_paths: {len(found)}/{len(all_files)} images in {d}")
        return found

    state = {
        "scores":      initial_scores,
        "overrides":   {},   # fname → "best" | "normal"
        "best_sel":    None,
        "normal_sel":  None,
        "model":       model,
        "device":      device,
        "image_paths": list(image_paths),
        "progress_html": render_progress(),
    }

    # ── core logic ──────────────────────────────────────────────────────────
    def imagereward_side(score, threshold):
        """Return "best" if score >= threshold, else "normal"."""
        return "best" if score >= threshold else "normal"

    def build_split(scores, overrides, threshold):
        """Returns (best_list, normal_list) sorted by score desc.
        Each item is (image_path, caption_string).
        Caption = score value + filename.
        Manual images get ✋ PREFIX so CSS can add red border.
        """
        best, normal = [], []
        for fname, score in sorted(scores.items(), key=lambda x: -x[1]["score"]):
            if score["score"] < -1000:  # failed/missing
                continue
            manual = fname in overrides
            side = overrides[fname] if manual else imagereward_side(score["score"], threshold)
            
            if manual:
                caption = f"✋ {score['score']:.3f} | {fname}"
            else:
                caption = f"{score['score']:.3f} | {fname}"
            entry = (score["path"], caption)
            
            if side == "best":
                best.append(entry)
            else:
                normal.append(entry)
        return best, normal

    def gallery_update(items):
        """Refresh gallery contents and clear any selected/zoomed preview."""
        return gr.update(value=items, selected_index=None)

    def clear_selection_state():
        state["best_sel"] = None
        state["normal_sel"] = None

    def status_line(best, normal, scores, overrides):
        n_manual = len(overrides)
        n_failed = sum(1 for v in scores.values() if v["score"] < -1000)
        n_total = len(scores)
        s = f"⬡  {len(best)} best  /  {len(normal)} normal  ({n_total} total)"
        if n_manual:
            s += f"  ✋ {n_manual} manual override{'s' if n_manual != 1 else ''}"
        if n_failed:
            s += f"  ⚠ {n_failed} failed"
        return s

    def slider_range(scores):
        """Get min/max/mid values for threshold slider."""
        vals = [v["score"] for v in scores.values() if v["score"] > -1000]
        if not vals:
            lo, hi = DEFAULT_SCORE_RANGE
            return lo, hi, THRESHOLD
        lo = round(min(vals) - 0.05, 3)
        hi = round(max(vals) + 0.05, 3)
        mid = round((lo + hi) / 2.0, 3)
        return lo, hi, mid

    def get_threshold_for_percentile(scores, percentile):
        """Calculate threshold to keep top N% of images (by score)."""
        vals = sorted([v["score"] for v in scores.values() if v["score"] > -1000], reverse=True)
        if not vals:
            return 0.0
        if percentile <= 0:
            return max(vals) + 0.01
        if percentile >= 100:
            return min(vals) - 0.01
        idx = max(0, int(len(vals) * (percentile / 100.0)) - 1)
        return round(vals[idx], 3)

    def make_histogram(scores, threshold):
        """Render histogram as PIL Image."""
        from PIL import ImageDraw, ImageFont
        vals = [v["score"] for v in scores.values() if v["score"] > -1000]
        if not vals:
            return None

        # Bin the scores
        lo, hi = min(vals), max(vals)
        if lo == hi:
            lo -= 0.05
            hi += 0.05
        n = 32
        w = (hi - lo) / n
        counts = [0] * n
        for v in vals:
            idx = min(int((v - lo) / w), n - 1)
            counts[idx] += 1

        # Drawing
        W, H = 300, 130
        PAD_L, PAD_R = 38, 8
        PAD_TOP, PAD_BOT = 18, 22
        cW = W - PAD_L - PAD_R

        img = Image.new("RGB", (W, H), "#0d0d11")
        d = ImageDraw.Draw(img)

        # Background
        d.rectangle([PAD_L, PAD_TOP, W - PAD_R, PAD_TOP + H - PAD_BOT], fill="#0f0f16")

        # Bars
        max_c = max(counts) if counts else 1
        bw = cW / n
        for i, c in enumerate(counts):
            if c == 0:
                continue
            bh = max(1, int((c / max_c) * (H - PAD_BOT - PAD_TOP - 2)))
            x0b = PAD_L + int(i * bw) + 1
            x1b = PAD_L + int((i + 1) * bw) - 1
            d.rectangle([x0b, PAD_TOP + (H - PAD_BOT - bh), x1b, PAD_TOP + H - PAD_BOT], fill="#3a7a3a")

        # Threshold line (dashed)
        tx = PAD_L + int(((threshold - lo) / (hi - lo)) * cW)
        tx = max(PAD_L, min(W - PAD_R, tx))
        for yy in range(PAD_TOP, PAD_TOP + H - PAD_BOT, 6):
            d.line([(tx, yy), (tx, min(yy + 3, PAD_TOP + H - PAD_BOT))], fill="#aadd66", width=2)

        # X-axis labels
        for frac, val in [(0.0, lo), (0.5, (lo+hi)/2), (1.0, hi)]:
            lx = PAD_L + int(frac * cW)
            d.text((lx, PAD_TOP + H - PAD_BOT + 4), f"{val:.3f}", fill="#667755", anchor="mt")

        # Title
        d.text((PAD_L, PAD_TOP - 14), f"ImageReward  threshold: {threshold:.3f}", fill="#99bb88")

        # Store geometry in state so on_hist_click can use it
        state["hist_geom"] = {
            "W": W, "H": H, "PAD_L": PAD_L, "PAD_R": PAD_R,
            "PAD_TOP": PAD_TOP, "PAD_BOT": PAD_BOT, "cW": cW,
            "lo": lo, "hi": hi,
        }

        return img

    # ── Gradio event handlers ───────────────────────────────────────────────
    def rescore(new_prompt, threshold, progress=gr.Progress()):
        """Re-score with a new prompt."""
        new_prompt = new_prompt.strip()
        if not new_prompt:
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                state["progress_html"],
                "⚠️ Prompt cannot be empty.",
                gr.update(),
                gr.update(),
            )
            return

        state["progress_html"] = render_progress(0, 0, "Preparing re-score")
        yield (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            state["progress_html"],
            "⏳ Scoring with new prompt…",
            gr.update(),
            gr.update(),
        )

        try:
            fresh_paths = scan_image_paths()
            state["image_paths"] = fresh_paths
            if not fresh_paths:
                state["progress_html"] = render_progress(0, 0, "No images loaded")
                yield (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    state["progress_html"],
                    "⚠️ No images found.",
                    gr.update(),
                    gr.update(),
                )
                return

            progress(0, desc=f"Scoring {len(fresh_paths)} images…")
            new_scores = {}
            for event in iter_imagereward_scores(fresh_paths, state["model"], state["device"], new_prompt):
                label = f"Scoring {event['total']} images (batch {event['batch_size']})"
                if event["type"] == "oom":
                    label = f"CUDA OOM, retrying with batch {event['batch_size']}"
                    state["progress_html"] = render_progress(event["done"], event["total"], label)
                    progress(event["done"] / max(event["total"], 1), desc=label)
                    yield (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        state["progress_html"],
                        f"⚠️ {label}",
                        gr.update(),
                        gr.update(),
                    )
                    continue

                new_scores = event["scores"]
                state["progress_html"] = render_progress(event["done"], event["total"], label)
                progress(event["done"] / max(event["total"], 1), desc=label)
                yield (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    state["progress_html"],
                    f"⏳ Scoring… {event['done']}/{event['total']}",
                    gr.update(),
                    gr.update(),
                )

            state["scores"] = new_scores
        except Exception as e:
            state["progress_html"] = render_progress(0, 0, "Error")
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                state["progress_html"],
                f"❌ Error: {e}",
                gr.update(),
                gr.update(),
            )
            return

        lo, hi, mid = slider_range(new_scores)
        clear_selection_state()
        best, normal = build_split(new_scores, state["overrides"], mid)
        status = status_line(best, normal, new_scores, state["overrides"])
        hist = make_histogram(new_scores, mid)
        state["progress_html"] = render_progress(len(fresh_paths), len(fresh_paths), "Scoring complete")

        yield (
            f"### 🏆 BEST — {len(best)} images",
            gallery_update(best),
            f"### ⚫ NORMAL — {len(normal)} images",
            gallery_update(normal),
            state["progress_html"],
            status,
            hist,
            gr.update(minimum=lo, maximum=hi, value=mid),
        )

    def change_folder(new_path, threshold, progress=gr.Progress()):
        """Load a new image folder."""
        new_path = (new_path or "").strip()
        if not new_path or not os.path.isdir(new_path):
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                state["progress_html"],
                f"⚠️ Invalid folder: {new_path!r}",
                gr.update(),
                gr.update(),
            )
            return

        state["source_dir"] = new_path
        state["progress_html"] = render_progress(0, 0, f"Scanning {new_path}")
        yield (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            state["progress_html"],
            f"⏳ Scanning {new_path}…",
            gr.update(),
            gr.update(),
        )

        fresh_paths = scan_image_paths()
        if not fresh_paths:
            state["progress_html"] = render_progress(0, 0, "No images loaded")
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                state["progress_html"],
                f"⚠️ No images in {new_path}",
                gr.update(),
                gr.update(),
            )
            return

        state["image_paths"] = fresh_paths
        progress(0, desc=f"Scoring {len(fresh_paths)} images…")
        wrapped_scores = {}
        for event in iter_imagereward_scores(fresh_paths, state["model"], state["device"], IR_PROMPT):
            label = f"Scoring {event['total']} images (batch {event['batch_size']})"
            if event["type"] == "oom":
                label = f"CUDA OOM, retrying with batch {event['batch_size']}"
                state["progress_html"] = render_progress(event["done"], event["total"], label)
                progress(event["done"] / max(event["total"], 1), desc=label)
                yield (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    state["progress_html"],
                    f"⚠️ {label}",
                    gr.update(),
                    gr.update(),
                )
                continue
            wrapped_scores = event["scores"]
            state["progress_html"] = render_progress(event["done"], event["total"], label)
            progress(event["done"] / max(event["total"], 1), desc=label)
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                state["progress_html"],
                f"⏳ Scoring… {event['done']}/{event['total']}",
                gr.update(),
                gr.update(),
            )

        state["scores"] = wrapped_scores
        progress(1, desc="Done")

        lo, hi, mid = slider_range(wrapped_scores)
        clear_selection_state()
        best, normal = build_split(wrapped_scores, state["overrides"], mid)
        status = status_line(best, normal, wrapped_scores, state["overrides"])
        hist = make_histogram(wrapped_scores, mid)
        state["progress_html"] = render_progress(len(fresh_paths), len(fresh_paths), "Scoring complete")

        yield (
            f"### 🏆 BEST — {len(best)} images",
            gallery_update(best),
            f"### ⚫ NORMAL — {len(normal)} images",
            gallery_update(normal),
            state["progress_html"],
            status,
            hist,
            gr.update(minimum=lo, maximum=hi, value=mid),
        )

    def update(threshold):
        """Threshold slider changed — re-split without re-scoring."""
        clear_selection_state()
        best, normal = build_split(state["scores"], state["overrides"], threshold)
        hist = make_histogram(state["scores"], threshold) if state["scores"] else None
        return (
            f"### 🏆 BEST — {len(best)} images",
            gallery_update(best),
            f"### ⚫ NORMAL — {len(normal)} images",
            gallery_update(normal),
            state["progress_html"],
            status_line(best, normal, state["scores"], state["overrides"]),
            hist,
        )

    def on_best_select(sel: gr.SelectData, threshold):
        """Track which image is selected in BEST gallery."""
        best, _ = build_split(state["scores"], state["overrides"], threshold)
        state["best_sel"] = os.path.basename(best[sel.index][0]) if sel.index < len(best) else None
        state["normal_sel"] = None
        fname = state["best_sel"] or "?"
        return gr.update(value=f"Selected in BEST: **{fname}**")

    def on_normal_select(sel: gr.SelectData, threshold):
        """Track which image is selected in NORMAL gallery."""
        _, normal = build_split(state["scores"], state["overrides"], threshold)
        state["normal_sel"] = os.path.basename(normal[sel.index][0]) if sel.index < len(normal) else None
        state["best_sel"] = None
        fname = state["normal_sel"] or "?"
        return gr.update(value=f"Selected in NORMAL: **{fname}**")

    def move_to_normal(threshold):
        """Move selected BEST image → NORMAL."""
        fname = state.get("best_sel")
        if fname:
            state["overrides"][fname] = "normal"
        state["best_sel"] = None
        state["normal_sel"] = None
        return update(threshold) + (gr.update(value=""),)

    def move_to_best(threshold):
        """Move selected NORMAL image → BEST."""
        fname = state.get("normal_sel")
        if fname:
            state["overrides"][fname] = "best"
        state["normal_sel"] = None
        state["best_sel"] = None
        return update(threshold) + (gr.update(value=""),)

    def on_hist_click(sel: gr.SelectData, threshold):
        """User clicked histogram — convert pixel coords to new threshold value."""
        geom = state.get("hist_geom")
        if not geom:
            return (gr.update(),) * 8
        try:
            cx, cy = sel.index   # pixel coords within the image
            W = geom["W"]
            H = geom["H"]
            PAD_L = geom["PAD_L"]
            PAD_R = geom["PAD_R"]
            PAD_TOP = geom["PAD_TOP"]
            PAD_BOT = geom["PAD_BOT"]
            cW = geom["cW"]
            lo = geom["lo"]
            hi = geom["hi"]

            # Check if click is within histogram bounds
            if cy >= PAD_TOP and cy <= H - PAD_BOT:
                # Convert pixel X to threshold value
                val = lo + ((cx - PAD_L) / cW) * (hi - lo)
                threshold = round(max(lo, min(hi, val)), 3)
            else:
                return (gr.update(),) * 8
        except Exception as e:
            print(f"[hist click] error: {e}")
            return (gr.update(),) * 8

        # Re-split with new threshold and update all displays
        clear_selection_state()
        best, normal = build_split(state["scores"], state["overrides"], threshold)
        hist = make_histogram(state["scores"], threshold) if state["scores"] else None
        return (
            f"### 🏆 BEST — {len(best)} images",
            gallery_update(best),
            f"### ⚫ NORMAL — {len(normal)} images",
            gallery_update(normal),
            state["progress_html"],
            status_line(best, normal, state["scores"], state["overrides"]),
            hist,
            gr.update(value=threshold),
        )

    def export_files(threshold):
        """Export BEST and NORMAL into folders."""
        best, normal = build_split(state["scores"], state["overrides"], threshold)
        failed = [f for f, v in state["scores"].items() if v["score"] < -1000]
        base = state.get("source_dir", source_dir)
        best_dir = os.path.join(base, "best")
        normal_dir = os.path.join(base, "normal")

        for d in (best_dir, normal_dir):
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    os.remove(fp)

        for path, _ in best:
            shutil.copy2(path, os.path.join(best_dir, os.path.basename(path)))
        for path, _ in normal:
            shutil.copy2(path, os.path.join(normal_dir, os.path.basename(path)))
        for fname in failed:
            src = state["scores"][fname]["path"]
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(normal_dir, fname))

        return (f"✅ {len(best)} → {best_dir}\n"
                f"✅ {len(normal)+len(failed)} → {normal_dir}")

    def set_threshold_from_percentile(percentile, threshold):
        """User changed percentile slider — calculate new threshold."""
        new_threshold = get_threshold_for_percentile(state["scores"], percentile)
        clear_selection_state()
        best, normal = build_split(state["scores"], state["overrides"], new_threshold)
        hist = make_histogram(state["scores"], new_threshold) if state["scores"] else None
        return (
            f"### 🏆 BEST — {len(best)} images",
            gallery_update(best),
            f"### ⚫ NORMAL — {len(normal)} images",
            gallery_update(normal),
            state["progress_html"],
            status_line(best, normal, state["scores"], state["overrides"]),
            hist,
            gr.update(value=new_threshold),
        )

    # ── initial values ──────────────────────────────────────────────────────
    lo, hi, mid = slider_range(initial_scores)

    # ── CSS ─────────────────────────────────────────────────────────────────
    css = """
    body, .gradio-container { background:#0d0d11 !important; color:#ddd8cc !important; }
    .gr-block,.gr-box,.panel { background:#14141c !important; border-color:#252530 !important; }

    /* Full browser width */
    .gradio-container { max-width: 100% !important; padding: 8px 12px !important; }
    .main { max-width: 100% !important; }
    footer { display: none !important; }

    h1 { font-family:'Courier New',monospace; letter-spacing:.18em; color:#aadd66;
         text-transform:uppercase; margin:0; font-size:1.4rem; }
    .subhead { color:#667755; font-family:monospace; font-size:.78em; margin-top:3px; }

    .folder-box { background:#1a2a1a !important; border:1px solid #335533 !important;
                  border-radius:8px; padding:10px 14px; margin-bottom:8px; }
    .folder-box label { color:#aadd88 !important; font-family:monospace !important;
                        font-size:.82em !important; text-transform:uppercase; }
    .folder-input textarea, .folder-input input {
        background:#0d160d !important; color:#ddd8cc !important;
        border:1px solid #334433 !important;
        font-family:monospace !important; font-size:.82em !important; }
    .folder-btn { background:#1a3a1a !important; color:#aadd88 !important;
                  border:1px solid #447744 !important;
                  font-family:monospace !important; width:100% !important;
                  margin-top:4px !important; }
    .folder-btn:hover { background:#225522 !important; }

    .reprompt-box { background:#1a1a26 !important; border:1px solid #2e2e44 !important;
                    border-radius:8px; padding:12px 14px; margin-bottom:8px; }
    .reprompt-box label { color:#aadd66 !important; font-family:monospace !important;
                          font-size:.82em !important; text-transform:uppercase; }
    .rescore-btn { background:#1e3a12 !important; color:#aadd66 !important;
                   border:1px solid #4a8820 !important; font-family:monospace !important; }
    .rescore-btn:hover { background:#2a5018 !important; }

    .status-md  { font-family:monospace; font-size:.82em; color:#88bb55;
                  background:#181820; border-radius:4px; padding:5px 10px; margin-top:6px; }
    .export-btn { background:#0e2a30 !important; color:#55ccdd !important;
                  border:1px solid #227788 !important; }
    .export-box textarea { font-family:monospace !important; font-size:.78em !important;
                           color:#55ddaa !important; background:#0d1a1c !important; }

    input[type=range] { accent-color:#aadd66; }

    .move-btn {
        font-family: monospace !important; font-size: .8em !important;
        padding: 4px 10px !important; border-radius: 4px !important;
    }
    .move-to-normal { background:#2a1010 !important; color:#ee8866 !important;
                      border:1px solid #663322 !important; }
    .move-to-best  { background:#102a10 !important; color:#88ee66 !important;
                     border:1px solid #226633 !important; }

    .best-head    { color:#88ee66 !important; font-family:monospace !important; }
    .normal-head  { color:#ee6655 !important; font-family:monospace !important; }

    .grid-wrap img { object-fit: contain !important; background: #0a0a12; }
    .grid-wrap .caption-label span,
    .grid-wrap span.svelte-1dv1zt9,
    .grid-wrap [class*="caption"] {
        font-family: monospace !important;
        font-size: .72em !important;
        color: #8899aa !important;
    }

    .grid-wrap img[alt^="✋"] {
        outline: 3px solid #dd3322 !important;
        outline-offset: -3px;
    }

    .move-col { display:flex; flex-direction:column; align-items:center;
                justify-content:center; gap:8px; padding:8px 4px;
                background:#0f0f16; border-radius:8px;
                border:1px solid #252535; }
    .move-col-label { font-family:monospace; font-size:.7em; color:#556644;
                      text-transform:uppercase; letter-spacing:.1em; }
    .move-hint { font-family:monospace; font-size:.65em; color:#445533;
                 text-align:center; line-height:1.4; }
    .sel-info p { font-family:monospace !important; font-size:.72em !important;
                  color:#aabb88 !important; text-align:center;
                  word-break:break-all; }
    .move-to-normal { background:#2a1010 !important; color:#ee8866 !important;
                      border:1px solid #663322 !important;
                      font-family:monospace !important; width:100% !important; }
    .move-to-best  { background:#102a10 !important; color:#88ee66 !important;
                     border:1px solid #226633 !important;
                     font-family:monospace !important; width:100% !important; }
    .move-to-normal:hover { background:#3a1a10 !important; }
    .move-to-best:hover  { background:#1a3a10 !important; }

    .hist-img img { cursor:crosshair !important; border-radius:6px; }
    .hist-img { margin-top:4px; }
    .progress-wrap { margin-top:8px; padding:8px 10px; background:#14141c;
                     border:1px solid #2a2a36; border-radius:8px; }
    .progress-label, .progress-meta { font-family:monospace; font-size:.76em; color:#aabb88; }
    .progress-meta { color:#778866; margin-top:4px; }
    .progress-track { margin-top:6px; height:10px; background:#0c0c12;
                      border-radius:999px; overflow:hidden; border:1px solid #2d3a22; }
    .progress-fill { height:100%; background:linear-gradient(90deg, #5c8f32, #aadd66); }
    """

    # ── layout ──────────────────────────────────────────────────────────────
    with gr.Blocks(title="ImageReward Scorer") as demo:

        gr.HTML("""
<h1>⬡ ImageReward</h1>
<div class='subhead'>Aesthetic preference scorer — interactive gallery
 &middot; click image to select, then use ← → buttons to move &middot; ✋ = manual override
</div>
""")

        with gr.Row(equal_height=False):

            # ── sidebar ──────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=290):

                with gr.Group(elem_classes=["folder-box"]):
                    gr.Markdown("#### 📁 Image folder")
                    folder_input = gr.Textbox(
                        value=source_dir, label="", show_label=False,
                        placeholder="/path/to/images", lines=1,
                        elem_classes=["folder-input"],
                    )
                    folder_btn = gr.Button("📂  Load folder & re-score",
                                           elem_classes=["folder-btn"])

                with gr.Group(elem_classes=["reprompt-box"]):
                    gr.Markdown("#### ⚡ Re-score with new prompt")
                    new_prompt = gr.Textbox(value=IR_PROMPT,
                                            label="Prompt",
                                            lines=3,
                                            placeholder="Describe the aesthetic you want to score...")
                    rescore_btn = gr.Button("⚡  Re-score images",
                                            elem_classes=["rescore-btn"])

                threshold_slider = gr.Slider(
                    minimum=lo, maximum=hi,
                    value=mid, step=0.01,
                    label="Score threshold  (≥ → BEST)",
                    interactive=True
                )

                percentile_slider = gr.Slider(
                    minimum=0, maximum=100,
                    value=50, step=1,
                    label="Or select top N%",
                    interactive=True
                )

                hist_plot = gr.Image(
                    value=make_histogram(initial_scores, mid) if initial_scores else None,
                    show_label=False,
                    interactive=False,
                    elem_classes=["hist-img"],
                )

                progress_html = gr.HTML(value=state["progress_html"])

                status = gr.Markdown(value="", elem_classes=["status-md"])

                export_btn = gr.Button("💾  Export folders",
                                       elem_classes=["export-btn"])
                export_result = gr.Textbox(label="Export result", lines=3,
                                           interactive=False,
                                           elem_classes=["export-box"])

            # ── gallery columns ───────────────────────────────────────────────
            with gr.Column(scale=5):

                with gr.Row():
                    best_head    = gr.Markdown("### 🏆 BEST",
                                               elem_classes=["best-head"])
                    normal_head  = gr.Markdown("### ⚫ NORMAL",
                                               elem_classes=["normal-head"])

                with gr.Row(equal_height=True):
                    best_gallery = gr.Gallery(
                        label="Best", show_label=False,
                        columns=5, height="80vh",
                        object_fit="contain",
                        allow_preview=True,
                        preview=True,
                        elem_classes=["grid-wrap"],
                    )

                    # ── centre move column ────────────────────────────────
                    with gr.Column(scale=0, min_width=90,
                                   elem_classes=["move-col"]):
                        gr.HTML("<div class='move-col-label'>MOVE</div>")
                        sel_info     = gr.Markdown("", elem_classes=["sel-info"])
                        move_norm_btn = gr.Button("→ NM", elem_classes=["move-btn","move-to-normal"])
                        move_best_btn = gr.Button("B ←",  elem_classes=["move-btn","move-to-best"])
                        gr.HTML("<div class='move-hint'>click image<br>then button</div>")

                    normal_gallery = gr.Gallery(
                        label="Normal", show_label=False,
                        columns=5, height="80vh",
                        object_fit="contain",
                        allow_preview=True,
                        preview=True,
                        elem_classes=["grid-wrap"],
                    )

        # ── wiring ───────────────────────────────────────────────────────────
        all_out = [best_head, best_gallery, normal_head, normal_gallery, progress_html, status, hist_plot]
        move_out = all_out + [sel_info]

        threshold_slider.change(fn=update, inputs=[threshold_slider], outputs=all_out)

        # percentile slider — calculate threshold from top N%
        percentile_out = all_out + [threshold_slider]
        percentile_slider.change(fn=set_threshold_from_percentile,
                                 inputs=[percentile_slider, threshold_slider],
                                 outputs=percentile_out)

        best_gallery.select(fn=on_best_select, inputs=[threshold_slider], outputs=[sel_info])
        normal_gallery.select(fn=on_normal_select, inputs=[threshold_slider], outputs=[sel_info])

        move_norm_btn.click(fn=move_to_normal, inputs=[threshold_slider], outputs=move_out)
        move_best_btn.click(fn=move_to_best, inputs=[threshold_slider], outputs=move_out)

        # histogram click → set threshold via gr.Image .select() pixel coords
        hist_click_out = [best_head, best_gallery, normal_head, normal_gallery, progress_html, status, hist_plot, threshold_slider]
        hist_plot.select(
            fn=on_hist_click,
            inputs=[threshold_slider],
            outputs=hist_click_out,
        )

        rescore_out = [best_head, best_gallery, normal_head, normal_gallery,
                       progress_html, status, hist_plot, threshold_slider]
        rescore_btn.click(
            fn=rescore,
            inputs=[new_prompt, threshold_slider],
            outputs=rescore_out,
        )

        folder_out = [best_head, best_gallery, normal_head, normal_gallery,
                      progress_html, status, hist_plot, threshold_slider]
        folder_btn.click(
            fn=change_folder,
            inputs=[folder_input, threshold_slider],
            outputs=folder_out,
        )

        export_btn.click(fn=export_files, inputs=[threshold_slider], outputs=export_result)

        demo.load(fn=update, inputs=[threshold_slider], outputs=all_out)

    demo.launch(server_name="0.0.0.0", server_port=7860,
                inbrowser=True, share=False,
                css=css,
                allowed_paths=["/"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== ImageReward — Interactive Aesthetic Scorer ===\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    # Load ImageReward model
    print("\nLoading ImageReward v1.0…")
    try:
        model = RM.load("ImageReward-v1.0")
    except Exception as e:
        sys.exit(f"Failed to load ImageReward: {e}\nTry: pip install image-reward")
    print("ImageReward Ready.")

    # Try to find default images folder
    source_dir = os.path.join(script_dir, INPUT_FOLDER_NAME)
    image_paths = []
    scores = {}

    if os.path.isdir(source_dir):
        image_paths = [
            os.path.join(source_dir, f)
            for f in sorted(os.listdir(source_dir))
            if f.lower().endswith(ALLOWED_EXTENSIONS)
            and os.path.isfile(os.path.join(source_dir, f))
        ]

    if image_paths:
        print(f"\nFound {len(image_paths)} images in {source_dir}. Scoring…")
        scores_raw = calculate_imagereward_scores(image_paths, model, device)
        scores = {}
        for fname, score_val in scores_raw.items():
            if fname in [os.path.basename(p) for p in image_paths]:
                path = next(p for p in image_paths if os.path.basename(p) == fname)
                scores[fname] = {"score": score_val, "path": path}
        print(f"Scoring done.")
    else:
        print(f"\nNo default images folder found — use the 📁 folder selector in the UI.")
        source_dir = script_dir

    print(f"Launching browser UI on http://localhost:7860 …\n")
    launch_ui(scores, image_paths, model, device, script_dir, source_dir)
