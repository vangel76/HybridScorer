"""
Prompt-Match Image Sorter  —  Interactive GUI
----------------------------------------------
- Scores images with CLIP against a positive (and optional negative) prompt
- Opens a browser UI with live threshold sliders
- Drag the slider → images instantly move between FOUND / NOT FOUND columns
- Run:  python promptmatch.py
"""

import os, sys, math, threading
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEARCH_PROMPT      = "woman"
NEGATIVE_PROMPT    = ""           # set to "" to disable

POSITIVE_THRESHOLD = 0.22
NEGATIVE_THRESHOLD = 0.22

INPUT_FOLDER_NAME  = "images"
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
BATCH_SIZE         = 64
CLIP_MODEL         = "ViT-L/14"   # "ViT-B/32" if VRAM is tight

THUMB_SIZE         = 220          # px — thumbnail size in the gallery
COPY_ON_EXPORT     = True         # create found/ notfound/ folders on export

# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------
def load_clip(device):
    try:
        import clip
    except ImportError:
        sys.exit("CLIP not installed.\nRun: pip install git+https://github.com/openai/CLIP.git")
    print(f"[CLIP] Loading {CLIP_MODEL} …")
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()
    print("[CLIP] Ready.")
    return model, preprocess, clip


def encode_prompt(clip_mod, model, device, prompt):
    tokens = clip_mod.tokenize([
        f"a photo of a {prompt}",
        f"a photo of {prompt}",
        prompt,
    ]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = F.normalize(feat, dim=-1)
        mean = F.normalize(feat.mean(dim=0, keepdim=True), dim=-1)
    return mean


def score_all(image_paths, model, preprocess, clip_mod, device, pos_emb, neg_emb):
    results = {}   # filename → {"pos": float, "neg": float|None, "path": str}

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Scoring"):
        batch = image_paths[i:i + BATCH_SIZE]
        imgs, valid = [], []
        for p in batch:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
                valid.append(p)
            except Exception as e:
                print(f"  [WARN] {p}: {e}")
                results[os.path.basename(p)] = {"pos": -1.0, "neg": None, "path": p}

        if not imgs:
            continue

        tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feat = F.normalize(model.encode_image(tensor), dim=-1)

        pos_sims = (feat @ pos_emb.T).squeeze(1).tolist()
        neg_sims = (feat @ neg_emb.T).squeeze(1).tolist() if neg_emb is not None else [None]*len(valid)

        for path, ps, ns in zip(valid, pos_sims, neg_sims):
            fname = os.path.basename(path)
            results[fname] = {
                "pos":  float(ps),
                "neg":  float(ns) if ns is not None else None,
                "path": path,
            }

        if device == "cuda":
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def launch_ui(initial_scores: dict, image_paths: list, clip_model, preprocess,
              clip_mod, device, script_dir: str):
    try:
        import gradio as gr
    except ImportError:
        sys.exit("Gradio not installed.\nRun: pip install gradio")

    import shutil, statistics

    # Mutable state — updated when user re-scores with new prompts
    state = {"scores": initial_scores}

    # ------------------------------------------------------------------ #
    def build_items(scores):
        return sorted(
            [(f, v) for f, v in scores.items() if v["pos"] >= 0],
            key=lambda x: -x[1]["pos"]
        )

    def get_failed(scores):
        return [f for f, v in scores.items() if v["pos"] < 0]

    def get_ranges(items):
        pos_vals = [v["pos"] for _, v in items]
        neg_vals = [v["neg"] for _, v in items if v["neg"] is not None]
        pos_min = round(min(pos_vals) - 0.01, 3) if pos_vals else 0.0
        pos_max = round(max(pos_vals) + 0.01, 3) if pos_vals else 1.0
        neg_min = round(min(neg_vals) - 0.01, 3) if neg_vals else 0.0
        neg_max = round(max(neg_vals) + 0.01, 3) if neg_vals else 1.0
        return pos_vals, neg_vals, pos_min, pos_max, neg_min, neg_max

    def stat_line(label, vals):
        if not vals:
            return ""
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"**{label}** — mean `{m:.4f}`  stdev `{s:.4f}`  range `{min(vals):.4f}` – `{max(vals):.4f}`"

    def make_stats_md(pos_prompt, neg_prompt, pos_vals, neg_vals):
        lines = [stat_line(f"🟢 `{pos_prompt}`", pos_vals)]
        if neg_prompt and neg_vals:
            lines.append(stat_line(f"🔴 `{neg_prompt}`", neg_vals))
        return "\n\n".join(filter(None, lines)) or "*No stats available*"

    # ------------------------------------------------------------------ #
    def split(scores_dict, pos_thresh, neg_thresh):
        items = build_items(scores_dict)
        found, notfound = [], []
        for fname, v in items:
            pos_ok = v["pos"] >= pos_thresh
            neg_ok = (v["neg"] is None) or (v["neg"] < neg_thresh)
            if pos_ok and neg_ok:
                found.append((v["path"], f"pos={v['pos']:.3f}"))
            else:
                notfound.append((v["path"], f"pos={v['pos']:.3f}"))
        return found, notfound

    def update(pos_thresh, neg_thresh):
        found, notfound = split(state["scores"], pos_thresh, neg_thresh)
        failed = get_failed(state["scores"])
        return (
            f"### ✅ FOUND — {len(found)} images",
            found or [],
            f"### ❌ NOT FOUND — {len(notfound)} images",
            notfound or [],
            f"⬡  {len(found)} found  /  {len(notfound)} not found  /  {len(failed)} failed",
        )

    def rescore(new_pos_prompt, new_neg_prompt, pos_thresh, neg_thresh):
        """Re-encode prompts and re-score all images — no restart needed."""
        new_pos_prompt = new_pos_prompt.strip()
        new_neg_prompt = new_neg_prompt.strip()
        if not new_pos_prompt:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                "⚠️  Positive prompt cannot be empty.",
            )

        yield (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            "⏳  Encoding prompts and scoring — please wait…",
        )

        try:
            pos_emb = encode_prompt(clip_mod, clip_model, device, new_pos_prompt)
            neg_emb = encode_prompt(clip_mod, clip_model, device, new_neg_prompt) if new_neg_prompt else None
            new_scores = score_all(image_paths, clip_model, preprocess, clip_mod,
                                   device, pos_emb, neg_emb)
            state["scores"] = new_scores
        except Exception as e:
            yield (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                f"❌ Scoring error: {e}",
            )
            return

        items = build_items(new_scores)
        pos_vals, neg_vals, pos_min, pos_max, neg_min, neg_max = get_ranges(items)
        stats = make_stats_md(new_pos_prompt, new_neg_prompt, pos_vals, neg_vals)

        found, notfound = split(new_scores, pos_thresh, neg_thresh)
        failed = get_failed(new_scores)

        yield (
            f"### ✅ FOUND — {len(found)} images",
            found or [],
            f"### ❌ NOT FOUND — {len(notfound)} images",
            notfound or [],
            stats,
            gr.update(minimum=pos_min, maximum=pos_max, value=pos_thresh),
            gr.update(minimum=neg_min if neg_vals else 0,
                      maximum=neg_max if neg_vals else 1,
                      value=neg_thresh,
                      interactive=bool(new_neg_prompt)),
            f"✅  Rescored {len(items)} images  |  {len(found)} found / {len(notfound)} not found / {len(failed)} failed",
        )

    def export(pos_thresh, neg_thresh):
        found, notfound = split(state["scores"], pos_thresh, neg_thresh)
        failed = get_failed(state["scores"])
        found_dir    = os.path.join(script_dir, "found")
        notfound_dir = os.path.join(script_dir, "notfound")
        os.makedirs(found_dir,    exist_ok=True)
        os.makedirs(notfound_dir, exist_ok=True)
        for d in (found_dir, notfound_dir):
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        for path, _ in found:
            shutil.copy2(path, os.path.join(found_dir, os.path.basename(path)))
        for path, _ in notfound:
            shutil.copy2(path, os.path.join(notfound_dir, os.path.basename(path)))
        for fname in failed:
            src = state["scores"][fname]["path"]
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(notfound_dir, fname))
        return (
            f"✅ Exported {len(found)} → found/   |   "
            f"{len(notfound)+len(failed)} → notfound/\n"
            f"Path: {script_dir}"
        )

    # ------------------------------------------------------------------ #
    # Initial ranges
    init_items = build_items(initial_scores)
    pos_vals, neg_vals, pos_min, pos_max, neg_min, neg_max = get_ranges(init_items)
    init_stats = make_stats_md(SEARCH_PROMPT, NEGATIVE_PROMPT, pos_vals, neg_vals)

    css = """
    body, .gradio-container { background: #0d0d11 !important; color: #ddd8cc !important; }
    .gr-block, .gr-box      { background: #14141c !important; border-color: #252530 !important; }
    .panel                  { background: #14141c !important; }

    h1 { font-family: 'Courier New', monospace; letter-spacing: 0.18em;
         color: #aadd66; text-transform: uppercase; margin: 0; font-size: 1.5rem; }
    .subhead { color: #667755; font-family: monospace; font-size: 0.8em; margin-top: 3px; }

    /* sidebar panel */
    .sidebar { padding: 12px !important; }

    /* re-prompt box */
    .reprompt-box { background: #1a1a26 !important; border: 1px solid #2e2e44 !important;
                    border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
    .reprompt-box label { color: #aadd66 !important; font-family: monospace !important;
                          font-size: 0.82em !important; text-transform: uppercase; }
    .reprompt-box input  { background: #0d0d16 !important; color: #ddd8cc !important;
                           border: 1px solid #333348 !important; font-family: monospace !important; }
    .rescore-btn { background: #1e3a12 !important; color: #aadd66 !important;
                   border: 1px solid #4a8820 !important; font-family: monospace !important;
                   font-size: 0.85em !important; }
    .rescore-btn:hover { background: #2a5018 !important; }

    /* status / export */
    .status-md  { font-family: monospace; font-size: 0.85em; color: #88bb55;
                  background: #181820; border-radius: 4px; padding: 5px 10px; }
    .export-btn { background: #0e2a30 !important; color: #55ccdd !important;
                  border: 1px solid #227788 !important; }
    .export-box textarea { font-family: monospace !important; font-size: 0.8em !important;
                           color: #55ddaa !important; background: #0d1a1c !important; }

    /* sliders */
    input[type=range] { accent-color: #aadd66; }

    /* gallery headers */
    .found-head    { color: #88ee66 !important; font-family: monospace !important; }
    .notfound-head { color: #ee6655 !important; font-family: monospace !important; }

    /* make galleries fill vertical space */
    .gallery-wrap .gr-gallery { min-height: 82vh !important; }
    """

    with gr.Blocks(title="PromptMatch", css=css, theme=gr.themes.Base()) as demo:

        gr.HTML("<h1>⬡ PromptMatch</h1><div class='subhead'>CLIP-powered interactive image sorter</div>")

        with gr.Row(equal_height=False):

            # ── SIDEBAR ──────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280, elem_classes=["sidebar"]):

                # Re-prompt section
                with gr.Group(elem_classes=["reprompt-box"]):
                    gr.Markdown("#### 🔄 Re-score with new prompts")
                    new_pos = gr.Textbox(
                        value=SEARCH_PROMPT, label="Positive prompt",
                        placeholder="e.g. woman in red dress",
                        lines=1,
                    )
                    new_neg = gr.Textbox(
                        value=NEGATIVE_PROMPT, label="Negative prompt  (leave empty to disable)",
                        placeholder="e.g. man, cartoon, drawing",
                        lines=1,
                    )
                    rescore_btn = gr.Button("⚡  Re-score images", elem_classes=["rescore-btn"])

                stats_box = gr.Markdown(value=init_stats)

                pos_slider = gr.Slider(
                    minimum=pos_min, maximum=pos_max,
                    value=POSITIVE_THRESHOLD, step=0.001,
                    label="Positive threshold  (≥ value → FOUND)",
                    interactive=True,
                )
                neg_slider = gr.Slider(
                    minimum=neg_min if neg_vals else 0,
                    maximum=neg_max if neg_vals else 1,
                    value=NEGATIVE_THRESHOLD, step=0.001,
                    label="Negative threshold  (< value → passes)",
                    interactive=bool(NEGATIVE_PROMPT),
                )

                status = gr.Markdown(value="", elem_classes=["status-md"])

                export_btn    = gr.Button("💾  Export folders", elem_classes=["export-btn"])
                export_result = gr.Textbox(
                    label="Export result", lines=3,
                    interactive=False, elem_classes=["export-box"],
                )

            # ── GALLERIES ────────────────────────────────────────────────
            with gr.Column(scale=5, elem_classes=["gallery-wrap"]):
                with gr.Row():
                    found_label    = gr.Markdown("### ✅ FOUND", elem_classes=["found-head"])
                    notfound_label = gr.Markdown("### ❌ NOT FOUND", elem_classes=["notfound-head"])

                with gr.Row(equal_height=True):
                    found_gallery = gr.Gallery(
                        label="Found", show_label=False,
                        columns=5, height="82vh",
                        object_fit="cover",
                        allow_preview=True,
                        preview=True,
                    )
                    notfound_gallery = gr.Gallery(
                        label="Not Found", show_label=False,
                        columns=5, height="82vh",
                        object_fit="cover",
                        allow_preview=True,
                        preview=True,
                    )

        # ── Wiring ───────────────────────────────────────────────────────
        slider_inputs  = [pos_slider, neg_slider]
        gallery_outputs = [found_label, found_gallery, notfound_label, notfound_gallery, status]

        pos_slider.change(fn=update, inputs=slider_inputs, outputs=gallery_outputs)
        neg_slider.change(fn=update, inputs=slider_inputs, outputs=gallery_outputs)

        rescore_outputs = [
            found_label, found_gallery,
            notfound_label, notfound_gallery,
            stats_box, pos_slider, neg_slider,
            status,
        ]
        rescore_btn.click(
            fn=rescore,
            inputs=[new_pos, new_neg, pos_slider, neg_slider],
            outputs=rescore_outputs,
        )

        export_btn.click(fn=export, inputs=slider_inputs, outputs=export_result)

        demo.load(fn=update, inputs=slider_inputs, outputs=gallery_outputs)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PromptMatch — Interactive Sorter ===")
    print(f"  Positive : {SEARCH_PROMPT!r}")
    neg_display = repr(NEGATIVE_PROMPT) if NEGATIVE_PROMPT else "(disabled)"
    print(f"  Negative : {neg_display}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, INPUT_FOLDER_NAME)

    if not os.path.isdir(source_dir):
        sys.exit(f"Input folder not found: {source_dir}")

    image_paths = [
        os.path.join(source_dir, f)
        for f in sorted(os.listdir(source_dir))
        if f.lower().endswith(ALLOWED_EXTENSIONS)
        and os.path.isfile(os.path.join(source_dir, f))
    ]
    if not image_paths:
        sys.exit("No images found.")
    print(f"\nFound {len(image_paths)} images. Scoring…")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    clip_model, preprocess, clip_mod = load_clip(device)
    pos_emb = encode_prompt(clip_mod, clip_model, device, SEARCH_PROMPT)
    neg_emb = encode_prompt(clip_mod, clip_model, device, NEGATIVE_PROMPT) if NEGATIVE_PROMPT else None

    scores = score_all(image_paths, clip_model, preprocess, clip_mod, device, pos_emb, neg_emb)

    print(f"\nScoring done. Launching browser UI on http://localhost:7860 …\n")
    launch_ui(scores, image_paths, clip_model, preprocess, clip_mod, device, script_dir)
