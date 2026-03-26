"""
PromptMatch — Interactive Image Sorter
---------------------------------------
• CLIP scoring with positive + optional negative prompt
• Native Gradio galleries — proven to work
• object_fit="contain" — full image always visible, never cropped
• Threshold sliders → live re-sort
• Re-score panel — new prompts without restart
• Manual move: select an image in a gallery, then click ← Move or Move →
  Manual overrides survive slider changes; cleared on re-score
"""

import os, sys, json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEARCH_PROMPT      = "woman"
NEGATIVE_PROMPT    = ""

POSITIVE_THRESHOLD = 0.22
NEGATIVE_THRESHOLD = 0.22

INPUT_FOLDER_NAME  = "images"
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
BATCH_SIZE         = 64
CLIP_MODEL         = "ViT-L/14"

# ---------------------------------------------------------------------------
# CLIP helpers
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
    results = {}
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
        neg_sims = (feat @ neg_emb.T).squeeze(1).tolist() if neg_emb is not None else [None] * len(valid)
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
# UI
# ---------------------------------------------------------------------------
def launch_ui(initial_scores, image_paths, clip_model, preprocess,
              clip_mod, device, script_dir, source_dir):
    try:
        import gradio as gr
    except ImportError:
        sys.exit("Gradio not installed.\nRun: pip install gradio")

    import shutil, statistics

    # ── shared mutable state ────────────────────────────────────────────────
    state = {
        "scores":    initial_scores,
        "overrides": {},   # fname → "found" | "notfound"
        "found_sel": None,
        "nf_sel":    None,
    }

    # ── core logic ──────────────────────────────────────────────────────────
    def clip_side(v, pos_t, neg_t):
        pos_ok = v["pos"] >= pos_t
        neg_ok = (v["neg"] is None) or (v["neg"] < neg_t)
        return "found" if (pos_ok and neg_ok) else "notfound"

    def build_split(scores, overrides, pos_t, neg_t):
        """Returns (found_list, notfound_list) sorted by score desc.
        Each item is (image_path, caption_string).
        Manual-overridden images get a ✋ marker in the caption.
        """
        found, notfound = [], []
        for fname, v in sorted(scores.items(), key=lambda x: -x[1]["pos"]):
            if v["pos"] < 0:
                continue
            manual = fname in overrides
            side   = overrides[fname] if manual else clip_side(v, pos_t, neg_t)
            marker = " ✋" if manual else ""
            caption = f"{fname}  {v['pos']:.3f}{marker}"
            entry = (v["path"], caption)
            if side == "found":
                found.append(entry)
            else:
                notfound.append(entry)
        return found, notfound

    def status_line(found, notfound, scores, overrides):
        n_manual = len(overrides)
        n_failed = sum(1 for v in scores.values() if v["pos"] < 0)
        s = f"⬡  {len(found)} found  /  {len(notfound)} not found"
        if n_manual:
            s += f"  ✋ {n_manual} manual override{'s' if n_manual != 1 else ''}"
        if n_failed:
            s += f"  ⚠ {n_failed} failed"
        return s

    def stat_line(label, vals):
        if not vals:
            return ""
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"**{label}** — mean `{m:.4f}` stdev `{s:.4f}` range `{min(vals):.4f}`–`{max(vals):.4f}`"

    def make_stats(pos_prompt, neg_prompt, scores):
        pos_vals = [v["pos"] for v in scores.values() if v["pos"] >= 0]
        neg_vals = [v["neg"] for v in scores.values()
                    if v.get("neg") is not None and v["neg"] >= 0]
        lines = [stat_line(f"🟢 `{pos_prompt}`", pos_vals)]
        if neg_prompt and neg_vals:
            lines.append(stat_line(f"🔴 `{neg_prompt}`", neg_vals))
        return "\n\n".join(filter(None, lines)) or "*No stats available*"

    def slider_range(scores):
        pos_vals = [v["pos"] for v in scores.values() if v["pos"] >= 0]
        neg_vals = [v["neg"] for v in scores.values()
                    if v.get("neg") is not None and v["neg"] >= 0]
        return (
            round(min(pos_vals) - 0.01, 3) if pos_vals else 0.0,
            round(max(pos_vals) + 0.01, 3) if pos_vals else 1.0,
            round(min(neg_vals) - 0.01, 3) if neg_vals else 0.0,
            round(max(neg_vals) + 0.01, 3) if neg_vals else 1.0,
            bool(neg_vals),
        )

    # ── Gradio event handlers ───────────────────────────────────────────────
    def update(pos_t, neg_t):
        found, notfound = build_split(state["scores"], state["overrides"], pos_t, neg_t)
        return (
            f"### ✅ FOUND — {len(found)} images",
            found,
            f"### ❌ NOT FOUND — {len(notfound)} images",
            notfound,
            status_line(found, notfound, state["scores"], state["overrides"]),
        )

    def on_found_select(sel: gr.SelectData, pos_t, neg_t):
        """Track which image is selected in FOUND gallery."""
        state["found_sel"] = sel.index
        state["nf_sel"]    = None
        found, notfound = build_split(state["scores"], state["overrides"], pos_t, neg_t)
        fname = os.path.basename(found[sel.index][0]) if sel.index < len(found) else "?"
        return gr.update(value=f"Selected in FOUND: **{fname}**")

    def on_nf_select(sel: gr.SelectData, pos_t, neg_t):
        """Track which image is selected in NOT FOUND gallery."""
        state["nf_sel"]    = sel.index
        state["found_sel"] = None
        found, notfound = build_split(state["scores"], state["overrides"], pos_t, neg_t)
        fname = os.path.basename(notfound[sel.index][0]) if sel.index < len(notfound) else "?"
        return gr.update(value=f"Selected in NOT FOUND: **{fname}**")

    def move_selected_to_notfound(pos_t, neg_t):
        """Move currently selected FOUND image → NOT FOUND."""
        idx = state.get("found_sel")
        if idx is not None:
            found, _ = build_split(state["scores"], state["overrides"], pos_t, neg_t)
            if idx < len(found):
                fname = os.path.basename(found[idx][0])
                state["overrides"][fname] = "notfound"
        state["found_sel"] = None
        return update(pos_t, neg_t) + (gr.update(value=""),)

    def move_selected_to_found(pos_t, neg_t):
        """Move currently selected NOT FOUND image → FOUND."""
        idx = state.get("nf_sel")
        if idx is not None:
            _, notfound = build_split(state["scores"], state["overrides"], pos_t, neg_t)
            if idx < len(notfound):
                fname = os.path.basename(notfound[idx][0])
                state["overrides"][fname] = "found"
        state["nf_sel"] = None
        return update(pos_t, neg_t) + (gr.update(value=""),)

    def rescore(new_pos, new_neg, pos_t, neg_t):
        new_pos = new_pos.strip()
        new_neg = new_neg.strip()
        if not new_pos:
            yield (gr.update(),) * 7 + ("⚠️ Positive prompt cannot be empty.",)
            return

        yield (gr.update(),) * 7 + ("⏳ Encoding & scoring — please wait…",)

        try:
            pos_emb = encode_prompt(clip_mod, clip_model, device, new_pos)
            neg_emb = encode_prompt(clip_mod, clip_model, device, new_neg) if new_neg else None
            new_scores = score_all(image_paths, clip_model, preprocess,
                                   clip_mod, device, pos_emb, neg_emb)
        except Exception as e:
            yield (gr.update(),) * 7 + (f"❌ Error: {e}",)
            return

        state["scores"]    = new_scores
        state["overrides"] = {}   # clear manual overrides on new prompt

        pos_min, pos_max, neg_min, neg_max, has_neg = slider_range(new_scores)
        stats = make_stats(new_pos, new_neg, new_scores)
        found, notfound = build_split(new_scores, {}, pos_t, neg_t)

        yield (
            f"### ✅ FOUND — {len(found)} images",
            found,
            f"### ❌ NOT FOUND — {len(notfound)} images",
            notfound,
            stats,
            gr.update(minimum=pos_min, maximum=pos_max, value=pos_t),
            gr.update(minimum=neg_min if has_neg else 0,
                      maximum=neg_max if has_neg else 1,
                      value=neg_t, interactive=has_neg),
            status_line(found, notfound, new_scores, {}),
        )

    def export_files(pos_t, neg_t):
        found, notfound = build_split(state["scores"], state["overrides"], pos_t, neg_t)
        failed = [f for f, v in state["scores"].items() if v["pos"] < 0]
        found_dir    = os.path.join(script_dir, "found")
        notfound_dir = os.path.join(script_dir, "notfound")
        for d in (found_dir, notfound_dir):
            os.makedirs(d, exist_ok=True)
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
        return (f"✅ {len(found)} → found/   {len(notfound) + len(failed)} → notfound/\n"
                f"Path: {script_dir}")

    # ── initial values ──────────────────────────────────────────────────────
    pos_min, pos_max, neg_min, neg_max, has_neg = slider_range(initial_scores)
    init_stats = make_stats(SEARCH_PROMPT, NEGATIVE_PROMPT, initial_scores)

    # ── CSS ─────────────────────────────────────────────────────────────────
    css = """
    body, .gradio-container { background:#0d0d11 !important; color:#ddd8cc !important; }
    .gr-block,.gr-box,.panel { background:#14141c !important; border-color:#252530 !important; }

    h1 { font-family:'Courier New',monospace; letter-spacing:.18em; color:#aadd66;
         text-transform:uppercase; margin:0; font-size:1.4rem; }
    .subhead { color:#667755; font-family:monospace; font-size:.78em; margin-top:3px; }

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

    /* move buttons */
    .move-btn {
        font-family: monospace !important; font-size: .8em !important;
        padding: 4px 10px !important; border-radius: 4px !important;
    }
    .move-to-nf { background:#2a1010 !important; color:#ee8866 !important;
                  border:1px solid #663322 !important; }
    .move-to-f  { background:#102a10 !important; color:#88ee66 !important;
                  border:1px solid #226633 !important; }

    /* gallery column headers */
    .found-head    { color:#88ee66 !important; font-family:monospace !important; }
    .notfound-head { color:#ee6655 !important; font-family:monospace !important; }

    /* make gallery images show full / uncropped */
    .grid-wrap img { object-fit: contain !important; background: #0a0a12; }

    /* centre move column */
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
    .move-to-nf { background:#2a1010 !important; color:#ee8866 !important;
                  border:1px solid #663322 !important;
                  font-family:monospace !important; width:100% !important; }
    .move-to-f  { background:#102a10 !important; color:#88ee66 !important;
                  border:1px solid #226633 !important;
                  font-family:monospace !important; width:100% !important; }
    .move-to-nf:hover { background:#3a1a10 !important; }
    .move-to-f:hover  { background:#1a3a10 !important; }
    """

    # ── layout ──────────────────────────────────────────────────────────────
    with gr.Blocks(title="PromptMatch", css=css, theme=gr.themes.Base()) as demo:

        gr.HTML("<h1>⬡ PromptMatch</h1>"
                "<div class='subhead'>CLIP-powered interactive image sorter"
                " · click image to select, then use ← → buttons to move · ✋ = manual override</div>")

        with gr.Row(equal_height=False):

            # ── sidebar ──────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=290):

                with gr.Group(elem_classes=["reprompt-box"]):
                    gr.Markdown("#### 🔄 Re-score with new prompts")
                    new_pos = gr.Textbox(value=SEARCH_PROMPT, label="Positive prompt",
                                         lines=1, placeholder="e.g. woman in red dress")
                    new_neg = gr.Textbox(value=NEGATIVE_PROMPT,
                                         label="Negative prompt (empty = disabled)",
                                         lines=1, placeholder="e.g. man, cartoon")
                    rescore_btn = gr.Button("⚡  Re-score images",
                                            elem_classes=["rescore-btn"])

                stats_box = gr.Markdown(value=init_stats)

                pos_slider = gr.Slider(minimum=pos_min, maximum=pos_max,
                                       value=POSITIVE_THRESHOLD, step=0.001,
                                       label="Positive threshold  (≥ → FOUND)",
                                       interactive=True)
                neg_slider = gr.Slider(minimum=neg_min if has_neg else 0,
                                       maximum=neg_max if has_neg else 1,
                                       value=NEGATIVE_THRESHOLD, step=0.001,
                                       label="Negative threshold  (< → passes)",
                                       interactive=has_neg)

                status = gr.Markdown(value="", elem_classes=["status-md"])

                export_btn    = gr.Button("💾  Export folders",
                                          elem_classes=["export-btn"])
                export_result = gr.Textbox(label="Export result", lines=3,
                                           interactive=False,
                                           elem_classes=["export-box"])

            # ── gallery columns ───────────────────────────────────────────────
            with gr.Column(scale=5):

                with gr.Row():
                    found_head    = gr.Markdown("### ✅ FOUND",
                                                elem_classes=["found-head"])
                    notfound_head = gr.Markdown("### ❌ NOT FOUND",
                                                elem_classes=["notfound-head"])

                with gr.Row(equal_height=True):
                    found_gallery = gr.Gallery(
                        label="Found", show_label=False,
                        columns=4, height="78vh",
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
                        move_nf_btn  = gr.Button("→ NF", elem_classes=["move-btn","move-to-nf"])
                        move_f_btn   = gr.Button("F ←",  elem_classes=["move-btn","move-to-f"])
                        gr.HTML("<div class='move-hint'>click image<br>then button</div>")

                    notfound_gallery = gr.Gallery(
                        label="Not Found", show_label=False,
                        columns=4, height="78vh",
                        object_fit="contain",
                        allow_preview=True,
                        preview=True,
                        elem_classes=["grid-wrap"],
                    )

        # ── wiring ───────────────────────────────────────────────────────────
        slider_in = [pos_slider, neg_slider]
        all_out   = [found_head, found_gallery,
                     notfound_head, notfound_gallery, status]
        move_out  = all_out + [sel_info]   # move buttons also clear sel_info

        pos_slider.change(fn=update, inputs=slider_in, outputs=all_out)
        neg_slider.change(fn=update, inputs=slider_in, outputs=all_out)

        # select handlers — just track which image is active, don't move yet
        found_gallery.select(
            fn=on_found_select, inputs=slider_in, outputs=[sel_info])
        notfound_gallery.select(
            fn=on_nf_select,    inputs=slider_in, outputs=[sel_info])

        # move buttons
        move_nf_btn.click(fn=move_selected_to_notfound,
                          inputs=slider_in, outputs=move_out)
        move_f_btn.click( fn=move_selected_to_found,
                          inputs=slider_in, outputs=move_out)

        rescore_out = [found_head, found_gallery,
                       notfound_head, notfound_gallery,
                       stats_box, pos_slider, neg_slider, status]
        rescore_btn.click(
            fn=rescore,
            inputs=[new_pos, new_neg, pos_slider, neg_slider],
            outputs=rescore_out,
        )

        export_btn.click(fn=export_files, inputs=slider_in, outputs=export_result)

        demo.load(fn=update, inputs=slider_in, outputs=all_out)

    demo.launch(server_name="0.0.0.0", server_port=7860,
                inbrowser=True, share=False)


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
    launch_ui(scores, image_paths, clip_model, preprocess, clip_mod, device,
              script_dir, source_dir)
