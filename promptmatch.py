"""
PromptMatch — Interactive Image Sorter
---------------------------------------
• CLIP scoring with positive + optional negative prompt
• Native Gradio galleries — proven to work
• object_fit="contain" — full image always visible, never cropped
• Threshold sliders → live re-sort
• Re-score panel — new prompts without restart
• Manual move: select an image in a gallery, then click ← Move or Move →
  Manual overrides persist through slider changes, re-scoring, and folder changes
• Created by vangel
"""

import os, sys, json, socket, string, warnings
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEARCH_PROMPT      = "woman"
NEGATIVE_PROMPT    = ""

POSITIVE_THRESHOLD = 0.14
NEGATIVE_THRESHOLD = 0.14

INPUT_FOLDER_NAME  = "images"
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.bmp', '.avif')
DEFAULT_BATCH_SIZE = 32   # SigLIP/BigG are large — 32 is safe on 32 GB VRAM
MAX_BATCH_SIZE     = 128
DEFAULT_SIMILARITY_RANGE = (-1.0, 1.0)


def require_cuda():
    if not torch.cuda.is_available():
        sys.exit(
            "CUDA is mandatory for this project.\n"
            "No CUDA device was detected by PyTorch."
        )


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


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def get_ephemeral_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def resolve_server_port(default_port, app_env_var):
    raw = os.getenv(app_env_var) or os.getenv("GRADIO_SERVER_PORT")
    if raw:
        try:
            port = int(raw)
        except ValueError:
            sys.exit(f"Invalid port value in {app_env_var}/GRADIO_SERVER_PORT: {raw!r}")
        if not is_port_available(port):
            sys.exit(
                f"Port {port} is already in use.\n"
                f"Set {app_env_var} or GRADIO_SERVER_PORT to a free port."
            )
        return port

    if is_port_available(default_port):
        return default_port

    return get_ephemeral_port()

# ── Model selection ──────────────────────────────────────────────────────────
# Pick ONE backend:
#
#  "openai"     — original OpenAI CLIP via the `clip` package
#                 Models: "ViT-L/14"  "ViT-L/14@336px"  "ViT-B/32"
#
#  "openclip"   — open_clip community models (much better, same API feel)
#                 Models: "ViT-H-14"        pretrained="laion2b_s32b_b79k"
#                         "ViT-bigG-14"     pretrained="laion2b_s39b_b160k"  ← best CLIP-class
#                 Install: pip install open_clip_torch
#
#  "siglip"     — Google SigLIP via HuggingFace transformers (best overall)
#                 Models: "google/siglip-so400m-patch14-384"   ← recommended
#                         "google/siglip-large-patch16-384"
#                 Install: pip install transformers
#
MODEL_BACKEND   = "openclip"
CLIP_MODEL      = "ViT-L/14"               # used when backend = "openai"
OPENCLIP_MODEL  = "ViT-bigG-14"            # used when backend = "openclip"
OPENCLIP_PRETRAINED = "laion2b_s39b_b160k" # used when backend = "openclip"
SIGLIP_MODEL    = "google/siglip-so400m-patch14-384"  # used when backend = "siglip"


def get_auto_batch_size(device, backend=None):
    """
    Pick a conservative batch size from currently free CUDA memory.

    When a larger vision backbone is selected, use a higher VRAM reference so
    we scale down earlier on smaller GPUs.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return DEFAULT_BATCH_SIZE

    reference_vram_gb = 20
    if backend is not None:
        if backend.backend == "openclip":
            if "bigG" in backend._openclip_model:
                reference_vram_gb = 32
            elif "ViT-H" in backend._openclip_model:
                reference_vram_gb = 24
        elif backend.backend == "openai" and "@336px" in backend._clip_model:
            reference_vram_gb = 24
        elif backend.backend == "siglip":
            if "large" in backend._siglip_model:
                reference_vram_gb = 24
            elif "base" in backend._siglip_model:
                reference_vram_gb = 16

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

# ---------------------------------------------------------------------------
# Model backend — unified loader / encoder / scorer
# ---------------------------------------------------------------------------

class ModelBackend:
    """Wraps OpenAI CLIP, OpenCLIP, and SigLIP behind one interface."""

    def __init__(self, device, backend=None, clip_model=None,
                 openclip_model=None, openclip_pretrained=None, siglip_model=None):
        self.device             = device
        self.backend            = backend            or MODEL_BACKEND
        self._clip_model        = clip_model         or CLIP_MODEL
        self._openclip_model    = openclip_model     or OPENCLIP_MODEL
        self._openclip_pre      = openclip_pretrained or OPENCLIP_PRETRAINED
        self._siglip_model      = siglip_model       or SIGLIP_MODEL
        self._load()

    # ── loaders ──────────────────────────────────────────────────────────────
    def _load(self):
        if self.backend == "openai":
            self._load_openai()
        elif self.backend == "openclip":
            self._load_openclip()
        elif self.backend == "siglip":
            self._load_siglip()
        else:
            sys.exit(f"Unknown MODEL_BACKEND: {self.backend!r}")

    def _load_openai(self):
        try:
            import clip as _clip
        except ImportError:
            sys.exit("OpenAI CLIP not installed.\nRun: pip install git+https://github.com/openai/CLIP.git")
        print(f"[OpenAI CLIP] Loading {self._clip_model} …")
        self._model, self._preprocess = _clip.load(self._clip_model, device=self.device)
        self._model.eval()
        self._clip_mod = _clip
        print("[OpenAI CLIP] Ready.")

    def _load_openclip(self):
        try:
            import open_clip
        except ImportError:
            sys.exit("OpenCLIP not installed.\nRun: pip install open_clip_torch")
        print(f"[OpenCLIP] Loading {self._openclip_model} / {self._openclip_pre} …")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._openclip_model, pretrained=self._openclip_pre, device=self.device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self._openclip_model)
        print("[OpenCLIP] Ready.")

    def _load_siglip(self):
        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError:
            sys.exit("transformers not installed.\nRun: pip install transformers")
        print(f"[SigLIP] Loading {self._siglip_model} …")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self._siglip_model)
        self._model     = AutoModel.from_pretrained(self._siglip_model, torch_dtype=dtype).to(self.device)
        self._model.eval()
        print("[SigLIP] Ready.")

    # ── text encoding ─────────────────────────────────────────────────────────
    def encode_text(self, prompt):
        """Returns normalised mean embedding (1, D) for a prompt string."""
        phrases = [
            f"a photo of a {prompt}",
            f"a photo of {prompt}",
            prompt,
        ]
        with torch.no_grad():
            if self.backend == "openai":
                toks = self._clip_mod.tokenize(phrases).to(self.device)
                feat = self._model.encode_text(toks)
            elif self.backend == "openclip":
                toks = self._tokenizer(phrases).to(self.device)
                feat = self._model.encode_text(toks)
            elif self.backend == "siglip":
                # SigLIP uses a HF processor for text
                inputs = self._processor(
                    text=phrases, return_tensors="pt", padding="max_length", truncation=True
                ).to(self.device)
                feat = self._model.get_text_features(**inputs)
            feat = F.normalize(feat.float(), dim=-1)
            mean = F.normalize(feat.mean(dim=0, keepdim=True), dim=-1)
        return mean   # (1, D)

    # ── image batch scoring ───────────────────────────────────────────────────
    def encode_images_batch(self, pil_images):
        """Returns normalised image embeddings (N, D)."""
        with torch.no_grad():
            if self.backend in ("openai", "openclip"):
                tensors = torch.stack([self._preprocess(img) for img in pil_images]).to(self.device)
                # Cast to model dtype (openclip bigG loads as float16 on CUDA)
                model_dtype = next(self._model.parameters()).dtype
                tensors = tensors.to(model_dtype)
                feat    = self._model.encode_image(tensors)
            elif self.backend == "siglip":
                inputs = self._processor(
                    images=pil_images, return_tensors="pt"
                ).to(self.device)
                # Cast pixel_values to match model dtype (float16 on CUDA)
                model_dtype = next(self._model.parameters()).dtype
                inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
                feat   = self._model.get_image_features(**inputs)
            feat = F.normalize(feat.float(), dim=-1)
        return feat   # (N, D)

    @property
    def name(self):
        if self.backend == "openai":    return f"OpenAI CLIP {self._clip_model}"
        if self.backend == "openclip":  return f"OpenCLIP {self._openclip_model}/{self._openclip_pre}"
        if self.backend == "siglip":    return f"SigLIP {self._siglip_model}"
        return self.backend


def score_all(image_paths, backend: ModelBackend, pos_emb, neg_emb,
              progress_cb=None):
    """Score images. progress_cb(done, total) called after each batch."""
    results = {}
    total   = len(image_paths)
    done    = 0
    batch_size = get_auto_batch_size(backend.device, backend)
    print(f"[PromptMatch] Using batch size {batch_size}")
    while done < total:
        current_size = min(batch_size, total - done)
        batch = image_paths[done:done + current_size]
        pil_imgs, valid = [], []
        for p in batch:
            try:
                pil_imgs.append(Image.open(p).convert("RGB"))
                valid.append(p)
            except Exception as e:
                print(f"  [WARN] {p}: {e}")
                results[os.path.basename(p)] = {"pos": -1.0, "neg": None, "path": p}
        if pil_imgs:
            try:
                feat = backend.encode_images_batch(pil_imgs)
                pos_sims = (feat @ pos_emb.T).squeeze(1).tolist()
                neg_sims = (feat @ neg_emb.T).squeeze(1).tolist() if neg_emb is not None else [None] * len(valid)
                for path, ps, ns in zip(valid, pos_sims, neg_sims):
                    fname = os.path.basename(path)
                    results[fname] = {
                        "pos":  float(ps),
                        "neg":  float(ns) if ns is not None else None,
                        "path": path,
                    }
            except Exception as e:
                if backend.device == "cuda" and is_cuda_oom_error(e) and current_size > 1:
                    batch_size = max(1, current_size // 2)
                    print(f"[PromptMatch] CUDA OOM, retrying with batch size {batch_size}")
                    torch.cuda.empty_cache()
                    if progress_cb:
                        progress_cb(done, total, batch_size, True)
                    continue
                import traceback
                print(f"  [WARN] batch error: {e}")
                traceback.print_exc()
                for p in valid:
                    results[os.path.basename(p)] = {"pos": -1.0, "neg": None, "path": p}
            if backend.device == "cuda":
                torch.cuda.empty_cache()
        done += len(batch)
        if progress_cb:
            progress_cb(done, total, batch_size, False)
    return results


# ---------------------------------------------------------------------------
# All available models — shown in UI dropdown
# ---------------------------------------------------------------------------
MODEL_CHOICES = [
    # label (shown in UI)                          backend      extra kwargs
    ("SigLIP  so400m-patch14-384  ★ recommended", "siglip",    {"siglip_model": "google/siglip-so400m-patch14-384"}),
    ("SigLIP  large-patch16-384",                 "siglip",    {"siglip_model": "google/siglip-large-patch16-384"}),
    ("SigLIP  base-patch16-224",                  "siglip",    {"siglip_model": "google/siglip-base-patch16-224"}),
    ("OpenCLIP  ViT-bigG-14  laion2b  ★ best CLIP","openclip", {"openclip_model": "ViT-bigG-14",  "openclip_pretrained": "laion2b_s39b_b160k"}),
    ("OpenCLIP  ViT-H-14  laion2b",               "openclip",  {"openclip_model": "ViT-H-14",     "openclip_pretrained": "laion2b_s32b_b79k"}),
    ("OpenCLIP  ViT-L-14  laion2b",               "openclip",  {"openclip_model": "ViT-L-14",     "openclip_pretrained": "laion2b_s32b_b82k"}),
    ("OpenAI CLIP  ViT-L/14@336px",               "openai",    {"clip_model": "ViT-L/14@336px"}),
    ("OpenAI CLIP  ViT-L/14",                     "openai",    {"clip_model": "ViT-L/14"}),
    ("OpenAI CLIP  ViT-B/32  (fastest)",          "openai",    {"clip_model": "ViT-B/32"}),
]
MODEL_LABELS = [m[0] for m in MODEL_CHOICES]

def label_for_backend(backend):
    """Return current model's label string."""
    for label, be, kwargs in MODEL_CHOICES:
        if be != backend.backend:
            continue
        if be == "openai"   and kwargs.get("clip_model")      == backend._clip_model:     return label
        if be == "openclip" and kwargs.get("openclip_model")  == backend._openclip_model: return label
        if be == "siglip"   and kwargs.get("siglip_model")    == backend._siglip_model:   return label
    return MODEL_LABELS[0]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def launch_ui(initial_scores, image_paths, backend, script_dir, source_dir, server_port):
    try:
        import gradio as gr
    except ImportError:
        sys.exit("Gradio not installed.\nRun: pip install gradio")

    import shutil, statistics
    device = backend.device

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
        "pm-folder-input": "Path to the image folder you want to score.",
        "pm-folder-btn": "Scan the folder above and re-score every image in it.",
        "pm-model-dd": "Choose which CLIP-family model PromptMatch should use.",
        "pm-reload-btn": "Load the selected model and re-score all images with it.",
        "pm-new-pos": "Describe what you want the app to find.",
        "pm-new-neg": "Optional: describe content that should count against a match.",
        "pm-rescore-btn": "Score all images again using the current prompts.",
        "pm-hist-plot": "Click in the histogram to set thresholds directly.",
        "pm-pos-slider": "Images at or above this positive score go to FOUND.",
        "pm-percentile-slider": "Automatically keep roughly the top N percent by positive score.",
        "pm-neg-slider": "If a negative prompt is active, lower values pass this filter.",
        "pm-export-btn": "Copy the current split into found and notfound folders.",
        "pm-export-result": "Shows where the exported files were written.",
        "pm-found-gallery": "Images currently classified as FOUND. Click one to select it.",
        "pm-move-nf-btn": "Move the selected FOUND image into NOT FOUND as a manual override.",
        "pm-move-f-btn": "Move the selected NOT FOUND image back into FOUND as a manual override.",
        "pm-notfound-gallery": "Images currently classified as NOT FOUND. Click one to select it."
    }

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
        "overrides":   {},   # fname → "found" | "notfound"
        "found_sel":   None,
        "nf_sel":      None,
        "backend":     backend,
        "image_paths": list(image_paths),  # refreshed on each rescore/reload
        "source_dir":  source_dir,
        "progress_html": render_progress(),
    }

    # ── core logic ──────────────────────────────────────────────────────────
    def clip_side(v, pos_t, neg_t):
        pos_ok = v["pos"] >= pos_t
        neg_ok = (v["neg"] is None) or (v["neg"] < neg_t)
        return "found" if (pos_ok and neg_ok) else "notfound"

    def build_split(scores, overrides, pos_t, neg_t):
        """Returns (found_list, notfound_list) sorted by score desc.
        Each item is (image_path, caption_string).
        Caption = score value only (shown under tile).
        Manual images get ✋ PREFIX so CSS [alt^="✋"] can add red border.
        Full filename shown in Gradio's zoom/preview caption.
        """
        found, notfound = [], []
        for fname, v in sorted(scores.items(), key=lambda x: -x[1]["pos"]):
            if v["pos"] < 0:
                continue
            manual = fname in overrides
            side   = overrides[fname] if manual else clip_side(v, pos_t, neg_t)
            # Caption used as alt= text by Gradio — prefix drives CSS red-border
            # Score shown under tile; filename visible only in zoom preview
            if manual:
                caption = f"✋ {v['pos']:.3f} | {fname}"
            else:
                caption = f"{v['pos']:.3f} | {fname}"
            entry = (v["path"], caption)
            if side == "found":
                found.append(entry)
            else:
                notfound.append(entry)
        return found, notfound

    def gallery_update(items):
        """Refresh gallery contents and clear any selected/zoomed preview."""
        return gr.update(value=items, selected_index=None)

    def clear_selection_state():
        state["found_sel"] = None
        state["nf_sel"] = None

    def status_line(found, notfound, scores, overrides):
        n_manual = len(overrides)
        n_failed = sum(1 for v in scores.values() if v["pos"] < 0)
        n_total  = len(scores)
        s = f"⬡  {len(found)} found  /  {len(notfound)} not found  ({n_total} total)"
        if n_manual:
            s += f"  ✋ {n_manual} manual override{'s' if n_manual != 1 else ''}"
        if n_failed:
            s += f"  ⚠ {n_failed} failed"
        return s

    def make_histogram(scores, pos_t, neg_t):
        """Render histogram as base64 PNG (Pillow). Returns HTML with img + click overlay."""
        from PIL import ImageDraw, ImageFont
        pos_vals = [v["pos"] for v in scores.values() if v["pos"] >= 0]
        neg_vals = [v["neg"] for v in scores.values()
                    if v.get("neg") is not None and v["neg"] >= 0]
        has_neg  = bool(neg_vals)

        def _bins(vals, n=32):
            if not vals: return [], 0.0, 1.0
            lo, hi = min(vals), max(vals)
            if lo == hi: lo -= 0.005; hi += 0.005
            w = (hi - lo) / n
            counts = [0] * n
            for v in vals:
                counts[min(int((v - lo) / w), n - 1)] += 1
            return counts, lo, hi

        pos_counts, pos_lo, pos_hi = _bins(pos_vals)
        neg_counts, neg_lo, neg_hi = _bins(neg_vals)

        # ── drawing constants ──────────────────────────────────────────────────
        W, CH = 300, 130        # canvas width, chart height
        PAD_L, PAD_R  = 38, 8
        PAD_TOP, PAD_BOT = 18, 22
        GAP   = 28              # gap between charts
        n_ch  = 2 if has_neg else 1
        H     = PAD_TOP + n_ch * CH + (n_ch - 1) * GAP + PAD_BOT

        img = Image.new("RGB", (W, H), "#0d0d11")
        d   = ImageDraw.Draw(img)

        def draw_chart(y0, counts, lo, hi, thresh, bar_rgb, line_rgb, label):
            cW = W - PAD_L - PAD_R
            max_c = max(counts) if counts else 1
            bw = cW / len(counts)
            # background
            d.rectangle([PAD_L, y0, W - PAD_R, y0 + CH], fill="#0f0f16")
            # bars
            for i, c in enumerate(counts):
                if c == 0: continue
                bh = max(1, int((c / max_c) * (CH - 2)))
                x0b = PAD_L + int(i * bw) + 1
                x1b = PAD_L + int((i + 1) * bw) - 1
                d.rectangle([x0b, y0 + CH - bh, x1b, y0 + CH], fill=bar_rgb)
            # threshold line (dashed)
            tx = PAD_L + int(((thresh - lo) / (hi - lo)) * cW)
            tx = max(PAD_L, min(W - PAD_R, tx))
            for yy in range(y0, y0 + CH, 6):
                d.line([(tx, yy), (tx, min(yy + 3, y0 + CH))],
                       fill=line_rgb, width=2)
            # x-axis labels
            for frac, val in [(0.0, lo), (0.5, (lo+hi)/2), (1.0, hi)]:
                lx = PAD_L + int(frac * cW)
                d.text((lx, y0 + CH + 4), f"{val:.3f}",
                       fill="#667755", anchor="mt")
            # title
            d.text((PAD_L, y0 - 14),
                   f"{label}   threshold: {thresh:.3f}",
                   fill="#99bb88")

        draw_chart(PAD_TOP, pos_counts, pos_lo, pos_hi,
                   pos_t, "#3a7a3a", "#aadd66", "▲ positive")
        if has_neg:
            draw_chart(PAD_TOP + CH + GAP, neg_counts, neg_lo, neg_hi,
                       neg_t, "#7a3a3a", "#dd6644", "▼ negative")

        # Store geometry in state so on_hist_click can use it
        if not scores or not pos_vals:
            return None
        state["hist_geom"] = {
            "W": W, "H": H, "PAD_L": PAD_L, "PAD_R": PAD_R,
            "PAD_TOP": PAD_TOP, "CH": CH, "GAP": GAP,
            "pos_lo": pos_lo, "pos_hi": pos_hi,
            "neg_lo": neg_lo, "neg_hi": neg_hi,
            "has_neg": has_neg,
        }
        return img   # PIL Image — fed directly to gr.Image

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
        pos_min = round(min(pos_vals) - 0.01, 3) if pos_vals else DEFAULT_SIMILARITY_RANGE[0]
        pos_max = round(max(pos_vals) + 0.01, 3) if pos_vals else DEFAULT_SIMILARITY_RANGE[1]
        # Default threshold = midpoint of the found (upper) half of score range
        pos_mid = round((pos_min + pos_max) / 2.0, 3)
        return (
            pos_min,
            pos_max,
            round(min(neg_vals) - 0.01, 3) if neg_vals else DEFAULT_SIMILARITY_RANGE[0],
            round(max(neg_vals) + 0.01, 3) if neg_vals else DEFAULT_SIMILARITY_RANGE[1],
            bool(neg_vals),
            pos_mid,
        )

    def get_threshold_for_percentile(scores, percentile):
        """Calculate threshold to keep top N% of images (by positive score)."""
        pos_vals = sorted([v["pos"] for v in scores.values() if v["pos"] >= 0], reverse=True)
        if not pos_vals:
            return 0.0
        if percentile <= 0:
            return max(pos_vals) + 0.01
        if percentile >= 100:
            return min(pos_vals) - 0.01
        idx = max(0, int(len(pos_vals) * (percentile / 100.0)) - 1)
        return round(pos_vals[idx], 3)

    # ── Gradio event handlers ───────────────────────────────────────────────
    def reload_model(model_label, pos_t, neg_t, pos_prompt, neg_prompt,
                     progress=gr.Progress()):
        """Load a different model, re-encode prompts, re-score all images."""
        # Find the chosen model config
        cfg = next((c for c in MODEL_CHOICES if c[0] == model_label), None)
        if cfg is None:
            yield (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                state["progress_html"],
                f"⚠️ Unknown model: {model_label}",
            )
            return

        label, be, kwargs = cfg
        state["progress_html"] = render_progress(0, 0, f"Loading {label}")
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            state["progress_html"],
            f"⏳ Loading {label} …",
        )

        try:
            new_backend = ModelBackend(device, backend=be, **kwargs)
            state["backend"] = new_backend
        except Exception as e:
            state["progress_html"] = render_progress(0, 0, "Load failed")
            yield (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                state["progress_html"],
                f"❌ Load error: {e}",
            )
            return

        pos_prompt = pos_prompt.strip() or SEARCH_PROMPT
        neg_prompt = neg_prompt.strip()

        try:
            fresh_paths = scan_image_paths()
            state["image_paths"] = fresh_paths
            pos_emb = new_backend.encode_text(pos_prompt)
            neg_emb = new_backend.encode_text(neg_prompt) if neg_prompt else None
            progress(0, desc=f"Scoring {len(fresh_paths)} images…")
            def _cb(done, total, batch_size, oom_retry):
                label = f"Scoring… {done}/{total} (batch {batch_size})"
                if oom_retry:
                    label = f"CUDA OOM, retrying with batch {batch_size}"
                state["progress_html"] = render_progress(done, total, label)
                progress(done / max(total, 1), desc=label)
            new_scores = score_all(fresh_paths, new_backend, pos_emb, neg_emb,
                                   progress_cb=_cb)
            progress(1, desc="Done")
        except Exception as e:
            state["progress_html"] = render_progress(0, 0, "Scoring failed")
            yield (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                state["progress_html"],
                f"❌ Scoring error: {e}",
            )
            return

        state["scores"]    = new_scores
        state["progress_html"] = render_progress(len(fresh_paths), len(fresh_paths), "Scoring complete")

        pos_min, pos_max, neg_min, neg_max, has_neg, pos_mid = slider_range(new_scores)
        neg_mid = round((neg_min + neg_max) / 2, 3) if has_neg else NEGATIVE_THRESHOLD
        stats = make_stats(pos_prompt, neg_prompt, new_scores)
        clear_selection_state()
        found, notfound = build_split(new_scores, state["overrides"], pos_mid, neg_mid)
        yield (
            f"### ✅ FOUND — {len(found)} images",
            gallery_update(found),
            f"### ❌ NOT FOUND — {len(notfound)} images",
            gallery_update(notfound),
            stats,
            gr.update(minimum=pos_min, maximum=pos_max, value=pos_mid),
            gr.update(minimum=neg_min if has_neg else 0,
                      maximum=neg_max if has_neg else 1,
                      value=neg_mid, interactive=has_neg),
            gr.update(),   # sel_info
            make_histogram(new_scores, pos_mid, neg_mid),
            state["progress_html"],
            status_line(found, notfound, new_scores, state["overrides"]),
        )

    def on_hist_click(sel: gr.SelectData, pos_t, neg_t):
        """gr.Image .select() gives pixel coords — map to threshold value."""
        geom = state.get("hist_geom")
        if not geom:
            return (gr.update(),) * 9
        try:
            cx, cy = sel.index   # pixel coords within the image
            W, H       = geom["W"], geom["H"]
            PAD_L      = geom["PAD_L"]
            PAD_R      = geom["PAD_R"]
            PAD_TOP    = geom["PAD_TOP"]
            CH         = geom["CH"]
            GAP        = geom["GAP"]
            cW         = W - PAD_L - PAD_R
            y0pos      = PAD_TOP
            y0neg      = PAD_TOP + CH + GAP

            if cy >= y0pos and cy <= y0pos + CH:
                lo, hi = geom["pos_lo"], geom["pos_hi"]
                val    = lo + ((cx - PAD_L) / cW) * (hi - lo)
                pos_t  = round(max(lo, min(hi, val)), 3)
            elif geom["has_neg"] and cy >= y0neg and cy <= y0neg + CH:
                lo, hi = geom["neg_lo"], geom["neg_hi"]
                val    = lo + ((cx - PAD_L) / cW) * (hi - lo)
                neg_t  = round(max(lo, min(hi, val)), 3)
            else:
                return (gr.update(),) * 9
        except Exception as e:
            print(f"[hist click] error: {e}")
            return (gr.update(),) * 9

        clear_selection_state()
        found, notfound = build_split(state["scores"], state["overrides"], pos_t, neg_t)
        return (
            f"### ✅ FOUND — {len(found)} images",
            gallery_update(found),
            f"### ❌ NOT FOUND — {len(notfound)} images",
            gallery_update(notfound),
            state["progress_html"],
            status_line(found, notfound, state["scores"], state["overrides"]),
            make_histogram(state["scores"], pos_t, neg_t),
            gr.update(value=pos_t),
            gr.update(value=neg_t),
        )

    def change_folder(new_path, pos_t, neg_t, pos_prompt, neg_prompt,
                      progress=gr.Progress()):
        """User typed a new input folder path — rescan and re-score."""
        new_path = (new_path or "").strip()
        if not new_path or not os.path.isdir(new_path):
            yield (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                state["progress_html"],
                f"⚠️ Invalid folder: {new_path!r}",
            )
            return

        state["source_dir"] = new_path
        pos_prompt = pos_prompt.strip() or SEARCH_PROMPT
        neg_prompt = neg_prompt.strip()

        state["progress_html"] = render_progress(0, 0, f"Scanning {new_path}")
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            state["progress_html"],
            f"⏳ Scanning {new_path} …",
        )

        fresh_paths = scan_image_paths()
        if not fresh_paths:
            state["progress_html"] = render_progress(0, 0, "No images found")
            yield (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                state["progress_html"],
                f"⚠️ No images found in {new_path}",
            )
            return

        state["image_paths"] = fresh_paths

        try:
            cur = state["backend"]
            pos_emb = cur.encode_text(pos_prompt)
            neg_emb = cur.encode_text(neg_prompt) if neg_prompt else None
            progress(0, desc=f"Scoring {len(fresh_paths)} images…")
            def _cb(done, total, batch_size, oom_retry):
                label = f"Scoring… {done}/{total} (batch {batch_size})"
                if oom_retry:
                    label = f"CUDA OOM, retrying with batch {batch_size}"
                state["progress_html"] = render_progress(done, total, label)
                progress(done / max(total, 1), desc=label)
            new_scores = score_all(fresh_paths, cur, pos_emb, neg_emb, progress_cb=_cb)
            progress(1, desc="Done")
        except Exception as e:
            state["progress_html"] = render_progress(0, 0, "Scoring failed")
            yield (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                state["progress_html"],
                f"❌ Scoring error: {e}",
            )
            return

        state["scores"] = new_scores
        state["progress_html"] = render_progress(len(fresh_paths), len(fresh_paths), "Scoring complete")
        pos_min, pos_max, neg_min, neg_max, has_neg, pos_mid = slider_range(new_scores)
        neg_mid  = round((neg_min + neg_max) / 2, 3) if has_neg else NEGATIVE_THRESHOLD
        stats    = make_stats(pos_prompt, neg_prompt, new_scores)
        clear_selection_state()
        found, notfound = build_split(new_scores, state["overrides"], pos_mid, neg_mid)
        yield (
            f"### ✅ FOUND — {len(found)} images",
            gallery_update(found),
            f"### ❌ NOT FOUND — {len(notfound)} images",
            gallery_update(notfound),
            stats,
            gr.update(minimum=pos_min, maximum=pos_max, value=pos_mid),
            gr.update(minimum=neg_min if has_neg else 0,
                      maximum=neg_max if has_neg else 1,
                      value=neg_mid, interactive=has_neg),
            make_histogram(new_scores, pos_mid, neg_mid),
            gr.update(),
            state["progress_html"],
            f"✅ Loaded {len(fresh_paths)} images from {new_path}",
        )

    def update(pos_t, neg_t):
        clear_selection_state()
        found, notfound = build_split(state["scores"], state["overrides"], pos_t, neg_t)
        hist = make_histogram(state["scores"], pos_t, neg_t) if state["scores"] else None
        return (
            f"### ✅ FOUND — {len(found)} images",
            gallery_update(found),
            f"### ❌ NOT FOUND — {len(notfound)} images",
            gallery_update(notfound),
            state["progress_html"],
            status_line(found, notfound, state["scores"], state["overrides"]),
            hist,
        )

    def on_found_select(sel: gr.SelectData, pos_t, neg_t):
        """Track which image is selected in FOUND gallery."""
        found, notfound = build_split(state["scores"], state["overrides"], pos_t, neg_t)
        state["found_sel"] = os.path.basename(found[sel.index][0]) if sel.index < len(found) else None
        state["nf_sel"]    = None
        fname = state["found_sel"] or "?"
        return gr.update(value=f"Selected in FOUND: **{fname}**")

    def on_nf_select(sel: gr.SelectData, pos_t, neg_t):
        """Track which image is selected in NOT FOUND gallery."""
        found, notfound = build_split(state["scores"], state["overrides"], pos_t, neg_t)
        state["nf_sel"] = os.path.basename(notfound[sel.index][0]) if sel.index < len(notfound) else None
        state["found_sel"] = None
        fname = state["nf_sel"] or "?"
        return gr.update(value=f"Selected in NOT FOUND: **{fname}**")

    def move_selected_to_notfound(pos_t, neg_t):
        """Move currently selected FOUND image → NOT FOUND."""
        fname = state.get("found_sel")
        if fname:
            state["overrides"][fname] = "notfound"
        state["found_sel"] = None
        state["nf_sel"] = None
        return update(pos_t, neg_t) + (gr.update(value=""),)

    def move_selected_to_found(pos_t, neg_t):
        """Move currently selected NOT FOUND image → FOUND."""
        fname = state.get("nf_sel")
        if fname:
            state["overrides"][fname] = "found"
        state["nf_sel"] = None
        state["found_sel"] = None
        return update(pos_t, neg_t) + (gr.update(value=""),)

    def rescore(new_pos, new_neg, pos_t, neg_t, progress=gr.Progress()):
        new_pos = new_pos.strip()
        new_neg = new_neg.strip()
        if not new_pos:
            yield (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                state["progress_html"],
                "⚠️ Positive prompt cannot be empty.",
            )
            return

        state["progress_html"] = render_progress(0, 0, "Encoding prompts")
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            state["progress_html"],
            "⏳ Encoding & scoring — please wait…",
        )

        try:
            cur = state["backend"]
            # Refresh image list — picks up files added/removed since startup
            fresh_paths = scan_image_paths()
            state["image_paths"] = fresh_paths
            n_new = len(fresh_paths) - len(state["scores"])
            if n_new != 0:
                print(f"  [INFO] Image folder changed: {len(fresh_paths)} files "
                      f"({'+ ' if n_new > 0 else ''}{n_new} vs previous)")
            pos_emb = cur.encode_text(new_pos)
            neg_emb = cur.encode_text(new_neg) if new_neg else None
            progress(0, desc=f"Scoring {len(fresh_paths)} images…")
            def _cb(done, total, batch_size, oom_retry):
                label = f"Scoring… {done}/{total} (batch {batch_size})"
                if oom_retry:
                    label = f"CUDA OOM, retrying with batch {batch_size}"
                state["progress_html"] = render_progress(done, total, label)
                progress(done / max(total, 1), desc=label)
            new_scores = score_all(fresh_paths, cur, pos_emb, neg_emb,
                                   progress_cb=_cb)
            progress(1, desc="Done")
        except Exception as e:
            state["progress_html"] = render_progress(0, 0, "Scoring failed")
            yield (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                state["progress_html"],
                f"❌ Error: {e}",
            )
            return

        state["scores"]    = new_scores
        state["progress_html"] = render_progress(len(fresh_paths), len(fresh_paths), "Scoring complete")

        pos_min, pos_max, neg_min, neg_max, has_neg, pos_mid = slider_range(new_scores)
        neg_mid = round((neg_min + neg_max) / 2, 3) if has_neg else NEGATIVE_THRESHOLD
        stats = make_stats(new_pos, new_neg, new_scores)
        # Use pos_mid — old pos_t may be outside the new model's score range
        clear_selection_state()
        found, notfound = build_split(new_scores, state["overrides"], pos_mid, neg_mid)

        yield (
            f"### ✅ FOUND — {len(found)} images",
            gallery_update(found),
            f"### ❌ NOT FOUND — {len(notfound)} images",
            gallery_update(notfound),
            stats,
            gr.update(minimum=pos_min, maximum=pos_max, value=pos_mid),
            gr.update(minimum=neg_min if has_neg else 0,
                      maximum=neg_max if has_neg else 1,
                      value=neg_mid, interactive=has_neg),
            make_histogram(new_scores, pos_mid, neg_mid),
            state["progress_html"],
            status_line(found, notfound, new_scores, state["overrides"]),
        )

    def export_files(pos_t, neg_t):
        found, notfound = build_split(state["scores"], state["overrides"], pos_t, neg_t)
        failed = [f for f, v in state["scores"].items() if v["pos"] < 0]
        base         = state.get("source_dir", source_dir)
        found_dir    = os.path.join(base, "found")
        notfound_dir = os.path.join(base, "notfound")
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
        return (f"✅ {len(found)} → {found_dir}\n"
                f"✅ {len(notfound)+len(failed)} → {notfound_dir}")

    def set_threshold_from_percentile(percentile, pos_t, neg_t):
        """User changed percentile slider — calculate new threshold."""
        new_pos_t = get_threshold_for_percentile(state["scores"], percentile)
        clear_selection_state()
        found, notfound = build_split(state["scores"], state["overrides"], new_pos_t, neg_t)
        return (
            f"### ✅ FOUND — {len(found)} images",
            gallery_update(found),
            f"### ❌ NOT FOUND — {len(notfound)} images",
            gallery_update(notfound),
            state["progress_html"],
            status_line(found, notfound, state["scores"], state["overrides"]),
            make_histogram(state["scores"], new_pos_t, neg_t),
            gr.update(value=new_pos_t),
        )

    # ── initial values ──────────────────────────────────────────────────────
    pos_min, pos_max, neg_min, neg_max, has_neg, pos_mid = slider_range(initial_scores)
    init_stats = (make_stats(SEARCH_PROMPT, NEGATIVE_PROMPT, initial_scores)
                  if initial_scores else "*No images loaded — use 📁 folder selector above*")

    # ── CSS ─────────────────────────────────────────────────────────────────
    css = """
    body, .gradio-container { background:#0d0d11 !important; color:#ddd8cc !important; }
    .gr-block,.gr-box,.panel { background:#14141c !important; border-color:#252530 !important; }

    /* Full browser width — remove Gradio's default max-width cap */
    .gradio-container { max-width: 100% !important; padding: 8px 12px !important; }
    .main { max-width: 100% !important; }
    footer { display: none !important; }

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

    /* Caption text under each tile: show score, hide the " | filename" part */
    .grid-wrap .caption-label span,
    .grid-wrap span.svelte-1dv1zt9,
    .grid-wrap [class*="caption"] {
        font-family: monospace !important;
        font-size: .72em !important;
        color: #8899aa !important;
    }

    /* Red border on manually-moved images (alt text starts with ✋) */
    .grid-wrap img[alt^="✋"] {
        outline: 3px solid #dd3322 !important;
        outline-offset: -3px;
    }

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

    /* model selector box */
    .model-box { background:#1a1a2a !important; border:1px solid #333355 !important;
                 border-radius:8px; padding:10px 14px; margin-bottom:8px; }
    .model-box label { color:#aaaaee !important; font-family:monospace !important;
                       font-size:.82em !important; text-transform:uppercase; }
    .reload-btn { background:#1a1a3a !important; color:#aaaaee !important;
                  border:1px solid #444488 !important;
                  font-family:monospace !important; width:100% !important;
                  margin-top:4px !important; }
    .reload-btn:hover { background:#222250 !important; }

    /* histogram — gr.Image, click sets threshold */
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
    with gr.Blocks(title="PromptMatch") as demo:

        gr.HTML("""
<h1>⬡ PromptMatch</h1>
<div class='subhead'>CLIP-powered interactive image sorter
 &middot; click image to select, then use &larr; &rarr; buttons to move &middot; ✋ = manual override
 &middot; <span style='color:#aadd66'>click histogram to set threshold</span>
 &middot; created by vangel</div>
""")


        with gr.Row(equal_height=False):

            # ── sidebar ──────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=290):

                with gr.Group(elem_classes=["folder-box"]):
                    gr.Markdown("#### 📁 Image folder")
                    folder_input = gr.Textbox(
                        value=source_dir, label="", show_label=False,
                        placeholder=folder_placeholder(), lines=1,
                        elem_classes=["folder-input"],
                        elem_id="pm-folder-input",
                    )
                    folder_btn = gr.Button("📂  Load folder & re-score",
                                           elem_classes=["folder-btn"],
                                           elem_id="pm-folder-btn")

                with gr.Group(elem_classes=["model-box"]):
                    gr.Markdown("#### 🧠 Model")
                    model_dd = gr.Dropdown(
                        choices=MODEL_LABELS,
                        value=label_for_backend(backend),
                        label="",
                        show_label=False,
                        interactive=True,
                        elem_id="pm-model-dd",
                    )
                    reload_btn = gr.Button("🔄  Load model & re-score",
                                           elem_classes=["reload-btn"],
                                           elem_id="pm-reload-btn")

                with gr.Group(elem_classes=["reprompt-box"]):
                    gr.Markdown("#### 🔄 Re-score with new prompts")
                    new_pos = gr.Textbox(value=SEARCH_PROMPT, label="Positive prompt",
                                         lines=1, placeholder="e.g. woman in red dress",
                                         elem_id="pm-new-pos")
                    new_neg = gr.Textbox(value=NEGATIVE_PROMPT,
                                         label="Negative prompt (empty = disabled)",
                                         lines=1, placeholder="e.g. man, cartoon",
                                         elem_id="pm-new-neg")
                    rescore_btn = gr.Button("⚡  Re-score images",
                                            elem_classes=["rescore-btn"],
                                            elem_id="pm-rescore-btn")

                stats_box = gr.Markdown(value=init_stats)

                hist_plot = gr.Image(
                    value=make_histogram(initial_scores, pos_mid,
                                         NEGATIVE_THRESHOLD) if initial_scores else None,
                    show_label=False,
                    interactive=False,
                    elem_classes=["hist-img"],
                    elem_id="pm-hist-plot",
                )

                progress_html = gr.HTML(value=state["progress_html"])

                pos_slider = gr.Slider(minimum=pos_min, maximum=pos_max,
                                       value=pos_mid, step=0.001,
                                       label="Positive threshold  (≥ → FOUND)",
                                       interactive=True,
                                       elem_id="pm-pos-slider")
                percentile_slider = gr.Slider(minimum=0, maximum=100,
                                              value=50, step=1,
                                              label="Or select top N%",
                                              interactive=True,
                                              elem_id="pm-percentile-slider")
                neg_slider = gr.Slider(minimum=neg_min if has_neg else 0,
                                       maximum=neg_max if has_neg else 1,
                                       value=NEGATIVE_THRESHOLD, step=0.001,
                                       label="Negative threshold  (< → passes)",
                                       interactive=has_neg,
                                       elem_id="pm-neg-slider")

                status = gr.Markdown(value="", elem_classes=["status-md"])

                export_btn    = gr.Button("💾  Export folders",
                                          elem_classes=["export-btn"],
                                          elem_id="pm-export-btn")
                export_result = gr.Textbox(label="Export result", lines=3,
                                           interactive=False,
                                           elem_classes=["export-box"],
                                           elem_id="pm-export-result")

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
                        columns=5, height="80vh",
                        object_fit="contain",
                        allow_preview=True,
                        preview=True,
                        elem_classes=["grid-wrap"],
                        elem_id="pm-found-gallery",
                    )

                    # ── centre move column ────────────────────────────────
                    with gr.Column(scale=0, min_width=90,
                                   elem_classes=["move-col"]):
                        gr.HTML("<div class='move-col-label'>MOVE</div>")
                        sel_info     = gr.Markdown("", elem_classes=["sel-info"])
                        move_nf_btn  = gr.Button("→ NF", elem_classes=["move-btn","move-to-nf"], elem_id="pm-move-nf-btn")
                        move_f_btn   = gr.Button("F ←",  elem_classes=["move-btn","move-to-f"], elem_id="pm-move-f-btn")
                        gr.HTML("<div class='move-hint'>click image<br>then button</div>")

                    notfound_gallery = gr.Gallery(
                        label="Not Found", show_label=False,
                        columns=5, height="80vh",
                        object_fit="contain",
                        allow_preview=True,
                        preview=True,
                        elem_classes=["grid-wrap"],
                        elem_id="pm-notfound-gallery",
                    )

        # ── wiring ───────────────────────────────────────────────────────────
        slider_in = [pos_slider, neg_slider]
        all_out   = [found_head, found_gallery,
                     notfound_head, notfound_gallery, progress_html, status, hist_plot]
        move_out  = all_out + [sel_info]   # move buttons also clear sel_info

        pos_slider.change(fn=update, inputs=slider_in, outputs=all_out)
        neg_slider.change(fn=update, inputs=slider_in, outputs=all_out)

        # percentile slider — calculate threshold from top N%
        percentile_out = all_out + [pos_slider]
        percentile_slider.change(fn=set_threshold_from_percentile,
                                 inputs=[percentile_slider, pos_slider, neg_slider],
                                 outputs=percentile_out)

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
                       stats_box, pos_slider, neg_slider, hist_plot, progress_html, status]
        rescore_btn.click(
            fn=rescore,
            inputs=[new_pos, new_neg, pos_slider, neg_slider],
            outputs=rescore_out,
        )

        reload_out = [found_head, found_gallery,
                      notfound_head, notfound_gallery,
                      stats_box, pos_slider, neg_slider, sel_info, hist_plot, progress_html, status]
        reload_btn.click(
            fn=reload_model,
            inputs=[model_dd, pos_slider, neg_slider, new_pos, new_neg],
            outputs=reload_out,
        )

        # histogram click → set threshold via gr.Image .select() pixel coords
        hist_click_out = [found_head, found_gallery,
                          notfound_head, notfound_gallery, progress_html, status,
                          hist_plot, pos_slider, neg_slider]
        hist_plot.select(
            fn=on_hist_click,
            inputs=slider_in,
            outputs=hist_click_out,
        )

        folder_out = [found_head, found_gallery,
                      notfound_head, notfound_gallery,
                      stats_box, pos_slider, neg_slider, hist_plot, sel_info, progress_html, status]
        folder_btn.click(
            fn=change_folder,
            inputs=[folder_input, pos_slider, neg_slider, new_pos, new_neg],
            outputs=folder_out,
        )

        export_btn.click(fn=export_files, inputs=slider_in, outputs=export_result)

        demo.load(fn=update, inputs=slider_in, outputs=all_out)

    demo.launch(server_name="0.0.0.0", server_port=server_port,
                inbrowser=True, share=False,
                css=css, theme=gr.themes.Base(),
                head=tooltip_head(tooltips),
                allowed_paths=get_allowed_paths(script_dir, source_dir))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PromptMatch — Interactive Sorter ===")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    require_cuda()
    device     = "cuda"
    print(f"Device: {device.upper()}")

    # Load model first — UI needs it ready
    backend = ModelBackend(device)
    print(f"  Model : {backend.name}")

    # Try to find a default folder to pre-load (optional — UI works without it)
    source_dir  = os.path.join(script_dir, INPUT_FOLDER_NAME)
    image_paths = []
    scores      = {}

    if os.path.isdir(source_dir):
        image_paths = [
            os.path.join(source_dir, f)
            for f in sorted(os.listdir(source_dir))
            if f.lower().endswith(ALLOWED_EXTENSIONS)
            and os.path.isfile(os.path.join(source_dir, f))
        ]

    if image_paths:
        print(f"\nFound {len(image_paths)} images in {source_dir}. Scoring…")
        pos_emb = backend.encode_text(SEARCH_PROMPT)
        neg_emb = backend.encode_text(NEGATIVE_PROMPT) if NEGATIVE_PROMPT else None
        from tqdm import tqdm as _tqdm
        _pbar = _tqdm(total=len(image_paths), desc="Scoring")
        def _main_cb(done, total):
            _pbar.n = done; _pbar.refresh()
        scores = score_all(image_paths, backend, pos_emb, neg_emb, progress_cb=_main_cb)
        _pbar.close()
        print(f"Scoring done.")
    else:
        print(f"\nNo default images folder found — use the 📁 folder selector in the UI.")
        source_dir = script_dir   # fallback: show script dir in the folder box

    server_port = resolve_server_port(7861, "PROMPTMATCH_PORT")
    print(f"Launching browser UI on http://localhost:{server_port} …\n")
    launch_ui(scores, image_paths, backend, script_dir, source_dir, server_port)
