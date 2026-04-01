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
import socket
import string
import sys
import tempfile
import types
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from importlib import import_module
from importlib.machinery import ModuleSpec

import gradio as gr
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# High-level app modes and default thresholds/prompts.
METHOD_PROMPTMATCH = "PromptMatch"
METHOD_IMAGEREWARD = "ImageReward"
DEFAULT_IR_NEGATIVE_PROMPT = ""
DEFAULT_IR_PENALTY_WEIGHT = 1.0
PROMPTMATCH_SLIDER_MIN = -1.0
PROMPTMATCH_SLIDER_MAX = 1.0
IMAGEREWARD_SLIDER_MIN = -5.0
IMAGEREWARD_SLIDER_MAX = 5.0
HIST_HEIGHT_SCALE = 0.7

SEARCH_PROMPT = "woman"
NEGATIVE_PROMPT = ""
NEGATIVE_THRESHOLD = 0.14
IR_PROMPT = (
    "masterpiece, best quality, ultra-detailed, cinematic, "
    "extremely beautiful woman, dramatic lighting, rim light, chiaroscuro, "
    "professional studio portrait, 8k, award-winning"
)
IMAGEREWARD_THRESHOLD = 0.5
INPUT_FOLDER_NAME = "images"
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".avif")
DEFAULT_BATCH_SIZE = 32
MAX_BATCH_SIZE = 128
PROMPTMATCH_PROXY_MAX_EDGE = 1024
PROMPTMATCH_PROXY_CACHE_ROOT = "HybridScorerPromptMatchProxyCache"
PROMPTMATCH_PROXY_PROGRESS_SHARE = 0.25


def get_imagereward_utils():
    # Avoid ImageReward's package-level ReFL import, which pulls in unrelated diffusers code.
    package_name = "ImageReward"
    module_name = f"{package_name}.utils"
    if module_name in sys.modules:
        return sys.modules[module_name]

    package_dir = None
    for entry in sys.path:
        candidate = os.path.join(entry, "ImageReward")
        if os.path.isdir(candidate):
            package_dir = candidate
            break

    if package_dir is None:
        sys.exit(
            "ImageReward not installed.\n"
            "Run: python -m pip install -r requirements.txt"
        )

    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__file__ = os.path.join(package_dir, "__init__.py")
        package.__path__ = [package_dir]
        package.__package__ = package_name
        spec = ModuleSpec(package_name, loader=None, is_package=True)
        spec.submodule_search_locations = [package_dir]
        package.__spec__ = spec
        sys.modules[package_name] = package

    try:
        return import_module(module_name)
    except Exception as exc:
        sys.exit(
            "ImageReward could not be imported.\n"
            f"{exc}"
        )


def require_cuda():
    if not torch.cuda.is_available():
        sys.exit(
            "CUDA is mandatory for this project.\n"
            "No CUDA device was detected by PyTorch."
        )


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


def normalize_folder_identity(folder):
    return os.path.normcase(os.path.abspath(folder))


def get_promptmatch_proxy_cache_dir(folder):
    folder_key = normalize_folder_identity(folder)
    digest = sha256(folder_key.encode("utf-8", errors="ignore")).hexdigest()
    return os.path.join(tempfile.gettempdir(), PROMPTMATCH_PROXY_CACHE_ROOT, digest)


def clear_promptmatch_proxy_cache(cache_dir):
    if cache_dir and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def build_promptmatch_proxy_path(original_path, cache_dir, max_edge=PROMPTMATCH_PROXY_MAX_EDGE):
    stat = os.stat(original_path)
    cache_key = "|".join([
        os.path.normcase(os.path.abspath(original_path)),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        str(max_edge),
    ])
    filename = f"{sha256(cache_key.encode('utf-8', errors='ignore')).hexdigest()}.jpg"
    return os.path.join(cache_dir, filename)


def ensure_promptmatch_proxy(original_path, cache_dir, max_edge=PROMPTMATCH_PROXY_MAX_EDGE):
    os.makedirs(cache_dir, exist_ok=True)
    proxy_path = build_promptmatch_proxy_path(original_path, cache_dir, max_edge=max_edge)
    if os.path.isfile(proxy_path):
        return proxy_path

    resampling = getattr(Image, "Resampling", Image)
    tmp_handle = tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".jpg", delete=False)
    tmp_path = tmp_handle.name
    tmp_handle.close()

    try:
        with Image.open(original_path) as src_img:
            img = src_img.convert("RGB")
            if max(img.size) > max_edge:
                img.thumbnail((max_edge, max_edge), resampling.LANCZOS)
            img.save(tmp_path, format="JPEG", quality=90)
        os.replace(tmp_path, proxy_path)
        return proxy_path
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def prepare_promptmatch_proxies(image_paths, cache_dir, progress_cb=None):
    os.makedirs(cache_dir, exist_ok=True)
    proxy_map = {}
    generated = 0
    reused = 0
    total = len(image_paths)
    if total == 0:
        return proxy_map, generated, reused

    max_workers = max(1, min(total, os.cpu_count() or 1))

    def _prepare_one(original_path):
        try:
            proxy_path = build_promptmatch_proxy_path(original_path, cache_dir)
            if os.path.isfile(proxy_path):
                return original_path, proxy_path, False, None
            proxy_path = ensure_promptmatch_proxy(original_path, cache_dir)
            return original_path, proxy_path, True, None
        except Exception as exc:
            return original_path, original_path, False, exc

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_prepare_one, original_path): original_path for original_path in image_paths}
        for future in as_completed(future_map):
            original_path, resolved_path, was_generated, exc = future.result()
            proxy_map[original_path] = resolved_path
            completed += 1
            if exc is not None:
                print(f"  [WARN] proxy fallback for {original_path}: {exc}")
            elif was_generated:
                generated += 1
            else:
                reused += 1
        if progress_cb:
            progress_cb(completed, total, generated, reused)

    return proxy_map, generated, reused


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


def is_cuda_oom_error(exc):
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def get_auto_batch_size(device, backend=None, reference_vram_gb=32):
    # Scale batch size from currently free VRAM so large models do not OOM instantly.
    if device != "cuda" or not torch.cuda.is_available():
        return DEFAULT_BATCH_SIZE

    if backend is not None:
        reference_vram_gb = 20
        if backend.backend == "openclip":
            model_name = backend._openclip_model.lower()
            if "bigg" in model_name or "xxlarge" in model_name:
                reference_vram_gb = 32
            elif "large_d_320" in model_name:
                reference_vram_gb = 28
            elif "vit-h" in model_name:
                reference_vram_gb = 24
            elif "base_w" in model_name:
                reference_vram_gb = 20
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


def iter_imagereward_scores(image_paths, model, device, prompt, source_paths=None):
    # Yield partial results so Gradio can update progress while large folders are scored.
    scores = {}
    total = len(image_paths)
    done = 0
    batch_size = get_auto_batch_size(device)
    model = model.to(device).eval()
    print(f"[ImageReward] Using batch size {batch_size}")

    while done < total:
        current_size = min(batch_size, total - done)
        batch_paths = image_paths[done:done + current_size]
        batch_source_paths = source_paths[done:done + current_size] if source_paths is not None else batch_paths
        batch_filenames = [os.path.basename(p) for p in batch_source_paths]

        try:
            with torch.no_grad():
                _, batch_rewards = model.inference_rank(prompt, batch_paths)
            for filename, source_path, reward in zip(batch_filenames, batch_source_paths, batch_rewards):
                scores[filename] = {"score": float(reward), "path": source_path}
            done += len(batch_paths)
            yield {"type": "progress", "done": done, "total": total, "batch_size": batch_size, "scores": dict(scores)}
        except Exception as exc:
            if device == "cuda" and is_cuda_oom_error(exc) and current_size > 1:
                batch_size = max(1, current_size // 2)
                print(f"[ImageReward] CUDA OOM, retrying with batch size {batch_size}")
                torch.cuda.empty_cache()
                yield {"type": "oom", "done": done, "total": total, "batch_size": batch_size, "scores": dict(scores)}
                continue

            print(f"Batch error at {done}: {exc}", file=sys.stderr)
            for filename, source_path in zip(batch_filenames, batch_source_paths):
                scores[filename] = {"score": -float('inf'), "path": source_path}
            done += len(batch_paths)
            yield {"type": "progress", "done": done, "total": total, "batch_size": batch_size, "scores": dict(scores)}
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()


class ModelBackend:
    # Small adapter that hides the differences between OpenAI CLIP, OpenCLIP, and SigLIP.
    def __init__(self, device, backend="openclip", clip_model="ViT-L/14",
                 openclip_model="ViT-bigG-14", openclip_pretrained="laion2b_s39b_b160k",
                 siglip_model="google/siglip-so400m-patch14-384"):
        self.device = device
        self.backend = backend
        self._clip_model = clip_model
        self._openclip_model = openclip_model
        self._openclip_pre = openclip_pretrained
        self._siglip_model = siglip_model
        self._load()

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
        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self._openclip_model, pretrained=self._openclip_pre, device=self.device
            )
        except RuntimeError as exc:
            if "install the latest timm" in str(exc).lower():
                raise RuntimeError(
                    "This OpenCLIP model requires a newer timm build.\n"
                    "Run: python -m pip install --upgrade timm"
                )
            raise
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self._openclip_model)
        print("[OpenCLIP] Ready.")

    def _load_siglip(self):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            sys.exit("transformers not installed.\nRun: pip install transformers")
        print(f"[SigLIP] Loading {self._siglip_model} …")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self._siglip_model)
        self._model = AutoModel.from_pretrained(self._siglip_model, torch_dtype=dtype).to(self.device)
        self._model.eval()
        print("[SigLIP] Ready.")

    def encode_text(self, prompt):
        # Average a few prompt phrasings to make matching a little less brittle.
        phrases = [f"a photo of a {prompt}", f"a photo of {prompt}", prompt]
        with torch.no_grad():
            if self.backend == "openai":
                toks = self._clip_mod.tokenize(phrases).to(self.device)
                feat = self._model.encode_text(toks)
            elif self.backend == "openclip":
                toks = self._tokenizer(phrases).to(self.device)
                feat = self._model.encode_text(toks)
            else:
                inputs = self._processor(text=phrases, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
                feat = self._model.get_text_features(**inputs)
            feat = F.normalize(feat.float(), dim=-1)
            return F.normalize(feat.mean(dim=0, keepdim=True), dim=-1)

    def encode_images_batch(self, pil_images):
        with torch.no_grad():
            if self.backend in ("openai", "openclip"):
                tensors = torch.stack([self._preprocess(img) for img in pil_images]).to(self.device)
                tensors = tensors.to(next(self._model.parameters()).dtype)
                feat = self._model.encode_image(tensors)
            else:
                inputs = self._processor(images=pil_images, return_tensors="pt").to(self.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(next(self._model.parameters()).dtype)
                feat = self._model.get_image_features(**inputs)
            return F.normalize(feat.float(), dim=-1)


def score_all(image_paths, backend, pos_emb, neg_emb, progress_cb=None, proxy_resolver=None):
    # PromptMatch scoring path: embed images in batches, then compare against text embeddings.
    results = {}
    total = len(image_paths)
    done = 0
    batch_size = get_auto_batch_size(backend.device, backend)
    print(f"[PromptMatch] Using batch size {batch_size}")

    def _score_items(valid_items):
        pil_imgs = []
        loaded = []
        for original_path, scoring_path in valid_items:
            try:
                with Image.open(scoring_path) as src_img:
                    pil_imgs.append(src_img.convert("RGB"))
                loaded.append((original_path, scoring_path))
            except Exception as exc:
                print(f"  [WARN] {original_path}: {exc}")
                results[os.path.basename(original_path)] = {"pos": 0.0, "neg": None, "path": original_path, "failed": True}
        if not pil_imgs:
            return

        feat = backend.encode_images_batch(pil_imgs)
        pos_sims = (feat @ pos_emb.T).squeeze(1).tolist()
        neg_sims = (feat @ neg_emb.T).squeeze(1).tolist() if neg_emb is not None else [None] * len(loaded)
        for (original_path, _), pos_score, neg_score in zip(loaded, pos_sims, neg_sims):
            results[os.path.basename(original_path)] = {
                "pos": float(pos_score),
                "neg": float(neg_score) if neg_score is not None else None,
                "path": original_path,
                "failed": False,
            }

    while done < total:
        current_size = min(batch_size, total - done)
        batch = image_paths[done:done + current_size]
        batch_start = done + 1
        batch_end = done + len(batch)
        print(f"[PromptMatch] Batch {batch_start}-{batch_end}/{total} ({len(batch)} images)")
        valid = []
        for original_path in batch:
            try:
                scoring_path = proxy_resolver(original_path) if proxy_resolver is not None else original_path
                valid.append((original_path, scoring_path))
            except Exception as exc:
                print(f"  [WARN] {original_path}: {exc}")
                results[os.path.basename(original_path)] = {"pos": 0.0, "neg": None, "path": original_path, "failed": True}
        if valid:
            try:
                _score_items(valid)
            except Exception as exc:
                if backend.device == "cuda" and is_cuda_oom_error(exc) and current_size > 1:
                    batch_size = max(1, current_size // 2)
                    print(f"[PromptMatch] CUDA OOM, retrying with batch size {batch_size}")
                    torch.cuda.empty_cache()
                    if progress_cb:
                        progress_cb(done, total, batch_size, True)
                    continue
                print(f"  [WARN] batch error, retrying individually: {exc}")
                recovered = 0
                failed = 0
                for item in valid:
                    try:
                        _score_items([item])
                        original_path, _ = item
                        if not results.get(os.path.basename(original_path), {}).get("failed", False):
                            recovered += 1
                        else:
                            failed += 1
                    except Exception as single_exc:
                        original_path, _ = item
                        print(f"  [WARN] single-image error for {original_path}: {single_exc}")
                        results[os.path.basename(original_path)] = {"pos": 0.0, "neg": None, "path": original_path, "failed": True}
                        failed += 1
                print(f"[PromptMatch] Individual retry result: {recovered} recovered, {failed} failed")
            if backend.device == "cuda":
                torch.cuda.empty_cache()
        done += len(batch)
        if progress_cb:
            progress_cb(done, total, batch_size, False)
    return results


MODEL_CHOICES = [
    ("SigLIP  so400m-patch14-384  ★ recommended", "siglip", {"siglip_model": "google/siglip-so400m-patch14-384"}),
    ("SigLIP  large-patch16-384", "siglip", {"siglip_model": "google/siglip-large-patch16-384"}),
    ("SigLIP  base-patch16-224", "siglip", {"siglip_model": "google/siglip-base-patch16-224"}),
    ("OpenCLIP  ViT-bigG-14  laion2b  ★ best CLIP", "openclip", {"openclip_model": "ViT-bigG-14", "openclip_pretrained": "laion2b_s39b_b160k"}),
    ("OpenCLIP  ViT-H-14  laion2b", "openclip", {"openclip_model": "ViT-H-14", "openclip_pretrained": "laion2b_s32b_b79k"}),
    ("OpenCLIP  ViT-L-14  laion2b", "openclip", {"openclip_model": "ViT-L-14", "openclip_pretrained": "laion2b_s32b_b82k"}),
    ("OpenCLIP  ConvNeXt-Base-W  laion2b", "openclip", {"openclip_model": "convnext_base_w", "openclip_pretrained": "laion2b_s13b_b82k"}),
    ("OpenCLIP  ConvNeXt-Large-D-320  laion2b", "openclip", {"openclip_model": "convnext_large_d_320", "openclip_pretrained": "laion2b_s29b_b131k_ft"}),
    ("OpenCLIP  ConvNeXt-XXLarge  laion2b", "openclip", {"openclip_model": "convnext_xxlarge", "openclip_pretrained": "laion2b_s34b_b82k_augreg"}),
    ("OpenAI CLIP  ViT-L/14@336px", "openai", {"clip_model": "ViT-L/14@336px"}),
    ("OpenAI CLIP  ViT-L/14", "openai", {"clip_model": "ViT-L/14"}),
    ("OpenAI CLIP  ViT-B/32  (fastest)", "openai", {"clip_model": "ViT-B/32"}),
]
MODEL_LABELS = [choice[0] for choice in MODEL_CHOICES]


def label_for_backend(backend):
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
    return "SELECTED", "REJECTED", "selected", "rejected"


def promptmatch_slider_range(scores):
    pos_vals = [v["pos"] for v in scores.values() if not v.get("failed", False)]
    neg_vals = [v["neg"] for v in scores.values() if not v.get("failed", False) and v.get("neg") is not None]
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


def status_line(method, left_items, right_items, scores, overrides):
    left_name, right_name, _, _ = method_labels(method)
    if not scores:
        failed = 0
    elif method == METHOD_PROMPTMATCH and all("pos" in v for v in scores.values()):
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
    # Build the two visible buckets while preserving any manual user overrides.
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
            side = overrides.get(fname)
            if side is None:
                if item.get("failed", False):
                    side = right_name
                else:
                    pos_ok = item["pos"] >= main_threshold
                    neg_ok = (item["neg"] is None) or (item["neg"] < aux_threshold)
                    side = left_name if (pos_ok and neg_ok) else right_name
            score_text = "FAILED" if item.get("failed", False) else f"{item['pos']:.3f}"
            caption = f"{'✋ ' if fname in overrides else ''}{score_text} | {fname}"
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

    # Start with the lighter recommended SigLIP model so first-run download size
    # and VRAM pressure are more reasonable than the much larger ViT-bigG-14.
    prompt_backend = ModelBackend(
        device,
        backend="siglip",
        siglip_model="google/siglip-so400m-patch14-384",
    )

    # Shared mutable state for the one-page app. Gradio callbacks update this in place.
    state = {
        "method": METHOD_PROMPTMATCH,
        "source_dir": source_dir,
        "scores": {},
        "overrides": {},
        "left_marked": [],
        "right_marked": [],
        "preview_fname": None,
        "backend": prompt_backend,
        "ir_model": None,
        "hist_geom": None,
        "proxy_folder_key": None,
        "proxy_cache_dir": None,
        "proxy_map": {},
        "use_proxy_display": True,
    }

    def sync_promptmatch_proxy_cache(folder):
        folder_key = normalize_folder_identity(folder)
        if state["proxy_folder_key"] != folder_key:
            clear_promptmatch_proxy_cache(state.get("proxy_cache_dir"))
            state["proxy_folder_key"] = folder_key
            state["proxy_cache_dir"] = get_promptmatch_proxy_cache_dir(folder)
            state["proxy_map"] = {}
        return state["proxy_cache_dir"]

    def tooltip_head(pairs):
        mapping = json.dumps(pairs)
        return f"""
<script>
(() => {{
  const tooltips = {mapping};
  // Hidden text inputs are used as tiny event bridges from custom JS back into Gradio callbacks.
  const pushThumbAction = (value) => {{
    const root = document.getElementById("hy-thumb-action");
    if (!root) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    input.value = value;
    input.dispatchEvent(new Event("input", {{ bubbles: true }}));
    input.dispatchEvent(new Event("change", {{ bubbles: true }}));
  }};
  const readMarkedState = () => {{
    const root = document.getElementById("hy-mark-state");
    if (!root) return {{ left: [], right: [] }};
    const input = root.querySelector("input, textarea");
    if (!input || !input.value) return {{ left: [], right: [] }};
    try {{
      return JSON.parse(input.value);
    }} catch {{
      return {{ left: [], right: [] }};
    }}
  }};
  let repaintTimers = [];
  const scheduleRepaint = () => {{
    // Gallery DOM mutates a moment after clicks; repaint a few times to catch the final layout.
    ensureThumbBehavior();
    for (const timer of repaintTimers) clearTimeout(timer);
    repaintTimers = [
      setTimeout(ensureThumbBehavior, 40),
      setTimeout(ensureThumbBehavior, 140),
      setTimeout(ensureThumbBehavior, 320),
    ];
  }};
  const ensureThumbBehavior = () => {{
    // Re-apply green/red borders after every gallery rerender and preview change.
    const markedState = readMarkedState();
    const allMarked = new Set([...(markedState.left || []), ...(markedState.right || [])]);
    const heldSet = new Set(markedState.held || []);
    for (const [galleryId, side] of [["hy-left-gallery", "left"], ["hy-right-gallery", "right"]]) {{
      const root = document.getElementById(galleryId);
      if (!root) continue;
      if (!root.dataset.hyShiftHooked) {{
        root.addEventListener("click", (event) => {{
          const card = event.target.closest("button");
          if (!card || !root.contains(card)) return;
          if (!event.shiftKey) {{
            setTimeout(scheduleRepaint, 30);
            setTimeout(scheduleRepaint, 140);
            setTimeout(scheduleRepaint, 320);
            return;
          }}
          const thumbButtons = Array.from(root.querySelectorAll("button")).filter((btn) => {{
            const img = btn.querySelector("img");
            const inDialog = !!btn.closest('[role="dialog"], [aria-modal="true"]');
            const hasCaption = !!(btn.querySelector(".caption-label span") || btn.querySelector('[class*="caption"]'));
            return !inDialog && !!img && hasCaption;
          }});
          const index = thumbButtons.indexOf(card);
          if (index < 0) return;
          event.preventDefault();
          event.stopPropagation();
          event.stopImmediatePropagation();
          pushThumbAction(`${{side}}:${{index}}:${{Date.now()}}`);
        }}, true);
        root.dataset.hyShiftHooked = "1";
      }}
      const thumbButtons = Array.from(root.querySelectorAll("button")).filter((btn) => {{
        const img = btn.querySelector("img");
        const inDialog = !!btn.closest('[role="dialog"], [aria-modal="true"]');
        const hasCaption = !!(btn.querySelector(".caption-label span") || btn.querySelector('[class*="caption"]'));
        return !inDialog && !!img && hasCaption;
      }});
      for (const card of thumbButtons) {{
        const img = card.querySelector("img");
        if (!img) continue;
        const captionEl = card.querySelector(".caption-label span") || card.querySelector('[class*="caption"]');
        const captionText = captionEl ? (captionEl.textContent || "") : "";
        const held = captionText.includes("✋ ");
        const parts = captionText.split("|");
        const fname = parts.length ? parts[parts.length - 1].trim() : "";
        const marked = (markedState[side] || []).includes(fname);
        if (!card) continue;
        card.style.position = "relative";
        card.style.boxSizing = "border-box";
        card.style.outline = marked ? "3px solid #58bb73" : (held ? "3px solid #dd3322" : "");
        card.style.outlineOffset = (marked || held) ? "-3px" : "";
        card.style.boxShadow = marked ? "inset 0 0 0 1px rgba(88,187,115,0.35)" : "";
        img.style.outline = "";
        img.style.outlineOffset = "";
      }}
    }}
    for (const el of document.querySelectorAll('[data-hy-preview-border="1"]')) {{
      el.style.outline = "";
      el.style.outlineOffset = "";
      el.style.boxShadow = "";
      el.style.border = "";
      el.style.borderRadius = "";
      el.style.background = "";
      el.style.padding = "";
      el.style.boxSizing = "";
      el.style.overflow = "";
      el.removeAttribute("data-hy-preview-border");
    }}
    const previewImages = Array.from(document.querySelectorAll(
      "#hy-left-gallery span.preview img, #hy-right-gallery span.preview img, #hy-left-gallery .preview img, #hy-right-gallery .preview img"
    )).filter((img) => {{
      const rect = img.getBoundingClientRect();
      const style = window.getComputedStyle(img);
      return rect.width >= 220 && rect.height >= 220 && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
    }});
    for (const chosen of previewImages) {{
      const captionText = chosen.getAttribute("alt") || "";
      const parts = captionText.split("|");
      const fname = parts.length ? parts[parts.length - 1].trim() : "";
      const marked = allMarked.has(fname);
      const held = heldSet.has(fname);
      if (!(marked || held)) continue;
      const color = marked ? "#58bb73" : "#dd3322";
      const mediaButton = chosen.closest("button.media-button");
      if (mediaButton && !mediaButton.matches("button.thumbnail-item, .thumbnail-item")) {{
        mediaButton.style.outline = "";
        mediaButton.style.outlineOffset = "";
        mediaButton.style.border = `4px solid ${{color}}`;
        mediaButton.style.boxShadow = "0 0 0 1px rgba(0, 0, 0, 0.28)";
        mediaButton.style.borderRadius = "10px";
        mediaButton.style.background = "#0a0a12";
        mediaButton.style.padding = "0";
        mediaButton.style.boxSizing = "border-box";
        mediaButton.style.overflow = "hidden";
        mediaButton.setAttribute("data-hy-preview-border", "1");
      }}
      chosen.style.border = "0";
      chosen.style.boxShadow = "none";
      chosen.style.borderRadius = "6px";
      chosen.setAttribute("data-hy-preview-border", "1");
    }}
  }};
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
    ensureThumbBehavior();
  }};
  const hookMarkState = () => {{
    const root = document.getElementById("hy-mark-state");
    if (!root || root.dataset.hyStateHooked) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    input.addEventListener("input", scheduleRepaint);
    input.addEventListener("change", scheduleRepaint);
    root.dataset.hyStateHooked = "1";
  }};
  applyTooltips();
  hookMarkState();
  scheduleRepaint();
  new MutationObserver(() => {{
    applyTooltips();
    hookMarkState();
    scheduleRepaint();
  }}).observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>
"""

    tooltips = {
        "hy-method": "Choose whether to sort by PromptMatch or ImageReward.",
        "hy-folder": "Path to the image folder you want to score. You can paste a full folder path here.",
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
        "hy-zoom-ui": "Choose how many thumbnails appear per row in both galleries.",
        "hy-use-proxy-display": "Show gallery images from cached proxies for faster browsing on large folders.",
        "hy-hist": "Histogram of current scores. In PromptMatch, click the top chart for positive threshold or bottom chart for negative threshold.",
        "hy-export": "COPY the current split into two SELECTED / REJECTED output folders inside source folder.",
        "hy-left-gallery": "Images currently in the left bucket. Click one to select it.",
        "hy-right-gallery": "Images currently in the right bucket. Click one to select it.",
        "hy-move-right": "Move all marked SELECTED images into REJECTED as manual overrides.",
        "hy-move-left": "Move all marked REJECTED images into SELECTED as manual overrides.",
        "hy-clear-status": "Remove manual override status from all marked images so they snap back to their scored bucket.",
    }

    def ensure_imagereward_model():
        if state["ir_model"] is None:
            state["ir_model"] = get_imagereward_utils().load("ImageReward-v1.0")
        return state["ir_model"]

    def gallery_update(items, columns=None):
        update_kwargs = {"value": items, "selected_index": None}
        if columns is not None:
            update_kwargs["columns"] = columns
        return gr.update(**update_kwargs)

    def gallery_display_items(items):
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

    def selection_info():
        left_count = len(state.get("left_marked", []))
        right_count = len(state.get("right_marked", []))
        if not left_count and not right_count:
            return "Shift+click thumbnails to mark multiple images."
        return f"Marked: **{left_count}** in SELECTED, **{right_count}** in REJECTED"

    def marked_state_json():
        return json.dumps({
            "left": state.get("left_marked", []),
            "right": state.get("right_marked", []),
            "held": list(state.get("overrides", {}).keys()),
            "preview": state.get("preview_fname"),
        })

    def render_histogram(method, scores, main_threshold, aux_threshold):
        # Draw a lightweight PIL histogram image instead of depending on a plotting library.
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
            W, CH = 300, max(60, int(130 * HIST_HEIGHT_SCALE))
            PAD_L, PAD_R = 38, 8
            PAD_TOP, PAD_BOT = max(10, int(18 * HIST_HEIGHT_SCALE)), max(14, int(22 * HIST_HEIGHT_SCALE))
            GAP = max(16, int(28 * HIST_HEIGHT_SCALE))
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
        W, H = 300, max(70, int(130 * HIST_HEIGHT_SCALE))
        PAD_L, PAD_R = 38, 8
        PAD_TOP, PAD_BOT = max(10, int(18 * HIST_HEIGHT_SCALE)), max(14, int(22 * HIST_HEIGHT_SCALE))
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
        # Single place that rebuilds gallery contents, status text, histogram, and marked-state JSON.
        left_name, right_name, _, _ = method_labels(state["method"])
        zoom_columns = int(state.get("zoom_columns", 5))
        left_items, right_items = build_split(
            state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold
        )
        left_names = {os.path.basename(path) for path, _ in left_items}
        right_names = {os.path.basename(path) for path, _ in right_items}
        state["left_marked"] = [name for name in state.get("left_marked", []) if name in left_names]
        state["right_marked"] = [name for name in state.get("right_marked", []) if name in right_names]
        return (
            f"### {left_name} — {len(left_items)} images",
            gallery_update(gallery_display_items(left_items), columns=zoom_columns),
            f"### {right_name} — {len(right_items)} images",
            gallery_update(gallery_display_items(right_items), columns=zoom_columns),
            status_line(state["method"], left_items, right_items, state["scores"], state["overrides"]),
            render_histogram(state["method"], state["scores"], main_threshold, aux_threshold),
            selection_info(),
            marked_state_json(),
        )

    def score_imagereward(folder_paths, positive_prompt, negative_prompt, penalty_weight, progress):
        # Optional penalty prompt is implemented as a second pass whose score is subtracted.
        model = ensure_imagereward_model()
        positive_prompt = (positive_prompt or "").strip() or IR_PROMPT
        negative_prompt = (negative_prompt or "").strip()
        penalty_weight = float(penalty_weight)
        proxy_map = {}
        cache_dir = state.get("proxy_cache_dir")
        scoring_paths = list(folder_paths)

        if cache_dir:
            def _proxy_prep_cb(done, total, generated, reused):
                desc = f"Preparing ImageReward proxies {done}/{total}"
                if generated or reused:
                    desc += f" ({generated} new, {reused} reused)"
                progress(PROMPTMATCH_PROXY_PROGRESS_SHARE * (done / max(total, 1)), desc=desc)

            progress(0, desc=f"Preparing ImageReward proxies 0/{len(folder_paths)}")
            proxy_map, generated, reused = prepare_promptmatch_proxies(
                folder_paths,
                cache_dir,
                progress_cb=_proxy_prep_cb,
            )
            state["proxy_map"] = dict(proxy_map)
            scoring_paths = [proxy_map.get(path, path) for path in folder_paths]
            print(f"[ImageReward] Proxy prep complete: {generated} new, {reused} reused")

        base_scores = {}
        progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Scoring {len(folder_paths)} images with ImageReward...")
        for event in iter_imagereward_scores(scoring_paths, model, device, positive_prompt, source_paths=folder_paths):
            if event["type"] == "oom":
                progress(
                    PROMPTMATCH_PROXY_PROGRESS_SHARE
                    + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                    desc=f"ImageReward OOM, retrying batch {event['batch_size']}",
                )
                continue
            progress(
                PROMPTMATCH_PROXY_PROGRESS_SHARE
                + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                desc=f"ImageReward {event['done']}/{event['total']}",
            )
            base_scores = event["scores"]

        penalty_scores = {}
        if negative_prompt:
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Applying penalty prompt to {len(folder_paths)} images...")
            for event in iter_imagereward_scores(scoring_paths, model, device, negative_prompt, source_paths=folder_paths):
                if event["type"] == "oom":
                    progress(
                        PROMPTMATCH_PROXY_PROGRESS_SHARE
                        + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                        desc=f"Penalty OOM, retrying batch {event['batch_size']}",
                    )
                    continue
                progress(
                    PROMPTMATCH_PROXY_PROGRESS_SHARE
                    + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                    desc=f"Penalty prompt {event['done']}/{event['total']}",
                )
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
        state["left_marked"] = []
        state["right_marked"] = []
        state["preview_fname"] = None
        if method == METHOD_PROMPTMATCH:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(label="Primary thresh (>=  SELECTED)", value=0.14, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(label="Neg threshold (< -> passes)", visible=True, value=NEGATIVE_THRESHOLD, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(value="PromptMatch sorts by text-image similarity. Use a positive prompt and optional negative prompt."),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(label="Primary thresh (>=  SELECTED)", value=IMAGEREWARD_THRESHOLD, minimum=IMAGEREWARD_SLIDER_MIN, maximum=IMAGEREWARD_SLIDER_MAX),
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
            selection_info(),
            marked_state_json(),
            main_upd,
            aux_upd,
        )

    def score_folder(method, folder, model_label, pos_prompt, neg_prompt, ir_prompt, ir_negative_prompt, ir_penalty_weight, progress=gr.Progress()):
        # Main entrypoint for "Run scoring"; both methods converge back into current_view().
        folder = (folder or "").strip()
        if not folder or not os.path.isdir(folder):
            return empty_result(f"Invalid folder: {folder!r}", method)

        image_paths = scan_image_paths(folder)
        if not image_paths:
            return empty_result(f"No images found in {folder}", method)

        state["method"] = method
        sync_promptmatch_proxy_cache(folder)
        state["source_dir"] = folder
        state["overrides"] = {}
        state["left_marked"] = []
        state["right_marked"] = []
        state["preview_fname"] = None

        if method == METHOD_PROMPTMATCH:
            cfg = get_model_config(model_label)
            if cfg is None:
                return empty_result(f"Unknown PromptMatch model: {model_label}", method)
            _, backend_name, kwargs = cfg
            if label_for_backend(state["backend"]) != model_label:
                try:
                    state["backend"] = ModelBackend(device, backend=backend_name, **kwargs)
                except Exception as exc:
                    return empty_result(str(exc), method)

            proxy_map = {}
            cache_dir = state.get("proxy_cache_dir")
            if cache_dir:
                def _proxy_prep_cb(done, total, generated, reused):
                    desc = f"Preparing PromptMatch proxies {done}/{total}"
                    if generated or reused:
                        desc += f" ({generated} new, {reused} reused)"
                    progress(PROMPTMATCH_PROXY_PROGRESS_SHARE * (done / max(total, 1)), desc=desc)

                progress(0, desc=f"Preparing PromptMatch proxies 0/{len(image_paths)}")
                proxy_map, generated, reused = prepare_promptmatch_proxies(
                    image_paths,
                    cache_dir,
                    progress_cb=_proxy_prep_cb,
                )
                state["proxy_map"] = dict(proxy_map)
                print(f"[PromptMatch] Proxy prep complete: {generated} new, {reused} reused")

            pos_prompt = (pos_prompt or "").strip() or SEARCH_PROMPT
            neg_prompt = (neg_prompt or "").strip()
            pos_emb = state["backend"].encode_text(pos_prompt)
            neg_emb = state["backend"].encode_text(neg_prompt) if neg_prompt else None
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Scoring {len(image_paths)} images with PromptMatch...")

            def _cb(done, total, batch_size, oom_retry):
                label = f"PromptMatch {done}/{total} (autobatch {batch_size})"
                if oom_retry:
                    label = f"PromptMatch OOM, retrying autobatch {batch_size}"
                progress(
                    PROMPTMATCH_PROXY_PROGRESS_SHARE
                    + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (done / max(total, 1))),
                    desc=label,
                )

            def _proxy_resolver(original_path):
                return proxy_map.get(original_path, original_path)

            state["scores"] = score_all(
                image_paths,
                state["backend"],
                pos_emb,
                neg_emb,
                progress_cb=_cb,
                proxy_resolver=_proxy_resolver,
            )
            pos_min, pos_max, pos_mid, neg_min, neg_max, neg_mid, has_neg = promptmatch_slider_range(state["scores"])
            left_head, left_gallery, right_head, right_gallery, status, hist, sel_info, mark_state = current_view(pos_mid, neg_mid)
            return (
                left_head,
                left_gallery,
                right_head,
                right_gallery,
                status,
                hist,
                sel_info,
                mark_state,
                gr.update(minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX, value=pos_mid, label="Primary thresh (>=  SELECTED)"),
                gr.update(minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX, value=neg_mid, visible=True, interactive=has_neg, label="Neg threshold (< -> passes)"),
            )

        state["scores"] = score_imagereward(
            image_paths,
            ir_prompt,
            ir_negative_prompt,
            ir_penalty_weight,
            progress,
        )
        lo, hi, mid = imagereward_slider_range(state["scores"])
        left_head, left_gallery, right_head, right_gallery, status, hist, sel_info, mark_state = current_view(mid, NEGATIVE_THRESHOLD)
        return (
            left_head,
            left_gallery,
            right_head,
            right_gallery,
            status,
            hist,
            sel_info,
            mark_state,
            gr.update(minimum=IMAGEREWARD_SLIDER_MIN, maximum=IMAGEREWARD_SLIDER_MAX, value=mid, label="Primary thresh (>=  SELECTED)"),
            gr.update(value=NEGATIVE_THRESHOLD, visible=False),
        )

    def update_split(main_threshold, aux_threshold):
        return current_view(main_threshold, aux_threshold)

    def update_proxy_display(use_proxy_display, main_threshold, aux_threshold):
        state["use_proxy_display"] = bool(use_proxy_display)
        return current_view(main_threshold, aux_threshold)

    def toggle_mark(action, main_threshold, aux_threshold):
        # Shift-click marks thumbnails for bulk move/clear actions without opening the preview.
        if not action:
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        try:
            side, raw_index, _ = str(action).split(":", 2)
            index = int(raw_index)
        except Exception:
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        left_items, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)
        items = left_items if side == "left" else right_items
        marked_key = "left_marked" if side == "left" else "right_marked"
        if 0 <= index < len(items):
            fname = os.path.basename(items[index][0])
            if fname in state[marked_key]:
                state[marked_key] = [name for name in state[marked_key] if name != fname]
            else:
                state[marked_key].append(fname)
        return current_view(main_threshold, aux_threshold)

    def remember_preview(side, main_threshold, aux_threshold, evt: gr.SelectData):
        left_items, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)
        items = left_items if side == "left" else right_items
        if 0 <= evt.index < len(items):
            state["preview_fname"] = os.path.basename(items[evt.index][0])
            return marked_state_json()
        return gr.update()

    def remember_preview_left(main_threshold, aux_threshold, evt: gr.SelectData):
        return remember_preview("left", main_threshold, aux_threshold, evt)

    def remember_preview_right(main_threshold, aux_threshold, evt: gr.SelectData):
        return remember_preview("right", main_threshold, aux_threshold, evt)

    def move_right(main_threshold, aux_threshold):
        left_name, right_name, _, _ = method_labels(state["method"])
        for fname in state["left_marked"]:
            state["overrides"][fname] = right_name
        state["left_marked"] = []
        state["right_marked"] = []
        return current_view(main_threshold, aux_threshold)

    def move_left(main_threshold, aux_threshold):
        left_name, right_name, _, _ = method_labels(state["method"])
        for fname in state["right_marked"]:
            state["overrides"][fname] = left_name
        state["left_marked"] = []
        state["right_marked"] = []
        return current_view(main_threshold, aux_threshold)

    def clear_status(main_threshold, aux_threshold):
        for fname in set(state["left_marked"] + state["right_marked"]):
            state["overrides"].pop(fname, None)
        state["left_marked"] = []
        state["right_marked"] = []
        return current_view(main_threshold, aux_threshold)

    def set_from_percentile(percentile, main_threshold, aux_threshold):
        new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
        return (*current_view(new_threshold, aux_threshold), gr.update(value=new_threshold))

    def update_zoom(zoom_value, main_threshold, aux_threshold):
        # Gallery zoom is implemented by changing the number of columns in both galleries.
        try:
            state["zoom_columns"] = max(2, min(10, int(zoom_value)))
        except Exception:
            state["zoom_columns"] = 5
        left_head, left_gallery, right_head, right_gallery, status, hist, sel_info, mark_state = current_view(main_threshold, aux_threshold)
        return left_head, left_gallery, right_head, right_gallery, status, hist, sel_info, mark_state

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
        # Export is a lossless copy, not a rewrite or recompression of the originals.
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
    .sidebar-box .gr-accordion { margin-bottom:10px !important; border:1px solid #2c2c39 !important; border-radius:8px !important; background:#151520 !important; }
    .sidebar-box .gr-accordion summary, .sidebar-box .gr-accordion button { font-family:monospace !important; font-size:.9rem !important; color:#d7dbc8 !important; }
    .method-note { font-family:monospace; color:#8e9d80; background:#11111a; border-radius:8px; padding:6px 9px; }
    .method-note p { margin:0 !important; font-family:monospace !important; font-size:.82rem !important; line-height:1.35 !important; color:#8e9d80 !important; }
    .status-md p { font-family:monospace !important; color:#9fc27c !important; }
    .hist-img img { cursor:crosshair !important; border-radius:6px; }
    .grid-wrap img { object-fit: contain !important; background: #0a0a12; }
    .grid-wrap .caption-label span, .grid-wrap [class*="caption"] { font-family:monospace !important; font-size:.72em !important; color:#8899aa !important; }
    .move-col { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px; padding:10px 6px; background:#0f0f16; border-radius:8px; border:1px solid #252535; }
    .move-col button { width:100%; }
    .sel-info p { font-family:monospace !important; font-size:.72em !important; color:#aabb88 !important; text-align:center; word-break:break-all; }
    #hy-folder textarea, #hy-folder input { min-height:60px !important; font-size:.96rem !important; }
    #hy-run, #hy-export { border-radius:8px !important; }
    #hy-run, #hy-run button, #hy-export, #hy-export button {
        background:#2f8f45 !important;
        background-image:none !important;
        border:1px solid #58bb73 !important;
        color:#f3fff2 !important;
        font-weight:700 !important;
        box-shadow:0 0 0 1px rgba(25, 55, 30, 0.15) inset !important;
    }
    #hy-run button, #hy-export button {
        min-height:40px !important;
        border-radius:8px !important;
    }
    #hy-run:hover, #hy-run button:hover, #hy-export:hover, #hy-export button:hover {
        background:#38a14f !important;
        background-image:none !important;
    }
    #hy-run button:disabled, #hy-export button:disabled {
        background:#256d35 !important;
        color:#d8ead8 !important;
    }
    #hy-zoom {
        background:transparent !important;
        border:0 !important;
        border-radius:0 !important;
        padding:0 !important;
        width:148px !important;
        max-width:148px !important;
        min-width:148px !important;
        height:auto !important;
        min-height:0 !important;
        margin-left:0 !important;
        overflow:visible !important;
        box-sizing:border-box !important;
    }
    #hy-zoom .head,
    #hy-zoom label,
    #hy-zoom .tab-like-container,
    #hy-zoom [data-testid="number-input"],
    #hy-zoom [data-testid="reset-button"],
    #hy-zoom .min_value,
    #hy-zoom .max_value { display:none !important; }
    #hy-zoom .wrap {
        gap:0 !important;
        width:100% !important;
        max-width:100% !important;
        overflow:visible !important;
        min-height:0 !important;
    }
    #hy-zoom .slider_input_container {
        padding:0 !important;
        gap:0 !important;
        width:100% !important;
        max-width:100% !important;
        overflow:visible !important;
        min-height:0 !important;
    }
    #hy-zoom input[type="range"] {
        margin:0 !important;
        height:14px !important;
        width:100% !important;
        min-width:100% !important;
        display:block !important;
    }
    .gallery-side { min-width:0; }
    .gallery-topbar { align-items:end; margin-bottom:8px; }
    .gallery-right-topbar { align-items:center; gap:12px; justify-content:space-between; }
    .zoom-inline-wrap { align-items:center; gap:8px; margin-left:auto; flex-wrap:nowrap; overflow:hidden; }
    .zoom-inline-label {
        flex:0 0 auto;
        overflow:hidden !important;
        min-width:34px !important;
        max-width:34px !important;
        line-height:1 !important;
    }
    .zoom-inline-label p {
        margin:0 !important;
        color:#8ec5ff !important;
        font-family:monospace !important;
        font-size:.72rem !important;
        line-height:1 !important;
        white-space:nowrap !important;
        overflow:hidden !important;
    }
    """

    with gr.Blocks(title="HybridSelector") as demo:
        gr.HTML("""
<h1>⬡ HybridSelector</h1>
<div class='subhead'>PromptMatch + ImageReward in one UI &middot; quick image triage &middot; created by vangel</div>
""")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=330, elem_classes=["sidebar-box"]):
                thumb_action = gr.Textbox(value="", visible="hidden", elem_id="hy-thumb-action")
                mark_state = gr.Textbox(value='{"left":[],"right":[]}', visible="hidden", elem_id="hy-mark-state")
                with gr.Accordion("1. Setup", open=True):
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
                    folder_input = gr.Textbox(
                        value=source_dir,
                        label="Image folder - paste a path here",
                        lines=2,
                        placeholder=folder_placeholder(),
                        elem_id="hy-folder",
                    )

                with gr.Accordion("2. Method settings", open=True):
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

                    run_btn = gr.Button("Run scoring", elem_id="hy-run", variant="primary")

                with gr.Accordion("3. Thresholds", open=False):
                    hist_plot = gr.Image(value=None, show_label=False, interactive=False, elem_classes=["hist-img"], elem_id="hy-hist")
                    main_slider = gr.Slider(minimum=-1.0, maximum=1.0, value=0.14, step=0.001, label="Primary thresh (>=  SELECTED)", elem_id="hy-main-slider")
                    aux_slider = gr.Slider(minimum=-1.0, maximum=1.0, value=NEGATIVE_THRESHOLD, step=0.001, label="Neg threshold (< -> passes)", elem_id="hy-aux-slider")
                    percentile_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Or keep top N%", elem_id="hy-percentile")
                    proxy_display_cb = gr.Checkbox(value=True, label="Use proxies for gallery display", elem_id="hy-use-proxy-display")
                    status_md = gr.Markdown("", elem_classes=["status-md"])

                with gr.Accordion("4. Export", open=False):
                    export_btn = gr.Button("Export folders", elem_id="hy-export", variant="primary")
                    export_tb = gr.Textbox(label="Export result", lines=3, interactive=False)

            with gr.Column(scale=5):
                with gr.Row(equal_height=False, elem_classes=["gallery-topbar"]):
                    with gr.Column(scale=1, elem_classes=["gallery-side"]):
                        left_head = gr.Markdown("### SELECTED")
                    with gr.Column(scale=0, min_width=100):
                        gr.HTML("")
                    with gr.Column(scale=1, elem_classes=["gallery-side"]):
                        with gr.Row(equal_height=False, elem_classes=["gallery-right-topbar"]):
                            right_head = gr.Markdown("### REJECTED")
                            with gr.Row(equal_height=False, elem_classes=["zoom-inline-wrap"]):
                                gr.Markdown("Tiles #", elem_classes=["zoom-inline-label"])
                                zoom_slider = gr.Slider(minimum=2, maximum=10, value=5, step=1, label="Thumbnail count", show_label=False, container=False, elem_id="hy-zoom")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes=["gallery-side"]):
                        left_gallery = gr.Gallery(show_label=False, columns=5, height="80vh", object_fit="contain", preview=True, allow_preview=True, elem_classes=["grid-wrap"], elem_id="hy-left-gallery")
                    with gr.Column(scale=0, min_width=100, elem_classes=["move-col"]):
                        sel_info = gr.Markdown("Shift+click thumbnails to mark multiple images.", elem_classes=["sel-info"])
                        move_right_btn = gr.Button("Move →", elem_id="hy-move-right")
                        move_left_btn = gr.Button("← Move", elem_id="hy-move-left")
                        clear_status_btn = gr.Button("Clear status", elem_id="hy-clear-status")
                    with gr.Column(scale=1, elem_classes=["gallery-side"]):
                        right_gallery = gr.Gallery(show_label=False, columns=5, height="80vh", object_fit="contain", preview=True, allow_preview=True, elem_classes=["grid-wrap"], elem_id="hy-right-gallery")

        method_dd.change(
            fn=configure_controls,
            inputs=[method_dd],
            outputs=[promptmatch_group, imagereward_group, main_slider, aux_slider, method_note],
        )

        run_btn.click(
            fn=score_folder,
            inputs=[method_dd, folder_input, model_dd, pos_prompt_tb, neg_prompt_tb, ir_prompt_tb, ir_negative_prompt_tb, ir_penalty_weight_tb],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider],
        )

        main_slider.change(fn=update_split, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state])
        aux_slider.change(fn=update_split, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state])
        percentile_slider.change(
            fn=set_from_percentile,
            inputs=[percentile_slider, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider],
        )
        zoom_slider.change(
            fn=update_zoom,
            inputs=[zoom_slider, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state],
        )
        proxy_display_cb.change(
            fn=update_proxy_display,
            inputs=[proxy_display_cb, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state],
        )
        left_gallery.select(fn=remember_preview_left, inputs=[main_slider, aux_slider], outputs=[mark_state])
        right_gallery.select(fn=remember_preview_right, inputs=[main_slider, aux_slider], outputs=[mark_state])
        thumb_action.change(fn=toggle_mark, inputs=[thumb_action, main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state])
        move_right_btn.click(fn=move_right, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state])
        move_left_btn.click(fn=move_left, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state])
        clear_status_btn.click(fn=clear_status, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state])
        hist_plot.select(fn=on_hist_click, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider])
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
