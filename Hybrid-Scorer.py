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

import base64
import io
import json
import math
import os
import re
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

APP_NAME = "HybridScorer"
APP_DISPLAY_NAME = "HybridSelector"


def load_app_version():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    version_path = os.path.join(script_dir, "VERSION")
    try:
        with open(version_path, "r", encoding="utf-8") as handle:
            version = handle.read().strip()
            if version:
                return version
    except OSError:
        pass
    return "0.0.0"


APP_VERSION = load_app_version()
APP_GITHUB_TAG = f"v{APP_VERSION}"
APP_WINDOW_TITLE = f"{APP_DISPLAY_NAME} {APP_GITHUB_TAG}"

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
PROMPT_GENERATOR_FLORENCE = "Florence-2"
PROMPT_GENERATOR_JOYCAPTION = "JoyCaption Beta One"
PROMPT_GENERATOR_JOYCAPTION_GGUF = "JoyCaption Beta One GGUF (Q4_K_M)"
PROMPT_GENERATOR_CHOICES = (
    PROMPT_GENERATOR_FLORENCE,
    PROMPT_GENERATOR_JOYCAPTION,
    PROMPT_GENERATOR_JOYCAPTION_GGUF,
)
DEFAULT_PROMPT_GENERATOR = PROMPT_GENERATOR_FLORENCE
FLORENCE_MODEL_ID = "florence-community/Florence-2-base"
FLORENCE_MAX_NEW_TOKENS = 256
JOYCAPTION_MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
JOYCAPTION_MAX_NEW_TOKENS = 320
JOYCAPTION_GGUF_REPO_ID = "cinnabrad/llama-joycaption-beta-one-hf-llava-mmproj-gguf"
JOYCAPTION_GGUF_FILENAME = "Llama-Joycaption-Beta-One-Hf-Llava-Q4_K_M.gguf"
JOYCAPTION_GGUF_MMPROJ_FILENAME = "llama-joycaption-beta-one-llava-mmproj-model-f16.gguf"
JOYCAPTION_GGUF_SETUP_HINT = "INSTALL_JOYCAPTION_GGUF=1 ./setup-venv312.sh"
DEFAULT_GENERATED_PROMPT_DETAIL = 2

SEARCH_PROMPT = "woman"
NEGATIVE_PROMPT = ""
NEGATIVE_THRESHOLD = 0.14
IR_PROMPT = (
    "masterpiece, best quality, ultra-detailed, cinematic, "
    "beautiful woman, dramatic lighting, chiaroscuro, "
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

    # ImageReward still imports a few helpers from transformers.modeling_utils
    # that now live in transformers.pytorch_utils in newer transformers builds.
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
        for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
            if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
                setattr(modeling_utils, name, getattr(pytorch_utils, name))
    except Exception:
        pass

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


def huggingface_file_cached(repo_id, filename):
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    try:
        cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
    except Exception:
        return False
    if not isinstance(cached, (str, bytes, os.PathLike)):
        return False
    return os.path.isfile(cached)


def describe_openai_clip_source(model_name):
    try:
        import clip as _clip
    except Exception:
        return "network or disk cache"
    url = _clip._MODELS.get(model_name)
    if not url:
        return "network or disk cache"
    filename = os.path.basename(url)
    cache_path = os.path.join(os.path.expanduser("~"), ".cache", "clip", filename)
    return "disk cache" if os.path.isfile(cache_path) else "network download"


def describe_openclip_source(model_name, pretrained_tag):
    try:
        import open_clip.pretrained as oc_pretrained
    except Exception:
        return "network or disk cache"
    cfg = oc_pretrained.get_pretrained_cfg(model_name, pretrained_tag)
    if not cfg:
        return "network or disk cache"
    if "file" in cfg:
        return "disk file"
    hf_ref = cfg.get("hf_hub", "")
    if hf_ref:
        model_id, filename = os.path.split(hf_ref)
        if model_id and filename and huggingface_file_cached(model_id, filename):
            return "disk cache"
        return "network download"
    return "network or disk cache"


def describe_siglip_source(model_name):
    cached = huggingface_file_cached(model_name, "config.json") and (
        huggingface_file_cached(model_name, "model.safetensors")
        or huggingface_file_cached(model_name, "pytorch_model.bin")
    )
    return "disk cache" if cached else "network download"


def describe_imagereward_source():
    cache_root = os.path.expanduser("~/.cache/ImageReward")
    model_ok = os.path.isfile(os.path.join(cache_root, "ImageReward.pt"))
    med_ok = os.path.isfile(os.path.join(cache_root, "med_config.json"))
    return "disk cache" if model_ok and med_ok else "network download"


def describe_florence_source():
    cached = huggingface_file_cached(FLORENCE_MODEL_ID, "config.json") and (
        huggingface_file_cached(FLORENCE_MODEL_ID, "model.safetensors")
        or huggingface_file_cached(FLORENCE_MODEL_ID, "pytorch_model.bin")
    )
    return "disk cache" if cached else "network download"


def describe_huggingface_transformers_source(repo_id):
    has_config = huggingface_file_cached(repo_id, "config.json")
    has_processor = (
        huggingface_file_cached(repo_id, "processor_config.json")
        or huggingface_file_cached(repo_id, "preprocessor_config.json")
        or huggingface_file_cached(repo_id, "tokenizer_config.json")
    )
    has_weights = (
        huggingface_file_cached(repo_id, "model.safetensors")
        or huggingface_file_cached(repo_id, "pytorch_model.bin")
        or huggingface_file_cached(repo_id, "model.safetensors.index.json")
    )
    return "disk cache" if has_config and has_processor and has_weights else "network download"


def describe_joycaption_gguf_source():
    cached = (
        huggingface_file_cached(JOYCAPTION_GGUF_REPO_ID, JOYCAPTION_GGUF_FILENAME)
        and huggingface_file_cached(JOYCAPTION_GGUF_REPO_ID, JOYCAPTION_GGUF_MMPROJ_FILENAME)
    )
    return "disk cache" if cached else "network download"


def describe_prompt_generator_source(generator_name):
    if generator_name == PROMPT_GENERATOR_FLORENCE:
        return describe_florence_source()
    if generator_name == PROMPT_GENERATOR_JOYCAPTION:
        return describe_huggingface_transformers_source(JOYCAPTION_MODEL_ID)
    if generator_name == PROMPT_GENERATOR_JOYCAPTION_GGUF:
        return describe_joycaption_gguf_source()
    return "network or disk cache"


def prompt_generator_supports_torch_cleanup(generator_name):
    return generator_name in {PROMPT_GENERATOR_FLORENCE, PROMPT_GENERATOR_JOYCAPTION}


def cuda_prefers_bfloat16():
    return (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )


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


def get_image_paths_signature(image_paths):
    digest = sha256()
    for path in image_paths:
        stat = os.stat(path)
        digest.update(os.path.normcase(os.path.abspath(path)).encode("utf-8", errors="ignore"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


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


def get_imagereward_penalty_offset(penalty_values):
    valid = [float(val) for val in penalty_values if val is not None]
    if not valid:
        return None
    return min(valid)


def compute_imagereward_final_score(base_score, penalty_score, penalty_weight, penalty_offset=None):
    final_score = float(base_score)
    if penalty_score is not None and penalty_offset is not None:
        effective_penalty = max(0.0, float(penalty_score) - float(penalty_offset))
        final_score = final_score - (float(penalty_weight) * effective_penalty)
    return float(final_score)


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


def slider_step_floor(value, step=0.001):
    return round(math.floor(float(value) / step) * step, 3)


def slider_step_ceil_exclusive(value, step=0.001):
    return round((math.floor(float(value) / step) + 1) * step, 3)


def extract_florence_caption(parsed, raw_text, task_prompt):
    def _clean(text):
        text = re.sub(r"<[^>]+>", " ", text or "")
        return re.sub(r"\s+", " ", text).strip()

    if isinstance(parsed, dict):
        value = parsed.get(task_prompt)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return ", ".join(value)
    if isinstance(parsed, str):
        return _clean(parsed)

    return _clean(raw_text)


def normalize_generated_prompt(text, keep_prose=False):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""

    lowered = text.lower()
    prefixes = (
        "in this image, we can see ",
        "in this image we can see ",
        "in this image, we see ",
        "in this image we see ",
        "this image shows ",
        "this image depicts ",
        "the image shows ",
        "the image depicts ",
        "this photo shows ",
        "this photo depicts ",
        "you are looking at ",
        "a photo of ",
        "an image of ",
        "image of ",
        "photo of ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    if keep_prose:
        text = re.sub(r"\s*([,:;])\s*", r"\1 ", text)
        text = re.sub(r"\s*\.\s*", ". ", text)
        text = re.sub(r"\s+", " ", text).strip(" ,;:-")
        return text.strip()

    text = re.sub(r"\s*[:;]\s*", ", ", text)
    text = re.sub(r"\.\s*", ", ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    parts = [part.strip(" ,.;:-") for part in text.split(",")]
    parts = [part for part in parts if part]
    return ", ".join(parts)


def florence_task_is_pure_text(task_prompt):
    return task_prompt in {
        "<CAPTION>",
        "<DETAILED_CAPTION>",
        "<MORE_DETAILED_CAPTION>",
        "<REGION_TO_CATEGORY>",
        "<REGION_TO_DESCRIPTION>",
        "<REGION_TO_OCR>",
    }


def florence_detail_config(detail_level):
    try:
        detail_level = int(detail_level)
    except Exception:
        detail_level = DEFAULT_GENERATED_PROMPT_DETAIL

    detail_level = max(1, min(3, detail_level))
    mapping = {
        1: ("Core facts", "<CAPTION>"),
        2: ("Balanced", "<DETAILED_CAPTION>"),
        3: ("Full", "<MORE_DETAILED_CAPTION>"),
    }
    return detail_level, *mapping[detail_level]


def joycaption_detail_config(detail_level):
    try:
        detail_level = int(detail_level)
    except Exception:
        detail_level = DEFAULT_GENERATED_PROMPT_DETAIL

    detail_level = max(1, min(3, detail_level))
    mapping = {
        1: (
            "Core facts",
            (
                "Write only a very short comma-separated tag list for this image. "
                "Use about 4 to 8 short tags total. "
                "Include only the main subject, medium or style, pose, setting, lighting, and one standout object if clearly visible. "
                "Do not write full sentences. "
                "Drop minor details completely and avoid repeated concepts. "
                "Avoid meta lead-ins like 'This image shows', 'In this image we can see', or 'You are looking at'. "
                "Write it so the result is useful as a text-to-image prompt."
            ),
        ),
        2: (
            "Balanced",
            (
                "Write one compact text-to-image prompt line for this image. "
                "Use about 18 to 35 words maximum. "
                "Prefer prompt fragments separated by commas, not full prose. "
                "Describe the visible subject, appearance, clothing, pose, background, lighting, framing, and the most notable objects. "
                "Keep only clearly useful details, avoid speculation, and keep it prompt-friendly. "
                "Avoid meta lead-ins like 'This image shows', 'In this image we can see', or 'You are looking at'. "
                "Write it so the result is useful as a text-to-image prompt."
            ),
        ),
        3: (
            "Full",
            (
                "Write one natural descriptive paragraph for this image. "
                "Use about 45 to 90 words maximum. "
                "Write in clear flowing prose, not a tag list. "
                "Include concrete visible details about subject, appearance, outfit, pose, composition, background, lighting, viewpoint, and notable objects. "
                "Use direct factual language that is still useful for a text-to-image prompt. "
                "Avoid meta lead-ins like 'This image shows', 'In this image we can see', or 'You are looking at'."
            ),
        ),
    }
    return detail_level, *mapping[detail_level]


def joycaption_max_new_tokens(detail_level):
    detail_level = max(1, min(3, int(detail_level)))
    mapping = {
        1: 24,
        2: 56,
        3: 128,
    }
    return mapping[detail_level]


def prompt_generator_detail_config(generator_name, detail_level):
    if generator_name == PROMPT_GENERATOR_FLORENCE:
        return florence_detail_config(detail_level)
    return joycaption_detail_config(detail_level)


def extract_joycaption_caption(text):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    text = re.sub(r"^(assistant|caption)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def image_to_data_url(image, image_format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


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
        "ir_penalty_weight": DEFAULT_IR_PENALTY_WEIGHT,
        "ir_cached_signature": None,
        "ir_cached_positive_prompt": None,
        "ir_cached_negative_prompt": None,
        "ir_cached_base_scores": None,
        "ir_cached_penalty_scores": None,
        "prompt_generator": DEFAULT_PROMPT_GENERATOR,
        "prompt_backend_cache": {},
        "generated_prompt": "",
        "generated_prompt_source": None,
        "generated_prompt_backend": DEFAULT_PROMPT_GENERATOR,
        "generated_prompt_detail": DEFAULT_GENERATED_PROMPT_DETAIL,
        "generated_prompt_variants": {},
        "generated_prompt_status": "Preview an image, then generate a prompt from it.",
    }

    def sync_promptmatch_proxy_cache(folder):
        folder_key = normalize_folder_identity(folder)
        if state["proxy_folder_key"] != folder_key:
            clear_promptmatch_proxy_cache(state.get("proxy_cache_dir"))
            state["proxy_folder_key"] = folder_key
            state["proxy_cache_dir"] = get_promptmatch_proxy_cache_dir(folder)
            state["proxy_map"] = {}
            state["ir_cached_signature"] = None
            state["ir_cached_positive_prompt"] = None
            state["ir_cached_negative_prompt"] = None
            state["ir_cached_base_scores"] = None
            state["ir_cached_penalty_scores"] = None
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
            const thumbButtons = Array.from(root.querySelectorAll("button")).filter((btn) => {{
              const img = btn.querySelector("img");
              const inDialog = !!btn.closest('[role="dialog"], [aria-modal="true"]');
              const hasCaption = !!(btn.querySelector(".caption-label span") || btn.querySelector('[class*="caption"]'));
              return !inDialog && !!img && hasCaption;
            }});
            const index = thumbButtons.indexOf(card);
            if (index >= 0) {{
              pushThumbAction(`preview:${{side}}:${{index}}:${{Date.now()}}`);
            }}
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
          pushThumbAction(`mark:${{side}}:${{index}}:${{Date.now()}}`);
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
        "hy-prompt-generator": "Choose which caption model should draft the prompt from the preview image.",
        "hy-generate-prompt": "Use the currently previewed image to generate a dense editable prompt with the selected caption backend.",
        "hy-generated-prompt": "Editable scratch prompt generated from the previewed image. You can tweak it before scoring or reinsert it into the active method.",
        "hy-generated-prompt-detail": "Choose whether the caption backend should describe only the core facts, a balanced amount of detail, or the full detailed prompt.",
        "hy-insert-prompt": "Copy the editable generated prompt back into the active method's main prompt field.",
        "hy-promptgen-status": "Small status readout for prompt generation.",
        "hy-run-pm": "Score the current folder with the selected method and prompts.",
        "hy-run-ir": "Score the current folder with the selected method and prompts.",
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
        "hy-fit-threshold": "Adjust the score threshold just enough so the marked images flip to the other bucket. Uses the previewed image if nothing is marked.",
        "hy-clear-status": "Remove manual override status from all marked images so they snap back to their scored bucket.",
    }

    def ensure_imagereward_model():
        if state["ir_model"] is None:
            state["ir_model"] = get_imagereward_utils().load("ImageReward-v1.0")
        return state["ir_model"]

    def ensure_florence_model():
        cached = state["prompt_backend_cache"].get(PROMPT_GENERATOR_FLORENCE)
        if cached and cached.get("model") is not None and cached.get("processor") is not None:
            return cached["model"], cached["processor"]

        try:
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Florence prompt generation needs a newer transformers build.\n"
                "Run setup again after updating requirements."
            ) from exc

        dtype = torch.float16 if device == "cuda" else torch.float32
        local_files_only = describe_florence_source() == "disk cache"
        print(f"[Florence] Loading {FLORENCE_MODEL_ID} …")
        processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_ID,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        model = Florence2ForConditionalGeneration.from_pretrained(
            FLORENCE_MODEL_ID,
            dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)
        model.eval()
        state["prompt_backend_cache"][PROMPT_GENERATOR_FLORENCE] = {
            "model": model,
            "processor": processor,
        }
        print("[Florence] Ready.")
        return model, processor

    def ensure_joycaption_model():
        cached = state["prompt_backend_cache"].get(PROMPT_GENERATOR_JOYCAPTION)
        if cached and cached.get("model") is not None and cached.get("processor") is not None:
            return cached["model"], cached["processor"]

        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "JoyCaption prompt generation needs a newer transformers build.\n"
                "Run setup again after updating requirements."
            ) from exc

        dtype = torch.bfloat16 if cuda_prefers_bfloat16() else torch.float16
        local_files_only = describe_prompt_generator_source(PROMPT_GENERATOR_JOYCAPTION) == "disk cache"
        print(f"[JoyCaption] Loading {JOYCAPTION_MODEL_ID} …")
        processor = AutoProcessor.from_pretrained(
            JOYCAPTION_MODEL_ID,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            JOYCAPTION_MODEL_ID,
            dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
        ).to(device)
        model.eval()
        state["prompt_backend_cache"][PROMPT_GENERATOR_JOYCAPTION] = {
            "model": model,
            "processor": processor,
        }
        print("[JoyCaption] Ready.")
        return model, processor

    def ensure_joycaption_gguf_model():
        cached = state["prompt_backend_cache"].get(PROMPT_GENERATOR_JOYCAPTION_GGUF)
        if cached and cached.get("llm") is not None:
            return cached["llm"]

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "GGUF prompt generation needs huggingface_hub.\n"
                "Run setup again after updating requirements."
            ) from exc

        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except ImportError as exc:
            raise RuntimeError(
                "JoyCaption GGUF support is optional and not installed.\n"
                f"Run: {JOYCAPTION_GGUF_SETUP_HINT}"
            ) from exc

        local_files_only = describe_prompt_generator_source(PROMPT_GENERATOR_JOYCAPTION_GGUF) == "disk cache"
        print(f"[JoyCaption GGUF] Loading {JOYCAPTION_GGUF_REPO_ID} / {JOYCAPTION_GGUF_FILENAME} …")
        model_path = hf_hub_download(
            repo_id=JOYCAPTION_GGUF_REPO_ID,
            filename=JOYCAPTION_GGUF_FILENAME,
            local_files_only=local_files_only,
        )
        mmproj_path = hf_hub_download(
            repo_id=JOYCAPTION_GGUF_REPO_ID,
            filename=JOYCAPTION_GGUF_MMPROJ_FILENAME,
            local_files_only=local_files_only,
        )
        chat_handler = Llava15ChatHandler(
            clip_model_path=mmproj_path,
            verbose=False,
        )
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            chat_format="llava-1-5",
            n_ctx=4096,
            n_gpu_layers=-1 if device == "cuda" else 0,
            verbose=False,
        )
        state["prompt_backend_cache"][PROMPT_GENERATOR_JOYCAPTION_GGUF] = {
            "llm": llm,
            "model_path": model_path,
            "mmproj_path": mmproj_path,
        }
        print("[JoyCaption GGUF] Ready.")
        return llm

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

    def active_targets(main_threshold, aux_threshold):
        left_items, right_items = build_split(
            state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold
        )
        left_names = {os.path.basename(path) for path, _ in left_items}
        right_names = {os.path.basename(path) for path, _ in right_items}
        left_marked = [name for name in state.get("left_marked", []) if name in left_names]
        right_marked = [name for name in state.get("right_marked", []) if name in right_names]
        if left_marked or right_marked:
            return left_marked, right_marked

        preview_fname = state.get("preview_fname")
        if preview_fname in left_names:
            return [preview_fname], []
        if preview_fname in right_names:
            return [], [preview_fname]
        return [], []

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

    def get_preview_image_path():
        preview_fname = state.get("preview_fname")
        if not preview_fname:
            return None, None
        item = state.get("scores", {}).get(preview_fname)
        if not item:
            return None, preview_fname
        return item.get("path"), preview_fname

    def generated_prompt_variants_for(preview_fname, generator_name, create=False):
        if not preview_fname:
            return {}
        preview_bucket = state["generated_prompt_variants"].get(preview_fname)
        if preview_bucket is None:
            if not create:
                return {}
            preview_bucket = {}
            state["generated_prompt_variants"][preview_fname] = preview_bucket
        backend_bucket = preview_bucket.get(generator_name)
        if backend_bucket is None:
            if not create:
                return {}
            backend_bucket = {}
            preview_bucket[generator_name] = backend_bucket
        return backend_bucket

    def select_cached_generated_prompt(generator_name, detail_level, current_generated_prompt):
        detail_level, detail_label, _ = prompt_generator_detail_config(generator_name, detail_level)
        state["prompt_generator"] = generator_name
        state["generated_prompt_detail"] = detail_level
        _, preview_fname = get_preview_image_path()
        if not preview_fname:
            preview_fname = state.get("generated_prompt_source")
        prompt_text = generated_prompt_variants_for(preview_fname, generator_name).get(detail_level)
        if prompt_text:
            state["generated_prompt"] = prompt_text
            state["generated_prompt_source"] = preview_fname
            state["generated_prompt_backend"] = generator_name
            state["generated_prompt_status"] = (
                f"Showing cached {detail_label.lower()} prompt for {preview_fname} via {generator_name}."
            )
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=prompt_text),
            )

        if preview_fname:
            state["generated_prompt_status"] = (
                f"{generator_name} {detail_label.lower()} prompt is not cached for {preview_fname}. Click generate to create it."
            )
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=current_generated_prompt),
            )

        state["generated_prompt_status"] = "Preview an image, then generate a prompt from it."
        return (
            gr.update(value=state["generated_prompt_status"]),
            gr.update(value=current_generated_prompt),
        )

    def run_florence_prompt_variant(image, task_prompt):
        model, processor = ensure_florence_model()
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=FLORENCE_MAX_NEW_TOKENS,
                num_beams=3,
            )

        raw_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        if florence_task_is_pure_text(task_prompt):
            parsed = raw_text
        else:
            parsed = processor.post_process_generation(
                raw_text,
                task=task_prompt,
                image_size=image.size,
            )
        return normalize_generated_prompt(
            extract_florence_caption(parsed, raw_text, task_prompt)
        )

    def run_joycaption_prompt_variant(generator_name, image, user_prompt, detail_level):
        system_prompt = (
            "You are a helpful image captioner. "
            "Describe only concrete visible content and write output that is useful as a text-to-image prompt. "
            "Follow the requested output style exactly, whether it asks for short tags, a compact prompt line, or natural prose. "
            "Do not begin with meta phrases like 'This image shows', 'In this image we can see', or 'You are looking at'."
        )
        max_new_tokens = joycaption_max_new_tokens(detail_level)

        if generator_name == PROMPT_GENERATOR_JOYCAPTION:
            model, processor = ensure_joycaption_model()
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            convo_string = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=[convo_string], images=[image], return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )[0]

            prompt_len = inputs["input_ids"].shape[1]
            text = processor.tokenizer.decode(
                generated_ids[prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return normalize_generated_prompt(
                extract_joycaption_caption(text),
                keep_prose=(detail_level == 3),
            )

        llm = ensure_joycaption_gguf_model()
        data_url = image_to_data_url(image)
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
        )
        try:
            text = response["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected JoyCaption GGUF response shape: {exc}") from exc
        return normalize_generated_prompt(
            extract_joycaption_caption(text),
            keep_prose=(detail_level == 3),
        )

    def generate_prompt_variant(generator_name, image, detail_level):
        _, _, detail_prompt = prompt_generator_detail_config(generator_name, detail_level)
        if generator_name == PROMPT_GENERATOR_FLORENCE:
            return run_florence_prompt_variant(image, detail_prompt)
        return run_joycaption_prompt_variant(generator_name, image, detail_prompt, detail_level)

    def score_imagereward(folder_paths, positive_prompt, negative_prompt, penalty_weight, progress):
        # Optional penalty prompt is implemented as a second pass whose score is subtracted.
        model = ensure_imagereward_model()
        positive_prompt = (positive_prompt or "").strip() or IR_PROMPT
        negative_prompt = (negative_prompt or "").strip()
        penalty_weight = float(penalty_weight)
        state["ir_penalty_weight"] = penalty_weight
        image_signature = get_image_paths_signature(folder_paths)
        proxy_map = {}
        cache_dir = state.get("proxy_cache_dir")
        scoring_paths = list(folder_paths)

        if cache_dir:
            print(f"[ImageReward] Proxy cache dir: {cache_dir}")
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
            print(f"[ImageReward] Proxy prep complete in {cache_dir}: {generated} new, {reused} reused")

        base_scores = {}
        can_reuse_base = (
            state.get("ir_cached_signature") == image_signature
            and state.get("ir_cached_positive_prompt") == positive_prompt
            and state.get("ir_cached_base_scores") is not None
        )
        if can_reuse_base:
            base_scores = dict(state["ir_cached_base_scores"])
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Reusing cached ImageReward positive pass for {len(folder_paths)} images")
        else:
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Scoring {len(folder_paths)} images with ImageReward...")
            for event in iter_imagereward_scores(scoring_paths, model, device, positive_prompt, source_paths=folder_paths):
                if event["type"] == "oom":
                    progress(
                        PROMPTMATCH_PROXY_PROGRESS_SHARE
                        + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                        desc=f"ImageReward OOM, retrying autobatch {event['batch_size']}",
                    )
                    continue
                progress(
                    PROMPTMATCH_PROXY_PROGRESS_SHARE
                    + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                    desc=f"ImageReward {event['done']}/{event['total']} (autobatch {event['batch_size']})",
                )
                base_scores = event["scores"]
            state["ir_cached_signature"] = image_signature
            state["ir_cached_positive_prompt"] = positive_prompt
            state["ir_cached_base_scores"] = dict(base_scores)

        penalty_scores = {}
        if negative_prompt:
            can_reuse_penalty = (
                state.get("ir_cached_signature") == image_signature
                and state.get("ir_cached_negative_prompt") == negative_prompt
                and state.get("ir_cached_penalty_scores") is not None
            )
            if can_reuse_penalty:
                penalty_scores = dict(state["ir_cached_penalty_scores"])
                progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Reusing cached penalty pass for {len(folder_paths)} images")
            else:
                progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Applying penalty prompt to {len(folder_paths)} images...")
                for event in iter_imagereward_scores(scoring_paths, model, device, negative_prompt, source_paths=folder_paths):
                    if event["type"] == "oom":
                        progress(
                            PROMPTMATCH_PROXY_PROGRESS_SHARE
                            + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                            desc=f"Penalty OOM, retrying autobatch {event['batch_size']}",
                        )
                        continue
                    progress(
                        PROMPTMATCH_PROXY_PROGRESS_SHARE
                        + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (event["done"] / max(event["total"], 1))),
                        desc=f"Penalty prompt {event['done']}/{event['total']} (autobatch {event['batch_size']})",
                    )
                    penalty_scores = event["scores"]
                state["ir_cached_signature"] = image_signature
                state["ir_cached_negative_prompt"] = negative_prompt
                state["ir_cached_penalty_scores"] = dict(penalty_scores)
        else:
            state["ir_cached_negative_prompt"] = ""
            state["ir_cached_penalty_scores"] = {}

        penalty_offset = get_imagereward_penalty_offset(
            item["score"] for item in penalty_scores.values()
        )
        wrapped = {}
        for path in folder_paths:
            fname = os.path.basename(path)
            base_item = base_scores.get(fname, {"score": -float("inf"), "path": path})
            penalty_item = penalty_scores.get(fname)
            penalty_value = penalty_item["score"] if penalty_item is not None else None
            final_score = compute_imagereward_final_score(
                base_item["score"],
                penalty_value,
                penalty_weight,
                penalty_offset=penalty_offset,
            )
            wrapped[fname] = {
                "score": float(final_score),
                "base": float(base_item["score"]),
                "penalty": float(penalty_value) if penalty_value is not None else None,
                "path": path,
            }
        return wrapped

    def recompute_imagereward_scores(penalty_weight):
        penalty_weight = float(penalty_weight)
        state["ir_penalty_weight"] = penalty_weight
        if state["method"] != METHOD_IMAGEREWARD or not state["scores"]:
            return False

        changed = False
        penalty_offset = get_imagereward_penalty_offset(
            item.get("penalty") for item in state["scores"].values() if "base" in item
        )
        for item in state["scores"].values():
            if "base" not in item:
                continue
            item["score"] = compute_imagereward_final_score(
                item["base"],
                item.get("penalty"),
                penalty_weight,
                penalty_offset=penalty_offset,
            )
            changed = True
        return changed

    def configure_controls(method):
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
                if backend_name == "openai":
                    source = describe_openai_clip_source(kwargs.get("clip_model"))
                elif backend_name == "openclip":
                    source = describe_openclip_source(kwargs.get("openclip_model"), kwargs.get("openclip_pretrained"))
                else:
                    source = describe_siglip_source(kwargs.get("siglip_model"))
                progress(0, desc=f"Loading PromptMatch model from {source}: {model_label}")
                try:
                    state["backend"] = ModelBackend(device, backend=backend_name, **kwargs)
                except Exception as exc:
                    return empty_result(str(exc), method)
            else:
                progress(0, desc=f"Using loaded PromptMatch model from memory: {model_label}")

            proxy_map = {}
            cache_dir = state.get("proxy_cache_dir")
            if cache_dir:
                print(f"[PromptMatch] Proxy cache dir: {cache_dir}")
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
                print(f"[PromptMatch] Proxy prep complete in {cache_dir}: {generated} new, {reused} reused")

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

        if state["ir_model"] is None:
            progress(0, desc=f"Loading ImageReward model from {describe_imagereward_source()}: ImageReward-v1.0")
        else:
            progress(0, desc="Using loaded ImageReward model from memory: ImageReward-v1.0")

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

    def update_prompt_generator(generator_name, detail_level, current_generated_prompt):
        return select_cached_generated_prompt(generator_name, detail_level, current_generated_prompt)

    def update_generated_prompt_detail(generator_name, detail_level, current_generated_prompt):
        return select_cached_generated_prompt(generator_name, detail_level, current_generated_prompt)

    def generate_prompt_from_preview(generator_name, current_generated_prompt, detail_level, progress=gr.Progress()):
        detail_level, detail_label, _ = prompt_generator_detail_config(generator_name, detail_level)
        state["prompt_generator"] = generator_name
        state["generated_prompt_detail"] = detail_level
        image_path, preview_fname = get_preview_image_path()
        if not image_path or not os.path.isfile(image_path):
            state["generated_prompt_status"] = "Select a preview image first, then generate a prompt."
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=current_generated_prompt),
                gr.update(),
            )

        variants = generated_prompt_variants_for(preview_fname, generator_name, create=True)
        if all(level in variants and variants[level] for level in (1, 2, 3)):
            cached_prompt = variants.get(detail_level)
            state["generated_prompt"] = cached_prompt
            state["generated_prompt_source"] = preview_fname
            state["generated_prompt_backend"] = generator_name
            state["generated_prompt_status"] = (
                f"Reused cached prompt set for {preview_fname} via {generator_name}. Showing {detail_label.lower()} detail."
            )
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=cached_prompt),
                gr.update(value=detail_level),
            )

        backend_cached = state["prompt_backend_cache"].get(generator_name)
        if not backend_cached:
            progress(0, desc=f"Loading {generator_name} from {describe_prompt_generator_source(generator_name)}")
        else:
            progress(0, desc=f"Using loaded {generator_name} backend from memory")

        try:
            with Image.open(image_path) as src_img:
                image = src_img.convert("RGB")

            for idx, level in enumerate((1, 2, 3), start=1):
                _, loop_label, _ = prompt_generator_detail_config(generator_name, level)
                progress(
                    0.2 + (0.7 * ((idx - 1) / 3.0)),
                    desc=f"Generating {loop_label.lower()} prompt from {preview_fname} via {generator_name}",
                )
                variants[level] = generate_prompt_variant(generator_name, image, level)
        except Exception as exc:
            state["generated_prompt_status"] = f"Prompt generation failed: {exc}"
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=current_generated_prompt),
                gr.update(),
            )
        finally:
            if device == "cuda" and prompt_generator_supports_torch_cleanup(generator_name):
                torch.cuda.empty_cache()

        prompt_text = variants.get(detail_level, "")
        if not prompt_text:
            state["generated_prompt_status"] = f"No usable prompt text was produced for {preview_fname}."
            return (
                gr.update(value=state["generated_prompt_status"]),
                gr.update(value=current_generated_prompt),
                gr.update(),
            )

        state["generated_prompt"] = prompt_text
        state["generated_prompt_source"] = preview_fname
        state["generated_prompt_backend"] = generator_name
        ready_count = sum(1 for level in (1, 2, 3) if variants.get(level))
        state["generated_prompt_status"] = (
            f"Generated {ready_count} prompt detail levels for {preview_fname} via {generator_name}. "
            f"Showing {detail_label.lower()} detail."
        )
        return (
            gr.update(value=state["generated_prompt_status"]),
            gr.update(value=prompt_text),
            gr.update(value=detail_level),
        )

    def insert_generated_prompt(method, prompt_text):
        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            state["generated_prompt_status"] = "Generated prompt is empty. Edit or generate a prompt first."
            return gr.update(value=state["generated_prompt_status"]), gr.update(), gr.update()

        state["generated_prompt"] = prompt_text
        target_label = "PromptMatch positive prompt" if method == METHOD_PROMPTMATCH else "ImageReward positive prompt"
        state["generated_prompt_status"] = f"Inserted generated prompt into {target_label}."
        if method == METHOD_PROMPTMATCH:
            return gr.update(value=state["generated_prompt_status"]), gr.update(value=prompt_text), gr.update()
        return gr.update(value=state["generated_prompt_status"]), gr.update(), gr.update(value=prompt_text)

    def update_proxy_display(use_proxy_display, main_threshold, aux_threshold):
        state["use_proxy_display"] = bool(use_proxy_display)
        return current_view(main_threshold, aux_threshold)

    def update_imagereward_penalty_weight(penalty_weight, main_threshold, aux_threshold):
        recomputed = recompute_imagereward_scores(penalty_weight)
        if not recomputed:
            return (
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
            )

        lo, hi, _ = imagereward_slider_range(state["scores"])
        clamped = round(max(lo, min(hi, float(main_threshold))), 3)
        return (
            *current_view(clamped, aux_threshold),
            gr.update(
                minimum=IMAGEREWARD_SLIDER_MIN,
                maximum=IMAGEREWARD_SLIDER_MAX,
                value=clamped,
                label="Primary thresh (>=  SELECTED)",
            ),
            gr.update(value=NEGATIVE_THRESHOLD, visible=False),
        )

    def handle_thumb_action(action, main_threshold, aux_threshold):
        # Custom JS reports both normal preview clicks and shift-click bulk marking.
        if not action:
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        try:
            verb, side, raw_index, _ = str(action).split(":", 3)
            index = int(raw_index)
        except Exception:
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        left_items, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)
        items = left_items if side == "left" else right_items
        if 0 <= index < len(items):
            fname = os.path.basename(items[index][0])
            if verb == "preview":
                state["preview_fname"] = fname
            else:
                marked_key = "left_marked" if side == "left" else "right_marked"
                if fname in state[marked_key]:
                    state[marked_key] = [name for name in state[marked_key] if name != fname]
                else:
                    state[marked_key].append(fname)
        return current_view(main_threshold, aux_threshold)

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

    def fit_threshold_to_targets(main_threshold, aux_threshold):
        left_targets, right_targets = active_targets(main_threshold, aux_threshold)
        targets = left_targets or right_targets
        if not targets or (left_targets and right_targets) or not state["scores"]:
            return (*current_view(main_threshold, aux_threshold), gr.update(), gr.update())

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
            *current_view(new_main, new_aux),
            gr.update(value=new_main),
            gr.update(value=new_aux),
        )

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
    .app-header {
        display:flex;
        align-items:baseline;
        justify-content:space-between;
        gap:12px;
        margin-bottom:8px;
        flex-wrap:wrap;
    }
    h1 { font-family:'Courier New',monospace; letter-spacing:.18em; color:#aadd66; text-transform:uppercase; margin:0; font-size:1.4rem; }
    .header-meta {
        color:#667755;
        font-family:monospace;
        font-size:.78rem;
        white-space:nowrap;
        margin-left:auto;
    }
    .top-controls-shell, .threshold-strip {
        background:#171722;
        border:1px solid #2c2c39;
        border-radius:10px;
        padding:10px;
        margin-bottom:10px;
    }
    .top-controls-shell .gr-group,
    .top-controls-shell .block,
    .threshold-strip .gr-group,
    .threshold-strip .block { gap:8px !important; }
    .top-controls-shell .gr-form,
    .top-controls-shell .gradio-row,
    .threshold-strip .gr-form,
    .threshold-strip .gradio-row { gap:8px !important; }
    .top-controls-shell label span, .threshold-strip label span { font-size:.84rem !important; }
    .method-note { font-family:monospace; color:#8e9d80; background:#11111a; border-radius:8px; padding:5px 8px; margin-bottom:2px !important; }
    .method-note p { margin:0 !important; font-family:monospace !important; font-size:.76rem !important; line-height:1.28 !important; color:#8e9d80 !important; }
    .promptgen-status { background:#121823; border:1px solid #293449; border-radius:8px; padding:5px 8px; }
    .promptgen-status p { margin:0 !important; font-family:monospace !important; font-size:.72rem !important; line-height:1.28 !important; color:#8ec5ff !important; }
    .status-md p { font-family:monospace !important; color:#9fc27c !important; }
    .hist-img img { cursor:crosshair !important; border-radius:6px; }
    .grid-wrap img { object-fit: contain !important; background: #0a0a12; }
    .grid-wrap .caption-label span, .grid-wrap [class*="caption"] { font-family:monospace !important; font-size:.72em !important; color:#8899aa !important; }
    .move-col { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px; padding:10px 6px; background:#0f0f16; border-radius:8px; border:1px solid #252535; }
    .move-col button { width:100%; }
    .sel-info p { font-family:monospace !important; font-size:1.08em !important; font-weight:700 !important; color:#aabb88 !important; text-align:center; word-break:break-all; }
    #hy-folder textarea, #hy-folder input { min-height:40px !important; font-size:.92rem !important; }
    #hy-top-tabs {
        border:1px solid #2b2f3b !important;
        border-radius:8px !important;
        background:#11131a !important;
        padding:8px !important;
    }
    #hy-top-tabs .tab-nav {
        gap:6px !important;
        padding:0 0 8px 0 !important;
        border-bottom:1px solid #252b36 !important;
        margin-bottom:8px !important;
    }
    #hy-top-tabs .tab-nav button {
        font-family:monospace !important;
        font-size:.78rem !important;
        line-height:1.1 !important;
        border-radius:7px !important;
        border:1px solid #313646 !important;
        background:#161924 !important;
        color:#bfc7d4 !important;
        padding:7px 10px !important;
        min-height:0 !important;
    }
    #hy-top-tabs .tab-nav button.selected {
        background:#243147 !important;
        color:#e4efff !important;
        border-color:#5677a8 !important;
    }
    #hy-top-tabs .tabitem { padding:2px 0 0 0 !important; }
    .tab-panel-card {
        background:#131722;
        border:1px solid #262c38;
        border-radius:8px;
        padding:8px;
    }
    .threshold-strip .gradio-row { align-items:end !important; }
    #hy-threshold-panel .wrap-inner { gap:8px !important; }
    .compact-actions { align-items:end !important; }
    .compact-actions button { width:100%; }
    #hy-run-pm, #hy-run-ir, #hy-export, #hy-generate-prompt, #hy-insert-prompt { border-radius:8px !important; }
    #hy-run-pm, #hy-run-pm button, #hy-run-ir, #hy-run-ir button, #hy-export, #hy-export button {
        background:#2f8f45 !important;
        background-image:none !important;
        border:1px solid #58bb73 !important;
        color:#f3fff2 !important;
        font-weight:700 !important;
        box-shadow:0 0 0 1px rgba(25, 55, 30, 0.15) inset !important;
    }
    #hy-run-pm button, #hy-run-ir button, #hy-export button, #hy-generate-prompt button, #hy-insert-prompt button {
        min-height:40px !important;
        border-radius:8px !important;
    }
    #hy-run-pm:hover, #hy-run-pm button:hover, #hy-run-ir:hover, #hy-run-ir button:hover, #hy-export:hover, #hy-export button:hover {
        background:#38a14f !important;
        background-image:none !important;
    }
    #hy-run-pm button:disabled, #hy-run-ir button:disabled, #hy-export button:disabled {
        background:#256d35 !important;
        color:#d8ead8 !important;
    }
    #hy-generate-prompt, #hy-generate-prompt button {
        background:#2b6dc9 !important;
        background-image:none !important;
        border:1px solid #6ea7ff !important;
        color:#f4f9ff !important;
        font-weight:700 !important;
        box-shadow:0 0 0 1px rgba(19, 42, 79, 0.2) inset !important;
    }
    #hy-generate-prompt:hover, #hy-generate-prompt button:hover {
        background:#347ce2 !important;
        background-image:none !important;
    }
    #hy-generate-prompt button:disabled {
        background:#254f86 !important;
        color:#d9e8ff !important;
    }
    #hy-insert-prompt, #hy-insert-prompt button {
        background:#8d5a19 !important;
        background-image:none !important;
        border:1px solid #d5a257 !important;
        color:#fff7ec !important;
        font-weight:700 !important;
        box-shadow:0 0 0 1px rgba(74, 45, 11, 0.2) inset !important;
    }
    #hy-insert-prompt:hover, #hy-insert-prompt button:hover {
        background:#a56b22 !important;
        background-image:none !important;
    }
    #hy-insert-prompt button:disabled {
        background:#704710 !important;
        color:#f4e4ce !important;
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

    with gr.Blocks(title=APP_WINDOW_TITLE) as demo:
        gr.HTML("""
<div class='app-header'>
  <h1>{title}</h1>
  <div class='header-meta'>{tag} &middot; created by vangel</div>
</div>
""".format(title=APP_NAME, tag=APP_GITHUB_TAG))

        thumb_action = gr.Textbox(value="", visible="hidden", elem_id="hy-thumb-action")
        mark_state = gr.Textbox(value='{"left":[],"right":[]}', visible="hidden", elem_id="hy-mark-state")

        with gr.Group(elem_classes=["top-controls-shell"]):
            with gr.Tabs(elem_id="hy-top-tabs"):
                with gr.Tab("Setup"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, min_width=260, elem_classes=["tab-panel-card"]):
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
                        with gr.Column(scale=2, min_width=420, elem_classes=["tab-panel-card"]):
                            folder_input = gr.Textbox(
                                value=source_dir,
                                label="Image folder - paste a path here",
                                lines=1,
                                placeholder=folder_placeholder(),
                                elem_id="hy-folder",
                            )

                with gr.Tab("Scoring + Prompt from preview"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, min_width=420, elem_classes=["tab-panel-card"]):
                            with gr.Group(visible=True) as promptmatch_group:
                                model_dd = gr.Dropdown(choices=MODEL_LABELS, value=label_for_backend(prompt_backend), label="PromptMatch model", elem_id="hy-model")
                                pos_prompt_tb = gr.Textbox(value=SEARCH_PROMPT, label="Positive prompt", lines=1, elem_id="hy-pos")
                                neg_prompt_tb = gr.Textbox(value=NEGATIVE_PROMPT, label="Negative prompt", lines=1, elem_id="hy-neg")
                                promptmatch_run_btn = gr.Button("Run scoring", elem_id="hy-run-pm", variant="primary")

                            with gr.Group(visible=False) as imagereward_group:
                                ir_prompt_tb = gr.Textbox(value=IR_PROMPT, label="ImageReward positive prompt", lines=2, elem_id="hy-ir-pos")
                                ir_negative_prompt_tb = gr.Textbox(
                                    value=DEFAULT_IR_NEGATIVE_PROMPT,
                                    label="Experimental penalty prompt",
                                    lines=1,
                                    placeholder="Optional: undesirable style or mood to subtract",
                                    elem_id="hy-ir-neg",
                                )
                                ir_penalty_weight_tb = gr.Slider(
                                    value=DEFAULT_IR_PENALTY_WEIGHT,
                                    label="Penalty weight",
                                    minimum=0.0,
                                    maximum=4.0,
                                    step=0.1,
                                    elem_id="hy-ir-weight",
                                )
                                imagereward_run_btn = gr.Button("Run scoring", elem_id="hy-run-ir", variant="primary")
                        with gr.Column(scale=1, min_width=420, elem_classes=["tab-panel-card"]):
                            gr.Markdown(
                                "Generate an editable prompt from the current preview image, then reinsert it into the active method.",
                                elem_classes=["method-note"],
                            )
                            with gr.Row(equal_height=True, elem_classes=["compact-actions"]):
                                with gr.Column(scale=3):
                                    prompt_generator_dd = gr.Dropdown(
                                        choices=list(PROMPT_GENERATOR_CHOICES),
                                        value=state["prompt_generator"],
                                        label="Prompt generator",
                                        elem_id="hy-prompt-generator",
                                    )
                                with gr.Column(scale=2, min_width=170):
                                    generate_prompt_btn = gr.Button("Generate prompt from preview", elem_id="hy-generate-prompt")
                            promptgen_status_md = gr.Markdown(
                                state["generated_prompt_status"],
                                elem_classes=["promptgen-status"],
                                elem_id="hy-promptgen-status",
                            )
                            generated_prompt_tb = gr.Textbox(
                                value=state["generated_prompt"],
                                label="Generated prompt",
                                lines=3,
                                placeholder="Preview an image, then generate an editable prompt here.",
                                elem_id="hy-generated-prompt",
                            )
                            with gr.Row(equal_height=True, elem_classes=["compact-actions"]):
                                with gr.Column(scale=3):
                                    generated_prompt_detail_slider = gr.Slider(
                                        minimum=1,
                                        maximum=3,
                                        value=DEFAULT_GENERATED_PROMPT_DETAIL,
                                        step=1,
                                        label="Prompt detail",
                                        elem_id="hy-generated-prompt-detail",
                                    )
                                with gr.Column(scale=2, min_width=170):
                                    insert_prompt_btn = gr.Button("Insert into active prompt", elem_id="hy-insert-prompt")

                with gr.Tab("Export"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=0, min_width=180, elem_classes=["tab-panel-card"]):
                            export_btn = gr.Button("Export folders", elem_id="hy-export", variant="primary")
                        with gr.Column(scale=1, min_width=420, elem_classes=["tab-panel-card"]):
                            export_tb = gr.Textbox(label="Export result", lines=3, interactive=False)

        with gr.Group(elem_id="hy-threshold-panel", elem_classes=["threshold-strip"]):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=280):
                    hist_plot = gr.Image(value=None, show_label=False, interactive=False, elem_classes=["hist-img"], elem_id="hy-hist")
                with gr.Column(scale=5, min_width=520):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, min_width=180):
                            main_slider = gr.Slider(minimum=-1.0, maximum=1.0, value=0.14, step=0.001, label="Primary thresh (>=  SELECTED)", elem_id="hy-main-slider")
                        with gr.Column(scale=1, min_width=180):
                            aux_slider = gr.Slider(minimum=-1.0, maximum=1.0, value=NEGATIVE_THRESHOLD, step=0.001, label="Neg threshold (< -> passes)", elem_id="hy-aux-slider")
                        with gr.Column(scale=1, min_width=180):
                            percentile_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Or keep top N%", elem_id="hy-percentile")
                    proxy_display_cb = gr.Checkbox(value=True, label="Use proxies for gallery display", elem_id="hy-use-proxy-display")
                    status_md = gr.Markdown("", elem_classes=["status-md"])

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
                fit_threshold_btn = gr.Button("Fit thresh", elem_id="hy-fit-threshold")
                move_left_btn = gr.Button("← Move", elem_id="hy-move-left")
                clear_status_btn = gr.Button("Clear status", elem_id="hy-clear-status")
            with gr.Column(scale=1, elem_classes=["gallery-side"]):
                right_gallery = gr.Gallery(show_label=False, columns=5, height="80vh", object_fit="contain", preview=True, allow_preview=True, elem_classes=["grid-wrap"], elem_id="hy-right-gallery")

        method_dd.change(
            fn=configure_controls,
            inputs=[method_dd],
            outputs=[promptmatch_group, imagereward_group, main_slider, aux_slider, method_note],
        )

        promptmatch_run_btn.click(
            fn=score_folder,
            inputs=[method_dd, folder_input, model_dd, pos_prompt_tb, neg_prompt_tb, ir_prompt_tb, ir_negative_prompt_tb, ir_penalty_weight_tb],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider],
        )
        imagereward_run_btn.click(
            fn=score_folder,
            inputs=[method_dd, folder_input, model_dd, pos_prompt_tb, neg_prompt_tb, ir_prompt_tb, ir_negative_prompt_tb, ir_penalty_weight_tb],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider],
        )
        prompt_generator_dd.change(
            fn=update_prompt_generator,
            inputs=[prompt_generator_dd, generated_prompt_detail_slider, generated_prompt_tb],
            outputs=[promptgen_status_md, generated_prompt_tb],
        )
        generate_prompt_btn.click(
            fn=generate_prompt_from_preview,
            inputs=[prompt_generator_dd, generated_prompt_tb, generated_prompt_detail_slider],
            outputs=[promptgen_status_md, generated_prompt_tb, generated_prompt_detail_slider],
        )
        generated_prompt_detail_slider.change(
            fn=update_generated_prompt_detail,
            inputs=[prompt_generator_dd, generated_prompt_detail_slider, generated_prompt_tb],
            outputs=[promptgen_status_md, generated_prompt_tb],
        )
        insert_prompt_btn.click(
            fn=insert_generated_prompt,
            inputs=[method_dd, generated_prompt_tb],
            outputs=[promptgen_status_md, pos_prompt_tb, ir_prompt_tb],
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
        ir_penalty_weight_tb.change(
            fn=update_imagereward_penalty_weight,
            inputs=[ir_penalty_weight_tb, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider],
        )
        thumb_action.change(fn=handle_thumb_action, inputs=[thumb_action, main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state])
        move_right_btn.click(fn=move_right, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state])
        fit_threshold_btn.click(fn=fit_threshold_to_targets, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider])
        move_left_btn.click(fn=move_left, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state])
        clear_status_btn.click(fn=clear_status, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state])
        hist_plot.select(fn=on_hist_click, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider])
        export_btn.click(fn=export_files, inputs=[main_slider, aux_slider], outputs=[export_tb])

    return demo, css, tooltip_head(tooltips)


if __name__ == "__main__":
    app, css, head = create_app()
    port = resolve_server_port(7862, "HYBRIDSELECTOR_PORT")
    print(f"Launching {APP_DISPLAY_NAME} {APP_GITHUB_TAG} on http://localhost:{port} ...")
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
