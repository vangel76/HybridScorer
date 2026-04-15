"""
HybridScorer — image triage

Single-file Gradio app (`Hybrid-Scorer.py`)

Modes:
- PromptMatch → CLIP-based text-image similarity
- ImageReward → aesthetic/preference scoring
- Similarity → image-image search from preview
- SamePerson → face search from preview
- LM Search → PromptMatch shortlist + local vision-language rerank

Core:
- manual bucket overrides
- preview-driven actions
- prompt generation from preview
- cached scoring + proxy images
- export to structured folders

See:
- README.md
- docs/architecture.md
- docs/behavior-notes.md

Important:
- callback return signatures must match exactly
- UI behavior may live in Python and injected JS
"""

import base64
import gc
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
import threading
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from hashlib import sha256
from importlib import metadata
from importlib import import_module
from importlib.machinery import ModuleSpec

import gradio as gr
import numpy as np
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
SETUP_SCRIPT_HINT = "setup_update-windows.bat" if os.name == "nt" else "./setup_update-linux.sh"

try:
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version
except Exception:
    SpecifierSet = None
    InvalidVersion = Exception
    Version = None

    def canonicalize_name(name):
        return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()

CACHE_MODE_PROJECT = "project"
CACHE_MODE_SYSTEM = "system"
ENV_CACHE_MODE = "HYBRIDSCORER_CACHE_MODE"
PROJECT_HF_CACHE_DIR = os.path.join("models", "huggingface")
PROJECT_CLIP_CACHE_DIR = os.path.join("models", "clip")
PROJECT_IMAGEREWARD_CACHE_DIR = os.path.join("models", "ImageReward")
PROJECT_INSIGHTFACE_CACHE_DIR = os.path.join("models", "insightface")
PROJECT_PROMPTMATCH_PROXY_CACHE_DIR = "cache"
STARTUP_EXTRA_REQUIREMENTS = (
    "image-reward==1.5",
    "llama-cpp-python>=0.3.7",
)


def default_cache_mode():
    return CACHE_MODE_PROJECT if os.name == "nt" else CACHE_MODE_SYSTEM


def _system_proxy_root():
    # On Linux, prefer /dev/shm (RAM-backed tmpfs) for proxy storage so resized
    # thumbnails never touch disk.  Fall back to the normal temp dir if /dev/shm
    # is absent (some containers) or not writable.
    shm = "/dev/shm"
    if os.path.isdir(shm) and os.access(shm, os.W_OK):
        return shm
    return tempfile.gettempdir()


@lru_cache(maxsize=1)
def get_cache_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fallback_mode = default_cache_mode()
    raw_mode = (os.getenv(ENV_CACHE_MODE) or fallback_mode).strip().lower()
    mode = raw_mode if raw_mode in {CACHE_MODE_PROJECT, CACHE_MODE_SYSTEM} else fallback_mode

    if mode == CACHE_MODE_PROJECT:
        return {
            "mode": mode,
            "script_dir": script_dir,
            "huggingface_dir": os.path.join(script_dir, PROJECT_HF_CACHE_DIR),
            "clip_dir": os.path.join(script_dir, PROJECT_CLIP_CACHE_DIR),
            "imagereward_dir": os.path.join(script_dir, PROJECT_IMAGEREWARD_CACHE_DIR),
            "insightface_dir": os.path.join(script_dir, PROJECT_INSIGHTFACE_CACHE_DIR),
            "proxy_root": os.path.join(script_dir, PROJECT_PROMPTMATCH_PROXY_CACHE_DIR),
        }

    home_cache_root = os.path.expanduser("~/.cache")
    return {
        "mode": mode,
        "script_dir": script_dir,
        "huggingface_dir": None,
        "clip_dir": os.path.join(home_cache_root, "clip"),
        "imagereward_dir": os.path.join(home_cache_root, "ImageReward"),
        "insightface_dir": os.path.join(home_cache_root, "insightface"),
        "proxy_root": os.path.join(_system_proxy_root(), PROMPTMATCH_PROXY_CACHE_ROOT),
    }

# High-level app modes and default thresholds/prompts.
METHOD_PROMPTMATCH = "PromptMatch"
METHOD_IMAGEREWARD = "ImageReward"
METHOD_LLMSEARCH = "LM Search"
METHOD_SIMILARITY = "Similarity"
METHOD_SAMEPERSON = "SamePerson"
METHOD_TAGMATCH = "TagMatch"
DEFAULT_IR_NEGATIVE_PROMPT = ""
DEFAULT_IR_PENALTY_WEIGHT = 1.0
SIMILARITY_TOPN_DEFAULT = 5
SIMILARITY_TOPN_SLIDER_MAX = 50
SIMILARITY_AUTO_TOPN_SCAN_LIMIT = 30
SIMILARITY_AUTO_TOPN_MIN = 3
SIMILARITY_AUTO_KNEE_MIN_DISTANCE = 0.06
PROMPTMATCH_SLIDER_MIN = -1.0
PROMPTMATCH_SLIDER_MAX = 1.0
IMAGEREWARD_SLIDER_MIN = -5.0
IMAGEREWARD_SLIDER_MAX = 5.0
TAGMATCH_SLIDER_MIN = 0.0
TAGMATCH_SLIDER_MAX = 100.0
TAGMATCH_DEFAULT_THRESHOLD = 20.0
HIST_HEIGHT_SCALE = 0.7
PROMPT_GENERATOR_FLORENCE = "Florence-2"
PROMPT_GENERATOR_JOYCAPTION = "JoyCaption Beta One"
PROMPT_GENERATOR_JOYCAPTION_GGUF = "JoyCaption Beta One GGUF (Q4_K_M)"
PROMPT_GENERATOR_WD_TAGS = "WD Tags (ONNX)"
PROMPT_GENERATOR_CHOICES = (
    PROMPT_GENERATOR_FLORENCE,
    PROMPT_GENERATOR_JOYCAPTION,
    PROMPT_GENERATOR_JOYCAPTION_GGUF,
)
# Superset including WD Tags — used for the prompt generator dropdown only.
# LLM Search backend dropdown uses PROMPT_GENERATOR_CHOICES (no WD Tags).
PROMPT_GENERATOR_ALL_CHOICES = (
    PROMPT_GENERATOR_FLORENCE,
    PROMPT_GENERATOR_JOYCAPTION,
    PROMPT_GENERATOR_JOYCAPTION_GGUF,
    PROMPT_GENERATOR_WD_TAGS,
)
DEFAULT_PROMPT_GENERATOR = PROMPT_GENERATOR_FLORENCE
FLORENCE_MODEL_ID = "florence-community/Florence-2-base"
FLORENCE_MAX_NEW_TOKENS = 256
JOYCAPTION_MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
JOYCAPTION_MAX_NEW_TOKENS = 320
JOYCAPTION_GGUF_REPO_ID = "cinnabrad/llama-joycaption-beta-one-hf-llava-mmproj-gguf"
JOYCAPTION_GGUF_FILENAME = "Llama-Joycaption-Beta-One-Hf-Llava-Q4_K_M.gguf"
JOYCAPTION_GGUF_MMPROJ_FILENAME = "llama-joycaption-beta-one-llava-mmproj-model-f16.gguf"
JOYCAPTION_GGUF_SETUP_HINT = "./setup_update-linux.sh"
DEFAULT_GENERATED_PROMPT_DETAIL = 2
DEFAULT_LLMSEARCH_BACKEND = PROMPT_GENERATOR_JOYCAPTION_GGUF
LLMSEARCH_SCORING_MODE_NUMERIC_V1 = "joycaption_numeric_v9"
LLMSEARCH_JOYCAPTION_CAPTION_TYPE = "descriptive"
LLMSEARCH_JOYCAPTION_CAPTION_LENGTH = "very short"
LLMSEARCH_JOYCAPTION_MAX_NEW_TOKENS = 128
LLMSEARCH_JOYCAPTION_TEMPERATURE = 0.6
LLMSEARCH_JOYCAPTION_TOP_P = 0.9
LLMSEARCH_JOYCAPTION_TOP_K = 0
LLMSEARCH_JOYCAPTION_SYSTEM_PROMPT = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."
LLMSEARCH_JOYCAPTION_HF_BATCH_SIZE = 4
LLMSEARCH_DEFAULT_PROMPT = "woman in red dress, cinematic portrait, soft warm light"
LLMSEARCH_SHORTLIST_DEFAULT = 32
LLMSEARCH_SHORTLIST_MIN = 8
LLMSEARCH_SHORTLIST_MAX = 512
TAGMATCH_WD_REPO_ID = "SmilingWolf/wd-eva02-large-tagger-v3"
TAGMATCH_WD_MODEL_FILE = "model.onnx"
TAGMATCH_WD_TAGS_FILE = "selected_tags.csv"
TAGMATCH_WD_IMAGE_SIZE = 448
TAGMATCH_WD_MIN_CACHE_PROB = 0.02
TAGMATCH_WD_BATCH_SIZE = 32
TAGMATCH_DEFAULT_TAGS = (
    "bad_anatomy, bad_hands, bad_feet, bad_proportions, deformed, extra_arms, extra_faces, extra_mouth, missing_limb, multiple_legs, multiple_heads, oversized_limbs, wrong_foot, artistic_error, glitch, blob, disembodied_limb"
)
FACE_MODEL_PACK = "buffalo_l"
FACE_MODEL_LABEL = f"InsightFace {FACE_MODEL_PACK}"
FACE_DET_SIZE = (640, 640)
FACE_EMBEDDING_ABSOLUTE_MAX_WORKERS = 16
FACE_EMBEDDING_MIN_FREE_VRAM_GB = 2.5
FACE_EMBEDDING_PER_WORKER_VRAM_GB = 0.6
PROMPTMATCH_HOST_MAX_WORKERS = 16

SEARCH_PROMPT = "ginger woman"
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
IMAGEREWARD_BASE_BATCH_SIZE = 48
IMAGEREWARD_MAX_BATCH_SIZE = 256
IMAGEREWARD_BATCH_AGGRESSION = 1.5
IMAGEREWARD_MIN_FREE_VRAM_GB = 2.0
PROMPTMATCH_BASE_BATCH_SIZE = 64
PROMPTMATCH_MAX_BATCH_SIZE = 384
PROMPTMATCH_BATCH_AGGRESSION = 1.75
PROMPTMATCH_MIN_FREE_VRAM_GB = 3.0
TAGMATCH_BASE_BATCH_SIZE = 16
TAGMATCH_MAX_BATCH_SIZE = 64
TAGMATCH_BATCH_AGGRESSION = 1.2
TAGMATCH_MIN_FREE_VRAM_GB = 2.5
PROMPTMATCH_PROXY_MAX_EDGE = 1024
PROMPTMATCH_PROXY_CACHE_ROOT = "HybridScorerPromptMatchProxyCache"
PROMPTMATCH_PROXY_PROGRESS_SHARE = 0.25
PROMPTMATCH_TORCH_THREAD_CAP = 8
EXPLICIT_PROMPT_WEIGHT_RE = re.compile(r"\(([^()]*?)\s*:\s*([0-9]*\.?[0-9]+)\)")


def normalize_prompt_text(text):
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    return text.strip(" ,")


def parse_requirement_entry(raw_line):
    line = (raw_line or "").strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("git+"):
        if "openai/CLIP.git" in line:
            return {
                "raw": line,
                "name": "clip",
                "spec": None,
            }
        return None

    match = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*(.*)$", line)
    if not match:
        return None
    name = match.group(1).strip()
    remainder = (match.group(2) or "").strip()
    marker = None
    if ";" in remainder:
        remainder, marker = remainder.split(";", 1)
        marker = marker.strip()
    return {
        "raw": line,
        "name": name,
        "spec": remainder.strip() or None,
        "marker": marker,
    }


def load_startup_requirements():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirement_sources = [
        os.path.join(script_dir, "requirements.txt"),
        os.path.join(script_dir, "requirements-gguf.txt"),
    ]
    parsed = []
    seen = set()
    for path in requirement_sources:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            continue
        for line in lines:
            entry = parse_requirement_entry(line)
            if not entry:
                continue
            key = canonicalize_name(entry["name"])
            if key in seen:
                continue
            seen.add(key)
            parsed.append(entry)
    for raw in STARTUP_EXTRA_REQUIREMENTS:
        entry = parse_requirement_entry(raw)
        if not entry:
            continue
        key = canonicalize_name(entry["name"])
        if key in seen:
            continue
        seen.add(key)
        parsed.append(entry)
    return parsed


def runtime_requirement_issues():
    issues = []
    installed_versions = {}
    try:
        for dist in metadata.distributions():
            dist_name = dist.metadata.get("Name") or dist.metadata.get("Summary") or ""
            if not dist_name:
                continue
            installed_versions[canonicalize_name(dist_name)] = dist.version
    except Exception:
        installed_versions = {}

    for requirement in load_startup_requirements():
        name = requirement["name"]
        installed_version = installed_versions.get(canonicalize_name(name))
        if installed_version is None:
            issues.append(f"{name}: not installed")
            continue

        spec = requirement.get("spec")
        if not spec or SpecifierSet is None or Version is None:
            continue
        try:
            spec_set = SpecifierSet(spec)
            version = Version(installed_version)
        except (InvalidVersion, Exception):
            continue
        if version not in spec_set:
            issues.append(f"{name}: installed {installed_version}, needs {spec}")
    return issues


def render_promptmatch_segments(segments, skip_weighted_index=None):
    rendered = []
    for index, segment in enumerate(segments):
        if segment["kind"] == "weighted" and index == skip_weighted_index:
            continue
        rendered.append(segment["text"])
    return normalize_prompt_text("".join(rendered))


def parse_promptmatch_weighted_prompt(prompt):
    prompt = prompt or ""
    segments = []
    weighted_fragments = []
    last_end = 0

    for match in EXPLICIT_PROMPT_WEIGHT_RE.finditer(prompt):
        start, end = match.span()
        if start > last_end:
            segments.append({"kind": "text", "text": prompt[last_end:start]})

        fragment_text = normalize_prompt_text(match.group(1))
        try:
            fragment_weight = float(match.group(2))
        except ValueError:
            fragment_text = ""
            fragment_weight = 0.0

        if fragment_text and fragment_weight > 0:
            segment_index = len(segments)
            segments.append({"kind": "weighted", "text": fragment_text, "weight": fragment_weight})
            weighted_fragments.append(
                {
                    "segment_index": segment_index,
                    "text": fragment_text,
                    "weight": fragment_weight,
                }
            )
        else:
            segments.append({"kind": "text", "text": match.group(0)})
        last_end = end

    if last_end < len(prompt):
        segments.append({"kind": "text", "text": prompt[last_end:]})

    return render_promptmatch_segments(segments), weighted_fragments, segments


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
    cache_dir = get_cache_config()["huggingface_dir"]
    try:
        cached = try_to_load_from_cache(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    except Exception:
        return False
    if not isinstance(cached, (str, bytes, os.PathLike)):
        return False
    return os.path.isfile(cached)


def huggingface_repo_cached(repo_id, filenames=None, suffixes=None):
    cache_dir = get_cache_config()["huggingface_dir"]
    if cache_dir is None:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = HF_HUB_CACHE
        except Exception:
            cache_dir = os.path.join(os.path.expanduser("~/.cache"), "huggingface", "hub")
    repo_dir = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}")
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return False
    exact_names = {name for name in (filenames or []) if name}
    suffix_list = tuple(suffixes or ())
    for root, _, files in os.walk(snapshots_dir):
        for name in files:
            if exact_names and name in exact_names:
                return True
            if suffix_list and name.endswith(suffix_list):
                return True
    return False


def describe_openai_clip_source(model_name):
    try:
        import clip as _clip
    except Exception:
        return "network or disk cache"
    url = _clip._MODELS.get(model_name)
    if not url:
        return "network or disk cache"
    filename = os.path.basename(url)
    cache_path = os.path.join(get_cache_config()["clip_dir"], filename)
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
        if model_id:
            if filename and huggingface_file_cached(model_id, filename):
                return "disk cache"
            if huggingface_repo_cached(
                model_id,
                filenames=[filename],
                suffixes=(".bin", ".pt", ".pth", ".safetensors"),
            ):
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
    cache_root = get_cache_config()["imagereward_dir"]
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


def describe_insightface_source():
    model_dir = os.path.join(get_cache_config()["insightface_dir"], "models", FACE_MODEL_PACK)
    if not os.path.isdir(model_dir):
        return "network download"
    for name in os.listdir(model_dir):
        if name.lower().endswith(".onnx"):
            return "disk cache"
    return "network download"


def face_embedding_worker_count(total_images):
    if total_images <= 1:
        return 1

    cpu_cap = os.cpu_count() or 1
    hard_cap = max(1, min(FACE_EMBEDDING_ABSOLUTE_MAX_WORKERS, cpu_cap, total_images))

    if not torch.cuda.is_available():
        return 1

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError:
        return max(1, min(2, hard_cap))

    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    # FaceAnalysis is loaded once per worker thread, so worker count needs to
    # scale with available VRAM instead of a fixed magic number.
    if free_gb <= FACE_EMBEDDING_MIN_FREE_VRAM_GB:
        return 1

    extra_budget_gb = max(0.0, free_gb - FACE_EMBEDDING_MIN_FREE_VRAM_GB)
    vram_cap = 1 + int(extra_budget_gb / FACE_EMBEDDING_PER_WORKER_VRAM_GB)

    # Keep smaller cards safe, but let larger cards actually use the GPU.
    if total_gb < 9:
        vram_cap = min(vram_cap, 4)
    elif total_gb < 12:
        vram_cap = min(vram_cap, 6)
    elif total_gb < 16:
        vram_cap = min(vram_cap, 8)
    elif total_gb < 24:
        vram_cap = min(vram_cap, 12)
    else:
        vram_cap = min(vram_cap, FACE_EMBEDDING_ABSOLUTE_MAX_WORKERS)

    # If there is obviously abundant free memory, allow a little extra headroom
    # beyond the simple per-worker estimate.
    if total_gb >= 24 and free_gb >= 16:
        vram_cap = max(vram_cap, min(FACE_EMBEDDING_ABSOLUTE_MAX_WORKERS, 12))
    if total_gb >= 32 and free_gb >= 20:
        vram_cap = max(vram_cap, min(FACE_EMBEDDING_ABSOLUTE_MAX_WORKERS, 14))

    return max(1, min(hard_cap, vram_cap))


def promptmatch_host_worker_count(total_items):
    if total_items <= 1:
        return 1
    cpu_cap = os.cpu_count() or 1
    return max(1, min(PROMPTMATCH_HOST_MAX_WORKERS, cpu_cap, total_items))


def configure_torch_cpu_threads():
    cpu_cap = os.cpu_count() or 1
    desired = max(1, min(PROMPTMATCH_TORCH_THREAD_CAP, cpu_cap))
    try:
        current = torch.get_num_threads()
    except Exception:
        current = None
    try:
        if current is None or current != desired:
            torch.set_num_threads(desired)
    except Exception:
        pass


def load_promptmatch_rgb_images(valid_items):
    if not valid_items:
        return [], []

    def _load_one(item):
        original_path, scoring_path = item
        with Image.open(scoring_path) as src_img:
            rgb = src_img.convert("RGB")
            rgb.load()
        return item, rgb, None

    loaded_items = []
    pil_imgs = []
    max_workers = promptmatch_host_worker_count(len(valid_items))
    if max_workers <= 1:
        results = [_load_one(item) for item in valid_items]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_load_one, valid_items))

    for item, image, exc in results:
        if exc is not None:
            raise exc
        loaded_items.append(item)
        pil_imgs.append(image)
    return loaded_items, pil_imgs


def promptmatch_timing_ms(start_time):
    return (time.perf_counter() - start_time) * 1000.0


def current_free_vram_gb():
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError:
        return None
    return free_bytes / (1024 ** 3), total_bytes / (1024 ** 3)


def promptmatch_log_batch_timing(prefix, batch_start, batch_end, total, timings):
    parts = []
    for key, value in timings.items():
        if value is None:
            continue
        if key in {"free_vram_gb"}:
            parts.append(f"{key}={value:.1f}GB")
        else:
            parts.append(f"{key}={value:.1f}ms")
    if parts:
        print(f"[PromptMatch] {prefix} {batch_start}-{batch_end}/{total}  " + "  ".join(parts))


def imagereward_log_batch_timing(prefix, batch_start, batch_end, total, timings):
    parts = []
    for key, value in timings.items():
        if value is None:
            continue
        if key in {"free_vram_gb"}:
            parts.append(f"{key}={value:.1f}GB")
        else:
            parts.append(f"{key}={value:.1f}ms")
    if parts:
        print(f"[ImageReward] {prefix} {batch_start}-{batch_end}/{total}  " + "  ".join(parts))


def prepare_promptmatch_loaded_batch(batch, proxy_resolver=None):
    load_started = time.perf_counter()
    valid_items = []
    failed_paths = []

    for original_path in batch:
        try:
            scoring_path = proxy_resolver(original_path) if proxy_resolver is not None else original_path
            valid_items.append((original_path, scoring_path))
        except Exception as exc:
            print(f"  [WARN] {original_path}: {exc}")
            failed_paths.append(original_path)

    if not valid_items:
        return [], [], failed_paths, {"load": promptmatch_timing_ms(load_started)}

    try:
        loaded_items, pil_imgs = load_promptmatch_rgb_images(valid_items)
        return loaded_items, pil_imgs, failed_paths, {"load": promptmatch_timing_ms(load_started)}
    except Exception as exc:
        print(f"  [WARN] batch image-load error, retrying individually: {exc}")

    loaded_items = []
    pil_imgs = []
    for original_path, scoring_path in valid_items:
        try:
            single_loaded, single_imgs = load_promptmatch_rgb_images([(original_path, scoring_path)])
            loaded_items.extend(single_loaded)
            pil_imgs.extend(single_imgs)
        except Exception as single_exc:
            print(f"  [WARN] {original_path}: {single_exc}")
            failed_paths.append(original_path)

    return loaded_items, pil_imgs, failed_paths, {"load": promptmatch_timing_ms(load_started)}


def prepare_imagereward_loaded_batch(batch_paths, batch_source_paths, model):
    load_started = time.perf_counter()

    def _load_one(item):
        scoring_path, source_path = item
        with Image.open(scoring_path) as src_img:
            rgb = src_img.convert("RGB")
            rgb.load()
        return scoring_path, source_path, rgb

    pairs = list(zip(batch_paths, batch_source_paths))
    workers = promptmatch_host_worker_count(len(pairs))
    loaded_images = []
    failed_entries = []

    if workers <= 1:
        for scoring_path, source_path in pairs:
            try:
                _, _, rgb = _load_one((scoring_path, source_path))
                loaded_images.append((scoring_path, source_path, rgb))
            except Exception as exc:
                failed_entries.append((scoring_path, source_path, exc))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_load_one, item): item for item in pairs}
            for future in as_completed(future_map):
                scoring_path, source_path = future_map[future]
                try:
                    _, _, rgb = future.result()
                    loaded_images.append((scoring_path, source_path, rgb))
                except Exception as exc:
                    failed_entries.append((scoring_path, source_path, exc))

    preprocess_started = time.perf_counter()

    def _preprocess_one(entry):
        scoring_path, source_path, rgb = entry
        tensor = model.preprocess(rgb)
        return scoring_path, source_path, tensor

    preprocessed = []
    if loaded_images:
        preprocess_workers = promptmatch_host_worker_count(len(loaded_images))
        if preprocess_workers <= 1:
            for entry in loaded_images:
                preprocessed.append(_preprocess_one(entry))
        else:
            with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                preprocessed = list(executor.map(_preprocess_one, loaded_images))

    timings = {
        "load": promptmatch_timing_ms(load_started),
        "preprocess": promptmatch_timing_ms(preprocess_started),
    }
    return preprocessed, failed_entries, timings


def describe_prompt_generator_source(generator_name):
    if generator_name == PROMPT_GENERATOR_FLORENCE:
        return describe_florence_source()
    if generator_name == PROMPT_GENERATOR_JOYCAPTION:
        return describe_huggingface_transformers_source(JOYCAPTION_MODEL_ID)
    if generator_name == PROMPT_GENERATOR_JOYCAPTION_GGUF:
        return describe_joycaption_gguf_source()
    if generator_name == PROMPT_GENERATOR_WD_TAGS:
        return "WD tagger ONNX (onnxruntime)"
    return "network or disk cache"


def prompt_generator_supports_torch_cleanup(generator_name):
    return generator_name in {PROMPT_GENERATOR_FLORENCE, PROMPT_GENERATOR_JOYCAPTION}


def llmsearch_backend_choices():
    return list(PROMPT_GENERATOR_CHOICES)


def llmsearch_backend_label(generator_name):
    return generator_name


def describe_llmsearch_backend_source(generator_name):
    return describe_prompt_generator_source(generator_name)


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
    return os.path.join(get_cache_config()["proxy_root"], digest)


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


def ensure_promptmatch_proxy(original_path, cache_dir, proxy_path=None, max_edge=PROMPTMATCH_PROXY_MAX_EDGE):
    os.makedirs(cache_dir, exist_ok=True)
    proxy_path = proxy_path or build_promptmatch_proxy_path(original_path, cache_dir, max_edge=max_edge)
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
            proxy_path = ensure_promptmatch_proxy(original_path, cache_dir, proxy_path=proxy_path)
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


def get_auto_batch_size(device, backend=None, reference_vram_gb=32, mode=None):
    # Scale batch size from currently free VRAM so large models do not OOM instantly.
    if device != "cuda" or not torch.cuda.is_available():
        return DEFAULT_BATCH_SIZE

    base_batch_size = DEFAULT_BATCH_SIZE
    max_batch_size = MAX_BATCH_SIZE
    aggression = 1.0
    reserve_gb = 2.0

    if mode == "imagereward":
        base_batch_size = IMAGEREWARD_BASE_BATCH_SIZE
        max_batch_size = IMAGEREWARD_MAX_BATCH_SIZE
        aggression = IMAGEREWARD_BATCH_AGGRESSION
        reference_vram_gb = 16
        reserve_gb = IMAGEREWARD_MIN_FREE_VRAM_GB
    elif mode == "tagmatch":
        base_batch_size = TAGMATCH_BASE_BATCH_SIZE
        max_batch_size = TAGMATCH_MAX_BATCH_SIZE
        aggression = TAGMATCH_BATCH_AGGRESSION
        reference_vram_gb = 14  # ViT-Large 448px — heavier per image than CLIP
        reserve_gb = TAGMATCH_MIN_FREE_VRAM_GB
    elif backend is not None:
        base_batch_size = PROMPTMATCH_BASE_BATCH_SIZE
        max_batch_size = PROMPTMATCH_MAX_BATCH_SIZE
        aggression = PROMPTMATCH_BATCH_AGGRESSION
        reference_vram_gb = 20
        reserve_gb = PROMPTMATCH_MIN_FREE_VRAM_GB
        if backend.backend == "openclip":
            model_name = backend._openclip_model.lower()
            if "bigg" in model_name or "xxlarge" in model_name:
                reference_vram_gb = 32
                max_batch_size = min(max_batch_size, 384)
            elif "large_d_320" in model_name:
                reference_vram_gb = 28
                max_batch_size = min(max_batch_size, 384)
            elif "vit-h" in model_name:
                reference_vram_gb = 24
                max_batch_size = min(max_batch_size, 448)
            elif "base_w" in model_name:
                reference_vram_gb = 20
                max_batch_size = min(max_batch_size, 288)
            elif "vit-l" in model_name:
                max_batch_size = min(max_batch_size, 288)
        elif backend.backend == "openai" and "@336px" in backend._clip_model:
            reference_vram_gb = 24
            max_batch_size = min(max_batch_size, 384)
        elif backend.backend == "siglip":
            if "large" in backend._siglip_model:
                reference_vram_gb = 24
                max_batch_size = min(max_batch_size, 288)
            elif "base" in backend._siglip_model:
                reference_vram_gb = 16
                max_batch_size = min(max_batch_size, 256)
            else:
                max_batch_size = min(max_batch_size, 256)

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError:
        return DEFAULT_BATCH_SIZE

    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    effective_free_gb = max(0.0, free_gb - reserve_gb)

    if mode == "imagereward":
        if total_gb < 9:
            base_batch_size = min(base_batch_size, 24)
            max_batch_size = min(max_batch_size, 64)
            reference_vram_gb = max(reference_vram_gb, 10)
            aggression *= 0.9
        elif total_gb < 12:
            base_batch_size = min(base_batch_size, 32)
            max_batch_size = min(max_batch_size, 96)
            reference_vram_gb = max(reference_vram_gb, 12)
        elif total_gb < 16:
            base_batch_size = min(base_batch_size, 40)
            max_batch_size = min(max_batch_size, 128)
            reference_vram_gb = max(reference_vram_gb, 14)
        elif total_gb < 24:
            max_batch_size = min(max_batch_size, 192)

        if free_gb >= 24:
            aggression *= 1.2
        elif free_gb >= 16:
            aggression *= 1.1
        elif free_gb >= 12:
            aggression *= 1.05

        if total_gb >= 16 and free_gb >= 12:
            base_batch_size = max(base_batch_size, 64)
        if total_gb >= 24 and free_gb >= 16:
            base_batch_size = max(base_batch_size, 80)
        if total_gb >= 32 and free_gb >= 24:
            max_batch_size = min(max_batch_size, 256)
    elif backend is not None:
        if total_gb < 9:
            base_batch_size = min(base_batch_size, 24)
            max_batch_size = min(max_batch_size, 64)
            reference_vram_gb = max(reference_vram_gb, 12)
            aggression *= 0.85
        elif total_gb < 12:
            base_batch_size = min(base_batch_size, 32)
            max_batch_size = min(max_batch_size, 96)
            reference_vram_gb = max(reference_vram_gb, 14)
            aggression *= 0.92
        elif total_gb < 16:
            base_batch_size = min(base_batch_size, 48)
            max_batch_size = min(max_batch_size, 160)
            reference_vram_gb = max(reference_vram_gb, 16)
        elif total_gb < 24:
            max_batch_size = min(max_batch_size, 256)

        if free_gb >= 24:
            aggression *= 1.15
        elif free_gb >= 16:
            aggression *= 1.08
        elif free_gb >= 12:
            aggression *= 1.04

        if total_gb >= 24 and free_gb >= 16:
            base_batch_size = max(base_batch_size, 96)
        if total_gb >= 32 and free_gb >= 24:
            base_batch_size = max(base_batch_size, 112)
    elif mode == "tagmatch":
        # WD ViT-Large 448px: heavier per image than CLIP; conservative VRAM caps
        if total_gb < 9:
            base_batch_size = min(base_batch_size, 8)
            max_batch_size = min(max_batch_size, 16)
            reference_vram_gb = max(reference_vram_gb, 16)
            aggression *= 0.85
        elif total_gb < 12:
            base_batch_size = min(base_batch_size, 12)
            max_batch_size = min(max_batch_size, 32)
            reference_vram_gb = max(reference_vram_gb, 14)
            aggression *= 0.9
        elif total_gb < 16:
            base_batch_size = min(base_batch_size, 16)
            max_batch_size = min(max_batch_size, 48)

        if free_gb >= 20:
            aggression *= 1.15
        elif free_gb >= 14:
            aggression *= 1.08

        if total_gb >= 16 and free_gb >= 10:
            base_batch_size = max(base_batch_size, 24)
        if total_gb >= 24 and free_gb >= 16:
            base_batch_size = max(base_batch_size, 32)

    if effective_free_gb <= 0:
        return 1 if mode == "imagereward" else max(1, min(8, max_batch_size))

    scaled = max(1, int(base_batch_size * aggression * (effective_free_gb / float(reference_vram_gb))))

    if scaled >= 32:
        scaled = max(32, (scaled // 16) * 16)
    elif scaled >= 16:
        scaled = max(16, (scaled // 8) * 8)
    elif scaled >= 8:
        scaled = max(8, (scaled // 4) * 4)

    return max(1, min(max_batch_size, scaled))


def iter_imagereward_scores(image_paths, model, device, prompt, source_paths=None):
    # Yield partial results so Gradio can update progress while large folders are scored.
    scores = {}
    total = len(image_paths)
    done = 0
    batch_size = get_auto_batch_size(device, mode="imagereward")
    model = model.to(device).eval()
    print(f"[ImageReward] Using batch size {batch_size}")
    text_input = model.blip.tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    ).to(device)

    def _mark_failed(source_path):
        scores[os.path.basename(source_path)] = {"score": -float("inf"), "path": source_path}

    def _submit_prefetch(executor, start_index, size):
        if start_index >= total:
            return None
        next_paths = image_paths[start_index:start_index + size]
        next_source_paths = source_paths[start_index:start_index + size] if source_paths is not None else next_paths
        return executor.submit(prepare_imagereward_loaded_batch, next_paths, next_source_paths, model)

    def _run_scoring_batch(loaded_tensors):
        if not loaded_tensors:
            return None, None

        transfer_started = time.perf_counter()
        loaded = [(scoring_path, source_path) for scoring_path, source_path, _ in loaded_tensors]
        tensors = [tensor for _, _, tensor in loaded_tensors]
        images = torch.stack(tensors, dim=0).to(device)
        transfer_ms = promptmatch_timing_ms(transfer_started)

        gpu_started = time.perf_counter()
        image_embeds = model.blip.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        batch_len = image_embeds.shape[0]
        text_output = model.blip.text_encoder(
            text_input.input_ids.expand(batch_len, -1),
            attention_mask=text_input.attention_mask.expand(batch_len, -1),
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].float()
        rewards = model.mlp(txt_features)
        rewards = (rewards - model.mean) / model.std
        if device == "cuda":
            torch.cuda.synchronize()
        gpu_ms = promptmatch_timing_ms(gpu_started)

        copy_started = time.perf_counter()
        rewards = torch.squeeze(rewards, dim=-1).detach().cpu().tolist()
        copy_ms = promptmatch_timing_ms(copy_started)

        for (_, source_path), reward in zip(loaded, rewards):
            scores[os.path.basename(source_path)] = {"score": float(reward), "path": source_path}
        return loaded, {
            "host_to_device": transfer_ms,
            "gpu_encode": gpu_ms,
            "device_to_host": copy_ms,
        }

    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetched = None
    prefetched_start = None
    prefetched_size = None
    prefetched_future = None
    try:
        while done < total:
            current_size = min(batch_size, total - done)
            batch_paths = image_paths[done:done + current_size]
            batch_source_paths = source_paths[done:done + current_size] if source_paths is not None else batch_paths
            batch_start = done + 1
            batch_end = done + len(batch_paths)
            print(f"[ImageReward] Batch {batch_start}-{batch_end}/{total} ({len(batch_paths)} images)")

            prefetch_wait_ms = 0.0
            if prefetched is not None and prefetched_start == done and prefetched_size == current_size:
                loaded_tensors, failed_entries, load_timings = prefetched
                prefetched = None
            else:
                wait_started = time.perf_counter()
                loaded_tensors, failed_entries, load_timings = prepare_imagereward_loaded_batch(batch_paths, batch_source_paths, model)
                prefetch_wait_ms = promptmatch_timing_ms(wait_started)

            for _, source_path, exc in failed_entries:
                print(f"  [WARN] {source_path}: {exc}")
                _mark_failed(source_path)

            next_start = done + len(batch_paths)
            next_size = min(batch_size, total - next_start) if next_start < total else 0
            prefetched_start = next_start if next_size else None
            prefetched_size = next_size if next_size else None
            prefetched_future = _submit_prefetch(prefetch_executor, next_start, next_size) if next_size else None

            try:
                with torch.inference_mode():
                    _, run_timings = _run_scoring_batch(loaded_tensors)
                vram_info = current_free_vram_gb()
                imagereward_log_batch_timing(
                    "score timings",
                    batch_start,
                    batch_end,
                    total,
                        {
                            "load": load_timings.get("load"),
                            "prefetch_wait": prefetch_wait_ms,
                            "preprocess": load_timings.get("preprocess"),
                            "host_to_device": run_timings.get("host_to_device") if run_timings else None,
                            "gpu_encode": run_timings.get("gpu_encode") if run_timings else None,
                            "device_to_host": run_timings.get("device_to_host") if run_timings else None,
                            "free_vram_gb": vram_info[0] if vram_info is not None else None,
                        },
                    )
                done += len(batch_paths)
                yield {"type": "progress", "done": done, "total": total, "batch_size": batch_size, "scores": dict(scores)}
            except Exception as exc:
                if device == "cuda" and is_cuda_oom_error(exc) and current_size > 1:
                    batch_size = max(1, current_size // 2)
                    print(f"[ImageReward] CUDA OOM, retrying with batch size {batch_size}")
                    if prefetched_future is not None:
                        prefetched_future.cancel()
                        prefetched_future = None
                    prefetched = None
                    prefetched_start = None
                    prefetched_size = None
                    prefetch_executor.shutdown(wait=False, cancel_futures=True)
                    prefetch_executor = ThreadPoolExecutor(max_workers=1)
                    torch.cuda.empty_cache()
                    yield {"type": "oom", "done": done, "total": total, "batch_size": batch_size, "scores": dict(scores)}
                    continue

                print(f"Batch error at {done}: {exc}", file=sys.stderr)
                for source_path in batch_source_paths:
                    _mark_failed(source_path)
                done += len(batch_paths)
                yield {"type": "progress", "done": done, "total": total, "batch_size": batch_size, "scores": dict(scores)}
            finally:
                if device == "cuda":
                    torch.cuda.empty_cache()

            if prefetched_future is not None:
                prefetched = prefetched_future.result()
                prefetched_future = None
            else:
                prefetched = None
                prefetched_start = None
                prefetched_size = None
    finally:
        if prefetched_future is not None:
            prefetched_future.cancel()
        prefetch_executor.shutdown(wait=False, cancel_futures=True)


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
                 siglip_model="google/siglip-so400m-patch14-384",
                 clip_cache_dir=None,
                 huggingface_cache_dir=None):
        self.device = device
        self.backend = backend
        self._clip_model = clip_model
        self._openclip_model = openclip_model
        self._openclip_pre = openclip_pretrained
        self._siglip_model = siglip_model
        cache_cfg = get_cache_config()
        self._clip_cache_dir = clip_cache_dir if clip_cache_dir is not None else cache_cfg["clip_dir"]
        self._huggingface_cache_dir = (
            huggingface_cache_dir if huggingface_cache_dir is not None else cache_cfg["huggingface_dir"]
        )
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
        self._model, self._preprocess = _clip.load(
            self._clip_model,
            device=self.device,
            download_root=self._clip_cache_dir,
        )
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
                self._openclip_model,
                pretrained=self._openclip_pre,
                device=self.device,
                cache_dir=self._huggingface_cache_dir,
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
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(
            self._siglip_model,
            cache_dir=self._huggingface_cache_dir,
            use_fast=False,
        )
        self._model = AutoModel.from_pretrained(
            self._siglip_model,
            dtype=dtype,
            cache_dir=self._huggingface_cache_dir,
        ).to(self.device)
        self._model.eval()

    def _encode_text_plain(self, prompt):
        prompt = normalize_prompt_text(prompt)
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

    def _blend_text_embeddings(self, weighted_prompts):
        mixed = None
        total_weight = 0.0

        for prompt_text, weight in weighted_prompts:
            prompt_text = normalize_prompt_text(prompt_text)
            weight = float(weight)
            if not prompt_text or weight <= 0:
                continue
            emb = self._encode_text_plain(prompt_text)
            mixed = (emb * weight) if mixed is None else (mixed + (emb * weight))
            total_weight += weight

        if mixed is None:
            fallback = normalize_prompt_text(weighted_prompts[0][0]) if weighted_prompts else ""
            return self._encode_text_plain(fallback)

        if total_weight > 0:
            mixed = mixed / total_weight
        return F.normalize(mixed, dim=-1)

    def encode_text(self, prompt):
        base_prompt, weighted_fragments, segments = parse_promptmatch_weighted_prompt(prompt)
        base_prompt = base_prompt or normalize_prompt_text(prompt)
        if not weighted_fragments or not base_prompt:
            return self._encode_text_plain(base_prompt or prompt)

        weighted_prompts = [(base_prompt, 1.0)]
        summary = []
        for fragment in weighted_fragments:
            frag_text = fragment["text"]
            frag_weight = fragment["weight"]
            if frag_weight > 1.0:
                weighted_prompts.append((frag_text, frag_weight - 1.0))
                summary.append(f"{frag_text} x{frag_weight:g}")
            elif frag_weight < 1.0:
                reduced_prompt = render_promptmatch_segments(
                    segments,
                    skip_weighted_index=fragment["segment_index"],
                )
                if reduced_prompt and reduced_prompt != base_prompt:
                    weighted_prompts.append((reduced_prompt, 1.0 - frag_weight))
                    summary.append(f"{frag_text} x{frag_weight:g}")

        if summary:
            print(f"[PromptMatch] Weighted prompt fragments: {', '.join(summary)}")
        return self._blend_text_embeddings(weighted_prompts)

    def encode_images_batch(self, pil_images, return_timings=False):
        timings = {}
        with torch.no_grad():
            if self.backend in ("openai", "openclip"):
                preprocess_started = time.perf_counter()
                preprocess_workers = promptmatch_host_worker_count(len(pil_images))
                if preprocess_workers <= 1:
                    processed = [self._preprocess(img) for img in pil_images]
                else:
                    with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                        processed = list(executor.map(self._preprocess, pil_images))
                timings["preprocess"] = promptmatch_timing_ms(preprocess_started)
                transfer_started = time.perf_counter()
                tensors = torch.stack(processed).to(self.device)
                tensors = tensors.to(next(self._model.parameters()).dtype)
                timings["host_to_device"] = promptmatch_timing_ms(transfer_started)
                gpu_started = time.perf_counter()
                feat = self._model.encode_image(tensors)
            elif self.backend == "siglip":
                preprocess_started = time.perf_counter()
                preprocess_workers = promptmatch_host_worker_count(len(pil_images))

                def _process_one(img):
                    batch = self._processor(images=img, return_tensors="pt")
                    return batch["pixel_values"].squeeze(0)

                if preprocess_workers <= 1:
                    processed = [_process_one(img) for img in pil_images]
                else:
                    with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                        processed = list(executor.map(_process_one, pil_images))
                timings["preprocess"] = promptmatch_timing_ms(preprocess_started)
                transfer_started = time.perf_counter()
                pixel_values = torch.stack(processed).to(self.device)
                pixel_values = pixel_values.to(next(self._model.parameters()).dtype)
                timings["host_to_device"] = promptmatch_timing_ms(transfer_started)
                gpu_started = time.perf_counter()
                feat = self._model.get_image_features(pixel_values=pixel_values)
            else:
                preprocess_started = time.perf_counter()
                inputs = self._processor(images=pil_images, return_tensors="pt").to(self.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(next(self._model.parameters()).dtype)
                timings["preprocess"] = promptmatch_timing_ms(preprocess_started)
                gpu_started = time.perf_counter()
                feat = self._model.get_image_features(**inputs)
            if self.device == "cuda":
                torch.cuda.synchronize()
            timings["gpu_encode"] = promptmatch_timing_ms(gpu_started)
            normalize_started = time.perf_counter()
            normalized = F.normalize(feat.float(), dim=-1)
            timings["normalize"] = promptmatch_timing_ms(normalize_started)
            if return_timings:
                return normalized, timings
            return normalized


def score_all(image_paths, backend, pos_emb, neg_emb, progress_cb=None, proxy_resolver=None):
    # PromptMatch scoring path: embed images in batches, then compare against text embeddings.
    results = {}
    total = len(image_paths)
    done = 0
    batch_size = get_auto_batch_size(backend.device, backend)
    print(f"[PromptMatch] Using batch size {batch_size}")

    def _mark_failed(original_path):
        results[os.path.basename(original_path)] = {"pos": 0.0, "neg": None, "path": original_path, "failed": True}

    def _submit_prefetch(executor, start_index, size):
        if start_index >= total:
            return None
        batch = image_paths[start_index:start_index + size]
        return executor.submit(prepare_promptmatch_loaded_batch, batch, proxy_resolver)

    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetched = None
    prefetched_start = None
    prefetched_size = None
    prefetched_future = None
    try:
        while done < total:
            current_size = min(batch_size, total - done)
            batch = image_paths[done:done + current_size]
            batch_start = done + 1
            batch_end = done + len(batch)
            print(f"[PromptMatch] Batch {batch_start}-{batch_end}/{total} ({len(batch)} images)")

            prefetch_wait_ms = 0.0
            if prefetched is not None and prefetched_start == done and prefetched_size == current_size:
                loaded, pil_imgs, failed_paths, load_timings = prefetched
                prefetched = None
            else:
                load_started = time.perf_counter()
                loaded, pil_imgs, failed_paths, load_timings = prepare_promptmatch_loaded_batch(batch, proxy_resolver)
                prefetch_wait_ms = promptmatch_timing_ms(load_started)

            for original_path in failed_paths:
                _mark_failed(original_path)

            next_start = done + len(batch)
            next_size = min(batch_size, total - next_start) if next_start < total else 0
            prefetched_start = next_start if next_size else None
            prefetched_size = next_size if next_size else None
            prefetched_future = _submit_prefetch(prefetch_executor, next_start, next_size) if next_size else None

            if pil_imgs:
                try:
                    encode_started = time.perf_counter()
                    feat, encode_timings = backend.encode_images_batch(pil_imgs, return_timings=True)
                    score_started = time.perf_counter()
                    pos_sims = (feat @ pos_emb.T).squeeze(1).tolist()
                    neg_sims = (feat @ neg_emb.T).squeeze(1).tolist() if neg_emb is not None else [None] * len(loaded)
                    for (original_path, _), pos_score, neg_score in zip(loaded, pos_sims, neg_sims):
                        results[os.path.basename(original_path)] = {
                            "pos": float(pos_score),
                            "neg": float(neg_score) if neg_score is not None else None,
                            "path": original_path,
                            "failed": False,
                        }
                    score_ms = promptmatch_timing_ms(score_started)
                    total_encode_ms = promptmatch_timing_ms(encode_started)
                    vram_info = current_free_vram_gb()
                    promptmatch_log_batch_timing(
                        "score timings",
                        batch_start,
                        batch_end,
                        total,
                        {
                            "load": load_timings.get("load"),
                            "prefetch_wait": prefetch_wait_ms,
                            "preprocess": encode_timings.get("preprocess"),
                            "host_to_device": encode_timings.get("host_to_device"),
                            "gpu_encode": encode_timings.get("gpu_encode"),
                            "normalize": encode_timings.get("normalize"),
                            "score_merge": score_ms,
                            "encode_total": total_encode_ms,
                            "free_vram_gb": vram_info[0] if vram_info is not None else None,
                        },
                    )
                except Exception as exc:
                    if backend.device == "cuda" and is_cuda_oom_error(exc) and current_size > 1:
                        batch_size = max(1, current_size // 2)
                        print(f"[PromptMatch] CUDA OOM, retrying with batch size {batch_size}")
                        if prefetched_future is not None:
                            prefetched_future.cancel()
                            prefetched_future = None
                        prefetched = None
                        prefetched_start = None
                        prefetched_size = None
                        prefetch_executor.shutdown(wait=False, cancel_futures=True)
                        prefetch_executor = ThreadPoolExecutor(max_workers=1)
                        torch.cuda.empty_cache()
                        if progress_cb:
                            progress_cb(done, total, batch_size, True)
                        continue
                    print(f"  [WARN] batch error, retrying individually: {exc}")
                    recovered = 0
                    failed = 0
                    for original_path, _ in loaded:
                        try:
                            single_loaded, single_imgs, single_failed = prepare_promptmatch_loaded_batch([original_path], proxy_resolver)
                            for failed_path in single_failed:
                                _mark_failed(failed_path)
                            if not single_imgs:
                                failed += 1
                                continue
                            single_feat = backend.encode_images_batch(single_imgs)
                            pos_score = float((single_feat @ pos_emb.T).squeeze().item())
                            neg_score = float((single_feat @ neg_emb.T).squeeze().item()) if neg_emb is not None else None
                            results[os.path.basename(original_path)] = {
                                "pos": pos_score,
                                "neg": neg_score,
                                "path": original_path,
                                "failed": False,
                            }
                            recovered += 1
                        except Exception as single_exc:
                            print(f"  [WARN] single-image error for {original_path}: {single_exc}")
                            _mark_failed(original_path)
                            failed += 1
                    print(f"[PromptMatch] Individual retry result: {recovered} recovered, {failed} failed")
                if backend.device == "cuda":
                    torch.cuda.empty_cache()

            done += len(batch)
            if progress_cb:
                progress_cb(done, total, batch_size, False)

            if prefetched_future is not None:
                prefetch_result_started = time.perf_counter()
                prefetched = prefetched_future.result()
                if prefetched and len(prefetched) >= 4:
                    loaded_prefetch, pil_prefetch, failed_prefetch, timing_prefetch = prefetched
                    timing_prefetch = dict(timing_prefetch or {})
                    timing_prefetch["prefetch_ready_wait"] = promptmatch_timing_ms(prefetch_result_started)
                    prefetched = (loaded_prefetch, pil_prefetch, failed_prefetch, timing_prefetch)
                prefetched_future = None
            else:
                prefetched = None
                prefetched_start = None
                prefetched_size = None
    finally:
        if prefetched_future is not None:
            prefetched_future.cancel()
        prefetch_executor.shutdown(wait=False, cancel_futures=True)
    return results


def encode_all_promptmatch_images(image_paths, backend, progress_cb=None, proxy_resolver=None):
    # Cacheable PromptMatch path: encode image features once, then reuse them for prompt changes.
    feature_paths = []
    feature_rows = []
    failed_paths = []
    failed_seen = set()
    total = len(image_paths)
    done = 0
    batch_size = get_auto_batch_size(backend.device, backend)
    print(f"[PromptMatch] Using batch size {batch_size}")

    def _mark_failed(original_path):
        if original_path not in failed_seen:
            failed_seen.add(original_path)
            failed_paths.append(original_path)

    def _submit_prefetch(executor, start_index, size):
        if start_index >= total:
            return None
        batch = image_paths[start_index:start_index + size]
        return executor.submit(prepare_promptmatch_loaded_batch, batch, proxy_resolver)

    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetched = None
    prefetched_start = None
    prefetched_size = None
    prefetched_future = None
    try:
        while done < total:
            current_size = min(batch_size, total - done)
            batch = image_paths[done:done + current_size]
            batch_start = done + 1
            batch_end = done + len(batch)
            print(f"[PromptMatch] Batch {batch_start}-{batch_end}/{total} ({len(batch)} images)")

            prefetch_wait_ms = 0.0
            if prefetched is not None and prefetched_start == done and prefetched_size == current_size:
                loaded, pil_imgs, batch_failed, load_timings = prefetched
                prefetched = None
            else:
                load_started = time.perf_counter()
                loaded, pil_imgs, batch_failed, load_timings = prepare_promptmatch_loaded_batch(batch, proxy_resolver)
                prefetch_wait_ms = promptmatch_timing_ms(load_started)

            for original_path in batch_failed:
                _mark_failed(original_path)

            next_start = done + len(batch)
            next_size = min(batch_size, total - next_start) if next_start < total else 0
            prefetched_start = next_start if next_size else None
            prefetched_size = next_size if next_size else None
            prefetched_future = _submit_prefetch(prefetch_executor, next_start, next_size) if next_size else None

            if pil_imgs:
                try:
                    encode_started = time.perf_counter()
                    feat, encode_timings = backend.encode_images_batch(pil_imgs, return_timings=True)
                    copy_started = time.perf_counter()
                    feat = feat.detach().cpu()
                    for (original_path, _), row in zip(loaded, feat):
                        feature_paths.append(original_path)
                        feature_rows.append(row)
                    copy_ms = promptmatch_timing_ms(copy_started)
                    total_encode_ms = promptmatch_timing_ms(encode_started)
                    vram_info = current_free_vram_gb()
                    promptmatch_log_batch_timing(
                        "cache timings",
                        batch_start,
                        batch_end,
                        total,
                        {
                            "load": load_timings.get("load"),
                            "prefetch_wait": prefetch_wait_ms,
                            "preprocess": encode_timings.get("preprocess"),
                            "host_to_device": encode_timings.get("host_to_device"),
                            "gpu_encode": encode_timings.get("gpu_encode"),
                            "normalize": encode_timings.get("normalize"),
                            "device_to_host": copy_ms,
                            "encode_total": total_encode_ms,
                            "free_vram_gb": vram_info[0] if vram_info is not None else None,
                        },
                    )
                except Exception as exc:
                    if backend.device == "cuda" and is_cuda_oom_error(exc) and current_size > 1:
                        batch_size = max(1, current_size // 2)
                        print(f"[PromptMatch] CUDA OOM, retrying with batch size {batch_size}")
                        if prefetched_future is not None:
                            prefetched_future.cancel()
                            prefetched_future = None
                        prefetched = None
                        prefetched_start = None
                        prefetched_size = None
                        prefetch_executor.shutdown(wait=False, cancel_futures=True)
                        prefetch_executor = ThreadPoolExecutor(max_workers=1)
                        torch.cuda.empty_cache()
                        if progress_cb:
                            progress_cb(done, total, batch_size, True)
                        continue
                    print(f"  [WARN] batch error, retrying individually: {exc}")
                    recovered = 0
                    failed = 0
                    for original_path, _ in loaded:
                        before_count = len(feature_paths)
                        before_failed = len(failed_paths)
                        try:
                            single_loaded, single_imgs, single_failed = prepare_promptmatch_loaded_batch([original_path], proxy_resolver)
                            for failed_path in single_failed:
                                _mark_failed(failed_path)
                            if not single_imgs:
                                failed += 1
                                continue
                            single_feat = backend.encode_images_batch(single_imgs).detach().cpu()
                            for (_, _), row in zip(single_loaded, single_feat):
                                feature_paths.append(original_path)
                                feature_rows.append(row)
                            if len(feature_paths) > before_count:
                                recovered += 1
                            elif len(failed_paths) > before_failed:
                                failed += 1
                        except Exception as single_exc:
                            print(f"  [WARN] single-image error for {original_path}: {single_exc}")
                            _mark_failed(original_path)
                            failed += 1
                    print(f"[PromptMatch] Individual retry result: {recovered} recovered, {failed} failed")
                if backend.device == "cuda":
                    torch.cuda.empty_cache()

            done += len(batch)
            if progress_cb:
                progress_cb(done, total, batch_size, False)

            if prefetched_future is not None:
                prefetch_result_started = time.perf_counter()
                prefetched = prefetched_future.result()
                if prefetched and len(prefetched) >= 4:
                    loaded_prefetch, pil_prefetch, failed_prefetch, timing_prefetch = prefetched
                    timing_prefetch = dict(timing_prefetch or {})
                    timing_prefetch["prefetch_ready_wait"] = promptmatch_timing_ms(prefetch_result_started)
                    prefetched = (loaded_prefetch, pil_prefetch, failed_prefetch, timing_prefetch)
                prefetched_future = None
            else:
                prefetched = None
                prefetched_start = None
                prefetched_size = None
    finally:
        if prefetched_future is not None:
            prefetched_future.cancel()
        prefetch_executor.shutdown(wait=False, cancel_futures=True)

    feature_tensor = torch.stack(feature_rows) if feature_rows else torch.empty((0, 0), dtype=torch.float32)
    return feature_paths, feature_tensor, failed_paths


def score_promptmatch_cached_features(feature_paths, image_features, failed_paths, pos_emb, neg_emb):
    results = {}
    pos_emb_cpu = pos_emb.detach().float().cpu()
    neg_emb_cpu = neg_emb.detach().float().cpu() if neg_emb is not None else None

    if feature_paths and image_features.numel():
        pos_sims = (image_features @ pos_emb_cpu.T).squeeze(1).tolist()
        neg_sims = (image_features @ neg_emb_cpu.T).squeeze(1).tolist() if neg_emb_cpu is not None else [None] * len(feature_paths)
        for original_path, pos_score, neg_score in zip(feature_paths, pos_sims, neg_sims):
            results[os.path.basename(original_path)] = {
                "pos": float(pos_score),
                "neg": float(neg_score) if neg_score is not None else None,
                "path": original_path,
                "failed": False,
            }

    for original_path in failed_paths:
        results[os.path.basename(original_path)] = {
            "pos": 0.0,
            "neg": None,
            "path": original_path,
            "failed": True,
        }

    return results


MODEL_CHOICES = [
    ("SigLIP  base-patch16-224  [~5 GB]", "siglip", {"siglip_model": "google/siglip-base-patch16-224"}),
    ("SigLIP  so400m-patch14-384  [<8 GB]  ★ recommended", "siglip", {"siglip_model": "google/siglip-so400m-patch14-384"}),
    ("OpenCLIP  ViT-L-14  laion2b  [<6 GB]", "openclip", {"openclip_model": "ViT-L-14", "openclip_pretrained": "laion2b_s32b_b82k"}),
    ("OpenCLIP  ConvNeXt-Base-W  laion2b  [<8 GB]", "openclip", {"openclip_model": "convnext_base_w", "openclip_pretrained": "laion2b_s13b_b82k"}),
    ("OpenCLIP  ViT-H-14  laion2b  [<10 GB]", "openclip", {"openclip_model": "ViT-H-14", "openclip_pretrained": "laion2b_s32b_b79k"}),
    ("OpenCLIP  ConvNeXt-Large-D-320  laion2b  [16+ GB]", "openclip", {"openclip_model": "convnext_large_d_320", "openclip_pretrained": "laion2b_s29b_b131k_ft"}),
    ("OpenCLIP  ViT-bigG-14  laion2b  [14-16 GB]  ★ best CLIP", "openclip", {"openclip_model": "ViT-bigG-14", "openclip_pretrained": "laion2b_s39b_b160k"}),
]
MODEL_LABELS = [choice[0] for choice in MODEL_CHOICES]
MODEL_STATUS_CACHED_MARKER = "🟢"
MODEL_STATUS_DOWNLOAD_MARKER = "🟠"


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


def promptmatch_model_status_map():
    mapping = {}
    for label, name, kwargs in MODEL_CHOICES:
        if name == "openai":
            source = describe_openai_clip_source(kwargs.get("clip_model"))
        elif name == "openclip":
            source = describe_openclip_source(kwargs.get("openclip_model"), kwargs.get("openclip_pretrained"))
        elif name == "siglip":
            source = describe_siglip_source(kwargs.get("siglip_model"))
        else:
            source = "network or disk cache"
        mapping[label] = {
            "cached": source in {"disk cache", "disk file"},
            "source": source,
        }
    return mapping


def promptmatch_model_status_json():
    return json.dumps(promptmatch_model_status_map())


def promptmatch_model_dropdown_choices():
    status_map = promptmatch_model_status_map()
    choices = []
    for label in MODEL_LABELS:
        entry = status_map.get(label) or {}
        marker = MODEL_STATUS_CACHED_MARKER if entry.get("cached") else MODEL_STATUS_DOWNLOAD_MARKER
        choices.append((f"{marker} {label}", label))
    return choices


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


def sanitize_export_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("._-")
    return cleaned


def uses_similarity_topn(method):
    return method in (METHOD_SIMILARITY, METHOD_SAMEPERSON)


def uses_pos_similarity_scores(method):
    return method in (METHOD_PROMPTMATCH, METHOD_LLMSEARCH, METHOD_SIMILARITY, METHOD_SAMEPERSON, METHOD_TAGMATCH)


def export_destination(folder, filename):
    return os.path.join(folder, filename)


def threshold_labels(method):
    if method == METHOD_LLMSEARCH:
        return (
            "Minimum LLM rerank score to keep (higher = fewer kept)",
            "Negative score is unused for LLM rerank search",
            "Minimum LLM rerank score",
            "Negative score",
        )
    if method == METHOD_SIMILARITY:
        return (
            "Minimum similarity to keep (higher = fewer kept)",
            "Negative similarity is unused for image similarity search",
            "Minimum similarity",
            "Negative similarity",
        )
    if method == METHOD_SAMEPERSON:
        return (
            "Minimum face similarity to keep (higher = fewer kept)",
            "Negative face similarity is unused for same-person search",
            "Minimum face similarity",
            "Negative face similarity",
        )
    if method == METHOD_PROMPTMATCH:
        return (
            "Minimum positive similarity to keep (higher = fewer kept)",
            "Maximum negative similarity allowed (lower = fewer kept)",
            "Min positive similarity",
            "Max negative similarity",
        )
    if method == METHOD_TAGMATCH:
        return (
            "Minimum artifact score to keep (higher = fewer kept)",
            "Negative score is unused for TagMatch",
            "Min artifact score",
            "Negative score",
        )
    return (
        "Minimum score to keep (higher = fewer kept)",
        "Maximum negative similarity allowed (lower = fewer kept)",
        "Minimum keep score",
        "Max negative similarity",
    )


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
    if uses_similarity_topn(method):
        valid_items = [
            item for item in (scores or {}).values()
            if not item.get("failed", False) and item.get("pos") is not None
        ]
        if not valid_items:
            return 0.0
        valid_items.sort(key=lambda item: -float(item["pos"]))
        query_offset = 1 if valid_items and valid_items[0].get("query") else 0
        vals = [float(item["pos"]) for item in valid_items[query_offset:]]
        if not vals:
            return round(float(valid_items[0]["pos"]), 3)
        try:
            top_n = int(round(float(percentile)))
        except Exception:
            top_n = SIMILARITY_TOPN_DEFAULT
        top_n = max(1, min(len(vals), top_n))
        return round(vals[top_n - 1], 3)
    if uses_pos_similarity_scores(method):
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


def percentile_slider_label(method):
    if method == METHOD_LLMSEARCH:
        return "Or keep top N%"
    if method == METHOD_SIMILARITY:
        return "Show the N most similar"
    if method == METHOD_SAMEPERSON:
        return "Show the N closest face matches"
    return "Or keep top N%"


def estimate_similarity_topn(scores=None):
    valid_items = [
        item for item in (scores or {}).values()
        if not item.get("failed", False) and item.get("pos") is not None
    ]
    if not valid_items:
        return 1

    valid_items.sort(key=lambda item: -float(item["pos"]))
    query_offset = 1 if valid_items and valid_items[0].get("query") else 0
    ranked_scores = [float(item["pos"]) for item in valid_items[query_offset:query_offset + SIMILARITY_AUTO_TOPN_SCAN_LIMIT]]

    if len(ranked_scores) < 2:
        return max(1, min(len(ranked_scores), SIMILARITY_TOPN_SLIDER_MAX))

    first = ranked_scores[0]
    last = ranked_scores[-1]
    if abs(first - last) < 1e-9:
        fallback = min(len(ranked_scores), SIMILARITY_TOPN_DEFAULT)
        return max(1, fallback)

    best_idx = 0
    best_distance = -float("inf")
    span = max(1, len(ranked_scores) - 1)
    for idx, score in enumerate(ranked_scores):
        x = idx / span
        y = (score - last) / (first - last)
        distance = y - (1.0 - x)
        if distance > best_distance:
            best_distance = distance
            best_idx = idx

    if best_distance < SIMILARITY_AUTO_KNEE_MIN_DISTANCE:
        fallback = min(len(ranked_scores), SIMILARITY_TOPN_DEFAULT)
        return max(1, min(fallback, SIMILARITY_TOPN_SLIDER_MAX))

    auto_topn = best_idx + 1
    min_topn = min(len(ranked_scores), max(1, SIMILARITY_AUTO_TOPN_MIN))
    auto_topn = max(min_topn, auto_topn)
    auto_topn = min(len(ranked_scores), auto_topn, SIMILARITY_TOPN_SLIDER_MAX)
    return max(1, auto_topn)


def similarity_topn_defaults(scores=None):
    valid_items = [
        item for item in (scores or {}).values()
        if not item.get("failed", False) and item.get("pos") is not None
    ]
    if not valid_items:
        return 1, SIMILARITY_TOPN_DEFAULT
    valid_items.sort(key=lambda item: -float(item["pos"]))
    query_offset = 1 if valid_items and valid_items[0].get("query") else 0
    similar_count = len(valid_items) - query_offset
    if similar_count <= 0:
        return 1, 1
    slider_max = min(SIMILARITY_TOPN_SLIDER_MAX, similar_count)
    auto_default = estimate_similarity_topn(scores)
    return slider_max, min(auto_default, slider_max)


def percentile_slider_update(method, scores=None):
    if uses_similarity_topn(method):
        max_items, default_items = similarity_topn_defaults(scores)
        return gr.update(
            minimum=1,
            maximum=max_items,
            value=default_items,
            step=1,
            label=percentile_slider_label(method),
        )
    return gr.update(
        minimum=0,
        maximum=100,
        value=50,
        step=1,
        label=percentile_slider_label(method),
    )


def percentile_reset_button_update(method, scores=None):
    if uses_similarity_topn(method):
        _, default_items = similarity_topn_defaults(scores)
        return gr.update(value=str(default_items))
    return gr.update(value="50%")


def slider_step_floor(value, step=0.001):
    return round(math.floor(float(value) / step) * step, 3)


def slider_step_ceil_exclusive(value, step=0.001):
    return round((math.floor(float(value) / step) + 1) * step, 3)


def clamp_threshold(value, lo, hi):
    return round(max(float(lo), min(float(hi), float(value))), 3)


def expand_slider_bounds(lo, hi, *values):
    safe_values = [float(v) for v in values if v is not None]
    if safe_values:
        lo = min(float(lo), *safe_values)
        hi = max(float(hi), *safe_values)
    # Floor lo / ceil hi so actual score values never fall outside the slider bounds.
    # round() can push lo above the true minimum, causing Gradio out-of-bounds errors.
    return math.floor(lo * 1000) / 1000, math.ceil(hi * 1000) / 1000


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


def wd_tags_detail_config(detail_level):
    try:
        detail_level = int(detail_level)
    except Exception:
        detail_level = DEFAULT_GENERATED_PROMPT_DETAIL
    detail_level = max(1, min(3, detail_level))
    # detail_prompt field holds the top-N count used by run_wd_tags_prompt_variant
    mapping = {
        1: ("Top 12 tags",  12),
        2: ("Top 36 tags",  36),
        3: ("Top 96 tags",  96),
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
    if generator_name == PROMPT_GENERATOR_WD_TAGS:
        return wd_tags_detail_config(detail_level)
    return joycaption_detail_config(detail_level)


def extract_joycaption_caption(text):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    text = re.sub(r"^(assistant|caption)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def llmsearch_joycaption_system_prompt():
    return LLMSEARCH_JOYCAPTION_SYSTEM_PROMPT


def build_llmsearch_joycaption_user_prompt(query_text):
    normalized_query = normalize_prompt_text(query_text or "")
    return (
        "You are an image evaluation model.\n\n"
        "Your task is to score how well an image matches a given user prompt.\n\n"
        "Scoring rules:\n"
        "- Return a single number from 0 to 100 (integer only).\n"
        "- 0 = the image does NOT match the described prompt at all.\n"
        "- 100 = the image clearly and strongly matches the described prompt.\n"
        "- Intermediate values reflect partial presence or uncertainty.\n\n"
        "Evaluation guidelines:\n"
        "- Focus ONLY on visible evidence in the image.\n"
        "- Do NOT hallucinate details.\n"
        "- Be strict: weak or partial matches = low score, strong obvious matches = high score.\n"
        "- If multiple traits, attributes, objects, actions, styles, or conditions are listed, evaluate their combined presence.\n"
        "- If nothing is visible, return 0.\n\n"
        "Output format:\n"
        "- Return ONLY the number.\n"
        "- No explanation, no text, no symbols, no percentage sign.\n\n"
        "Prompt:\n"
        f"{normalized_query or 'no traits provided'}"
    )


def normalize_llmsearch_candidate_text(text):
    text = extract_joycaption_caption(text)
    text = re.sub(r"^(search prompt|user search prompt|match summary|caption)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[-*]\s*", ", ", text)
    return normalize_generated_prompt(text)


def extract_llmsearch_numeric_score(text):
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        raise ValueError("LLM rerank backend returned an empty score.")
    match = re.fullmatch(r"(100|[0-9]{1,2})", cleaned)
    if not match:
        raise ValueError(f"LLM rerank backend did not return a usable 0-100 integer: {cleaned!r}")
    score_value = int(match.group(1))
    return max(0, min(100, score_value))


def image_to_data_url(image, image_format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def joycaption_gguf_prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((336, 336), Image.Resampling.BILINEAR)


def status_line(method, left_items, right_items, scores, overrides):
    left_name, right_name, _, _ = method_labels(method)
    if not scores:
        failed = 0
    elif uses_pos_similarity_scores(method) and all("pos" in v for v in scores.values()):
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

    if uses_pos_similarity_scores(method) and not all("pos" in item for item in scores.values()):
        return left, right
    if method == METHOD_IMAGEREWARD and not all("score" in item for item in scores.values()):
        return left, right

    if uses_pos_similarity_scores(method):
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
            query_suffix = " | QUERY" if item.get("query") else ""
            caption = f"{'✋ ' if fname in overrides else ''}{score_text}{query_suffix} | {fname}"
            if item.get("base_pos") is not None:
                caption += f"  (shortlist {item['base_pos']:.3f})"
            if method == METHOD_LLMSEARCH:
                raw_response = (item.get("caption") or "").strip()
                if raw_response:
                    caption += f"  [LLM: {raw_response}]"
                elif item.get("reason"):
                    caption += f"  [LLM error: {item['reason']}]"
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
    configure_torch_cpu_threads()
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
        clip_cache_dir=get_cache_config()["clip_dir"],
        huggingface_cache_dir=get_cache_config()["huggingface_dir"],
    )

    # Shared mutable state for the one-page app. Gradio callbacks update this in place.
    state = {
        "method": METHOD_PROMPTMATCH,
        "source_dir": source_dir,
        "view_mode": "scored",
        "browse_items": [],
        "browse_status": "",
        "scores": {},
        "overrides": {},
        "last_scored_method": None,
        "last_scored_folder_key": None,
        "last_promptmatch_model_label": None,
        "left_marked": [],
        "right_marked": [],
        "preview_fname": None,
        "backend": prompt_backend,
        "ir_model": None,
        "face_backend": None,
        "tagmatch_backend": None,
        "hist_geom": None,
        "hist_width": 300,
        "proxy_folder_key": None,
        "proxy_cache_dir": None,
        "proxy_signature": None,
        "proxy_map": {},
        "use_proxy_display": True,
        "ir_penalty_weight": DEFAULT_IR_PENALTY_WEIGHT,
        "pm_cached_signature": None,
        "pm_cached_model_label": None,
        "pm_cached_feature_paths": None,
        "pm_cached_image_features": None,
        "pm_cached_failed_paths": None,
        "face_cached_signature": None,
        "face_cached_feature_paths": None,
        "face_cached_embeddings": None,
        "face_cached_failures": None,
        "ir_cached_signature": None,
        "ir_cached_positive_prompt": None,
        "ir_cached_negative_prompt": None,
        "ir_cached_base_scores": None,
        "ir_cached_penalty_scores": None,
        "llmsearch_backend": DEFAULT_LLMSEARCH_BACKEND,
        "llmsearch_shortlist_size": LLMSEARCH_SHORTLIST_DEFAULT,
        "llmsearch_cached_signature": None,
        "llmsearch_cached_prompt": None,
        "llmsearch_cached_backend": None,
        "llmsearch_cached_scoring_mode": None,
        "llmsearch_cached_shortlist_size": None,
        "llmsearch_cached_model_label": None,
        "llmsearch_cached_scores": None,
        "llmsearch_cached_captions": {},
        "tagmatch_cached_signature": None,
        "tagmatch_cached_feature_paths": None,
        "tagmatch_cached_tag_vectors": None,
        "prompt_generator": DEFAULT_PROMPT_GENERATOR,
        "prompt_backend_cache": {},
        "generated_prompt": "",
        "generated_prompt_source": None,
        "generated_prompt_backend": DEFAULT_PROMPT_GENERATOR,
        "generated_prompt_detail": DEFAULT_GENERATED_PROMPT_DETAIL,
        "generated_prompt_variants": {},
        "generated_prompt_status": "Preview an image, then generate a prompt from it.",
        "similarity_query_fname": None,
        "similarity_model_label": None,
        "sameperson_query_fname": None,
        "sameperson_model_label": None,
        "mode_thresholds": {
            METHOD_PROMPTMATCH: {"main": None, "aux": None},
            METHOD_IMAGEREWARD: {"main": None, "aux": None},
            METHOD_LLMSEARCH: {"main": None, "aux": None},
            METHOD_SIMILARITY: {"main": None, "aux": None},
            METHOD_SAMEPERSON: {"main": None, "aux": None},
            METHOD_TAGMATCH: {"main": None, "aux": None},
        },
        "zoom_columns": 5,
    }

    def is_browse_mode():
        return state.get("view_mode") == "browse"

    def set_scored_mode():
        state["view_mode"] = "scored"

    def set_browse_mode(items=None, status_text=""):
        state["view_mode"] = "browse"
        state["browse_items"] = list(items or [])
        state["browse_status"] = status_text

    def sync_promptmatch_proxy_cache(folder):
        folder_key = normalize_folder_identity(folder)
        if state["proxy_folder_key"] != folder_key:
            clear_promptmatch_proxy_cache(state.get("proxy_cache_dir"))
            state["proxy_folder_key"] = folder_key
            state["proxy_cache_dir"] = get_promptmatch_proxy_cache_dir(folder)
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
        return state["proxy_cache_dir"]

    def can_reuse_proxy_map(image_paths, image_signature):
        proxy_map = state.get("proxy_map") or {}
        if state.get("proxy_signature") != image_signature or not proxy_map:
            return False
        return all(path in proxy_map for path in image_paths)

    def clear_preview_search_context():
        state["similarity_query_fname"] = None
        state["similarity_model_label"] = None
        state["sameperson_query_fname"] = None
        state["sameperson_model_label"] = None

    def remember_mode_thresholds(method, main_threshold, aux_threshold):
        if method not in state["mode_thresholds"]:
            state["mode_thresholds"][method] = {}
        state["mode_thresholds"][method]["main"] = float(main_threshold)
        state["mode_thresholds"][method]["aux"] = float(aux_threshold)

    def recalled_mode_thresholds(method, default_main, default_aux):
        entry = state.get("mode_thresholds", {}).get(method) or {}
        main_value = entry.get("main")
        aux_value = entry.get("aux")
        has_saved = (main_value is not None) or (aux_value is not None)
        if main_value is None:
            main_value = default_main
        if aux_value is None:
            aux_value = default_aux
        return float(main_value), float(aux_value), bool(has_saved)

    def tooltip_head(pairs):
        mapping = json.dumps(pairs)
        return fr"""
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
  const pushShortcutAction = (value) => {{
    const root = document.getElementById("hy-shortcut-action");
    if (!root) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    input.value = value;
    input.dispatchEvent(new Event("input", {{ bubbles: true }}));
    input.dispatchEvent(new Event("change", {{ bubbles: true }}));
  }};
  const pushHistWidth = (value) => {{
    const root = document.getElementById("hy-hist-width");
    if (!root) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    const normalized = String(value);
    if (input.value === normalized) return;
    input.value = normalized;
    input.dispatchEvent(new Event("input", {{ bubbles: true }}));
    input.dispatchEvent(new Event("change", {{ bubbles: true }}));
  }};
  const pushPreviewFname = (fname) => {{
    if (!fname) return;
    pushThumbAction(`previewfname:${{fname}}:${{Date.now()}}`);
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
  const readPromptMatchModelStatus = () => {{
    const root = document.getElementById("hy-model-status");
    if (!root) return {{}};
    const input = root.querySelector("input, textarea");
    if (!input || !input.value) return {{}};
    try {{
      return JSON.parse(input.value);
    }} catch {{
      return {{}};
    }}
  }};
  const paintPromptMatchModelNode = (node, entry, colors) => {{
    if (!(node instanceof HTMLElement) || !entry) return;
    const color = entry.cached ? colors.cached : colors.download;
    node.style.setProperty("color", color, "important");
    node.style.setProperty("-webkit-text-fill-color", color, "important");
    node.style.setProperty("font-weight", entry.cached ? "600" : "500", "important");
  }};
  const findPromptMatchModelStatusEntry = (text, statusMap, knownLabels) => {{
    const raw = `${{text || ""}}`.trim();
    if (!raw) return null;
    const normalized = raw.replace(/\\s+/g, " ").trim();
    if (statusMap[normalized]) return statusMap[normalized];
    let bestLabel = "";
    for (const label of knownLabels) {{
      if (normalized.includes(label) && label.length > bestLabel.length) {{
        bestLabel = label;
      }}
    }}
    return bestLabel ? statusMap[bestLabel] : null;
  }};
  const schedulePromptMatchModelAvailability = () => {{
    applyPromptMatchModelAvailability();
    requestAnimationFrame(() => applyPromptMatchModelAvailability());
    setTimeout(applyPromptMatchModelAvailability, 0);
    setTimeout(applyPromptMatchModelAvailability, 60);
    setTimeout(applyPromptMatchModelAvailability, 180);
  }};
  let repaintTimers = [];
  let activeDrag = null;
  let histResizeObserver = null;
  let activeHoverInfo = null;
  let activeDialogPreviewFname = "";
  let activeDialogSelection = {{ side: "", index: -1 }};
  const findWeightedPromptSpan = (value, selectionStart, selectionEnd) => {{
    const weightedRe = /\\(([^()]*?)\\s*:\\s*([0-9]*\\.?[0-9]+)\\)/g;
    let match;
    while ((match = weightedRe.exec(value)) !== null) {{
      const fullStart = match.index;
      const fullEnd = fullStart + match[0].length;
      const hasSelection = selectionStart !== selectionEnd;
      const insideMatch = hasSelection
        ? (selectionStart >= fullStart && selectionEnd <= fullEnd)
        : (selectionStart >= fullStart && selectionStart <= fullEnd);
      if (!insideMatch) continue;
      return {{
        fullStart,
        fullEnd,
        text: match[1],
        weight: Number.parseFloat(match[2]),
      }};
    }}
    return null;
  }};
  const formatPromptWeight = (weight) => {{
    const rounded = Math.round(weight * 10) / 10;
    return rounded.toFixed(1);
  }};
  const dispatchTextboxEvents = (input) => {{
    input.dispatchEvent(new Event("input", {{ bubbles: true }}));
    input.dispatchEvent(new Event("change", {{ bubbles: true }}));
  }};
  const getPromptRootForElement = (element) => {{
    if (!element || typeof element.closest !== "function") return null;
    return element.closest("#hy-pos, #hy-neg, #hy-ir-pos, #hy-ir-neg");
  }};
  const adjustPromptWeight = (input, delta) => {{
    if (!input) return false;
    const value = input.value || "";
    const selectionStart = input.selectionStart ?? 0;
    const selectionEnd = input.selectionEnd ?? selectionStart;
    const weighted = findWeightedPromptSpan(value, selectionStart, selectionEnd);
    if (weighted) {{
      const newWeight = Math.max(0.1, (weighted.weight || 1.0) + delta);
      const replacement = Math.abs(newWeight - 1.0) < 1e-9
        ? weighted.text
        : `(${{weighted.text}}:${{formatPromptWeight(newWeight)}})`;
      input.value = value.slice(0, weighted.fullStart) + replacement + value.slice(weighted.fullEnd);
      const innerStart = weighted.fullStart + (replacement === weighted.text ? 0 : 1);
      const innerEnd = innerStart + weighted.text.length;
      input.setSelectionRange(innerStart, innerEnd);
      dispatchTextboxEvents(input);
      return true;
    }}
    if (selectionStart === selectionEnd) return false;
    const selectedText = value.slice(selectionStart, selectionEnd);
    const leadingWs = (selectedText.match(/^\\s*/) || [""])[0];
    const trailingWs = (selectedText.match(/\\s*$/) || [""])[0];
    const coreText = selectedText.slice(leadingWs.length, selectedText.length - trailingWs.length);
    if (!coreText) return false;
    const baseWeight = delta >= 0 ? 1.1 : 0.9;
    const replacement = `${{leadingWs}}(${{coreText}}:${{formatPromptWeight(baseWeight)}})${{trailingWs}}`;
    input.value = value.slice(0, selectionStart) + replacement + value.slice(selectionEnd);
    const innerStart = selectionStart + leadingWs.length + 1;
    const innerEnd = innerStart + coreText.length;
    input.setSelectionRange(innerStart, innerEnd);
    dispatchTextboxEvents(input);
    return true;
  }};
  const hookPromptWeightHotkeys = () => {{
    for (const id of ["hy-pos", "hy-neg"]) {{
      const root = document.getElementById(id);
      if (!root) continue;
      const input = root.querySelector("input, textarea");
      if (!input || input.dataset.hyWeightHooked) continue;
      input.addEventListener("keydown", (event) => {{
        if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
        let delta = null;
        const key = event.key || "";
        const code = event.code || "";
        if (key === "+" || key === "=" || code === "NumpadAdd") {{
          delta = 0.1;
        }} else if (key === "-" || key === "_" || code === "NumpadSubtract") {{
          delta = -0.1;
        }}
        if (delta === null) return;
        if (!adjustPromptWeight(input, delta)) return;
        event.preventDefault();
        event.stopPropagation();
      }});
      input.dataset.hyWeightHooked = "1";
    }}
  }};
  const hookRunScoringHotkeys = () => {{
    if (document.body.dataset.hyRunHotkeyHooked) return;
    document.addEventListener("keydown", (event) => {{
      if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
      const key = event.key || "";
      const code = event.code || "";
      if (key !== "Enter" && code !== "NumpadEnter") return;
      const promptRoot = getPromptRootForElement(event.target) || getPromptRootForElement(document.activeElement);
      const promptId = promptRoot ? promptRoot.id : "";
      if (!promptId) return;
      pushShortcutAction(`run:${{promptId}}:${{Date.now()}}`);
      event.preventDefault();
      event.stopPropagation();
    }}, true);
    document.body.dataset.hyRunHotkeyHooked = "1";
  }};
  const hookPreviewDialogTracking = () => {{
    if (document.body.dataset.hyPreviewTrackHooked) return;
    document.addEventListener("click", (event) => {{
      const target = event.target;
      if (!(target instanceof Element)) return;
      const dialogRoot = target.closest('[role="dialog"], [aria-modal="true"]');
      if (!dialogRoot) return;
      disablePreviewDialogNavigation(dialogRoot);
      if (isDialogMainImageClick(event, dialogRoot)) {{
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        closePreviewDialog(dialogRoot);
        return;
      }}
      if (shouldBlockDialogNavigationClick(target, dialogRoot)) {{
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        return;
      }}
      const thumbInfo = getDialogThumbTargetInfo(target);
      if (thumbInfo) {{
        setTimeout(syncDialogPreviewTarget, 40);
        setTimeout(syncDialogPreviewTarget, 140);
        setTimeout(syncDialogPreviewTarget, 280);
        return;
      }}
      setTimeout(syncDialogPreviewTarget, 40);
      setTimeout(syncDialogPreviewTarget, 140);
    }}, true);
    document.addEventListener("keydown", (event) => {{
      const dialogRoot = document.querySelector('[role="dialog"], [aria-modal="true"]');
      if (!dialogRoot) return;
      const key = event.key || "";
      const deltas = {{
        ArrowLeft: -1,
        ArrowUp: -1,
        PageUp: -1,
        ArrowRight: 1,
        ArrowDown: 1,
        PageDown: 1,
        Home: 0,
        End: 0,
      }};
      if (!(key in deltas)) return;
      disablePreviewDialogNavigation(dialogRoot);
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
    }}, true);
    document.body.dataset.hyPreviewTrackHooked = "1";
  }};
  const hookInlinePreviewNavigationLock = () => {{
    if (document.body.dataset.hyInlinePreviewLockHooked) return;
    document.addEventListener("click", (event) => {{
      const target = event.target;
      if (!(target instanceof Element)) return;
      const previewRoot = target.closest("#hy-left-gallery .preview, #hy-right-gallery .preview");
      if (!previewRoot) return;
      if (event.button !== 0) return;
      if (target.closest(".thumbnails, .thumbnail-item, .thumbnail-small, .thumbnail-lg")) return;
      const control = target.closest("button, a, input, textarea, select, label");
      if (control && !control.closest(".media-button")) return;
      const closeButton = findInlinePreviewCloseButton(previewRoot);
      if (!closeButton) return;
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      closeButton.click();
    }}, true);
    document.addEventListener("keydown", (event) => {{
      const key = event.key || "";
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End"].includes(key)) return;
      const target = event.target instanceof Element ? event.target : null;
      const active = document.activeElement instanceof Element ? document.activeElement : null;
      const inPreview = !!(
        target?.closest?.("#hy-left-gallery .preview, #hy-right-gallery .preview")
        || active?.closest?.("#hy-left-gallery .preview, #hy-right-gallery .preview")
      );
      if (!inPreview) return;
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
    }}, true);
    document.body.dataset.hyInlinePreviewLockHooked = "1";
  }};
  const hookHistogramResize = () => {{
    const root = document.getElementById("hy-hist");
    if (!root) return;
    const measure = () => {{
      const rect = root.getBoundingClientRect();
      const width = Math.max(220, Math.round(rect.width || root.clientWidth || 300) - 2);
      pushHistWidth(width);
      syncHistogramHoverLine();
    }};
    if (!histResizeObserver) {{
      histResizeObserver = new ResizeObserver(() => {{
        window.requestAnimationFrame(measure);
      }});
    }}
    if (root.dataset.hyResizeHooked !== "1") {{
      histResizeObserver.observe(root);
      root.dataset.hyResizeHooked = "1";
    }}
    measure();
  }};
  const parseMainScoreFromCaption = (captionText) => {{
    const cleaned = (captionText || "").replace(/^✋\s*/, "").trim();
    const match = cleaned.match(/^-?\d+(?:\.\d+)?/);
    return match ? Number.parseFloat(match[0]) : null;
  }};
  const extractFnameFromCaption = (captionText) => {{
    const text = (captionText || "").trim();
    if (!text) return "";
    const parts = text.split("|");
    return parts.length ? parts[parts.length - 1].trim() : "";
  }};
  const resolveImageSourceTokens = (src) => {{
    const tokens = new Set();
    if (!src) return tokens;
    const raw = String(src);
    const decoded = (() => {{
      try {{
        return decodeURIComponent(raw);
      }} catch {{
        return raw;
      }}
    }})();
    for (const candidate of [raw, decoded]) {{
      if (!candidate) continue;
      tokens.add(candidate);
      for (const part of candidate.split(/[/?#&=]+/)) {{
        const clean = (part || "").trim();
        if (clean) tokens.add(clean);
      }}
      const slashParts = candidate.split("/");
      const tail = slashParts.length ? slashParts[slashParts.length - 1] : "";
      if (tail) tokens.add(tail);
    }}
    return tokens;
  }};
  const extractFnameFromDialogImage = (img, markedState) => {{
    if (!img) return "";
    const direct = extractFnameFromCaption(
      img.getAttribute("alt")
      || img.getAttribute("aria-label")
      || img.getAttribute("title")
      || ""
    );
    if (direct) return direct;
    const mediaLookup = markedState.media_lookup || {{}};
    const sourceTokens = new Set([
      ...resolveImageSourceTokens(img.currentSrc || ""),
      ...resolveImageSourceTokens(img.src || ""),
    ]);
    for (const token of sourceTokens) {{
      if (mediaLookup[token]) return mediaLookup[token];
    }}
    return "";
  }};
  const extractFnameFromDialogText = (dialogRoot, markedState) => {{
    if (!dialogRoot) return "";
    const knownNames = [
      ...(Array.isArray(markedState.left_order) ? markedState.left_order : []),
      ...(Array.isArray(markedState.right_order) ? markedState.right_order : []),
    ];
    if (!knownNames.length) return "";
    const sortedNames = knownNames.slice().sort((a, b) => b.length - a.length);
    const dialogRect = dialogRoot.getBoundingClientRect();
    const thumbs = getDialogThumbImages(dialogRoot);
    const stripTop = thumbs.length
      ? Math.min(...thumbs.map((img) => img.getBoundingClientRect().top))
      : Number.POSITIVE_INFINITY;
    let bestFname = "";
    let bestScore = -Infinity;
    for (const node of Array.from(dialogRoot.querySelectorAll("div, span, p, figcaption, label"))) {{
      if (!(node instanceof Element)) continue;
      if (node.closest("button, [role='button']")) continue;
      if (node.querySelector("img")) continue;
      const rect = node.getBoundingClientRect();
      const style = window.getComputedStyle(node);
      if (
        rect.width <= 0
        || rect.height <= 0
        || style.display === "none"
        || style.visibility === "hidden"
        || style.opacity === "0"
      ) continue;
      const text = (node.textContent || "").trim();
      if (!text || text.length > 300) continue;
      let fname = "";
      const parsed = extractFnameFromCaption(text);
      if (parsed && knownNames.includes(parsed)) {{
        fname = parsed;
      }} else {{
        for (const candidate of sortedNames) {{
          if (text.includes(candidate)) {{
            fname = candidate;
            break;
          }}
        }}
      }}
      if (!fname) continue;
      const centerX = rect.left + (rect.width / 2);
      const dialogCenterX = dialogRect.left + (dialogRect.width / 2);
      let score = 0;
      if (text.includes("|")) score += 800;
      if (Number.isFinite(stripTop) && rect.bottom <= stripTop + 18) {{
        score += 500;
        score -= Math.abs(stripTop - rect.bottom);
      }}
      if (rect.width < dialogRect.width * 0.9) score += 80;
      score += rect.top / 6;
      score -= Math.abs(centerX - dialogCenterX) / 4;
      if (score > bestScore) {{
        bestScore = score;
        bestFname = fname;
      }}
    }}
    return bestFname;
  }};
  const getDialogOrder = (markedState, side) => {{
    if (side === "left") return Array.isArray(markedState.left_order) ? markedState.left_order : [];
    if (side === "right") return Array.isArray(markedState.right_order) ? markedState.right_order : [];
    return [];
  }};
  const updateDialogSelectionFromFname = (markedState, fname) => {{
    if (!fname) return false;
    for (const side of ["left", "right"]) {{
      const order = getDialogOrder(markedState, side);
      const idx = order.indexOf(fname);
      if (idx >= 0) {{
        activeDialogSelection = {{ side, index: idx }};
        return true;
      }}
    }}
    return false;
  }};
  const getActiveDialogImage = () => {{
    const candidates = Array.from(document.querySelectorAll('[role="dialog"] img, [aria-modal="true"] img')).filter((img) => {{
      const rect = img.getBoundingClientRect();
      const style = window.getComputedStyle(img);
      return rect.width >= 220 && rect.height >= 220 && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
    }});
    if (!candidates.length) return null;
    candidates.sort((a, b) => (b.getBoundingClientRect().width * b.getBoundingClientRect().height) - (a.getBoundingClientRect().width * a.getBoundingClientRect().height));
    return candidates[0];
  }};
  const getDialogThumbImages = (dialogRoot) => {{
    const root = dialogRoot || document.querySelector('[role="dialog"], [aria-modal="true"]');
    if (!root) return [];
    const candidates = Array.from(root.querySelectorAll("img")).filter((img) => {{
      const rect = img.getBoundingClientRect();
      const style = window.getComputedStyle(img);
      return rect.width > 12
        && rect.height > 12
        && rect.width <= 180
        && rect.height <= 180
        && style.display !== "none"
        && style.visibility !== "hidden"
        && style.opacity !== "0";
    }});
    if (candidates.length <= 1) return candidates;
    const rows = [];
    for (const img of candidates) {{
      const rect = img.getBoundingClientRect();
      const centerY = rect.top + (rect.height / 2);
      let row = rows.find((entry) => Math.abs(entry.centerY - centerY) <= 22);
      if (!row) {{
        row = {{ centerY, images: [] }};
        rows.push(row);
      }}
      row.images.push(img);
      row.centerY = row.images.reduce((sum, entry) => {{
        const entryRect = entry.getBoundingClientRect();
        return sum + entryRect.top + (entryRect.height / 2);
      }}, 0) / row.images.length;
    }}
    rows.sort((a, b) => {{
      if (b.images.length !== a.images.length) return b.images.length - a.images.length;
      return b.centerY - a.centerY;
    }});
    const strip = rows[0]?.images || [];
    strip.sort((a, b) => a.getBoundingClientRect().left - b.getBoundingClientRect().left);
    return strip;
  }};
  const getDialogThumbTargetInfo = (target) => {{
    const dialogRoot = target.closest('[role="dialog"], [aria-modal="true"]');
    if (!dialogRoot) return null;
    const targetImg = target instanceof HTMLImageElement ? target : target.querySelector?.("img");
    if (!(targetImg instanceof HTMLImageElement)) return null;
    const thumbs = getDialogThumbImages(dialogRoot);
    const idx = thumbs.indexOf(targetImg);
    if (idx < 0) return null;
    return {{ dialogRoot, targetImg, idx }};
  }};
  const getDialogControlLabel = (element) => {{
    if (!(element instanceof Element)) return "";
    return `${{element.getAttribute("aria-label") || ""}} ${{element.getAttribute("title") || ""}} ${{(element.textContent || "").trim()}}`.trim().toLowerCase();
  }};
  const findInlinePreviewCloseButton = (previewRoot) => {{
    if (!(previewRoot instanceof Element)) return null;
    const candidates = Array.from(previewRoot.querySelectorAll("button")).filter((button) => {{
      return !button.matches(".media-button, .thumbnail-item, .thumbnail-small, .thumbnail-lg")
        && !button.closest(".thumbnails");
    }});
    return candidates.length ? candidates[candidates.length - 1] : null;
  }};
  const findCloseButton = (root) => {{
    if (!(root instanceof Element)) return null;
    if (root.matches?.("#hy-left-gallery .preview, #hy-right-gallery .preview")) {{
      const previewClose = findInlinePreviewCloseButton(root);
      if (previewClose) return previewClose;
    }}
    return Array.from(root.querySelectorAll("button")).find((button) => {{
      const label = getDialogControlLabel(button);
      return label === "x" || label === "×" || label.includes("close");
    }}) || null;
  }};
  const hideDialogThumbStrip = (dialogRoot) => {{
    const thumbs = getDialogThumbImages(dialogRoot);
    if (!thumbs.length) return;
    const parentCounts = new Map();
    for (const img of thumbs) {{
      const host = img.closest("button, [role='button'], .thumbnail-item, .gallery-item") || img;
      host.style.display = "none";
      host.style.pointerEvents = "none";
      host.setAttribute("aria-hidden", "true");
      if (host.parentElement) {{
        parentCounts.set(host.parentElement, (parentCounts.get(host.parentElement) || 0) + 1);
      }}
    }}
    for (const [parent, count] of parentCounts.entries()) {{
      if (count >= Math.max(3, Math.ceil(thumbs.length * 0.5))) {{
        parent.style.display = "none";
        parent.setAttribute("aria-hidden", "true");
      }}
    }}
  }};
  const disablePreviewDialogNavigation = (dialogRoot) => {{
    const root = dialogRoot || document.querySelector('[role="dialog"], [aria-modal="true"]');
    if (!root) return;
    hideDialogThumbStrip(root);
    const mainImg = getActiveDialogImage();
    if (mainImg) {{
      const host = mainImg.closest("button, [role='button']") || mainImg;
      host.style.pointerEvents = "none";
      host.style.cursor = "default";
      mainImg.style.pointerEvents = "none";
      mainImg.style.cursor = "default";
    }}
  }};
  const isDialogMainImageClick = (event, dialogRoot) => {{
    if (!event || !(dialogRoot instanceof Element)) return false;
    if (event.button !== 0) return false;
    const target = event.target;
    if (!(target instanceof Element)) return false;
    if (target.closest("button, a, input, textarea, select, label")) return false;
    const dialogImg = getActiveDialogImage();
    if (!(dialogImg instanceof HTMLImageElement)) return false;
    const rect = dialogImg.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    return (
      event.clientX >= rect.left
      && event.clientX <= rect.right
      && event.clientY >= rect.top
      && event.clientY <= rect.bottom
    );
  }};
  const shouldBlockDialogNavigationClick = (target, dialogRoot) => {{
    if (!(target instanceof Element) || !dialogRoot) return false;
    if (target === dialogRoot) return false;
    const control = target.closest("button, a");
    if (control) {{
      const label = getDialogControlLabel(control);
      const rect = control.getBoundingClientRect();
      const dialogRect = dialogRoot.getBoundingClientRect();
      if (rect.top <= dialogRect.top + 80) return false;
      if (
        label === "x"
        || label === "×"
        || label.includes("close")
        || label.includes("download")
        || label.includes("share")
        || label.includes("fullscreen")
        || label.includes("expand")
      ) {{
        return false;
      }}
    }}
    return true;
  }};
  const parseCssColor = (value) => {{
    const match = String(value || "").match(/rgba?\\(([^)]+)\\)/i);
    if (!match) return null;
    const parts = match[1].split(",").map((part) => Number.parseFloat(part.trim()));
    if (parts.length < 3 || parts.some((part) => !Number.isFinite(part))) return null;
    return {{
      r: parts[0],
      g: parts[1],
      b: parts[2],
      a: Number.isFinite(parts[3]) ? parts[3] : 1,
    }};
  }};
  const thumbSelectionScore = (img) => {{
    const host = img.closest("button, [role='button'], .thumbnail-item, .gallery-item") || img;
    const hostStyle = window.getComputedStyle(host);
    const imgStyle = window.getComputedStyle(img);
    const hostClasses = String(host.className || "").toLowerCase();
    const imgClasses = String(img.className || "").toLowerCase();
    let score = 0;
    if (host.getAttribute("aria-selected") === "true" || img.getAttribute("aria-selected") === "true") score += 1000;
    if (host.getAttribute("aria-current") === "true" || img.getAttribute("aria-current") === "true") score += 1000;
    if (host.dataset.selected === "true" || img.dataset.selected === "true") score += 1000;
    if (/(^|\\s)(selected|active|current)(\\s|$)/.test(hostClasses)) score += 300;
    if (/(^|\\s)(selected|active|current)(\\s|$)/.test(imgClasses)) score += 300;
    const outlineWidth = Number.parseFloat(hostStyle.outlineWidth || "0") || 0;
    const borderWidth = Number.parseFloat(hostStyle.borderTopWidth || "0") || 0;
    const boxShadow = `${{hostStyle.boxShadow || ""}} ${{imgStyle.boxShadow || ""}}`.toLowerCase();
    if (outlineWidth > 0) score += outlineWidth * 25;
    if (borderWidth > 0) score += borderWidth * 15;
    if (boxShadow && boxShadow !== "none") score += 30;
    const colors = [
      parseCssColor(hostStyle.outlineColor),
      parseCssColor(hostStyle.borderTopColor),
      parseCssColor(hostStyle.boxShadow),
      parseCssColor(imgStyle.outlineColor),
      parseCssColor(imgStyle.borderTopColor),
      parseCssColor(imgStyle.boxShadow),
    ].filter(Boolean);
    for (const color of colors) {{
      const blueBias = color.b - Math.max(color.r, color.g);
      const cyanBias = Math.min(color.g, color.b) - color.r;
      if (blueBias > 20) score += 40 + blueBias;
      if (cyanBias > 20) score += 40 + cyanBias;
      score += (color.a || 0) * 10;
    }}
    return score;
  }};
  const getSelectedDialogThumbIndex = (dialogRoot) => {{
    const thumbs = getDialogThumbImages(dialogRoot);
    if (!thumbs.length) return -1;
    let bestIndex = -1;
    let bestScore = -1;
    thumbs.forEach((img, idx) => {{
      const score = thumbSelectionScore(img);
      if (score > bestScore) {{
        bestScore = score;
        bestIndex = idx;
      }}
    }});
    return bestScore > 0 ? bestIndex : -1;
  }};
  const syncDialogPreviewTarget = () => {{
    const dialogRoot = document.querySelector('[role="dialog"], [aria-modal="true"]');
    const dialogImg = getActiveDialogImage();
    const markedState = readMarkedState();
    if (!activeDialogSelection.side) {{
      updateDialogSelectionFromFname(markedState, markedState.preview || "");
    }}
    if (!dialogRoot || !dialogImg) {{
      activeDialogPreviewFname = "";
      activeDialogSelection = {{ side: "", index: -1 }};
      return;
    }}
    disablePreviewDialogNavigation(dialogRoot);
    const fname = extractFnameFromDialogText(dialogRoot, markedState) || extractFnameFromDialogImage(dialogImg, markedState);
    if (!fname || fname === activeDialogPreviewFname) return;
    updateDialogSelectionFromFname(markedState, fname);
    activeDialogPreviewFname = fname;
    pushPreviewFname(fname);
  }};
  const resolveDialogPreviewFnameForAction = (dialogRoot, markedState) => {{
    const fromText = extractFnameFromDialogText(dialogRoot, markedState);
    if (fromText) {{
      updateDialogSelectionFromFname(markedState, fromText);
      return fromText;
    }}
    const dialogImg = getActiveDialogImage();
    const fromImage = extractFnameFromDialogImage(dialogImg, markedState);
    if (fromImage) {{
      updateDialogSelectionFromFname(markedState, fromImage);
      return fromImage;
    }}
    const selectedThumbIndex = getSelectedDialogThumbIndex(dialogRoot);
    if (selectedThumbIndex >= 0) {{
      let side = activeDialogSelection.side || "";
      if (!side) {{
        side = updateDialogSelectionFromFname(markedState, markedState.preview || "") ? activeDialogSelection.side : "";
      }}
      const order = getDialogOrder(markedState, side);
      if (selectedThumbIndex < order.length) {{
        activeDialogSelection = {{ side, index: selectedThumbIndex }};
        return order[selectedThumbIndex] || "";
      }}
    }}
    const order = getDialogOrder(markedState, activeDialogSelection.side);
    if (order.length && activeDialogSelection.index >= 0 && activeDialogSelection.index < order.length) {{
      return order[activeDialogSelection.index] || "";
    }}
    return activeDialogPreviewFname || markedState.preview || "";
  }};
  const closePreviewDialog = (dialogRoot) => {{
    const root = dialogRoot || document.querySelector('[role="dialog"], [aria-modal="true"]');
    if (!root) return false;
    const closeButton = findCloseButton(root);
    if (closeButton) {{
      closeButton.click();
      return true;
    }}
    const escapeEvent = new KeyboardEvent("keydown", {{
      key: "Escape",
      code: "Escape",
      keyCode: 27,
      which: 27,
      bubbles: true,
    }});
    document.dispatchEvent(escapeEvent);
    return true;
  }};
  const hookDialogActionHandoff = () => {{
    if (document.body.dataset.hyDialogActionHooked) return;
    document.addEventListener("click", (event) => {{
      const target = event.target;
      if (!(target instanceof Element)) return;
      const actionRoot = target.closest("#hy-fit-threshold, #hy-move-right, #hy-move-left");
      if (!actionRoot) return;
      const dialogRoot = document.querySelector('[role="dialog"], [aria-modal="true"]');
      if (!dialogRoot) return;
      const markedState = readMarkedState();
      const fname = resolveDialogPreviewFnameForAction(dialogRoot, markedState);
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      closePreviewDialog(dialogRoot);
      pushThumbAction(`dialogactionjson:${{JSON.stringify({{
        action: actionRoot.id || "",
        fname: fname || "",
        ts: Date.now(),
      }})}}`);
      setTimeout(scheduleRepaint, 40);
      setTimeout(scheduleRepaint, 140);
    }}, true);
    document.body.dataset.hyDialogActionHooked = "1";
  }};
  const getHoverScores = (markedState) => {{
    if (!activeHoverInfo || !activeHoverInfo.fname) return null;
    const lookup = markedState.score_lookup || {{}};
    const scored = lookup[activeHoverInfo.fname] || null;
    if (scored) return scored;
    if (Number.isFinite(activeHoverInfo.main)) {{
      return {{
        main: activeHoverInfo.main,
        neg: Number.isFinite(activeHoverInfo.neg) ? activeHoverInfo.neg : null,
      }};
    }}
    return null;
  }};
  const syncHistogramHoverLine = () => {{
    const root = document.getElementById("hy-hist");
    if (!root) return;
    root.style.position = "relative";
    const markedState = readMarkedState();
    const geom = markedState.hist_geom || null;
    const img = root.querySelector("img");
    let line = root.querySelector(".hy-hover-line-main");
    if (!line) {{
      line = document.createElement("div");
      line.className = "hy-hover-line hy-hover-line-main";
      root.appendChild(line);
    }}
    let negLine = root.querySelector(".hy-hover-line-neg");
    if (!negLine) {{
      negLine = document.createElement("div");
      negLine.className = "hy-hover-line hy-hover-line-neg";
      root.appendChild(negLine);
    }}
    const hoverScores = getHoverScores(markedState);
    if (!geom || !img || !hoverScores || !Number.isFinite(hoverScores.main)) {{
      line.style.opacity = "0";
      negLine.style.opacity = "0";
      return;
    }}
    const usesPositiveSimilarityChart = ["PromptMatch", "Similarity", "SamePerson", "TagMatch", "LM Search"].includes(geom.method);
    const chartLo = usesPositiveSimilarityChart ? geom.pos_lo : geom.lo;
    const chartHi = usesPositiveSimilarityChart ? geom.pos_hi : geom.hi;
    if (!Number.isFinite(chartLo) || !Number.isFinite(chartHi) || Math.abs(chartHi - chartLo) < 1e-9) {{
      line.style.opacity = "0";
      negLine.style.opacity = "0";
      return;
    }}
    const scaleX = img.clientWidth / Math.max(geom.W || 1, 1);
    const scaleY = img.clientHeight / Math.max(geom.H || 1, 1);
    const chartX = (geom.PAD_L + (((hoverScores.main - chartLo) / (chartHi - chartLo)) * (geom.W - geom.PAD_L - geom.PAD_R))) * scaleX;
    const clampedX = Math.max(geom.PAD_L * scaleX, Math.min((geom.W - geom.PAD_R) * scaleX, chartX));
    const chartTop = (geom.PAD_TOP || 0) * scaleY;
    const chartHeight = (usesPositiveSimilarityChart ? geom.CH : (geom.H - geom.PAD_TOP - geom.PAD_BOT)) * scaleY;
    line.style.left = `${{Math.round(img.offsetLeft + clampedX)}}px`;
    line.style.top = `${{Math.round(img.offsetTop + chartTop)}}px`;
    line.style.height = `${{Math.max(12, Math.round(chartHeight))}}px`;
    line.style.opacity = "1";
    negLine.style.opacity = "0";
    if (usesPositiveSimilarityChart && geom.has_neg && Number.isFinite(hoverScores.neg) && Number.isFinite(geom.neg_lo) && Number.isFinite(geom.neg_hi) && Math.abs(geom.neg_hi - geom.neg_lo) >= 1e-9) {{
      const negX = (geom.PAD_L + (((hoverScores.neg - geom.neg_lo) / (geom.neg_hi - geom.neg_lo)) * (geom.W - geom.PAD_L - geom.PAD_R))) * scaleX;
      const clampedNegX = Math.max(geom.PAD_L * scaleX, Math.min((geom.W - geom.PAD_R) * scaleX, negX));
      const negTop = (geom.PAD_TOP + geom.CH + geom.GAP) * scaleY;
      negLine.style.left = `${{Math.round(img.offsetLeft + clampedNegX)}}px`;
      negLine.style.top = `${{Math.round(img.offsetTop + negTop)}}px`;
      negLine.style.height = `${{Math.max(12, Math.round(geom.CH * scaleY))}}px`;
      negLine.style.opacity = "1";
    }}
  }};
  const scheduleRepaint = () => {{
    // Gallery DOM mutates a moment after clicks; repaint a few times to catch the final layout.
    ensureThumbBehavior();
    hookHistogramResize();
    syncHistogramHoverLine();
    syncDialogPreviewTarget();
    disablePreviewDialogNavigation();
    for (const timer of repaintTimers) clearTimeout(timer);
    repaintTimers = [
      setTimeout(ensureThumbBehavior, 40),
      setTimeout(syncHistogramHoverLine, 60),
      setTimeout(syncDialogPreviewTarget, 80),
      setTimeout(disablePreviewDialogNavigation, 100),
      setTimeout(ensureThumbBehavior, 140),
      setTimeout(syncHistogramHoverLine, 170),
      setTimeout(syncDialogPreviewTarget, 200),
      setTimeout(disablePreviewDialogNavigation, 220),
      setTimeout(ensureThumbBehavior, 320),
      setTimeout(syncHistogramHoverLine, 350),
      setTimeout(syncDialogPreviewTarget, 380),
      setTimeout(disablePreviewDialogNavigation, 400),
    ];
  }};
  const ensureThumbBehavior = () => {{
    // Re-apply green/red borders after every gallery rerender and preview change.
    const markedState = readMarkedState();
    const allMarked = new Set([...(markedState.left || []), ...(markedState.right || [])]);
    const heldSet = new Set(markedState.held || []);
    const clearDropTarget = (root) => {{
      if (!root) return;
      root.style.boxShadow = "";
      root.style.borderColor = "";
    }};
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
              activeDialogSelection = {{ side, index }};
              activeDialogPreviewFname = card.dataset.hyFname || "";
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
      if (!root.dataset.hyDropHooked) {{
        root.addEventListener("dragenter", (event) => {{
          if (!activeDrag || activeDrag.side === side) return;
          event.preventDefault();
          root.style.boxShadow = "0 0 0 3px rgba(125, 168, 255, 0.35)";
          root.style.borderColor = "#7da8ff";
        }});
        root.addEventListener("dragover", (event) => {{
          if (!activeDrag || activeDrag.side === side) return;
          event.preventDefault();
          if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
        }});
        root.addEventListener("dragleave", (event) => {{
          if (!root.contains(event.relatedTarget)) {{
            clearDropTarget(root);
          }}
        }});
        root.addEventListener("drop", (event) => {{
          if (!activeDrag || activeDrag.side === side) return;
          event.preventDefault();
          clearDropTarget(root);
          pushThumbAction(`dropjson:${{JSON.stringify({{
            source_side: activeDrag.side,
            source_index: activeDrag.index,
            target_side: side,
            fnames: activeDrag.fnames || [],
            ts: Date.now(),
          }})}}`);
          activeDrag = null;
          scheduleRepaint();
        }});
        root.dataset.hyDropHooked = "1";
      }}
      const thumbButtons = Array.from(root.querySelectorAll("button")).filter((btn) => {{
        const img = btn.querySelector("img");
        const inDialog = !!btn.closest('[role="dialog"], [aria-modal="true"]');
        const hasCaption = !!(btn.querySelector(".caption-label span") || btn.querySelector('[class*="caption"]'));
        return !inDialog && !!img && hasCaption;
      }});
      thumbButtons.forEach((card, index) => {{
        const img = card.querySelector("img");
        if (!img) return;
        const captionEl = card.querySelector(".caption-label span") || card.querySelector('[class*="caption"]');
        const captionText = captionEl ? (captionEl.textContent || "") : "";
        const held = captionText.includes("✋ ");
        const parts = captionText.split("|");
        const fname = parts.length ? parts[parts.length - 1].trim() : "";
        const marked = (markedState[side] || []).includes(fname);
        card.style.position = "relative";
        card.style.boxSizing = "border-box";
        card.style.outline = marked ? "3px solid #58bb73" : (held ? "3px solid #dd3322" : "");
        card.style.outlineOffset = (marked || held) ? "-3px" : "";
        card.style.boxShadow = marked ? "inset 0 0 0 1px rgba(88,187,115,0.35)" : "";
        card.style.cursor = "grab";
        card.draggable = true;
        card.dataset.hySide = side;
        card.dataset.hyIndex = String(index);
        card.dataset.hyFname = fname;
        const scoreLookup = markedState.score_lookup || {{}};
        const hoverScores = scoreLookup[fname] || null;
        const hoverMain = hoverScores && Number.isFinite(hoverScores.main)
          ? hoverScores.main
          : parseMainScoreFromCaption(captionText);
        card.dataset.hyHoverMain = Number.isFinite(hoverMain) ? String(hoverMain) : "";
        card.dataset.hyHoverNeg = hoverScores && Number.isFinite(hoverScores.neg) ? String(hoverScores.neg) : "";
        if (!card.dataset.hyHoverHooked) {{
          card.addEventListener("mouseenter", (event) => {{
            const target = event.currentTarget;
            if (!target) return;
            const main = Number.parseFloat(target.dataset.hyHoverMain || "");
            const neg = Number.parseFloat(target.dataset.hyHoverNeg || "");
            activeHoverInfo = {{
              fname: target.dataset.hyFname || "",
              main: Number.isFinite(main) ? main : null,
              neg: Number.isFinite(neg) ? neg : null,
            }};
            syncHistogramHoverLine();
          }});
          card.addEventListener("mouseleave", () => {{
            activeHoverInfo = null;
            syncHistogramHoverLine();
          }});
          card.dataset.hyHoverHooked = "1";
        }}
        if (!card.dataset.hyDragHooked) {{
          card.addEventListener("dragstart", (event) => {{
            const dragSide = card.dataset.hySide || side;
            const dragIndex = Number.parseInt(card.dataset.hyIndex || String(index), 10);
            const dragFname = card.dataset.hyFname || fname;
            const currentMarkedState = readMarkedState();
            const markedNames = Array.isArray(currentMarkedState[dragSide]) ? currentMarkedState[dragSide] : [];
            const isMarked = markedNames.includes(dragFname);
            const dragNames = isMarked && markedNames.length > 1 ? markedNames.slice() : [dragFname];
            activeDrag = {{ side: dragSide, index: dragIndex, fnames: dragNames }};
            card.style.opacity = "0.55";
            if (event.dataTransfer) {{
              event.dataTransfer.effectAllowed = "move";
              event.dataTransfer.setData("text/plain", dragNames.join("\\n"));
            }}
          }});
          card.addEventListener("dragend", () => {{
            card.style.opacity = "";
            activeDrag = null;
            clearDropTarget(document.getElementById("hy-left-gallery"));
            clearDropTarget(document.getElementById("hy-right-gallery"));
          }});
          card.dataset.hyDragHooked = "1";
        }}
        img.style.outline = "";
        img.style.outlineOffset = "";
      }});
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
    hookRunScoringHotkeys();
    hookPromptWeightHotkeys();
    ensureThumbBehavior();
    applyPromptMatchModelAvailability();
  }};
  const applyPromptMatchModelAvailability = () => {{
    const statusMap = readPromptMatchModelStatus();
    const knownLabels = Object.keys(statusMap || {{}});
    if (!knownLabels.length) return;
    const colors = {{
      cached: "#8fdc7e",
      download: "#e7b062",
    }};
    const applyToNode = (node) => {{
      if (!(node instanceof HTMLElement)) return;
      const text = node instanceof HTMLInputElement
        ? (node.value || "").trim()
        : ((node.textContent || "").trim());
      const entry = findPromptMatchModelStatusEntry(text, statusMap, knownLabels);
      if (!entry) return;
      paintPromptMatchModelNode(node, entry, colors);
      for (const child of node.querySelectorAll("*")) {{
        paintPromptMatchModelNode(child, entry, colors);
      }}
    }};
    const root = document.getElementById("hy-model");
    if (root) {{
      const input = root.querySelector("input");
      if (input) applyToNode(input);
      for (const node of root.querySelectorAll("span, div")) applyToNode(node);
    }}
    for (const node of document.querySelectorAll('[role="option"], [role="listbox"], [role="listbox"] *')) {{
      applyToNode(node);
    }}
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
  const hookModelStatusState = () => {{
    const root = document.getElementById("hy-model-status");
    if (!root || root.dataset.hyStateHooked) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    const repaint = () => schedulePromptMatchModelAvailability();
    input.addEventListener("input", repaint);
    input.addEventListener("change", repaint);
    root.dataset.hyStateHooked = "1";
  }};
  const hookPromptMatchModelAvailability = () => {{
    const root = document.getElementById("hy-model");
    if (!root || root.dataset.hyAvailabilityHooked) return;
    const repaint = () => schedulePromptMatchModelAvailability();
    for (const eventName of ["click", "pointerdown", "mousedown", "focusin", "focusout", "input", "change", "keydown", "keyup"]) {{
      root.addEventListener(eventName, repaint);
    }}
    const docRepaint = (event) => {{
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest('#hy-model, [role="option"], [role="listbox"]')) {{
        schedulePromptMatchModelAvailability();
      }}
    }};
    document.addEventListener("click", docRepaint, true);
    document.addEventListener("pointerup", docRepaint, true);
    document.addEventListener("keyup", docRepaint, true);
    root.dataset.hyAvailabilityHooked = "1";
  }};
  applyTooltips();
  hookMarkState();
  hookModelStatusState();
  hookPromptMatchModelAvailability();
  hookHistogramResize();
  hookPreviewDialogTracking();
  hookInlinePreviewNavigationLock();
  hookDialogActionHandoff();
  scheduleRepaint();
  new MutationObserver((mutations) => {{
    let dialogMutation = false;
    const relevantMutation = mutations.some((mutation) => {{
      const nodes = [...mutation.addedNodes, ...mutation.removedNodes];
      return nodes.some((node) => {{
        if (!(node instanceof Element)) return false;
        if (node.closest('[role="dialog"], [aria-modal="true"]') || node.matches?.('[role="dialog"], [aria-modal="true"]')) {{
          dialogMutation = true;
          return false;
        }}
        return !!node.closest('#hy-left-gallery, #hy-right-gallery, #hy-hist, .sidebar-box');
      }});
    }});
    if (dialogMutation) {{
      setTimeout(syncDialogPreviewTarget, 30);
      setTimeout(syncDialogPreviewTarget, 120);
      setTimeout(disablePreviewDialogNavigation, 45);
      setTimeout(disablePreviewDialogNavigation, 135);
    }}
    if (!relevantMutation) return;
    applyTooltips();
    hookMarkState();
    hookModelStatusState();
    hookPromptMatchModelAvailability();
    hookHistogramResize();
    hookPreviewDialogTracking();
    hookInlinePreviewNavigationLock();
    hookDialogActionHandoff();
    scheduleRepaint();
  }}).observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>
"""

    tooltips = {
        "hy-method": "Choose whether to sort by PromptMatch or ImageReward.",
        "hy-folder": "Path to the image folder you want to score. You can paste a full folder path here.",
        "hy-load-folder": "Load the current folder into unscored browse mode and prepare proxies for faster gallery display.",
        "hy-model": "Choose the PromptMatch model. Cached models are shown in green text, and models that still need a first download are shown in amber.",
        "hy-llm-model": "Choose the PromptMatch model used for the fast shortlist stage before the vision-LLM rerank pass.",
        "hy-llm-backend": "Choose the local vision-language backend used to rerank shortlisted images at a deeper semantic level.",
        "hy-llm-prompt": "Natural-language search request for the hybrid PromptMatch plus vision-LLM rerank mode.",
        "hy-llm-shortlist": "How many top PromptMatch candidates should be sent into the slower vision-LLM rerank stage.",
        "hy-pos": "Describe what you want to find in the images. PromptMatch also supports fragment weights like beautiful (blonde:1.2) woman. Select text and press Ctrl +/- to wrap or adjust it by 0.1. Press Ctrl+Enter to run scoring.",
        "hy-neg": "Optional PromptMatch negative prompt that counts against a match. Weighted fragments like (text:1.3) also work here. Select text and press Ctrl +/- to wrap or adjust it by 0.1. Press Ctrl+Enter to run scoring.",
        "hy-ir-pos": "Describe the style or aesthetic you want ImageReward to favor. Press Ctrl+Enter to run scoring.",
        "hy-ir-neg": "Optional experimental penalty prompt. Its score is subtracted from the positive style score. Press Ctrl+Enter to run scoring.",
        "hy-ir-weight": "How strongly the penalty prompt should reduce the final ImageReward score.",
        "hy-run-llm": "Run hybrid image search: PromptMatch first shortlists likely matches, then the local vision-language backend reranks the top candidates.",
        "hy-prompt-generator": "Choose which caption model should draft the prompt from the preview image.",
        "hy-generate-prompt": "Use the currently previewed image to draft an editable prompt with the selected caption backend.",
        "hy-find-similar": "Use the currently previewed image as the query and rank the current folder by visual similarity with the active PromptMatch model.",
        "hy-generated-prompt": "Editable scratch prompt generated from the previewed image. You can tweak it before scoring or reinsert it into the active method.",
        "hy-generated-prompt-detail": "Choose whether the caption backend should describe only the core facts, a balanced amount of detail, or the full detailed prompt.",
        "hy-insert-prompt": "Copy the editable generated prompt back into the active method's main prompt field.",
        "hy-promptgen-status": "Small status readout for prompt generation.",
        "hy-run-pm": "Score the current folder with the selected method and prompts. Ctrl+Enter from a PromptMatch prompt box does the same.",
        "hy-run-ir": "Score the current folder with the selected method and prompts. Ctrl+Enter from an ImageReward prompt box does the same.",
        "hy-main-slider": "Minimum score needed to stay in SELECTED. Raising it keeps fewer images. Click the histogram to set it visually.",
        "hy-aux-slider": "Maximum negative-prompt similarity allowed in PromptMatch. Lowering it rejects more images that match the negative prompt.",
        "hy-main-mid": "Set the main threshold to 50% of the current score range.",
        "hy-aux-mid": "Set the negative threshold to 50% of the current negative-score range.",
        "hy-keep-pm-thresholds": "Keep the exact PromptMatch thresholds when rerunning with changed prompts. They reset automatically if folder or PromptMatch model changes.",
        "hy-keep-ir-thresholds": "Keep the exact ImageReward threshold when rerunning with changed prompts. It resets automatically if the folder changes.",
        "hy-percentile": "Automatically set the main threshold to keep roughly the top N percent, or show the N most similar images in FSI mode.",
        "hy-percentile-mid": "Reset the helper control to its default value for the current mode.",
        "hy-zoom-ui": "Choose how many thumbnails appear per row in both galleries.",
        "hy-use-proxy-display": "Show gallery images from cached proxies for faster browsing on large folders.",
        "hy-hist": "Histogram of current scores. In PromptMatch, click the top chart for positive threshold or bottom chart for negative threshold.",
        "hy-export": "COPY the current split into two SELECTED / REJECTED output folders inside source folder.",
        "hy-left-gallery": "Images currently in the left bucket. Click one to select it, Shift+click to mark, or drag an image to the other gallery.",
        "hy-right-gallery": "Images currently in the right bucket. Click one to select it, Shift+click to mark, or drag an image to the other gallery.",
        "hy-export-left-enabled": "Include the left bucket in the next export run.",
        "hy-export-right-enabled": "Include the right bucket in the next export run.",
        "hy-export-move-enabled": "Move files into the export folders instead of copying them. Disabled by default.",
        "hy-export-left-name": "Editable export folder name for the left bucket. Export writes directly into source_folder/<name>.",
        "hy-export-right-name": "Editable export folder name for the right bucket. Export writes directly into source_folder/<name>.",
        "hy-move-right": "Move all marked SELECTED images into REJECTED as manual overrides.",
        "hy-move-left": "Move all marked REJECTED images into SELECTED as manual overrides.",
        "hy-pin-selected": "Pin the currently marked or previewed images to their current bucket as manual overrides without moving them.",
        "hy-fit-threshold": "Adjust the score threshold just enough so the marked images flip to the other bucket. Uses the previewed image if nothing is marked.",
        "hy-clear-status": "Remove manual override status from all marked images so they snap back to their scored bucket.",
        "hy-clear-all-status": "Remove manual override status from every pinned image in the current folder so everything snaps back to the scored buckets.",
    }

    def ensure_imagereward_model():
        if state["ir_model"] is None:
            state["ir_model"] = get_imagereward_utils().load(
                "ImageReward-v1.0",
                download_root=get_cache_config()["imagereward_dir"],
            )
        return state["ir_model"]

    def release_inactive_gpu_models(target_method):
        released = []

        def _clear_torch_model(model):
            if model is None:
                return
            try:
                model.to("cpu")
            except Exception:
                pass

        if target_method != METHOD_IMAGEREWARD and state.get("ir_model") is not None:
            _clear_torch_model(state.get("ir_model"))
            state["ir_model"] = None
            released.append("ImageReward")

        if target_method != METHOD_SAMEPERSON and state.get("face_backend") is not None:
            state["face_backend"] = None
            state["face_backend_builder"] = None
            state["face_backend_worker_local"] = None
            released.append("InsightFace")

        if target_method != METHOD_TAGMATCH and state.get("tagmatch_backend") is not None:
            state["tagmatch_backend"] = None
            released.append("TagMatch")

        if state.get("prompt_backend_cache"):
            for backend_name, cached in list((state.get("prompt_backend_cache") or {}).items()):
                if isinstance(cached, dict):
                    _clear_torch_model(cached.get("model"))
                released.append(backend_name)
            state["prompt_backend_cache"] = {}
            # Drop loop variables so CPython's refcount immediately reaches zero for
            # llama_cpp Llama objects (which have no "model" key and are freed by __del__).
            try:
                del backend_name, cached
            except NameError:
                pass

        if released:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[GPU] Released inactive models: {', '.join(released)}")

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
            cache_dir=get_cache_config()["huggingface_dir"],
        )
        model = Florence2ForConditionalGeneration.from_pretrained(
            FLORENCE_MODEL_ID,
            dtype=dtype,
            local_files_only=local_files_only,
            cache_dir=get_cache_config()["huggingface_dir"],
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
            cache_dir=get_cache_config()["huggingface_dir"],
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            JOYCAPTION_MODEL_ID,
            dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
            cache_dir=get_cache_config()["huggingface_dir"],
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
            from llama_cpp._utils import suppress_stdout_stderr
            import llama_cpp.llama_cpp as llama_cpp_backend
        except ImportError as exc:
            raise RuntimeError(
                "JoyCaption GGUF support is not installed.\n"
                f"Run: {JOYCAPTION_GGUF_SETUP_HINT}"
            ) from exc

        if device == "cuda":
            supports_gpu_offload = getattr(llama_cpp_backend, "llama_supports_gpu_offload", None)
            if callable(supports_gpu_offload) and not supports_gpu_offload():
                raise RuntimeError(
                    "JoyCaption GGUF is installed with a CPU-only llama-cpp-python build.\n"
                    "Reinstall the GGUF runtime with CUDA enabled by rerunning setup with:\n"
                    f"{JOYCAPTION_GGUF_SETUP_HINT}"
                )

        local_files_only = describe_prompt_generator_source(PROMPT_GENERATOR_JOYCAPTION_GGUF) == "disk cache"
        print(f"[JoyCaption GGUF] Loading {JOYCAPTION_GGUF_REPO_ID} / {JOYCAPTION_GGUF_FILENAME} …")
        model_path = hf_hub_download(
            repo_id=JOYCAPTION_GGUF_REPO_ID,
            filename=JOYCAPTION_GGUF_FILENAME,
            local_files_only=local_files_only,
            cache_dir=get_cache_config()["huggingface_dir"],
        )
        mmproj_path = hf_hub_download(
            repo_id=JOYCAPTION_GGUF_REPO_ID,
            filename=JOYCAPTION_GGUF_MMPROJ_FILENAME,
            local_files_only=local_files_only,
            cache_dir=get_cache_config()["huggingface_dir"],
        )
        with suppress_stdout_stderr(disable=False):
            chat_handler = Llava15ChatHandler(
                clip_model_path=mmproj_path,
                verbose=False,
            )
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                chat_format="llava-1-5",
                n_ctx=2048,
                n_batch=2048,
                n_ubatch=512,
                n_threads=max(1, (os.cpu_count() or 1) - 1),
                n_threads_batch=max(1, (os.cpu_count() or 1) - 1),
                n_gpu_layers=-1 if device == "cuda" else 0,
                flash_attn=(device == "cuda"),
                verbose=False,
            )
        state["prompt_backend_cache"][PROMPT_GENERATOR_JOYCAPTION_GGUF] = {
            "llm": llm,
            "model_path": model_path,
            "mmproj_path": mmproj_path,
        }
        print("[JoyCaption GGUF] Ready.")
        return llm

    def tagmatch_prepare_image(image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        # WD tagger standard: pad to square with white background, then resize.
        w, h = image.size
        max_dim = max(w, h)
        canvas = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        canvas.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
        canvas = canvas.resize((TAGMATCH_WD_IMAGE_SIZE, TAGMATCH_WD_IMAGE_SIZE), Image.Resampling.BICUBIC)
        # WD tagger (SmilingWolf wdv3): raw float32 [0, 255], BGR channel order.
        # Not [0,1] and not RGB — wrong range/order produces near-zero logits → all scores ≈ 50.
        arr = np.array(canvas, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB → BGR
        return arr  # (448, 448, 3) NHWC, BGR, [0, 255]

    def ensure_tagmatch_model():
        cached = state.get("tagmatch_backend")
        if cached is not None:
            return cached

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime-gpu is required for TagMatch scoring.\n"
                "It should already be installed. Run the setup script again if missing."
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required for TagMatch model download.") from exc

        import csv as _csv

        print(f"[TagMatch] Loading {TAGMATCH_WD_REPO_ID} / {TAGMATCH_WD_MODEL_FILE} …")
        model_path = hf_hub_download(
            repo_id=TAGMATCH_WD_REPO_ID,
            filename=TAGMATCH_WD_MODEL_FILE,
            cache_dir=get_cache_config()["huggingface_dir"],
        )
        tags_path = hf_hub_download(
            repo_id=TAGMATCH_WD_REPO_ID,
            filename=TAGMATCH_WD_TAGS_FILE,
            cache_dir=get_cache_config()["huggingface_dir"],
        )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        print(f"[TagMatch] ONNX input tensor name: {input_name!r}")

        tags = []
        with open(tags_path, newline="", encoding="utf-8") as _f:
            reader = _csv.DictReader(_f)
            for row in reader:
                tags.append(row["name"].lower())

        backend = {"session": session, "tags": tags, "input_name": input_name}
        state["tagmatch_backend"] = backend
        print(f"[TagMatch] Ready — {len(tags)} tags loaded.")
        return backend

    def score_tagmatch_folder(image_paths, query_tags_str, progress):
        backend = ensure_tagmatch_model()
        session = backend["session"]
        tags = backend["tags"]
        input_name = backend["input_name"]

        image_signature = get_image_paths_signature(image_paths)
        can_reuse = (
            state.get("tagmatch_cached_signature") == image_signature
            and state.get("tagmatch_cached_tag_vectors") is not None
            and state.get("tagmatch_cached_feature_paths") is not None
        )

        if not can_reuse:
            batch_size = get_auto_batch_size(device, mode="tagmatch")
            print(f"[TagMatch] Running inference on {len(image_paths)} images "
                  f"(batch size {batch_size})")
            tag_vectors = {}
            total = len(image_paths)
            for batch_start in range(0, total, batch_size):
                batch_paths = image_paths[batch_start:batch_start + batch_size]
                batch_tensors = []
                batch_valid_paths = []
                for p in batch_paths:
                    disp = state.get("proxy_map", {}).get(p, p)
                    try:
                        with Image.open(disp) as src:
                            arr = tagmatch_prepare_image(src)
                        batch_tensors.append(arr)
                        batch_valid_paths.append(p)
                    except Exception:
                        tag_vectors[p] = {}
                if batch_tensors:
                    try:
                        batch_np = np.stack(batch_tensors, axis=0)
                        raw_out = session.run(None, {input_name: batch_np})[0]  # (B, num_tags), probabilities [0,1]
                        for i, p in enumerate(batch_valid_paths):
                            row = raw_out[i]
                            # numpy threshold: find indices above cutoff, then build a small dict
                            keep = np.where(row >= TAGMATCH_WD_MIN_CACHE_PROB)[0]
                            tag_vectors[p] = {tags[j]: float(row[j]) for j in keep}
                    except Exception as _e:
                        print(f"[TagMatch] Batch inference error: {_e}")
                        for p in batch_valid_paths:
                            tag_vectors[p] = {}
                done = min(batch_start + batch_size, total)
                progress(done / max(total, 1), desc=f"TagMatch inference {done}/{total}")

            state["tagmatch_cached_signature"] = image_signature
            state["tagmatch_cached_feature_paths"] = list(image_paths)
            state["tagmatch_cached_tag_vectors"] = tag_vectors
        else:
            print(f"[TagMatch] Reusing cached tag vectors for {len(image_paths)} images")
            tag_vectors = state["tagmatch_cached_tag_vectors"]

        # Score from cache using the current query tags (fast re-score on tag change)
        query_tags = [t.strip().lower() for t in (query_tags_str or "").split(",") if t.strip()]
        tag_set = set(tags)
        missing = [t for t in query_tags if t not in tag_set]
        if missing:
            print(f"[TagMatch] WARNING: these query tags are not in the model vocabulary and will score 0: {missing}")
        results = {}
        for p in image_paths:
            tv = tag_vectors.get(p, {})
            # Sum matching tag probabilities directly — no division by tag count.
            # Each matched tag contributes its full probability (0–1) × 100 to the score.
            # Having more tags in the query doesn't deflate individual matches.
            score = min(100.0, sum(tv.get(t, 0.0) for t in query_tags) * 100.0)
            results[os.path.basename(p)] = {
                "pos": score,
                "neg": None,
                "path": p,
                "failed": False,
            }
        return results

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

    def ui_visibility_updates():
        browse_mode = is_browse_mode()
        return (
            gr.update(visible=not browse_mode),
            gr.update(visible=not browse_mode),
            gr.update(visible=not browse_mode),
            gr.update(visible=not browse_mode),
            gr.update(visible=not browse_mode),
            gr.update(visible=not browse_mode),
            gr.update(visible=not browse_mode),
        )

    def selection_info():
        left_count = len(state.get("left_marked", []))
        right_count = len(state.get("right_marked", []))
        if not left_count and not right_count:
            return "Shift+click to mark multiple, or drag & drop."
        return f"Marked: **{left_count}** in SELECTED, **{right_count}** in REJECTED"

    def marked_state_json(visible_fnames=None):
        score_lookup = {}
        media_lookup = {}
        left_order = []
        right_order = []
        visible_names = set(visible_fnames or [])
        for original_path, _ in state.get("browse_items", []):
            original_base = os.path.basename(original_path)
            if visible_names and original_base not in visible_names:
                continue
            media_lookup[original_base] = original_base
            media_lookup[original_path] = original_base
            display_path = state.get("proxy_map", {}).get(original_path, original_path)
            display_base = os.path.basename(display_path)
            media_lookup[display_base] = original_base
            media_lookup[display_path] = original_base
        for fname, item in state.get("scores", {}).items():
            if visible_names and fname not in visible_names:
                continue
            original_path = item.get("path")
            if original_path:
                original_base = os.path.basename(original_path)
                media_lookup[original_base] = fname
                media_lookup[original_path] = fname
                display_path = state.get("proxy_map", {}).get(original_path, original_path)
                display_base = os.path.basename(display_path)
                media_lookup[display_base] = fname
                media_lookup[display_path] = fname
            if uses_pos_similarity_scores(state["method"]):
                if item.get("failed", False) or "pos" not in item:
                    continue
                score_lookup[fname] = {
                    "main": round(float(item["pos"]), 6),
                    "neg": round(float(item["neg"]), 6) if item.get("neg") is not None else None,
                }
            else:
                if item.get("score", -1001) <= -1000:
                    continue
                score_lookup[fname] = {
                    "main": round(float(item["score"]), 6),
                    "neg": None,
                }
        return json.dumps({
            "left": state.get("left_marked", []),
            "right": state.get("right_marked", []),
            "held": list(state.get("overrides", {}).keys()),
            "preview": state.get("preview_fname"),
            "hist_geom": state.get("hist_geom"),
            "score_lookup": score_lookup,
            "media_lookup": media_lookup,
            "left_order": left_order,
            "right_order": right_order,
        })

    def active_targets(main_threshold, aux_threshold, preview_override=None):
        left_items, right_items = build_split(
            state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold
        )
        left_names = {os.path.basename(path) for path, _ in left_items}
        right_names = {os.path.basename(path) for path, _ in right_items}
        left_marked = [name for name in state.get("left_marked", []) if name in left_names]
        right_marked = [name for name in state.get("right_marked", []) if name in right_names]
        if left_marked or right_marked:
            return left_marked, right_marked

        preview_fname = preview_override or state.get("preview_fname")
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
        _, _, main_hist_label, aux_hist_label = threshold_labels(method)
        try:
            hist_width = max(220, int(state.get("hist_width", 300)))
        except Exception:
            hist_width = 300

        def draw_axis_labels(draw, left_x, center_x, right_x, y, lo, hi):
            labels = (
                (left_x, f"{lo:.3f}", "left"),
                (center_x, f"{((lo + hi) / 2):.3f}", "center"),
                (right_x, f"{hi:.3f}", "right"),
            )
            for x, text, align in labels:
                bbox = draw.textbbox((0, 0), text)
                width = bbox[2] - bbox[0]
                if align == "center":
                    tx = x - (width / 2)
                elif align == "right":
                    tx = x - width
                else:
                    tx = x
                draw.text((int(tx), y), text, fill="#667755")

        if uses_pos_similarity_scores(method):
            if not all("pos" in item for item in scores.values()):
                state["hist_geom"] = None
                return None
            pos_vals = [v["pos"] for v in scores.values() if not v.get("failed", False)]
            neg_vals = [v["neg"] for v in scores.values() if not v.get("failed", False) and v.get("neg") is not None]
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
            W, CH = hist_width, max(60, int(130 * HIST_HEIGHT_SCALE))
            PAD_L, PAD_R = 12, 12
            PAD_TOP, PAD_BOT = max(10, int(18 * HIST_HEIGHT_SCALE)), max(14, int(22 * HIST_HEIGHT_SCALE))
            GAP = max(16, int(28 * HIST_HEIGHT_SCALE))
            n_ch = 2 if has_neg else 1
            H = PAD_TOP + n_ch * CH + (n_ch - 1) * GAP + PAD_BOT
            img = Image.new("RGB", (W, H), "#0d0d11")
            draw = ImageDraw.Draw(img)

            def draw_chart(y0, counts, lo, hi, threshold, bar_rgb, line_rgb, label, left_tint=None, right_tint=None):
                cW = W - PAD_L - PAD_R
                max_c = max(counts) if counts else 1
                bw = cW / len(counts)
                draw.rectangle([PAD_L, y0, W - PAD_R, y0 + CH], fill="#0f0f16")
                tx = PAD_L + int(((threshold - lo) / (hi - lo)) * cW)
                tx = max(PAD_L, min(W - PAD_R, tx))
                if left_tint and tx > PAD_L:
                    draw.rectangle([PAD_L, y0, tx, y0 + CH], fill=left_tint)
                if right_tint and tx < (W - PAD_R):
                    draw.rectangle([tx, y0, W - PAD_R, y0 + CH], fill=right_tint)
                for i, count in enumerate(counts):
                    if count == 0:
                        continue
                    bh = max(1, int((count / max_c) * (CH - 2)))
                    x0 = PAD_L + int(i * bw) + 1
                    x1 = PAD_L + int((i + 1) * bw) - 1
                    draw.rectangle([x0, y0 + CH - bh, x1, y0 + CH], fill=bar_rgb)
                for yy in range(y0, y0 + CH, 6):
                    draw.line([(tx, yy), (tx, min(yy + 3, y0 + CH))], fill=line_rgb, width=2)
                draw_axis_labels(draw, PAD_L, PAD_L + (cW / 2), W - PAD_R, y0 + CH + 4, lo, hi)
                draw.text((PAD_L, y0 - 14), f"{label} threshold: {threshold:.3f}", fill="#99bb88")

            draw_chart(
                PAD_TOP,
                pos_counts,
                pos_lo,
                pos_hi,
                main_threshold,
                "#3a7a3a",
                "#aadd66",
                main_hist_label,
                left_tint="#241416",
                right_tint="#142418",
            )
            if has_neg:
                draw_chart(
                    PAD_TOP + CH + GAP,
                    neg_counts,
                    neg_lo,
                    neg_hi,
                    aux_threshold,
                    "#7a3a3a",
                    "#dd6644",
                    aux_hist_label,
                    left_tint="#142418",
                    right_tint="#241416",
                )

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
        W, H = hist_width, max(70, int(130 * HIST_HEIGHT_SCALE))
        PAD_L, PAD_R = 12, 12
        PAD_TOP, PAD_BOT = max(10, int(18 * HIST_HEIGHT_SCALE)), max(14, int(22 * HIST_HEIGHT_SCALE))
        cW = W - PAD_L - PAD_R
        img = Image.new("RGB", (W, H), "#0d0d11")
        draw = ImageDraw.Draw(img)
        draw.rectangle([PAD_L, PAD_TOP, W - PAD_R, PAD_TOP + H - PAD_BOT], fill="#0f0f16")
        tx = PAD_L + int(((main_threshold - lo) / (hi - lo)) * cW)
        tx = max(PAD_L, min(W - PAD_R, tx))
        if tx > PAD_L:
            draw.rectangle([PAD_L, PAD_TOP, tx, PAD_TOP + H - PAD_BOT], fill="#241416")
        if tx < (W - PAD_R):
            draw.rectangle([tx, PAD_TOP, W - PAD_R, PAD_TOP + H - PAD_BOT], fill="#142418")
        max_c = max(counts) if counts else 1
        bw = cW / bins
        for i, count in enumerate(counts):
            if count == 0:
                continue
            bh = max(1, int((count / max_c) * (H - PAD_BOT - PAD_TOP - 2)))
            x0 = PAD_L + int(i * bw) + 1
            x1 = PAD_L + int((i + 1) * bw) - 1
            draw.rectangle([x0, PAD_TOP + (H - PAD_BOT - bh), x1, PAD_TOP + H - PAD_BOT], fill="#3a7a3a")
        for yy in range(PAD_TOP, PAD_TOP + H - PAD_BOT, 6):
            draw.line([(tx, yy), (tx, min(yy + 3, PAD_TOP + H - PAD_BOT))], fill="#aadd66", width=2)
        draw_axis_labels(draw, PAD_L, PAD_L + (cW / 2), W - PAD_R, PAD_TOP + H - PAD_BOT + 4, lo, hi)
        draw.text((PAD_L, PAD_TOP - 14), f"{main_hist_label}: {main_threshold:.3f}", fill="#99bb88")
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
        zoom_columns = int(state.get("zoom_columns", 5))
        if is_browse_mode():
            browse_items = list(state.get("browse_items", []))
            left_names = {os.path.basename(path) for path, _ in browse_items}
            left_order = [os.path.basename(path) for path, _ in browse_items]
            state["left_marked"] = []
            state["right_marked"] = []
            state["hist_geom"] = None
            status = state.get("browse_status") or "Unscored browse mode. Preview an image to search or generate a prompt."
            return (
                f"**UNSCORED**  •  **{len(browse_items)} images**",
                gallery_update(gallery_display_items(browse_items), columns=zoom_columns),
                "",
                gallery_update([], columns=zoom_columns),
                status,
                None,
                "Unscored browse mode. Preview an image to search or generate a prompt.",
                json.dumps({
                    **json.loads(marked_state_json(left_names)),
                    "left_order": left_order,
                    "right_order": [],
                }),
            )

        remember_mode_thresholds(state["method"], main_threshold, aux_threshold)
        left_items, right_items = build_split(
            state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold
        )
        left_names = {os.path.basename(path) for path, _ in left_items}
        right_names = {os.path.basename(path) for path, _ in right_items}
        left_order = [os.path.basename(path) for path, _ in left_items]
        right_order = [os.path.basename(path) for path, _ in right_items]
        state["left_marked"] = [name for name in state.get("left_marked", []) if name in left_names]
        state["right_marked"] = [name for name in state.get("right_marked", []) if name in right_names]
        visible_names = left_names | right_names
        status = status_line(state["method"], left_items, right_items, state["scores"], state["overrides"])
        if state["method"] == METHOD_LLMSEARCH and state.get("llmsearch_backend"):
            backend_label = state.get("llmsearch_backend") or DEFAULT_LLMSEARCH_BACKEND
            status = f"LLM rerank via {backend_label}  •  {status}"
        if state["method"] == METHOD_SIMILARITY and state.get("similarity_query_fname"):
            model_label = state.get("similarity_model_label") or "PromptMatch model"
            status = f"Similarity from {state['similarity_query_fname']} via {model_label}  •  {status}"
        elif state["method"] == METHOD_SAMEPERSON and state.get("sameperson_query_fname"):
            model_label = state.get("sameperson_model_label") or FACE_MODEL_LABEL
            status = f"Same person from {state['sameperson_query_fname']} via {model_label}  •  {status}"
        return (
            f"**{len(left_items)} images**",
            gallery_update(gallery_display_items(left_items), columns=zoom_columns),
            f"**{len(right_items)} images**",
            gallery_update(gallery_display_items(right_items), columns=zoom_columns),
            status,
            render_histogram(state["method"], state["scores"], main_threshold, aux_threshold),
            selection_info(),
            json.dumps({
                **json.loads(marked_state_json(visible_names)),
                "left_order": left_order,
                "right_order": right_order,
            }),
        )

    def get_preview_image_path():
        preview_fname = state.get("preview_fname")
        if not preview_fname:
            return None, None
        if is_browse_mode():
            for path, _ in state.get("browse_items", []):
                if os.path.basename(path) == preview_fname:
                    return path, preview_fname
            return None, preview_fname
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

    def run_joycaption_prompt_variant(generator_name, image, user_prompt, detail_level, system_prompt=None, normalizer=None, max_new_tokens_override=None, stop_sequences=None, temperature_override=None, top_p_override=None, top_k_override=None):
        system_prompt = system_prompt or (
            "You are a helpful image captioner. "
            "Describe only concrete visible content and write output that is useful as a text-to-image prompt. "
            "Follow the requested output style exactly, whether it asks for short tags, a compact prompt line, or natural prose. "
            "Do not begin with meta phrases like 'This image shows', 'In this image we can see', or 'You are looking at'."
        )
        if normalizer is None:
            normalizer = lambda text: normalize_generated_prompt(
                extract_joycaption_caption(text),
                keep_prose=(detail_level == 3),
            )
        max_new_tokens = max_new_tokens_override if max_new_tokens_override is not None else joycaption_max_new_tokens(detail_level)
        temperature = temperature_override if temperature_override is not None else 0.0
        top_p = top_p_override if top_p_override is not None else 1.0
        top_k = top_k_override if top_k_override is not None else 0
        do_sample = bool(temperature and temperature > 0.0)

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
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    use_cache=True,
                )[0]

            prompt_len = inputs["input_ids"].shape[1]
            text = processor.tokenizer.decode(
                generated_ids[prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return normalizer(text)

        llm = ensure_joycaption_gguf_model()
        image = joycaption_gguf_prepare_image(image)
        data_url = image_to_data_url(image)
        try:
            from llama_cpp._utils import suppress_stdout_stderr
        except ImportError:
            suppress_stdout_stderr = None
        completion_params = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "stop": stop_sequences or ["</s>", "User:", "Assistant:"],
            "repeat_penalty": 1.1,
        }
        if top_k and top_k > 0:
            completion_params["top_k"] = top_k
        if suppress_stdout_stderr is None:
            response = llm.create_chat_completion(**completion_params)
        else:
            with suppress_stdout_stderr(disable=False):
                response = llm.create_chat_completion(**completion_params)
        try:
            text = response["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected JoyCaption GGUF response shape: {exc}") from exc
        return normalizer(text)

    def run_wd_tags_prompt_variant(image, top_n):
        """Run WD tagger on a single PIL image and return the top-N tags as a comma-separated string."""
        backend = ensure_tagmatch_model()
        session = backend["session"]
        tags = backend["tags"]
        input_name = backend["input_name"]
        arr = tagmatch_prepare_image(image)          # (448, 448, 3) float32
        batch_np = arr[np.newaxis, ...]              # (1, 448, 448, 3)
        raw_out = session.run(None, {input_name: batch_np})[0][0]  # (num_tags,)
        probs = 1.0 / (1.0 + np.exp(-raw_out))
        tag_prob_pairs = sorted(
            ((tags[j], float(probs[j])) for j in range(len(tags))),
            key=lambda x: x[1],
            reverse=True,
        )
        top_tags = [t for t, p in tag_prob_pairs[:top_n] if p >= 0.05]
        return ", ".join(top_tags)

    def generate_prompt_variant(generator_name, image, detail_level):
        _, _, detail_prompt = prompt_generator_detail_config(generator_name, detail_level)
        if generator_name == PROMPT_GENERATOR_FLORENCE:
            return run_florence_prompt_variant(image, detail_prompt)
        if generator_name == PROMPT_GENERATOR_WD_TAGS:
            return run_wd_tags_prompt_variant(image, detail_prompt)  # detail_prompt is top_n int
        return run_joycaption_prompt_variant(generator_name, image, detail_prompt, detail_level)

    class VisionLLMRerankBackend:
        def __init__(self, backend_id):
            self.backend_id = backend_id

        def describe_source(self):
            return describe_llmsearch_backend_source(self.backend_id)

        def is_available(self):
            return True

        def load(self, progress):
            if self.backend_id == PROMPT_GENERATOR_FLORENCE:
                progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
                ensure_florence_model()
            elif self.backend_id == PROMPT_GENERATOR_JOYCAPTION:
                progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
                ensure_joycaption_model()
            elif self.backend_id == PROMPT_GENERATOR_JOYCAPTION_GGUF:
                progress(0, desc=f"Loading LLM rerank backend from {self.describe_source()}: {self.backend_id}")
                ensure_joycaption_gguf_model()
            else:
                raise RuntimeError(f"Unknown LLM rerank backend: {self.backend_id}")

        def release(self):
            return None

        def uses_direct_numeric_score(self):
            return self.backend_id in {PROMPT_GENERATOR_JOYCAPTION, PROMPT_GENERATOR_JOYCAPTION_GGUF}

        def candidate_text(self, image, query_text):
            if self.backend_id == PROMPT_GENERATOR_FLORENCE:
                return run_florence_prompt_variant(image, "<MORE_DETAILED_CAPTION>")

            user_prompt = build_llmsearch_joycaption_user_prompt(query_text)
            return run_joycaption_prompt_variant(
                self.backend_id,
                image,
                user_prompt,
                2,
                system_prompt=llmsearch_joycaption_system_prompt(),
                normalizer=normalize_llmsearch_candidate_text,
            )

        def score_candidate(self, image, query_text):
            if not self.uses_direct_numeric_score():
                caption_text = self.candidate_text(image, query_text)
                return llmsearch_similarity(state["backend"].encode_text((query_text or "").strip()), caption_text), caption_text

            raw_score_text = run_joycaption_prompt_variant(
                self.backend_id,
                image,
                build_llmsearch_joycaption_user_prompt(query_text),
                1,
                system_prompt=llmsearch_joycaption_system_prompt(),
                normalizer=extract_joycaption_caption,
                max_new_tokens_override=LLMSEARCH_JOYCAPTION_MAX_NEW_TOKENS,
                stop_sequences=[" ", "\n", "\t", ",", ".", "%", "</s>", "User:", "Assistant:"],
                temperature_override=LLMSEARCH_JOYCAPTION_TEMPERATURE,
                top_p_override=LLMSEARCH_JOYCAPTION_TOP_P,
                top_k_override=LLMSEARCH_JOYCAPTION_TOP_K,
            )
            return float(extract_llmsearch_numeric_score(raw_score_text)), raw_score_text

        def score_candidates_batch(self, images, query_text):
            """Score a batch of PIL images with the HF JoyCaption backend.
            Returns a list of (score_float, raw_text) pairs, one per image.
            Falls back to sequential for non-HF backends."""
            if self.backend_id != PROMPT_GENERATOR_JOYCAPTION:
                return [self.score_candidate(img, query_text) for img in images]
            model, processor = ensure_joycaption_model()
            user_prompt = build_llmsearch_joycaption_user_prompt(query_text)
            conversation = [
                {"role": "system", "content": llmsearch_joycaption_system_prompt()},
                {"role": "user", "content": user_prompt},
            ]
            convo_string = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            n = len(images)
            inputs = processor(text=[convo_string] * n, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)
            do_sample = bool(LLMSEARCH_JOYCAPTION_TEMPERATURE and LLMSEARCH_JOYCAPTION_TEMPERATURE > 0.0)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=LLMSEARCH_JOYCAPTION_MAX_NEW_TOKENS,
                    do_sample=do_sample,
                    temperature=LLMSEARCH_JOYCAPTION_TEMPERATURE,
                    top_p=LLMSEARCH_JOYCAPTION_TOP_P,
                    top_k=LLMSEARCH_JOYCAPTION_TOP_K,
                    use_cache=True,
                )
            prompt_len = inputs["input_ids"].shape[1]
            results = []
            for i in range(n):
                text = processor.tokenizer.decode(
                    generated_ids[i][prompt_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                raw = extract_joycaption_caption(text)
                try:
                    score = float(extract_llmsearch_numeric_score(raw))
                except Exception:
                    score = 0.0
                results.append((score, raw))
            return results

    def get_llmsearch_backend(backend_id):
        return VisionLLMRerankBackend(backend_id)

    def llmsearch_caption_cache_key(backend_id, image_signature, query_text):
        return (
            str(backend_id),
            LLMSEARCH_SCORING_MODE_NUMERIC_V1,
            str(image_signature),
            normalize_prompt_text(query_text or ""),
        )

    def llmsearch_similarity(query_embedding, text):
        text = normalize_prompt_text(text)
        if not text:
            return -1.0
        text_embedding = state["backend"].encode_text(text)
        score = float((query_embedding @ text_embedding.T).squeeze().item())
        return round(score, 6)

    def score_llmsearch_candidates(candidate_paths, query_text, backend_id, image_signature, progress):
        backend = get_llmsearch_backend(backend_id)
        backend.load(progress)
        cache_key = llmsearch_caption_cache_key(backend_id, image_signature, query_text)
        caption_cache = state["llmsearch_cached_captions"].setdefault(cache_key, {})
        total = len(candidate_paths)
        results = {}
        # Batch pre-pass for HF JoyCaption: score all uncached images in chunks to improve GPU utilization.
        # The main loop below reads from cache; any image not pre-scored falls back to sequential inference.
        if backend_id == PROMPT_GENERATOR_JOYCAPTION:
            needs_inference = [
                p for p in candidate_paths
                if not (isinstance(caption_cache.get(p), dict) and caption_cache[p].get("score") is not None)
            ]
            if needs_inference:
                print(f"[LLM Search] Batch scoring {len(needs_inference)}/{total} uncached images "
                      f"(batch size {LLMSEARCH_JOYCAPTION_HF_BATCH_SIZE})")
            for batch_start in range(0, len(needs_inference), LLMSEARCH_JOYCAPTION_HF_BATCH_SIZE):
                batch_paths = needs_inference[batch_start:batch_start + LLMSEARCH_JOYCAPTION_HF_BATCH_SIZE]
                loaded = []
                for p in batch_paths:
                    disp = state.get("proxy_map", {}).get(p, p)
                    try:
                        with Image.open(disp) as src:
                            loaded.append((p, src.convert("RGB")))
                    except Exception:
                        pass  # Will be retried sequentially in main loop
                if loaded:
                    try:
                        batch_scored = backend.score_candidates_batch([img for _, img in loaded], query_text)
                        for (p, _), (sv, ct) in zip(loaded, batch_scored):
                            caption_cache[p] = {"score": float(sv), "text": ct}
                    except Exception:
                        pass  # Batch failed entirely; main loop will retry individually
                done_so_far = batch_start + len(batch_paths)
                progress(done_so_far / max(total, 1), desc=f"LLM reranking {done_so_far}/{total} via {backend_id}")
        for index, original_path in enumerate(candidate_paths, start=1):
            cached_value = caption_cache.get(original_path)
            failed_reason = None
            score_value = None
            caption_text = ""
            if isinstance(cached_value, dict):
                score_value = cached_value.get("score")
                caption_text = cached_value.get("text") or ""
            elif isinstance(cached_value, str):
                caption_text = cached_value

            if score_value is None and not caption_text and backend.uses_direct_numeric_score():
                display_path = state.get("proxy_map", {}).get(original_path, original_path)
                try:
                    with Image.open(display_path) as src_img:
                        image = src_img.convert("RGB")
                    score_value, caption_text = backend.score_candidate(image, query_text)
                    caption_cache[original_path] = {
                        "score": float(score_value),
                        "text": caption_text,
                    }
                except Exception as exc:
                    failed_reason = str(exc) or "LLM rerank backend failed."
                    score_value = 0.0
                    caption_text = ""
            elif not caption_text:
                display_path = state.get("proxy_map", {}).get(original_path, original_path)
                try:
                    with Image.open(display_path) as src_img:
                        image = src_img.convert("RGB")
                    caption_text = backend.candidate_text(image, query_text)
                    caption_cache[original_path] = caption_text
                except Exception as exc:
                    failed_reason = str(exc) or "LLM rerank backend failed."
                    caption_text = ""
            if failed_reason:
                score_value = 0.0 if backend.uses_direct_numeric_score() else -1.0
            elif score_value is None:
                try:
                    score_value = llmsearch_similarity(state["backend"].encode_text((query_text or "").strip()), caption_text)
                except Exception as exc:
                    failed_reason = str(exc) or "LLM rerank text scoring failed."
                    score_value = -1.0

            results[os.path.basename(original_path)] = {
                "pos": float(score_value),
                "neg": None,
                "path": original_path,
                "failed": bool(failed_reason),
                "caption": caption_text,
                "reason": failed_reason,
            }
            progress(index / max(total, 1), desc=f"LLM reranking {index}/{total} via {backend_id}")

        return results

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

        if cache_dir and can_reuse_proxy_map(folder_paths, image_signature):
            proxy_map = dict(state.get("proxy_map") or {})
            scoring_paths = [proxy_map.get(path, path) for path in folder_paths]
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Reusing cached ImageReward proxies for {len(folder_paths)} images")
        elif cache_dir:
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
            state["proxy_signature"] = image_signature
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
        main_label, aux_label, _, _ = threshold_labels(method)
        if method == METHOD_PROMPTMATCH:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(label=main_label, value=0.14, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(label=aux_label, visible=True, value=NEGATIVE_THRESHOLD, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                percentile_slider_update(method),
                percentile_reset_button_update(method),
                gr.update(value="PromptMatch sorts by text-image similarity. Use a positive prompt and optional negative prompt. Fragment weights like (blonde:1.2) are supported."),
            )
        if method == METHOD_LLMSEARCH:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(label=main_label, value=0.14, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(visible=False, value=NEGATIVE_THRESHOLD, label=aux_label, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                percentile_slider_update(method),
                percentile_reset_button_update(method),
                gr.update(value="LM Search uses PromptMatch to shortlist likely matches, then reranks the top candidates with a local vision-language model."),
            )
        if method == METHOD_SIMILARITY:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(label=main_label, value=0.5, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(visible=False, value=NEGATIVE_THRESHOLD, label=aux_label, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                percentile_slider_update(method),
                percentile_reset_button_update(method),
                gr.update(value="Similarity search ranks the current folder by image-image similarity using the active PromptMatch model."),
            )
        if method == METHOD_SAMEPERSON:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(label=main_label, value=0.5, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(visible=False, value=NEGATIVE_THRESHOLD, label=aux_label, minimum=PROMPTMATCH_SLIDER_MIN, maximum=PROMPTMATCH_SLIDER_MAX),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                percentile_slider_update(method),
                percentile_reset_button_update(method),
                gr.update(value=f"Same-person search ranks the current folder by face identity similarity using {FACE_MODEL_LABEL}."),
            )
        if method == METHOD_TAGMATCH:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(label=main_label, value=TAGMATCH_DEFAULT_THRESHOLD, minimum=TAGMATCH_SLIDER_MIN, maximum=TAGMATCH_SLIDER_MAX),
                gr.update(visible=False, value=NEGATIVE_THRESHOLD, label=aux_label, minimum=TAGMATCH_SLIDER_MIN, maximum=TAGMATCH_SLIDER_MAX),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                percentile_slider_update(method),
                percentile_reset_button_update(method),
                gr.update(value="TagMatch scores images by WD tagger confidence for the specified tags. Use comma-separated booru-style tags like: body_horror, extra_fingers, deformed."),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(label=main_label, value=IMAGEREWARD_THRESHOLD, minimum=IMAGEREWARD_SLIDER_MIN, maximum=IMAGEREWARD_SLIDER_MAX),
            gr.update(visible=False, value=NEGATIVE_THRESHOLD),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            percentile_slider_update(method),
            percentile_reset_button_update(method),
            gr.update(value="ImageReward sorts by aesthetic preference. Optional penalty prompt subtracts a second style score."),
        )

    def empty_result(message, method):
        _, _, _, _, main_upd, aux_upd, _, _, _, percentile_upd, percentile_mid_upd, _ = configure_controls(method)
        set_scored_mode()
        state["browse_items"] = []
        state["browse_status"] = ""
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
            percentile_upd,
            percentile_mid_upd,
            promptmatch_model_status_json(),
            *ui_visibility_updates(),
        )

    def render_view_with_controls(main_threshold, aux_threshold):
        return (*current_view(main_threshold, aux_threshold), *ui_visibility_updates())

    def load_folder_for_browse(folder, main_threshold, aux_threshold, progress=gr.Progress()):
        folder = (folder or "").strip()
        if not folder or not os.path.isdir(folder):
            state["source_dir"] = folder
            state["scores"] = {}
            state["overrides"] = {}
            state["preview_fname"] = None
            state["left_marked"] = []
            state["right_marked"] = []
            clear_preview_search_context()
            set_browse_mode([], f"Invalid folder: {folder!r}")
            return (*render_view_with_controls(main_threshold, aux_threshold),)

        image_paths = scan_image_paths(folder)
        if not image_paths:
            state["source_dir"] = folder
            state["scores"] = {}
            state["overrides"] = {}
            state["preview_fname"] = None
            state["left_marked"] = []
            state["right_marked"] = []
            clear_preview_search_context()
            set_browse_mode([], f"No images found in {folder}")
            return (*render_view_with_controls(main_threshold, aux_threshold),)

        sync_promptmatch_proxy_cache(folder)
        image_signature = get_image_paths_signature(image_paths)
        proxy_map = {}
        cache_dir = state.get("proxy_cache_dir")
        if cache_dir and can_reuse_proxy_map(image_paths, image_signature):
            proxy_map = dict(state.get("proxy_map") or {})
            progress(1.0, desc=f"Reusing cached proxies for {len(image_paths)} images")
        elif cache_dir:
            def _proxy_prep_cb(done, total, generated, reused):
                desc = f"Preparing browse proxies {done}/{total}"
                if generated or reused:
                    desc += f" ({generated} new, {reused} reused)"
                progress(done / max(total, 1), desc=desc)

            progress(0, desc=f"Preparing browse proxies 0/{len(image_paths)}")
            proxy_map, _, _ = prepare_promptmatch_proxies(
                image_paths,
                cache_dir,
                progress_cb=_proxy_prep_cb,
            )
            state["proxy_map"] = dict(proxy_map)
            state["proxy_signature"] = image_signature

        state["source_dir"] = folder
        state["scores"] = {}
        state["overrides"] = {}
        state["preview_fname"] = None
        state["left_marked"] = []
        state["right_marked"] = []
        clear_preview_search_context()
        browse_items = [(path, os.path.basename(path)) for path in image_paths]
        set_browse_mode(
            browse_items,
            f"Browse mode for {folder}. {len(image_paths)} images loaded. Preview an image to search or generate a prompt.",
        )
        return (*render_view_with_controls(main_threshold, aux_threshold),)

    def refresh_promptmatch_model_dropdown(current_model_label):
        selected = current_model_label if current_model_label in MODEL_LABELS else MODEL_LABELS[0]
        return gr.update(choices=promptmatch_model_dropdown_choices(), value=selected)

    def middle_threshold_values(method):
        if uses_pos_similarity_scores(method):
            if state["scores"]:
                _, _, pos_mid, _, _, neg_mid, has_neg = promptmatch_slider_range(state["scores"])
            else:
                pos_mid, neg_mid, has_neg = 0.14, NEGATIVE_THRESHOLD, True
            return round(float(pos_mid), 3), round(float(neg_mid), 3), bool(has_neg)
        if state["scores"]:
            _, _, main_mid = imagereward_slider_range(state["scores"])
        else:
            main_mid = IMAGEREWARD_THRESHOLD
        return round(float(main_mid), 3), NEGATIVE_THRESHOLD, False

    def ensure_promptmatch_backend_loaded(model_label, progress):
        cfg = get_model_config(model_label)
        if cfg is None:
            raise RuntimeError(f"Unknown PromptMatch model: {model_label}")
        _, backend_name, kwargs = cfg
        if label_for_backend(state["backend"]) != model_label:
            if backend_name == "openai":
                source = describe_openai_clip_source(kwargs.get("clip_model"))
            elif backend_name == "openclip":
                source = describe_openclip_source(kwargs.get("openclip_model"), kwargs.get("openclip_pretrained"))
            else:
                source = describe_siglip_source(kwargs.get("siglip_model"))
            progress(0, desc=f"Loading PromptMatch model from {source}: {model_label}")
            state["backend"] = ModelBackend(
                device,
                backend=backend_name,
                clip_cache_dir=get_cache_config()["clip_dir"],
                huggingface_cache_dir=get_cache_config()["huggingface_dir"],
                **kwargs,
            )
        else:
            progress(0, desc=f"Using loaded PromptMatch model from memory: {model_label}")
        return state["backend"]

    def ensure_face_backend_loaded(progress):
        def build_face_backend():
            face_root = get_cache_config()["insightface_dir"]
            os.environ.setdefault("INSIGHTFACE_HOME", face_root)

            try:
                import onnxruntime as ort
            except Exception as exc:
                raise RuntimeError(
                    "InsightFace runtime is missing. Re-run the normal setup script to install onnxruntime-gpu."
                ) from exc

            try:
                # Silence ONNX Runtime's info-level provider/session spam from
                # each worker-local InsightFace initialization.
                ort.set_default_logger_severity(3)
            except Exception:
                pass

            available_providers = set(ort.get_available_providers())
            if "CUDAExecutionProvider" not in available_providers:
                raise RuntimeError(
                    "InsightFace needs a CUDA-enabled onnxruntime-gpu install, but CUDAExecutionProvider is unavailable. "
                    "Re-run the normal setup script so the GPU runtime is installed."
                )

            try:
                from insightface.app import FaceAnalysis
            except Exception as exc:
                raise RuntimeError(
                    "InsightFace could not be imported. Re-run the normal setup script to install the face-search dependencies."
                ) from exc

            try:
                backend = FaceAnalysis(
                    name=FACE_MODEL_PACK,
                    root=face_root,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    allowed_modules=["detection", "recognition"],
                )
                backend.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
            except Exception as exc:
                raise RuntimeError(f"InsightFace {FACE_MODEL_PACK} could not be initialized: {exc}") from exc
            return backend

        if state.get("face_backend") is not None:
            progress(0, desc=f"Using loaded face model from memory: {FACE_MODEL_LABEL}")
            return state["face_backend"]

        progress(0, desc=f"Loading face model from {describe_insightface_source()}: {FACE_MODEL_LABEL}")
        state["face_backend_builder"] = build_face_backend
        state["face_backend_worker_local"] = threading.local()
        state["face_backend"] = build_face_backend()
        return state["face_backend"]

    def ensure_promptmatch_feature_cache(image_paths, model_label, progress, reuse_desc, encode_desc, progress_label):
        image_signature = get_image_paths_signature(image_paths)
        can_reuse_promptmatch_cache = (
            state.get("pm_cached_signature") == image_signature
            and state.get("pm_cached_model_label") == model_label
            and state.get("pm_cached_feature_paths") is not None
            and state.get("pm_cached_image_features") is not None
            and state.get("pm_cached_failed_paths") is not None
        )

        proxy_map = {}
        cache_dir = state.get("proxy_cache_dir")
        if can_reuse_promptmatch_cache:
            proxy_map = dict(state.get("proxy_map") or {})
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=reuse_desc.format(count=len(image_paths)))
        elif cache_dir and can_reuse_proxy_map(image_paths, image_signature):
            proxy_map = dict(state.get("proxy_map") or {})
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=f"Reusing cached PromptMatch proxies for {len(image_paths)} images")
        elif cache_dir:
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
            state["proxy_signature"] = image_signature
            print(f"[PromptMatch] Proxy prep complete in {cache_dir}: {generated} new, {reused} reused")

        if not can_reuse_promptmatch_cache:
            progress(PROMPTMATCH_PROXY_PROGRESS_SHARE, desc=encode_desc.format(count=len(image_paths)))

            def _cb(done, total, batch_size, oom_retry):
                label = f"{progress_label} {done}/{total} (autobatch {batch_size})"
                if oom_retry:
                    label = f"{progress_label} OOM, retrying autobatch {batch_size}"
                progress(
                    PROMPTMATCH_PROXY_PROGRESS_SHARE
                    + ((1.0 - PROMPTMATCH_PROXY_PROGRESS_SHARE) * (done / max(total, 1))),
                    desc=label,
                )

            def _proxy_resolver(original_path):
                return proxy_map.get(original_path, original_path)

            feature_paths, image_features, failed_paths = encode_all_promptmatch_images(
                image_paths,
                state["backend"],
                progress_cb=_cb,
                proxy_resolver=_proxy_resolver,
            )
            state["pm_cached_signature"] = image_signature
            state["pm_cached_model_label"] = model_label
            state["pm_cached_feature_paths"] = list(feature_paths)
            state["pm_cached_image_features"] = image_features
            state["pm_cached_failed_paths"] = list(failed_paths)

        return (
            image_signature,
            list(state.get("pm_cached_feature_paths") or []),
            state.get("pm_cached_image_features"),
            list(state.get("pm_cached_failed_paths") or []),
        )

    def choose_primary_face(faces):
        if not faces:
            return None

        def face_area(face):
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) < 4:
                return 0.0
            return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

        return max(faces, key=face_area)

    def ensure_face_feature_cache(image_paths, progress):
        image_signature = get_image_paths_signature(image_paths)
        can_reuse_face_cache = (
            state.get("face_cached_signature") == image_signature
            and state.get("face_cached_feature_paths") is not None
            and state.get("face_cached_embeddings") is not None
            and state.get("face_cached_failures") is not None
        )
        if can_reuse_face_cache:
            progress(0, desc=f"Reusing cached face embeddings for {len(image_paths)} images")
            return (
                image_signature,
                list(state.get("face_cached_feature_paths") or []),
                state.get("face_cached_embeddings"),
                dict(state.get("face_cached_failures") or {}),
            )

        ensure_face_backend_loaded(progress)
        total = len(image_paths)
        worker_count = face_embedding_worker_count(total)
        results = {}
        worker_desc = f"Extracting face embeddings 0/{total} (workers {worker_count})"
        extraction_started = time.perf_counter()
        vram_info = current_free_vram_gb()
        if vram_info is not None:
            print(
                f"[InsightFace] Using {worker_count} workers "
                f"(free_vram={vram_info[0]:.1f}GB)"
            )

        def worker_backend():
            local_state = state.get("face_backend_worker_local")
            if local_state is None:
                local_state = threading.local()
                state["face_backend_worker_local"] = local_state
            backend = getattr(local_state, "backend", None)
            if backend is None:
                builder = state.get("face_backend_builder")
                if builder is None:
                    raise RuntimeError("Face backend builder is unavailable.")
                backend = builder()
                local_state.backend = backend
            return backend

        def extract_one(image_path):
            try:
                with Image.open(image_path) as src_img:
                    rgb = src_img.convert("RGB")
                    bgr = np.asarray(rgb)[:, :, ::-1].copy()
                primary_face = choose_primary_face(worker_backend().get(bgr))
                if primary_face is None:
                    return image_path, None, "No face detected."

                embedding = getattr(primary_face, "normed_embedding", None)
                if embedding is None:
                    raw_embedding = getattr(primary_face, "embedding", None)
                    if raw_embedding is None:
                        return image_path, None, "No face embedding returned."
                    embedding = F.normalize(torch.as_tensor(raw_embedding, dtype=torch.float32), dim=0).cpu().numpy()

                return image_path, torch.as_tensor(embedding, dtype=torch.float32), None
            except Exception as exc:
                return image_path, None, str(exc) or "Face analysis failed."

        completed = 0
        progress(0, desc=worker_desc)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(extract_one, image_path): image_path for image_path in image_paths}
            for future in as_completed(future_map):
                image_path, embedding, failure = future.result()
                results[image_path] = (embedding, failure)
                completed += 1
                progress(
                    completed / max(total, 1),
                    desc=f"Extracting face embeddings {completed}/{total} (workers {worker_count})",
                )

        feature_paths = []
        face_embeddings = []
        failures = {}
        for image_path in image_paths:
            embedding, failure = results.get(image_path, (None, "Face analysis failed."))
            if embedding is None:
                failures[image_path] = failure or "Face analysis failed."
                continue
            feature_paths.append(image_path)
            face_embeddings.append(embedding)

        if face_embeddings:
            embedding_tensor = torch.stack(face_embeddings, dim=0)
            embedding_tensor = F.normalize(embedding_tensor, dim=1)
        else:
            embedding_tensor = torch.empty((0, 0), dtype=torch.float32)

        state["face_cached_signature"] = image_signature
        state["face_cached_feature_paths"] = list(feature_paths)
        state["face_cached_embeddings"] = embedding_tensor
        state["face_cached_failures"] = dict(failures)
        print(
            f"[InsightFace] Face embedding pass finished in "
            f"{(time.perf_counter() - extraction_started):.2f}s "
            f"for {total} images with {worker_count} workers"
        )
        return image_signature, feature_paths, embedding_tensor, failures

    def score_similarity_cached_features(feature_paths, image_features, failed_paths, query_path):
        if not feature_paths or image_features is None or image_features.numel() == 0:
            raise RuntimeError("No PromptMatch image embeddings are available for similarity search.")
        if query_path in failed_paths:
            raise RuntimeError(f"{os.path.basename(query_path)} could not be encoded for similarity search.")
        try:
            query_index = feature_paths.index(query_path)
        except ValueError as exc:
            raise RuntimeError(f"{os.path.basename(query_path)} is missing from the cached PromptMatch embeddings.") from exc

        results = {}
        query_feature = image_features[query_index:query_index + 1]
        sims = (image_features @ query_feature.T).squeeze(1).tolist()
        query_fname = os.path.basename(query_path)
        for original_path, score in zip(feature_paths, sims):
            fname = os.path.basename(original_path)
            results[fname] = {
                "pos": float(score),
                "neg": None,
                "path": original_path,
                "failed": False,
                "query": fname == query_fname,
            }

        for original_path in failed_paths:
            fname = os.path.basename(original_path)
            results[fname] = {
                "pos": 0.0,
                "neg": None,
                "path": original_path,
                "failed": True,
                "query": fname == query_fname,
            }
        return results

    def score_sameperson_cached_features(feature_paths, face_embeddings, failures, query_path):
        if not feature_paths or face_embeddings is None or face_embeddings.numel() == 0:
            raise RuntimeError("No face embeddings are available for same-person search.")
        if query_path in failures:
            raise RuntimeError(f"{os.path.basename(query_path)}: {failures[query_path]}")
        try:
            query_index = feature_paths.index(query_path)
        except ValueError as exc:
            raise RuntimeError(f"{os.path.basename(query_path)} is missing from the cached face embeddings.") from exc

        results = {}
        query_embedding = face_embeddings[query_index:query_index + 1]
        sims = (face_embeddings @ query_embedding.T).squeeze(1).tolist()
        query_fname = os.path.basename(query_path)
        for original_path, score in zip(feature_paths, sims):
            fname = os.path.basename(original_path)
            results[fname] = {
                "pos": float(score),
                "neg": None,
                "path": original_path,
                "failed": False,
                "query": fname == query_fname,
            }

        for failed_path, reason in failures.items():
            fname = os.path.basename(failed_path)
            results[fname] = {
                "pos": 0.0,
                "neg": None,
                "path": failed_path,
                "failed": True,
                "query": fname == query_fname,
                "reason": reason,
            }
        return results

    def score_folder(method, folder, model_label, pos_prompt, neg_prompt, ir_prompt, ir_negative_prompt, ir_penalty_weight, llm_model_label, llm_prompt, llm_backend_id, llm_shortlist_size, tagmatch_tags, main_threshold, aux_threshold, keep_pm_thresholds, keep_ir_thresholds, progress=gr.Progress()):
        # Main entrypoint for "Run scoring"; both methods converge back into current_view().
        folder = (folder or "").strip()
        if not folder or not os.path.isdir(folder):
            return empty_result(f"Invalid folder: {folder!r}", method)
        main_label, aux_label, _, _ = threshold_labels(method)
        previous_method = state.get("method")
        recalled_main, recalled_aux, has_recalled = recalled_mode_thresholds(method, main_threshold, aux_threshold)
        requested_main = float(main_threshold)
        requested_aux = float(aux_threshold)
        if previous_method != method and has_recalled:
            requested_main = recalled_main
            requested_aux = recalled_aux

        image_paths = scan_image_paths(folder)
        if not image_paths:
            return empty_result(f"No images found in {folder}", method)
        image_signature = get_image_paths_signature(image_paths)
        folder_key = normalize_folder_identity(folder)
        previous_folder = state.get("source_dir")
        previous_folder_key = normalize_folder_identity(previous_folder) if previous_folder else None
        available_names = {os.path.basename(path) for path in image_paths}
        preserved_overrides = {}
        if previous_folder_key == folder_key:
            # Keep manual pins across rescoring the same folder, but drop them once
            # the file is gone from that folder.
            preserved_overrides = {
                fname: side
                for fname, side in state.get("overrides", {}).items()
                if fname in available_names
            }
        preserve_promptmatch_thresholds = (
            bool(keep_pm_thresholds)
            and method == METHOD_PROMPTMATCH
            and state.get("last_scored_method") == METHOD_PROMPTMATCH
            and state.get("last_scored_folder_key") == folder_key
            and state.get("last_promptmatch_model_label") == model_label
        )
        preserve_imagereward_threshold = (
            bool(keep_ir_thresholds)
            and method == METHOD_IMAGEREWARD
            and state.get("last_scored_method") == METHOD_IMAGEREWARD
            and state.get("last_scored_folder_key") == folder_key
        )

        state["method"] = method
        set_scored_mode()
        sync_promptmatch_proxy_cache(folder)
        state["source_dir"] = folder
        state["overrides"] = preserved_overrides
        state["left_marked"] = []
        state["right_marked"] = []
        state["preview_fname"] = None
        clear_preview_search_context()
        release_inactive_gpu_models(method)

        if method == METHOD_LLMSEARCH:
            llm_model_label = llm_model_label if llm_model_label in MODEL_LABELS else label_for_backend(prompt_backend)
            llm_backend_id = llm_backend_id if llm_backend_id in llmsearch_backend_choices() else DEFAULT_LLMSEARCH_BACKEND
            llm_prompt = (llm_prompt or "").strip() or LLMSEARCH_DEFAULT_PROMPT
            try:
                shortlist_size = int(float(llm_shortlist_size))
            except Exception:
                shortlist_size = LLMSEARCH_SHORTLIST_DEFAULT
            shortlist_size = max(LLMSEARCH_SHORTLIST_MIN, min(LLMSEARCH_SHORTLIST_MAX, shortlist_size))
            state["llmsearch_backend"] = llm_backend_id
            state["llmsearch_shortlist_size"] = shortlist_size

            can_reuse_llm_cache = (
                state.get("llmsearch_cached_signature") == image_signature
                and state.get("llmsearch_cached_prompt") == llm_prompt
                and state.get("llmsearch_cached_backend") == llm_backend_id
                and state.get("llmsearch_cached_scoring_mode") == LLMSEARCH_SCORING_MODE_NUMERIC_V1
                and state.get("llmsearch_cached_shortlist_size") == shortlist_size
                and state.get("llmsearch_cached_model_label") == llm_model_label
                and state.get("llmsearch_cached_scores") is not None
            )
            if can_reuse_llm_cache:
                state["scores"] = dict(state["llmsearch_cached_scores"])
            else:
                try:
                    ensure_promptmatch_backend_loaded(llm_model_label, progress)
                    _, feature_paths, image_features, failed_paths = ensure_promptmatch_feature_cache(
                        image_paths,
                        llm_model_label,
                        progress,
                        reuse_desc="Reusing cached LLM-search shortlist embeddings for {count} images",
                        encode_desc="Encoding LLM-search shortlist embeddings for {count} images...",
                        progress_label="LLM shortlist",
                    )
                except Exception as exc:
                    return empty_result(str(exc), method)

                shortlist_query_emb = state["backend"].encode_text(llm_prompt)
                shortlist_scores = score_promptmatch_cached_features(
                    feature_paths,
                    image_features,
                    failed_paths,
                    shortlist_query_emb,
                    None,
                )
                ranked_candidates = [
                    item for item in shortlist_scores.values()
                    if not item.get("failed", False) and item.get("pos") is not None
                ]
                ranked_candidates.sort(key=lambda item: -float(item["pos"]))
                candidate_paths = [item["path"] for item in ranked_candidates[:shortlist_size]]
                if not candidate_paths:
                    return empty_result("LM search could not shortlist any usable images.", method)

                try:
                    llm_candidate_scores = score_llmsearch_candidates(
                        candidate_paths,
                        llm_prompt,
                        llm_backend_id,
                        image_signature,
                        progress,
                    )
                except Exception as exc:
                    return empty_result(f"LLM rerank failed: {exc}", method)

                shortlist_floor_candidates = [
                    item["pos"] for item in llm_candidate_scores.values()
                    if not item.get("failed", False)
                ]
                shortlist_floor = min(shortlist_floor_candidates) if shortlist_floor_candidates else -0.2
                reject_floor = max(-1.0, float(shortlist_floor) - 0.05)
                wrapped_scores = {}
                shortlisted_names = set(llm_candidate_scores.keys())
                for path in image_paths:
                    fname = os.path.basename(path)
                    base_item = shortlist_scores.get(fname)
                    llm_item = llm_candidate_scores.get(fname)
                    if llm_item is not None:
                        wrapped_scores[fname] = {
                            **llm_item,
                            "base_pos": float(base_item["pos"]) if base_item and base_item.get("pos") is not None else None,
                        }
                        continue
                    wrapped_scores[fname] = {
                        "pos": float(reject_floor),
                        "neg": None,
                        "path": path,
                        "failed": False,
                        "base_pos": float(base_item["pos"]) if base_item and base_item.get("pos") is not None else None,
                        "shortlisted": False,
                    }

                state["scores"] = wrapped_scores
                state["llmsearch_cached_signature"] = image_signature
                state["llmsearch_cached_prompt"] = llm_prompt
                state["llmsearch_cached_backend"] = llm_backend_id
                state["llmsearch_cached_scoring_mode"] = LLMSEARCH_SCORING_MODE_NUMERIC_V1
                state["llmsearch_cached_shortlist_size"] = shortlist_size
                state["llmsearch_cached_model_label"] = llm_model_label
                state["llmsearch_cached_scores"] = dict(wrapped_scores)

            pos_vals = [
                item["pos"] for item in state["scores"].values()
                if item.get("base_pos") is not None and not item.get("failed", False)
            ]
            pos_min, pos_max, pos_mid, neg_min, neg_max, _, _ = promptmatch_slider_range(state["scores"])
            default_main = round(min(pos_vals), 3) if pos_vals else pos_mid
            next_main = clamp_threshold(requested_main, pos_min, pos_max) if (previous_method != METHOD_LLMSEARCH and has_recalled) else default_main
            safe_pos_min, safe_pos_max = expand_slider_bounds(pos_min, pos_max, next_main)
            safe_neg_min, safe_neg_max = expand_slider_bounds(neg_min, neg_max, NEGATIVE_THRESHOLD)
            left_head, left_gallery, right_head, right_gallery, status, hist, sel_info, mark_state = current_view(next_main, NEGATIVE_THRESHOLD)
            return (
                left_head,
                left_gallery,
                right_head,
                right_gallery,
                status,
                hist,
                sel_info,
                mark_state,
                gr.update(minimum=safe_pos_min, maximum=safe_pos_max, value=next_main, label=main_label),
                gr.update(minimum=safe_neg_min, maximum=safe_neg_max, value=NEGATIVE_THRESHOLD, visible=False, interactive=False, label=aux_label),
                percentile_slider_update(METHOD_LLMSEARCH, state["scores"]),
                percentile_reset_button_update(METHOD_LLMSEARCH, state["scores"]),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        if method == METHOD_PROMPTMATCH:
            try:
                ensure_promptmatch_backend_loaded(model_label, progress)
                _, feature_paths, image_features, failed_paths = ensure_promptmatch_feature_cache(
                    image_paths,
                    model_label,
                    progress,
                    reuse_desc="Reusing cached PromptMatch image embeddings for {count} images",
                    encode_desc="Encoding PromptMatch image embeddings for {count} images...",
                    progress_label="PromptMatch",
                )
            except Exception as exc:
                return empty_result(str(exc), method)

            pos_prompt = (pos_prompt or "").strip() or SEARCH_PROMPT
            neg_prompt = (neg_prompt or "").strip()
            pos_emb = state["backend"].encode_text(pos_prompt)
            neg_emb = state["backend"].encode_text(neg_prompt) if neg_prompt else None
            state["scores"] = score_promptmatch_cached_features(
                feature_paths,
                image_features,
                failed_paths,
                pos_emb,
                neg_emb,
            )
            pos_min, pos_max, pos_mid, neg_min, neg_max, neg_mid, has_neg = promptmatch_slider_range(state["scores"])
            if preserve_promptmatch_thresholds or (previous_method != METHOD_PROMPTMATCH and has_recalled):
                next_main = clamp_threshold(requested_main, pos_min, pos_max)
                next_aux = clamp_threshold(requested_aux, neg_min, neg_max)
            else:
                next_main = pos_mid
                next_aux = neg_mid
            safe_pos_min, safe_pos_max = expand_slider_bounds(pos_min, pos_max, requested_main, next_main)
            safe_neg_min, safe_neg_max = expand_slider_bounds(neg_min, neg_max, requested_aux, next_aux)
            state["last_scored_method"] = METHOD_PROMPTMATCH
            state["last_scored_folder_key"] = folder_key
            state["last_promptmatch_model_label"] = model_label
            left_head, left_gallery, right_head, right_gallery, status, hist, sel_info, mark_state = current_view(next_main, next_aux)
            return (
                left_head,
                left_gallery,
                right_head,
                right_gallery,
                status,
                hist,
                sel_info,
                mark_state,
                gr.update(minimum=safe_pos_min, maximum=safe_pos_max, value=next_main, label=main_label),
                gr.update(minimum=safe_neg_min, maximum=safe_neg_max, value=next_aux, visible=True, interactive=has_neg, label=aux_label),
                percentile_slider_update(METHOD_PROMPTMATCH, state["scores"]),
                percentile_reset_button_update(METHOD_PROMPTMATCH, state["scores"]),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        if method == METHOD_TAGMATCH:
            try:
                state["scores"] = score_tagmatch_folder(image_paths, tagmatch_tags, progress)
            except Exception as exc:
                return empty_result(str(exc), method)
            pos_vals = [
                v["pos"] for v in state["scores"].values()
                if not v.get("failed", False)
            ]
            if pos_vals:
                tm_lo = min(pos_vals)
                tm_hi = max(pos_vals)
                tm_mid = (tm_lo + tm_hi) / 2.0
            else:
                tm_lo, tm_hi, tm_mid = TAGMATCH_SLIDER_MIN, TAGMATCH_SLIDER_MAX, TAGMATCH_DEFAULT_THRESHOLD
            if previous_method != METHOD_TAGMATCH and has_recalled:
                next_main = clamp_threshold(requested_main, tm_lo, tm_hi)
            else:
                next_main = tm_mid
            safe_lo, safe_hi = expand_slider_bounds(tm_lo, tm_hi, requested_main, next_main)
            left_head, left_gallery, right_head, right_gallery, status, hist, sel_info, mark_state = current_view(next_main, NEGATIVE_THRESHOLD)
            return (
                left_head,
                left_gallery,
                right_head,
                right_gallery,
                status,
                hist,
                sel_info,
                mark_state,
                gr.update(minimum=safe_lo, maximum=safe_hi, value=next_main, label=main_label),
                gr.update(value=NEGATIVE_THRESHOLD, visible=False),
                percentile_slider_update(METHOD_TAGMATCH, state["scores"]),
                percentile_reset_button_update(METHOD_TAGMATCH, state["scores"]),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
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
        if preserve_imagereward_threshold or (previous_method != METHOD_IMAGEREWARD and has_recalled):
            next_main = clamp_threshold(requested_main, lo, hi)
        else:
            next_main = mid
        safe_lo, safe_hi = expand_slider_bounds(lo, hi, requested_main, next_main)
        state["last_scored_method"] = METHOD_IMAGEREWARD
        state["last_scored_folder_key"] = folder_key
        left_head, left_gallery, right_head, right_gallery, status, hist, sel_info, mark_state = current_view(next_main, NEGATIVE_THRESHOLD)
        return (
            left_head,
            left_gallery,
            right_head,
            right_gallery,
                status,
                hist,
                sel_info,
                mark_state,
            gr.update(minimum=safe_lo, maximum=safe_hi, value=next_main, label=main_label),
            gr.update(value=NEGATIVE_THRESHOLD, visible=False),
            percentile_slider_update(METHOD_IMAGEREWARD, state["scores"]),
            percentile_reset_button_update(METHOD_IMAGEREWARD, state["scores"]),
            promptmatch_model_status_json(),
            *ui_visibility_updates(),
        )

    def find_similar_images(folder, model_label, main_threshold, aux_threshold, progress=gr.Progress()):
        folder = (folder or "").strip()
        recalled_main, _, has_recalled = recalled_mode_thresholds(METHOD_SIMILARITY, main_threshold, aux_threshold)
        if not folder or not os.path.isdir(folder):
            return (
                gr.update(value=f"Invalid folder: {folder!r}"),
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        image_paths = scan_image_paths(folder)
        if not image_paths:
            return (
                gr.update(value=f"No images found in {folder}"),
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        _, preview_fname = get_preview_image_path()
        if not preview_fname:
            return (
                gr.update(value="Select a preview image first, then find similar images."),
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        folder_name_map = {os.path.basename(path): path for path in image_paths}
        query_path = folder_name_map.get(preview_fname)
        if not query_path:
            return (
                gr.update(value=f"{preview_fname} is not part of the current folder, so similarity search cannot run."),
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        previous_folder = state.get("source_dir")
        previous_folder_key = normalize_folder_identity(previous_folder) if previous_folder else None
        folder_key = normalize_folder_identity(folder)
        available_names = set(folder_name_map.keys())
        preserved_overrides = {}
        if previous_folder_key == folder_key:
            preserved_overrides = {
                fname: side
                for fname, side in state.get("overrides", {}).items()
                if fname in available_names
            }

        release_inactive_gpu_models(METHOD_SIMILARITY)
        try:
            sync_promptmatch_proxy_cache(folder)
            ensure_promptmatch_backend_loaded(model_label, progress)
            _, feature_paths, image_features, failed_paths = ensure_promptmatch_feature_cache(
                image_paths,
                model_label,
                progress,
                reuse_desc="Reusing cached similarity embeddings for {count} images",
                encode_desc="Encoding similarity embeddings for {count} images...",
                progress_label="Similarity",
            )
            similarity_scores = score_similarity_cached_features(
                feature_paths,
                image_features,
                failed_paths,
                query_path,
            )
        except Exception as exc:
            return (
                gr.update(value=f"Similarity search failed: {exc}"),
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        state["method"] = METHOD_SIMILARITY
        set_scored_mode()
        state["source_dir"] = folder
        state["scores"] = similarity_scores
        state["overrides"] = preserved_overrides
        state["left_marked"] = []
        state["right_marked"] = []
        state["preview_fname"] = preview_fname
        clear_preview_search_context()
        state["similarity_query_fname"] = preview_fname
        state["similarity_model_label"] = model_label

        pos_min, pos_max, _, neg_min, neg_max, _, _ = promptmatch_slider_range(state["scores"])
        _, default_top_n = similarity_topn_defaults(state["scores"])
        default_main = threshold_for_percentile(METHOD_SIMILARITY, state["scores"], default_top_n)
        next_main = clamp_threshold(recalled_main, pos_min, pos_max) if has_recalled else default_main
        safe_pos_min, safe_pos_max = expand_slider_bounds(pos_min, pos_max, next_main)
        safe_neg_min, safe_neg_max = expand_slider_bounds(neg_min, neg_max, NEGATIVE_THRESHOLD)
        status_text = f"Similarity search using {model_label} from {preview_fname}."
        return (
            gr.update(value=status_text),
            *current_view(next_main, NEGATIVE_THRESHOLD),
            gr.update(
                minimum=safe_pos_min,
                maximum=safe_pos_max,
                value=next_main,
                label=threshold_labels(METHOD_SIMILARITY)[0],
            ),
            gr.update(
                minimum=safe_neg_min,
                maximum=safe_neg_max,
                value=NEGATIVE_THRESHOLD,
                visible=False,
                interactive=False,
                label=threshold_labels(METHOD_SIMILARITY)[1],
            ),
            percentile_slider_update(METHOD_SIMILARITY, state["scores"]),
            percentile_reset_button_update(METHOD_SIMILARITY, state["scores"]),
            promptmatch_model_status_json(),
            *ui_visibility_updates(),
        )

    def find_same_person_images(folder, main_threshold, aux_threshold, progress=gr.Progress()):
        folder = (folder or "").strip()
        recalled_main, _, has_recalled = recalled_mode_thresholds(METHOD_SAMEPERSON, main_threshold, aux_threshold)
        if not folder or not os.path.isdir(folder):
            return (
                gr.update(value=f"Invalid folder: {folder!r}"),
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        image_paths = scan_image_paths(folder)
        if not image_paths:
            return (
                gr.update(value=f"No images found in {folder}"),
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        _, preview_fname = get_preview_image_path()
        if not preview_fname:
            return (
                gr.update(value="Select a preview image first, then find the same person."),
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        folder_name_map = {os.path.basename(path): path for path in image_paths}
        query_path = folder_name_map.get(preview_fname)
        if not query_path:
            return (
                gr.update(value=f"{preview_fname} is not part of the current folder, so same-person search cannot run."),
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        previous_folder = state.get("source_dir")
        previous_folder_key = normalize_folder_identity(previous_folder) if previous_folder else None
        folder_key = normalize_folder_identity(folder)
        available_names = set(folder_name_map.keys())
        preserved_overrides = {}
        if previous_folder_key == folder_key:
            preserved_overrides = {
                fname: side
                for fname, side in state.get("overrides", {}).items()
                if fname in available_names
            }

        release_inactive_gpu_models(METHOD_SAMEPERSON)
        try:
            sync_promptmatch_proxy_cache(folder)
            _, feature_paths, face_embeddings, failures = ensure_face_feature_cache(image_paths, progress)
            sameperson_scores = score_sameperson_cached_features(
                feature_paths,
                face_embeddings,
                failures,
                query_path,
            )
        except Exception as exc:
            return (
                gr.update(value=f"Same-person search failed: {exc}"),
                *current_view(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                promptmatch_model_status_json(),
                *ui_visibility_updates(),
            )

        state["method"] = METHOD_SAMEPERSON
        set_scored_mode()
        state["source_dir"] = folder
        state["scores"] = sameperson_scores
        state["overrides"] = preserved_overrides
        state["left_marked"] = []
        state["right_marked"] = []
        state["preview_fname"] = preview_fname
        clear_preview_search_context()
        state["sameperson_query_fname"] = preview_fname
        state["sameperson_model_label"] = FACE_MODEL_LABEL

        pos_min, pos_max, _, neg_min, neg_max, _, _ = promptmatch_slider_range(state["scores"])
        _, default_top_n = similarity_topn_defaults(state["scores"])
        default_main = threshold_for_percentile(METHOD_SAMEPERSON, state["scores"], default_top_n)
        next_main = clamp_threshold(recalled_main, pos_min, pos_max) if has_recalled else default_main
        safe_pos_min, safe_pos_max = expand_slider_bounds(pos_min, pos_max, next_main)
        safe_neg_min, safe_neg_max = expand_slider_bounds(neg_min, neg_max, NEGATIVE_THRESHOLD)
        status_text = f"Same-person search using {FACE_MODEL_LABEL} from {preview_fname}."
        return (
            gr.update(value=status_text),
            *current_view(next_main, NEGATIVE_THRESHOLD),
            gr.update(
                minimum=safe_pos_min,
                maximum=safe_pos_max,
                value=next_main,
                label=threshold_labels(METHOD_SAMEPERSON)[0],
            ),
            gr.update(
                minimum=safe_neg_min,
                maximum=safe_neg_max,
                value=NEGATIVE_THRESHOLD,
                visible=False,
                interactive=False,
                label=threshold_labels(METHOD_SAMEPERSON)[1],
            ),
            percentile_slider_update(METHOD_SAMEPERSON, state["scores"]),
            percentile_reset_button_update(METHOD_SAMEPERSON, state["scores"]),
            promptmatch_model_status_json(),
            *ui_visibility_updates(),
        )

    def handle_shortcut_action(action, method, folder, model_label, pos_prompt, neg_prompt, ir_prompt, ir_negative_prompt, ir_penalty_weight, llm_model_label, llm_prompt, llm_backend_id, llm_shortlist_size, tagmatch_tags, main_threshold, aux_threshold, keep_pm_thresholds, keep_ir_thresholds, progress=gr.Progress()):
        action = (action or "").strip()
        if not action.startswith("run:"):
            return empty_result("Shortcut action ignored.", method)

        prompt_id = action.split(":", 2)[1] if ":" in action else ""
        if prompt_id in ("hy-pos", "hy-neg"):
            method = METHOD_PROMPTMATCH
        elif prompt_id in ("hy-ir-pos", "hy-ir-neg"):
            method = METHOD_IMAGEREWARD

        return score_folder(
            method,
            folder,
            model_label,
            pos_prompt,
            neg_prompt,
            ir_prompt,
            ir_negative_prompt,
            ir_penalty_weight,
            llm_model_label,
            llm_prompt,
            llm_backend_id,
            llm_shortlist_size,
            tagmatch_tags,
            main_threshold,
            aux_threshold,
            keep_pm_thresholds,
            keep_ir_thresholds,
            progress=progress,
        )

    def update_split(main_threshold, aux_threshold):
        return render_view_with_controls(main_threshold, aux_threshold)

    def update_histogram_only(main_threshold, aux_threshold):
        return render_histogram(state["method"], state["scores"], main_threshold, aux_threshold)

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
            return gr.update(value=state["generated_prompt_status"]), gr.update(), gr.update(), gr.update()

        state["generated_prompt"] = prompt_text
        if method == METHOD_PROMPTMATCH:
            target_label = "PromptMatch positive prompt"
        elif method == METHOD_LLMSEARCH:
            target_label = "LM search prompt"
        else:
            target_label = "ImageReward positive prompt"
        state["generated_prompt_status"] = f"Inserted generated prompt into {target_label}."
        if method == METHOD_PROMPTMATCH:
            return gr.update(value=state["generated_prompt_status"]), gr.update(value=prompt_text), gr.update(), gr.update()
        if method == METHOD_LLMSEARCH:
            return gr.update(value=state["generated_prompt_status"]), gr.update(), gr.update(), gr.update(value=prompt_text)
        return gr.update(value=state["generated_prompt_status"]), gr.update(), gr.update(value=prompt_text), gr.update()

    def update_proxy_display(use_proxy_display, main_threshold, aux_threshold):
        state["use_proxy_display"] = bool(use_proxy_display)
        return render_view_with_controls(main_threshold, aux_threshold)

    def update_imagereward_penalty_weight(penalty_weight, main_threshold, aux_threshold):
        recomputed = recompute_imagereward_scores(penalty_weight)
        if not recomputed:
            return (
                *render_view_with_controls(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
            )

        lo, hi, _ = imagereward_slider_range(state["scores"])
        clamped = clamp_threshold(main_threshold, lo, hi)
        safe_lo, safe_hi = expand_slider_bounds(lo, hi, main_threshold, clamped)
        main_label, _, _, _ = threshold_labels(METHOD_IMAGEREWARD)
        return (
            *render_view_with_controls(clamped, aux_threshold),
            gr.update(
                minimum=safe_lo,
                maximum=safe_hi,
                value=clamped,
                label=main_label,
            ),
            gr.update(value=NEGATIVE_THRESHOLD, visible=False),
        )

    def handle_thumb_action(action, main_threshold, aux_threshold):
        # Custom JS reports preview clicks, shift-click bulk marking, and drag-drop moves.
        noop = (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
        )
        with_slider_skips = lambda view: (*view, gr.skip(), gr.skip())
        if not action:
            return noop
        if str(action).startswith("previewfname:"):
            try:
                _, fname, _ = str(action).split(":", 2)
            except Exception:
                return noop
            visible_names = {os.path.basename(path) for path, _ in state.get("browse_items", [])}
            if fname and (fname in state.get("scores", {}) or fname in visible_names):
                state["preview_fname"] = fname
            return noop
        if str(action).startswith("dialogactionjson:"):
            try:
                payload = json.loads(str(action)[17:])
                action_id = str(payload.get("action", "") or "")
                fname = str(payload.get("fname", "") or "")
            except Exception:
                return noop
            visible_names = {os.path.basename(path) for path, _ in state.get("browse_items", [])}
            if fname and (fname in state.get("scores", {}) or fname in visible_names):
                state["preview_fname"] = fname
            if action_id == "hy-move-right":
                return with_slider_skips(move_right(main_threshold, aux_threshold, preview_override=fname))
            if action_id == "hy-move-left":
                return with_slider_skips(move_left(main_threshold, aux_threshold, preview_override=fname))
            if action_id == "hy-fit-threshold":
                return fit_threshold_to_targets(main_threshold, aux_threshold, preview_override=fname)
            return noop
        if str(action).startswith("dropjson:"):
            try:
                payload = json.loads(str(action)[9:])
                side = payload["source_side"]
                index = int(payload["source_index"])
                target_side = payload["target_side"]
                drop_fnames = [str(name) for name in payload.get("fnames", []) if str(name)]
            except Exception:
                return noop
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
            return with_slider_skips(render_view_with_controls(main_threshold, aux_threshold))
        parts = str(action).split(":")
        verb = parts[0] if parts else ""
        if is_browse_mode():
            left_items = list(state.get("browse_items", []))
            right_items = []
        else:
            left_items, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)

        try:
            _, side, raw_index, _ = parts
            index = int(raw_index)
        except Exception:
            return noop
        items = left_items if side == "left" else right_items
        if 0 <= index < len(items):
            fname = os.path.basename(items[index][0])
            if verb == "preview":
                state["preview_fname"] = fname
                return noop
            if is_browse_mode():
                state["preview_fname"] = fname
                return noop
            else:
                marked_key = "left_marked" if side == "left" else "right_marked"
                if fname in state[marked_key]:
                    state[marked_key] = [name for name in state[marked_key] if name != fname]
                else:
                    state[marked_key].append(fname)
        return with_slider_skips(render_view_with_controls(main_threshold, aux_threshold))

    def handle_hist_width(width_value, main_threshold, aux_threshold):
        try:
            next_width = max(220, int(float(width_value)))
        except Exception:
            return gr.skip()
        if abs(next_width - int(state.get("hist_width", 300))) < 2:
            return gr.skip()
        state["hist_width"] = next_width
        return render_histogram(state["method"], state["scores"], main_threshold, aux_threshold)

    def move_right(main_threshold, aux_threshold, preview_override=None):
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        left_name, right_name, _, _ = method_labels(state["method"])
        left_targets, _ = active_targets(main_threshold, aux_threshold, preview_override=preview_override)
        targets = left_targets or list(state.get("left_marked", []))
        for fname in targets:
            state["overrides"][fname] = right_name
        state["left_marked"] = []
        state["right_marked"] = []
        return render_view_with_controls(main_threshold, aux_threshold)

    def move_left(main_threshold, aux_threshold, preview_override=None):
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        left_name, right_name, _, _ = method_labels(state["method"])
        _, right_targets = active_targets(main_threshold, aux_threshold, preview_override=preview_override)
        targets = right_targets or list(state.get("right_marked", []))
        for fname in targets:
            state["overrides"][fname] = left_name
        state["left_marked"] = []
        state["right_marked"] = []
        return render_view_with_controls(main_threshold, aux_threshold)

    def pin_selected(main_threshold, aux_threshold, preview_override=None):
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        left_name, right_name, _, _ = method_labels(state["method"])
        left_targets, right_targets = active_targets(main_threshold, aux_threshold, preview_override=preview_override)
        for fname in left_targets:
            state["overrides"][fname] = left_name
        for fname in right_targets:
            state["overrides"][fname] = right_name
        state["left_marked"] = []
        state["right_marked"] = []
        return render_view_with_controls(main_threshold, aux_threshold)

    def clear_status(main_threshold, aux_threshold):
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        for fname in set(state["left_marked"] + state["right_marked"]):
            state["overrides"].pop(fname, None)
        state["left_marked"] = []
        state["right_marked"] = []
        return render_view_with_controls(main_threshold, aux_threshold)

    def clear_all_status(main_threshold, aux_threshold):
        if is_browse_mode():
            return render_view_with_controls(main_threshold, aux_threshold)
        state["overrides"].clear()
        state["left_marked"] = []
        state["right_marked"] = []
        return render_view_with_controls(main_threshold, aux_threshold)

    def fit_threshold_to_targets(main_threshold, aux_threshold, preview_override=None):
        if is_browse_mode():
            return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())
        left_targets, right_targets = active_targets(main_threshold, aux_threshold, preview_override=preview_override)
        targets = left_targets or right_targets
        if not targets or (left_targets and right_targets) or not state["scores"]:
            return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())

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
            *render_view_with_controls(new_main, new_aux),
            gr.update(value=new_main),
            gr.update(value=new_aux),
        )

    def set_from_percentile(percentile, main_threshold, aux_threshold):
        new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
        return (*render_view_with_controls(new_threshold, aux_threshold), gr.update(value=new_threshold))

    def update_histogram_from_percentile(percentile, aux_threshold):
        new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
        return render_histogram(state["method"], state["scores"], new_threshold, aux_threshold)

    def reset_main_threshold_to_middle(main_threshold, aux_threshold):
        new_main, _, _ = middle_threshold_values(state["method"])
        return (
            *render_view_with_controls(new_main, aux_threshold),
            gr.update(value=new_main),
            gr.update(),
        )

    def reset_aux_threshold_to_middle(main_threshold, aux_threshold):
        if state["method"] != METHOD_PROMPTMATCH:
            return (
                *render_view_with_controls(main_threshold, aux_threshold),
                gr.update(),
                gr.update(),
            )
        _, new_aux, _ = middle_threshold_values(METHOD_PROMPTMATCH)
        return (
            *render_view_with_controls(main_threshold, new_aux),
            gr.update(),
            gr.update(value=new_aux),
        )

    def reset_percentile_to_middle(main_threshold, aux_threshold):
        if uses_similarity_topn(state["method"]):
            _, percentile = similarity_topn_defaults(state["scores"])
        else:
            percentile = 50
        if state["scores"]:
            new_threshold = threshold_for_percentile(state["method"], state["scores"], percentile)
        else:
            new_threshold = float(main_threshold)
        return (
            *render_view_with_controls(new_threshold, aux_threshold),
            gr.update(value=new_threshold),
            gr.update(value=percentile),
        )

    def update_zoom(zoom_value, main_threshold, aux_threshold):
        # Invert the slider so dragging right makes thumbnails larger by reducing columns.
        try:
            slider_value = max(2, min(10, int(zoom_value)))
            state["zoom_columns"] = 12 - slider_value
        except Exception:
            state["zoom_columns"] = 5
        return render_view_with_controls(main_threshold, aux_threshold)

    def on_hist_click(sel: gr.SelectData, main_threshold, aux_threshold):
        geom = state.get("hist_geom")
        if not geom:
            return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())
        try:
            cx, cy = sel.index
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
                    val = lo + ((cx - PAD_L) / cW) * (hi - lo)
                    main_threshold = round(max(lo, min(hi, val)), 3)
                elif geom["has_neg"] and y0neg <= cy <= y0neg + CH:
                    lo, hi = geom["neg_lo"], geom["neg_hi"]
                    val = lo + ((cx - PAD_L) / cW) * (hi - lo)
                    aux_threshold = round(max(lo, min(hi, val)), 3)
                else:
                    return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())
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
                    return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())
        except Exception:
            return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(), gr.update())

        return (*render_view_with_controls(main_threshold, aux_threshold), gr.update(value=main_threshold), gr.update(value=aux_threshold))

    def export_files(main_threshold, aux_threshold, export_left_enabled, export_right_enabled, export_move_enabled, export_left_name, export_right_name):
        # Export is a lossless copy, not a rewrite or recompression of the originals.
        if is_browse_mode():
            return (*render_view_with_controls(main_threshold, aux_threshold), "Export is unavailable in browse mode. Run scoring or a search first.")
        left_items, right_items = build_split(state["method"], state["scores"], state["overrides"], main_threshold, aux_threshold)
        left_name, right_name, left_dirname, right_dirname = method_labels(state["method"])
        base = state["source_dir"]
        targets = []
        if export_left_enabled:
            targets.append((left_name, sanitize_export_name(export_left_name) or left_dirname, left_items))
        if export_right_enabled:
            targets.append((right_name, sanitize_export_name(export_right_name) or right_dirname, right_items))
        if not targets:
            return (*render_view_with_controls(main_threshold, aux_threshold), "Enable at least one bucket for export.")
        target_names = [folder_name for _, folder_name, _ in targets]
        if len(set(target_names)) != len(target_names):
            return (*render_view_with_controls(main_threshold, aux_threshold), "Export folder names must be different when both buckets are enabled.")

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
                clear_preview_search_context()
        return (*render_view_with_controls(main_threshold, aux_threshold), "\n".join(lines))

    css = """
    html, body { height:100% !important; overflow:hidden !important; }
    body, .gradio-container { background:#0d0d11 !important; color:#ddd8cc !important; }
    body { margin:0 !important; }
    .gr-block,.gr-box,.panel { background:#14141c !important; border-color:#252530 !important; }
    .gradio-container {
        max-width: 100% !important;
        margin:0 !important;
        padding: 6px 2px !important;
        min-height:100vh !important;
        height:100vh !important;
        overflow:hidden !important;
    }
    .main {
        max-width: 100% !important;
        padding:0 !important;
        height:100% !important;
        overflow:hidden !important;
    }
    .main > .gradio-row { gap:6px !important; }
    .app-shell {
        gap:6px !important;
        align-items:flex-start !important;
    }
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
    .header-meta { color:#667755; font-family:monospace; font-size:.78rem; white-space:nowrap; margin-left:auto; }
    .sidebar-box {
        background:#171722;
        border:1px solid #2c2c39;
        border-radius:6px;
        padding:3px;
        display:block !important;
        flex:0 0 320px !important;
        width:320px !important;
        max-width:320px !important;
        align-self:flex-start !important;
        position:sticky !important;
        top:6px !important;
        height:calc(100vh - 80px) !important;
        max-height:calc(100vh - 80px) !important;
        overflow-y:auto !important;
        overflow-x:hidden !important;
        scrollbar-gutter:stable !important;
    }
    .sidebar-scroll {
        margin-top:2px !important;
        max-height:none !important;
        height:auto !important;
        overflow:visible !important;
        padding-right:2px !important;
        background:transparent !important;
        border:0 !important;
        box-shadow:none !important;
    }
    .gallery-pane {
        min-width:0 !important;
        overflow:visible !important;
    }
    .sidebar-box .gr-accordion { margin-bottom:2px !important; border:1px solid #2c2c39 !important; border-radius:4px !important; background:#151520 !important; }
    .sidebar-scroll > .gr-accordion,
    .sidebar-scroll > .gr-group,
    .sidebar-scroll > .block,
    .sidebar-scroll .gr-accordion,
    .sidebar-scroll .gr-accordion .content,
    .sidebar-scroll .gr-accordion .content > div {
        flex:none !important;
        min-height:auto !important;
        max-height:none !important;
        height:auto !important;
        overflow:visible !important;
    }
    .sidebar-box .gr-accordion summary, .sidebar-box .gr-accordion button { font-family:monospace !important; font-size:.9rem !important; color:#d7dbc8 !important; }
    .sidebar-box .gr-accordion summary { padding-top:2px !important; padding-bottom:2px !important; min-height:0 !important; }
    .sidebar-box .gr-accordion .label-wrap { padding-top:0 !important; padding-bottom:0 !important; }
    .sidebar-box .gr-accordion .content { padding-top:2px !important; padding-bottom:2px !important; }
    .sidebar-box .gr-accordion summary,
    .sidebar-box .gr-accordion .label-wrap,
    .sidebar-box .gr-accordion button {
        background:linear-gradient(180deg, rgba(38, 43, 56, 0.95), rgba(27, 31, 41, 0.95)) !important;
        border-bottom:1px solid rgba(255, 255, 255, 0.04) !important;
    }
    #hy-acc-setup {
        border-color:#455774 !important;
        box-shadow:inset 3px 0 0 #7da8ff !important;
    }
    #hy-acc-setup summary, #hy-acc-setup .label-wrap, #hy-acc-setup button {
        background:linear-gradient(180deg, rgba(39, 49, 66, 0.98), rgba(26, 33, 46, 0.98)) !important;
    }
    #hy-acc-scoring {
        border-color:#53684e !important;
        box-shadow:inset 3px 0 0 #7fb06d !important;
    }
    #hy-acc-scoring summary, #hy-acc-scoring .label-wrap, #hy-acc-scoring button {
        background:linear-gradient(180deg, rgba(42, 57, 46, 0.98), rgba(28, 38, 31, 0.98)) !important;
    }
    #hy-acc-prompt {
        border-color:#2f6971 !important;
        box-shadow:inset 3px 0 0 #58c7d6 !important;
    }
    #hy-acc-prompt summary, #hy-acc-prompt .label-wrap, #hy-acc-prompt button {
        background:linear-gradient(180deg, rgba(22, 56, 60, 0.98), rgba(14, 38, 41, 0.98)) !important;
    }
    #hy-acc-thresholds {
        border-color:#67507e !important;
        box-shadow:inset 3px 0 0 #b590e8 !important;
    }
    #hy-acc-thresholds summary, #hy-acc-thresholds .label-wrap, #hy-acc-thresholds button {
        background:linear-gradient(180deg, rgba(49, 36, 58, 0.98), rgba(33, 24, 39, 0.98)) !important;
    }
    #hy-acc-export {
        border-color:#7a6930 !important;
        box-shadow:inset 3px 0 0 #d9c06a !important;
    }
    #hy-acc-export summary, #hy-acc-export .label-wrap, #hy-acc-export button {
        background:linear-gradient(180deg, rgba(61, 54, 24, 0.98), rgba(43, 38, 17, 0.98)) !important;
    }
    .sidebar-box .gr-group, .sidebar-box .block { gap:2px !important; }
    .sidebar-box .gr-form, .sidebar-box .gradio-row { gap:2px !important; }
    .threshold-row { align-items:end; gap:6px; overflow:hidden; }
    .threshold-row .gr-slider, .threshold-row input[type=range] { min-width:0; }
    .threshold-row input[type=number] { min-width:0; width:64px !important; max-width:64px !important; }
    .threshold-actions {
        min-width:58px !important;
        gap:4px !important;
    }
    .threshold-mid button {
        min-width:50px !important;
        height:36px !important;
        padding:0 8px !important;
        font-family:monospace !important;
        background:#2a2435 !important;
        border:1px solid #5d4f77 !important;
        color:#e0d7f1 !important;
    }
    .threshold-mid button:hover {
        background:#362d45 !important;
        border-color:#8d77b4 !important;
    }
    .sidebar-box label { margin-bottom:1px !important; }
    .sidebar-box label span { margin-bottom:0 !important; line-height:1.05 !important; }
    .sidebar-box .form > *, .sidebar-box .wrap > * { margin-top:0 !important; margin-bottom:0 !important; }
    .sidebar-box .gradio-container-4-0, .sidebar-box .gradio-container-3-0 { gap:2px !important; }
    .method-note { font-family:monospace; color:#8e9d80; background:#11111a; border-radius:4px; padding:2px 4px; }
    .method-note p { margin:0 !important; font-family:monospace !important; font-size:.82rem !important; line-height:1.35 !important; color:#8e9d80 !important; }
    .promptgen-status p { margin:0 !important; font-family:monospace !important; font-size:.76rem !important; line-height:1.35 !important; color:#8ec5ff !important; }
    .preview-action-stack { gap:4px !important; }
    .preview-action-stack .gr-button,
    .preview-action-stack > .gr-button,
    .preview-action-stack button {
        margin-top:0 !important;
        margin-bottom:0 !important;
    }
    .preview-prompt-group {
        margin-top:4px !important;
        padding-top:4px !important;
        border-top:1px solid rgba(88, 199, 214, 0.22) !important;
    }
    .status-md p { font-family:monospace !important; color:#9fc27c !important; }
    .hist-img, .hist-img > div { width:100% !important; max-width:100% !important; }
    .hist-img img { cursor:crosshair !important; border-radius:6px; width:100% !important; max-width:100% !important; display:block !important; }
    .hy-hover-line {
        position:absolute;
        width:0;
        border-left:1px dashed rgba(190, 230, 190, 0.75);
        pointer-events:none;
        z-index:3;
        opacity:0;
        transition:opacity 0.06s linear;
    }
    .hy-hover-line-neg {
        border-left-color: rgba(220, 165, 145, 0.78);
    }
    #hy-left-gallery, #hy-left-gallery > div {
        background:linear-gradient(180deg, rgba(18, 36, 24, 0.96), rgba(13, 18, 15, 0.98)) !important;
        border:1px solid #284732 !important;
        border-radius:10px !important;
        box-shadow:inset 0 1px 0 rgba(110, 170, 120, 0.08) !important;
    }
    #hy-right-gallery, #hy-right-gallery > div {
        background:linear-gradient(180deg, rgba(40, 22, 22, 0.96), rgba(19, 14, 14, 0.98)) !important;
        border:1px solid #5b2f35 !important;
        border-radius:10px !important;
        box-shadow:inset 0 1px 0 rgba(195, 110, 110, 0.08) !important;
    }
    .grid-wrap img { object-fit: contain !important; background: #0a0a12; }
    .grid-wrap .caption-label span, .grid-wrap [class*="caption"] { font-family:monospace !important; font-size:.72em !important; color:#8899aa !important; }
    .move-col { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px; padding:10px 6px; background:#0f0f16; border-radius:8px; border:1px solid #252535; }
    .move-col button { width:100%; }
    #hy-move-left button, #hy-move-right button {
        font-size:1.12rem !important;
        line-height:1.05 !important;
        letter-spacing:.01em !important;
    }
    #hy-move-left, #hy-move-left button {
        background:#2f8f45 !important;
        background-image:none !important;
        border:1px solid #58bb73 !important;
        color:#f3fff2 !important;
        font-weight:700 !important;
    }
    #hy-move-left:hover, #hy-move-left button:hover {
        background:#38a14f !important;
        background-image:none !important;
    }
    #hy-move-right, #hy-move-right button {
        background:#aa3a3a !important;
        background-image:none !important;
        border:1px solid #dc7c7c !important;
        color:#fff2f2 !important;
        font-weight:700 !important;
    }
    #hy-move-right:hover, #hy-move-right button:hover {
        background:#bf4747 !important;
        background-image:none !important;
    }
    #hy-fit-threshold, #hy-fit-threshold button {
        background:#7fbf6a !important;
        background-image:none !important;
        border:1px solid #b2e39d !important;
        color:#10230f !important;
        font-weight:700 !important;
    }
    #hy-fit-threshold:hover, #hy-fit-threshold button:hover {
        background:#92cf7b !important;
        background-image:none !important;
    }
    #hy-clear-status, #hy-clear-status button,
    #hy-clear-all-status, #hy-clear-all-status button {
        background:#d2b44a !important;
        background-image:none !important;
        border:1px solid #ecd98d !important;
        color:#2d2408 !important;
        font-weight:700 !important;
    }
    #hy-clear-status:hover, #hy-clear-status button:hover,
    #hy-clear-all-status:hover, #hy-clear-all-status button:hover {
        background:#dfc35a !important;
        background-image:none !important;
    }
    .sel-info p { font-family:monospace !important; font-size:1.08em !important; font-weight:700 !important; color:#aabb88 !important; text-align:center; word-break:break-all; }
    #hy-folder textarea, #hy-folder input { min-height:36px !important; font-size:.96rem !important; }
    #hy-run-pm, #hy-run-ir, #hy-export, #hy-generate-prompt, #hy-find-similar, #hy-find-same-person, #hy-insert-prompt { border-radius:6px !important; }
    #hy-run-pm, #hy-run-pm button, #hy-run-ir, #hy-run-ir button, #hy-export, #hy-export button {
        background:#2f8f45 !important;
        background-image:none !important;
        border:1px solid #58bb73 !important;
        color:#f3fff2 !important;
        font-weight:700 !important;
        box-shadow:0 0 0 1px rgba(25, 55, 30, 0.15) inset !important;
    }
    #hy-run-pm button, #hy-run-ir button, #hy-export button, #hy-generate-prompt button, #hy-find-similar button, #hy-find-same-person button, #hy-insert-prompt button {
        min-height:34px !important;
        border-radius:6px !important;
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
    #hy-find-similar, #hy-find-similar button {
        background:#1f7b7b !important;
        background-image:none !important;
        border:1px solid #66caca !important;
        color:#f0ffff !important;
        font-weight:700 !important;
        box-shadow:0 0 0 1px rgba(12, 55, 55, 0.22) inset !important;
    }
    #hy-find-similar:hover, #hy-find-similar button:hover {
        background:#259191 !important;
        background-image:none !important;
    }
    #hy-find-similar button:disabled {
        background:#1b5f5f !important;
        color:#d6efef !important;
    }
    #hy-find-same-person, #hy-find-same-person button {
        background:#6f4aa9 !important;
        background-image:none !important;
        border:1px solid #b590e8 !important;
        color:#f7f1ff !important;
        font-weight:700 !important;
        box-shadow:0 0 0 1px rgba(54, 32, 83, 0.2) inset !important;
    }
    #hy-find-same-person:hover, #hy-find-same-person button:hover {
        background:#8057bf !important;
        background-image:none !important;
    }
    #hy-find-same-person button:disabled {
        background:#5a3a86 !important;
        color:#eadfff !important;
    }
    #hy-left-gallery .preview .thumbnails,
    #hy-right-gallery .preview .thumbnails,
    #hy-left-gallery .preview .thumbnail-small,
    #hy-right-gallery .preview .thumbnail-small {
        display:none !important;
        pointer-events:none !important;
    }
    #hy-left-gallery .preview,
    #hy-right-gallery .preview {
        display:flex !important;
        flex-direction:column !important;
    }
    #hy-left-gallery .preview .media-button,
    #hy-right-gallery .preview .media-button {
        pointer-events:none !important;
        cursor:default !important;
        height:auto !important;
        flex:1 1 auto !important;
        min-height:0 !important;
    }
    #hy-left-gallery .preview [data-testid="detailed-image"],
    #hy-right-gallery .preview [data-testid="detailed-image"],
    #hy-left-gallery .preview [data-testid="detailed-video"],
    #hy-right-gallery .preview [data-testid="detailed-video"] {
        cursor:default !important;
        pointer-events:none !important;
    }
    [role="dialog"] .thumbnail-item,
    [aria-modal="true"] .thumbnail-item,
    [role="dialog"] [data-testid*="thumbnail"],
    [aria-modal="true"] [data-testid*="thumbnail"] {
        display:none !important;
        pointer-events:none !important;
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
    .gallery-topbar { align-items:end; margin-bottom:8px; gap:14px !important; flex-wrap:nowrap !important; }
    .gallery-header-slot { min-width:0; overflow:hidden !important; }
    .gallery-header-spacer {
        flex:0 0 auto !important;
        min-width:100px !important;
    }
    .gallery-zoom-slot {
        margin-left:auto !important;
        display:flex !important;
        justify-content:flex-end !important;
        align-items:flex-end !important;
        min-width:230px !important;
    }
    .gallery-head-row {
        display:flex !important;
        width:100% !important;
        align-items:center !important;
        justify-content:space-between !important;
        gap:6px !important;
        flex-wrap:nowrap !important;
        min-width:0 !important;
        overflow:hidden !important;
    }
    .gallery-head-row .markdown { min-width:0 !important; flex:0 0 auto !important; }
    .gallery-head-fill {
        flex:1 1 auto !important;
        min-width:0 !important;
        width:100% !important;
    }
    .gallery-head-row .markdown h3,
    .gallery-head-row .markdown p {
        margin:0 !important;
    }
    .gallery-head-row .markdown p {
        font-family:monospace !important;
        font-size:.82rem !important;
        color:#bfc8b5 !important;
    }
    .gallery-export-toggle {
        flex:0 0 auto !important;
        min-width:0 !important;
        width:auto !important;
        max-width:max-content !important;
        margin:0 !important;
        padding:0 !important;
        background:transparent !important;
        border:0 !important;
        box-shadow:none !important;
    }
    .gallery-export-toggle label {
        margin:0 !important;
        min-height:0 !important;
        gap:5px !important;
        padding:0 !important;
        justify-content:flex-start !important;
        width:fit-content !important;
        background:transparent !important;
        border:0 !important;
        box-shadow:none !important;
    }
    .gallery-export-toggle .wrap {
        width:fit-content !important;
        min-width:0 !important;
        max-width:max-content !important;
        padding:0 !important;
        margin:0 !important;
    }
    .gallery-export-toggle .checkbox,
    .gallery-export-toggle .checkbox-wrap {
        margin:0 !important;
        flex:0 0 auto !important;
    }
    .gallery-export-toggle span {
        font-family:monospace !important;
        font-size:.68rem !important;
        color:#aebaa0 !important;
        letter-spacing:.03em !important;
        text-transform:uppercase !important;
        white-space:nowrap !important;
    }
    .export-move-toggle {
        min-width:0 !important;
    }
    .export-move-toggle label {
        align-items:center !important;
        gap:8px !important;
        padding:6px 10px !important;
        border:1px solid #9f7b2c !important;
        border-radius:6px !important;
        background:linear-gradient(180deg, rgba(70, 56, 18, 0.92), rgba(49, 38, 12, 0.92)) !important;
        box-shadow:inset 0 1px 0 rgba(255, 226, 154, 0.08) !important;
    }
    .export-move-toggle .wrap {
        width:100% !important;
        max-width:none !important;
    }
    .export-move-toggle .checkbox,
    .export-move-toggle .checkbox-wrap {
        transform:scale(1.16) !important;
        transform-origin:left center !important;
    }
    .export-move-toggle input[type="checkbox"] {
        accent-color:#e1b24b !important;
    }
    .export-move-toggle span {
        font-size:.8rem !important;
        font-weight:800 !important;
        letter-spacing:.06em !important;
        color:#f1d78b !important;
        text-transform:uppercase !important;
    }
    .export-move-toggle:hover label {
        border-color:#d3a84a !important;
        background:linear-gradient(180deg, rgba(84, 67, 23, 0.95), rgba(58, 45, 16, 0.95)) !important;
    }
    .gallery-export-name {
        flex:0 1 156px !important;
        min-width:120px !important;
        max-width:180px !important;
    }
    .gallery-export-name textarea,
    .gallery-export-name input {
        min-height:31px !important;
        height:31px !important;
        font-family:monospace !important;
        font-size:.92rem !important;
        font-weight:700 !important;
        letter-spacing:.01em !important;
        color:#f3f1e6 !important;
        padding:4px 10px !important;
    }
    .gallery-count {
        flex:0 0 auto !important;
        min-width:max-content !important;
    }
    .gallery-count p {
        font-family:monospace !important;
        font-size:.82rem !important;
        font-weight:700 !important;
        color:#d7decf !important;
        white-space:nowrap !important;
    }
    .export-options-row {
        align-items:center !important;
        gap:14px !important;
        margin-bottom:10px !important;
        flex-wrap:nowrap !important;
    }
    .zoom-inline-wrap { align-items:center; gap:8px; margin-left:auto; flex-wrap:nowrap; overflow:hidden; justify-content:flex-end !important; width:100% !important; }
    .zoom-inline-label {
        flex:0 0 auto;
        overflow:visible !important;
        min-width:65px !important;
        max-width:65px !important;
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
    @media (max-width: 900px) {
        html, body {
            height:auto !important;
            overflow:auto !important;
        }
        .gradio-container,
        .main {
            height:auto !important;
            min-height:0 !important;
            overflow:visible !important;
        }
        .sidebar-box {
            position:static !important;
            top:auto !important;
            width:auto !important;
            max-width:none !important;
            height:auto !important;
            max-height:none !important;
            overflow:visible !important;
        }
        .sidebar-scroll,
        .gallery-pane {
            max-height:none !important;
            overflow:visible !important;
        }
        .sidebar-scroll { padding-right:0 !important; }
    }
    """

    with gr.Blocks(title=APP_WINDOW_TITLE) as demo:
        gr.HTML("""
<div class='app-header'>
  <h1>{title}</h1>
  <div class='header-meta'>{tag} &middot; created by vangel</div>
</div>
""".format(title=APP_NAME, tag=APP_GITHUB_TAG))

        with gr.Row(equal_height=False, elem_classes=["app-shell"]):
            with gr.Column(scale=1, min_width=300, elem_classes=["sidebar-box"]):
                thumb_action = gr.Textbox(value="", visible="hidden", elem_id="hy-thumb-action")
                hist_width_tb = gr.Textbox(value="300", visible="hidden", elem_id="hy-hist-width")
                shortcut_action = gr.Textbox(value="", visible="hidden", elem_id="hy-shortcut-action")
                mark_state = gr.Textbox(value='{"left":[],"right":[]}', visible="hidden", elem_id="hy-mark-state")
                model_status_state = gr.Textbox(value=promptmatch_model_status_json(), visible="hidden", elem_id="hy-model-status")
                with gr.Group(elem_classes=["sidebar-scroll"]):
                    with gr.Accordion("1. Setup", open=True, elem_id="hy-acc-setup"):
                        method_dd = gr.Dropdown(
                            choices=[METHOD_PROMPTMATCH, METHOD_IMAGEREWARD, METHOD_LLMSEARCH, METHOD_TAGMATCH],
                            value=METHOD_PROMPTMATCH,
                            label="Method",
                            elem_id="hy-method",
                        )
                        method_note = gr.Markdown(
                            "PromptMatch sorts by text-image similarity. Use a positive prompt and optional negative prompt. Fragment weights like (blonde:1.2) are supported.",
                            elem_classes=["method-note"],
                        )
                        folder_input = gr.Textbox(
                            value=source_dir,
                            label="Image folder - paste a path here",
                            lines=1,
                            placeholder=folder_placeholder(),
                            elem_id="hy-folder",
                        )
                        load_folder_btn = gr.Button("Load folder", elem_id="hy-load-folder")

                    with gr.Accordion("2. SCORING & Method/Settings", open=True, elem_id="hy-acc-scoring"):
                        with gr.Group(visible=True) as promptmatch_group:
                            model_dd = gr.Dropdown(
                                choices=promptmatch_model_dropdown_choices(),
                                value=label_for_backend(prompt_backend),
                                label="PromptMatch model",
                                elem_id="hy-model",
                            )
                            pos_prompt_tb = gr.Textbox(value=SEARCH_PROMPT, label="Positive prompt", lines=1, elem_id="hy-pos")
                            neg_prompt_tb = gr.Textbox(value=NEGATIVE_PROMPT, label="Negative prompt", lines=1, elem_id="hy-neg")
                            promptmatch_run_btn = gr.Button("Run scoring", elem_id="hy-run-pm", variant="primary")

                        with gr.Group(visible=False) as imagereward_group:
                            ir_prompt_tb = gr.Textbox(value=IR_PROMPT, label="ImageReward positive prompt", lines=3, elem_id="hy-ir-pos")
                            ir_negative_prompt_tb = gr.Textbox(
                                value=DEFAULT_IR_NEGATIVE_PROMPT,
                                label="Experimental penalty prompt",
                                lines=2,
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

                        with gr.Group(visible=False) as llmsearch_group:
                            llm_model_dd = gr.Dropdown(
                                choices=promptmatch_model_dropdown_choices(),
                                value=label_for_backend(prompt_backend),
                                label="PromptMatch shortlist model",
                                elem_id="hy-llm-model",
                            )
                            llm_backend_dd = gr.Dropdown(
                                choices=llmsearch_backend_choices(),
                                value=DEFAULT_LLMSEARCH_BACKEND,
                                label="Vision LM backend",
                                elem_id="hy-llm-backend",
                            )
                            llm_prompt_tb = gr.Textbox(
                                value=LLMSEARCH_DEFAULT_PROMPT,
                                label="LM search prompt",
                                lines=2,
                                elem_id="hy-llm-prompt",
                            )
                            llm_shortlist_slider = gr.Slider(
                                minimum=LLMSEARCH_SHORTLIST_MIN,
                                maximum=LLMSEARCH_SHORTLIST_MAX,
                                value=LLMSEARCH_SHORTLIST_DEFAULT,
                                step=1,
                                label="Shortlist size",
                                elem_id="hy-llm-shortlist",
                            )
                            llmsearch_run_btn = gr.Button("Run scoring", elem_id="hy-run-llm", variant="primary")

                        with gr.Group(visible=False) as tagmatch_group:
                            tagmatch_tags_tb = gr.Textbox(
                                value=TAGMATCH_DEFAULT_TAGS,
                                label="Tags to detect (comma-separated booru-style tags)",
                                lines=3,
                                elem_id="hy-tagmatch-tags",
                            )
                            tagmatch_run_btn = gr.Button("Run scoring", elem_id="hy-run-tagmatch", variant="primary")

                    with gr.Accordion("3. Actions from preview image", open=False, elem_id="hy-acc-prompt"):
                        with gr.Column(elem_classes=["preview-action-stack"]):
                            find_same_person_btn = gr.Button("Find same person", elem_id="hy-find-same-person")
                            find_similar_btn = gr.Button("Find similar images", elem_id="hy-find-similar")
                        with gr.Group(elem_classes=["preview-prompt-group"]):
                            gr.Markdown("**Prompt from image**", elem_classes=["method-note"])
                            prompt_generator_dd = gr.Dropdown(
                                choices=list(PROMPT_GENERATOR_ALL_CHOICES),
                                value=state["prompt_generator"],
                                label="Prompt generator",
                                elem_id="hy-prompt-generator",
                            )
                            generate_prompt_btn = gr.Button("Prompt from image", elem_id="hy-generate-prompt")
                            promptgen_status_md = gr.Markdown(
                                state["generated_prompt_status"],
                                elem_classes=["promptgen-status"],
                                elem_id="hy-promptgen-status",
                            )
                            generated_prompt_tb = gr.Textbox(
                                value=state["generated_prompt"],
                                label="Generated prompt",
                                lines=4,
                                placeholder="Preview an image, then generate an editable prompt here.",
                                elem_id="hy-generated-prompt",
                            )
                            generated_prompt_detail_slider = gr.Slider(
                                minimum=1,
                                maximum=3,
                                value=DEFAULT_GENERATED_PROMPT_DETAIL,
                                step=1,
                                label="Prompt detail",
                                elem_id="hy-generated-prompt-detail",
                            )
                            insert_prompt_btn = gr.Button("Insert into active prompt", elem_id="hy-insert-prompt")

                    with gr.Accordion("4. Thresholds", open=True, elem_id="hy-acc-thresholds"):
                        hist_plot = gr.Image(value=None, show_label=False, interactive=False, elem_classes=["hist-img"], elem_id="hy-hist")
                        with gr.Row(equal_height=False, elem_classes=["threshold-row"]):
                            main_slider = gr.Slider(
                                minimum=-1.0,
                                maximum=1.0,
                                value=0.14,
                                step=0.001,
                                label=threshold_labels(METHOD_PROMPTMATCH)[0],
                                elem_id="hy-main-slider",
                                buttons=[],
                            )
                            with gr.Column(scale=0, min_width=58, elem_classes=["threshold-actions"]):
                                main_mid_btn = gr.Button("50%", elem_id="hy-main-mid", scale=0, min_width=58, elem_classes=["threshold-mid"])
                        with gr.Row(equal_height=False, elem_classes=["threshold-row"]):
                            aux_slider = gr.Slider(
                                minimum=-1.0,
                                maximum=1.0,
                                value=NEGATIVE_THRESHOLD,
                                step=0.001,
                                label=threshold_labels(METHOD_PROMPTMATCH)[1],
                                elem_id="hy-aux-slider",
                                buttons=[],
                            )
                            aux_mid_btn = gr.Button("50%", elem_id="hy-aux-mid", scale=0, min_width=50, visible=True, elem_classes=["threshold-mid"])
                        keep_pm_thresholds_cb = gr.Checkbox(
                            value=True,
                            label="Keep PromptMatch thresholds on prompt reruns",
                            visible=True,
                            elem_id="hy-keep-pm-thresholds",
                        )
                        keep_ir_thresholds_cb = gr.Checkbox(
                            value=True,
                            label="Keep ImageReward threshold on prompt reruns",
                            visible=False,
                            elem_id="hy-keep-ir-thresholds",
                        )
                        with gr.Row(equal_height=False, elem_classes=["threshold-row"]):
                            percentile_slider = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=50,
                                step=1,
                                label=percentile_slider_label(METHOD_PROMPTMATCH),
                                elem_id="hy-percentile",
                                buttons=[],
                            )
                            percentile_mid_btn = gr.Button("50%", elem_id="hy-percentile-mid", scale=0, min_width=50, elem_classes=["threshold-mid"])
                        proxy_display_cb = gr.Checkbox(value=True, label="Use proxies for gallery display", elem_id="hy-use-proxy-display")
                        status_md = gr.Markdown("", elem_classes=["status-md"])

                    with gr.Accordion("5. Export", open=False, elem_id="hy-acc-export") as export_acc:
                        with gr.Row(equal_height=False, elem_classes=["export-options-row"]):
                            move_export_cb = gr.Checkbox(
                                value=False,
                                label="Move instead of copy",
                                container=False,
                                scale=0,
                                min_width=150,
                                elem_id="hy-export-move-enabled",
                                elem_classes=["gallery-export-toggle", "export-move-toggle"],
                            )
                        export_btn = gr.Button("Export folders", elem_id="hy-export", variant="primary")
                        export_tb = gr.Textbox(label="Export result", lines=3, interactive=False)

            with gr.Column(scale=5, elem_classes=["gallery-pane"]):
                with gr.Row(equal_height=False, elem_classes=["gallery-topbar"]):
                    with gr.Column(scale=1, elem_classes=["gallery-side", "gallery-header-slot"]):
                        with gr.Row(equal_height=False, elem_classes=["gallery-head-row"]):
                            left_export_cb = gr.Checkbox(
                                value=True,
                                label="Export",
                                container=False,
                                scale=0,
                                min_width=62,
                                elem_id="hy-export-left-enabled",
                                elem_classes=["gallery-export-toggle"],
                            )
                            gr.HTML("", elem_classes=["gallery-head-fill"])
                            left_export_name_tb = gr.Textbox(value="selected", show_label=False, container=False, lines=1, elem_id="hy-export-left-name", elem_classes=["gallery-export-name"])
                            left_head = gr.Markdown("**0 images**", elem_classes=["gallery-count"])
                    with gr.Column(scale=0, min_width=100, elem_classes=["gallery-header-spacer"]) as gallery_header_spacer_col:
                        gr.HTML("")
                    with gr.Column(scale=1, elem_classes=["gallery-side", "gallery-header-slot"]) as right_header_col:
                        with gr.Row(equal_height=False, elem_classes=["gallery-head-row"]):
                            right_export_cb = gr.Checkbox(
                                value=True,
                                label="Export",
                                container=False,
                                scale=0,
                                min_width=62,
                                elem_id="hy-export-right-enabled",
                                elem_classes=["gallery-export-toggle"],
                            )
                            gr.HTML("", elem_classes=["gallery-head-fill"])
                            right_export_name_tb = gr.Textbox(value="rejected", show_label=False, container=False, lines=1, elem_id="hy-export-right-name", elem_classes=["gallery-export-name"])
                            right_head = gr.Markdown("**0 images**", elem_classes=["gallery-count"])
                    with gr.Column(scale=0, elem_classes=["gallery-zoom-slot"]):
                        with gr.Row(equal_height=False, elem_classes=["zoom-inline-wrap"]):
                            gr.Markdown("Tile Size", elem_classes=["zoom-inline-label"])
                            zoom_slider = gr.Slider(minimum=2, maximum=10, value=7, step=1, label="Thumbnail count", show_label=False, container=False, elem_id="hy-zoom")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes=["gallery-side"]):
                        left_gallery = gr.Gallery(show_label=False, columns=5, height="calc(100vh - 130px)", object_fit="contain", preview=True, allow_preview=True, elem_classes=["grid-wrap"], elem_id="hy-left-gallery")
                    with gr.Column(scale=0, min_width=100, elem_classes=["move-col"]) as move_controls_col:
                        sel_info = gr.Markdown("Shift+click thumbnails to mark multiple images.", elem_classes=["sel-info"])
                        move_right_btn = gr.Button("Move >>", elem_id="hy-move-right")
                        fit_threshold_btn = gr.Button("Fit thresh to filter image", elem_id="hy-fit-threshold")
                        pin_selected_btn = gr.Button("Pin selected", elem_id="hy-pin-selected")
                        move_left_btn = gr.Button("<< Move", elem_id="hy-move-left")
                        clear_status_btn = gr.Button("Clear marked", elem_id="hy-clear-status")
                        clear_all_status_btn = gr.Button("Clear all", elem_id="hy-clear-all-status")
                    with gr.Column(scale=1, elem_classes=["gallery-side"]) as right_gallery_col:
                        right_gallery = gr.Gallery(show_label=False, columns=5, height="calc(100vh - 130px)", object_fit="contain", preview=True, allow_preview=True, elem_classes=["grid-wrap"], elem_id="hy-right-gallery")

        method_dd.change(
            fn=configure_controls,
            inputs=[method_dd],
            outputs=[promptmatch_group, imagereward_group, llmsearch_group, tagmatch_group, main_slider, aux_slider, aux_mid_btn, keep_pm_thresholds_cb, keep_ir_thresholds_cb, percentile_slider, percentile_mid_btn, method_note],
        )

        _score_folder_inputs = [method_dd, folder_input, model_dd, pos_prompt_tb, neg_prompt_tb, ir_prompt_tb, ir_negative_prompt_tb, ir_penalty_weight_tb, llm_model_dd, llm_prompt_tb, llm_backend_dd, llm_shortlist_slider, tagmatch_tags_tb, main_slider, aux_slider, keep_pm_thresholds_cb, keep_ir_thresholds_cb]
        _score_folder_outputs = [left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider, percentile_slider, percentile_mid_btn, model_status_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col]

        promptmatch_run_btn.click(fn=score_folder, inputs=_score_folder_inputs, outputs=_score_folder_outputs)
        imagereward_run_btn.click(fn=score_folder, inputs=_score_folder_inputs, outputs=_score_folder_outputs)
        llmsearch_run_btn.click(fn=score_folder, inputs=_score_folder_inputs, outputs=_score_folder_outputs)
        tagmatch_run_btn.click(fn=score_folder, inputs=_score_folder_inputs, outputs=_score_folder_outputs)
        folder_input.submit(
            fn=load_folder_for_browse,
            inputs=[folder_input, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col],
        )
        load_folder_btn.click(
            fn=load_folder_for_browse,
            inputs=[folder_input, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col],
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
        find_similar_btn.click(
            fn=find_similar_images,
            inputs=[folder_input, model_dd, main_slider, aux_slider],
            outputs=[promptgen_status_md, left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider, percentile_slider, percentile_mid_btn, model_status_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col],
        )
        find_same_person_btn.click(
            fn=find_same_person_images,
            inputs=[folder_input, main_slider, aux_slider],
            outputs=[promptgen_status_md, left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider, percentile_slider, percentile_mid_btn, model_status_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col],
        )
        generated_prompt_detail_slider.change(
            fn=update_generated_prompt_detail,
            inputs=[prompt_generator_dd, generated_prompt_detail_slider, generated_prompt_tb],
            outputs=[promptgen_status_md, generated_prompt_tb],
        )
        insert_prompt_btn.click(
            fn=insert_generated_prompt,
            inputs=[method_dd, generated_prompt_tb],
            outputs=[promptgen_status_md, pos_prompt_tb, ir_prompt_tb, llm_prompt_tb],
        )

        main_slider.input(fn=update_histogram_only, inputs=[main_slider, aux_slider], outputs=[hist_plot], queue=False)
        aux_slider.input(fn=update_histogram_only, inputs=[main_slider, aux_slider], outputs=[hist_plot], queue=False)
        main_slider.release(fn=update_split, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col], queue=False)
        aux_slider.release(fn=update_split, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col], queue=False)
        percentile_slider.input(
            fn=update_histogram_from_percentile,
            inputs=[percentile_slider, aux_slider],
            outputs=[hist_plot],
            queue=False,
        )
        percentile_slider.release(
            fn=set_from_percentile,
            inputs=[percentile_slider, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider],
            queue=False,
        )
        main_mid_btn.click(
            fn=reset_main_threshold_to_middle,
            inputs=[main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider],
        )
        aux_mid_btn.click(
            fn=reset_aux_threshold_to_middle,
            inputs=[main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider],
        )
        percentile_mid_btn.click(
            fn=reset_percentile_to_middle,
            inputs=[main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, percentile_slider],
        )
        zoom_slider.change(
            fn=update_zoom,
            inputs=[zoom_slider, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col],
        )
        proxy_display_cb.change(
            fn=update_proxy_display,
            inputs=[proxy_display_cb, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col],
        )
        ir_penalty_weight_tb.change(
            fn=update_imagereward_penalty_weight,
            inputs=[ir_penalty_weight_tb, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider],
        )
        shortcut_action.change(
            fn=handle_shortcut_action,
            inputs=[shortcut_action, method_dd, folder_input, model_dd, pos_prompt_tb, neg_prompt_tb, ir_prompt_tb, ir_negative_prompt_tb, ir_penalty_weight_tb, llm_model_dd, llm_prompt_tb, llm_backend_dd, llm_shortlist_slider, tagmatch_tags_tb, main_slider, aux_slider, keep_pm_thresholds_cb, keep_ir_thresholds_cb],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, main_slider, aux_slider, percentile_slider, percentile_mid_btn, model_status_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col],
        )
        model_status_state.change(
            fn=refresh_promptmatch_model_dropdown,
            inputs=[model_dd],
            outputs=[model_dd],
        )
        model_status_state.change(
            fn=refresh_promptmatch_model_dropdown,
            inputs=[llm_model_dd],
            outputs=[llm_model_dd],
        )
        hist_width_tb.change(fn=handle_hist_width, inputs=[hist_width_tb, main_slider, aux_slider], outputs=[hist_plot])
        thumb_action.change(
            fn=handle_thumb_action,
            inputs=[thumb_action, main_slider, aux_slider],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider],
        )
        move_right_btn.click(fn=move_right, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col])
        fit_threshold_btn.click(fn=fit_threshold_to_targets, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider])
        pin_selected_btn.click(fn=pin_selected, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col])
        move_left_btn.click(fn=move_left, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col])
        clear_status_btn.click(fn=clear_status, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col])
        clear_all_status_btn.click(fn=clear_all_status, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col])
        hist_plot.select(fn=on_hist_click, inputs=[main_slider, aux_slider], outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, main_slider, aux_slider])
        export_btn.click(
            fn=export_files,
            inputs=[main_slider, aux_slider, left_export_cb, right_export_cb, move_export_cb, left_export_name_tb, right_export_name_tb],
            outputs=[left_head, left_gallery, right_head, right_gallery, status_md, hist_plot, sel_info, mark_state, left_export_cb, left_export_name_tb, export_acc, move_controls_col, gallery_header_spacer_col, right_header_col, right_gallery_col, export_tb],
        )

    return demo, css, tooltip_head(tooltips)


def create_setup_required_app(requirement_issues):
    issue_lines = "\n".join(f"    <li><code>{issue}</code></li>" for issue in requirement_issues) or "    <li><code>Unknown dependency mismatch</code></li>"
    css = """
    body, .gradio-container { background:#120d0d !important; color:#f4eaea !important; }
    .gradio-container { max-width:960px !important; margin:0 auto !important; padding:24px 16px 40px !important; }
    .setup-alert {
        border:2px solid #d86161;
        border-radius:14px;
        background:linear-gradient(180deg, rgba(62, 19, 19, 0.96), rgba(28, 11, 11, 0.98));
        box-shadow:0 0 0 2px rgba(255,255,255,0.03) inset, 0 18px 50px rgba(0,0,0,0.35);
        padding:22px 22px 18px;
    }
    .setup-alert h1 {
        margin:0 0 12px 0;
        color:#ffb3b3;
        font-size:2rem;
        line-height:1.1;
        font-family:monospace;
        text-transform:uppercase;
        letter-spacing:.05em;
    }
    .setup-alert p, .setup-alert li, .setup-alert code {
        font-family:monospace !important;
        font-size:1rem !important;
        line-height:1.55 !important;
        color:#f7eaea !important;
    }
    .setup-alert .cmd {
        margin:14px 0;
        padding:12px 14px;
        border-radius:10px;
        background:#0f0b0b;
        border:1px solid #7d3d3d;
        color:#fff1f1;
    }
    .setup-alert .soft {
        color:#e3c9c9 !important;
    }
    """
    body = f"""
<div class="setup-alert">
  <h1>Setup Update Required</h1>
  <p>The app code was updated, but the current <code>venv312</code> does not match the packages this version expects.</p>
  <p>Please rerun setup before using this version:</p>
  <div class="cmd"><code>{SETUP_SCRIPT_HINT}</code></div>
  <p>Dependency issues found:</p>
  <ul class="soft">
{issue_lines}
  </ul>
  <p class="soft">After setup finishes, restart the launcher and reload the page.</p>
</div>
"""
    with gr.Blocks(title=APP_WINDOW_TITLE) as demo:
        gr.HTML(body)
    return demo, css, ""


if __name__ == "__main__":
    requirement_issues = runtime_requirement_issues()
    if requirement_issues:
        print("[Startup check] Dependency mismatch detected. Please rerun setup.")
        for issue in requirement_issues:
            print(f"  - {issue}")
        app, css, head = create_setup_required_app(requirement_issues)
    else:
        app, css, head = create_app()
    port = resolve_server_port(7862, "HYBRIDSELECTOR_PORT")
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
