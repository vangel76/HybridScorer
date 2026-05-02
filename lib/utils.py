import base64
import io
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from importlib import metadata
from importlib import import_module
from importlib.machinery import ModuleSpec

import torch
from PIL import Image

from .config import (
    STARTUP_EXTRA_REQUIREMENTS,
    ALLOWED_EXTENSIONS,
    PROMPTMATCH_PROXY_MAX_EDGE,
    PROMPTMATCH_PROXY_CACHE_ROOT,
    PROMPTMATCH_TORCH_THREAD_CAP,
    PROMPTMATCH_HOST_MAX_WORKERS,
    FACE_EMBEDDING_ABSOLUTE_MAX_WORKERS,
    FACE_EMBEDDING_MIN_FREE_VRAM_GB,
    FACE_EMBEDDING_PER_WORKER_VRAM_GB,
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
    IMAGEREWARD_BASE_BATCH_SIZE,
    IMAGEREWARD_MAX_BATCH_SIZE,
    IMAGEREWARD_BATCH_AGGRESSION,
    IMAGEREWARD_MIN_FREE_VRAM_GB,
    PROMPTMATCH_BASE_BATCH_SIZE,
    PROMPTMATCH_MAX_BATCH_SIZE,
    PROMPTMATCH_BATCH_AGGRESSION,
    PROMPTMATCH_MIN_FREE_VRAM_GB,
    TAGMATCH_BASE_BATCH_SIZE,
    TAGMATCH_MAX_BATCH_SIZE,
    TAGMATCH_BATCH_AGGRESSION,
    TAGMATCH_MIN_FREE_VRAM_GB,
    EXPLICIT_PROMPT_WEIGHT_RE,
    FLORENCE_MODEL_ID,
    JOYCAPTION_MODEL_ID,
    JOYCAPTION_NF4_MODEL_ID,
    HUIHUI_GEMMA4_MODEL_ID,
    TAGMATCH_WD_REPO_ID,
    TAGMATCH_WD_MODEL_FILE,
    TAGMATCH_WD_TAGS_FILE,
    FACE_MODEL_PACK,
    PROMPT_GENERATOR_FLORENCE,
    PROMPT_GENERATOR_JOYCAPTION,
    PROMPT_GENERATOR_JOYCAPTION_NF4,
    PROMPT_GENERATOR_HUIHUI_GEMMA4,
    PROMPT_GENERATOR_WD_TAGS,
    PROMPT_GENERATOR_CHOICES,
    get_cache_config,
)

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
    script_dir = os.path.dirname(script_dir)
    requirement_sources = [
        os.path.join(script_dir, "requirements.txt"),
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
        from transformers import BertTokenizer
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils

        if not hasattr(modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
            def _get_all_tied_weights_keys(self):
                mapping = getattr(self, "_all_tied_weights_keys", None)
                if mapping is None:
                    try:
                        mapping = self.get_expanded_tied_weights_keys(all_submodels=True)
                    except Exception:
                        static_mapping = getattr(self, "_tied_weights_keys", None)
                        mapping = dict(static_mapping) if isinstance(static_mapping, dict) else {}
                    self._all_tied_weights_keys = mapping
                return mapping

            def _set_all_tied_weights_keys(self, value):
                self._all_tied_weights_keys = value

            modeling_utils.PreTrainedModel.all_tied_weights_keys = property(
                _get_all_tied_weights_keys,
                _set_all_tied_weights_keys,
            )

        if not hasattr(modeling_utils.PreTrainedModel, "get_head_mask"):
            def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                if head_mask.dim() != 5:
                    raise AssertionError(f"head_mask.dim != 5, got {head_mask.dim()}")
                return head_mask.to(dtype=self.dtype)

            def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is not None:
                    head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                    if is_attention_chunked:
                        head_mask = head_mask.unsqueeze(-1)
                else:
                    head_mask = [None] * num_hidden_layers
                return head_mask

            modeling_utils.PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d
            modeling_utils.PreTrainedModel.get_head_mask = _get_head_mask

        for name in ("apply_chunking_to_forward", "prune_linear_layer"):
            if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
                setattr(modeling_utils, name, getattr(pytorch_utils, name))
        if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
            # Transformers 5.x removed this helper entirely, but ImageReward's BLIP code
            # still imports it from modeling_utils using the older signature.
            def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                mask = torch.ones(n_heads, head_size)
                heads = set(heads) - already_pruned_heads
                for head in heads:
                    head = head - sum(1 if pruned < head else 0 for pruned in already_pruned_heads)
                    mask[head] = 0
                mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(mask))[mask].long()
                return heads, index

            setattr(modeling_utils, "find_pruneable_heads_and_indices", _find_pruneable_heads_and_indices)

        if not getattr(BertTokenizer, "_hybridscorer_additional_special_tokens_patch", False):
            original_add_special_tokens = BertTokenizer.add_special_tokens

            def _patched_add_special_tokens(self, special_tokens_dict, *args, **kwargs):
                added = original_add_special_tokens(self, special_tokens_dict, *args, **kwargs)
                extra_tokens = special_tokens_dict.get("additional_special_tokens")
                if extra_tokens is not None:
                    remembered = list(getattr(self, "_hybridscorer_additional_special_tokens", []))
                    for token in extra_tokens:
                        token_text = str(token)
                        if token_text not in remembered:
                            remembered.append(token_text)
                    self._hybridscorer_additional_special_tokens = remembered
                return added

            BertTokenizer.add_special_tokens = _patched_add_special_tokens
            BertTokenizer.additional_special_tokens_ids = property(
                lambda self: [
                    self.convert_tokens_to_ids(token)
                    for token in getattr(self, "_hybridscorer_additional_special_tokens", [])
                ]
            )
            BertTokenizer._hybridscorer_additional_special_tokens_patch = True
    except Exception:
        pass

    package_dir = None
    for entry in sys.path:
        candidate = os.path.join(entry, "ImageReward")
        if os.path.isdir(candidate):
            package_dir = candidate
            break

    if package_dir is None:
        raise RuntimeError(
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
        utils_module = import_module(module_name)
        if not getattr(utils_module, "_hybridscorer_local_cache_patch", False):
            original_download = utils_module.ImageReward_download

            def _patched_imagereward_download(url, root):
                filename = os.path.basename(url)
                download_target = os.path.join(root, filename)
                if os.path.isfile(download_target):
                    return download_target
                return original_download(url, root)

            utils_module.ImageReward_download = _patched_imagereward_download
            utils_module._hybridscorer_local_cache_patch = True
        return utils_module
    except Exception as exc:
        raise RuntimeError(
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


def describe_tagmatch_source():
    cached = (
        huggingface_file_cached(TAGMATCH_WD_REPO_ID, TAGMATCH_WD_MODEL_FILE)
        and huggingface_file_cached(TAGMATCH_WD_REPO_ID, TAGMATCH_WD_TAGS_FILE)
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

    return max(1, min(hard_cap, vram_cap) // 2)


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
        return item, rgb

    loaded_items = []
    pil_imgs = []
    max_workers = promptmatch_host_worker_count(len(valid_items))
    if max_workers <= 1:
        results = [_load_one(item) for item in valid_items]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_load_one, valid_items))

    for item, image in results:
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


def _log_batch_timing(tag, prefix, batch_start, batch_end, total, timings):
    parts = []
    for key, value in timings.items():
        if value is None:
            continue
        if key == "free_vram_gb":
            parts.append(f"{key}={value:.1f}GB")
        else:
            parts.append(f"{key}={value:.1f}ms")
    if parts:
        print(f"[{tag}] {prefix} {batch_start}-{batch_end}/{total}  " + "  ".join(parts))


def promptmatch_log_batch_timing(prefix, batch_start, batch_end, total, timings):
    _log_batch_timing("PromptMatch", prefix, batch_start, batch_end, total, timings)


def imagereward_log_batch_timing(prefix, batch_start, batch_end, total, timings):
    _log_batch_timing("ImageReward", prefix, batch_start, batch_end, total, timings)


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
    if generator_name == PROMPT_GENERATOR_JOYCAPTION_NF4:
        return describe_huggingface_transformers_source(JOYCAPTION_NF4_MODEL_ID)
    if generator_name == PROMPT_GENERATOR_HUIHUI_GEMMA4:
        return describe_huggingface_transformers_source(HUIHUI_GEMMA4_MODEL_ID)
    if generator_name == PROMPT_GENERATOR_WD_TAGS:
        return "WD tagger ONNX (onnxruntime)"
    return "network or disk cache"


def prompt_generator_supports_torch_cleanup(generator_name):
    return generator_name in {
        PROMPT_GENERATOR_FLORENCE,
        PROMPT_GENERATOR_JOYCAPTION,
        PROMPT_GENERATOR_JOYCAPTION_NF4,
        PROMPT_GENERATOR_HUIHUI_GEMMA4,
    }


def prompt_backend_warning_text(generator_name):
    if generator_name == PROMPT_GENERATOR_HUIHUI_GEMMA4:
        return " This backend uses a less-filtered abliterated Gemma 4 model and may produce less-filtered text."
    return ""



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
    # Yield partial results so the UI job stream can update progress while large folders are scored.
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
