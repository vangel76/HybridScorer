import os
import gc
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .backend import ModelBackend
from .config import (
    METHOD_IMAGEREWARD,
    METHOD_SAMEPERSON,
    METHOD_TAGMATCH,
    METHOD_OBJECTSEARCH,
    DINOV2_MODEL_ID,
    DINOV2_INPUT_SIZE,
    DINOV2_PATCH_DIM,
    DINOV2_PATCHES_PER_IMAGE,
    DINOV2_BASE_BATCH_SIZE,
    DINOV2_MAX_BATCH_SIZE,
    DINOV2_BATCH_AGGRESSION,
    DINOV2_MIN_FREE_VRAM_GB,
    OBJECTSEARCH_FAISS_TOP_K_PATCHES,
    PROMPT_GENERATOR_FLORENCE,
    PROMPT_GENERATOR_JOYCAPTION,
    PROMPT_GENERATOR_HUIHUI_GEMMA4,
    PROMPT_GENERATOR_JOYCAPTION_GGUF,
    FLORENCE_MODEL_ID,
    JOYCAPTION_MODEL_ID,
    HUIHUI_GEMMA4_MODEL_ID,
    HUIHUI_GEMMA4_PROCESSOR_MODEL_ID,
    JOYCAPTION_GGUF_REPO_ID,
    JOYCAPTION_GGUF_FILENAME,
    JOYCAPTION_GGUF_MMPROJ_FILENAME,
    JOYCAPTION_GGUF_SETUP_HINT,
    TAGMATCH_WD_REPO_ID,
    TAGMATCH_WD_MODEL_FILE,
    TAGMATCH_WD_TAGS_FILE,
    TAGMATCH_WD_IMAGE_SIZE,
    FACE_MODEL_PACK,
    FACE_DET_SIZE,
    FACE_MODEL_LABEL,
    PROMPTMATCH_PROXY_PROGRESS_SHARE,
    get_cache_config,
)
from .utils import (
    get_imagereward_utils,
    cuda_prefers_bfloat16,
    describe_florence_source,
    describe_prompt_generator_source,
    describe_insightface_source,
    describe_openai_clip_source,
    describe_openclip_source,
    describe_siglip_source,
    face_embedding_worker_count,
    current_free_vram_gb,
    get_image_paths_signature,
    prepare_promptmatch_proxies,
    load_promptmatch_rgb_images,
)
from .helpers import get_model_config, label_for_backend
from .scoring import encode_all_promptmatch_images
from .state_helpers import can_reuse_proxy_map


def _clear_torch_model(model):
    if model is None:
        return
    try:
        model.to("cpu")
    except Exception:
        pass


def _cached_hf_backend(state, key):
    cached = state["prompt_backend_cache"].get(key)
    if cached and cached.get("model") is not None and cached.get("processor") is not None:
        return cached["model"], cached["processor"]
    return None


def _load_tagmatch_tags(tags_path):
    import csv as _csv
    tags = []
    with open(tags_path, newline="", encoding="utf-8") as _f:
        reader = _csv.DictReader(_f)
        for row in reader:
            name = (row.get("name") or "").strip().lower()
            if name:
                tags.append(name)
    return tags


def ensure_imagereward_model(state):
    if state["ir_model"] is None:
        state["ir_model"] = get_imagereward_utils().load(
            "ImageReward-v1.0",
            download_root=get_cache_config()["imagereward_dir"],
        )
    return state["ir_model"]


def release_inactive_gpu_models(state, target_method):
    released = []

    if target_method != METHOD_IMAGEREWARD and state.get("ir_model") is not None:
        _clear_torch_model(state["ir_model"])
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

    if target_method != METHOD_OBJECTSEARCH and state.get("dinov2_backend") is not None:
        dinov2 = state["dinov2_backend"]
        if isinstance(dinov2, dict):
            _clear_torch_model(dinov2.get("model"))
        state["dinov2_backend"] = None
        state["os_cached_faiss_index"] = None
        state["os_cached_patch_gpu_tensor"] = None
        released.append("DINOv2")

    if state.get("prompt_backend_cache"):
        for backend_name, cached in list(state["prompt_backend_cache"].items()):
            if isinstance(cached, dict):
                _clear_torch_model(cached.get("model"))
            released.append(backend_name)
        state["prompt_backend_cache"] = {}
        # Drop loop variables so CPython's refcount immediately reaches zero for
        # llama_cpp Llama objects (which have no "model" key and are freed by __del__).
        del backend_name, cached

    if released:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[GPU] Released inactive models: {', '.join(released)}")


def ensure_florence_model(state, device):
    hit = _cached_hf_backend(state, PROMPT_GENERATOR_FLORENCE)
    if hit:
        return hit

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


def ensure_joycaption_model(state, device):
    hit = _cached_hf_backend(state, PROMPT_GENERATOR_JOYCAPTION)
    if hit:
        return hit

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


def ensure_huihui_gemma4_model(state, device):
    hit = _cached_hf_backend(state, PROMPT_GENERATOR_HUIHUI_GEMMA4)
    if hit:
        return hit

    try:
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Huihui Gemma 4 prompt generation needs a newer transformers build with Gemma 4 multimodal support.\n"
            "Run setup again after updating requirements."
        ) from exc

    if not all(
        hasattr(transformers, name)
        for name in ("Gemma4Processor", "Gemma4ImageProcessor", "Gemma4ForConditionalGeneration")
    ):
        raise RuntimeError(
            "The installed transformers build does not include Gemma 4 runtime classes yet.\n"
            f"Detected transformers=={getattr(transformers, '__version__', 'unknown')}.\n"
            "This environment has Gemma 3 classes but not Gemma 4, so Huihui Gemma 4 cannot run here.\n"
            "Update transformers to a Gemma-4-capable release by rerunning setup after refreshing requirements."
        )

    dtype = torch.bfloat16 if cuda_prefers_bfloat16() else torch.float16
    local_files_only = describe_prompt_generator_source(PROMPT_GENERATOR_HUIHUI_GEMMA4) == "disk cache"
    print(f"[Huihui Gemma 4] Loading {HUIHUI_GEMMA4_MODEL_ID} …")
    processor_repo_id = HUIHUI_GEMMA4_MODEL_ID
    processor_load_errors = []
    try:
        processor = AutoProcessor.from_pretrained(
            HUIHUI_GEMMA4_MODEL_ID,
            padding_side="left",
            local_files_only=local_files_only,
            trust_remote_code=True,
            cache_dir=get_cache_config()["huggingface_dir"],
        )
    except Exception as primary_exc:
        processor_load_errors.append(f"{HUIHUI_GEMMA4_MODEL_ID}: {primary_exc}")
        processor_repo_id = HUIHUI_GEMMA4_PROCESSOR_MODEL_ID
        try:
            processor = AutoProcessor.from_pretrained(
                HUIHUI_GEMMA4_PROCESSOR_MODEL_ID,
                padding_side="left",
                local_files_only=local_files_only,
                trust_remote_code=True,
                cache_dir=get_cache_config()["huggingface_dir"],
            )
            print(
                "[Huihui Gemma 4] Repo processor metadata was incompatible; "
                f"reusing processor assets from {HUIHUI_GEMMA4_PROCESSOR_MODEL_ID}."
            )
        except Exception as fallback_exc:
            processor_load_errors.append(f"{HUIHUI_GEMMA4_PROCESSOR_MODEL_ID}: {fallback_exc}")
            raise RuntimeError(
                "Huihui Gemma 4 processor loading failed.\n"
                "Tried the finetuned repo processor assets and the base Gemma 4 processor assets.\n"
                + "\n".join(processor_load_errors)
            ) from fallback_exc
    model = AutoModelForImageTextToText.from_pretrained(
        HUIHUI_GEMMA4_MODEL_ID,
        dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=True,
        cache_dir=get_cache_config()["huggingface_dir"],
    ).to(device)
    model.eval()
    state["prompt_backend_cache"][PROMPT_GENERATOR_HUIHUI_GEMMA4] = {
        "model": model,
        "processor": processor,
        "processor_repo_id": processor_repo_id,
    }
    print("[Huihui Gemma 4] Ready.")
    return model, processor


def ensure_joycaption_gguf_model(state, device):
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


def ensure_tagmatch_model(state, device):
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

    tags = _load_tagmatch_tags(tags_path)
    backend = {"session": session, "tags": tags, "input_name": input_name}
    state["tagmatch_backend"] = backend
    print(f"[TagMatch] Ready — {len(tags)} tags loaded.")
    return backend


def load_tagmatch_vocabulary(state):
    cached = state.get("tagmatch_vocab_tags")
    if cached:
        return cached

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for TagMatch tag suggestions.") from exc

    tags_path = hf_hub_download(
        repo_id=TAGMATCH_WD_REPO_ID,
        filename=TAGMATCH_WD_TAGS_FILE,
        cache_dir=get_cache_config()["huggingface_dir"],
    )

    tags = _load_tagmatch_tags(tags_path)
    state["tagmatch_vocab_tags"] = tags
    state["tagmatch_vocab_json"] = json.dumps(tags)
    print(f"[TagMatch] Autocomplete vocabulary ready — {len(tags)} tags loaded.")
    return tags


def refresh_tagmatch_vocab_state(state, method):
    cached_json = state.get("tagmatch_vocab_json") or "[]"
    if method != METHOD_TAGMATCH:
        return cached_json
    try:
        load_tagmatch_vocabulary(state)
    except Exception as exc:
        print(f"[TagMatch] Autocomplete vocabulary unavailable: {exc}")
        return cached_json
    return state.get("tagmatch_vocab_json") or "[]"


def ensure_promptmatch_backend_loaded(state, device, model_label, progress):
    cfg = get_model_config(model_label)
    if cfg is None:
        raise RuntimeError(f"Unknown PromptMatch model: {model_label}")
    _, backend_name, kwargs = cfg
    if state.get("backend") is None or label_for_backend(state["backend"]) != model_label:
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


def ensure_face_backend_loaded(state, progress):
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


def ensure_promptmatch_feature_cache(state, device, image_paths, model_label, progress, reuse_desc, encode_desc, progress_label):
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
    elif cache_dir and can_reuse_proxy_map(state, image_paths, image_signature):
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


def encode_single_promptmatch_image(state, query_path):
    loaded_items, pil_imgs = load_promptmatch_rgb_images([(query_path, query_path)])
    if not loaded_items or not pil_imgs:
        raise RuntimeError(f"{os.path.basename(query_path)} could not be loaded for similarity search.")
    features = state["backend"].encode_images_batch(pil_imgs)
    if features is None or not hasattr(features, "shape") or int(features.shape[0]) < 1:
        raise RuntimeError(f"{os.path.basename(query_path)} could not be encoded for similarity search.")
    return features[0:1].detach().float().cpu()


def choose_primary_face(faces):
    if not faces:
        return None

    def face_area(face):
        bbox = getattr(face, "bbox", None)
        if bbox is None or len(bbox) < 4:
            return 0.0
        return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

    return max(faces, key=face_area)


def ensure_face_feature_cache(state, image_paths, progress):
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

    ensure_face_backend_loaded(state, progress)
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


def encode_single_face_embedding(state, query_path, progress):
    backend = ensure_face_backend_loaded(state, progress)
    try:
        with Image.open(query_path) as src_img:
            rgb = src_img.convert("RGB")
            bgr = np.asarray(rgb)[:, :, ::-1].copy()
    except Exception as exc:
        raise RuntimeError(f"{os.path.basename(query_path)} could not be opened for same-person search: {exc}") from exc

    primary_face = choose_primary_face(backend.get(bgr))
    if primary_face is None:
        raise RuntimeError(f"{os.path.basename(query_path)}: No face detected.")

    embedding = getattr(primary_face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(primary_face, "embedding", None)
        if embedding is None:
            raise RuntimeError(f"{os.path.basename(query_path)}: No face embedding returned.")

    return F.normalize(torch.as_tensor(embedding, dtype=torch.float32), dim=0).view(1, -1)


def ensure_dinov2_backend(state, device, progress=None):
    if state.get("dinov2_backend") is not None:
        if progress is not None:
            progress(0, desc=f"Using loaded DINOv2 model from memory: {DINOV2_MODEL_ID}")
        return state["dinov2_backend"]

    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        raise RuntimeError("transformers is required for DINOv2 but could not be imported.") from exc

    hf_cache = get_cache_config().get("huggingface_dir")
    cache_kwargs = {"cache_dir": hf_cache} if hf_cache else {}
    if progress is not None:
        progress(0, desc=f"Loading DINOv2 model: {DINOV2_MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID, **cache_kwargs)
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID, **cache_kwargs)
    model = model.to(device).eval()
    state["dinov2_backend"] = {"model": model, "processor": processor, "device": device}
    print(f"[DINOv2] Loaded {DINOV2_MODEL_ID} on {device}")
    return state["dinov2_backend"]


def _extract_dinov2_patches_batch(dinov2_backend, pil_images):
    model = dinov2_backend["model"]
    processor = dinov2_backend["processor"]
    device = dinov2_backend["device"]

    resized = [img.convert("RGB").resize((DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), Image.Resampling.BILINEAR) for img in pil_images]
    inputs = processor(images=resized, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    patch_tokens = outputs.last_hidden_state[:, 1:, :]
    patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
    return patch_tokens.cpu().numpy()


def ensure_objectsearch_feature_cache(state, device, image_paths, progress=None):
    image_signature = get_image_paths_signature(image_paths)
    can_reuse = (
        state.get("os_cached_signature") == image_signature
        and state.get("os_cached_model_id") == DINOV2_MODEL_ID
        and state.get("os_cached_feature_paths") is not None
        and state.get("os_cached_patch_features") is not None
        and state.get("os_cached_patch_image_idx") is not None
        and state.get("os_cached_faiss_index") is not None
        and state.get("os_cached_failures") is not None
    )
    if can_reuse:
        if progress is not None:
            progress(0, desc=f"Reusing cached DINOv2 patch embeddings for {len(image_paths)} images")
        return (
            image_signature,
            list(state["os_cached_feature_paths"]),
            state["os_cached_patch_features"],
            state["os_cached_patch_image_idx"],
            state["os_cached_faiss_index"],
            dict(state["os_cached_failures"]),
        )

    ensure_dinov2_backend(state, device, progress)
    dinov2_backend = state["dinov2_backend"]

    vram_info = current_free_vram_gb()
    free_gb = vram_info[0] if vram_info else None
    batch_size = DINOV2_BASE_BATCH_SIZE
    if free_gb is not None and free_gb > DINOV2_MIN_FREE_VRAM_GB:
        extra = free_gb - DINOV2_MIN_FREE_VRAM_GB
        scaled = int(DINOV2_BASE_BATCH_SIZE * (1 + extra * DINOV2_BATCH_AGGRESSION / 8.0))
        batch_size = min(DINOV2_MAX_BATCH_SIZE, max(DINOV2_BASE_BATCH_SIZE, scaled))

    total = len(image_paths)
    feature_paths = []
    all_patches = []
    patch_image_idx = []
    failures = {}
    completed = 0

    if progress is not None:
        progress(0, desc=f"Extracting DINOv2 patch embeddings 0/{total} (batch {batch_size})")

    for batch_start in range(0, total, batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        pil_batch = []
        valid_paths = []
        for path in batch_paths:
            try:
                with Image.open(path) as src:
                    pil_batch.append(src.convert("RGB").copy())
                valid_paths.append(path)
            except Exception as exc:
                failures[path] = str(exc) or "Could not open image."

        if pil_batch:
            try:
                patches = _extract_dinov2_patches_batch(dinov2_backend, pil_batch)
            except Exception as exc:
                for path in valid_paths:
                    if path not in failures:
                        failures[path] = str(exc) or "DINOv2 extraction failed."
                patches = None

            if patches is not None:
                for path, img_patches in zip(valid_paths, patches):
                    image_idx = len(feature_paths)
                    feature_paths.append(path)
                    all_patches.append(img_patches)
                    patch_image_idx.extend([image_idx] * img_patches.shape[0])

        completed += len(batch_paths)
        if progress is not None:
            progress(completed / max(total, 1), desc=f"Extracting DINOv2 patch embeddings {completed}/{total}")

    if all_patches:
        flat_patches = np.concatenate(all_patches, axis=0).astype(np.float32)
        patch_image_idx_arr = np.array(patch_image_idx, dtype=np.int32)
    else:
        flat_patches = np.empty((0, DINOV2_PATCH_DIM), dtype=np.float32)
        patch_image_idx_arr = np.empty((0,), dtype=np.int32)

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required for ObjectSearch but is not installed. Run the setup script.") from exc

    index = faiss.IndexFlatIP(DINOV2_PATCH_DIM)
    if flat_patches.shape[0] > 0:
        index.add(flat_patches)

    gpu_tensor = None
    if torch.cuda.is_available() and flat_patches.shape[0] > 0:
        try:
            gpu_tensor = torch.from_numpy(flat_patches).to("cuda")
            print(f"[DINOv2] Patch gallery uploaded to GPU ({flat_patches.shape[0]} patches)")
        except Exception as exc:
            print(f"[DINOv2] GPU tensor upload failed ({exc}), will use FAISS CPU for search")

    state["os_cached_signature"] = image_signature
    state["os_cached_model_id"] = DINOV2_MODEL_ID
    state["os_cached_feature_paths"] = list(feature_paths)
    state["os_cached_patch_features"] = flat_patches
    state["os_cached_patch_image_idx"] = patch_image_idx_arr
    state["os_cached_faiss_index"] = index
    state["os_cached_patch_gpu_tensor"] = gpu_tensor
    state["os_cached_failures"] = dict(failures)

    print(f"[DINOv2] Patch cache built: {len(feature_paths)} images, {flat_patches.shape[0]} patches, {len(failures)} failures")
    return image_signature, feature_paths, flat_patches, patch_image_idx_arr, index, failures


def encode_single_objectsearch_query(state, device, query_path, progress=None):
    ensure_dinov2_backend(state, device, progress)
    try:
        with Image.open(query_path) as src:
            pil_img = src.convert("RGB").copy()
    except Exception as exc:
        raise RuntimeError(f"{os.path.basename(query_path)} could not be opened for object search: {exc}") from exc
    patches = _extract_dinov2_patches_batch(state["dinov2_backend"], [pil_img])
    return patches[0].astype(np.float32)
