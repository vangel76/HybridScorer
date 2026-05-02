import os
import re
import tempfile
from functools import lru_cache

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
    "accelerate",
    "bitsandbytes",
)
SETUP_SCRIPT_HINT = "setup_update-windows.bat" if os.name == "nt" else "./setup_update-linux.sh"

METHOD_PROMPTMATCH = "PromptMatch"
METHOD_IMAGEREWARD = "ImageReward"
METHOD_LLMSEARCH = "LM Search"
METHOD_SIMILARITY = "Similarity"
METHOD_SAMEPERSON = "SamePerson"
METHOD_TAGMATCH = "TagMatch"
METHOD_OBJECTSEARCH = "ObjectSearch"
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
TAGMATCH_SLIDER_PREPROCESS_MIN = -0.01
TAGMATCH_DEFAULT_THRESHOLD = 20.0
HIST_HEIGHT_SCALE = 0.7
PROMPT_GENERATOR_FLORENCE = "Florence-2"
PROMPT_GENERATOR_JOYCAPTION = "JoyCaption Beta One"
PROMPT_GENERATOR_JOYCAPTION_NF4 = "JoyCaption Beta One NF4"
PROMPT_GENERATOR_HUIHUI_GEMMA4 = "Huihui Gemma 4 E4B"
PROMPT_GENERATOR_WD_TAGS = "WD Tags (ONNX)"
PROMPT_GENERATOR_CHOICES = (
    PROMPT_GENERATOR_FLORENCE,
    PROMPT_GENERATOR_JOYCAPTION,
    PROMPT_GENERATOR_JOYCAPTION_NF4,
    PROMPT_GENERATOR_HUIHUI_GEMMA4,
)
PROMPT_GENERATOR_ALL_CHOICES = (*PROMPT_GENERATOR_CHOICES, PROMPT_GENERATOR_WD_TAGS)
DEFAULT_PROMPT_GENERATOR = PROMPT_GENERATOR_FLORENCE
FLORENCE_MODEL_ID = "florence-community/Florence-2-base"
FLORENCE_MAX_NEW_TOKENS = 256
JOYCAPTION_MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
JOYCAPTION_NF4_MODEL_ID = "John6666/llama-joycaption-beta-one-hf-llava-nf4"
JOYCAPTION_MAX_NEW_TOKENS = 320
HUIHUI_GEMMA4_MODEL_ID = "huihui-ai/Huihui-gemma-4-E4B-it-abliterated"
HUIHUI_GEMMA4_PROCESSOR_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_GENERATED_PROMPT_DETAIL = 2
DEFAULT_LLMSEARCH_BACKEND = PROMPT_GENERATOR_JOYCAPTION_NF4
LLMSEARCH_SCORING_MODE_NUMERIC_V1 = "joycaption_numeric_v9"
LLMSEARCH_JOYCAPTION_CAPTION_TYPE = "descriptive"
LLMSEARCH_JOYCAPTION_CAPTION_LENGTH = "very short"
LLMSEARCH_JOYCAPTION_MAX_NEW_TOKENS = 128
LLMSEARCH_JOYCAPTION_TEMPERATURE = 0.6
LLMSEARCH_JOYCAPTION_TOP_P = 0.9
LLMSEARCH_JOYCAPTION_TOP_K = 0
LLMSEARCH_HUIHUI_GEMMA4_MAX_NEW_TOKENS = 24
LLMSEARCH_HUIHUI_GEMMA4_TEMPERATURE = 0.0
LLMSEARCH_HUIHUI_GEMMA4_TOP_P = 1.0
LLMSEARCH_HUIHUI_GEMMA4_TOP_K = 0
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
TAGMATCH_AUTOCOMPLETE_MAX_SUGGESTIONS = 12
TAGMATCH_DEFAULT_TAGS = (
    "bad_anatomy, bad_hands, bad_feet, bad_proportions, deformed, extra_arms, extra_faces, extra_mouth, missing_limb, multiple_legs, multiple_heads, oversized_limbs, wrong_foot, artistic_error, glitch, blob, disembodied_limb"
)
DINOV2_MODEL_ID = "facebook/dinov2-base"
DINOV2_INPUT_SIZE = 224
DINOV2_PATCH_DIM = 768
DINOV2_PATCHES_PER_IMAGE = 256
DINOV2_BASE_BATCH_SIZE = 24
DINOV2_MAX_BATCH_SIZE = 96
DINOV2_BATCH_AGGRESSION = 1.4
DINOV2_MIN_FREE_VRAM_GB = 3.0
OBJECTSEARCH_FAISS_TOP_K_PATCHES = 8

FACE_MODEL_PACK = "buffalo_l"
FACE_MODEL_LABEL = f"InsightFace {FACE_MODEL_PACK}"
FACE_DET_SIZE = (640, 640)
FACE_EMBEDDING_ABSOLUTE_MAX_WORKERS = 16
FACE_EMBEDDING_MIN_FREE_VRAM_GB = 2.5
FACE_EMBEDDING_PER_WORKER_VRAM_GB = 0.6
PROMPTMATCH_HOST_MAX_WORKERS = 16

VIEW_WITH_CONTROLS_OUTPUT_KEYS = (
    "left_head",
    "left_gallery",
    "right_head",
    "right_gallery",
    "status_md",
    "hist_plot",
    "sel_info",
    "mark_state",
    "external_query_image",
    "clear_external_query_btn",
    "left_export_cb",
    "left_export_name_tb",
    "export_acc",
    "move_controls_col",
    "gallery_header_spacer_col",
    "right_header_col",
    "right_gallery_col",
)
SCORE_FOLDER_OUTPUT_KEYS = (
    "left_head",
    "left_gallery",
    "right_head",
    "right_gallery",
    "status_md",
    "hist_plot",
    "sel_info",
    "mark_state",
    "external_query_image",
    "clear_external_query_btn",
    "main_slider",
    "aux_slider",
    "percentile_slider",
    "percentile_mid_btn",
    "model_status_state",
    "left_export_cb",
    "left_export_name_tb",
    "export_acc",
    "move_controls_col",
    "gallery_header_spacer_col",
    "right_header_col",
    "right_gallery_col",
)
PREVIEW_SEARCH_OUTPUT_KEYS = (
    "promptgen_status_md",
    *SCORE_FOLDER_OUTPUT_KEYS,
)

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
TAGMATCH_BATCH_AGGRESSION = 1.1
TAGMATCH_MIN_FREE_VRAM_GB = 3.0
PROMPTMATCH_PROXY_MAX_EDGE = 1024
PROMPTMATCH_PROXY_CACHE_ROOT = "HybridScorerPromptMatchProxyCache"
PROMPTMATCH_PROXY_PROGRESS_SHARE = 0.25
PROMPTMATCH_TORCH_THREAD_CAP = 8
EXPLICIT_PROMPT_WEIGHT_RE = re.compile(r"\(([^()]*?)\s*:\s*([0-9]*\.?[0-9]+)\)")

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


def default_cache_mode():
    return CACHE_MODE_PROJECT if os.name == "nt" else CACHE_MODE_SYSTEM


def _system_proxy_root():
    # Prefer /dev/shm (RAM-backed tmpfs) so proxy thumbnails never touch disk.
    shm = "/dev/shm"
    if os.path.isdir(shm) and os.access(shm, os.W_OK):
        return shm
    return tempfile.gettempdir()


@lru_cache(maxsize=1)
def get_cache_config():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
