import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from .config import get_cache_config
from .utils import (
    normalize_prompt_text,
    render_promptmatch_segments,
    parse_promptmatch_weighted_prompt,
    promptmatch_host_worker_count,
    promptmatch_timing_ms,
)


class ModelBackend:
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

    def _extract_feature_tensor(self, output, kind="feature"):
        if torch.is_tensor(output):
            return output

        _keys = (f"{kind}_embeds", "text_embeds", "image_embeds", "pooler_output", "last_hidden_state")

        getter = output.get if isinstance(output, dict) else lambda k: getattr(output, k, None)
        for key in _keys:
            value = getter(key)
            if torch.is_tensor(value):
                if key == "last_hidden_state" and value.ndim >= 3:
                    return value[:, 0, :]
                return value

        if isinstance(output, (list, tuple)):
            for value in output:
                if torch.is_tensor(value):
                    return value

        raise TypeError(f"Unsupported {kind} output type from backend {self.backend}: {type(output).__name__}")

    def _encode_text_plain(self, prompt):
        prompt = normalize_prompt_text(prompt)
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
                feat = self._extract_feature_tensor(feat, kind="text")
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
        model_dtype = next(self._model.parameters()).dtype
        with torch.no_grad():
            preprocess_started = time.perf_counter()
            preprocess_workers = promptmatch_host_worker_count(len(pil_images))
            if self.backend in ("openai", "openclip"):
                _fn = self._preprocess
            else:
                def _fn(img):
                    return self._processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            if preprocess_workers <= 1:
                processed = [_fn(img) for img in pil_images]
            else:
                with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                    processed = list(executor.map(_fn, pil_images))
            timings["preprocess"] = promptmatch_timing_ms(preprocess_started)

            transfer_started = time.perf_counter()
            tensors = torch.stack(processed).to(self.device).to(model_dtype)
            timings["host_to_device"] = promptmatch_timing_ms(transfer_started)

            gpu_started = time.perf_counter()
            if self.backend in ("openai", "openclip"):
                feat = self._model.encode_image(tensors)
            else:
                feat = self._model.get_image_features(pixel_values=tensors)
                feat = self._extract_feature_tensor(feat, kind="image")
            if self.device == "cuda":
                torch.cuda.synchronize()
            timings["gpu_encode"] = promptmatch_timing_ms(gpu_started)

            normalize_started = time.perf_counter()
            normalized = F.normalize(feat.float(), dim=-1)
            timings["normalize"] = promptmatch_timing_ms(normalize_started)
            if return_timings:
                return normalized, timings
            return normalized
