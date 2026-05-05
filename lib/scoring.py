import os
import sys
import time
import torch
from concurrent.futures import ThreadPoolExecutor

from .config import DEFAULT_BATCH_SIZE
from .utils import (
    get_auto_batch_size,
    prepare_promptmatch_loaded_batch,
    promptmatch_timing_ms,
    current_free_vram_gb,
    promptmatch_log_batch_timing,
    prepare_imagereward_loaded_batch,
    imagereward_log_batch_timing,
    is_cuda_oom_error,
)


def _make_result_entry(original_path, pos_score, neg_score, failed=False):
    return {
        "pos": float(pos_score) if not failed else 0.0,
        "neg": (float(neg_score) if neg_score is not None else None) if not failed else None,
        "path": original_path,
        "failed": failed,
    }


def _submit_promptmatch_prefetch(executor, image_paths, start_index, batch_size, total, proxy_resolver):
    if start_index >= total:
        return None, None, None
    size = min(batch_size, total - start_index)
    batch = image_paths[start_index:start_index + size]
    future = executor.submit(prepare_promptmatch_loaded_batch, batch, proxy_resolver)
    return future, start_index, size


def _annotate_prefetch_timing(prefetched, prefetch_result_started):
    loaded_prefetch, pil_prefetch, failed_prefetch, timing_prefetch = prefetched
    timing_prefetch = dict(timing_prefetch or {})
    timing_prefetch["prefetch_ready_wait"] = promptmatch_timing_ms(prefetch_result_started)
    return (loaded_prefetch, pil_prefetch, failed_prefetch, timing_prefetch)


def _reset_oom_prefetch(prefetch_executor, prefetched_future):
    if prefetched_future is not None:
        prefetched_future.cancel()
    prefetch_executor.shutdown(wait=False, cancel_futures=True)
    torch.cuda.empty_cache()
    return ThreadPoolExecutor(max_workers=1), None, None, None


def _run_promptmatch_batches(image_paths, backend, batch_size, proxy_resolver, progress_cb, on_batch, mark_failed_cb):
    """
    Core prefetch/OOM/retry loop for PromptMatch encoding.

    Calls on_batch(loaded, feat, encode_timings, prefetch_wait_ms, batch_start, batch_end, total)
    for each successfully encoded batch.  Calls mark_failed_cb(path) for load failures and
    unrecoverable per-image errors.  Returns the (possibly halved) batch_size so callers can
    see the final value, though it is not currently used by either wrapper.
    """
    total = len(image_paths)
    done = 0
    batches_since_cache_clear = 0

    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetched = None
    prefetch_valid = False          # True when prefetched covers exactly [done, done+current_size)
    prefetched_future = None
    # Track what the prefetch covers so we can validate it after an OOM resize.
    _prefetched_start = None
    _prefetched_size = None

    try:
        while done < total:
            current_size = min(batch_size, total - done)
            batch = image_paths[done:done + current_size]
            batch_start = done + 1
            batch_end = done + len(batch)
            print(f"[PromptMatch] Batch {batch_start}-{batch_end}/{total} ({len(batch)} images)")

            prefetch_wait_ms = 0.0
            if prefetch_valid and prefetched is not None and _prefetched_start == done and _prefetched_size == current_size:
                loaded, pil_imgs, batch_failed, load_timings = prefetched
                prefetched = None
                prefetch_valid = False
            else:
                load_started = time.perf_counter()
                loaded, pil_imgs, batch_failed, load_timings = prepare_promptmatch_loaded_batch(batch, proxy_resolver)
                prefetch_wait_ms = promptmatch_timing_ms(load_started)

            for original_path in batch_failed:
                mark_failed_cb(original_path)

            next_start = done + len(batch)
            prefetched_future, _prefetched_start, _prefetched_size = _submit_promptmatch_prefetch(
                prefetch_executor, image_paths, next_start, batch_size, total, proxy_resolver
            )

            if pil_imgs:
                try:
                    encode_started = time.perf_counter()
                    feat, encode_timings = backend.encode_images_batch(pil_imgs, return_timings=True)
                    encode_total_ms = promptmatch_timing_ms(encode_started)
                    on_batch(loaded, feat, encode_timings, encode_total_ms, prefetch_wait_ms, batch_start, batch_end, total, load_timings)
                    batches_since_cache_clear += 1
                    if backend.device == "cuda" and batches_since_cache_clear >= 4:
                        torch.cuda.empty_cache()
                        batches_since_cache_clear = 0
                except Exception as exc:
                    if backend.device == "cuda" and is_cuda_oom_error(exc) and current_size > 1:
                        batch_size = max(1, current_size // 2)
                        print(f"[PromptMatch] CUDA OOM, retrying with batch size {batch_size}")
                        prefetch_executor, prefetched_future, prefetched, _prefetched_start = _reset_oom_prefetch(prefetch_executor, prefetched_future)
                        _prefetched_size = None
                        prefetch_valid = False
                        batches_since_cache_clear = 0
                        torch.cuda.empty_cache()
                        if progress_cb:
                            progress_cb(done, total, batch_size, True)
                        continue
                    print(f"  [WARN] batch error, retrying individually: {exc}")
                    recovered = 0
                    failed_count = 0
                    for original_path, _ in loaded:
                        try:
                            single_loaded, single_imgs, single_failed, _ = prepare_promptmatch_loaded_batch([original_path], proxy_resolver)
                            for failed_path in single_failed:
                                mark_failed_cb(failed_path)
                            if not single_imgs:
                                failed_count += 1
                                continue
                            single_feat = backend.encode_images_batch(single_imgs)
                            on_batch(single_loaded, single_feat, {}, 0.0, 0.0, batch_start, batch_end, total, {})
                            recovered += 1
                        except Exception as single_exc:
                            print(f"  [WARN] single-image error for {original_path}: {single_exc}")
                            mark_failed_cb(original_path)
                            failed_count += 1
                    print(f"[PromptMatch] Individual retry result: {recovered} recovered, {failed_count} failed")

            done += len(batch)
            if progress_cb:
                progress_cb(done, total, batch_size, False)

            if prefetched_future is not None:
                prefetch_result_started = time.perf_counter()
                prefetched = _annotate_prefetch_timing(prefetched_future.result(), prefetch_result_started)
                prefetch_valid = True
                prefetched_future = None
            else:
                prefetched = None
                prefetch_valid = False
                _prefetched_start = None
                _prefetched_size = None
    finally:
        if prefetched_future is not None:
            prefetched_future.cancel()
        prefetch_executor.shutdown(wait=False, cancel_futures=True)

    return batch_size


def score_all(image_paths, backend, pos_emb, neg_emb, progress_cb=None, proxy_resolver=None):
    results = {}
    batch_size = get_auto_batch_size(backend.device, backend)
    print(f"[PromptMatch] Using batch size {batch_size}")

    def _mark_failed(original_path):
        results[os.path.basename(original_path)] = _make_result_entry(original_path, 0.0, None, failed=True)

    def _on_batch(loaded, feat, encode_timings, encode_total_ms, prefetch_wait_ms, batch_start, batch_end, total, load_timings):
        score_started = time.perf_counter()
        pos_sims = (feat @ pos_emb.T).squeeze(1).tolist()
        neg_sims = (feat @ neg_emb.T).squeeze(1).tolist() if neg_emb is not None else [None] * len(loaded)
        for (original_path, _), pos_score, neg_score in zip(loaded, pos_sims, neg_sims):
            results[os.path.basename(original_path)] = _make_result_entry(original_path, pos_score, neg_score)
        score_ms = promptmatch_timing_ms(score_started)
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
                "encode_total": encode_total_ms or None,
                "free_vram_gb": vram_info[0] if vram_info is not None else None,
            },
        )

    _run_promptmatch_batches(image_paths, backend, batch_size, proxy_resolver, progress_cb, _on_batch, _mark_failed)
    return results


def encode_all_promptmatch_images(image_paths, backend, progress_cb=None, proxy_resolver=None):
    feature_paths = []
    feature_rows = []
    _failed_paths = []
    failed_seen = set()
    batch_size = get_auto_batch_size(backend.device, backend)
    print(f"[PromptMatch] Using batch size {batch_size}")

    def _mark_failed(original_path):
        if original_path not in failed_seen:
            failed_seen.add(original_path)
            _failed_paths.append(original_path)

    def _on_batch(loaded, feat, encode_timings, encode_total_ms, prefetch_wait_ms, batch_start, batch_end, total, load_timings):
        copy_started = time.perf_counter()
        feat_cpu = feat.detach().cpu()
        for (original_path, _), row in zip(loaded, feat_cpu):
            feature_paths.append(original_path)
            feature_rows.append(row)
        copy_ms = promptmatch_timing_ms(copy_started)
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
                "encode_total": encode_total_ms or None,
                "free_vram_gb": vram_info[0] if vram_info is not None else None,
            },
        )

    _run_promptmatch_batches(image_paths, backend, batch_size, proxy_resolver, progress_cb, _on_batch, _mark_failed)

    feature_tensor = torch.stack(feature_rows) if feature_rows else torch.empty((0, 0), dtype=torch.float32)
    return feature_paths, feature_tensor, _failed_paths


def score_promptmatch_cached_features(feature_paths, image_features, failed_paths, pos_emb, neg_emb):
    results = {}
    pos_emb_cpu = pos_emb.detach().float().cpu()
    neg_emb_cpu = neg_emb.detach().float().cpu() if neg_emb is not None else None

    if feature_paths and image_features.numel():
        pos_sims = (image_features @ pos_emb_cpu.T).squeeze(1).tolist()
        neg_sims = (image_features @ neg_emb_cpu.T).squeeze(1).tolist() if neg_emb_cpu is not None else [None] * len(feature_paths)
        for original_path, pos_score, neg_score in zip(feature_paths, pos_sims, neg_sims):
            results[os.path.basename(original_path)] = _make_result_entry(original_path, pos_score, neg_score)

    for original_path in failed_paths:
        results[os.path.basename(original_path)] = _make_result_entry(original_path, 0.0, None, failed=True)

    return results
