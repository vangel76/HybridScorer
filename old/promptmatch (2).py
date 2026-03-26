import os
import shutil
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration  — change these between runs
# ---------------------------------------------------------------------------
SEARCH_PROMPT   = "pants"        # what you ARE looking for
NEGATIVE_PROMPT = ""          # what you are NOT looking for  (set to "" to disable)

# Thresholds
POSITIVE_THRESHOLD = 0.15        # image must score >= this for the positive prompt
NEGATIVE_THRESHOLD = 0.15        # image must score <  this for the negative prompt
                                 # (if negative prompt is empty, this is ignored)

# ---------------------------------------------------------------------------
# Advanced settings  (usually leave as-is)
# ---------------------------------------------------------------------------
INPUT_FOLDER_NAME  = "images"
FOLDER_NAMES       = {"found": "found", "notfound": "notfound"}
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
BATCH_SIZE         = 64
CLIP_MODEL         = "ViT-L/14"  # use "ViT-B/32" if VRAM is tight

SHOW_PLOT     = True
SAVE_PLOT     = True
PLOT_FILENAME = "promptmatch_confidence_distribution.png"

# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------
def load_clip(device):
    try:
        import clip
    except ImportError:
        sys.exit("CLIP not installed.\nRun: pip install git+https://github.com/openai/CLIP.git")
    print(f"Loading CLIP model: {CLIP_MODEL}")
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()
    print("CLIP loaded.")
    return model, preprocess, clip


def encode_prompt(clip, model, device, prompt):
    """Encode a text prompt into a normalised embedding (average of 3 phrasings)."""
    tokens = clip.tokenize([
        f"a photo of a {prompt}",
        f"a photo of {prompt}",
        prompt,
    ]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = F.normalize(features, dim=-1)
        mean = features.mean(dim=0, keepdim=True)
        mean = F.normalize(mean, dim=-1)
    return mean                                          # shape: (1, D)


def score_images(image_paths, model, preprocess, clip, device,
                 pos_embedding, neg_embedding):
    """
    Returns dict: filename → {"pos": float, "neg": float or None}
    """
    scores = {}

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="CLIP scoring"):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        images, valid = [], []

        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                images.append(img)
                valid.append(os.path.basename(p))
            except Exception as e:
                print(f"  [WARN] Cannot open {p}: {e}", file=sys.stderr)
                scores[os.path.basename(p)] = {"pos": -1.0, "neg": -1.0}

        if not images:
            continue

        image_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            img_features = model.encode_image(image_tensor)
            img_features = F.normalize(img_features, dim=-1)   # (B, D)

        pos_sims = (img_features @ pos_embedding.T).squeeze(1).tolist()

        if neg_embedding is not None:
            neg_sims = (img_features @ neg_embedding.T).squeeze(1).tolist()
        else:
            neg_sims = [None] * len(valid)

        for fname, ps, ns in zip(valid, pos_sims, neg_sims):
            scores[fname] = {"pos": float(ps), "neg": float(ns) if ns is not None else None}

        if device == "cuda":
            torch.cuda.empty_cache()

    return scores


def is_found(entry):
    """Apply positive + optional negative threshold logic."""
    pos_ok = entry["pos"] >= POSITIVE_THRESHOLD
    if entry["neg"] is None:
        return pos_ok
    neg_ok = entry["neg"] < NEGATIVE_THRESHOLD
    return pos_ok and neg_ok


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------
def suggest_threshold(label, scores_list):
    import statistics
    mean  = statistics.mean(scores_list)
    stdev = statistics.stdev(scores_list) if len(scores_list) > 1 else 0.0
    suggestion = mean + 0.25 * stdev
    print(f"\n  [{label}] mean={mean:.4f}  stdev={stdev:.4f}  → suggested threshold: {suggestion:.4f}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_distribution(scores, pos_thresh, neg_thresh, filename, base_dir):
    pos_vals = [v["pos"] for v in scores.values() if v["pos"] >= 0]
    neg_vals = [v["neg"] for v in scores.values()
                if v["neg"] is not None and v["neg"] >= 0]

    n_plots = 2 if neg_vals else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    axes[0].hist(pos_vals, bins=40, color='steelblue', edgecolor='black', alpha=0.75)
    axes[0].axvline(pos_thresh, color='green', linestyle='--',
                    label=f'Threshold = {pos_thresh:.3f}')
    axes[0].set_title(f'Positive prompt: "{SEARCH_PROMPT}"')
    axes[0].set_xlabel('Cosine similarity')
    axes[0].set_ylabel('Images')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.4)

    if neg_vals:
        axes[1].hist(neg_vals, bins=40, color='salmon', edgecolor='black', alpha=0.75)
        axes[1].axvline(neg_thresh, color='red', linestyle='--',
                        label=f'Threshold = {neg_thresh:.3f}')
        axes[1].set_title(f'Negative prompt: "{NEGATIVE_PROMPT}"')
        axes[1].set_xlabel('Cosine similarity  (must be BELOW threshold to pass)')
        axes[1].set_ylabel('Images')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.4)

    plt.tight_layout()
    if SAVE_PLOT:
        save_path = os.path.join(base_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved: {save_path}")
    if SHOW_PLOT:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Prompt-Match Image Sorter (CLIP + Negative Prompt) ===")
    print(f"  Positive  : {SEARCH_PROMPT!r}  (threshold >= {POSITIVE_THRESHOLD})")
    if NEGATIVE_PROMPT:
        print(f"  Negative  : {NEGATIVE_PROMPT!r}  (must score < {NEGATIVE_THRESHOLD} to pass)")
    else:
        print("  Negative  : (disabled)")
    print(f"  Model     : {CLIP_MODEL}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, INPUT_FOLDER_NAME)

    if not os.path.isdir(source_dir):
        sys.exit(f"Input folder not found: {source_dir}")

    # Output folders
    dest_folders = {}
    for key, name in FOLDER_NAMES.items():
        p = os.path.join(script_dir, name)
        os.makedirs(p, exist_ok=True)
        dest_folders[key] = p

    # Collect images
    image_paths = [
        os.path.join(source_dir, f)
        for f in sorted(os.listdir(source_dir))
        if f.lower().endswith(ALLOWED_EXTENSIONS)
        and os.path.isfile(os.path.join(source_dir, f))
    ]
    if not image_paths:
        sys.exit("No images found in input folder.")
    print(f"\nFound {len(image_paths)} images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    model, preprocess, clip = load_clip(device)

    # Encode prompts
    pos_embedding = encode_prompt(clip, model, device, SEARCH_PROMPT)
    neg_embedding = encode_prompt(clip, model, device, NEGATIVE_PROMPT) if NEGATIVE_PROMPT else None

    # Score
    scores = score_images(image_paths, model, preprocess, clip, device,
                          pos_embedding, neg_embedding)

    # Calibration hints
    pos_vals = [v["pos"] for v in scores.values() if v["pos"] >= 0]
    neg_vals = [v["neg"] for v in scores.values() if v["neg"] is not None and v["neg"] >= 0]
    if pos_vals:
        suggest_threshold("POSITIVE", pos_vals)
    if neg_vals:
        suggest_threshold("NEGATIVE", neg_vals)

    # Split
    found_list    = [(f, v) for f, v in scores.items() if v["pos"] >= 0 and is_found(v)]
    notfound_list = [(f, v) for f, v in scores.items() if v["pos"] >= 0 and not is_found(v)]
    failed_list   = [f      for f, v in scores.items() if v["pos"] < 0]

    found_list.sort(key=lambda x: -x[1]["pos"])

    print(f"\n--- Results ---")
    print(f"  FOUND     : {len(found_list)}")
    print(f"  NOT FOUND : {len(notfound_list)}")
    if failed_list:
        print(f"  FAILED    : {len(failed_list)}")

    # Top 10 with both scores shown
    print("\n  Top 10 matches:")
    all_valid = sorted([(f, v) for f, v in scores.items() if v["pos"] >= 0],
                       key=lambda x: -x[1]["pos"])
    for fname, v in all_valid[:10]:
        neg_str = f"  neg={v['neg']:.4f}" if v["neg"] is not None else ""
        tag = "✓ FOUND" if is_found(v) else "  blocked"
        print(f"    pos={v['pos']:.4f}{neg_str}  {tag}  {fname}")

    # Plot
    if (SHOW_PLOT or SAVE_PLOT):
        plot_distribution(scores, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD,
                          PLOT_FILENAME, script_dir)

    # Copy
    print("\nCopying files...")
    copied = skipped = 0

    def copy_file(fname, dest_key):
        global copied, skipped
        src  = os.path.join(source_dir, fname)
        dest = os.path.join(dest_folders[dest_key], fname)
        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception as e:
            print(f"  [WARN] {fname}: {e}", file=sys.stderr)
            skipped += 1

    for fname, _ in tqdm(found_list,    desc="Copying FOUND"):
        copy_file(fname, "found")
    for fname, _ in tqdm(notfound_list, desc="Copying NOT FOUND"):
        copy_file(fname, "notfound")
    for fname    in tqdm(failed_list,   desc="Copying FAILED → notfound"):
        copy_file(fname, "notfound")

    print(f"\n=== Done ===")
    print(f"  Copied  : {copied}  |  Skipped: {skipped}")
    print(f"  Found   → {dest_folders['found']}")
    print(f"  Notfound→ {dest_folders['notfound']}")
