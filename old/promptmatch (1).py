import os
import shutil
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration  — only these two lines need changing between runs
# ---------------------------------------------------------------------------
SEARCH_PROMPT = "short pants"          # what you are looking for
CONFIDENCE_THRESHOLD = 0.13      # lower = more permissive  |  start here, tune if needed

# ---------------------------------------------------------------------------
# Advanced settings  (usually leave as-is)
# ---------------------------------------------------------------------------
INPUT_FOLDER_NAME  = "images"
FOLDER_NAMES       = {"found": "found", "notfound": "notfound"}
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
BATCH_SIZE         = 64          # RTX 5090 can handle large batches easily
CLIP_MODEL         = "ViT-L/14"  # best accuracy; use "ViT-B/32" if VRAM is tight

SHOW_PLOT     = True
SAVE_PLOT     = True
PLOT_FILENAME = "promptmatch_confidence_distribution.png"

# ---------------------------------------------------------------------------
# CLIP scoring
# ---------------------------------------------------------------------------
def load_clip(device):
    try:
        import clip
    except ImportError:
        sys.exit("CLIP not installed. Run:  pip install git+https://github.com/openai/CLIP.git")
    print(f"Loading CLIP model: {CLIP_MODEL}")
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()
    print("CLIP loaded.")
    return model, preprocess, clip


def score_images(image_paths, model, preprocess, clip, device, prompt):
    """
    Returns dict: filename → similarity score (float, roughly 0.1–0.35 typical range).
    Higher = more similar to the prompt.
    """
    # Encode several phrasings of the prompt for robustness
    text_tokens = clip.tokenize([
        f"a photo of a {prompt}",
        f"a photo of {prompt}",
        prompt,
    ]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)          # (3, D)
        text_features = F.normalize(text_features, dim=-1)
        # Average the phrasings into one embedding
        text_mean = text_features.mean(dim=0, keepdim=True)     # (1, D)
        text_mean = F.normalize(text_mean, dim=-1)

    scores = {}

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="CLIP scoring"):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        images = []
        valid  = []

        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                images.append(img)
                valid.append(os.path.basename(p))
            except Exception as e:
                print(f"  [WARN] Cannot open {p}: {e}", file=sys.stderr)
                scores[os.path.basename(p)] = -1.0

        if not images:
            continue

        image_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)

        # Cosine similarity vs averaged text embedding
        sims = (image_features @ text_mean.T).squeeze(1)        # (B,)

        for fname, sim in zip(valid, sims.tolist()):
            scores[fname] = float(sim)

        if device == "cuda":
            torch.cuda.empty_cache()

    return scores


# ---------------------------------------------------------------------------
# Auto-threshold helper: picks the natural valley in the distribution
# Call this on a small test run to find a good CONFIDENCE_THRESHOLD value
# ---------------------------------------------------------------------------
def suggest_threshold(scores_list):
    import statistics
    mean = statistics.mean(scores_list)
    stdev = statistics.stdev(scores_list) if len(scores_list) > 1 else 0.0
    suggestion = mean + 0.25 * stdev
    print(f"\n  Score stats  →  mean={mean:.4f}  stdev={stdev:.4f}")
    print(f"  Auto-suggested threshold: {suggestion:.4f}")
    print(f"  (Set CONFIDENCE_THRESHOLD = {suggestion:.3f} in the config above)")
    return suggestion


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_distribution(scores_list, threshold, filename, base_dir):
    plt.figure(figsize=(10, 5))
    plt.hist(scores_list, bins=40, color='steelblue', edgecolor='black', alpha=0.75)
    plt.axvline(threshold, color='orange', linestyle='--',
                label=f'Threshold = {threshold:.3f}')
    plt.title(f'CLIP Similarity Distribution\nPrompt: "{SEARCH_PROMPT}"')
    plt.xlabel('Cosine similarity to prompt  (higher = better match)')
    plt.ylabel('Number of images')
    plt.legend()
    plt.grid(axis='y', alpha=0.4)
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
    print("=== Prompt-Match Image Sorter (CLIP) ===")
    print(f"  Prompt    : {SEARCH_PROMPT!r}")
    print(f"  Threshold : similarity >= {CONFIDENCE_THRESHOLD}")
    print(f"  Model     : {CLIP_MODEL}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, INPUT_FOLDER_NAME)

    if not os.path.isdir(source_dir):
        sys.exit(f"Input folder not found: {source_dir}")

    # Create output folders
    dest_folders = {}
    for key, name in FOLDER_NAMES.items():
        p = os.path.join(script_dir, name)
        os.makedirs(p, exist_ok=True)
        dest_folders[key] = p
        print(f"  Folder ready: {p}")

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

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    # Load CLIP
    model, preprocess, clip = load_clip(device)

    # Score all images
    scores = score_images(image_paths, model, preprocess, clip, device, SEARCH_PROMPT)

    # Print score range so you can calibrate the threshold
    valid_scores = [s for s in scores.values() if s >= 0]
    if valid_scores:
        suggest_threshold(valid_scores)

    # Split into buckets
    found_list    = [(f, s) for f, s in scores.items() if s >= CONFIDENCE_THRESHOLD]
    notfound_list = [(f, s) for f, s in scores.items() if 0 <= s < CONFIDENCE_THRESHOLD]
    failed_list   = [f      for f, s in scores.items() if s < 0]

    # Sort found by score descending
    found_list.sort(key=lambda x: -x[1])

    print(f"\n--- Results ---")
    print(f"  FOUND     : {len(found_list)} images")
    print(f"  NOT FOUND : {len(notfound_list)} images")
    if failed_list:
        print(f"  FAILED    : {len(failed_list)} images (load errors → notfound)")

    # Sanity check: show top 10 matches with their scores
    print("\n  Top 10 matches (highest similarity):")
    all_valid = sorted([(f, s) for f, s in scores.items() if s >= 0], key=lambda x: -x[1])
    for fname, sim in all_valid[:10]:
        tag = "✓ FOUND" if sim >= CONFIDENCE_THRESHOLD else "  below threshold"
        print(f"    {sim:.4f}  {tag}  {fname}")

    # Plot
    if valid_scores and (SHOW_PLOT or SAVE_PLOT):
        plot_distribution(valid_scores, CONFIDENCE_THRESHOLD, PLOT_FILENAME, script_dir)

    # Copy files
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
    print(f"  Copied  : {copied}")
    print(f"  Skipped : {skipped}")
    print(f"  Found   → {dest_folders['found']}")
    print(f"  Notfound→ {dest_folders['notfound']}")
