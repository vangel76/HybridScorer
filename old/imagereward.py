import os
import shutil
import math
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import ImageReward as RM  # pip install image-reward

# --- Configuration ---
INPUT_FOLDER_NAME = "images"
PERCENT_WORST = 25
PERCENT_BEST = 25
FOLDER_NAMES = {"worst": "worst", "middle": "normal", "best": "best"}

# --- Control Flags ---
ENABLE_SCORE_PREFIX = False   # True = add "000_" to "100_" prefix
ENABLE_CATEGORY_SORTING = True  # True = sort into worst/normal/best

# --- Plotting / Other ---
SHOW_PLOT = True
SAVE_PLOT = True
PLOT_FILENAME = "imagereward_score_distribution.png"
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
BATCH_SIZE = 32  # ImageReward is VRAM-heavy → keep small

# --- ImageReward-specific ---
# Strong prompt optimized for femme fatale aesthetic scoring (2025–2026 style)
IR_PROMPT = (
    "masterpiece, best quality, ultra-detailed, cinematic, high fashion photography, "
    "extremely beautiful and seductive femme fatale woman, 25-35 years old, sharp elegant features, "
    "intense piercing gaze directly at viewer, flawless skin, dramatic lighting, rim light, chiaroscuro, "
    "moody atmosphere, volumetric god rays, professional studio portrait, vogue style, "
    "holding a gun, confident dangerous expression, "
    "highly detailed face and eyes, 8k, award-winning"
)

# --- Validation ---
if not (0 <= PERCENT_WORST <= 100 and 0 <= PERCENT_BEST <= 100):
    sys.exit("Error: Percentages must be 0-100.")
if PERCENT_WORST + PERCENT_BEST > 100 and ENABLE_CATEGORY_SORTING:
    sys.exit("Error: PERCENT_WORST + PERCENT_BEST > 100 when sorting enabled.")

# --- Helper Functions ---
def create_folders(base_path, folder_map):
    paths = {}
    print("\nCreating destination folders...")
    for key, name in folder_map.items():
        folder_path = os.path.join(base_path, name)
        paths[key] = folder_path
        if key == 'middle' or (ENABLE_CATEGORY_SORTING and key in ['worst', 'best']):
            os.makedirs(folder_path, exist_ok=True)
            print(f"Folder ready: {folder_path}")
    return paths

def calculate_imagereward_scores(image_paths, model, device):
    """
    Batched scoring using inference_rank - much faster on RTX 5090
    """
    scores = {}
    model = model.to(device).eval()

    # Process in chunks to avoid OOM (adjust chunk_size based on your VRAM)
    # 5090 has huge VRAM → you can try 32–64 or even more
    chunk_size = 32   # Start here, increase to 64/96 if stable

    for i in tqdm(range(0, len(image_paths), chunk_size), desc="Batched scoring with ImageReward"):
        batch_paths = image_paths[i:i + chunk_size]
        batch_filenames = [os.path.basename(p) for p in batch_paths]

        try:
            with torch.no_grad():
                # inference_rank returns (ranking_indices, reward_scores)
                _, batch_rewards = model.inference_rank(IR_PROMPT, batch_paths)

                # batch_rewards is list of floats (higher = better preference)
                for filename, reward in zip(batch_filenames, batch_rewards):
                    scores[filename] = float(reward)

                # Optional: print progress sample
                print(f"  Batch done - sample: {batch_filenames[0]} → {batch_rewards[0]:.4f}")

        except Exception as e:
            print(f"Batch error on chunk starting at {i}: {e}", file=sys.stderr)
            # Fallback: mark whole batch as failed
            for filename in batch_filenames:
                scores[filename] = -float('inf')

        if device == 'cuda':
            torch.cuda.empty_cache()

    return scores

def plot_score_distribution(scores_list, best_thresh, worst_thresh, filename, base_dir):
    if not plt:
        print("Matplotlib not available.")
        return
    plt.figure(figsize=(10, 6))
    plt.hist(scores_list, bins=30, color='teal', edgecolor='black', alpha=0.7)
    plt.title('ImageReward Score Distribution (Higher = Better)')
    plt.xlabel('ImageReward Score')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', alpha=0.5)

    if ENABLE_CATEGORY_SORTING:
        if best_thresh is not None:
            plt.axvline(best_thresh, color='green', linestyle='--', label=f'Best ≥ {best_thresh:.3f}')
        if worst_thresh is not None:
            plt.axvline(worst_thresh, color='red', linestyle='--', label=f'Worst < {worst_thresh:.3f}')
        plt.legend()

    plt.tight_layout()
    save_path = os.path.join(base_dir, filename)
    if SAVE_PLOT:
        plt.savefig(save_path)
        print(f"Plot saved: {save_path}")
    if SHOW_PLOT:
        plt.show()
    plt.close()

# --- Main ---
if __name__ == "__main__":
    print("--- AI Image Sorting Tool (Using ImageReward v1.5) ---")
    print("Better aesthetic & human-preference scoring than CLIP!")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, INPUT_FOLDER_NAME)
    dest_base_dir = script_dir

    if not os.path.isdir(source_dir):
        sys.exit(f"Input folder not found: {source_dir}")

    dest_folders = create_folders(dest_base_dir, FOLDER_NAMES)

    # Find images
    image_paths = [
        os.path.join(source_dir, f) for f in os.listdir(source_dir)
        if f.lower().endswith(ALLOWED_EXTENSIONS) and os.path.isfile(os.path.join(source_dir, f))
    ]
    if not image_paths:
        sys.exit("No images found.")
    print(f"Found {len(image_paths)} images.")

    # Load ImageReward
    print("\nLoading ImageReward-v1.5...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    if device == "cpu":
        print("Warning: CPU mode will be slow.")

    try:
        model = RM.load("ImageReward-v1.0")  # or "ImageReward-v1.0" if you prefer
    except Exception as e:
        sys.exit(f"Failed to load ImageReward: {e}\nTry: pip install image-reward")

    # Score
    print("Scoring images...")
    image_scores = calculate_imagereward_scores(image_paths, model, device)

    valid_scores = {k: v for k, v in image_scores.items() if v != -float('inf')}
    failed_count = len(image_scores) - len(valid_scores)
    if failed_count > 0:
        print(f"Warning: {failed_count} images failed scoring (will go to 'worst' or 'normal').")

    # Sort (descending: higher score = better)
    sorted_images = sorted(valid_scores.items(), key=lambda x: (-x[1], x[0]))
    n_scored = len(sorted_images)

    # Calculate splits
    best_threshold_score = worst_threshold_score = None
    n_best = n_worst = n_middle = 0

    if ENABLE_CATEGORY_SORTING:
        n_best = math.floor(n_scored * (PERCENT_BEST / 100.0))
        n_worst = math.ceil(n_scored * (PERCENT_WORST / 100.0))
        n_middle = n_scored - n_best - n_worst

        best_threshold_score = None
        worst_threshold_score = None

        if n_scored > 0:
            if n_best > 0:
                best_threshold_score = sorted_images[n_best - 1][1]
            if n_worst > 0:
                worst_threshold_score = sorted_images[n_scored - n_worst][1]

        # Safe printing
        print(f"Best ({PERCENT_BEST}%): ~{n_best} images", end="")
        if best_threshold_score is not None:
            print(f" (≥ {best_threshold_score:.3f})")
        else:
            print(" (no valid scores)")

        print(f"Normal ({100 - PERCENT_WORST - PERCENT_BEST}%): ~{n_middle} images")

        print(f"Worst ({PERCENT_WORST}%): ~{n_worst} images", end="")
        if worst_threshold_score is not None:
            print(f" (< {worst_threshold_score:.3f})")
        else:
            print(" (no valid scores)")
    else:
        n_middle = n_scored
        print("All scored images → 'normal' folder")

    # Plot
    if SHOW_PLOT or SAVE_PLOT:
        scores_list = [score for _, score in sorted_images]
        plot_score_distribution(scores_list, best_threshold_score, worst_threshold_score, PLOT_FILENAME, script_dir)

    # Copy files
    print("\nCopying files...")
    copied = failed_copied = skipped = 0

    for i, (filename, score) in enumerate(tqdm(sorted_images, desc="Copying scored")):
        src = os.path.join(source_dir, filename)
        if not os.path.exists(src):
            skipped += 1
            continue

        if ENABLE_SCORE_PREFIX:
            rank_pct = 100.0 if n_scored == 1 else ((n_scored - 1 - i) / (n_scored - 1)) * 100
            prefix = f"{int(round(rank_pct)):03d}_"
            new_name = prefix + filename
        else:
            new_name = filename

        if ENABLE_CATEGORY_SORTING:
            if i < n_best:
                dest_key = "best"
            elif i >= n_scored - n_worst:
                dest_key = "worst"
            else:
                dest_key = "middle"
        else:
            dest_key = "middle"

        dest = os.path.join(dest_folders[dest_key], new_name)
        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception as e:
            print(f"Copy error {filename}: {e}", file=sys.stderr)
            skipped += 1

    # Handle failed images
    failed_dest_key = "worst" if ENABLE_CATEGORY_SORTING else "middle"
    failed_dest = dest_folders[failed_dest_key]
    for filename, score in image_scores.items():
        if score == -float('inf'):
            src = os.path.join(source_dir, filename)
            if os.path.exists(src):
                new_name = f"000_{filename}" if ENABLE_SCORE_PREFIX else filename
                dest = os.path.join(failed_dest, new_name)
                try:
                    shutil.copy2(src, dest)
                    failed_copied += 1
                except:
                    skipped += 1

    # Summary
    print("\n--- Done ---")
    print(f"Copied {copied} scored images")
    if failed_copied > 0:
        print(f"Copied {failed_copied} failed images to '{FOLDER_NAMES[failed_dest_key]}'")
    if skipped > 0:
        print(f"Skipped {skipped} files (errors)")
    print(f"Originals remain in: {source_dir}")
    print(f"Sorted copies in: {dest_base_dir}")
    if ENABLE_SCORE_PREFIX:
        print("Filenames have score prefix (000–100)")
