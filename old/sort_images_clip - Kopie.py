import os
import shutil
import math
import sys
import torch
from PIL import Image
# Check Pillow version for ANTIALIAS attribute
from PIL import __version__ as PILLOW_VERSION
_PILLOW_INTERPOLATION = {}
if hasattr(Image, 'Resampling'):
    _PILLOW_INTERPOLATION['bicubic'] = Image.Resampling.BICUBIC
    _PILLOW_INTERPOLATION['bilinear'] = Image.Resampling.BILINEAR
    _PILLOW_INTERPOLATION['nearest'] = Image.Resampling.NEAREST
    _PILLOW_INTERPOLATION['lanczos'] = Image.Resampling.LANCZOS
else:
    if hasattr(Image, 'LANCZOS'):
         _PILLOW_INTERPOLATION['antialias'] = Image.LANCZOS
    else:
        _PILLOW_INTERPOLATION['antialias'] = Image.BICUBIC
    _PILLOW_INTERPOLATION['bicubic'] = Image.BICUBIC
    _PILLOW_INTERPOLATION['bilinear'] = Image.BILINEAR
    _PILLOW_INTERPOLATION['nearest'] = Image.NEAREST

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: Matplotlib not found (pip install matplotlib). Plotting disabled.", file=sys.stderr)
    plt = None

from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import requests

# --- Configuration ---
INPUT_FOLDER_NAME = "images"
PERCENT_WORST = 25
PERCENT_BEST = 40
FOLDER_NAMES = { "worst": "worst", "middle": "normal", "best": "best" }

# --- !!! NEW CONTROL FLAGS !!! ---
ENABLE_SCORE_PREFIX = True   # True = add "000_" to "100_" prefix, False = use original filenames
ENABLE_CATEGORY_SORTING = False # True = sort into worst/normal/best, False = copy all scored images to 'normal' folder

# --- Plotting / Other Config ---
SHOW_PLOT = True
SAVE_PLOT = True
PLOT_FILENAME = "clip_score_distribution.png"
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
BATCH_SIZE = 16

# --- CLIP Prompts ---
POSITIVE_PROMPTS = [
    "sexy", "rape", "pain", "suffering", "raw sex", "bruised", "oral sex", "surprised", "shocked", "pain",
    "a high-resolution photograph", "realistic photo", "dynamic lighting", "detailed skin",
    "good quality image", "photorealistic", "clear image", "good composition",
]
NEGATIVE_PROMPTS = [
    "bad eyes", "low quality", "unrealistic", "blurry image", "cartoon", "drawing",
    "illustration", "deformed hand", "artifact", "extra fingers", "poorly drawn", "text",
    "letters", "flat lighting",
]

# --- Validation & Calculation ---
if not (0 <= PERCENT_WORST <= 100 and 0 <= PERCENT_BEST <= 100): sys.exit("Error: Percentages must be 0-100.")
if PERCENT_WORST + PERCENT_BEST > 100 and ENABLE_CATEGORY_SORTING:
    sys.exit("Error: PERCENT_WORST + PERCENT_BEST > 100 when ENABLE_CATEGORY_SORTING is True.")
PERCENT_MIDDLE = 100 - PERCENT_WORST - PERCENT_BEST

# --- Helper Functions ---
# Updated create_folders
def create_folders(base_path, folder_map):
    paths = {}
    print("\nCreating destination folders...")
    try:
        if not os.path.isdir(base_path): sys.exit(f"Error: Script's base path not found? {base_path}")

        for key, name in folder_map.items():
            folder_path = os.path.join(base_path, name)
            paths[key] = folder_path # Store the potential path

            should_create = False # Default to not creating
            if key == 'middle': # Always create the 'normal' ('middle') folder
                should_create = True
            elif ENABLE_CATEGORY_SORTING and (key == 'worst' or key == 'best'):
                # Only create 'worst' and 'best' if category sorting is enabled
                should_create = True

            if should_create:
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    print(f"Created folder: {folder_path}")
                else:
                    print(f"Folder already exists: {folder_path}")

        # Ensure essential keys exist for later logic, pointing to where folder *would* be
        for key in folder_map.keys():
             if key not in paths:
                  paths[key] = os.path.join(base_path, folder_map[key])

        return paths
    except OSError as e: sys.exit(f"Error creating directories in {base_path}: {e}")


def calculate_clip_scores(image_paths, model, processor, device):
    scores = {}
    model.eval()
    all_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        num_batches = math.ceil(len(image_paths) / BATCH_SIZE)
        for i in tqdm(range(num_batches), desc="Scoring Images (Batches)"):
            batch_paths = image_paths[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            batch_images, valid_paths_in_batch = [], []
            for img_path in batch_paths:
                filename = os.path.basename(img_path)
                try:
                    image = Image.open(img_path).convert("RGB")
                    batch_images.append(image)
                    valid_paths_in_batch.append(img_path)
                except Exception as e:
                    print(f"\nWarning: Could not load image '{filename}'. Assigning lowest score. Error: {e}", file=sys.stderr)
                    scores[filename] = -float('inf')
            if not batch_images: continue
            try:
                image_inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
                image_features = model.get_image_features(**image_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).cpu().numpy()
                for j, full_path in enumerate(valid_paths_in_batch):
                    filename = os.path.basename(full_path)
                    if filename not in scores or scores[filename] != -float('inf'):
                        pos_sim = similarity[j, :len(POSITIVE_PROMPTS)].mean()
                        neg_sim = similarity[j, len(POSITIVE_PROMPTS):].mean()
                        score = pos_sim - neg_sim
                        scores[filename] = score
            except Exception as e:
                 print(f"\nError processing CLIP batch: {e}", file=sys.stderr)
                 for full_path in valid_paths_in_batch:
                     filename = os.path.basename(full_path)
                     scores[filename] = -float('inf')
            finally:
                 if 'image_inputs' in locals(): del image_inputs
                 if 'image_features' in locals(): del image_features
                 if device == 'cuda': torch.cuda.empty_cache()
    return scores

def plot_score_distribution(scores_list, best_threshold_score, worst_threshold_score, filename, base_dir):
    if not plt: print("Plotting disabled: matplotlib not imported."); return
    if not scores_list: print("No valid scores to plot."); return
    print("\nGenerating score distribution plot...")
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(scores_list, bins='auto', color='skyblue', edgecolor='black')
        plt.title('Distribution of Image CLIP Scores')
        plt.xlabel('CLIP Score (Higher is Better)')
        plt.ylabel('Number of Images')
        plt.grid(axis='y', alpha=0.75)
        plot_legend = False
        if ENABLE_CATEGORY_SORTING: # Only show lines if sorting
            if best_threshold_score is not None:
                plt.axvline(best_threshold_score, color='green', linestyle='dashed', linewidth=1.5, label=f'Best Threshold ({best_threshold_score:.3f})')
                plot_legend = True
            if worst_threshold_score is not None:
                 label = f'Worst Threshold ({worst_threshold_score:.3f})'
                 if best_threshold_score is not None and abs(worst_threshold_score - best_threshold_score) < 1e-5 : label = f'Worst/Best Thresh ({worst_threshold_score:.3f})'
                 plt.axvline(worst_threshold_score, color='red', linestyle='dashed', linewidth=1.5, label=label)
                 plot_legend = True
        if plot_legend: plt.legend()
        plt.tight_layout()
        if SAVE_PLOT:
            save_path = os.path.join(base_dir, filename)
            plt.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        if SHOW_PLOT:
            print("Displaying plot window...")
            plt.show()
        plt.close()
    except Exception as e: print(f"Error generating plot: {e}", file=sys.stderr)

# --- Main Script ---
if __name__ == "__main__":
    print("--- AI Image Automatic Sorting Tool (Using CLIP) ---")
    copy_desc = "COPIES images"
    if ENABLE_SCORE_PREFIX: copy_desc += " with score prefix"
    if ENABLE_CATEGORY_SORTING: copy_desc += " into category folders."
    else: copy_desc += f" into '{FOLDER_NAMES['middle']}' folder."
    print(copy_desc)

    if ENABLE_CATEGORY_SORTING: print(f"Thresholds: Worst {PERCENT_WORST}%, Normal {PERCENT_MIDDLE}%, Best {PERCENT_BEST}%")
    if plt and (SHOW_PLOT or SAVE_PLOT): print(f"Plotting: Show={SHOW_PLOT}, Save={SAVE_PLOT} (as {PLOT_FILENAME})")

    # 1. Determine Paths
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd(); print(f"Warning: Using CWD: {script_dir}")
    source_dir = os.path.join(script_dir, INPUT_FOLDER_NAME)
    dest_base_dir = script_dir
    print(f"Source directory (expected): {source_dir}")
    print(f"Destination base directory: {dest_base_dir}")
    if not os.path.isdir(source_dir): sys.exit(f"\nError: Input folder '{INPUT_FOLDER_NAME}' not found at {source_dir}")

    # 2. Create Destination Folders
    dest_folders = create_folders(dest_base_dir, FOLDER_NAMES)

    # 3. Find Images
    print(f"\nScanning '{source_dir}' for images ({', '.join(ALLOWED_EXTENSIONS)})...")
    image_paths = []
    for fname in os.listdir(source_dir):
        if fname.lower().endswith(ALLOWED_EXTENSIONS):
            full_path = os.path.join(source_dir, fname)
            if os.path.isfile(full_path): image_paths.append(full_path)
    if not image_paths: sys.exit(f"No images found in '{source_dir}'. Exiting.")
    print(f"Found {len(image_paths)} images.")

    # 4. Load CLIP Model
    print(f"\nLoading CLIP model '{CLIP_MODEL_NAME}'...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device.upper()}")
        if device == "cpu": print("Warning: Processing on CPU will be very slow.")
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    except Exception as e: sys.exit(f"\nError loading CLIP model: {e}")

    # 5. Score Images
    print("Calculating CLIP scores...")
    image_scores = calculate_clip_scores(image_paths, model, processor, device)
    if not image_scores: sys.exit("No images could be scored. Exiting.")
    valid_scores = {k: v for k, v in image_scores.items() if v is not None and v != -float('inf')}
    failed_prefix = "000_" if ENABLE_SCORE_PREFIX else ""
    failed_dest_key_report = "worst" if ENABLE_CATEGORY_SORTING else "middle" # For reporting only
    if len(valid_scores) < len(image_scores):
         failed_count = len(image_scores) - len(valid_scores)
         print(f"Warning: {failed_count} images failed scoring/loading (will be prefixed {failed_prefix} and copied to '{FOLDER_NAMES[failed_dest_key_report]}').")

    # 6. Sort and Calculate Splits
    print("\nSorting images based on CLIP scores...")
    sorted_images = sorted(valid_scores.items(), key=lambda item: (-item[1], item[0]))

    n_scored = len(sorted_images)
    best_threshold_score, worst_threshold_score = None, None
    n_best, n_worst, n_middle = 0, 0, 0

    if n_scored == 0 and len(image_scores) > 0: # All failed
        failed_dest_key = "worst" if ENABLE_CATEGORY_SORTING else "middle"
        failed_dest_foldername = FOLDER_NAMES[failed_dest_key]
        print(f"No images successfully scored. Copying all failed images (prefixed {failed_prefix}) to '{failed_dest_foldername}'...")
        fail_copied_count, skipped_count = 0, 0
        for filename, score in image_scores.items():
             if score == -float('inf'):
                 source_path = os.path.join(source_dir, filename)
                 if os.path.exists(source_path):
                     try:
                         new_filename = f"{failed_prefix}{filename}" if ENABLE_SCORE_PREFIX else filename
                         dest_path = os.path.join(dest_folders[failed_dest_key], new_filename)
                         shutil.copy2(source_path, dest_path)
                         fail_copied_count += 1
                     except Exception as e: skipped_count += 1; print(f"Err copy failed: {e}", file=sys.stderr)
                 else: skipped_count += 1
        print(f"Copied {fail_copied_count} failed files. Skipped {skipped_count}.")
        sys.exit(0)
    elif n_scored == 0: # None found/scored
         sys.exit("No images found or scored. Nothing to copy or plot.")

    # Calculate counts only if sorting is enabled
    if ENABLE_CATEGORY_SORTING:
        n_best = math.floor(n_scored * (PERCENT_BEST / 100.0))
        n_worst = math.ceil(n_scored * (PERCENT_WORST / 100.0))
        if n_scored > 0 and n_worst + n_best > n_scored: n_worst = n_scored - n_best
        n_middle = n_scored - n_best - n_worst
        if n_middle < 0: n_middle = 0

        print(f"Total successfully scored: {n_scored}")
        print(f"  '{FOLDER_NAMES['best']}' ({PERCENT_BEST}%): ~{n_best} images")
        print(f"  '{FOLDER_NAMES['middle']}' ({PERCENT_MIDDLE}%): ~{n_middle} images")
        print(f"  '{FOLDER_NAMES['worst']}' ({PERCENT_WORST}%): ~{n_worst} images")

        # Determine score thresholds for plot
        if n_best > 0 and n_best <= n_scored: best_threshold_score = sorted_images[n_best - 1][1]; print(f"  Score threshold for 'best': >= {best_threshold_score:.4f}")
        if n_worst > 0 and n_worst < n_scored: worst_threshold_score = sorted_images[n_scored - n_worst][1]; print(f"  Score threshold for 'worst': < {worst_threshold_score:.4f}")
        elif n_worst == n_scored: worst_threshold_score = sorted_images[0][1]; print(f"  Score threshold for 'worst': <= {worst_threshold_score:.4f} (All images)")
    else:
        n_middle = n_scored
        print(f"Total successfully scored: {n_scored} (All will be copied to '{FOLDER_NAMES['middle']}')")


    # --- Plot Distribution ---
    if plt and (SHOW_PLOT or SAVE_PLOT):
        scores_list_for_plot = [score for filename, score in sorted_images]
        plot_best_thresh = best_threshold_score if ENABLE_CATEGORY_SORTING else None
        plot_worst_thresh = worst_threshold_score if ENABLE_CATEGORY_SORTING else None
        plot_score_distribution(scores_list_for_plot, plot_best_thresh, plot_worst_thresh, PLOT_FILENAME, script_dir)


    # 7. Copy Files with Conditional Prefix and Sorting
    copy_desc = "Copying successfully scored files"
    if ENABLE_SCORE_PREFIX: copy_desc += " with score prefix"
    print(f"\n{copy_desc}...")

    copied_count, skipped_count = 0, 0
    for i, (original_filename, score) in enumerate(tqdm(sorted_images, desc="Copying Scored Files")):
        source_path = os.path.join(source_dir, original_filename)
        if not os.path.exists(source_path):
             skipped_count += 1
             continue

        try:
            # Determine filename (conditional prefix)
            if ENABLE_SCORE_PREFIX:
                if n_scored == 1: rank_percentage = 100.0
                else: rank_percentage = ((n_scored - 1.0) - i) / (n_scored - 1.0) * 100.0
                score_int = int(round(rank_percentage))
                score_prefix = f"{score_int:03d}"
                new_filename = f"{score_prefix}_{original_filename}"
            else:
                new_filename = original_filename

            # Determine category (conditional sorting)
            if ENABLE_CATEGORY_SORTING:
                if i < n_best: dest_key = "best"
                elif i >= n_scored - n_worst: dest_key = "worst"
                else: dest_key = "middle"
            else:
                dest_key = "middle" # All go to normal if sorting disabled

            # Construct destination path
            dest_path = os.path.join(dest_folders[dest_key], new_filename)

            # Perform copy
            shutil.copy2(source_path, dest_path)
            copied_count += 1

        except Exception as e:
            err_dest_key = "unknown"
            if 'dest_key' in locals(): err_dest_key = dest_key
            print(f"\nError copying file {original_filename} (as {new_filename}) to {err_dest_key}: {e}", file=sys.stderr)
            skipped_count += 1

    # Handle images that failed scoring (copy to 'worst' OR 'normal' with conditional prefix)
    fail_copied_count = 0
    failed_dest_key = "worst" if ENABLE_CATEGORY_SORTING else "middle" # <<< Use determined dest key
    failed_dest_foldername = FOLDER_NAMES[failed_dest_key]
    print(f"\nCopying files that failed scoring/loading (prefixed {failed_prefix}) to '{failed_dest_foldername}'...")

    for original_filename, score in image_scores.items():
         if score == -float('inf'):
             source_path = os.path.join(source_dir, original_filename)
             if os.path.exists(source_path):
                 try:
                     # Determine filename for failed (conditional prefix)
                     if ENABLE_SCORE_PREFIX:
                         new_filename = f"000_{original_filename}"
                     else:
                         new_filename = original_filename

                     # Use the determined destination key (worst or middle)
                     dest_path = os.path.join(dest_folders[failed_dest_key], new_filename)
                     shutil.copy2(source_path, dest_path)
                     fail_copied_count += 1
                 except Exception as e:
                     print(f"Error copying failed file {original_filename}: {e}", file=sys.stderr)
                     skipped_count += 1
             else: skipped_count += 1

    # --- Final Summary ---
    print("\n--- Process Complete ---")
    print(f"Successfully copied {copied_count} images.")
    if fail_copied_count > 0:
         # Use the actual folder name where failed files went
         print(f"Copied {fail_copied_count} failed images to the '{failed_dest_foldername}' folder.")
    if skipped_count > 0: print(f"Skipped {skipped_count} files (errors or file not found).")
    print(f"\nOriginal images remain in: {source_dir}")
    if ENABLE_CATEGORY_SORTING:
        print(f"Sorted copies are in folders ('{FOLDER_NAMES['worst']}', '{FOLDER_NAMES['middle']}', '{FOLDER_NAMES['best']}') within: {dest_base_dir}")
    else:
        # Updated message
        print(f"All image copies (including failed ones) are in folder '{FOLDER_NAMES['middle']}' within: {dest_base_dir}")

    if ENABLE_SCORE_PREFIX: print("Filenames include percentage score prefix.")
    else: print("Original filenames were used.")

    if plt and SAVE_PLOT: print(f"Score distribution graph saved as {PLOT_FILENAME}")
    print("------------------------")