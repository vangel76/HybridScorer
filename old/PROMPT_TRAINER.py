# --- START OF FILE PROMPT_TRAINER.py ---

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import numpy as np
import math
import sys
import collections

# --- Configuration ---
GOOD_IMAGE_FOLDER = "good_training_images"  # Folder with your good example images
BAD_IMAGE_FOLDER = "bad_training_images"    # Folder with your bad example images
OUTPUT_POSITIVE_PROMPTS_FILE = "learned_positive_prompts.txt"
OUTPUT_NEGATIVE_PROMPTS_FILE = "learned_negative_prompts.txt"

# --- !!! NEW: Automatic Keyword Generation Control !!! ---
ENABLE_AUTO_KEYWORD_GENERATION = True # True = Generate keywords from images, False = Use CANDIDATE_PROMPTS_FILE
CANDIDATE_PROMPTS_FILE = "candidate_prompts.txt" # Used if ENABLE_AUTO_KEYWORD_GENERATION = False
NUM_KEYWORDS_FROM_GOOD_SET = 40 # Number of keywords to try and extract from good image captions
NUM_KEYWORDS_FROM_BAD_SET = 40  # Number of keywords to try and extract from bad image captions
MIN_KEYWORD_FREQ = 1 # Minimum frequency for a keyword to be considered (if fewer than NUM_KEYWORDS are found)

NUM_POSITIVE_TO_EXTRACT = 10  # How many top positive prompts to suggest from candidates
NUM_NEGATIVE_TO_EXTRACT = 10  # How many top negative prompts to suggest from candidates

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-large" # "Salesforce/blip-image-captioning-large" for higher quality but slower
BATCH_SIZE_CLIP = 16 # For CLIP image processing
BATCH_SIZE_CAPTION = 8 # For captioning (can be smaller due to memory)
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')

# Attempt to import NLTK and download necessary resources
NLTK_AVAILABLE = False
STOPWORDS = set()
LEMMATIZER = None

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer

    # Structure: resource_key : (download_id, path_to_find_for_verification)
    _NLTK_RESOURCES_TO_CHECK = {
        'punkt_main_data': ('punkt', 'tokenizers/punkt'),
        'punkt_english_tabular_data': ('punkt', 'tokenizers/punkt_tab/english'), # For word_tokenize -> sent_tokenize
        'averaged_perceptron_tagger_main_data': ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        'averaged_perceptron_tagger_english_data': ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger_eng'), # For pos_tag
        'wordnet_data': ('wordnet', 'corpora/wordnet'),
        'stopwords_data': ('stopwords', 'corpora/stopwords')
    }
    _missing_nltk_resources_keys = [] # Tracks keys of resources confirmed missing after all attempts
    _download_attempt_failed_for_keys = [] # Tracks keys of resources for which download explicitly failed

    print("Checking NLTK resources...")
    for resource_key, (download_id, path_to_find) in _NLTK_RESOURCES_TO_CHECK.items():
        try:
            nltk.data.find(path_to_find)
            # print(f"NLTK resource for '{resource_key}' (path: {path_to_find}) found.")
        except LookupError:
            print(f"NLTK resource for '{resource_key}' (path: {path_to_find}) not found. Attempting to download package '{download_id}'...")
            download_successful = False
            try:
                print(f"  Attempting: nltk.download('{download_id}')")
                nltk.download(download_id) 
                nltk.data.find(path_to_find) 
                print(f"  NLTK resource for '{resource_key}' (path: {path_to_find}) successfully downloaded/verified via package '{download_id}'.")
                download_successful = True
            except LookupError as e_verify_after_download:
                print(f"  Verification failed for '{resource_key}' (path: {path_to_find}) after download attempt of '{download_id}': {e_verify_after_download}", file=sys.stderr)
                _download_attempt_failed_for_keys.append(resource_key)
            except Exception as e_download:
                print(f"  Error during download attempt of '{download_id}' for '{resource_key}': {e_download}", file=sys.stderr)
                _download_attempt_failed_for_keys.append(resource_key)
            
            if not download_successful:
                if resource_key not in _missing_nltk_resources_keys:
                     _missing_nltk_resources_keys.append(resource_key)

    # Final check: any resource for which download failed or was never found is missing
    for resource_key, (download_id, path_to_find) in _NLTK_RESOURCES_TO_CHECK.items():
        if resource_key in _download_attempt_failed_for_keys and resource_key not in _missing_nltk_resources_keys:
            _missing_nltk_resources_keys.append(resource_key)
        elif resource_key not in _download_attempt_failed_for_keys: 
            try:
                nltk.data.find(path_to_find)
            except LookupError:
                if resource_key not in _missing_nltk_resources_keys:
                    _missing_nltk_resources_keys.append(resource_key)


    if not _missing_nltk_resources_keys:
        STOPWORDS = set(nltk_stopwords.words('english'))
        LEMMATIZER = WordNetLemmatizer()
        NLTK_AVAILABLE = True
        print("All required NLTK resources are available.")
    else:
        # NLTK_AVAILABLE remains False
        print("\nERROR: One or more essential NLTK resources are missing or could not be downloaded/verified:", file=sys.stderr)
        unique_missing_keys = sorted(list(set(_missing_nltk_resources_keys)))
        for res_key in unique_missing_keys:
            download_id_for_missing, path_for_missing = _NLTK_RESOURCES_TO_CHECK[res_key]
            print(f"  - Check for '{res_key}' (path: '{path_for_missing}') failed. Associated NLTK download package: '{download_id_for_missing}'.", file=sys.stderr)
        
        print("\nPlease try downloading them manually. Open a Python interpreter and run:", file=sys.stderr)
        # Suggest unique download_ids based on the failed keys
        suggested_download_ids = sorted(list(set([_NLTK_RESOURCES_TO_CHECK[key][0] for key in unique_missing_keys])))
        for dl_id in suggested_download_ids:
            print(f"  >>> import nltk; nltk.download('{dl_id}')", file=sys.stderr)
        print("\nIf manual download also fails, check your internet connection, NLTK configuration (download directory permissions), or try a different NLTK mirror if applicable.", file=sys.stderr)
        print("Automatic keyword generation will be disabled.", file=sys.stderr)

except ImportError:
    # NLTK_AVAILABLE remains False.
    print("ERROR: NLTK library not found (pip install nltk).", file=sys.stderr)
    print("Automatic keyword generation will be disabled.", file=sys.stderr)


# --- Default Candidate Prompts (if CANDIDATE_PROMPTS_FILE is not found and auto-gen is off) ---
DEFAULT_CANDIDATE_PROMPTS = [
    # Custom
    "dead woman", "death", "despair", "rape", "strangled", "raw sex", 
    "dynamic lighting", "detailed skin", "low quality", "cartoon", "drawing", "distorted", "dreamy",
    "illustration", "deformed hand", "artifact", "extra fingers", "poorly drawn", "text",
    "letters", "flat lighting",
    # Quality
    "high quality", "low quality", "high resolution", "low resolution", "sharp", "blurry",
    "clear image", "noisy image", "grainy", "pixelated", "compressed", "artifacting",
    "well-lit", "poorly lit", "dark image", "bright image", "overexposed", "underexposed",
    "good contrast", "low contrast", "vibrant colors", "dull colors", "monochrome",
    # Aesthetics & Style
    "beautiful", "ugly", "aesthetic", "unaesthetic", "appealing", "unappealing",
    "photorealistic", "realistic photo", "unrealistic", "cartoon", "anime", "manga", "drawing",
    "illustration", "painting", "sketch", "3d render", "cgi",
    "masterpiece", "amateur", "professional", "snapshot",
    # Composition
    "good composition", "bad composition", "well-composed", "poorly composed",
    "centered", "off-center", "rule of thirds", "leading lines", "balanced", "unbalanced",
    "cluttered", "minimalist", "dynamic pose", "static pose",
    # Common Issues / Content
    "watermark", "text", "logo", "signature", "username", "date stamp",
    "cropped", "badly cropped", "out of frame",
    "deformed", "malformed", "mutated", "extra limbs", "missing limbs", "deformed hands", "extra fingers",
    "ugly face", "beautiful face", "expressive", "emotionless",
    "abstract", "surreal", "fantasy", "sci-fi", "portrait", "landscape", "still life",
    "intricate details", "simple details", "flat shading", "dramatic lighting",
    "erotic", "sexy", "seductive", "provocative", "explicit", "nsfw",
    "cute", "adorable", "creepy", "disturbing", "gore", "violence"
]

# --- Helper Functions ---
def list_image_files(folder_path):
    image_paths = []
    if not os.path.isdir(folder_path):
        print(f"Warning: Folder '{folder_path}' not found.", file=sys.stderr)
        return []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(ALLOWED_EXTENSIONS):
            full_path = os.path.join(folder_path, fname)
            if os.path.isfile(full_path):
                image_paths.append(full_path)
    return image_paths

def generate_captions_for_images(image_paths, caption_model, caption_processor, device):
    captions = []
    if not image_paths: return captions
    print(f"Generating captions for {len(image_paths)} images...")
    caption_model.eval()
    num_batches = math.ceil(len(image_paths) / BATCH_SIZE_CAPTION)
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating Captions"):
            batch_paths = image_paths[i * BATCH_SIZE_CAPTION : (i + 1) * BATCH_SIZE_CAPTION]
            raw_images = []
            for img_path in batch_paths:
                try:
                    raw_images.append(Image.open(img_path).convert("RGB"))
                except Exception as e:
                    print(f"Warning: Could not load image '{os.path.basename(img_path)}' for captioning. Skipping. Error: {e}", file=sys.stderr)
            if not raw_images: continue

            try:
                inputs = caption_processor(images=raw_images, return_tensors="pt", padding=True).to(device)
                outputs = caption_model.generate(**inputs, max_length=50, num_beams=4) 
                batch_captions = caption_processor.batch_decode(outputs, skip_special_tokens=True)
                captions.extend(batch_captions)
            except Exception as e:
                print(f"Error during caption generation for batch: {e}", file=sys.stderr)
            finally:
                if 'inputs' in locals(): del inputs
                if 'outputs' in locals(): del outputs
                if device == 'cuda': torch.cuda.empty_cache()
    return captions

def extract_keywords_from_captions(captions_list, num_keywords_to_extract, min_freq=1):
    if not NLTK_AVAILABLE or not captions_list:
        return []

    print(f"Extracting keywords from {len(captions_list)} captions...")
    all_words = []
    for caption in captions_list:
        tokens = nltk.word_tokenize(caption.lower())
        tagged_tokens = nltk.pos_tag(tokens)
        for word, tag in tagged_tokens:
            if word.isalpha() and word not in STOPWORDS and len(word) > 2: 
                if tag.startswith('NN') or tag.startswith('JJ'):
                    lemma = LEMMATIZER.lemmatize(word, pos='n' if tag.startswith('NN') else 'a')
                    all_words.append(lemma)

    if not all_words: return []
    word_counts = collections.Counter(all_words)
    frequent_enough_keywords = [word for word, count in word_counts.items() if count >= min_freq]
    most_common_keywords = [word for word, count in word_counts.most_common() if word in frequent_enough_keywords]

    return most_common_keywords[:num_keywords_to_extract]


def load_clip_image_features(image_paths, clip_model, clip_processor, device):
    all_image_features_list = []
    if not image_paths:
        return torch.empty(0, clip_model.config.projection_dim if clip_model else 768), [] 

    num_batches = math.ceil(len(image_paths) / BATCH_SIZE_CLIP)
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc=f"Processing images with CLIP"):
            batch_paths = image_paths[i * BATCH_SIZE_CLIP : (i + 1) * BATCH_SIZE_CLIP]
            batch_images = []
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    batch_images.append(image)
                except Exception as e:
                    print(f"Warning: Could not load image '{os.path.basename(img_path)}' for CLIP. Skipping. Error: {e}", file=sys.stderr)
            if not batch_images: continue

            try:
                inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True).to(device)
                image_features = clip_model.get_image_features(**inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                all_image_features_list.append(image_features.cpu())
            except Exception as e:
                print(f"Error processing CLIP image batch: {e}", file=sys.stderr)
            finally:
                if 'inputs' in locals(): del inputs
                if 'image_features' in locals() and device == 'cuda': torch.cuda.empty_cache()

    if not all_image_features_list:
        return torch.empty(0, clip_model.config.projection_dim if clip_model else 768), []
    return torch.cat(all_image_features_list, dim=0), image_paths


def load_manual_candidate_prompts(filepath):
    if os.path.exists(filepath):
        print(f"Loading candidate prompts from '{filepath}'...")
        with open(filepath, 'r', encoding='utf-8') as f:
            candidates = [line.strip() for line in f if line.strip()]
        if candidates:
            print(f"Loaded {len(candidates)} candidate prompts.")
            return candidates
        else:
            print(f"Warning: Candidate prompt file '{filepath}' is empty. Using default prompts.", file=sys.stderr)
    else:
        print(f"Warning: Candidate prompt file '{filepath}' not found. Using default prompts.", file=sys.stderr)
    return DEFAULT_CANDIDATE_PROMPTS.copy()


# --- Main Script ---
if __name__ == "__main__":
    print("--- CLIP Prompt Trainer (with Auto Keyword Generation) ---")
    if ENABLE_AUTO_KEYWORD_GENERATION and not NLTK_AVAILABLE:
        print("\n--- CRITICAL ERROR ---", file=sys.stderr)
        print("Automatic keyword generation is enabled in the script's configuration,", file=sys.stderr)
        print("but NLTK is not fully available due to missing resources (see details printed earlier).", file=sys.stderr)
        print("Please resolve NLTK resource issues or set ENABLE_AUTO_KEYWORD_GENERATION = False in the script.", file=sys.stderr)
        sys.exit(1)

    # 1. Setup Paths
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd(); print(f"Warning: Using CWD for script_dir: {script_dir}")
    good_folder_path = os.path.join(script_dir, GOOD_IMAGE_FOLDER)
    bad_folder_path = os.path.join(script_dir, BAD_IMAGE_FOLDER)
    candidate_prompts_path = os.path.join(script_dir, CANDIDATE_PROMPTS_FILE)
    output_positive_path = os.path.join(script_dir, OUTPUT_POSITIVE_PROMPTS_FILE)
    output_negative_path = os.path.join(script_dir, OUTPUT_NEGATIVE_PROMPTS_FILE)

    # 2. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    print(f"\nLoading CLIP model '{CLIP_MODEL_NAME}'...")
    try:
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model.eval()
    except Exception as e: sys.exit(f"Error loading CLIP model: {e}")

    caption_model, caption_processor = None, None
    if ENABLE_AUTO_KEYWORD_GENERATION:
        print(f"\nLoading Captioning model '{CAPTION_MODEL_NAME}'...")
        try:
            caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME).to(device)
            caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
        except Exception as e:
            print(f"Error loading Captioning model: {e}. Disabling auto keyword generation.", file=sys.stderr)
            ENABLE_AUTO_KEYWORD_GENERATION = False

    # 3. Prepare Candidate Prompts
    candidate_prompts = []
    good_image_paths = list_image_files(good_folder_path)
    bad_image_paths = list_image_files(bad_folder_path)

    if not good_image_paths and not bad_image_paths:
        sys.exit("Error: No images found in good or bad training folders. Cannot proceed.")

    if ENABLE_AUTO_KEYWORD_GENERATION and caption_model and caption_processor:
        print("\n--- Automatic Keyword Generation ---")
        keywords_from_good = []
        if good_image_paths:
            good_captions = generate_captions_for_images(good_image_paths, caption_model, caption_processor, device)
            keywords_from_good = extract_keywords_from_captions(good_captions, NUM_KEYWORDS_FROM_GOOD_SET, MIN_KEYWORD_FREQ)
            print(f"Extracted {len(keywords_from_good)} keywords from 'good' images: {keywords_from_good[:10]}...")
        else: print("No 'good' images to generate keywords from.")

        keywords_from_bad = []
        if bad_image_paths:
            bad_captions = generate_captions_for_images(bad_image_paths, caption_model, caption_processor, device)
            keywords_from_bad = extract_keywords_from_captions(bad_captions, NUM_KEYWORDS_FROM_BAD_SET, MIN_KEYWORD_FREQ)
            print(f"Extracted {len(keywords_from_bad)} keywords from 'bad' images: {keywords_from_bad[:10]}...")
        else: print("No 'bad' images to generate keywords from.")

        candidate_prompts = list(set(keywords_from_good + keywords_from_bad)) 
        print(f"Total unique candidate keywords from auto-generation: {len(candidate_prompts)}")
        if not candidate_prompts:
            print("Warning: Automatic keyword generation yielded no candidates. Falling back to manual/default.", file=sys.stderr)
            ENABLE_AUTO_KEYWORD_GENERATION = False 

    if not ENABLE_AUTO_KEYWORD_GENERATION or not candidate_prompts: 
        print("\n--- Manual/Default Candidate Prompts ---")
        candidate_prompts = load_manual_candidate_prompts(candidate_prompts_path)

    if not candidate_prompts:
        sys.exit("Error: No candidate prompts to work with (neither auto-generated nor manually provided).")
    print(f"Using {len(candidate_prompts)} candidate prompts for scoring.")


    # 4. Get CLIP Text Features for Candidate Prompts
    with torch.no_grad():
        text_inputs = clip_processor(text=candidate_prompts, return_tensors="pt", padding=True).to(device)
        candidate_text_features = clip_model.get_text_features(**text_inputs)
        candidate_text_features /= candidate_text_features.norm(dim=-1, keepdim=True)
        candidate_text_features = candidate_text_features.cpu()

    # 5. Get CLIP Image Features for Good and Bad Images
    good_image_features, _ = load_clip_image_features(good_image_paths, clip_model, clip_processor, device)
    if good_image_features.ndim == 0 or good_image_features.shape[0] == 0:
        print("Warning: No 'good' image features extracted for CLIP.", file=sys.stderr)

    bad_image_features, _ = load_clip_image_features(bad_image_paths, clip_model, clip_processor, device)
    if bad_image_features.ndim == 0 or bad_image_features.shape[0] == 0:
        print("Warning: No 'bad' image features extracted for CLIP.", file=sys.stderr)

    if (good_image_features.ndim == 0 or good_image_features.shape[0] == 0) and \
       (bad_image_features.ndim == 0 or bad_image_features.shape[0] == 0):
        sys.exit("Error: No image features extracted for CLIP from either good or bad folders. Cannot score prompts.")

    # 6. Calculate Average Similarities and Score Prompts
    prompt_scores = {}
    print("\nCalculating prompt similarities with CLIP features...")
    for i, prompt_text in enumerate(tqdm(candidate_prompts, desc="Scoring Prompts")):
        current_text_feature = candidate_text_features[i].unsqueeze(0)
        prompt_scores[prompt_text] = {'avg_good': 0.0, 'avg_bad': 0.0, 'pos_potential': -float('inf'), 'neg_potential': -float('inf')}

        if good_image_features.ndim > 0 and good_image_features.shape[0] > 0:
            good_sim = (good_image_features @ current_text_feature.T).mean().item()
            prompt_scores[prompt_text]['avg_good'] = good_sim

        if bad_image_features.ndim > 0 and bad_image_features.shape[0] > 0:
            bad_sim = (bad_image_features @ current_text_feature.T).mean().item()
            prompt_scores[prompt_text]['avg_bad'] = bad_sim

        if good_image_features.shape[0] > 0 and bad_image_features.shape[0] > 0:
            prompt_scores[prompt_text]['pos_potential'] = prompt_scores[prompt_text]['avg_good'] - prompt_scores[prompt_text]['avg_bad']
            prompt_scores[prompt_text]['neg_potential'] = prompt_scores[prompt_text]['avg_bad'] - prompt_scores[prompt_text]['avg_good']
        elif good_image_features.shape[0] > 0:
            prompt_scores[prompt_text]['pos_potential'] = prompt_scores[prompt_text]['avg_good']
            prompt_scores[prompt_text]['neg_potential'] = -prompt_scores[prompt_text]['avg_good']
        elif bad_image_features.shape[0] > 0:
            prompt_scores[prompt_text]['neg_potential'] = prompt_scores[prompt_text]['avg_bad']
            prompt_scores[prompt_text]['pos_potential'] = -prompt_scores[prompt_text]['avg_bad']


    # 7. Sort and Select Prompts
    sorted_positive = sorted([item for item in prompt_scores.items() if item[1]['pos_potential'] > -float('inf')], key=lambda item: item[1]['pos_potential'], reverse=True)
    selected_positive_prompts = [item[0] for item in sorted_positive[:NUM_POSITIVE_TO_EXTRACT]]

    sorted_negative = sorted([item for item in prompt_scores.items() if item[1]['neg_potential'] > -float('inf')], key=lambda item: item[1]['neg_potential'], reverse=True)
    selected_negative_prompts = [item[0] for item in sorted_negative[:NUM_NEGATIVE_TO_EXTRACT]]

    # 8. Output Results
    print("\n--- Suggested Positive Prompts ---")
    if selected_positive_prompts:
        for i, p in enumerate(selected_positive_prompts):
            score_data = prompt_scores[p]
            print(f"{i+1}. \"{p}\" (Score: {score_data['pos_potential']:.4f}, AvgGood: {score_data['avg_good']:.4f}, AvgBad: {score_data['avg_bad']:.4f})")
        with open(output_positive_path, 'w', encoding='utf-8') as f:
            for p in selected_positive_prompts: f.write(p + "\n")
        print(f"\nSaved {len(selected_positive_prompts)} positive prompts to '{output_positive_path}'")
    else: print("No suitable positive prompts found.")

    print("\n--- Suggested Negative Prompts ---")
    if selected_negative_prompts:
        for i, p in enumerate(selected_negative_prompts):
            score_data = prompt_scores[p]
            print(f"{i+1}. \"{p}\" (Score: {score_data['neg_potential']:.4f}, AvgGood: {score_data['avg_good']:.4f}, AvgBad: {score_data['avg_bad']:.4f})")
        with open(output_negative_path, 'w', encoding='utf-8') as f:
            for p in selected_negative_prompts: f.write(p + "\n")
        print(f"\nSaved {len(selected_negative_prompts)} negative prompts to '{output_negative_path}'")
    else: print("No suitable negative prompts found.")

    print("\n--- Process Complete ---")
    print("Review the .txt files. You can copy-paste these prompts into your STANDARD.py script.")
    if ENABLE_AUTO_KEYWORD_GENERATION:
        print("Keywords were auto-generated. For different results, try varying NUM_KEYWORDS_FROM_GOOD_SET/BAD_SET or MIN_KEYWORD_FREQ.")
    else:
        print("Keywords were loaded from candidate_prompts.txt or defaults. Consider curating this file for your specific domain.")

# --- END OF FILE PROMPT_TRAINER.py ---