# -*- coding: utf-8 -*-
"""
Optical Illusions Dataset Generator

Generates Ebbinghaus, MullerLyer, Ponzo, VerticalHorizontal, Zollner, Poggendorff illusions.
Creates images at multiple resolutions (384, 768, 1152).
Includes pixel size in filenames and metadata.
Generates balanced diff=0/!=0 cases.
Creates no-title versions only.
Saves comprehensive metadata (JSON/CSV).

Run with --illusion parameter to generate just one illusion type:
python script.py --illusion VerticalHorizontal
"""


import os
import json
import random
import logging
import shutil
import pandas as pd
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from Pyllusion import pyllusion

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base configuration
RESOLUTIONS = [384, 768, 1152] # <<<--- UPDATED RESOLUTIONS
# Define all available illusion types
ALL_ILLUSION_TYPES = ["Ebbinghaus", "MullerLyer", "Ponzo", "VerticalHorizontal", "Zollner", "Poggendorff"]
# This will be set in main() based on command line args
ILLUSION_TYPES = ALL_ILLUSION_TYPES.copy()

def sanitize_filename(name):
    """Sanitize a string for use in filenames."""
    name = str(name)
    name = name.replace(' ', '_').replace(':', '_').replace('/', '_').replace('\\', '_')
    # Handle scientific notation if Pyllusion parameters result in it
    name = name.replace('e-', 'eneg').replace('e+', 'epos')
    name = name.replace('-', 'neg').replace('+', 'pos')
    # Remove decimals, keeping sign info
    name = name.replace('.', 'p')
    return name


def get_relative_path(full_path, base_dir):
    """Get relative path from base_dir to full_path."""
    try:
        rel_path = os.path.relpath(full_path, base_dir)
        return rel_path.replace(os.sep, '/') # Use forward slashes for consistency
    except ValueError:
        logging.warning(f"Could not determine relative path for {full_path} from {base_dir}. Returning full path.")
        return full_path


def create_directory_structure():
    """Create directory structure for no-title images."""
    notitle_base = "vlms-are-biased-notitle"
    temp_dir = "temp_illusion_output"
    dirs = {
        "notitle_base_dir": notitle_base, "notitle_dirs": {}, "notitle_img_dirs": {},
        "temp_dir": temp_dir
    }

    illusion_types_lower = [it.lower() for it in ILLUSION_TYPES]

    logging.info("Creating directory structures...")
    os.makedirs(notitle_base, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    print(f"  Ensured: {temp_dir}")

    base_path = dirs["notitle_base_dir"]
    for illusion_lower in illusion_types_lower:
        illusion_dir = os.path.join(base_path, illusion_lower)
        img_dir = os.path.join(illusion_dir, "images")
        try:
            os.makedirs(illusion_dir, exist_ok=True); print(f"  Ensured: {illusion_dir}")
            os.makedirs(img_dir, exist_ok=True); print(f"  Ensured: {img_dir}")
            dirs["notitle_dirs"][illusion_lower] = illusion_dir
            dirs["notitle_img_dirs"][illusion_lower] = img_dir
        except OSError as e:
            logging.error(f"ERROR creating directory {illusion_dir} or {img_dir}: {e}")
            raise # Stop if essential directories can't be made

    logging.info("Directory structures created/verified.")
    return dirs


def define_illusion_parameters():
    """Define the parameters for each illusion type."""
    # Keep the same parameter definitions as before
    illusion_params = {
        "Ebbinghaus": {
            "diff_nonzero": {"strengths": [3, 5], "diffs": [0.5, 0.6, 0.7], "with_negative": True},
            "diff_zero": {"strengths": [3, 4, 5, 6, 7, 8], "diffs": [0], "with_negative": True},
            "questions": {"Q1": "Are the two inner circles equal in size? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q2": "Do the two inner circles have the same size? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q3": "Is this an example of the Ebbinghaus illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic": "Ebbinghaus illusion", "class": pyllusion.Ebbinghaus
        },
        "MullerLyer": {
            "diff_nonzero": {"strengths": [30, 50], "diffs": [0.3, 0.4, 0.5], "with_negative": True},
            "diff_zero": {"strengths": [30, 35, 40, 45, 50, 55], "diffs": [0], "with_negative": True},
            "questions": {"Q1": "Are the two horizontal lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q2": "Do the two horizontal lines have the same length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q3": "Is this an example of the Müller-Lyer illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic": "Müller-Lyer illusion", "class": pyllusion.MullerLyer
        },
        "Ponzo": {
            "diff_nonzero": {"strengths": [15, 18], "diffs": [0.15, 0.20, 0.25], "with_negative": True},
            "diff_zero": {"strengths": [15, 17, 19, 21, 23, 25], "diffs": [0], "with_negative": True},
            "questions": {"Q1": "Are the two horizontal lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q2": "Do the two horizontal lines have the same length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q3": "Is this an example of the Ponzo illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic": "Ponzo illusion", "class": pyllusion.Ponzo
        },
        "VerticalHorizontal": {
            "diff_nonzero": {"size_mins": [0.5, 1.0], "diffs": [0.3, 0.4, 0.5], "with_negative": False},
            "diff_zero": {"size_mins": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "diffs": [0], "with_negative": False},
            "questions": {"Q1": "Are the horizontal and vertical lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q2": "Do the horizontal and vertical lines have the same length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q3": "Is this an example of the Vertical–Horizontal illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic": "Vertical-Horizontal illusion", "class": pyllusion.VerticalHorizontal
        },
        "Zollner": {
            "diff_nonzero": {"strengths": [60, 75], "diffs": [2, 2.5, 3], "with_negative": True},
            "diff_zero": {"strengths": [50, 55, 60, 65, 70, 75], "diffs": [0], "with_negative": True},
            "questions": {"Q1": "Are the two horizontal lines parallel? Answer in curly brackets, e.g., {Yes} or {No}.", # Adjusted question for clarity
                          "Q2": "Do the two horizontal lines run parallel? Answer in curly brackets, e.g., {Yes} or {No}.", # Adjusted question
                          "Q3": "Is this an example of the Zöllner illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic": "Zöllner illusion", "class": pyllusion.Zollner
        },
        "Poggendorff": {
             "diff_nonzero": {"strengths": [40, 45, 50], "diffs": [0.1, 0.2], "with_negative": True},
             "diff_zero": {"strengths": [30, 35, 40, 45, 50, 55], "diffs": [0], "with_negative": True},
             "questions": {"Q1": "Are the two diagonal line segments aligned? Answer in curly brackets, e.g., {Yes} or {No}.",
                           "Q2": "Do the two diagonal lines form a straight line? Answer in curly brackets, e.g., {Yes} or {No}.",
                           "Q3": "Is this an example of the Poggendorff illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic": "Poggendorff illusion", "class": pyllusion.Poggendorff
        }
    }
    return illusion_params

def generate_parameter_combinations(params):
    """Generate all parameter combinations for an illusion type."""
    # Keep the same logic as before
    combinations = []
    is_vh_illusion = "size_mins" in params["diff_zero"]
    if is_vh_illusion:
        for size_min in params["diff_zero"]["size_mins"]: combinations.append({"size_min": size_min, "diff": 0, "is_diff_zero": True})
        for size_min in params["diff_nonzero"]["size_mins"]:
            for diff in params["diff_nonzero"]["diffs"]: combinations.append({"size_min": size_min, "diff": diff, "is_diff_zero": False})
    else:
        for strength in params["diff_zero"]["strengths"]:
            combinations.append({"strength": strength, "diff": 0, "is_diff_zero": True})
            if params["diff_zero"]["with_negative"]: combinations.append({"strength": -strength, "diff": 0, "is_diff_zero": True})
        for strength in params["diff_nonzero"]["strengths"]:
            for diff in params["diff_nonzero"]["diffs"]:
                combinations.append({"strength": strength, "diff": diff, "is_diff_zero": False})
                if params["diff_nonzero"]["with_negative"]: combinations.append({"strength": -strength, "diff": -diff, "is_diff_zero": False})
    return combinations


def generate_illusion_images(illusion_name, illusion_class, params, combinations, dirs):
    """Generate illusion images and metadata for all combinations and resolutions."""
    notitle_metadata = []
    sample_id = 1

    # --- ADJUST PROGRESS BAR TOTAL ---
    total_images_to_generate = len(combinations) * len(RESOLUTIONS) # Combinations * Resolutions (No-Title only)
    progress_bar = tqdm(total=total_images_to_generate, desc=f"Generating {illusion_name}", unit="image", ncols=100)

    questions, topic = params["questions"], params["topic"]
    illusion_type_lower = illusion_name.lower() # For directory path

    for combo in combinations:
        diff, is_diff_zero = combo["diff"], combo["is_diff_zero"]
        ground_truth = "Yes" if is_diff_zero else "No"
        expected_bias = "No" if is_diff_zero else "Yes"

        # Format base ID and parameter string for filename base
        base_combo_id = f"{illusion_name}_{sample_id:03d}"
        if "size_min" in combo:
            param_str = f"min{sanitize_filename(combo['size_min'])}_diff{sanitize_filename(diff)}"
            params_dict = {"size_min": combo["size_min"]}
        else:
            param_str = f"str{sanitize_filename(combo['strength'])}_diff{sanitize_filename(diff)}"
            params_dict = {"strength": combo["strength"]}
        params_dict["diff"] = diff # Add diff to params_dict regardless

        # --- LOOP THROUGH RESOLUTIONS ---
        for resolution in RESOLUTIONS:
            try:
                # Create illusion instance (needs to be recreated if parameters change per resolution)
                if illusion_name == "VerticalHorizontal":
                    # For VerticalHorizontal, adjust gap parameter based on resolution
                    illusion = illusion_class(difference=diff, size_min=combo["size_min"])
                else:
                    illusion = illusion_class(illusion_strength=combo["strength"], difference=diff)

                # --- Generate No-Title Version ---
                notitle_filename = f"{illusion_name}_{sample_id:03d}_{param_str}_notitle_px{resolution}.png" # Add resolution
                temp_notitle_path = os.path.join(dirs["temp_dir"], notitle_filename)
                final_notitle_path = os.path.join(dirs["notitle_img_dirs"][illusion_type_lower], notitle_filename)

                # Save the no-title image with specific size
                illusion.to_image(save_path=temp_notitle_path, width=resolution, height=resolution) # <<<--- ADDED SIZE

                shutil.copy2(temp_notitle_path, final_notitle_path)
                progress_bar.update(1) # Update after image generation

                # Metadata for no-title (Q1, Q2, Q3)
                for q_key in ["Q1", "Q2", "Q3"]:
                    notitle_meta_id = f"{base_combo_id}_{q_key}_notitle_px{resolution}" # Add resolution to ID
                    notitle_metadata.append({
                        "ID": notitle_meta_id,
                        "image_path": os.path.join("images", notitle_filename), # Relative path within illusion type dir
                        "topic": topic,
                        "prompt": questions[q_key],
                        "ground_truth": ground_truth,
                        "expected_bias": expected_bias,
                        "with_title": False,
                        "type_of_question": q_key,
                        "pixel": resolution,
                        "metadata": {
                            "illusion_type": illusion_name,
                            "is_diff_zero": is_diff_zero,
                            "prompt_type": q_key,
                            "pixel": resolution, # <<<--- ADDED RESOLUTION TO METADATA
                            **params_dict # Unpack strength/size_min and diff
                        }
                    })

            except Exception as e:
                logging.error(f"Error generating {illusion_name} (res {resolution}, combo {combo}): {e}", exc_info=True)
                progress_bar.update(1) # Update for the attempted image
                continue # Skip to next resolution or combination

        # Increment sample_id *after* processing all resolutions for a combination
        sample_id += 1

    progress_bar.close()
    if progress_bar.n < progress_bar.total:
         logging.warning(f"{illusion_name} progress bar finished early ({progress_bar.n}/{progress_bar.total}). Check logs.")
    return notitle_metadata


def save_metadata(metadata, directory, illusion_type=None):
    """Save metadata to both JSON and CSV files."""
    if not metadata:
        logging.warning(f"No metadata to save for directory {directory}{' (' + illusion_type + ')' if illusion_type else ''}")
        return False

    # Ensure directory exists
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        logging.error(f"Cannot create directory {directory} for metadata: {e}")
        return False

    base_filename = illusion_type if illusion_type else "illusion"
    json_filename = f"{base_filename}_metadata.json"
    csv_filename = f"{base_filename}_metadata.csv"
    json_filepath = os.path.join(directory, json_filename)
    csv_filepath = os.path.join(directory, csv_filename)

    # Save JSON
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved JSON: {json_filepath} ({len(metadata)} entries)")
    except Exception as e:
        logging.error(f"Error saving JSON {json_filepath}: {e}")
        return False # Stop if JSON fails

    # Save CSV
    try:
        df = pd.DataFrame(metadata)
        # Flatten metadata dictionary into separate columns for CSV
        meta_df = pd.json_normalize(df['metadata'])
        meta_df.columns = ['meta_' + str(col) for col in meta_df.columns] # Prefix meta columns
        df = df.drop('metadata', axis=1)
        df = pd.concat([df, meta_df], axis=1)

        # Define preferred column order
        preferred_cols = ['ID', 'image_path', 'topic', 'prompt', 'ground_truth', 'expected_bias', 'with_title', 'type_of_question']
        meta_cols = sorted([col for col in df.columns if col.startswith('meta_')])
        # Order columns
        final_cols_ordered = []
        remaining_cols = list(df.columns)
        for col in preferred_cols + meta_cols:
             if col in remaining_cols:
                 final_cols_ordered.append(col)
                 remaining_cols.remove(col)
        final_cols_ordered.extend(sorted(remaining_cols)) # Add any others sorted

        df = df[final_cols_ordered]
        df.to_csv(csv_filepath, index=False, encoding='utf-8')
        logging.info(f"Saved CSV: {csv_filepath} ({len(df)} rows)")
        return True
    except Exception as e:
        logging.error(f"Error saving CSV {csv_filepath}: {e}")
        return False


def check_balance(combinations):
    """Check balance between diff=0 and diff!=0 combinations."""
    diff_zero_count = sum(1 for combo in combinations if combo["is_diff_zero"])
    diff_nonzero_count = len(combinations) - diff_zero_count
    return diff_zero_count, diff_nonzero_count


def main():
    """Main function to generate the dataset."""
    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser(description='Generate optical illusion dataset with customization options')
    parser.add_argument('--illusion', type=str, choices=ALL_ILLUSION_TYPES, 
                        help='Generate only one specific illusion type')
    args = parser.parse_args()
    
    # Update ILLUSION_TYPES based on command-line argument
    global ILLUSION_TYPES
    if args.illusion:
        ILLUSION_TYPES = [args.illusion]
        logging.info(f"Generating ONLY {args.illusion} illusion as specified by command-line argument")
    else:
        logging.info(f"Generating ALL illusion types: {', '.join(ILLUSION_TYPES)}")

    logging.info(f"Starting illusion dataset generation for resolutions: {RESOLUTIONS}")

    dirs = create_directory_structure()
    illusion_params = define_illusion_parameters()
    all_notitle_metadata = []

    for illusion_name, params in illusion_params.items():
        # Skip illusions not in ILLUSION_TYPES
        if illusion_name not in ILLUSION_TYPES:
            logging.info(f"Skipping {illusion_name} (not selected for generation)")
            continue
            
        logging.info(f"--- Processing {illusion_name} illusion ---")
        illusion_type_lower = illusion_name.lower()
        combinations = generate_parameter_combinations(params)
        diff_zero, diff_nonzero = check_balance(combinations)
        logging.info(f"{illusion_name}: {len(combinations)} combinations ({diff_zero} diff=0, {diff_nonzero} diff!=0)")
        if diff_zero != diff_nonzero:
            logging.warning(f"IMBALANCE for {illusion_name}: diff=0 ({diff_zero}) != diff!=0 ({diff_nonzero})")

        notitle_meta = generate_illusion_images(
            illusion_name, params["class"], params, combinations, dirs
        )
        all_notitle_metadata.extend(notitle_meta)

        # Save illusion-specific metadata
        save_metadata(notitle_meta, dirs["notitle_dirs"][illusion_type_lower], illusion_type_lower)

    # Save combined metadata
    logging.info("--- Saving Combined Metadata ---")
    save_metadata(all_notitle_metadata, dirs["notitle_base_dir"])

    # Clean up temporary directory
    try:
        if os.path.exists(dirs["temp_dir"]):
            shutil.rmtree(dirs["temp_dir"])
            logging.info("Temporary directory cleaned up")
    except Exception as e:
        logging.warning(f"Failed to clean up temporary directory '{dirs['temp_dir']}': {e}")

    logging.info("--- Dataset generation complete ---")

    # Print summary
    print("\n--- Final Summary ---")
    print(f"Resolutions generated: {RESOLUTIONS}")
    print(f"Illusion types generated: {', '.join(ILLUSION_TYPES)}")
    print(f"Total no-title metadata entries: {len(all_notitle_metadata)}")

    total_images_counted = 0
    for illusion_lower in [it.lower() for it in ILLUSION_TYPES]:
        try:
            no_title_img_count = len([f for f in os.listdir(dirs["notitle_img_dirs"][illusion_lower]) if f.endswith('.png')])
            print(f"  {illusion_lower.capitalize()}: {no_title_img_count} no-title images")
            total_images_counted += no_title_img_count
        except FileNotFoundError:
             print(f"  {illusion_lower.capitalize()}: Image directory not found (check generation errors).")
        except Exception as e:
             print(f"  Error counting images for {illusion_lower}: {e}")

    print(f"\nTotal images found in output directories: {total_images_counted}")
    expected_total = 0
    for illusion_name in ILLUSION_TYPES:
        if illusion_name in illusion_params:
            expected_total += len(generate_parameter_combinations(illusion_params[illusion_name])) * len(RESOLUTIONS) # Only no-title
    print(f"Expected total images based on parameters: {expected_total}")
    if total_images_counted != expected_total:
        logging.warning("Mismatch between expected and found images. Check logs for errors during generation.")

    print(f"\nOutput structure:")
    print(f"  No-title base directory: {dirs['notitle_base_dir']}")


if __name__ == "__main__":
    main()