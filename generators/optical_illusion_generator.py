"""
Optical Illusions Dataset Generator - "notitle" versions only.
Generates specified optical illusions at multiple resolutions.
Metadata saved per illusion type and combined.
"""
import os
import json
import random
import logging 
import shutil
import pandas as pd
import argparse 
from tqdm import tqdm

try:
    from Pyllusion import pyllusion
    HAS_PYLLUSION = True
except ImportError:
    HAS_PYLLUSION = False

from utils import sanitize_filename as common_sanitize_filename # Using the one from main utils
from utils import save_metadata_files # For saving JSON/CSV

# --- Configuration ---
RESOLUTIONS = [384, 768, 1152] # Standard resolutions
ALL_ILLUSION_TYPES = ["Ebbinghaus", "MullerLyer", "Ponzo", "VerticalHorizontal", "Zollner", "Poggendorff"]


# Base output directory for "notitle" illusions
BASE_NOTITLE_OUTPUT_DIR = "vlms-are-biased-notitle"
TEMP_OUTPUT_DIR_BASE = "temp_optical_illusions_output" # Temporary working directory

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_filename(name_str):
    """Wrapper for common sanitize_filename, handles scientific notation from Pyllusion."""
    name_str = str(name_str)
    # Pyllusion parameters can sometimes result in scientific notation in strings
    name_str = name_str.replace('e-', 'eneg').replace('e+', 'epos')
    name_str = name_str.replace('-', 'neg').replace('+', 'pos')
    name_str = name_str.replace('.', 'p') # Replace decimal point
    return common_sanitize_filename(name_str)

def _create_illusion_output_dirs(illusion_type_name_sanitized):
    """
    Creates the 'notitle' directory structure for a specific illusion type.
    Example: vlms-are-biased-notitle/ebbinghaus_illusion/images/
    Returns a dictionary with paths.
    """
    illusion_base_dir = os.path.join(BASE_NOTITLE_OUTPUT_DIR, illusion_type_name_sanitized)
    illusion_img_dir = os.path.join(illusion_base_dir, "images")
    
    os.makedirs(illusion_img_dir, exist_ok=True) # Creates base if not existing
    logging.info(f"  Ensured output directory for {illusion_type_name_sanitized}: {illusion_img_dir}")
    
    return {
        "base_dir": illusion_base_dir, # For metadata files
        "img_dir": illusion_img_dir    # For PNG images
    }

def define_illusion_parameters_config():
    """
    Defines parameters, questions, and Pyllusion class for each illusion type.
    (This function content is from the original optical_illusion_generator.py)
    """
    illusion_params_config = {
        "Ebbinghaus": {
            "diff_nonzero": {"strengths": [3, 5], "diffs": [0.5, 0.6, 0.7], "with_negative": True},
            "diff_zero": {"strengths": [3, 4, 5, 6, 7, 8], "diffs": [0], "with_negative": True},
            "questions": {"Q1": "Are the two inner circles equal in size? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q2": "Do the two inner circles have the same size? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q3": "Is this an example of the Ebbinghaus illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic_display_name": "Ebbinghaus Illusion", "pyllusion_class": pyllusion.Ebbinghaus
        },
        "MullerLyer": {
            "diff_nonzero": {"strengths": [30, 50], "diffs": [0.3, 0.4, 0.5], "with_negative": True},
            "diff_zero": {"strengths": [30, 35, 40, 45, 50, 55], "diffs": [0], "with_negative": True},
            "questions": {"Q1": "Are the two horizontal lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q2": "Do the two horizontal lines have the same length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q3": "Is this an example of the Müller-Lyer illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic_display_name": "Müller-Lyer Illusion", "pyllusion_class": pyllusion.MullerLyer
        },
        "Ponzo": {
            "diff_nonzero": {"strengths": [15, 18], "diffs": [0.15, 0.20, 0.25], "with_negative": True},
            "diff_zero": {"strengths": [15, 17, 19, 21, 23, 25], "diffs": [0], "with_negative": True},
            "questions": {"Q1": "Are the two horizontal lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q2": "Do the two horizontal lines have the same length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q3": "Is this an example of the Ponzo illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic_display_name": "Ponzo Illusion", "pyllusion_class": pyllusion.Ponzo
        },
        "VerticalHorizontal": { # Renamed for consistency as topic ID
            "diff_nonzero": {"size_mins": [0.5, 1.0], "diffs": [0.3, 0.4, 0.5], "with_negative": False},
            "diff_zero": {"size_mins": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "diffs": [0], "with_negative": False},
            "questions": {"Q1": "Are the horizontal and vertical lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q2": "Do the horizontal and vertical lines have the same length? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q3": "Is this an example of the Vertical-Horizontal illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic_display_name": "Vertical-Horizontal Illusion", "pyllusion_class": pyllusion.VerticalHorizontal
        },
        "Zollner": { # Renamed for consistency as topic ID
            "diff_nonzero": {"strengths": [60, 75], "diffs": [2, 2.5, 3], "with_negative": True},
            "diff_zero": {"strengths": [50, 55, 60, 65, 70, 75], "diffs": [0], "with_negative": True},
            "questions": {"Q1": "Are the main diagonal lines parallel? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q2": "Do the main diagonal lines run parallel to each other? Answer in curly brackets, e.g., {Yes} or {No}.",
                          "Q3": "Is this an example of the Zöllner illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic_display_name": "Zöllner Illusion", "pyllusion_class": pyllusion.Zollner # Corrected class name
        },
        "Poggendorff": {
             "diff_nonzero": {"strengths": [40, 45, 50], "diffs": [0.1, 0.2], "with_negative": True},
             "diff_zero": {"strengths": [30, 35, 40, 45, 50, 55], "diffs": [0], "with_negative": True},
             "questions": {"Q1": "Are the two diagonal line segments collinear (would they form a straight line if extended)? Answer in curly brackets, e.g., {Yes} or {No}.",
                           "Q2": "Do the two diagonal line segments appear to be part of the same continuous straight line? Answer in curly brackets, e.g., {Yes} or {No}.",
                           "Q3": "Is this an example of the Poggendorff illusion? Answer in curly brackets, e.g., {Yes} or {No}."},
            "topic_display_name": "Poggendorff Illusion", "pyllusion_class": pyllusion.Poggendorff
        }
    }
    return illusion_params_config

def _generate_parameter_combinations_for_illusion(illusion_config):
    """Generates all parameter combinations for a single illusion type based on its config."""
    # (Logic from original optical_illusion_generator.py)
    combinations = []
    is_vh_illusion = "size_mins" in illusion_config["diff_zero"] # Check if VerticalHorizontal type params
    
    if is_vh_illusion:
        # For VerticalHorizontal (uses size_min)
        for sz_min in illusion_config["diff_zero"]["size_mins"]:
            combinations.append({"size_min": sz_min, "difference_param": 0, "is_difference_zero": True})
        for sz_min in illusion_config["diff_nonzero"]["size_mins"]:
            for diff_val in illusion_config["diff_nonzero"]["diffs"]:
                combinations.append({"size_min": sz_min, "difference_param": diff_val, "is_difference_zero": False})
    else:
        # For other illusions (using strength)
        for strength_val in illusion_config["diff_zero"]["strengths"]:
            combinations.append({"strength_param": strength_val, "difference_param": 0, "is_difference_zero": True})
            if illusion_config["diff_zero"]["with_negative"]: # Add negative strength if applicable
                combinations.append({"strength_param": -strength_val, "difference_param": 0, "is_difference_zero": True})
        
        for strength_val in illusion_config["diff_nonzero"]["strengths"]:
            for diff_val in illusion_config["diff_nonzero"]["diffs"]:
                combinations.append({"strength_param": strength_val, "difference_param": diff_val, "is_difference_zero": False})
                if illusion_config["diff_nonzero"]["with_negative"]: # Add negative strength and diff
                    combinations.append({"strength_param": -strength_val, "difference_param": -diff_val, "is_difference_zero": False})
    return combinations

def _generate_illusion_images_and_metadata(
    illusion_name_key, # Key from ALL_ILLUSION_TYPES, e.g., "Ebbinghaus"
    illusion_full_config, 
    param_combinations_list, 
    output_dirs_for_illusion, # Dict with "base_dir" and "img_dir"
    temp_dir_path):

    """Generates "notitle" illusion images and metadata for all combinations and resolutions."""
    
    # This function is adapted from the original `generate_illusion_images`
    # It now saves directly to the "notitle" structure and uses passed-in directories.

    all_metadata_for_this_illusion = []
    unique_sample_id_counter = 1 # Used for base part of filename, increments per combination

    # Prepare for Pyllusion class instantiation
    PyllusionClass = illusion_full_config["pyllusion_class"]
    question_templates = illusion_full_config["questions"]
    topic_display = illusion_full_config["topic_display_name"]
    
    # For progress bar: total images = num_combinations * num_resolutions
    total_images_for_illusion = len(param_combinations_list) * len(RESOLUTIONS)
    progress_bar = tqdm(total=total_images_for_illusion, 
                        desc=f"Generating {illusion_name_key}", unit="image", ncols=100, leave=False)

    for combo_params in param_combinations_list:
        difference_value = combo_params["difference_param"]
        is_diff_zero_case = combo_params["is_difference_zero"]
        
        ground_truth_answer = "Yes" if is_diff_zero_case else "No" # For Q1/Q2 type questions
        # Expected bias for Q1/Q2: if it's truly different (GT=No), bias is to say Yes (it looks same)
        # If it's truly same (GT=Yes), bias is to say No (it looks different) - this might need review per illusion
        # For simplicity, let's assume illusion makes different things look same, and same things look different.
        # So, if GT=No (actually different), biased answer is Yes.
        # If GT=Yes (actually same), biased answer is No.
        expected_bias_answer_q1q2 = "No" if is_diff_zero_case else "Yes" 

        # Create a base filename part from parameters for this combination
        base_filename_id_part = f"{sanitize_filename(illusion_name_key)}_{unique_sample_id_counter:03d}"
        param_str_for_filename = ""
        pyllusion_call_params = {"difference": difference_value} # Common Pyllusion param name
        metadata_params_to_store = {"difference_value": difference_value}

        if "size_min" in combo_params: # For VerticalHorizontal
            param_str_for_filename = f"min{sanitize_filename(combo_params['size_min'])}_diff{sanitize_filename(difference_value)}"
            pyllusion_call_params["size_min"] = combo_params['size_min']
            metadata_params_to_store["size_min"] = combo_params['size_min']
        elif "strength_param" in combo_params: # For others
            param_str_for_filename = f"str{sanitize_filename(combo_params['strength_param'])}_diff{sanitize_filename(difference_value)}"
            pyllusion_call_params["illusion_strength"] = combo_params['strength_param']
            metadata_params_to_store["strength"] = combo_params['strength_param']
        
        # Loop through resolutions for this parameter combination
        for res_px in RESOLUTIONS:
            try:
                # Instantiate Pyllusion object with current parameters
                # Note: VerticalHorizontal takes 'difference' and 'size_min'. Others take 'illusion_strength' and 'difference'.
                if illusion_name_key == "VerticalHorizontal":
                    illusion_instance = PyllusionClass(difference=pyllusion_call_params["difference"], 
                                                       size_min=pyllusion_call_params["size_min"])
                else:
                    illusion_instance = PyllusionClass(illusion_strength=pyllusion_call_params["illusion_strength"], 
                                                       difference=pyllusion_call_params["difference"])

                # --- Generate "notitle" image ---
                # Filename: <IllusionType>_<sample_id>_<param_str>_notitle_px<resolution>.png
                notitle_img_basename = f"{base_filename_id_part}_{param_str_for_filename}_notitle_px{res_px}.png"
                # Temporary path for Pyllusion to save to
                temp_img_path = os.path.join(temp_dir_path, notitle_img_basename) 
                # Final path in the "notitle" structure
                final_img_path = os.path.join(output_dirs_for_illusion["img_dir"], notitle_img_basename)

                # Save image using Pyllusion, specifying width and height for resolution
                illusion_instance.to_image(save_path=temp_img_path, width=res_px, height=res_px)
                
                # Copy from temp to final location (or move if preferred)
                shutil.copy2(temp_img_path, final_img_path)
                progress_bar.update(1) # Update after successful image generation and copy

                # --- Create Metadata Entries for this image ---
                # One entry per question type (Q1, Q2, Q3)
                for q_key, q_template_text in question_templates.items():
                    # ID for metadata entry: <IllusionType>_<sample_id>_<param_str>_notitle_px<res>_<Qkey>
                    meta_id_str = f"{base_filename_id_part}_{param_str_for_filename}_notitle_px{res_px}_{q_key}"
                    
                    # Q3 is an identification question ("Is this Ebbinghaus illusion?")
                    # GT for Q3 is always "Yes" as we are generating that illusion.
                    # Bias for Q3 would be "No" if model fails to identify it.
                    current_gt = ground_truth_answer if q_key != "Q3" else "Yes"
                    current_bias_exp = expected_bias_answer_q1q2 if q_key != "Q3" else "No"


                    all_metadata_for_this_illusion.append({
                        "ID": meta_id_str,
                        # Image path relative to the specific illusion's "notitle" image directory
                        "image_path": os.path.join("images", notitle_img_basename).replace("\\", "/"), 
                        "topic": topic_display, # User-friendly topic name
                        "prompt": q_template_text,
                        "ground_truth": current_gt,
                        "expected_bias": current_bias_exp,
                        "with_title": False, # This is for "notitle" generation
                        "type_of_question": q_key,
                        "pixel": res_px,
                        "metadata": { # Detailed parameters
                            "illusion_type_key": illusion_name_key, # e.g., "Ebbinghaus"
                            "is_difference_zero": is_diff_zero_case,
                            "prompt_type_key": q_key,
                            "resolution_px": res_px, 
                            **metadata_params_to_store # Unpack strength/size_min and difference_value
                        }
                    })
            except Exception as e:
                logging.error(f"Error generating {illusion_name_key} (res {res_px}, combo {combo_params}): {e}", exc_info=True)
                progress_bar.update(1) # Update for the attempted image even if it failed
                continue # Skip to next resolution or combination

        unique_sample_id_counter += 1 # Increment after all resolutions for a param combination

    progress_bar.close()
    if progress_bar.n < progress_bar.total:
         logging.warning(f"{illusion_name_key} progress bar finished early ({progress_bar.n}/{progress_bar.total}). Check logs for errors.")
    
    return all_metadata_for_this_illusion


def main(specific_illusion=None): 
    """
    Main function to generate "notitle" optical illusion datasets.
    If `specific_illusion` is provided, only that illusion is generated.
    Otherwise, all illusions defined in `ALL_ILLUSION_TYPES` are generated.
    """
    if not HAS_PYLLUSION:
        logging.error("Pyllusion library is not installed. Optical illusion generation cannot proceed.")
        print("Please install Pyllusion: pip install pyllusion")
        return

    # Determine which illusion types to generate
    if specific_illusion and specific_illusion in ALL_ILLUSION_TYPES:
        illusion_types_to_process = [specific_illusion]
        logging.info(f"Generating ONLY {specific_illusion} illusion as specified.")
    elif specific_illusion == "all" or specific_illusion is None : # Allow "all" from CLI
        illusion_types_to_process = ALL_ILLUSION_TYPES
        logging.info(f"Generating ALL optical illusion types: {', '.join(illusion_types_to_process)}")
    elif specific_illusion:
        logging.error(f"Specified illusion type '{specific_illusion}' is not recognized. Available: {ALL_ILLUSION_TYPES}")
        return
    else:
        illusion_types_to_process = ALL_ILLUSION_TYPES


    logging.info(f"Starting 'notitle' optical illusion dataset generation for resolutions: {RESOLUTIONS}")

    os.makedirs(BASE_NOTITLE_OUTPUT_DIR, exist_ok=True)
    if os.path.exists(TEMP_OUTPUT_DIR_BASE):
        shutil.rmtree(TEMP_OUTPUT_DIR_BASE)
    os.makedirs(TEMP_OUTPUT_DIR_BASE, exist_ok=True)
    logging.info(f"  Using temporary directory: {TEMP_OUTPUT_DIR_BASE}")

    all_illusion_configs = define_illusion_parameters_config()
    

    for illusion_key_name in illusion_types_to_process:
        if illusion_key_name not in all_illusion_configs:
            logging.warning(f"Configuration for illusion type '{illusion_key_name}' not found. Skipping.")
            continue
            
        logging.info(f"--- Processing {illusion_key_name} illusion ---")
        
    
        illusion_dir_name = sanitize_filename(f"{illusion_key_name}_illusion".lower())

        current_illusion_output_dirs = _create_illusion_output_dirs(illusion_dir_name)
        
        current_illusion_params = all_illusion_configs[illusion_key_name]
        param_combos = _generate_parameter_combinations_for_illusion(current_illusion_params)
        
    

        metadata_for_this_type = _generate_illusion_images_and_metadata(
            illusion_key_name, 
            current_illusion_params, 
            param_combos, 
            current_illusion_output_dirs,
            TEMP_OUTPUT_DIR_BASE # Pass path to temp working directory
        )
        
        # Save metadata for this specific illusion type
        if metadata_for_this_type:
            meta_filename_prefix = f"{illusion_dir_name}_notitle"
            save_metadata_files(
                metadata_for_this_type, 
                current_illusion_output_dirs["base_dir"], # Save to "vlms-are-biased-notitle/ebbinghaus_illusion/"
                meta_filename_prefix
            )

    try:
        if os.path.exists(TEMP_OUTPUT_DIR_BASE):
            shutil.rmtree(TEMP_OUTPUT_DIR_BASE)
            logging.info(f"  Temporary directory '{TEMP_OUTPUT_DIR_BASE}' cleaned up.")
    except Exception as e:
        logging.warning(f"  Failed to clean up temporary directory '{TEMP_OUTPUT_DIR_BASE}': {e}")

    logging.info("--- Optical Illusion 'notitle' dataset generation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Optical Illusion "notitle" datasets.')
    parser.add_argument('--illusion', type=str, choices=ALL_ILLUSION_TYPES + ["all"], default="all",
                        help='Generate a specific illusion type, or "all" (default).')
    args = parser.parse_args()
    
    illusion_to_run = args.illusion
    if args.illusion == "all":
        illusion_to_run = None

    main(specific_illusion=illusion_to_run)