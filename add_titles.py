# add_titles.py
# -*- coding: utf-8 -*-
"""
Script to add "in_image_title" to already generated "notitle" images and update metadata.
This script should be run AFTER `main.py` has generated the base "notitle" datasets.
It will also clean up the temporary directories used by the 'notitle' generators.
"""
import os
import argparse
import time
import sys
import glob
import json
import shutil # Ensure shutil is imported for rmtree
import re
from tqdm import tqdm
from utils import (add_title_to_png, load_title, create_directory_structure, 
                   save_metadata_files, sanitize_filename, TITLE_TYPES)

AVAILABLE_TOPICS = [
    'chess_pieces', 'xiangqi_pieces', 'chess_board', 'go_board',
    'xiangqi_board', 'sudoku_board', 'dice_patterned_grid', 'tally_patterned_grid',
    'ebbinghaus_illusion', 'mullerlyer_illusion', 'poggendorff_illusion',
    'ponzo_illusion', 'verticalhorizontal_illusion', 'zollner_illusion',
    # Add 'animals', 'logos', 'flags' here if/when they are implemented
]

def read_notitle_topic_metadata(topic_sanitized):
    """Reads the "<topic_sanitized>_notitle_metadata.json" file for a specific topic."""
    notitle_meta_dir = f"vlms-are-biased-notitle/{topic_sanitized}"
    meta_file = os.path.join(notitle_meta_dir, f"{topic_sanitized}_notitle_metadata.json")
    if not os.path.exists(meta_file):
        print(f"Warning: Metadata file not found for topic '{topic_sanitized}': {meta_file}")
        return []
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"  Successfully read {len(metadata)} metadata entries from {meta_file}")
        return metadata
    except Exception as e:
        print(f"Error reading metadata from {meta_file}: {e}")
        return []

def find_notitle_images_for_topic(topic_sanitized):
    """Finds all 'notitle' PNG images for a given topic."""
    img_dir = f"vlms-are-biased-notitle/{topic_sanitized}/images"
    if not os.path.exists(img_dir):
        print(f"Error: 'notitle' images directory not found: {img_dir}")
        return []
    image_files = glob.glob(os.path.join(img_dir, "*.png"))
    return image_files

def cleanup_notitle_temp_dir(topic_sanitized):
    """Cleans up the temporary directory for a given 'notitle' topic."""
    # Construct the expected temp directory name based on convention
    # e.g., temp_chess_pieces_output, temp_ebbinghaus_illusion_output
    # Note: patterned_grid and optical_illusion generators might have slightly different temp dir names
    # in their original setup. This needs to be consistent.
    # Assuming: temp_<topic_sanitized>_output
    
    # Special handling for patterned grids if its temp dir was different
    if topic_sanitized == "dice_patterned_grid" or topic_sanitized == "tally_patterned_grid":
        # Patterned grid generator might have used a shared temp like "temp_patterned_grid_output"
        # Or if it created "temp_dice_patterned_grid_output", this will work.
        # If it was "temp_grid_output" (from its original dice_tally_generator.py), that's an issue.
        # Let's assume the new convention temp_<topic_id>_output is followed by generators.
        # If patterned_grid_generator used a single "temp_patterned_grid_output",
        # we should only clean it once, perhaps after both dice and tally are processed by add_titles.
        # For now, this will try to clean temp_dice_patterned_grid_output and temp_tally_patterned_grid_output
        pass # No special handling for now, just use the standard name

    temp_dir_path = f"temp_{topic_sanitized}_output"
    
    # Specific temp dir for optical illusions from its generator
    if "illusion" in topic_sanitized:
        temp_dir_path = "temp_optical_illusions_output" # As defined in optical_illusion_generator.py

    if os.path.exists(temp_dir_path):
        try:
            shutil.rmtree(temp_dir_path)
            print(f"  Successfully cleaned up temporary directory: {temp_dir_path}")
        except Exception as e:
            print(f"  Warning: Failed to clean up temporary directory {temp_dir_path}: {e}")
    else:
        print(f"  Note: Temporary directory {temp_dir_path} not found for cleanup (already cleaned or never created).")


def process_topic_for_titling(topic_name, dry_run=False, exclude_questions=None):
    """Adds "in_image_title" to images of a single topic and generates corresponding metadata."""
    if exclude_questions is None: exclude_questions = []

    print(f"\n=== Processing topic: {topic_name} ===")
    topic_sanitized = sanitize_filename(topic_name.lower())
    output_dirs = create_directory_structure(topic_sanitized, title_types_to_create=['in_image_title'])
    notitle_images = find_notitle_images_for_topic(topic_sanitized)

    if not notitle_images:
        print(f"  No 'notitle' images found for topic '{topic_sanitized}'. Skipping.")
        if not dry_run: # Attempt cleanup even if no images, as temp dir might exist from failed run
            cleanup_notitle_temp_dir(topic_sanitized)
        return [], False 
    
    print(f"  Found {len(notitle_images)} 'notitle' images for '{topic_sanitized}'.")
    original_notitle_metadata = read_notitle_topic_metadata(topic_sanitized)
    if not original_notitle_metadata:
        print(f"  Warning: No 'notitle' metadata found for '{topic_sanitized}'. Titles will be added to images, but no new metadata generated.")
    
    filtered_notitle_metadata = [entry for entry in original_notitle_metadata if entry.get("type_of_question") not in exclude_questions]
    num_excluded = len(original_notitle_metadata) - len(filtered_notitle_metadata)
    if num_excluded > 0: print(f"  Excluded {num_excluded} metadata entries based on question types: {exclude_questions}.")
    
    meta_by_filename = {}
    for entry in filtered_notitle_metadata:
        basename = os.path.basename(entry.get("image_path", ""))
        if basename: meta_by_filename.setdefault(basename, []).append(entry)

    in_image_title_metadata_for_topic = []
    images_created_count = 0
    title_text_to_add = load_title(topic_name, "in_image_title") # `topic_name` for consistency in defaults
    print(f"  Using title: '{title_text_to_add}'")

    progress_bar = tqdm(total=len(notitle_images), desc=f"Titling {topic_sanitized}", unit="image", ncols=100)
    for notitle_image_path in notitle_images:
        notitle_basename = os.path.basename(notitle_image_path)
        titled_basename = ""
        if "_notitle_" in notitle_basename:
            titled_basename = notitle_basename.replace("_notitle_", "_in_image_title_")
        else:
            name_part, ext_part = os.path.splitext(notitle_basename)
            px_match = re.search(r"_px\d+$", name_part)
            if px_match:
                titled_basename = f"{name_part[:-len(px_match.group(0))]}_in_image_title{px_match.group(0)}{ext_part}"
            else:
                titled_basename = f"{name_part}_in_image_title{ext_part}"
        
        final_titled_path = os.path.join(output_dirs["img_dirs"]["in_image_title"], titled_basename)

        if not dry_run:
            os.makedirs(os.path.dirname(final_titled_path), exist_ok=True)
            if add_title_to_png(notitle_image_path, final_titled_path, title_text_to_add):
                images_created_count += 1
        else: images_created_count += 1

        if notitle_basename in meta_by_filename:
            for notitle_meta_entry in meta_by_filename[notitle_basename]:
                titled_meta_entry = notitle_meta_entry.copy()
                titled_meta_entry["ID"] = titled_meta_entry["ID"].replace("_notitle_", "_in_image_title_")
                new_relative_image_path = os.path.join("images", titled_basename).replace("\\", "/")
                titled_meta_entry["image_path"] = new_relative_image_path
                titled_meta_entry["with_title"] = True
                in_image_title_metadata_for_topic.append(titled_meta_entry)
        progress_bar.update(1)
    progress_bar.close()

    print(f"  Processed {images_created_count} images for '{topic_sanitized}'. Generated {len(in_image_title_metadata_for_topic)} metadata entries.")

    if not dry_run:
        if in_image_title_metadata_for_topic:
            save_metadata_files(
                in_image_title_metadata_for_topic,
                output_dirs["meta_dirs"]["in_image_title"], 
                f"{topic_sanitized}_in_image_title"
            )
        # Cleanup temp dir for this topic AFTER successful processing
        cleanup_notitle_temp_dir(topic_sanitized) 
    
    return in_image_title_metadata_for_topic, True

def main():
    parser = argparse.ArgumentParser(description='Add "in_image_title" to existing "notitle" datasets and cleanup temp folders.')
    parser.add_argument('--topic', type=str, choices=AVAILABLE_TOPICS + ['all'], default='all',
                        help='Which topic to process. Use "all" for all available topics.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate the process: count images and metadata entries without creating files or cleaning temp folders.')
    parser.add_argument('--exclude-questions', type=str, nargs='*', default=['Q3'],
                        help='Question types (e.g., Q1 Q2 Q3) to exclude from the titled metadata. Default: Q3.')
    args = parser.parse_args()
    
    print("========== Adding 'in_image_title' to Datasets & Cleaning Temp ==========")
    # ... (rest of print statements) ...
    
    start_time = time.time()
    all_topics_combined_in_image_title_metadata = []
    topics_to_process = AVAILABLE_TOPICS if args.topic == 'all' else ([args.topic] if args.topic in AVAILABLE_TOPICS else [])

    if not topics_to_process and args.topic != 'all':
        print(f"Error: Specified topic '{args.topic}' is not recognized. Available: {AVAILABLE_TOPICS}")
        sys.exit(1)
            
    for current_topic_name in topics_to_process:
        topic_specific_metadata, success = process_topic_for_titling(
            current_topic_name, args.dry_run, args.exclude_questions
        )
        if success and topic_specific_metadata:
            all_topics_combined_in_image_title_metadata.extend(topic_specific_metadata)
    
    if not args.dry_run and all_topics_combined_in_image_title_metadata:
        combined_output_dir = "vlms-are-biased-in_image_title"
        os.makedirs(combined_output_dir, exist_ok=True)
        combined_json_path = os.path.join(combined_output_dir, "in_image_title.json")
        try:
            with open(combined_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_topics_combined_in_image_title_metadata, f, indent=2)
            print(f"\nSuccessfully saved combined 'in_image_title' metadata to: {combined_json_path} ({len(all_topics_combined_in_image_title_metadata)} entries)")
        except Exception as e:
            print(f"\nERROR writing combined JSON metadata file {combined_json_path}: {e}")
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTitle addition and cleanup process completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")
    print("====================== Titling & Cleanup Complete ======================")

if __name__ == "__main__":
    main()