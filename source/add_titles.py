# -*- coding: utf-8 -*-
import os
import argparse
import time
import sys
import glob
import json
from tqdm import tqdm
from utils import add_title_to_png, load_title, create_directory_structure, save_metadata_files, TITLE_TYPES

# Updated list of all available topics
TOPICS = [
    'chess_grid',
    'chess_pieces', 
    'dice',
    'ebbinghaus',
    'go_grid',
    'mullerlyer',
    'poggendorff',
    'ponzo',
    'sudoku_grid',
    'tally',
    'verticalhorizontal',
    'xiangqi_grid',
    'xiangqi_pieces',
    'zollner'
]

def read_notitle_metadata(topic=None):
    """Read the metadata from the general metadata file"""
    # Use the general metadata file
    meta_path = "vlms-are-biased-notitle/an_notitle.json"
    
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)
        
        # Count total entries and Q3 entries for diagnostics
        total_entries = len(all_metadata)
        q3_entries = sum(1 for entry in all_metadata if entry.get("type_of_question") == "Q3")
        print(f"Total entries in an_notitle.json: {total_entries}, of which {q3_entries} are Q3")
            
        # If topic is specified, filter for that topic
        if topic:
            topic_metadata = [entry for entry in all_metadata 
                             if topic.lower() in entry.get("image_path", "").lower()]
            
            topic_total = len(topic_metadata)
            topic_q3 = sum(1 for entry in topic_metadata if entry.get("type_of_question") == "Q3")
            print(f"Found {topic_total} entries for topic '{topic}', of which {topic_q3} are Q3")
            return topic_metadata
        else:
            return all_metadata
    
    except Exception as e:
        print(f"Error reading metadata from {meta_path}: {e}")
        return []

def find_notitle_images(topic):
    """Find all notitle images for a given topic"""
    img_dir = f"vlms-are-biased-notitle/{topic}/images"
    if not os.path.exists(img_dir):
        print(f"Error: notitle images directory not found: {img_dir}")
        return []
    
    image_files = glob.glob(os.path.join(img_dir, "*.png"))
    return image_files

def add_titles_to_images(topic, title_types=None, dry_run=False, exclude_questions=None):
    """Add titles to notitle images for a given topic"""
    if title_types is None:
        # By default, only process titled versions (skip notitle)
        title_types = [t for t in TITLE_TYPES if t != 'notitle']
    
    if exclude_questions is None:
        exclude_questions = []
    
    print(f"=== Adding titles to {topic} images ===")
    print(f"Title types to process: {title_types}")
    print(f"Excluding questions: {exclude_questions}")
    
    # Create directory structure for titled versions
    dirs = create_directory_structure(topic, title_types=title_types)
    
    # Find all notitle images
    notitle_images = find_notitle_images(topic)
    if not notitle_images:
        print(f"No notitle images found for topic {topic}")
        return {}, False
    
    print(f"Found {len(notitle_images)} notitle images")
    
    # Read notitle metadata to help create titled metadata
    notitle_metadata = read_notitle_metadata(topic)
    if not notitle_metadata:
        print(f"Warning: No metadata found for notitle images. Will generate images but not metadata.")
    
    # Filter out excluded question types
    if exclude_questions and notitle_metadata:
        original_count = len(notitle_metadata)
        notitle_metadata = [entry for entry in notitle_metadata 
                           if entry.get("type_of_question") not in exclude_questions]
        filtered_count = original_count - len(notitle_metadata)
        print(f"Filtered out {filtered_count} entries with question types: {exclude_questions}")
        print(f"Remaining entries after filtering: {len(notitle_metadata)}")
    
    # Create a mapping from image filenames (without path) to a LIST of metadata entries
    # This allows multiple metadata entries per image (for different question types)
    meta_by_filename = {}
    for entry in notitle_metadata:
        image_path = entry.get("image_path", "")
        filename = os.path.basename(image_path) if image_path else ""
        if filename:
            if filename not in meta_by_filename:
                meta_by_filename[filename] = []
            meta_by_filename[filename].append(entry)
    
    # Count unique images and total metadata entries for verification
    unique_images = len(meta_by_filename)
    total_meta_entries = sum(len(entries) for entries in meta_by_filename.values())
    print(f"Mapped {unique_images} unique image filenames to {total_meta_entries} metadata entries")
    
    # Process each title type
    total_images = 0
    all_metadata = {title_type: [] for title_type in title_types}
    
    for title_type in title_types:
        if title_type == 'notitle':
            continue  # Skip notitle as we already have these
            
        print(f"\nProcessing title type: {title_type}")
        title_text = load_title(topic, title_type)
        print(f"  Title text: '{title_text}'")
        
        # Create progress bar
        progress = tqdm(total=len(notitle_images), desc=f"Adding {title_type} titles", unit="image")
        
        titled_count = 0
        metadata_count = 0
        
        for notitle_path in notitle_images:
            # Get the base filename without path
            notitle_filename = os.path.basename(notitle_path)
            
            # Replace 'notitle' with the current title_type in the filename
            titled_filename = notitle_filename.replace('_notitle_', f'_{title_type}_')
            titled_path = os.path.join(dirs["img_dirs"][title_type], titled_filename)
            
            # Skip if the titled file already exists
            if os.path.exists(titled_path) and not dry_run:
                # Check if we need to process metadata for this image even though the image exists
                if notitle_filename in meta_by_filename:
                    for notitle_entry in meta_by_filename[notitle_filename]:
                        titled_entry = notitle_entry.copy()
                        
                        # Update the entry for the titled version
                        titled_entry["ID"] = titled_entry["ID"].replace("_notitle_", f"_{title_type}_")
                        titled_entry["image_path"] = titled_entry["image_path"].replace("_notitle_", f"_{title_type}_")
                        titled_entry["with_title"] = True
                        
                        all_metadata[title_type].append(titled_entry)
                        metadata_count += 1
                
                progress.update(1)
                continue
                
            if not dry_run:
                # Create the titled image
                os.makedirs(os.path.dirname(titled_path), exist_ok=True)
                if add_title_to_png(notitle_path, titled_path, title_text):
                    titled_count += 1
                    
                    # Find corresponding metadata entries by filename
                    if notitle_filename in meta_by_filename:
                        for notitle_entry in meta_by_filename[notitle_filename]:
                            titled_entry = notitle_entry.copy()
                            
                            # Update the entry for the titled version
                            titled_entry["ID"] = titled_entry["ID"].replace("_notitle_", f"_{title_type}_")
                            titled_entry["image_path"] = titled_entry["image_path"].replace("_notitle_", f"_{title_type}_")
                            titled_entry["with_title"] = True
                            
                            all_metadata[title_type].append(titled_entry)
                            metadata_count += 1
            else:
                titled_count += 1
                # Also count metadata in dry run
                if notitle_filename in meta_by_filename:
                    metadata_count += len(meta_by_filename[notitle_filename])
                
                progress.update(1)
                continue
                
            progress.update(1)
            
        progress.close()
        total_images += titled_count
        print(f"  Added {titled_count} {title_type} images with {metadata_count} metadata entries")
        
        # Write topic-specific metadata
        if not dry_run and notitle_metadata and title_type in all_metadata:
            save_metadata_files(
                all_metadata[title_type],
                dirs["meta_dirs"][title_type],
                f"{topic}_{title_type}"
            )
    
    print(f"\nTotal images created: {total_images}")
    return all_metadata, True

def main():
    parser = argparse.ArgumentParser(description='Add titles to existing notitle images')
    parser.add_argument('--topic', type=str, choices=TOPICS + ['all'], default='all',
                      help=f'Which topic to process (default: all)')
    parser.add_argument('--title-types', type=str, nargs='+', choices=TITLE_TYPES,
                      default=['withtitle'],
                      help='Which title types to generate (default: withtitle bias_mitigating)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Dry run - only count images without creating them')
    parser.add_argument('--exclude-questions', type=str, nargs='+', default=['Q3'],
                      help='Question types to exclude from metadata (default: Q3)')
    args = parser.parse_args()
    
    print("========== Adding Titles to Notitle Images ==========")
    print(f"Topic(s): {args.topic}")
    print(f"Title types: {args.title_types}")
    print(f"Exclude questions: {args.exclude_questions}")
    print(f"Dry run: {args.dry_run}")
    print("==================================================")
    
    start_time = time.time()
    
    # Dictionary to collect all metadata across topics for combined file
    combined_metadata = {title_type: [] for title_type in args.title_types}
    
    try:
        # Read all metadata first for statistics
        all_metadata = read_notitle_metadata()
        if args.exclude_questions:
            original_count = len(all_metadata)
            filtered_metadata = [entry for entry in all_metadata 
                               if entry.get("type_of_question") not in args.exclude_questions]
            filtered_count = original_count - len(filtered_metadata)
            print(f"Overall, after filtering Q3: {len(filtered_metadata)} entries (removed {filtered_count})")
        
        topics = []
        if args.topic == 'all':
            topics = TOPICS
        else:
            topics = [args.topic]
            
        for topic in topics:
            topic_metadata, success = add_titles_to_images(topic, args.title_types, args.dry_run, args.exclude_questions)
            
            # Collect metadata for combined file
            if success and not args.dry_run:
                for title_type in args.title_types:
                    if title_type in topic_metadata:
                        combined_metadata[title_type].extend(topic_metadata[title_type])
        
        # Save combined metadata file for each title type
        if not args.dry_run:
            for title_type in args.title_types:
                if combined_metadata[title_type]:
                    # Create the main output directory
                    output_dir = f"vlms-are-biased-{title_type}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save combined metadata file
                    output_path = os.path.join(output_dir, f"an_{title_type}.json")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(combined_metadata[title_type], f, indent=2)
                    
                    print(f"Saved combined metadata to {output_path} ({len(combined_metadata[title_type])} entries)")
            
    except Exception as e:
        print(f"\n!!! FATAL ERROR: {e} !!!")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nProcessing completed in {duration:.2f} seconds.")
        if duration > 60:
            minutes, seconds = divmod(duration, 60)
            print(f"That's {int(minutes)} minutes and {seconds:.2f} seconds.")
        print("====================== Complete ======================")

if __name__ == "__main__":
    main()