"""
Common utility functions for all board/grid type dataset generators.
(e.g., Chess Board, Go Board, Sudoku Board, Xiangqi Board)
These generators typically produce SVG images that need conversion and have
specific metadata structures.
"""
import os
import re
import json
import shutil 
import pandas as pd

try:
    from cairosvg import svg2png
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False
    # This warning will be prominent when grid_utils is imported.
    print("####################################################################")
    print("# WARNING: cairosvg library not found or could not be imported.    #")
    print("# SVG to PNG conversion for board generators WILL FAIL.          #")
    print("# Please install cairosvg: pip install cairosvg                    #")
    print("# Ensure its dependencies (like Cairo graphics library) are also   #")
    print("# correctly installed on your system.                            #")
    print("####################################################################")

try:
    from utils import sanitize_filename as common_sanitize_filename
except ImportError:
    print("ERROR in grid_utils.py: Could not import common_sanitize_filename from utils.py.")
    print("Ensure utils.py is accessible in the Python path.")
    def common_sanitize_filename(name_str):
        name_str = str(name_str)
        name_str = re.sub(r'[^\w\-]+', '_', name_str)
        name_str = re.sub(r'_+', '_', name_str)
        name_str = name_str.strip('_')
        return name_str if name_str else "sanitized_empty_fallback"

def sanitize_filename(name):
    """Wrapper for the common sanitize_filename for local use within grid_utils."""
    return common_sanitize_filename(name)

def create_directory_structure(board_type_sanitized_id):
    """
    Creates the "notitle" directory structure for saving files for a specific board type.
    `board_type_sanitized_id` is the sanitized ID for the board (e.g., "chess_board", "go_board").
    
    Returns a dictionary containing paths for:
        "notitle_img_dir": Directory for "notitle" PNG images.
        "notitle_meta_dir": Directory for "notitle" metadata files (JSON/CSV).
        "temp_dir": Temporary directory for intermediate files for this board type.
    """
    # Base directory for all "notitle" datasets (e.g., "vlms-are-biased-notitle")
    base_notitle_dir_parent = "vlms-are-biased-notitle"
    
    # Specific base directory for this board type under the "notitle" parent
    # e.g., vlms-are-biased-notitle/chess_board/
    notitle_board_topic_base_dir = os.path.join(base_notitle_dir_parent, board_type_sanitized_id)
    
    # Image directory within this board type's "notitle" directory
    # e.g., vlms-are-biased-notitle/chess_board/images/
    notitle_board_img_dir = os.path.join(notitle_board_topic_base_dir, "images")
    
    # Metadata files (.json, .csv) will be stored in notitle_board_topic_base_dir
    # e.g., vlms-are-biased-notitle/chess_board/chess_board_notitle_metadata.json
    
    # Create all necessary directories
    os.makedirs(notitle_board_img_dir, exist_ok=True) # This also creates parent dirs if they don't exist
    # No print statement here by default, caller (generator) can log if needed.
    
    # Temporary directory for intermediate processing specific to this board type
    # e.g., temp_chess_board_output
    temp_dir_for_board = f"temp_{board_type_sanitized_id}_output"
    os.makedirs(temp_dir_for_board, exist_ok=True)
    # No print statement here by default.
    
    return {
        "notitle_img_dir": notitle_board_img_dir,
        "notitle_meta_dir": notitle_board_topic_base_dir, # Metadata stored alongside "images" folder
        "temp_dir": temp_dir_for_board
    }

def svg_to_png_direct(svg_content, output_path, scale=2.0, output_size=768, output_height=None, maintain_aspect=True, aspect_ratio=None):
    """
    Convert SVG content (as a string) directly to a PNG file.
    `output_size` is typically used as the width for aspect ratio calculations.
    `aspect_ratio` is height/width.
    """
    if not HAS_CAIROSVG:
        print(f"  CRITICAL ERROR: cairosvg is not available. Cannot convert SVG to PNG for {os.path.basename(output_path)}.")
        return False
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    target_output_width = output_size
    target_output_height = output_height # Can be None initially

    if target_output_height is None: # If not explicitly provided
        if maintain_aspect and aspect_ratio is not None and aspect_ratio > 0:
            # Calculate height based on target width (output_size) and aspect_ratio
            target_output_height = int(target_output_width * aspect_ratio)
        else:
            # Default to square if no aspect ratio provided or not maintaining aspect
            target_output_height = target_output_width 
    
    try:
        svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=output_path,
            scale=scale, # This scale factor is applied *before* output_width/height constraints
            background_color="white", # Ensure consistent white background
            output_width=target_output_width,   # Target width in pixels
            output_height=target_output_height  # Target height in pixels
        )
        return True
    except Exception as e:
        print(f"  ERROR converting SVG to PNG for {os.path.basename(output_path)}: {e}")
        if not svg_content or not svg_content.strip():
            print("    Hint: The SVG content provided was empty.")
        return False

def write_metadata_files(metadata_rows_list, dirs_struct, board_id_filename_prefix):
    """
    Writes metadata (list of dictionaries) to JSON and CSV files.
    These are for the "notitle" versions generated by board generators.
    
    `dirs_struct`: The dictionary returned by `create_directory_structure()`.
                   Expects `dirs_struct["notitle_meta_dir"]`.
    `board_id_filename_prefix`: The sanitized ID for the board type (e.g., "chess_board").
                                Used to construct filenames like "chess_board_notitle_metadata.json".
    """
    if not metadata_rows_list:
        print(f"  INFO: No metadata rows to write for {board_id_filename_prefix}.")
        return False # Indicate nothing was written
    
    # Metadata files will be named: <board_id_filename_prefix>_notitle_metadata.json/csv
    # Stored in the directory specified by dirs_struct["notitle_meta_dir"]
    meta_output_directory = dirs_struct["notitle_meta_dir"]
    metadata_file_basename = f"{board_id_filename_prefix}_notitle_metadata" # Base for .json and .csv
    
    json_output_path = os.path.join(meta_output_directory, f"{metadata_file_basename}.json")
    csv_output_path = os.path.join(meta_output_directory, f"{metadata_file_basename}.csv")
    
    try:
        # Ensure the output directory exists (should be created by create_directory_structure)
        os.makedirs(meta_output_directory, exist_ok=True)

        # Write JSON file (indented for human readability)
        with open(json_output_path, 'w', encoding='utf-8') as f_json:
            json.dump(metadata_rows_list, f_json, indent=2, ensure_ascii=False)
        print(f"    Successfully wrote JSON metadata: {json_output_path} ({len(metadata_rows_list)} entries)")
            
        # Prepare data for CSV: flatten the 'metadata' sub-dictionary
        flat_data_for_csv = []
        for single_entry in metadata_rows_list:
            entry_copy = single_entry.copy()
            # Safely pop the 'metadata' field; if it doesn't exist, nested_meta_dict will be empty.
            nested_meta_dict = entry_copy.pop('metadata', {}) 
            
            if isinstance(nested_meta_dict, dict): # Ensure it's a dict before iterating
                for meta_key, meta_value in nested_meta_dict.items():
                    # Serialize complex types (like dicts/lists within metadata) to JSON strings for CSV
                    entry_copy[f'meta_{meta_key}'] = json.dumps(meta_value) if isinstance(meta_value, (dict, list)) else meta_value
            elif nested_meta_dict is not None: # If 'metadata' was some other non-null, non-dict value
                 entry_copy['meta_value_direct'] = nested_meta_dict # Store under a generic key

            flat_data_for_csv.append(entry_copy)

        if not flat_data_for_csv: # Should not happen if metadata_rows_list was not empty
            print(f"  INFO: No data structure formed for CSV output for {board_id_filename_prefix}.")
            return True # JSON might have succeeded

        df_for_csv = pd.DataFrame(flat_data_for_csv)

        # Define a preferred order for columns in the CSV for better consistency and readability
        preferred_column_order = [
            'ID', 'image_path', 'topic', 'prompt', 
            'ground_truth', 'expected_bias', 'with_title', 
            'type_of_question', 'pixel' # 'pixel' promoted to top-level for board grids
        ]
        
        # Get all columns that start with 'meta_' and sort them alphabetically
        meta_detail_columns = sorted([col for col in df_for_csv.columns if col.startswith('meta_')])
        
        # Construct the final ordered list of columns for the CSV
        final_csv_columns = []
        # Add preferred columns first, if they exist in the DataFrame
        for col_name in preferred_column_order:
            if col_name in df_for_csv.columns:
                final_csv_columns.append(col_name)
        # Add all 'meta_*' detail columns
        final_csv_columns.extend(meta_detail_columns)
        # Add any remaining columns from the DataFrame (not in preferred or meta_ list)
        # These are sorted alphabetically for consistent ordering.
        other_remaining_columns = sorted([col for col in df_for_csv.columns if col not in final_csv_columns])
        final_csv_columns.extend(other_remaining_columns)

        # Reorder DataFrame columns according to the final_csv_columns list
        df_for_csv = df_for_csv[final_csv_columns]
        
        df_for_csv.to_csv(csv_output_path, index=False, encoding='utf-8')
        print(f"    Successfully wrote CSV metadata: {csv_output_path} ({len(df_for_csv)} rows)")
            
        return True # Success
    except Exception as e:
        print(f"  ERROR writing metadata files for {board_id_filename_prefix} to {meta_output_directory}: {e}")
        import traceback
        traceback.print_exc()
        return False