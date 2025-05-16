# chinese_chess_grid_generator.py
# -*- coding: utf-8 -*-
"""
Chinese Chessboard Grid Generator - Refactored Structure (Grid Only)
Generates images of Chinese Chessboard grids with variations in rows/columns.
Follows the structure of the Western Chessboard generator.
Does NOT draw pieces or river text.
"""
import os
import json # Needed for metadata structure if not handled by grid_utils dump
import shutil
from tqdm import tqdm
import svgwrite
import pandas as pd # Needed for metadata structure if not handled by grid_utils dump
import sys
import re # Keep re as sanitize_filename might use it indirectly or for future needs

# Import shared utilities - assuming grid_utils.py is in the same directory or Python path
try:
    # Assuming grid_utils has: create_directory_structure, sanitize_filename,
    # svg_to_png_direct, write_metadata_files
    import grid_utils
except ImportError:
    print("ERROR: grid_utils.py not found. Please ensure it's in the same directory or PYTHONPATH.")
    sys.exit(1)

# --- Constants --- (Need to be accessible to the function)
STANDARD_BOARD_WIDTH = 9
STANDARD_BOARD_HEIGHT = 10
BOARD_TYPE_NAME = "Xiangqi" # Consistent topic name
BOARD_ID = "xiangqi_grid" # Standardized ID convention

# --- Chinese Chess Board Grid Class (Manages Dimensions) ---
class ChineseChessboardGrid:
    """Represents a Chinese Chess board grid, allowing modifications."""
    def __init__(self, rows=STANDARD_BOARD_HEIGHT, cols=STANDARD_BOARD_WIDTH):
        self.rows = max(2, rows) # Min 2 rows for river logic
        self.cols = max(1, cols) # Min 1 col
        # River row index (0-based). Row 4 is the last row before the river (index 4).
        # The gap is between row index river_row and river_row + 1
        self.river_row = 4
        # Adjust river_row if initial rows are different from standard
        # Ensure river row index makes sense (at least one row above and one below the gap)
        self.river_row = max(0, min(self.river_row, self.rows - 2))


    def copy(self):
        """Create a deep copy of the board state"""
        board = ChineseChessboardGrid(self.rows, self.cols)
        board.river_row = self.river_row
        return board

    def add_row(self, position):
        """Add a row at the specified logical position ('first', 'last', 'middle_top', 'middle_bottom')."""
        # Determine the actual insertion index
        if position == "middle_top": # Insert *after* river_row (index river_row + 1)
            insert_idx = self.river_row + 1
        elif position == "middle_bottom": # Insert after the row that *was* originally row 7 (index 6)
             # This means inserting at index river_row + 2 relative to the *current* river position
             insert_idx = self.river_row + 2
        elif position == "first" or position == "top":
            insert_idx = 0
        elif position == "last" or position == "bottom":
            insert_idx = self.rows
        else:
            print(f"Warning: Invalid add row position '{position}', defaulting to 'last'.")
            insert_idx = self.rows

        # Clamp index
        insert_idx = max(0, min(insert_idx, self.rows))

        self.rows += 1
        # Adjust river position if the insertion happens at or before the river's visual middle
        # i.e., if inserting at index <= river_row + 1
        if insert_idx <= self.river_row + 1:
            self.river_row += 1
        # Ensure river_row stays valid
        self.river_row = max(0, min(self.river_row, self.rows - 2))

        return {"action": "add_row", 
                # "position": position, 
                "insert_index": insert_idx, "dimension": "row"}

    def remove_row(self, position):
        """Remove a row at the specified logical position ('first', 'last', 'middle_top', 'middle_bottom')."""
        if self.rows <= 2:
            print("Cannot remove row: board must have at least two rows.")
            return None

        # Determine the actual removal index
        # Note: These indices refer to the rows *before* removal.
        if position == "middle_top": # Remove the row *before* the river (index river_row)
            remove_idx = self.river_row
        elif position == "middle_bottom": # Remove the row *after* the river (index river_row + 1)
             remove_idx = self.river_row + 1
        elif position == "first" or position == "top":
            remove_idx = 0
        elif position == "last" or position == "bottom":
            remove_idx = self.rows - 1
        else:
            print(f"Warning: Invalid remove row position '{position}', defaulting to 'last'.")
            remove_idx = self.rows - 1

        # Clamp index (should be valid if logic above is correct, but check anyway)
        if not (0 <= remove_idx < self.rows):
            print(f"Cannot remove row: index {remove_idx} out of bounds (0-{self.rows-1}). Invalid state?")
            return None

        original_river_row = self.river_row
        self.rows -= 1
        # Adjust river position if a row at or before the original river line is removed
        if remove_idx <= original_river_row:
            self.river_row -= 1
        # Ensure river_row remains valid after removal
        self.river_row = max(0, min(self.river_row, self.rows - 2))

        return {"action": "remove_row", 
                # "position": position, 
                "remove_index": remove_idx, "dimension": "row"}

    def add_column(self, position):
        """Add a column at the specified position ('first', 'last')."""
        # Simplified like Western Chess, only first/last for columns
        if position == "first" or position == "left":
            insert_idx = 0
        elif position == "last" or position == "right":
            insert_idx = self.cols
        else:
            print(f"Warning: Invalid add column position '{position}', defaulting to 'last'.")
            insert_idx = self.cols

        # Clamp index
        insert_idx = max(0, min(insert_idx, self.cols))

        self.cols += 1
        return {"action": "add_column", "position": position, "insert_index": insert_idx, "dimension": "col"}

    def remove_column(self, position):
        """Remove a column at the specified position ('first', 'last')."""
        # Simplified like Western Chess, only first/last for columns
        if self.cols <= 1:
            print("Cannot remove column: board must have at least one column.")
            return None

        if position == "first" or position == "left":
            remove_idx = 0
        elif position == "last" or position == "right":
            remove_idx = self.cols - 1
        else:
            print(f"Warning: Invalid remove column position '{position}', defaulting to 'last'.")
            remove_idx = self.cols - 1

        # Clamp index (redundant given logic, but safe)
        if not (0 <= remove_idx < self.cols):
            print(f"Cannot remove column: index {remove_idx} out of bounds (0-{self.cols-1}).")
            return None

        self.cols -= 1
        return {"action": "remove_column", 
                # "position": position, 
            "remove_index": remove_idx, "dimension": "col"}


# --- Drawing Function (Improved version) ---
def draw_chinese_chessboard_grid(board, size=800, show_coordinates=False):
    """Generate SVG image of a Chinese chess board grid (no pieces, no river text)
       using the dimensions from the board object."""
    if board.rows <= 0 or board.cols <= 0: return "" # Handle empty board case

    # Calculate base dimensions based on standard aspect ratio for consistent look
    h_intervals = STANDARD_BOARD_WIDTH - 1.0
    v_intervals = STANDARD_BOARD_HEIGHT - 1.0
    standard_aspect_ratio = v_intervals / h_intervals
    
    # Colors and Padding - match exactly with second code
    board_color = "#f9e9a9"  # Light yellowish board color
    grid_color = "#000000"   # Black grid lines
    border_color = "#d66500" # Brown/orange border
    padding = size / 12      # Match the second code's padding
    
    # Calculate board dimensions matching the second code's approach
    board_width_px = size - (padding * 2)
    board_height_px = board_width_px * standard_aspect_ratio
    
    # Set total dimensions precisely as in the second code
    total_width = board_width_px + padding * 2
    total_height = board_height_px + padding * 2
    
    # Create SVG with the same profile as second code
    dwg = svgwrite.Drawing(size=(f"{total_width:.2f}", f"{total_height:.2f}"), profile='full', debug=False)
    
    # Add border
    dwg.add(dwg.rect((0, 0), (total_width, total_height), fill=border_color))
    
    # Grid area start coordinates
    grid_start_x = padding
    grid_start_y = padding

    # Inner board background with stroke just like second code
    dwg.add(dwg.rect((grid_start_x, grid_start_y), (board_width_px, board_height_px), 
                     fill=board_color, stroke=grid_color, stroke_width=2))
    
    # Calculate grid intervals
    h_interval = board_width_px / max(1, board.cols - 1)
    v_interval = board_height_px / max(1, board.rows - 1)

    # Grid Lines with same stroke-width as second code
    grid_line_width = 2.0
    line_style = {"stroke": grid_color, "stroke_width": grid_line_width}
    
    # Horizontal lines
    for i in range(board.rows):
        y = grid_start_y + i * v_interval
        dwg.add(dwg.line((grid_start_x, y), (grid_start_x + board_width_px, y), **line_style))

    # Vertical lines with river gap
    for i in range(board.cols):
        x = grid_start_x + i * h_interval
        if i == 0 or i == (board.cols - 1):
            # Outer columns always have a full line
            dwg.add(dwg.line((x, grid_start_y), (x, grid_start_y + board_height_px), **line_style))
        else:
            # Inner columns: draw with river gap - adapt to current board dimensions
            river_row_idx = min(board.river_row, 4) if board.rows >= 10 else board.river_row
            river_gap_start_y = grid_start_y + river_row_idx * v_interval
            river_gap_end_y = grid_start_y + (river_row_idx + 1) * v_interval
            
            # Draw two segments with the river gap
            dwg.add(dwg.line((x, grid_start_y), (x, river_gap_start_y), **line_style))
            dwg.add(dwg.line((x, river_gap_end_y), (x, grid_start_y + board_height_px), **line_style))

    # Draw palaces
    palace_min_col_idx = 3
    palace_max_col_idx = 5
    if board.cols > palace_max_col_idx:
        palace_w = 2 * h_interval
        palace_h = 2 * v_interval
        
        # Top Palace (first 3 rows)
        if board.rows >= 3:
            top_palace_x = grid_start_x + palace_min_col_idx * h_interval
            top_palace_y = grid_start_y
            dwg.add(dwg.line((top_palace_x, top_palace_y), 
                            (top_palace_x + palace_w, top_palace_y + palace_h), **line_style))
            dwg.add(dwg.line((top_palace_x + palace_w, top_palace_y), 
                            (top_palace_x, top_palace_y + palace_h), **line_style))

        # Bottom Palace (last 3 rows)
        if board.rows >= 3:
            bottom_row_idx = max(0, board.rows - 3)
            bottom_palace_x = grid_start_x + palace_min_col_idx * h_interval
            bottom_palace_y = grid_start_y + bottom_row_idx * v_interval
            dwg.add(dwg.line((bottom_palace_x, bottom_palace_y), 
                            (bottom_palace_x + palace_w, bottom_palace_y + palace_h), **line_style))
            dwg.add(dwg.line((bottom_palace_x + palace_w, bottom_palace_y), 
                            (bottom_palace_x, bottom_palace_y + palace_h), **line_style))

    # Coordinates - keep this code if needed
    if show_coordinates and board.cols > 0 and board.rows > 0:
        # (coordinate rendering code would remain unchanged)
        pass

    # Return SVG content as a string with XML declaration
    svg_string = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + dwg.tostring()
    return svg_string

# --- Main Generation Function (Uses Utilities, follows structure of Code 2) ---
def create_chinese_chessboard_dataset(quality_scale=5.0, svg_size=800):
    """
    Generates a Chinese Chessboard Grid dataset with variations.
    Outputs grid images (no pieces, no river text), CSV, and JSON.
    Focuses on dimension-based questions (Q1 rows, Q2 cols) and standard check (Q3).
    Action details (insert/remove index) are stored directly in metadata.
    """
    # BOARD_ID defined above constants
    PIXEL_SIZES = [384, 768, 1152]
    
    dirs = grid_utils.create_directory_structure(BOARD_ID)
    temp_dir = dirs["temp_dir"]

    notitle_metadata_rows = []

    print(f"--- Starting {BOARD_TYPE_NAME} Dataset Generation ---")
    print(f"Standard Size: {STANDARD_BOARD_HEIGHT} rows x {STANDARD_BOARD_WIDTH} columns")

    # --- Define Variations (Unchanged) ---
    variations = [
        {"name": "remove_row_before_river", "action": lambda b: b.remove_row("middle_top"), "desc": "Remove row before river"},
        {"name": "remove_row_after_river", "action": lambda b: b.remove_row("middle_bottom"), "desc": "Remove row after river"},
        {"name": "add_row_in_river_top", "action": lambda b: b.add_row("middle_top"), "desc": "Add row within river (top part)"},
        {"name": "add_row_in_river_bottom", "action": lambda b: b.add_row("middle_bottom"), "desc": "Add row within river (bottom part)"},
        {"name": "remove_first_col", "action": lambda b: b.remove_column("first"), "desc": "Remove first column"},
        {"name": "remove_last_col", "action": lambda b: b.remove_column("last"), "desc": "Remove last column"},
        {"name": "add_first_col", "action": lambda b: b.add_column("first"), "desc": "Add first column"},
        {"name": "add_last_col", "action": lambda b: b.add_column("last"), "desc": "Add last column"}
    ]

    # --- Define Prompt Sets (Unchanged) ---
    ROW_PROMPTS = [
        f"How many horizontal lines are there on this board?  Answer with a number in curly brackets, e.g., {{9}}.",
        f"Count the horizontal lines on this board. Answer with a number in curly brackets, e.g., {{9}}."
    ]
    COL_PROMPTS = [
        f"How many vertical lines are there on this board? Answer with a number in curly brackets, e.g., {{9}}.",
        f"Count the vertical lines on this board. Answer with a number in curly brackets, e.g., {{9}}."
    ]
    STD_CHECK_PROMPT_Q3 = f"Is this a {STANDARD_BOARD_HEIGHT}x{STANDARD_BOARD_WIDTH} {BOARD_TYPE_NAME} board? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    STD_CHECK_GT_Q3 = "No"
    STD_CHECK_BIAS_Q3 = "Yes"

    total_variations = len(variations)
    total_images_expected = total_variations * len(PIXEL_SIZES)
    total_metadata_entries_expected = total_variations * len(PIXEL_SIZES) * 3  # No title entries only

    progress = tqdm(total=total_images_expected, desc=f"Processing {BOARD_TYPE_NAME}", unit="image", ncols=100)
    generated_metadata_count = 0
    skipped_variations = 0
    skipped_sizes = 0
    metadata_creation_failures = 0
    images_processed_count = 0 # Track successful image copies

    # --- Loop through variations ---
    for var_idx, variant in enumerate(variations):
        board = ChineseChessboardGrid(STANDARD_BOARD_HEIGHT, STANDARD_BOARD_WIDTH)
        action_result = variant["action"](board)

        if action_result is None:
            print(f"Skipping variant {variant['name']} due to action failure.")
            skipped_variations += 1
            progress.update(len(PIXEL_SIZES))
            continue # Skip to the next variation

        dimension_focus = action_result.get("dimension", None)
        prompts_q1q2 = []
        dimension_ground_truth = ""
        dimension_expected_bias = ""

        if dimension_focus == "row":
            prompts_q1q2 = ROW_PROMPTS
            dimension_ground_truth = str(board.rows)
            dimension_expected_bias = str(STANDARD_BOARD_HEIGHT)
        elif dimension_focus == "col":
            prompts_q1q2 = COL_PROMPTS
            dimension_ground_truth = str(board.cols)
            dimension_expected_bias = str(STANDARD_BOARD_WIDTH)
        else:
            print(f"Error: Could not determine dimension focus for variant {variant['name']}. Skipping.")
            skipped_variations += 1
            progress.update(len(PIXEL_SIZES))
            continue

        # --- Prepare Base ID ---
        variant_id_num = var_idx + 1
        sanitized_name = grid_utils.sanitize_filename(variant['name'])
        base_id = f"{BOARD_ID}_{variant_id_num:02d}_{dimension_focus}_{sanitized_name}"

        # --- Generate images for each pixel size ---
        for pixel_size in PIXEL_SIZES:
            current_images_processed = 0 # Track images for this size iteration
            try:
                # --- File Naming ---
                no_title_filename = f"{base_id}_notitle_px{pixel_size}.png"
                temp_no_title_path = os.path.join(temp_dir, no_title_filename)
                final_no_title_path = os.path.join(dirs["notitle_img_dir"], no_title_filename)

                # --- Generate No Title Image ---
                svg_content = draw_chinese_chessboard_grid(board, size=svg_size, show_coordinates=False)
                if not svg_content:
                     print(f"Error: Failed SVG generation for {no_title_filename}. Skipping size.")
                     skipped_sizes += 1; progress.update(1); continue

                adjusted_scale = quality_scale # Use scale directly
                aspect_ratio = (STANDARD_BOARD_HEIGHT - 1.0) / (STANDARD_BOARD_WIDTH - 1.0)
                svg_success = grid_utils.svg_to_png_direct(
                    svg_content, temp_no_title_path,
                    scale=adjusted_scale, 
                    output_size=pixel_size,
                    output_height=int(pixel_size * aspect_ratio)  # Ensure consistent aspect ratio
                )
                if not svg_success or not os.path.exists(temp_no_title_path):
                    print(f"Error: Failed PNG generation for {no_title_filename}. Skipping size.")
                    skipped_sizes += 1; progress.update(1); continue

                progress.update(1); current_images_processed += 1
                os.makedirs(os.path.dirname(final_no_title_path), exist_ok=True)
                shutil.copy2(temp_no_title_path, final_no_title_path)

                images_processed_count += current_images_processed # Add successfully processed images for this size

                # --- Prepare Metadata (MODIFIED structure) ---
                metadata_common_info = {
                    "action": action_result.get("action", "unknown"),
                    # "position": action_result.get("position"), # Optional: Include if needed and present
                    "dimension_modified": dimension_focus,
                    "original_dimensions": f"{STANDARD_BOARD_HEIGHT}x{STANDARD_BOARD_WIDTH}",
                    "new_dimensions": f"{board.rows}x{board.cols}",
                    "pixel": pixel_size,
                    # Removed "action_details" key
                }
                # Directly add the relevant index key if it exists in action_result
                if "insert_index" in action_result:
                    metadata_common_info["insert_index"] = action_result["insert_index"]
                if "remove_index" in action_result:
                    metadata_common_info["remove_index"] = action_result["remove_index"]
                # Optionally add position directly if present and desired
                # if "position" in action_result:
                #     metadata_common_info["position"] = action_result["position"]

                # Add "no_title" metadata (Q1, Q2, Q3) - always generated if no-title image exists
                if os.path.exists(final_no_title_path): # Check if base image was successfully copied
                    img_rel_path_no = os.path.join("images", no_title_filename).replace("\\", "/")
                    # Generate Q1 and Q2
                    for i in range(len(prompts_q1q2)):
                        q_type = f"Q{i+1}"
                        prompt_text = prompts_q1q2[i]
                        meta_id = f"{base_id}_notitle_px{pixel_size}_{q_type}"
                        try:
                            notitle_metadata_rows.append({
                                "ID": meta_id, "image_path": img_rel_path_no,
                                "prompt": prompt_text, "ground_truth": dimension_ground_truth,
                                "expected_bias": dimension_expected_bias, "with_title": False,
                                "type_of_question": q_type, "topic": "Xiangqi Grid", # Use constant
                                "pixel": pixel_size,
                                "metadata": metadata_common_info.copy() # Use modified info
                            })
                            generated_metadata_count += 1
                        except Exception as e: print(f"Err meta {meta_id}: {e}"); metadata_creation_failures += 1

                    # Generate Q3
                    q_type_q3 = f"Q{len(prompts_q1q2) + 1}"
                    meta_id_q3 = f"{base_id}_notitle_px{pixel_size}_{q_type_q3}"
                    try:
                        notitle_metadata_rows.append({
                            "ID": meta_id_q3, "image_path": img_rel_path_no,
                            "prompt": STD_CHECK_PROMPT_Q3, "ground_truth": STD_CHECK_GT_Q3,
                            "expected_bias": STD_CHECK_BIAS_Q3, "with_title": False,
                            "type_of_question": q_type_q3, "topic": "Xiangqi Grid", # Use constant
                            "pixel": pixel_size,
                            "metadata": metadata_common_info.copy() # Use modified info
                        })
                        generated_metadata_count += 1
                    except Exception as e: print(f"Err meta {meta_id_q3}: {e}"); metadata_creation_failures += 1
                # --- End Metadata ---

            except Exception as e:
                 # Ensure progress bar is updated for both slots on critical error for the size
                 update_count = max(0, 1 - current_images_processed)
                 progress.update(update_count)
                 print(f"\nCRITICAL Error processing variant '{variant['name']}' size {pixel_size}: {e}\n")
                 skipped_sizes += 1 # Count the size skip


    progress.close()

    # --- Write Metadata Files ---
    print("\nWriting metadata files...")
    # Use BOARD_ID for filenames
    write_success_no = grid_utils.write_metadata_files(
        notitle_metadata_rows, dirs, f"{BOARD_ID}_notitle", is_with_title=False
    )

    if write_success_no: print(f"  Successfully wrote no-title metadata ({len(notitle_metadata_rows)} entries)")
    else: print("  ERROR writing no-title metadata")

    # --- Final Summary ---
    print(f"\n--- {BOARD_TYPE_NAME} Dataset Generation Summary ---")
    print(f"Total Variations Processed: {len(variations) - skipped_variations} / {len(variations)}")
    print(f"Image Generation Attempts (Slots): {progress.n}") # Total updates = slots processed
    print(f"Successfully Generated & Copied Images: {images_processed_count}")
    print(f"Skipped Variants (Action Failure): {skipped_variations}")
    print(f"Skipped Sizes (PNG/Copy Failures): {skipped_sizes}")
    print(f"Metadata Row Creation Failures: {metadata_creation_failures}")
    print(f"\nGenerated Metadata Entries (Total): {generated_metadata_count}")
    print(f"  - No Title: {len(notitle_metadata_rows)}")

    # Optional: Clean up temp directory
    try:
        if os.path.exists(temp_dir):
            print(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
            print("  Temporary directory removed.")
    except Exception as e:
        print(f"  Warning: Could not remove temporary directory {temp_dir}: {e}")

    return generated_metadata_count # Return total metadata entries generated

# --- Main Execution Guard ---
if __name__ == "__main__":
    # Make sure grid_utils is available
    if 'grid_utils' not in sys.modules:
         print("ERROR: grid_utils module not loaded correctly. Please ensure it's importable.")
         sys.exit(1)

    print(f"=== Running {BOARD_TYPE_NAME} Generator Standalone ===")

    # Define parameters for standalone run
    svg_render_size = 800
    png_conversion_scale = 5.0

    # Generate the dataset
    create_chinese_chessboard_dataset(
        quality_scale=png_conversion_scale,
        svg_size=svg_render_size
    )

    print("\nStandalone process completed.")