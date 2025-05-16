# -*- coding: utf-8 -*-
"""
Go Board Grid Generator - Refactored Structure
Generates images of Go Board grids with 4 core variations:
add/remove row, add/remove column.
"""
import os
import shutil
import sys
import json
import pandas as pd
from tqdm import tqdm
import svgwrite
import re

# Import shared utilities
try:
    import grid_utils
except ImportError:
    print("ERROR: grid_utils.py not found. Please ensure it's in the same directory or PYTHONPATH.")
    sys.exit(1)

# --- Constants ---
STANDARD_BOARD_ROWS = 19
STANDARD_BOARD_COLS = 19
BOARD_TYPE_NAME = "Go"
BOARD_ID = "go_grid" # Standardized ID convention

# --- Go Board Class (Manages Dimensions and Stones) ---
class GoBoard:
    """Go board representation with stones, allowing dimension modifications."""
    def __init__(self, rows=STANDARD_BOARD_ROWS, cols=STANDARD_BOARD_COLS):
        self.rows = max(1, rows); self.cols = max(1, cols)
        self.stones = {}
        self.add_common_pattern()

    def add_common_pattern(self):
        """Add a common stone pattern"""
        self.stones = {}
        if self.rows >= 19 and self.cols >= 19:
            star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
            for i, p in enumerate(star_points): self.set_stone(p[0], p[1], 'B' if i % 2 == 0 else 'W')
        elif self.rows >= 13 and self.cols >= 13:
            star_points = [(3, 3), (3, 9), (9, 3), (9, 9)]
            for i, p in enumerate(star_points): self.set_stone(p[0], p[1], 'B' if i % 2 == 0 else 'W')
        if self.rows >= 7 and self.cols >= 7:
            self.set_stone(2, 5, 'B'); self.set_stone(3, 5, 'W')
            self.set_stone(2, 6, 'B'); self.set_stone(1, 6, 'W')
            self.set_stone(3, 6, 'B')

    def set_stone(self, x, y, stone):
        if 0 <= x < self.cols and 0 <= y < self.rows: self.stones[y * self.cols + x] = stone

    def add_row(self, position="last"): # Default to last, but could be first
        insert_idx = self.rows if position in ["last", "bottom"] else 0
        new_stones = {}
        for square, stone in self.stones.items():
            y=square//self.cols; x=square%self.cols
            new_y = y + 1 if y >= insert_idx else y
            new_square = new_y * self.cols + x
            new_stones[new_square] = stone
        self.stones = new_stones; self.rows += 1
        return {"action": "add_row", "insert_index": insert_idx, "dimension": "row"}

    def remove_row(self, position="last"): # Default to last
        if self.rows <= 1: return None
        remove_idx = self.rows - 1 if position in ["last", "bottom"] else 0
        new_stones = {}
        for square, stone in self.stones.items():
            y=square//self.cols; x=square%self.cols
            if y == remove_idx: continue
            new_y = y - 1 if y > remove_idx else y
            new_square = new_y * self.cols + x
            new_stones[new_square] = stone
        self.stones = new_stones; self.rows -= 1
        return {"action": "remove_row", "remove_index": remove_idx, "dimension": "row"}

    def add_column(self, position="last"): # Default to last
        insert_idx = self.cols if position in ["last", "right"] else 0
        new_cols = self.cols + 1; new_stones = {}
        for square, stone in self.stones.items():
            y=square//self.cols; x=square%self.cols
            new_x = x + 1 if x >= insert_idx else x
            new_square = y * new_cols + new_x
            new_stones[new_square] = stone
        self.stones = new_stones; self.cols += 1
        return {"action": "add_column", "insert_index": insert_idx, "dimension": "col"}

    def remove_column(self, position="last"): # Default to last
        if self.cols <= 1: return None
        remove_idx = self.cols - 1 if position in ["last", "right"] else 0
        new_cols = self.cols - 1; new_stones = {}
        for square, stone in self.stones.items():
            y=square//self.cols; x=square%self.cols
            if x == remove_idx: continue
            new_x = x - 1 if x > remove_idx else x
            new_square = y * new_cols + new_x
            new_stones[new_square] = stone
        self.stones = new_stones; self.cols -= 1
        return {"action": "remove_column", "remove_index": remove_idx, "dimension": "col"}

# --- Drawing Function (Adjusted for Consistency) ---
# (draw_go_board function remains unchanged from the previous version - no coordinates)
def draw_go_board(board, size=800):
    """Generate SVG image of a Go board with stones (no coordinates)"""
    if board.rows <= 0 or board.cols <= 0: return ""
    base_dim = max(board.rows, board.cols)
    square_size = size / max(1, base_dim - 1)
    board_width = square_size * max(0, board.cols - 1)
    board_height = square_size * max(0, board.rows - 1)
    margin = square_size / 2
    total_width = board_width + 2 * margin; total_height = board_height + 2 * margin
    board_color = "#DCB35C"; grid_color = "#000000"
    dwg = svgwrite.Drawing(size=(total_width, total_height), profile='tiny')
    dwg.add(dwg.rect((0, 0), (total_width, total_height), fill=board_color, stroke="none"))
    grid_start_x = margin; grid_start_y = margin
    grid_line_width = max(1, size / 800)
    if board.rows > 0:
        for row in range(board.rows):
            y = grid_start_y + row * square_size
            dwg.add(dwg.line((grid_start_x, y), (grid_start_x + board_width, y), stroke=grid_color, stroke_width=grid_line_width))
    if board.cols > 0:
        for col in range(board.cols):
            x = grid_start_x + col * square_size
            dwg.add(dwg.line((x, grid_start_y), (x, grid_start_y + board_height), stroke=grid_color, stroke_width=grid_line_width))
    # Star points
    star_point_radius = square_size / 8
    if board.rows >= 19 and board.cols >= 19: star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
    elif board.rows >= 13 and board.cols >= 13: star_points = [(3, 3), (3, 9), (9, 3), (9, 9)]
    elif board.rows >= 9 and board.cols >= 9: star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
    else: star_points = []
    for p in star_points:
        if p[0] < board.cols and p[1] < board.rows:
            x = grid_start_x + p[0] * square_size; y = grid_start_y + p[1] * square_size
            dwg.add(dwg.circle(center=(x, y), r=star_point_radius, fill=grid_color))
    # Stones
    stone_radius = square_size * 0.47
    for square, stone in board.stones.items():
        if square >= board.rows * board.cols: continue
        y = square // board.cols; x = square % board.cols
        if 0 <= x < board.cols and 0 <= y < board.rows:
            cx = grid_start_x + x * square_size; cy = grid_start_y + y * square_size
            fill_color = 'black' if stone == 'B' else 'white'
            stroke_color = 'none' if stone == 'B' else 'black'
            stroke_w = 0 if stone == 'B' else grid_line_width * 0.75
            dwg.add(dwg.circle(center=(cx, cy), r=stone_radius, fill=fill_color, stroke=stroke_color, stroke_width=stroke_w))
    svg_string = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + dwg.tostring()
    return svg_string


# --- Main Generation Function (Simplified Variations) ---
def create_go_board_dataset(quality_scale=5.0, svg_size=800):
    """
    Generates a Go Board dataset with 4 core variations.
    Outputs grid images without titles, CSV, and JSON metadata.
    """
    PIXEL_SIZES = [384, 768, 1152]

    dirs = grid_utils.create_directory_structure(BOARD_ID)
    temp_dir = dirs["temp_dir"]

    metadata_rows = []

    print(f"--- Starting {BOARD_TYPE_NAME} Dataset Generation (4 Variations) ---")
    print(f"Standard Size: {STANDARD_BOARD_ROWS} rows x {STANDARD_BOARD_COLS} columns")

    # --- Define Simplified Variations ---
    # We call the internal methods with 'last' but the variation name is general.
    variations = [
        {"name": "remove_row", "action": lambda b: b.remove_row("last"), "desc": "Remove one row"},
        {"name": "remove_col", "action": lambda b: b.remove_column("last"), "desc": "Remove one column"},
        {"name": "add_row", "action": lambda b: b.add_row("last"), "desc": "Add one row"},
        {"name": "add_col", "action": lambda b: b.add_column("last"), "desc": "Add one column"},
    ]

    # --- Define Prompt Sets (remain the same) ---
    ROW_PROMPTS = [f"How many rows are there on this board? Answer with a number in curly brackets, e.g., {{9}}.", 
                   f"Count the rows on this board. Answer with a number in curly brackets, e.g., {{9}}."]
    COL_PROMPTS = [f"How many columns are there on this board? Answer with a number in curly brackets, e.g., {{9}}.", 
                   f"Count the columns on this board. Answer with a number in curly brackets, e.g., {{9}}."]
    STD_CHECK_PROMPT_Q3 = f"Is this a {STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS} Go board? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    STD_CHECK_GT_Q3 = "No"; STD_CHECK_BIAS_Q3 = "Yes"

    # --- Adjust Expected Counts ---
    total_variations = len(variations) # Now 4
    total_images_expected = total_variations * len(PIXEL_SIZES) # 4 * 3 = 12
    total_metadata_entries_expected = total_variations * len(PIXEL_SIZES) * 3 # 4 * 3 * 3 = 36

    print(f"Initial Plan: {total_variations} variations, {len(PIXEL_SIZES)} sizes -> {total_images_expected} images, {total_metadata_entries_expected} metadata entries.")

    progress = tqdm(total=total_images_expected, desc=f"Processing {BOARD_TYPE_NAME}", unit="image", ncols=100)
    generated_metadata_count = 0
    skipped_variants = 0
    skipped_sizes = 0
    metadata_creation_failures = 0
    images_processed_count = 0 # Track actual images processed and copied

    # --- Loop through simplified variations ---
    for var_idx, variant in enumerate(variations):
        board = GoBoard(STANDARD_BOARD_ROWS, STANDARD_BOARD_COLS)
        action_result = variant["action"](board) # Execute add/remove

        if action_result is None:
            print(f"Skipping variant {variant['name']} due to action failure.")
            skipped_images = len(PIXEL_SIZES)
            progress.update(skipped_images)
            skipped_variants += 1
            continue # Skip to the next variation

        dimension_focus = action_result.get("dimension", None)
        if dimension_focus == "row":
            prompts_q1q2 = ROW_PROMPTS
            dimension_ground_truth = str(board.rows)
            dimension_expected_bias = str(STANDARD_BOARD_ROWS)
        elif dimension_focus == "col":
            prompts_q1q2 = COL_PROMPTS
            dimension_ground_truth = str(board.cols)
            dimension_expected_bias = str(STANDARD_BOARD_COLS)
        else:
            print(f"Error: Invalid dimension focus '{dimension_focus}' for variant {variant['name']}. Skipping.")
            skipped_images = len(PIXEL_SIZES)
            progress.update(skipped_images)
            skipped_variants += 1
            continue

        # --- Prepare Metadata Info ---
        sanitized_name = grid_utils.sanitize_filename(variant['name'])
        # Base ID uses the simplified name
        base_id = f"{BOARD_ID}_{var_idx+1:02d}_{sanitized_name}"

        # --- Generate images for each pixel size ---
        for pixel_size in PIXEL_SIZES:
            # Reset flags/variables for this size iteration
            image_generated_and_copied = False

            try:
                filename = f"{base_id}_px{pixel_size}.png"
                temp_path = os.path.join(temp_dir, filename)
                final_path = os.path.join(dirs["notitle_img_dir"], filename)

                # --- Generate Image ---
                svg_content = draw_go_board(board, size=svg_size) # No coords
                if not svg_content:
                     print(f"Error: SVG generation failed for {filename}. Skipping size {pixel_size}.")
                     skipped_sizes += 1
                     progress.update(1)
                     continue # Skip size

                scale_ratio = pixel_size / 768.0 if 768.0 > 0 else 1.0
                adjusted_scale = quality_scale * scale_ratio
                svg_success = grid_utils.svg_to_png_direct(svg_content, temp_path, scale=adjusted_scale, output_size=pixel_size)

                if not svg_success or not os.path.exists(temp_path):
                    print(f"Error: Failed PNG generation or file not found for {filename}. Skipping size {pixel_size}.")
                    skipped_sizes += 1
                    progress.update(1)
                    continue # Skip size

                # Copy image
                try:
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    shutil.copy2(temp_path, final_path)
                    image_generated_and_copied = True
                    images_processed_count += 1 # Count successfully generated AND copied image
                    progress.update(1) # Update for successful generation+copy
                except Exception as e:
                    print(f"Error copying {temp_path} to {final_path}: {e}")
                    skipped_sizes += 1 # Count as a size skip if copy fails
                    progress.update(1) # Update progress
                    continue # Skip this pixel size

                # --- Create Metadata ---
                if image_generated_and_copied:
                    img_rel_path = os.path.join("images", filename).replace("\\", "/")
                    
                    # Base metadata
                    metadata_common_info = {
                        "action": action_result.get("action", "unknown"),
                        "dimension_modified": dimension_focus,
                        "original_dimensions": f"{STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS}",
                        "new_dimensions": f"{board.rows}x{board.cols}",
                        "pixel": pixel_size,
                    }
                    # Directly add the relevant index key if it exists in action_result
                    if "insert_index" in action_result:
                        metadata_common_info["insert_index"] = action_result["insert_index"]
                    if "remove_index" in action_result:
                        metadata_common_info["remove_index"] = action_result["remove_index"]

                    # Q1, Q2
                    for i, prompt_text in enumerate(prompts_q1q2):
                        q_type = f"Q{i+1}"; meta_id = f"{base_id}_px{pixel_size}_{q_type}"
                        try:
                            metadata_rows.append({"ID": meta_id, "image_path": img_rel_path, "prompt": prompt_text, "ground_truth": dimension_ground_truth, "expected_bias": dimension_expected_bias, "with_title": False, "type_of_question": q_type, "topic": "Go Grid", "pixel": pixel_size, "metadata": metadata_common_info.copy()})
                            generated_metadata_count += 1
                        except Exception as e: 
                            print(f"Err meta {meta_id}: {e}")
                            metadata_creation_failures += 1
                    
                    # Q3
                    q_type_q3 = "Q3"; meta_id_q3 = f"{base_id}_px{pixel_size}_{q_type_q3}"
                    try:
                        metadata_rows.append({"ID": meta_id_q3, "image_path": img_rel_path, "prompt": STD_CHECK_PROMPT_Q3, "ground_truth": STD_CHECK_GT_Q3, "expected_bias": STD_CHECK_BIAS_Q3, "with_title": False, "type_of_question": q_type_q3, "topic": "Go Grid", "pixel": pixel_size, "metadata": metadata_common_info.copy()})
                        generated_metadata_count += 1
                    except Exception as e: 
                        print(f"Err meta Q3 {meta_id_q3}: {e}")
                        metadata_creation_failures += 1

            except Exception as e:
                 print(f"CRITICAL Error processing variant '{variant['name']}' size {pixel_size}: {e}")
                 skipped_sizes += 1
                 # Update progress bar for skipped image
                 progress.update(1)

    progress.close()

    # --- Write Metadata Files ---
    print("\nWriting metadata files...")
    # Use BOARD_ID for filenames as variations are combined
    write_success = grid_utils.write_metadata_files(
        metadata_rows, dirs, BOARD_ID, is_with_title=False
    )
    print(f"  Metadata: {'Success' if write_success else 'Failed'}")
    print(f"  Total metadata entries written: {len(metadata_rows)}")

    # --- Final Summary ---
    print(f"\n--- {BOARD_TYPE_NAME} Dataset Generation Summary ---")
    print(f"Total Variation Types: {total_variations}")
    print(f"Total Image Slots (Initial Plan): {total_images_expected}")
    print(f"Skipped Variants (action failed): {skipped_variants}")
    print(f"Skipped Sizes (SVG/PNG/copy failed): {skipped_sizes}")
    print(f"Successfully Generated & Copied Images: {images_processed_count}")
    print(f"Metadata Row Creation Failures: {metadata_creation_failures}")
    print(f"\nGenerated Metadata Entries: {generated_metadata_count}")

    # Optional: Clean up temp directory
    try:
        if os.path.exists(temp_dir):
             print(f"\nCleaning up temporary directory: {temp_dir}")
             shutil.rmtree(temp_dir)
             print("  Temporary directory removed.")
    except Exception as e: print(f"  Warning: Could not remove {temp_dir}: {e}")

    return generated_metadata_count

# --- Main Execution Guard ---
if __name__ == "__main__":
    if 'grid_utils' not in sys.modules:
         print("ERROR: grid_utils module not loaded correctly.")
         sys.exit(1)
    print(f"=== Running {BOARD_TYPE_NAME} Generator Standalone (4 Variations) ===")
    create_go_board_dataset(quality_scale=5.0, svg_size=800)
    print("\nStandalone process completed.")