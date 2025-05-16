# -*- coding: utf-8 -*-
"""
Sudoku Grid Generator - Variations with Visual Structure Difference
Generates images of Sudoku grids with dimension variations, always highlighting
the 3x3 block structure based on the original 9x9 grid's position.
Coordinates are NOT displayed. All variations are saved to the same metadata files.
"""
import os
import shutil
import sys
import json
import pandas as pd
from tqdm import tqdm
import svgwrite
import re

# Import shared utilities - assuming grid_utils.py is in the same directory or Python path
try:
    import grid_utils
except ImportError:
    print("ERROR: grid_utils.py not found. Please ensure it's in the same directory or PYTHONPATH.")
    sys.exit(1)

# --- Constants ---
STANDARD_BOARD_ROWS = 9
STANDARD_BOARD_COLS = 9
BOARD_TYPE_NAME = "Sudoku" # Consistent topic name
BOARD_ID = "sudoku_grid" # Standardized ID convention

# --- Sudoku Grid Class (Manages Dimensions and Numbers) ---
class SudokuGrid:
    """Sudoku grid representation with example numbers, allowing dimension modifications."""
    def __init__(self, rows=STANDARD_BOARD_ROWS, cols=STANDARD_BOARD_COLS):
        self.rows = max(1, rows)
        self.cols = max(1, cols)
        self._original_rows = STANDARD_BOARD_ROWS
        self._original_cols = STANDARD_BOARD_COLS
        self.numbers = {}
        self.add_example_numbers()

    def add_example_numbers(self):
        """Add standard example pattern to the top-left 9x9 area."""
        self.numbers = {}
        if self.rows >= self._original_rows and self.cols >= self._original_cols:
            example_pattern = {
                (0, 0): 5, (1, 0): 3, (4, 0): 7, (0, 1): 6, (3, 1): 1,
                (4, 1): 9, (5, 1): 5, (1, 2): 9, (2, 2): 8, (7, 2): 6,
                (0, 3): 8, (4, 3): 6, (8, 3): 3, (0, 4): 4, (3, 4): 8,
                (5, 4): 3, (8, 4): 1, (0, 5): 7, (4, 5): 2, (8, 5): 6,
                (1, 6): 6, (6, 6): 2, (7, 6): 8, (3, 7): 4, (4, 7): 1,
                (5, 7): 9, (8, 7): 5, (4, 8): 8, (7, 8): 7, (8, 8): 9
            }
            for (x, y), number in example_pattern.items():
                 if x < self.cols and y < self.rows:
                    self.set_number(x, y, number)
        elif self.rows >= 4 and self.cols >= 4: # Smaller pattern for smaller grids
             if 0 < self.cols and 0 < self.rows: self.set_number(0, 0, 1)
             if 1 < self.cols and 1 < self.rows: self.set_number(1, 1, 2)
             if 2 < self.cols and 2 < self.rows: self.set_number(2, 2, 3)
             if 3 < self.cols and 3 < self.rows: self.set_number(3, 3, 4)

    def set_number(self, x, y, number):
        """Set a number at a specific position (x,y are 0-indexed)"""
        if 0 <= x < self.cols and 0 <= y < self.rows and isinstance(number, int) and 1 <= number <= 9:
            square = y * self.cols + x
            self.numbers[square] = number

    def add_row(self, position):
        """Add a row. Returns action details including original content start."""
        insert_idx = -1; position_str = None; orig_content_start_y = 0
        if position in ["first", "top"]: insert_idx = 0; position_str = "first"; orig_content_start_y = 1
        elif position in ["last", "bottom"]: insert_idx = self.rows; position_str = "last"; orig_content_start_y = 0
        else: insert_idx = self.rows; position_str = "last"; orig_content_start_y = 0
        new_numbers = {}
        for square, number in self.numbers.items():
            y_old=square//self.cols; x_old=square%self.cols; y_new=y_old+1 if y_old>=insert_idx else y_old
            new_square = y_new * self.cols + x_old; new_numbers[new_square] = number
        self.numbers = new_numbers; self.rows += 1
        return {"action": "add", "dimension": "row", "position": position_str, "insert_index": insert_idx,
                "new_dimensions": f"{self.rows}x{self.cols}", "orig_start_x_in_new": 0, "orig_start_y_in_new": orig_content_start_y}

    def remove_row(self, position):
        """Remove a row. Returns action details."""
        if self.rows <= 1: return None
        remove_idx = -1; position_str = None
        if position in ["first", "top"]: remove_idx = 0; position_str = "first"
        elif position in ["last", "bottom"]: remove_idx = self.rows - 1; position_str = "last"
        else: remove_idx = self.rows - 1; position_str = "last"
        new_numbers = {}
        for square, number in self.numbers.items():
            y_old=square//self.cols; x_old=square%self.cols;
            if y_old == remove_idx: continue
            y_new = y_old - 1 if y_old > remove_idx else y_old
            new_square = y_new * self.cols + x_old; new_numbers[new_square] = number
        self.numbers = new_numbers; self.rows -= 1
        return {"action": "remove", "dimension": "row", "position": position_str, "remove_index": remove_idx,
                "new_dimensions": f"{self.rows}x{self.cols}"}

    def add_column(self, position):
        """Add a column. Returns action details including original content start."""
        insert_idx = -1; position_str = None; orig_content_start_x = 0
        if position in ["first", "left"]: insert_idx = 0; position_str = "first"; orig_content_start_x = 1
        elif position in ["last", "right"]: insert_idx = self.cols; position_str = "last"; orig_content_start_x = 0
        else: insert_idx = self.cols; position_str = "last"; orig_content_start_x = 0
        new_cols = self.cols + 1; new_numbers = {}
        for square, number in self.numbers.items():
            y_old=square//self.cols; x_old=square%self.cols
            x_new = x_old + 1 if x_old >= insert_idx else x_old
            new_square = y_old * new_cols + x_new; new_numbers[new_square] = number
        self.numbers = new_numbers; self.cols = new_cols
        return {"action": "add", "dimension": "col", "position": position_str, "insert_index": insert_idx,
                "new_dimensions": f"{self.rows}x{self.cols}", "orig_start_x_in_new": orig_content_start_x, "orig_start_y_in_new": 0}

    def remove_column(self, position):
        """Remove a column. Returns action details."""
        if self.cols <= 1: return None
        remove_idx = -1; position_str = None
        if position in ["first", "left"]: remove_idx = 0; position_str = "first"
        elif position in ["last", "right"]: remove_idx = self.cols - 1; position_str = "last"
        else: remove_idx = self.cols - 1; position_str = "last"
        new_cols = self.cols - 1; new_numbers = {}
        for square, number in self.numbers.items():
            y_old=square//self.cols; x_old=square%self.cols
            if x_old == remove_idx: continue
            x_new = x_old - 1 if x_old > remove_idx else x_old
            if 0 <= x_new < new_cols: new_square = y_old * new_cols + x_new; new_numbers[new_square] = number
        self.numbers = new_numbers; self.cols = new_cols
        return {"action": "remove", "dimension": "col", "position": position_str, "remove_index": remove_idx,
                "new_dimensions": f"{self.rows}x{self.cols}"}

# --- Drawing Function (No Coordinates, Bold 3x3 Structure Lines) ---
def draw_sudoku_grid(grid, size=800,
                     orig_grid_start_x_in_new=0, orig_grid_start_y_in_new=0):
    """
    Generate SVG image of a Sudoku grid WITHOUT coordinates.
    Bold lines emphasize the original 9x9's 3x3 block structure based on its position.
    """
    if grid.rows <= 0 or grid.cols <= 0: return ""
    base_dim = max(grid.rows, grid.cols); square_size = max(size / base_dim, 20)
    grid_width = square_size * grid.cols; grid_height = square_size * grid.rows
    margin = square_size / 4; total_width = grid_width + 2 * margin; total_height = grid_height + 2 * margin
    background_color="#FFFFFF"; grid_color="#000000"; number_color="#000000"
    line_thickness_outer=2.0; line_thickness_3x3_boundary=2.0; line_thickness_standard=0.5
    dwg = svgwrite.Drawing(size=(total_width, total_height))
    dwg.add(dwg.rect((0, 0), (total_width, total_height), fill=background_color, stroke="none"))
    grid_start_x = margin; grid_start_y = margin
    # Horizontal lines
    for i in range(grid.rows + 1):
        y = grid_start_y + i * square_size; stroke_width = line_thickness_standard
        is_outer_border = (i == 0 or i == grid.rows); relative_row = i - orig_grid_start_y_in_new
        is_3x3_boundary = ((relative_row in [0, 3, 6, 9]) and (orig_grid_start_y_in_new <= i <= orig_grid_start_y_in_new + STANDARD_BOARD_ROWS) and (0 <= i <= grid.rows))
        if is_outer_border: stroke_width = line_thickness_outer
        if is_3x3_boundary: stroke_width = line_thickness_3x3_boundary
        dwg.add(dwg.line((grid_start_x, y), (grid_start_x + grid_width, y), stroke=grid_color, stroke_width=stroke_width))
    # Vertical lines
    for i in range(grid.cols + 1):
        x = grid_start_x + i * square_size; stroke_width = line_thickness_standard
        is_outer_border = (i == 0 or i == grid.cols); relative_col = i - orig_grid_start_x_in_new
        is_3x3_boundary = ((relative_col in [0, 3, 6, 9]) and (orig_grid_start_x_in_new <= i <= orig_grid_start_x_in_new + STANDARD_BOARD_COLS) and (0 <= i <= grid.cols))
        if is_outer_border: stroke_width = line_thickness_outer
        if is_3x3_boundary: stroke_width = line_thickness_3x3_boundary
        dwg.add(dwg.line((x, grid_start_y), (x, grid_start_y + grid_height), stroke=grid_color, stroke_width=stroke_width))
    # Numbers
    number_font_size = square_size * 0.6; number_font_family = "Arial, sans-serif"; text_baseline_correction = square_size * 0.05
    for square, number in grid.numbers.items():
        if square >= grid.rows * grid.cols: continue
        y = square // grid.cols; x = square % grid.cols
        if 0 <= x < grid.cols and 0 <= y < grid.rows:
            cx = grid_start_x + (x + 0.5) * square_size; cy = grid_start_y + (y + 0.5) * square_size
            dwg.add(dwg.text(str(number), insert=(cx, cy + text_baseline_correction), fill=number_color, font_size=number_font_size,
                             font_family=number_font_family, font_weight="bold", text_anchor="middle", dominant_baseline="central"))
    svg_string = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + dwg.tostring()
    return svg_string

# --- Main Generation Function (Following Chess structure) ---
def create_sudoku_grid_dataset(quality_scale=5.0, svg_size=800):
    """
    Generates Sudoku Grid dataset variations. NO coordinates are displayed.
    Bold lines emphasize the 3x3 block structure relative to original 9x9 position.
    Uses the same prompt structure and filename conventions as Chess code.
    """
    PIXEL_SIZES = [384, 768, 1152]
    
    dirs = grid_utils.create_directory_structure(BOARD_ID)
    temp_dir = dirs["temp_dir"]

    # Use single list to hold all metadata (removed 'withtitle' list)
    notitle_metadata_rows = []

    print(f"--- Starting {BOARD_TYPE_NAME} Dataset Generation (No Coordinates, Chess-like Structure) ---")
    print(f"Standard Size: {STANDARD_BOARD_ROWS} rows x {STANDARD_BOARD_COLS} columns")

    # Define variations (same as before)
    add_variations = [
        {"name": "add_first_row", "action": lambda g: g.add_row("first"), "desc": "Add row top"},
        {"name": "add_last_row", "action": lambda g: g.add_row("last"), "desc": "Add row bottom"},
        {"name": "add_first_col", "action": lambda g: g.add_column("first"), "desc": "Add col left"},
        {"name": "add_last_col", "action": lambda g: g.add_column("last"), "desc": "Add col right"},
    ]
    remove_variations = [
        {"name": "remove_first_row", "action": lambda g: g.remove_row("first"), "desc": "Remove first row"},
        {"name": "remove_last_row", "action": lambda g: g.remove_row("last"), "desc": "Remove last row"},
        {"name": "remove_first_col", "action": lambda g: g.remove_column("first"), "desc": "Remove first col"},
        {"name": "remove_last_col", "action": lambda g: g.remove_column("last"), "desc": "Remove last col"},
    ]

    # Define prompts (like Chess code)
    ROW_PROMPTS = [
        f"How many rows are there on this puzzle? Answer with a number in curly brackets, e.g., {{9}}.", 
        f"Count the rows on this puzzle. Answer with a number in curly brackets, e.g., {{9}}."
    ]
    COL_PROMPTS = [
        f"How many columns are there on this puzzle? Answer with a number in curly brackets, e.g., {{9}}.", 
        f"Count the columns on this puzzle. Answer with a number in curly brackets, e.g., {{9}}."
    ]
    # Define Standard Check Prompt and Answers for Q3 (like Chess code)
    STD_CHECK_PROMPT_Q3 = f"Is this a {STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS} {BOARD_TYPE_NAME}? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    STD_CHECK_GT_Q3 = "No"  # Variations are never standard
    STD_CHECK_BIAS_Q3 = "Yes" # Biased answer assumes standard

    total_variations = len(add_variations) + len(remove_variations)
    # Only no-title images now (removed 'with_title')
    total_images_expected = total_variations * len(PIXEL_SIZES)
    # Just Q1, Q2, Q3 for notitle (3) per variation/size
    total_metadata_entries_expected = total_variations * len(PIXEL_SIZES) * 3

    print(f"Initial Plan: {total_variations} variations, {len(PIXEL_SIZES)} sizes -> {total_images_expected} images, {total_metadata_entries_expected} metadata entries.")

    progress = tqdm(total=total_images_expected, desc=f"Processing {BOARD_TYPE_NAME}", unit="image", ncols=100)
    generated_metadata_count = 0
    skipped_variants = 0
    skipped_sizes = 0
    images_processed_count = 0

    # Process all variations (add and remove)
    all_variations = []
    for var_idx, variant in enumerate(add_variations):
        variant["variant_set"] = "add"
        all_variations.append((var_idx, variant))
    
    for var_idx, variant in enumerate(remove_variations):
        variant["variant_set"] = "remove"
        all_variations.append((var_idx + len(add_variations), variant))

    for var_idx, variant_info in all_variations:
        var_idx_num = var_idx + 1  # Start from 1 for IDs
        variant = variant_info
        variant_set = variant["variant_set"]
        
        # Create a fresh grid for each variation
        grid = SudokuGrid(STANDARD_BOARD_ROWS, STANDARD_BOARD_COLS)
        action_result = variant["action"](grid)

        if action_result is None:
            print(f"Skipping variant '{variant['name']}' due to action failure.")
            skipped_variants += 1
            progress.update(len(PIXEL_SIZES))
            continue

        # Determine dimension focus
        dimension_focus = action_result.get("dimension", "unknown")
        
        # Set original grid start coordinates based on action result
        orig_grid_start_x_in_new = action_result.get("orig_start_x_in_new", 0)
        orig_grid_start_y_in_new = action_result.get("orig_start_y_in_new", 0)

        if dimension_focus not in ["row", "col"]:
             print(f"Error: Undetermined dimension focus for '{variant['name']}'. Skipping.")
             skipped_variants += 1
             progress.update(len(PIXEL_SIZES))
             continue

        # Choose appropriate prompts based on dimension
        prompts_q1q2 = ROW_PROMPTS if dimension_focus == "row" else COL_PROMPTS
        dimension_ground_truth = str(grid.rows) if dimension_focus == "row" else str(grid.cols)
        dimension_expected_bias = str(STANDARD_BOARD_ROWS) if dimension_focus == "row" else str(STANDARD_BOARD_COLS)

        # Create filename components (like Chess code)
        sanitized_name = grid_utils.sanitize_filename(variant['name'])
        base_id = f"{BOARD_ID}_{var_idx_num:02d}_{dimension_focus}_{sanitized_name}"

        # Create metadata common info (similar to Chess code)
        metadata_common_info = {
            "action": action_result.get("action", "unknown"),
            "dimension_modified": dimension_focus,
            "position": action_result.get("position", "unknown"),
            "original_dimensions": f"{STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS}",
            "new_dimensions": action_result.get("new_dimensions", f"{grid.rows}x{grid.cols}"),
            "visual_cue": "original_structure_bold_lines",
            "show_coordinates": False,
        }
        
        # Add appropriate index details based on action
        if "insert_index" in action_result:
            metadata_common_info["insert_index"] = action_result["insert_index"]
        if "remove_index" in action_result:
            metadata_common_info["remove_index"] = action_result["remove_index"]

        # Generate images and metadata for each pixel size
        for pixel_size in PIXEL_SIZES:
            try:
                metadata_common_info["pixel"] = pixel_size
                no_title_filename = f"{base_id}_notitle_px{pixel_size}.png"
                temp_no_title_path = os.path.join(temp_dir, no_title_filename)
                final_no_title_path = os.path.join(dirs["notitle_img_dir"], no_title_filename)

                # Generate No Title Image
                svg_content = draw_sudoku_grid(
                    grid, size=svg_size,
                    orig_grid_start_x_in_new=orig_grid_start_x_in_new,
                    orig_grid_start_y_in_new=orig_grid_start_y_in_new
                )
                
                if not svg_content:
                    skipped_sizes += 1
                    progress.update(1)
                    continue
                
                scale_ratio = pixel_size / 768.0 if 768.0 > 0 else 1.0
                adjusted_scale = quality_scale * scale_ratio
                svg_success = grid_utils.svg_to_png_direct(svg_content, temp_no_title_path, scale=adjusted_scale, output_size=pixel_size)
                
                if not svg_success or not os.path.exists(temp_no_title_path):
                    skipped_sizes += 1
                    progress.update(1)
                    continue
                
                os.makedirs(os.path.dirname(final_no_title_path), exist_ok=True)
                shutil.copy2(temp_no_title_path, final_no_title_path)
                images_processed_count += 1
                progress.update(1)

                # Add metadata for no-title image (Q1, Q2, Q3)
                img_rel_path_no = os.path.join("images", no_title_filename).replace("\\", "/")
                
                # Q1 and Q2 for no-title
                for i in range(len(prompts_q1q2)):
                    q_type = f"Q{i+1}"
                    prompt_text = prompts_q1q2[i]
                    meta_id = f"{base_id}_notitle_px{pixel_size}_{q_type}"
                    
                    notitle_metadata_rows.append({
                        "ID": meta_id,
                        "image_path": img_rel_path_no,
                        "prompt": prompt_text,
                        "ground_truth": dimension_ground_truth,
                        "expected_bias": dimension_expected_bias,
                        "with_title": False,
                        "type_of_question": q_type,
                        "topic": "Sudoku Grid",
                        "pixel": pixel_size,
                        "metadata": metadata_common_info.copy(),
                    })
                    generated_metadata_count += 1
                
                # Q3 for no-title
                q_type_q3 = "Q3"
                meta_id_q3 = f"{base_id}_notitle_px{pixel_size}_{q_type_q3}"
                
                notitle_metadata_rows.append({
                    "ID": meta_id_q3,
                    "image_path": img_rel_path_no,
                    "prompt": STD_CHECK_PROMPT_Q3,
                    "ground_truth": STD_CHECK_GT_Q3,
                    "expected_bias": STD_CHECK_BIAS_Q3,
                    "with_title": False,
                    "type_of_question": q_type_q3,
                    "topic": "Sudoku Grid",
                    "pixel": pixel_size,
                    "metadata": metadata_common_info.copy(),
                })
                generated_metadata_count += 1

            except Exception as e:
                update_count = max(0, 1 - (progress.n - images_processed_count))
                progress.update(update_count)
                print(f"\nCRITICAL Error processing variant '{variant['name']}' size {pixel_size}: {e}\n")
                skipped_sizes += 1

    progress.close()

    # Write metadata files (only no-title now)
    print("\nWriting metadata files...")
    write_success_no = grid_utils.write_metadata_files(
        notitle_metadata_rows,
        dirs, BOARD_ID, is_with_title=False
    )

    print(f"  Metadata: No title={'Success' if write_success_no else 'Failed'}")
    print(f"  Total metadata entries written: {len(notitle_metadata_rows)}")

    # Final Summary
    print(f"\n--- {BOARD_TYPE_NAME} Dataset Generation Summary ---")
    print(f"Total Image Slots (Initial Plan): {total_images_expected}")
    print(f"Skipped Variants (Action Failure): {skipped_variants}")
    print(f"Skipped Sizes (PNG/Copy Failures): {skipped_sizes}")
    print(f"Successfully Generated & Copied Images: {images_processed_count}")
    print(f"\nGenerated Metadata Entries (Total): {generated_metadata_count}")
    print(f"  - No Title: {len(notitle_metadata_rows)} (Q1, Q2, Q3)")

    # Clean up temp
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
         try: import grid_utils; print("Imported grid_utils.")
         except ImportError: print("ERROR: grid_utils.py not found."); sys.exit(1)

    print(f"=== Running {BOARD_TYPE_NAME} Generator Standalone (No Coordinates, Chess-like Structure) ===")
    create_sudoku_grid_dataset(quality_scale=5.0, svg_size=800)
    print("\nStandalone process completed.")