# generators/sudoku_board_generator.py
# -*- coding: utf-8 -*-
"""
Sudoku Board Generator - Generates "notitle" images of Sudoku boards
with variations, highlighting 3x3 block structure.
"""
import os
import shutil
import svgwrite # For drawing the Sudoku board
from tqdm import tqdm
import sys

import grid_utils # Common utilities for grid/board generators

# --- Constants for Sudoku Board ---
STANDARD_BOARD_ROWS = 9
STANDARD_BOARD_COLS = 9
BOARD_TYPE_NAME = "Sudoku Board" 
BOARD_ID = "sudoku_board"    
PIXEL_SIZES = [384, 768, 1152]

# --- SudokuGrid Class (from original sudoku_grid_generator.py) ---
class SudokuGrid:
    def __init__(self, rows=STANDARD_BOARD_ROWS, cols=STANDARD_BOARD_COLS):
        self.rows = max(1, rows)
        self.cols = max(1, cols)
        self._original_rows = STANDARD_BOARD_ROWS 
        self._original_cols = STANDARD_BOARD_COLS
        self.numbers = {} 
        self._add_example_numbers()

    def _add_example_numbers(self):
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
            for (c_idx, r_idx), number in example_pattern.items():
                 if c_idx < self.cols and r_idx < self.rows:
                    self.set_number(c_idx, r_idx, number)
        elif self.rows >= 4 and self.cols >= 4:
             if 0 < self.cols and 0 < self.rows: self.set_number(0, 0, 1)
             if 1 < self.cols and 1 < self.rows: self.set_number(1, 1, 2)
             if 2 < self.cols and 2 < self.rows: self.set_number(2, 2, 3)
             if 3 < self.cols and 3 < self.rows: self.set_number(3, 3, 4)

    def set_number(self, col_idx, row_idx, number_val):
        if 0 <= col_idx < self.cols and 0 <= row_idx < self.rows and \
           isinstance(number_val, int) and 1 <= number_val <= 9:
            square_idx = row_idx * self.cols + col_idx
            self.numbers[square_idx] = number_val

    def add_row(self, position="last"):
        insert_idx = 0 if position == "first" else self.rows
        orig_content_start_y_in_new = 1 if position == "first" else 0
        
        new_numbers_map = {}
        for sq_idx, num_val in self.numbers.items():
            r_old = sq_idx // self.cols
            c_old = sq_idx % self.cols
            r_new = r_old + 1 if r_old >= insert_idx else r_old
            new_sq_idx = r_new * self.cols + c_old 
            new_numbers_map[new_sq_idx] = num_val
        self.numbers = new_numbers_map
        self.rows += 1
        return {"action": "add_row", "dimension": "row", "position_added": position, 
                "insert_index": insert_idx, "orig_start_y_in_new": orig_content_start_y_in_new}

    def remove_row(self, position="last"):
        if self.rows <= 1: return None
        remove_idx = 0 if position == "first" else self.rows - 1
        # Determine where original (0,0)'s content would shift *to* if first row is removed
        # If first row (idx 0) is removed, original content at (any_col, 1) becomes (any_col, 0).
        # So, orig_start_y_in_new indicates the new y-coordinate of what was originally at y=0 (if not removed)
        # or y=1 (if y=0 was removed).
        orig_content_start_y_in_new = 0 # Default: no shift if removing last or if more than 1 row and removing first
        if position == "first" and self.rows > 1: # If removing the top row and there are other rows
             pass # Original content effectively shifts "up", so (0,0) of original is gone.
                  # The content from original (0,1) is now at (0,0) in new grid.
                  # So, the "start" of the original content (that remains) is still at y=0 of the new grid.

        new_numbers_map = {}
        for sq_idx, num_val in self.numbers.items():
            r_old = sq_idx // self.cols
            c_old = sq_idx % self.cols
            if r_old == remove_idx: continue
            r_new = r_old - 1 if r_old > remove_idx else r_old
            new_sq_idx = r_new * self.cols + c_old
            new_numbers_map[new_sq_idx] = num_val
        self.numbers = new_numbers_map
        self.rows -= 1
        return {"action": "remove_row", "dimension": "row", "position_removed": position, 
                "remove_index": remove_idx, "orig_start_y_in_new": orig_content_start_y_in_new}


    def add_column(self, position="last"):
        insert_idx = 0 if position == "first" else self.cols
        orig_content_start_x_in_new = 1 if position == "first" else 0
        new_col_count = self.cols + 1
        
        new_numbers_map = {}
        for sq_idx, num_val in self.numbers.items():
            r_old = sq_idx // self.cols
            c_old = sq_idx % self.cols
            c_new = c_old + 1 if c_old >= insert_idx else c_old
            new_sq_idx = r_old * new_col_count + c_new 
            new_numbers_map[new_sq_idx] = num_val
        self.numbers = new_numbers_map
        self.cols = new_col_count
        return {"action": "add_column", "dimension": "col", "position_added": position, 
                "insert_index": insert_idx, "orig_start_x_in_new": orig_content_start_x_in_new}

    def remove_column(self, position="last"):
        if self.cols <= 1: return None
        remove_idx = 0 if position == "first" else self.cols - 1
        orig_content_start_x_in_new = 0 # Similar logic to remove_row for y-shift
        
        new_col_count = self.cols - 1
        new_numbers_map = {}
        for sq_idx, num_val in self.numbers.items():
            r_old = sq_idx // self.cols
            c_old = sq_idx % self.cols
            if c_old == remove_idx: continue
            c_new = c_old - 1 if c_old > remove_idx else c_old
            if new_col_count > 0:
                 new_sq_idx = r_old * new_col_count + c_new
                 new_numbers_map[new_sq_idx] = num_val
        self.numbers = new_numbers_map
        self.cols = new_col_count
        return {"action": "remove_column", "dimension": "col", "position_removed": position, 
                "remove_index": remove_idx, "orig_start_x_in_new": orig_content_start_x_in_new}

# --- Drawing Function for Sudoku Board SVG ---
def draw_sudoku_board_svg(grid_obj, render_size=800,
                          orig_grid_start_col_in_new=0, orig_grid_start_row_in_new=0):
    if grid_obj.rows <= 0 or grid_obj.cols <= 0: return ""

    margin_ratio = 0.05
    effective_render_size = render_size * (1 - 2 * margin_ratio)
    
    cell_width = effective_render_size / grid_obj.cols if grid_obj.cols > 0 else effective_render_size
    cell_height = effective_render_size / grid_obj.rows if grid_obj.rows > 0 else effective_render_size
    actual_cell_size = min(cell_width, cell_height) 

    grid_pixel_width = grid_obj.cols * actual_cell_size
    grid_pixel_height = grid_obj.rows * actual_cell_size
    
    margin_pixels = render_size * margin_ratio
    total_svg_width = grid_pixel_width + 2 * margin_pixels
    total_svg_height = grid_pixel_height + 2 * margin_pixels

    # CORRECTED: Use profile='full' to support 'dominant-baseline'
    dwg = svgwrite.Drawing(size=(f"{total_svg_width:.2f}", f"{total_svg_height:.2f}"), profile='full')
    dwg.add(dwg.rect((0,0), (total_svg_width, total_svg_height), fill="#FFFFFF")) 

    grid_origin_x = margin_pixels
    grid_origin_y = margin_pixels
    
    line_color = "#000000"
    thickness_standard = max(0.5, 0.5 * (render_size / 800.0)) 
    thickness_3x3_boundary = max(1.5, 2.0 * (render_size / 800.0))
    thickness_outer_border = max(1.5, 2.0 * (render_size / 800.0))

    for r_idx in range(grid_obj.rows + 1):
        y_pos = grid_origin_y + r_idx * actual_cell_size
        current_thickness = thickness_standard
        is_current_grid_outer = (r_idx == 0 or r_idx == grid_obj.rows)
        r_idx_relative_to_original_start = r_idx - orig_grid_start_row_in_new
        is_original_3x3_boundary = (
            (0 <= r_idx_relative_to_original_start <= grid_obj._original_rows) and \
            (r_idx_relative_to_original_start % 3 == 0)
        )
        if is_current_grid_outer: current_thickness = thickness_outer_border
        elif is_original_3x3_boundary: current_thickness = thickness_3x3_boundary
        dwg.add(dwg.line((grid_origin_x, y_pos), (grid_origin_x + grid_pixel_width, y_pos), 
                         stroke=line_color, stroke_width=current_thickness))

    for c_idx in range(grid_obj.cols + 1):
        x_pos = grid_origin_x + c_idx * actual_cell_size
        current_thickness = thickness_standard
        is_current_grid_outer = (c_idx == 0 or c_idx == grid_obj.cols)
        c_idx_relative_to_original_start = c_idx - orig_grid_start_col_in_new
        is_original_3x3_boundary = (
            (0 <= c_idx_relative_to_original_start <= grid_obj._original_cols) and \
            (c_idx_relative_to_original_start % 3 == 0)
        )
        if is_current_grid_outer: current_thickness = thickness_outer_border
        elif is_original_3x3_boundary: current_thickness = thickness_3x3_boundary
        dwg.add(dwg.line((x_pos, grid_origin_y), (x_pos, grid_origin_y + grid_pixel_height), 
                         stroke=line_color, stroke_width=current_thickness))

    number_font_size = actual_cell_size * 0.6
    font_family = "Arial, sans-serif"
    # dominant-baseline="central" works well with text_anchor="middle" for true centering
    # The text_y_offset might not be needed if dominant-baseline="central" is effective.
    # Test to see. If text still looks off, text_y_offset = actual_cell_size * 0.05 can be re-added.
    text_y_offset = 0 # Initially try without manual offset when using dominant-baseline

    for sq_idx, number_val in grid_obj.numbers.items():
        if sq_idx >= grid_obj.rows * grid_obj.cols: continue 
        r_val = sq_idx // grid_obj.cols
        c_val = sq_idx % grid_obj.cols
        if 0 <= c_val < grid_obj.cols and 0 <= r_val < grid_obj.rows:
            center_x = grid_origin_x + (c_val + 0.5) * actual_cell_size
            center_y = grid_origin_y + (r_val + 0.5) * actual_cell_size
            dwg.add(dwg.text(str(number_val), insert=(center_x, center_y + text_y_offset),
                             fill="#000000", font_size=f"{number_font_size:.2f}px",
                             font_family=font_family, font_weight="bold",
                             text_anchor="middle", dominant_baseline="central")) # Added dominant-baseline
                             
    svg_str = dwg.tostring()
    return '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_str if not svg_str.startswith('<?xml') else svg_str

# --- Main Sudoku Board Dataset Generation Function ---
def create_sudoku_board_dataset(quality_scale=5.0, svg_size=800):
    output_dirs = grid_utils.create_directory_structure(BOARD_ID)
    temp_dir_path = output_dirs["temp_dir"]
    all_notitle_metadata = []

    print(f"  Starting {BOARD_TYPE_NAME} 'notitle' dataset generation...")
    variations_specs = [
        {"name": "add_first_row", "action_func": lambda g: g.add_row("first")},
        {"name": "add_last_row", "action_func": lambda g: g.add_row("last")},
        {"name": "add_first_col", "action_func": lambda g: g.add_column("first")},
        {"name": "add_last_col", "action_func": lambda g: g.add_column("last")},
        {"name": "remove_first_row", "action_func": lambda g: g.remove_row("first")},
        {"name": "remove_last_row", "action_func": lambda g: g.remove_row("last")},
        {"name": "remove_first_col", "action_func": lambda g: g.remove_column("first")},
        {"name": "remove_last_col", "action_func": lambda g: g.remove_column("last")},
    ]
    row_prompts_q1q2 = [
        f"How many rows are there on this puzzle? Answer with a number in curly brackets, e.g., {{{STANDARD_BOARD_ROWS}}}.",
        f"Count the rows on this puzzle. Answer with a number in curly brackets, e.g., {{{STANDARD_BOARD_ROWS}}}."
    ]
    col_prompts_q1q2 = [
        f"How many columns are there on this puzzle? Answer with a number in curly brackets, e.g., {{{STANDARD_BOARD_COLS}}}.",
        f"Count the columns on this puzzle. Answer with a number in curly brackets, e.g., {{{STANDARD_BOARD_COLS}}}."
    ]
    std_check_prompt_q3 = f"Is this a {STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS} Sudoku puzzle? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    std_check_gt_q3, std_check_bias_q3 = "No", "Yes"

    num_variations = len(variations_specs)
    total_images_to_generate = num_variations * len(PIXEL_SIZES)
    progress_bar = tqdm(total=total_images_to_generate, desc=f"Generating {BOARD_TYPE_NAME}", unit="image", ncols=100)

    for var_idx, var_spec in enumerate(variations_specs):
        current_grid = SudokuGrid(STANDARD_BOARD_ROWS, STANDARD_BOARD_COLS)
        action_result_details = var_spec["action_func"](current_grid)
        if action_result_details is None: progress_bar.update(len(PIXEL_SIZES)); continue

        dim_focus = action_result_details.get("dimension")
        orig_start_col = action_result_details.get("orig_start_x_in_new", 0)
        orig_start_row = action_result_details.get("orig_start_y_in_new", 0)

        prompts_q1q2, gt_q1q2, bias_q1q2 = [], "", ""
        if dim_focus == "row": prompts_q1q2, gt_q1q2, bias_q1q2 = row_prompts_q1q2, str(current_grid.rows), str(STANDARD_BOARD_ROWS)
        elif dim_focus == "col": prompts_q1q2, gt_q1q2, bias_q1q2 = col_prompts_q1q2, str(current_grid.cols), str(STANDARD_BOARD_COLS)
        else: progress_bar.update(len(PIXEL_SIZES)); continue
            
        sanitized_var_name = grid_utils.sanitize_filename(var_spec['name'])
        base_id_prefix = f"{BOARD_ID}_{var_idx+1:02d}_{dim_focus}_{sanitized_var_name}"

        for px_size in PIXEL_SIZES:
            img_basename = f"{base_id_prefix}_px{px_size}.png"
            temp_png_path = os.path.join(temp_dir_path, img_basename)
            final_png_path = os.path.join(output_dirs["notitle_img_dir"], img_basename)

            board_svg_content = draw_sudoku_board_svg(current_grid, render_size=svg_size,
                                                     orig_grid_start_col_in_new=orig_start_col,
                                                     orig_grid_start_row_in_new=orig_start_row)
            if not board_svg_content: progress_bar.update(1); continue
            
            current_scale = quality_scale * (px_size / float(svg_size if svg_size > 0 else px_size))
            if not grid_utils.svg_to_png_direct(board_svg_content, temp_png_path, scale=current_scale, output_size=px_size):
                progress_bar.update(1); continue
            
            try: shutil.copy2(temp_png_path, final_png_path)
            except Exception: progress_bar.update(1); continue
            progress_bar.update(1)

            img_path_rel = os.path.join("images", img_basename).replace("\\", "/")
            common_meta = {"action_type": action_result_details.get("action"), "dimension_modified": dim_focus,
                           "original_dimensions": f"{STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS}",
                           "new_dimensions": f"{current_grid.rows}x{current_grid.cols}",
                           "pixel_size": px_size, "variation_name": var_spec['name'],
                           "visual_cue": "original_3x3_structure_bold_lines",
                           "orig_content_start_col": orig_start_col, 
                           "orig_content_start_row": orig_start_row
                          }
            if "insert_index" in action_result_details: common_meta["insert_index"] = action_result_details["insert_index"]
            if "remove_index" in action_result_details: common_meta["remove_index"] = action_result_details["remove_index"]

            for q_idx, prompt in enumerate(prompts_q1q2):
                q_label = f"Q{q_idx+1}"; meta_id = f"{base_id_prefix}_px{px_size}_{q_label}"
                all_notitle_metadata.append({"ID": meta_id, "image_path": img_path_rel, "topic": BOARD_TYPE_NAME,
                                             "prompt": prompt, "ground_truth": gt_q1q2, "expected_bias": bias_q1q2,
                                             "with_title": False, "type_of_question": q_label, "pixel": px_size,
                                             "metadata": common_meta.copy()})
            meta_id_q3 = f"{base_id_prefix}_px{px_size}_Q3"
            all_notitle_metadata.append({"ID": meta_id_q3, "image_path": img_path_rel, "topic": BOARD_TYPE_NAME,
                                         "prompt": std_check_prompt_q3, "ground_truth": std_check_gt_q3, "expected_bias": std_check_bias_q3,
                                         "with_title": False, "type_of_question": "Q3", "pixel": px_size,
                                         "metadata": common_meta.copy()})
    progress_bar.close()

    print(f"\n  Saving 'notitle' metadata for {BOARD_TYPE_NAME}...")
    grid_utils.write_metadata_files(all_notitle_metadata, output_dirs, BOARD_ID)
    
    final_img_count = 0
    try:
        if os.path.exists(output_dirs["notitle_img_dir"]):
            final_img_count = len([f for f in os.listdir(output_dirs["notitle_img_dir"]) if f.endswith('.png')])
    except Exception: pass
    print(f"\n  --- {BOARD_TYPE_NAME} 'notitle' Generation Summary ---")
    print(f"  Actual 'notitle' images: {final_img_count} (Expected: {total_images_to_generate})")
    print(f"  Total 'notitle' metadata entries: {len(all_notitle_metadata)}")
    try:
        if os.path.exists(temp_dir_path): shutil.rmtree(temp_dir_path)
        print(f"  Cleaned temp directory: {temp_dir_path}")
    except Exception as e: print(f"  Warning: Failed temp cleanup {temp_dir_path}: {e}")
    print(f"  {BOARD_TYPE_NAME} 'notitle' dataset generation finished.")

if __name__ == '__main__':
    print(f"Testing {BOARD_TYPE_NAME} Generator directly...")
    if not os.path.exists("vlms-are-biased-notitle"): os.makedirs("vlms-are-biased-notitle")
    create_sudoku_board_dataset(quality_scale=5.0, svg_size=800)
    print(f"\nDirect test of {BOARD_TYPE_NAME} Generator complete.")