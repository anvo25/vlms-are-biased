# generators/go_board_generator.py
# -*- coding: utf-8 -*-
"""
Go Board Generator - Generates "notitle" images of Go boards
with variations in rows/columns.
"""
import os
import shutil
import svgwrite # For drawing the Go board
from tqdm import tqdm
import sys

import grid_utils # Common utilities for grid/board generators

# --- Constants for Go Board ---
STANDARD_BOARD_ROWS = 19
STANDARD_BOARD_COLS = 19
BOARD_TYPE_NAME = "Go Board" # For metadata 'topic'
BOARD_ID = "go_board"    # For directory and filename prefixing
PIXEL_SIZES = [384, 768, 1152]

# --- GoBoard Class ---
class GoBoard:
    """Represents a Go board grid, allowing dimension modifications and stone placement."""
    def __init__(self, rows=STANDARD_BOARD_ROWS, cols=STANDARD_BOARD_COLS):
        self.rows = max(1, rows)
        self.cols = max(1, cols)
        self.stones = {} # Stores stone color ('B' or 'W') at square_index (rank * cols + file)
        self._add_common_pattern() # Add a few stones for visual interest

    def _to_square_index(self, file_idx, rank_idx):
        if 0 <= file_idx < self.cols and 0 <= rank_idx < self.rows:
            return rank_idx * self.cols + file_idx
        return None

    def set_stone_at_coords(self, file_idx, rank_idx, stone_color): # stone_color 'B' or 'W'
        sq_idx = self._to_square_index(file_idx, rank_idx)
        if sq_idx is not None:
            self.stones[sq_idx] = stone_color

    def _add_common_pattern(self):
        """Adds a few stones to make the board look more like a game in progress."""
        # Star points are common reference points on Go boards
        if self.rows >= 19 and self.cols >= 19: # For standard 19x19
            star_points_coords = [(3,3), (3,9), (3,15), (9,3), (9,9), (9,15), (15,3), (15,9), (15,15)]
            for i, (f, r) in enumerate(star_points_coords):
                self.set_stone_at_coords(f, r, 'B' if i % 2 == 0 else 'W')
        elif self.rows >= 13 and self.cols >= 13: # For 13x13
            star_points_coords = [(3,3), (3,9), (9,3), (9,9)]
            for i, (f, r) in enumerate(star_points_coords):
                self.set_stone_at_coords(f, r, 'B' if i % 2 == 0 else 'W')
        # Add a few more arbitrary stones for smaller boards if they fit
        if self.rows >= 7 and self.cols >= 7:
            self.set_stone_at_coords(2, 5, 'B'); self.set_stone_at_coords(3, 5, 'W')
            self.set_stone_at_coords(2, 6, 'B'); self.set_stone_at_coords(1, 6, 'W')

    def add_row(self, position="last"):
        insert_idx = 0 if position == "first" else self.rows
        new_stones_map = {}
        for sq_idx, stone in self.stones.items():
            current_rank = sq_idx // self.cols
            current_file = sq_idx % self.cols
            new_rank_val = current_rank + 1 if current_rank >= insert_idx else current_rank
            # New square index is based on the original number of columns
            new_sq_idx = new_rank_val * self.cols + current_file
            new_stones_map[new_sq_idx] = stone
        self.stones = new_stones_map
        self.rows += 1
        return {"action": "add_row", "dimension": "row", "position_added": position, "insert_index": insert_idx}

    def remove_row(self, position="last"):
        if self.rows <= 1: return None
        remove_rank_idx = 0 if position == "first" else self.rows - 1
        new_stones_map = {}
        for sq_idx, stone in self.stones.items():
            current_rank = sq_idx // self.cols
            current_file = sq_idx % self.cols
            if current_rank == remove_rank_idx: continue # Skip stones in the removed row
            new_rank_val = current_rank - 1 if current_rank > remove_rank_idx else current_rank
            # New square index uses original number of columns before row count changes
            new_sq_idx = new_rank_val * self.cols + current_file
            new_stones_map[new_sq_idx] = stone
        self.stones = new_stones_map
        self.rows -= 1
        return {"action": "remove_row", "dimension": "row", "position_removed": position, "remove_index": remove_rank_idx}

    def add_column(self, position="last"):
        insert_idx = 0 if position == "first" else self.cols
        new_col_count = self.cols + 1
        new_stones_map = {}
        for sq_idx, stone in self.stones.items():
            current_rank = sq_idx // self.cols
            current_file = sq_idx % self.cols
            new_file_val = current_file + 1 if current_file >= insert_idx else current_file
            new_sq_idx = current_rank * new_col_count + new_file_val # Use new column count
            new_stones_map[new_sq_idx] = stone
        self.stones = new_stones_map
        self.cols = new_col_count
        return {"action": "add_column", "dimension": "col", "position_added": position, "insert_index": insert_idx}

    def remove_column(self, position="last"):
        if self.cols <= 1: return None
        remove_file_idx = 0 if position == "first" else self.cols - 1
        new_col_count = self.cols - 1
        new_stones_map = {}
        for sq_idx, stone in self.stones.items():
            current_rank = sq_idx // self.cols
            current_file = sq_idx % self.cols
            if current_file == remove_file_idx: continue
            new_file_val = current_file - 1 if current_file > remove_file_idx else current_file
            if new_col_count > 0 : # Ensure valid index for new_col_count
                 new_sq_idx = current_rank * new_col_count + new_file_val
                 new_stones_map[new_sq_idx] = stone
            # If new_col_count is 0, stones can't be placed, effectively clearing them.
            # However, remove_column should not lead to 0 cols if self.cols > 1 initially.
        self.stones = new_stones_map
        self.cols = new_col_count
        return {"action": "remove_column", "dimension": "col", "position_removed": position, "remove_index": remove_file_idx}

# --- Drawing Function for Go Board SVG ---
def draw_go_board_svg(board_obj, render_size=800):
    """Generates an SVG string for a Go board with grid lines and stones."""
    if board_obj.rows <= 0 or board_obj.cols <= 0: return ""

    # Margin around the grid
    margin = render_size * 0.05  # 5% margin
    
    # The Go grid is drawn on intersections, so we need (rows-1) intervals vertically
    # and (cols-1) intervals horizontally.
    # Effective drawing area for the grid itself:
    grid_area_width = render_size - 2 * margin
    grid_area_height = render_size - 2 * margin

    # Size of one "square" or interval in the grid
    # If rows/cols is 1, there are 0 intervals, handle to avoid division by zero.
    interval_width = grid_area_width / (board_obj.cols - 1) if board_obj.cols > 1 else grid_area_width
    interval_height = grid_area_height / (board_obj.rows - 1) if board_obj.rows > 1 else grid_area_height

    total_svg_width = render_size
    total_svg_height = render_size # Assume square SVG output for Go by default

    dwg = svgwrite.Drawing(size=(f"{total_svg_width:.2f}", f"{total_svg_height:.2f}"), profile='tiny')
    
    board_bg_color = "#DCB35C" # Traditional Go board wood color
    grid_line_color = "#000000"
    dwg.add(dwg.rect(insert=(0,0), size=(total_svg_width, total_svg_height), fill=board_bg_color))

    grid_origin_x = margin
    grid_origin_y = margin
    
    # Grid line thickness
    line_thickness = max(1, render_size / 800.0) # Scale thickness with size

    # Draw horizontal lines
    if board_obj.rows > 0:
        for r_idx in range(board_obj.rows):
            y_pos = grid_origin_y + r_idx * interval_height if board_obj.rows > 1 else grid_origin_y + grid_area_height / 2
            # For a single row, line is drawn at its y-coordinate.
            # The actual grid lines span the full width of the grid area.
            dwg.add(dwg.line(start=(grid_origin_x, y_pos), 
                             end=(grid_origin_x + grid_area_width, y_pos), 
                             stroke=grid_line_color, stroke_width=line_thickness))
    # Draw vertical lines
    if board_obj.cols > 0:
        for c_idx in range(board_obj.cols):
            x_pos = grid_origin_x + c_idx * interval_width if board_obj.cols > 1 else grid_origin_x + grid_area_width / 2
            dwg.add(dwg.line(start=(x_pos, grid_origin_y), 
                             end=(x_pos, grid_origin_y + grid_area_height), 
                             stroke=grid_line_color, stroke_width=line_thickness))

    # Draw star points (Hoshi)
    star_point_radius = interval_width / 8.0 # Adjust size relative to interval
    star_point_coords = []
    if board_obj.rows == 19 and board_obj.cols == 19: # Standard 19x19 hoshi
        points = [3, 9, 15]
        star_point_coords = [(r, c) for r in points for c in points]
    elif board_obj.rows == 13 and board_obj.cols == 13: # Common 13x13 hoshi
        points = [3, 6, 9] # Center is (6,6) for 0-indexed
        star_point_coords = [(r,c) for r in points for c in points if (r in [3,9] or c in [3,9] or (r==6 and c==6))]
    elif board_obj.rows == 9 and board_obj.cols == 9: # Common 9x9 hoshi
        points = [2, 4, 6] # Center is (4,4)
        star_point_coords = [(r,c) for r in points for c in points if (r in [2,6] or c in [2,6] or (r==4 and c==4))]
    
    for r_coord, c_coord in star_point_coords:
        # Ensure star point is within current board dimensions before drawing
        if r_coord < board_obj.rows and c_coord < board_obj.cols:
            star_x = grid_origin_x + c_coord * interval_width
            star_y = grid_origin_y + r_coord * interval_height
            dwg.add(dwg.circle(center=(star_x, star_y), r=star_point_radius, fill=grid_line_color))

    # Draw stones
    stone_radius_ratio = 0.47 # Slightly less than half the interval for spacing
    actual_stone_radius = min(interval_width, interval_height) * stone_radius_ratio
    
    for sq_idx, stone_color_char in board_obj.stones.items():
        # Ensure stone is within the current board dimensions if board was shrunk
        if sq_idx >= board_obj.rows * board_obj.cols: continue

        rank_idx = sq_idx // board_obj.cols
        file_idx = sq_idx % board_obj.cols
        
        # Check bounds again, especially if cols changed after stone was placed
        if rank_idx >= board_obj.rows or file_idx >= board_obj.cols: continue

        stone_center_x = grid_origin_x + file_idx * interval_width
        stone_center_y = grid_origin_y + rank_idx * interval_height
        
        fill_c = 'black' if stone_color_char == 'B' else 'white'
        stroke_c = 'none' if stone_color_char == 'B' else 'black' # Black stones often have no stroke
        stroke_w = 0 if stone_color_char == 'B' else line_thickness * 0.75
        
        dwg.add(dwg.circle(center=(stone_center_x, stone_center_y), r=actual_stone_radius,
                           fill=fill_c, stroke=stroke_c, stroke_width=stroke_w))

    svg_content_str = dwg.tostring()
    if not svg_content_str.startswith('<?xml'):
        svg_content_str = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_content_str
    return svg_content_str

# --- Main Go Board Dataset Generation Function ---
def create_go_board_dataset(quality_scale=5.0, svg_size=800):
    """Generates the "notitle" Go Board dataset with 4 core variations."""
    output_dirs = grid_utils.create_directory_structure(BOARD_ID)
    temp_dir_path = output_dirs["temp_dir"]
    all_notitle_metadata = []

    print(f"  Starting {BOARD_TYPE_NAME} 'notitle' dataset generation...")
    # Simplified variations for Go board (add/remove one row/col from end)
    # Since Go boards are symmetric, add/remove first is visually similar to add/remove last
    # after rotation. We'll stick to 'last' for simplicity of implementation.
    variations_specs = [
        {"name": "remove_row_last", "action_func": lambda b: b.remove_row("last")},
        {"name": "add_row_last",    "action_func": lambda b: b.add_row("last")},
        {"name": "remove_col_last", "action_func": lambda b: b.remove_column("last")},
        {"name": "add_col_last",     "action_func": lambda b: b.add_column("last")},
    ]
    # Prompts
    row_prompts_q1q2 = [
        "How many horizontal lines are there on this board? Answer with a number in curly brackets, e.g., {9}.",
        "Count the horizontal lines on this board. Answer with a number in curly brackets, e.g., {9}."
    ]
    col_prompts_q1q2 = [
        "How many vertical lines are there on this board? Answer with a number in curly brackets, e.g., {9}.",
        "Count the vertical lines on this board. Answer with a number in curly brackets, e.g., {9}."
    ] # Note: For Go, "lines" is more accurate than "rows/columns" for user understanding.
      # However, the ground truth will be board_obj.rows and board_obj.cols.
    
    std_check_prompt_q3 = f"Is this a {STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS} {BOARD_TYPE_NAME}? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    std_check_gt_q3, std_check_bias_q3 = "No", "Yes"

    num_variations = len(variations_specs)
    total_images_to_generate = num_variations * len(PIXEL_SIZES)
    progress_bar = tqdm(total=total_images_to_generate, desc=f"Generating {BOARD_TYPE_NAME}", unit="image", ncols=100)

    # (Loop and metadata generation logic identical to chess_board_generator,
    #  just replace ChessBoard with GoBoard and draw_chess_board_svg with draw_go_board_svg)

    for var_idx, var_spec in enumerate(variations_specs):
        current_board = GoBoard(STANDARD_BOARD_ROWS, STANDARD_BOARD_COLS) # Start fresh
        action_result_details = var_spec["action_func"](current_board)
        if action_result_details is None:
            progress_bar.update(len(PIXEL_SIZES)); continue

        dimension_focus = action_result_details.get("dimension")
        prompts_for_q1q2, gt_q1q2, bias_q1q2 = [], "", ""
        if dimension_focus == "row":
            prompts_for_q1q2, gt_q1q2, bias_q1q2 = row_prompts_q1q2, str(current_board.rows), str(STANDARD_BOARD_ROWS)
        elif dimension_focus == "col":
            prompts_for_q1q2, gt_q1q2, bias_q1q2 = col_prompts_q1q2, str(current_board.cols), str(STANDARD_BOARD_COLS)
        else:
            progress_bar.update(len(PIXEL_SIZES)); continue
            
        sanitized_var_name = grid_utils.sanitize_filename(var_spec['name'])
        base_id_prefix = f"{BOARD_ID}_{var_idx+1:02d}_{dimension_focus}_{sanitized_var_name}"

        for px_size in PIXEL_SIZES:
            img_basename = f"{base_id_prefix}_px{px_size}.png" # No "notitle" in filename itself
            temp_png_path = os.path.join(temp_dir_path, img_basename)
            final_png_path = os.path.join(output_dirs["notitle_img_dir"], img_basename)

            board_svg_content = draw_go_board_svg(current_board, render_size=svg_size)
            if not board_svg_content: progress_bar.update(1); continue
            
            current_scale = quality_scale * (px_size / float(svg_size if svg_size > 0 else px_size))
            if not grid_utils.svg_to_png_direct(board_svg_content, temp_png_path, scale=current_scale, output_size=px_size):
                progress_bar.update(1); continue
            
            try: shutil.copy2(temp_png_path, final_png_path)
            except Exception: progress_bar.update(1); continue
            progress_bar.update(1)

            image_path_relative = os.path.join("images", img_basename).replace("\\", "/")
            common_meta = {
                "action_type": action_result_details.get("action"), "dimension_modified": dimension_focus,
                "original_dimensions": f"{STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS}",
                "new_dimensions": f"{current_board.rows}x{current_board.cols}",
                "pixel_size": px_size, "variation_name": var_spec['name'],
            }
            if "insert_index" in action_result_details: common_meta["insert_index"] = action_result_details["insert_index"]
            if "remove_index" in action_result_details: common_meta["remove_index"] = action_result_details["remove_index"]

            for q_idx, prompt_text in enumerate(prompts_for_q1q2):
                q_label = f"Q{q_idx+1}"; meta_id = f"{base_id_prefix}_px{px_size}_{q_label}"
                all_notitle_metadata.append({"ID": meta_id, "image_path": image_path_relative, "topic": BOARD_TYPE_NAME,
                                             "prompt": prompt_text, "ground_truth": gt_q1q2, "expected_bias": bias_q1q2,
                                             "with_title": False, "type_of_question": q_label, "pixel": px_size,
                                             "metadata": common_meta.copy()})
            meta_id_q3 = f"{base_id_prefix}_px{px_size}_Q3"
            all_notitle_metadata.append({"ID": meta_id_q3, "image_path": image_path_relative, "topic": BOARD_TYPE_NAME,
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
    create_go_board_dataset(quality_scale=5.0, svg_size=800)
    print(f"\nDirect test of {BOARD_TYPE_NAME} Generator complete.")