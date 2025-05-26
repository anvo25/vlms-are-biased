# generators/chess_board_generator.py
# -*- coding: utf-8 -*-
"""
Chess Board Generator - Generates "notitle" images of Chess boards
with variations in rows/columns.
"""
import os
import shutil
import svgwrite # For drawing the board
from tqdm import tqdm
import sys

# Use grid_utils for common board generation functionalities
import grid_utils 

# --- Constants for Chess Board ---
STANDARD_BOARD_ROWS = 8
STANDARD_BOARD_COLS = 8
BOARD_TYPE_NAME = "Chess Board" # For metadata 'topic'
BOARD_ID = "chess_board" # For directory and filename prefixing
PIXEL_SIZES = [384, 768, 1152] # Standard resolutions

# --- ChessBoard Class (Manages Dimensions and optional piece placement) ---
class ChessBoard:
    """Represents a Chess board grid, allowing dimension modifications."""
    def __init__(self, rows=STANDARD_BOARD_ROWS, cols=STANDARD_BOARD_COLS):
        self.rows = max(1, rows)
        self.cols = max(1, cols)
        # `pieces` could be used if we were drawing pieces, but for grid-only, it's not essential.
        # self.pieces = {} # Example: {(file, rank): 'P'}
        # For now, this class primarily manages dimensions.

    def add_row(self, position="last"):
        """Adds a row to the board. 'position' can be 'first' or 'last'."""
        insert_idx = 0 if position == "first" else self.rows # Insert at top or bottom
        # If pieces were managed, their ranks would need adjustment here.
        self.rows += 1
        return {"action": "add_row", "dimension": "row", "position_added": position, "insert_index": insert_idx}

    def remove_row(self, position="last"):
        """Removes a row from the board. 'position' can be 'first' or 'last'."""
        if self.rows <= 1: return None # Cannot remove if only one row or less
        remove_idx = 0 if position == "first" else self.rows - 1
        # If pieces were managed, pieces on this row would be removed, others adjusted.
        self.rows -= 1
        return {"action": "remove_row", "dimension": "row", "position_removed": position, "remove_index": remove_idx}

    def add_column(self, position="last"):
        """Adds a column. 'position' can be 'first' (left) or 'last' (right)."""
        insert_idx = 0 if position == "first" else self.cols
        # If pieces were managed, their files would need adjustment.
        self.cols += 1
        return {"action": "add_column", "dimension": "col", "position_added": position, "insert_index": insert_idx}

    def remove_column(self, position="last"):
        """Removes a column. 'position' can be 'first' or 'last'."""
        if self.cols <= 1: return None
        remove_idx = 0 if position == "first" else self.cols - 1
        # If pieces were managed, pieces on this col would be removed, others adjusted.
        self.cols -= 1
        return {"action": "remove_column", "dimension": "col", "position_removed": position, "remove_index": remove_idx}

# --- Drawing Function for Chess Board SVG ---
def draw_chess_board_svg(board_obj, render_size=800, show_coords=False):
    """
    Generates an SVG string for a chess board with alternating square colors.
    Does not draw pieces or algebraic coordinates by default.
    """
    if board_obj.rows <= 0 or board_obj.cols <= 0: return ""

    # Determine square size based on the larger dimension to fit within render_size
    # Add a small margin around the board.
    margin_ratio = 0.05 # 5% margin
    effective_render_size = render_size * (1 - 2 * margin_ratio)
    
    square_width = effective_render_size / board_obj.cols
    square_height = effective_render_size / board_obj.rows
    # For a proportional board, usually square_width == square_height.
    # If we want strictly square cells, take the minimum.
    actual_square_size = min(square_width, square_height)

    board_pixel_width = board_obj.cols * actual_square_size
    board_pixel_height = board_obj.rows * actual_square_size
    
    margin_pixels = render_size * margin_ratio
    total_svg_width = board_pixel_width + 2 * margin_pixels
    total_svg_height = board_pixel_height + 2 * margin_pixels

    dwg = svgwrite.Drawing(size=(f"{total_svg_width:.2f}", f"{total_svg_height:.2f}"), profile='tiny')
    # Optional: background for the whole SVG (e.g., if margins are transparent)
    # dwg.add(dwg.rect(insert=(0,0), size=(total_svg_width, total_svg_height), fill="#cccccc")) # Light grey bg

    # Colors for the squares
    color_light = "#f0d9b5" # Typical light square color
    color_dark = "#b58863"  # Typical dark square color

    grid_origin_x = margin_pixels
    grid_origin_y = margin_pixels

    for r_idx in range(board_obj.rows):
        for c_idx in range(board_obj.cols):
            square_x = grid_origin_x + c_idx * actual_square_size
            square_y = grid_origin_y + r_idx * actual_square_size
            
            # Determine square color based on position (alternating pattern)
            current_color = color_light if (r_idx + c_idx) % 2 == 0 else color_dark
            
            dwg.add(dwg.rect(insert=(square_x, square_y), size=(actual_square_size, actual_square_size),
                             fill=current_color, stroke=grid_utils.sanitize_filename("none"))) # No stroke for individual squares usually

    # Optional: Add an outer border for the board area
    dwg.add(dwg.rect(insert=(grid_origin_x, grid_origin_y), size=(board_pixel_width, board_pixel_height),
                     fill="none", stroke="#333333", stroke_width=max(1, render_size/400.0)))
    
    # show_coords logic would go here if needed, but paper says no coordinates for board tasks.

    svg_content_str = dwg.tostring()
    if not svg_content_str.startswith('<?xml'):
        svg_content_str = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_content_str
    return svg_content_str

# --- Main Dataset Generation Function ---
def create_chess_board_dataset(quality_scale=5.0, svg_size=800):
    """
    Generates the "notitle" Chess Board dataset with variations.
    """
    # Get the directory structure for "notitle" outputs for this board type
    # `grid_utils.create_directory_structure` expects the sanitized board ID.
    output_dirs = grid_utils.create_directory_structure(BOARD_ID)
    temp_dir_path = output_dirs["temp_dir"]

    all_notitle_metadata = []

    print(f"  Starting {BOARD_TYPE_NAME} 'notitle' dataset generation...")
    print(f"    Standard Size: {STANDARD_BOARD_ROWS} rows x {STANDARD_BOARD_COLS} columns")

    # Define variations for chess board (add/remove row/col from first/last)
    variations_specs = [
        {"name": "remove_first_row", "action_func": lambda b: b.remove_row("first")},
        {"name": "remove_last_row",  "action_func": lambda b: b.remove_row("last")},
        {"name": "add_first_row",    "action_func": lambda b: b.add_row("first")},
        {"name": "add_last_row",     "action_func": lambda b: b.add_row("last")},
        {"name": "remove_first_col", "action_func": lambda b: b.remove_column("first")},
        {"name": "remove_last_col",  "action_func": lambda b: b.remove_column("last")},
        {"name": "add_first_col",    "action_func": lambda b: b.add_column("first")},
        {"name": "add_last_col",     "action_func": lambda b: b.add_column("last")},
    ]

    # Prompts for row/column counting and standard check
    row_prompts_q1q2 = [
        "How many rows are there on this board? Answer with a number in curly brackets, e.g., {9}.",
        "Count the rows on this board. Answer with a number in curly brackets, e.g., {9}."
    ]
    col_prompts_q1q2 = [
        "How many columns are there on this board? Answer with a number in curly brackets, e.g., {{9}.",
        "Count the columns on this board. Answer with a number in curly brackets, e.g., {{9}}."
    ]
    std_check_prompt_q3 = f"Is this a {STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS} {BOARD_TYPE_NAME}? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    std_check_gt_q3 = "No"  # Since all are variations from standard
    std_check_bias_q3 = "Yes" # Bias is to assume standard

    num_variations = len(variations_specs)
    total_images_to_generate = num_variations * len(PIXEL_SIZES)
    
    progress_bar = tqdm(total=total_images_to_generate, desc=f"Generating {BOARD_TYPE_NAME}", unit="image", ncols=100)

    for var_idx, var_spec in enumerate(variations_specs):
        # Create a fresh board for each variation from standard dimensions
        current_board = ChessBoard(STANDARD_BOARD_ROWS, STANDARD_BOARD_COLS)
        action_result_details = var_spec["action_func"](current_board)

        if action_result_details is None: # Action might fail (e.g., removing from 1x1 board)
            print(f"  Skipping variation '{var_spec['name']}' as action failed.")
            progress_bar.update(len(PIXEL_SIZES)) # Skip all resolutions for this failed var
            continue

        dimension_focus = action_result_details.get("dimension", "unknown") # "row" or "col"
        prompts_for_q1q2 = []
        ground_truth_for_q1q2 = ""
        expected_bias_for_q1q2 = ""

        if dimension_focus == "row":
            prompts_for_q1q2 = row_prompts_q1q2
            ground_truth_for_q1q2 = str(current_board.rows)
            expected_bias_for_q1q2 = str(STANDARD_BOARD_ROWS)
        elif dimension_focus == "col":
            prompts_for_q1q2 = col_prompts_q1q2
            ground_truth_for_q1q2 = str(current_board.cols)
            expected_bias_for_q1q2 = str(STANDARD_BOARD_COLS)
        else:
            print(f"  Error: Unknown dimension focus '{dimension_focus}' for variation '{var_spec['name']}'. Skipping.")
            progress_bar.update(len(PIXEL_SIZES))
            continue
        
        sanitized_variation_name = grid_utils.sanitize_filename(var_spec['name'])
        # Base ID for filenames and metadata entries
        # Format: <board_id>_<var_idx>_<dim_focus>_<var_name_sanitized>
        base_id_prefix = f"{BOARD_ID}_{var_idx+1:02d}_{dimension_focus}_{sanitized_variation_name}"

        for px_size in PIXEL_SIZES:
            # Filename for the "notitle" image (no "notitle" in name, as it's implied by output dir)
            # Format: <base_id_prefix>_px<size>.png
            img_basename = f"{base_id_prefix}_px{px_size}.png"
            temp_png_path = os.path.join(temp_dir_path, img_basename)
            final_png_path = os.path.join(output_dirs["notitle_img_dir"], img_basename)

            board_svg_content = draw_chess_board_svg(current_board, render_size=svg_size)
            if not board_svg_content:
                print(f"  ERROR: SVG generation failed for {img_basename}. Skipping size {px_size}.")
                progress_bar.update(1); continue

            # Scale for SVG to PNG conversion
            # Assuming svg_size (render_size for SVG) is the reference for px_size output.
            # If px_size is different from svg_size, scaling might be needed.
            # For simplicity, using a fixed quality scale adjusted by pixel ratio if desired.
            current_scale = quality_scale * (px_size / float(svg_size if svg_size > 0 else px_size) )
            
            if not grid_utils.svg_to_png_direct(board_svg_content, temp_png_path, 
                                                scale=current_scale, output_size=px_size):
                print(f"  ERROR: PNG conversion failed for {temp_png_path}. Skipping.")
                progress_bar.update(1); continue
            
            # Copy from temp to final "notitle" image directory
            try:
                shutil.copy2(temp_png_path, final_png_path)
            except Exception as e:
                print(f"  ERROR: Failed to copy {temp_png_path} to {final_png_path}: {e}")
                progress_bar.update(1); continue
            
            progress_bar.update(1) # Successful image generation and copy

            # --- Prepare Metadata ---
            # Relative path for the image, to be stored in metadata
            image_path_relative = os.path.join("images", img_basename).replace("\\", "/")
            
            # Common metadata payload for this variation and resolution
            common_metadata_payload = {
                "action_type": action_result_details.get("action", "unknown"),
                "dimension_modified": dimension_focus,
                "original_dimensions": f"{STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS}",
                "new_dimensions": f"{current_board.rows}x{current_board.cols}",
                "pixel_size": px_size,
                "variation_name": var_spec['name'],
                # Add specific action details like index if present in action_result_details
            }
            if "insert_index" in action_result_details:
                common_metadata_payload["insert_index"] = action_result_details["insert_index"]
            if "remove_index" in action_result_details:
                common_metadata_payload["remove_index"] = action_result_details["remove_index"]

            # Metadata for Q1 and Q2
            for q_idx, current_prompt in enumerate(prompts_for_q1q2):
                q_type_label = f"Q{q_idx + 1}"
                meta_id = f"{base_id_prefix}_px{px_size}_{q_type_label}" # Unique ID
                all_notitle_metadata.append({
                    "ID": meta_id, "image_path": image_path_relative,
                    "topic": BOARD_TYPE_NAME, "prompt": current_prompt,
                    "ground_truth": ground_truth_for_q1q2, "expected_bias": expected_bias_for_q1q2,
                    "with_title": False, "type_of_question": q_type_label,
                    "pixel": px_size, # For direct access, though also in common_metadata_payload
                    "metadata": common_metadata_payload.copy()
                })
            
            # Metadata for Q3
            meta_id_q3 = f"{base_id_prefix}_px{px_size}_Q3"
            all_notitle_metadata.append({
                "ID": meta_id_q3, "image_path": image_path_relative,
                "topic": BOARD_TYPE_NAME, "prompt": std_check_prompt_q3,
                "ground_truth": std_check_gt_q3, "expected_bias": std_check_bias_q3,
                "with_title": False, "type_of_question": "Q3",
                "pixel": px_size, 
                "metadata": common_metadata_payload.copy()
            })

    progress_bar.close()

    # Save all collected "notitle" metadata for Chess Board
    print(f"\n  Saving 'notitle' metadata for {BOARD_TYPE_NAME}...")
    # `grid_utils.write_metadata_files` expects `dirs` (from create_directory_structure),
    # and a `board_id_prefix` (which is BOARD_ID here for the combined file).
    grid_utils.write_metadata_files(
        all_notitle_metadata,
        output_dirs, # Contains "notitle_meta_dir"
        BOARD_ID     # Used to form filename like "chess_board_notitle_metadata.json"
    )

    # Summary for this generator
    final_img_count = 0
    try:
        if os.path.exists(output_dirs["notitle_img_dir"]):
            final_img_count = len([f for f in os.listdir(output_dirs["notitle_img_dir"]) if f.endswith('.png')])
    except Exception as e: print(f"  Warning: Could not count final images for {BOARD_ID}: {e}")

    print(f"\n  --- {BOARD_TYPE_NAME} 'notitle' Generation Summary ---")
    print(f"  Actual 'notitle' images generated: {final_img_count} (Expected: {total_images_to_generate})")
    print(f"  Total 'notitle' metadata entries created: {len(all_notitle_metadata)}")
    
    # Clean up temp directory
    try:
        if os.path.exists(temp_dir_path):
            shutil.rmtree(temp_dir_path)
            print(f"  Cleaned up temporary directory: {temp_dir_path}")
    except Exception as e:
        print(f"  Warning: Failed to clean up temp directory {temp_dir_path}: {e}")
        
    print(f"  {BOARD_TYPE_NAME} 'notitle' dataset generation finished.")

if __name__ == '__main__':
    print(f"Testing {BOARD_TYPE_NAME} Generator directly...")
    # This requires grid_utils.py to be importable
    # And will create output directories in the current working directory
    if not os.path.exists("vlms-are-biased-notitle"): # Ensure parent for test
        os.makedirs("vlms-are-biased-notitle")
    create_chess_board_dataset(quality_scale=5.0, svg_size=800)
    print(f"\nDirect test of {BOARD_TYPE_NAME} Generator complete.")