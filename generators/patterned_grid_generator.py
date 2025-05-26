# generators/patterned_grid_generator.py
# -*- coding: utf-8 -*-
"""
Patterned Grid Generator - Generates "notitle" images of grids with
dice-like circle patterns or tally mark patterns, with one anomalous cell.
Outputs to 'dice' and 'tally' subdirectories under vlms-are-biased-notitle/.
"""
import os
import random
import shutil
import string # For column labels A, B, C...
import math   # For shape drawing (star, triangle)
from tqdm import tqdm
import numpy as np # For grid layout calculations if needed
import sys

# Assuming grid_utils.py and utils.py are accessible from this script's execution context
# (e.g. if main.py sets up paths or if run as part of a package)
import grid_utils # For create_directory_structure, svg_to_png_direct
from utils import sanitize_filename # For filenames

# --- Global Constants ---
PIXEL_SIZES = [384, 768, 1152]
# Number of unique (grid_size, cell_selection_pair) combinations
NUM_ACTION_GROUPS = 14 
# For each group, we generate 2 grid types (dice, tally) and 2 action types (remove, replace/add)
# So, total of 4 specific (grid_type, action_type) pairs per group.
# The actions (remove, replace/add) are applied to the *same* anomalous cell per group.
MIN_GRID_DIM = 6
MAX_GRID_DIM = 12

# Define topic IDs for sub-directory creation and metadata
DICE_TOPIC_ID = "dice_patterned_grid" # Changed from just "dice"
TALLY_TOPIC_ID = "tally_patterned_grid" # Changed from just "tally"

# --- Helper: Pattern Logic ---
def get_pattern_count_for_cell(row_idx, col_idx, pattern_type_id, total_rows, total_cols):
    """
    Calculates the number of shapes (dots for dice, lines for tally)
    that should be in a cell based on its position and the pattern type.
    `pattern_type_id = 7` refers to the "increasing-then-decreasing" pattern from paper.
    """
    if total_cols <= 0: return 0 # Avoid division by zero if cols is 0 or less
    
    if pattern_type_id == 7: # Symmetric increasing-then-decreasing from edges to center
        # Distance from the closer vertical edge (0-indexed)
        distance_from_vertical_edge = min(col_idx, total_cols - 1 - col_idx)
        # Count starts at 1 at the edge, increases towards center.
        # Example: 6x6 grid (cols 0-5).
        # col 0: min(0, 5) = 0. count = 1
        # col 1: min(1, 4) = 1. count = 2
        # col 2: min(2, 3) = 2. count = 3
        # col 3: min(3, 2) = 2. count = 3
        # col 4: min(4, 1) = 1. count = 2
        # col 5: min(5, 0) = 0. count = 1
        count = distance_from_vertical_edge + 1
        return int(count) # Ensure integer count
    else:
        # Fallback for unknown patterns, though only type 7 is used.
        print(f"  Warning: Unknown pattern_type_id {pattern_type_id}. Defaulting to 1 count.")
        return 1

# --- SVG Drawing Functions (from original dice_tally_generator.py, adapted) ---

def generate_patterned_grid_svg(
    grid_rows, grid_cols, pattern_id, cell_render_size, cell_spacing_pixels,
    shape_draw_type, base_shape_color,
    anomalous_cell_coords, # (row_idx, col_idx) of the cell with exception
    exception_details,     # Dict: e.g. {'operation': 'remove', 'shape_index_if_dice': 0}
                           #      or {'operation': 'add'} for tally
                           #      or {'operation': 'replace', 'new_shape_type': 'square', 'shape_index_if_dice': 0}
    show_alphanumeric_labels=True, label_font_size_px=12):
    """
    Generates an SVG for a patterned grid with one anomalous cell.
    `exception_details` drives how the `anomalous_cell_coords` differs from standard pattern.
    """
    # Calculate overall SVG dimensions including margins for labels
    label_margin_factor = 2.0 # Multiplier for font size to estimate label margin
    margin_left_for_labels = (label_font_size_px * label_margin_factor * 1.5) if show_alphanumeric_labels else cell_spacing_pixels
    margin_top_for_grid = cell_spacing_pixels # Top margin for the grid part
    margin_bottom_for_labels = (label_font_size_px * label_margin_factor * 1.2) if show_alphanumeric_labels else cell_spacing_pixels

    grid_actual_width = grid_cols * (cell_render_size + cell_spacing_pixels) - cell_spacing_pixels
    grid_actual_height = grid_rows * (cell_render_size + cell_spacing_pixels) - cell_spacing_pixels
    
    total_svg_width = grid_actual_width + margin_left_for_labels + cell_spacing_pixels # Right margin also cell_spacing
    total_svg_height = grid_actual_height + margin_top_for_grid + margin_bottom_for_labels

    svg_output = f'<svg width="{total_svg_width}" height="{total_svg_height}" xmlns="http://www.w3.org/2000/svg">\n'
    svg_output += f'<rect width="{total_svg_width}" height="{total_svg_height}" fill="white"/>\n' # White SVG background

    # Draw Column Labels (A, B, C...) if enabled
    if show_alphanumeric_labels:
        label_y_position = margin_top_for_grid + grid_actual_height + margin_bottom_for_labels * 0.6
        for c_idx in range(grid_cols):
            col_char_label = string.ascii_uppercase[c_idx % 26]
            if c_idx >= 26: # For columns beyond Z (AA, AB...)
                col_char_label = string.ascii_uppercase[(c_idx // 26) - 1] + col_char_label
            
            label_x_position = margin_left_for_labels + c_idx * (cell_render_size + cell_spacing_pixels) + cell_render_size / 2
            svg_output += f'<text x="{label_x_position}" y="{label_y_position}" text-anchor="middle" dominant-baseline="middle" '
            svg_output += f'font-family="Arial, sans-serif" font-size="{label_font_size_px}" fill="black" font-weight="bold">{col_char_label}</text>\n'

    # Draw Row Labels (1, 2, 3...) if enabled
    if show_alphanumeric_labels:
         label_x_position = margin_left_for_labels * 0.4 # Position row labels to the left of grid
         for r_idx in range(grid_rows):
            row_num_label = str(r_idx + 1)
            label_y_position = margin_top_for_grid + r_idx * (cell_render_size + cell_spacing_pixels) + cell_render_size / 2
            svg_output += f'<text x="{label_x_position}" y="{label_y_position}" text-anchor="middle" dominant-baseline="middle" '
            svg_output += f'font-family="Arial, sans-serif" font-size="{label_font_size_px}" fill="black" font-weight="bold">{row_num_label}</text>\n'

    # Draw Grid Cells and Shapes within them
    for r_idx_current_cell in range(grid_rows):
        for c_idx_current_cell in range(grid_cols):
            cell_top_left_x = margin_left_for_labels + c_idx_current_cell * (cell_render_size + cell_spacing_pixels)
            cell_top_left_y = margin_top_for_grid + r_idx_current_cell * (cell_render_size + cell_spacing_pixels)
            
            # Draw cell border
            svg_output += f'<rect x="{cell_top_left_x}" y="{cell_top_left_y}" width="{cell_render_size}" height="{cell_render_size}" fill="none" stroke="black" stroke-width="1"/>\n'
            
            # Determine standard count for this cell based on pattern
            standard_shapes_count = get_pattern_count_for_cell(r_idx_current_cell, c_idx_current_cell, pattern_id, grid_rows, grid_cols)
            
            is_this_the_anomalous_cell = (anomalous_cell_coords is not None and \
                                          r_idx_current_cell == anomalous_cell_coords[0] and \
                                          c_idx_current_cell == anomalous_cell_coords[1])
            
            effective_exception_to_apply = None # This will hold modified exception_details if valid for this cell
            
            # If this is the anomalous cell, validate and prepare the exception
            if is_this_the_anomalous_cell and exception_details:
                op_type = exception_details.get('operation')
                if op_type == 'remove':
                    # For remove, standard count must be > 0. Dice also needs valid index.
                    can_remove = standard_shapes_count > 0
                    if shape_draw_type != 'tally': # Dice/other shapes need index
                        can_remove = can_remove and (0 <= exception_details.get('shape_index_if_dice', -1) < standard_shapes_count)
                    if can_remove:
                        effective_exception_to_apply = {'operation': 'remove', 'shape_index_if_dice': exception_details.get('shape_index_if_dice')}
                
                elif op_type == 'add': # Tally only usually
                    effective_exception_to_apply = {'operation': 'add'} # Add is generally always possible
                
                elif op_type == 'replace': # Dice only usually
                    # For replace, standard count must be > 0 and index valid
                    can_replace = standard_shapes_count > 0 and \
                                  (0 <= exception_details.get('shape_index_if_dice', -1) < standard_shapes_count)
                    if can_replace:
                        effective_exception_to_apply = {
                            'operation': 'replace',
                            'new_shape_type': exception_details.get('new_shape_type', shape_draw_type), # Fallback to original if not specified
                            'new_shape_color': exception_details.get('new_shape_color', base_shape_color),
                            'shape_index_if_dice': exception_details.get('shape_index_if_dice')
                        }
            
            # Draw shapes if standard count > 0 OR if it's an anomalous cell being added to (from 0)
            should_draw_shapes_in_cell = standard_shapes_count > 0 or \
                                         (is_this_the_anomalous_cell and effective_exception_to_apply and \
                                          effective_exception_to_apply.get('operation') == 'add')
            
            if should_draw_shapes_in_cell:
                if shape_draw_type == 'tally':
                    svg_output += _draw_tally_marks_in_cell(
                        cell_top_left_x, cell_top_left_y, cell_render_size, 
                        standard_shapes_count,
                        is_this_the_anomalous_cell, 
                        effective_exception_to_apply, # Pass processed exception
                        base_shape_color
                    )
                else: # For dice (circles) or other shapes
                     svg_output += _draw_dice_like_shapes_in_cell(
                         cell_top_left_x, cell_top_left_y, cell_render_size, 
                         standard_shapes_count,
                         shape_draw_type, base_shape_color,
                         is_this_the_anomalous_cell,
                         effective_exception_to_apply # Pass processed exception
                     )
    svg_output += '</svg>'
    return svg_output

# Helper: Draw Tally Marks
def _draw_tally_marks_in_cell(cell_x, cell_y, cell_dim, standard_line_count, 
                             is_anomalous, exception_to_apply, line_color='black'):
    svg_lines = ''
    padding = cell_dim * 0.15 # Padding inside the cell
    mark_line_width_px = max(1.5, cell_dim * 0.035) # Thickness of tally lines
    mark_line_height_px = cell_dim - 2 * padding    # Length of vertical tally lines
    available_width_for_tallies = cell_dim - 2 * padding

    # Determine the final number of tally lines to draw based on exception
    final_line_count = standard_line_count
    if is_anomalous and exception_to_apply:
        operation = exception_to_apply.get('operation')
        if operation == 'remove' and standard_line_count > 0:
            final_line_count -= 1
        elif operation == 'add':
            final_line_count += 1
    final_line_count = max(0, final_line_count) # Cannot have negative lines

    if final_line_count == 0: return svg_lines

    # Layout parameters for tally marks
    line_spacing_internal_ratio = 0.1 # Spacing between vertical lines in a group of 4
    group_spacing_ratio = 0.15      # Spacing after a full group of 5

    line_internal_spacing_px = available_width_for_tallies * line_spacing_internal_ratio
    spacing_after_full_group_px = available_width_for_tallies * group_spacing_ratio
    # Width taken by 4 vertical lines + 3 internal spaces (approximate, diagonal adds width)
    width_of_four_verticals = 3 * line_internal_spacing_px 

    # Generate line coordinates
    lines_to_render = []
    current_x_offset = padding # Start drawing from left padding edge
    
    num_full_tally_groups = final_line_count // 5
    num_remaining_single_lines = final_line_count % 5

    # Draw full groups of 5
    for _ in range(num_full_tally_groups):
        group_start_x_abs = cell_x + current_x_offset
        # Draw 4 vertical lines
        for i in range(4):
            line_x_abs = group_start_x_abs + i * line_internal_spacing_px
            lines_to_render.append({
                'x1': line_x_abs, 'y1': cell_y + padding, 
                'x2': line_x_abs, 'y2': cell_y + padding + mark_line_height_px
            })
        # Draw diagonal strike-through line for the 5th mark
        diag_x1_abs = group_start_x_abs - line_internal_spacing_px * 0.2 # Start slightly before first vertical
        diag_x2_abs = group_start_x_abs + width_of_four_verticals - line_internal_spacing_px * 0.8 # End slightly after last vertical based on 4 lines
        diag_y1_abs = cell_y + padding + mark_line_height_px * 0.1 # Diagonal starts near top
        diag_y2_abs = cell_y + padding + mark_line_height_px * 0.9 # Diagonal ends near bottom
        lines_to_render.append({
            'x1': diag_x1_abs, 'y1': diag_y1_abs, 
            'x2': diag_x2_abs, 'y2': diag_y2_abs
        })
        current_x_offset += width_of_four_verticals + spacing_after_full_group_px # Advance X for next group

    # Draw remaining single vertical lines
    singles_start_x_abs = cell_x + current_x_offset
    for i in range(num_remaining_single_lines):
        line_x_abs = singles_start_x_abs + i * line_internal_spacing_px
        lines_to_render.append({
            'x1': line_x_abs, 'y1': cell_y + padding, 
            'x2': line_x_abs, 'y2': cell_y + padding + mark_line_height_px
        })

    # Convert line coordinates to SVG line elements
    for line_coords in lines_to_render:
        svg_lines += f'<line x1="{line_coords["x1"]:.2f}" y1="{line_coords["y1"]:.2f}" x2="{line_coords["x2"]:.2f}" y2="{line_coords["y2"]:.2f}" '
        svg_lines += f'stroke="{line_color}" stroke-width="{mark_line_width_px:.2f}" stroke-linecap="round"/>\n'
    return svg_lines

# Helper: Draw Dice-like shapes (circles, squares, etc.)
def _draw_dice_like_shapes_in_cell(
    cell_x, cell_y, cell_dim, standard_shape_count, 
    shape_type_to_draw, shape_color_to_use,
    is_anomalous, exception_to_apply):
    # (This function is very similar to draw_shapes from original, needs careful adaptation)
    # (It now uses exception_to_apply which has processed 'operation' and related details)
    
    svg_shapes = ''
    cell_padding = cell_dim * 0.15 # Padding within the cell for shapes
    drawable_area_size = cell_dim - (2 * cell_padding)
    base_indiv_shape_size = drawable_area_size * 0.3 # Default size for one shape
    max_shapes_before_rescale = 6 # If more shapes, they might need to be smaller

    # Determine final number of shapes for scaling and positions
    count_for_layout = standard_shape_count
    shape_details_list = [] # List of {'x', 'y', 'size', 'type', 'color'} for each shape

    if is_anomalous and exception_to_apply:
        operation = exception_to_apply.get('operation')
        idx_for_dice_ops = exception_to_apply.get('shape_index_if_dice', 0)

        if operation == 'remove':
            count_for_layout -= 1
            # Create initial list and then remove one
            initial_positions = _get_dice_shape_positions(standard_shape_count, cell_x, cell_y, cell_dim, cell_padding, base_indiv_shape_size)
            for i, (px, py) in enumerate(initial_positions):
                if i == idx_for_dice_ops: continue # Skip the removed shape
                shape_details_list.append({'x':px, 'y':py, 'size':base_indiv_shape_size, 'type':shape_type_to_draw, 'color':shape_color_to_use})

        elif operation == 'replace':
            count_for_layout = standard_shape_count # Count doesn't change
            new_type_for_replaced = exception_to_apply.get('new_shape_type', shape_type_to_draw)
            new_color_for_replaced = exception_to_apply.get('new_shape_color', shape_color_to_use)
            initial_positions = _get_dice_shape_positions(standard_shape_count, cell_x, cell_y, cell_dim, cell_padding, base_indiv_shape_size)
            for i, (px, py) in enumerate(initial_positions):
                current_type = new_type_for_replaced if i == idx_for_dice_ops else shape_type_to_draw
                current_color = new_color_for_replaced if i == idx_for_dice_ops else shape_color_to_use
                shape_details_list.append({'x':px, 'y':py, 'size':base_indiv_shape_size, 'type':current_type, 'color':current_color})
        
        elif operation == 'add': # This is less common for dice, more for tally, but handled if used
            count_for_layout += 1
            # This requires getting positions for count_for_layout shapes
            # (Implementation for adding shapes to dice patterns needs careful position logic)
            # For now, let's assume 'add' mainly applies to tally; dice uses replace/remove.
            # If 'add' is used for dice, we'd need robust `_get_next_dice_shape_position`
            # For simplicity, if 'add' is forced on dice, it might just draw `standard_shape_count + 1` shapes
            # using the standard layout for that new count.
            all_positions_for_added = _get_dice_shape_positions(count_for_layout, cell_x, cell_y, cell_dim, cell_padding, base_indiv_shape_size)
            for (px,py) in all_positions_for_added:
                shape_details_list.append({'x':px, 'y':py, 'size':base_indiv_shape_size, 'type':shape_type_to_draw, 'color':shape_color_to_use})

        else: # No valid operation, draw standard
            positions = _get_dice_shape_positions(standard_shape_count, cell_x, cell_y, cell_dim, cell_padding, base_indiv_shape_size)
            for (px,py) in positions:
                shape_details_list.append({'x':px, 'y':py, 'size':base_indiv_shape_size, 'type':shape_type_to_draw, 'color':shape_color_to_use})
    else: # Not anomalous, or no valid exception, draw standard
        positions = _get_dice_shape_positions(standard_shape_count, cell_x, cell_y, cell_dim, cell_padding, base_indiv_shape_size)
        for (px,py) in positions:
            shape_details_list.append({'x':px, 'y':py, 'size':base_indiv_shape_size, 'type':shape_type_to_draw, 'color':shape_color_to_use})

    # Adjust shape size if many shapes are present
    actual_num_shapes_to_draw = len(shape_details_list)
    if actual_num_shapes_to_draw == 0: return svg_shapes
    
    scale_factor_for_many_shapes = 1.0
    if actual_num_shapes_to_draw > max_shapes_before_rescale:
        scale_factor_for_many_shapes = min(1.0, math.sqrt(max_shapes_before_rescale / float(actual_num_shapes_to_draw)))
    
    final_indiv_shape_size = max(5, base_indiv_shape_size * scale_factor_for_many_shapes)

    # Re-calculate positions if size changed significantly (optional, for now use original relative positions)
    # For simplicity, we'll just update the size in shape_details_list.
    # A more complex version would re-run _get_dice_shape_positions with the new final_indiv_shape_size
    # and count_for_layout, but that might create circular dependencies if not handled carefully.
    
    for shape_detail in shape_details_list:
        # Update size. Positions might need re-centering if size changes drastically,
        # but for now, assume _get_dice_shape_positions gives top-left for the base_indiv_shape_size.
        shape_detail['size'] = final_indiv_shape_size
        svg_shapes += _draw_single_geometric_shape(
            shape_detail['x'], shape_detail['y'], 
            shape_detail['size'], 
            shape_detail['type'], shape_detail['color']
        )
    return svg_shapes

# Helper: Get Dice Shape Positions (from original)
def _get_dice_shape_positions(num_shapes, cell_tl_x, cell_tl_y, cell_full_dim, padding_in_cell, indiv_shape_actual_size):
    # (This function is mostly from the original `get_dice_positions`)
    if num_shapes <= 0: return []
    
    draw_area_start_x = cell_tl_x + padding_in_cell
    draw_area_start_y = cell_tl_y + padding_in_cell
    draw_area_dim_w = max(0, cell_full_dim - 2 * padding_in_cell)
    draw_area_dim_h = max(0, cell_full_dim - 2 * padding_in_cell) # Assume square cells for dice
    
    center_x_abs = draw_area_start_x + draw_area_dim_w / 2
    center_y_abs = draw_area_start_y + draw_area_dim_h / 2

    # Define key positions (top-left of where a shape of indiv_shape_actual_size would be placed)
    # These are relative to the drawable area within the cell.
    pos_center = (center_x_abs - indiv_shape_actual_size / 2, center_y_abs - indiv_shape_actual_size / 2)
    pos_top_left = (draw_area_start_x, draw_area_start_y)
    pos_top_right = (draw_area_start_x + draw_area_dim_w - indiv_shape_actual_size, draw_area_start_y)
    pos_bottom_left = (draw_area_start_x, draw_area_start_y + draw_area_dim_h - indiv_shape_actual_size)
    pos_bottom_right = (draw_area_start_x + draw_area_dim_w - indiv_shape_actual_size, draw_area_start_y + draw_area_dim_h - indiv_shape_actual_size)
    pos_center_left = (draw_area_start_x, center_y_abs - indiv_shape_actual_size / 2) # Mid-left
    pos_center_right = (draw_area_start_x + draw_area_dim_w - indiv_shape_actual_size, center_y_abs - indiv_shape_actual_size / 2) # Mid-right

    # Standard dice patterns for 1-6 shapes
    dice_dot_patterns = {
        1: [pos_center],
        2: [pos_top_left, pos_bottom_right],
        3: [pos_top_left, pos_center, pos_bottom_right],
        4: [pos_top_left, pos_top_right, pos_bottom_left, pos_bottom_right],
        5: [pos_top_left, pos_top_right, pos_center, pos_bottom_left, pos_bottom_right],
        6: [pos_top_left, pos_top_right, pos_center_left, pos_center_right, pos_bottom_left, pos_bottom_right]
    }
    
    if num_shapes in dice_dot_patterns:
        return dice_dot_patterns[num_shapes]
    else: # For > 6 shapes, use a grid layout within the cell
        positions_list = []
        # Determine grid dimensions for shapes within the cell (e.g., 3x3 for 7-9 shapes)
        grid_cols_for_shapes = max(1, int(np.ceil(np.sqrt(num_shapes))))
        grid_rows_for_shapes = max(1, int(np.ceil(num_shapes / float(grid_cols_for_shapes))))
        
        # Try to make the grid more "square-like" if possible
        if grid_cols_for_shapes > 1 and grid_rows_for_shapes * (grid_cols_for_shapes - 1) >= num_shapes:
            grid_cols_for_shapes -= 1
            grid_rows_for_shapes = max(1, int(np.ceil(num_shapes / float(grid_cols_for_shapes))))
            
        # Spacing between shapes in the mini-grid
        spacing_x_shapes = (draw_area_dim_w - indiv_shape_actual_size) / (grid_cols_for_shapes - 1) if grid_cols_for_shapes > 1 else 0
        spacing_y_shapes = (draw_area_dim_h - indiv_shape_actual_size) / (grid_rows_for_shapes - 1) if grid_rows_for_shapes > 1 else 0
        
        # Total width/height of the shape grid itself
        shape_grid_total_w = (grid_cols_for_shapes - 1) * spacing_x_shapes + indiv_shape_actual_size if grid_cols_for_shapes > 0 else 0
        shape_grid_total_h = (grid_rows_for_shapes - 1) * spacing_y_shapes + indiv_shape_actual_size if grid_rows_for_shapes > 0 else 0
        
        # Starting point to center the shape grid within the drawable area
        start_x_for_shape_grid = draw_area_start_x + (draw_area_dim_w - shape_grid_total_w) / 2
        start_y_for_shape_grid = draw_area_start_y + (draw_area_dim_h - shape_grid_total_h) / 2
        
        shapes_placed_count = 0
        for r_s_idx in range(grid_rows_for_shapes):
            for c_s_idx in range(grid_cols_for_shapes):
                if shapes_placed_count < num_shapes:
                    # Top-left coordinate for this shape
                    px_shape = start_x_for_shape_grid + c_s_idx * spacing_x_shapes if grid_cols_for_shapes > 1 else start_x_for_shape_grid
                    py_shape = start_y_for_shape_grid + r_s_idx * spacing_y_shapes if grid_rows_for_shapes > 1 else start_y_for_shape_grid
                    positions_list.append((px_shape, py_shape))
                    shapes_placed_count += 1
                else: break
            if shapes_placed_count >= num_shapes: break
        return positions_list

# Helper: Draw Single Geometric Shape (from original)
def _draw_single_geometric_shape(shape_tl_x, shape_tl_y, shape_dim_size, shape_type_str, shape_fill_color):
    # (This function is mostly from the original `draw_shape`)
    svg_element = ''
    if shape_dim_size <= 0: return "" # Cannot draw shape with no size

    # Calculate center and radius based on top-left (shape_tl_x, shape_tl_y) and dimension (shape_dim_size)
    center_x_of_shape = shape_tl_x + shape_dim_size / 2.0
    center_y_of_shape = shape_tl_y + shape_dim_size / 2.0
    radius_for_shape = shape_dim_size / 2.0
    
    # Effective drawing radius/side, slightly smaller for visual appeal (avoid touching theoretical bounds)
    effective_draw_radius = max(0.1, radius_for_shape * 0.9)
    effective_draw_side = max(0.1, shape_dim_size * 0.9)

    if shape_type_str == 'circle':
        svg_element += f'<circle cx="{center_x_of_shape:.2f}" cy="{center_y_of_shape:.2f}" r="{effective_draw_radius:.2f}" fill="{shape_fill_color}"/>\n'
    elif shape_type_str == 'square':
        # For square, shape_tl_x,y is already top-left if effective_draw_side is shape_dim_size*0.9
        # Need to adjust top-left for the smaller square to keep it centered
        sq_draw_tl_x = center_x_of_shape - effective_draw_side / 2.0
        sq_draw_tl_y = center_y_of_shape - effective_draw_side / 2.0
        svg_element += f'<rect x="{sq_draw_tl_x:.2f}" y="{sq_draw_tl_y:.2f}" width="{effective_draw_side:.2f}" height="{effective_draw_side:.2f}" fill="{shape_fill_color}"/>\n'
    elif shape_type_str == 'triangle': # Equilateral triangle pointing up, centered
        height_of_triangle = max(0.1, effective_draw_radius * 1.5) # Triangle height based on radius
        # Base width of triangle = height / (sqrt(3)/2) if equilateral
        base_width_triangle = height_of_triangle / (math.sqrt(3)/2.0) * 1.0 if math.sqrt(3) > 0 else height_of_triangle

        # Points: p1 (top), p2 (bottom-left), p3 (bottom-right)
        p1 = (center_x_of_shape, center_y_of_shape - effective_draw_radius) # Top point
        p2 = (center_x_of_shape - base_width_triangle / 2.0, center_y_of_shape + effective_draw_radius * 0.5) # Bottom-left
        p3 = (center_x_of_shape + base_width_triangle / 2.0, center_y_of_shape + effective_draw_radius * 0.5) # Bottom-right
        svg_element += f'<polygon points="{p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}" fill="{shape_fill_color}"/>\n'
    elif shape_type_str == 'star': # 5-point star
        outer_r_star = max(0.1, radius_for_shape * 0.95)
        inner_r_star = outer_r_star * 0.38 # Standard ratio for inner radius of 5-point star
        num_star_points = 5
        
        all_star_points_coords = []
        for i in range(num_star_points * 2): # Iterate through outer and inner vertices
            current_radius_for_vertex = outer_r_star if i % 2 == 0 else inner_r_star
            # Angle for each vertex. Offset by -PI/2 to make one point go upwards.
            angle_rad = math.pi * i / num_star_points - (math.pi / 2.0)
            
            px_vertex = center_x_of_shape + current_radius_for_vertex * math.cos(angle_rad)
            py_vertex = center_y_of_shape + current_radius_for_vertex * math.sin(angle_rad)
            all_star_points_coords.append(f"{px_vertex:.2f},{py_vertex:.2f}")
        
        svg_element += '<polygon points="'
        svg_element += " ".join(all_star_points_coords)
        svg_element += f'" fill="{shape_fill_color}"/>\n'
    else: # Fallback for unknown shape type
        print(f"  Warning: Unknown shape_type '{shape_type_str}' in _draw_single_geometric_shape. Drawing a small black circle as fallback.");
        svg_element += f'<circle cx="{center_x_of_shape:.2f}" cy="{center_y_of_shape:.2f}" r="{max(0.1, radius_for_shape * 0.5):.2f}" fill="black"/>\n'
    return svg_element


# --- Helper: Select Anomalous Cell ---
def select_anomalous_cell_for_grid(grid_r, grid_c, previously_selected_cells_for_size=None):
    # (Logic from original `select_edge_cell` - aims to pick non-edge cells)
    if previously_selected_cells_for_size is None: previously_selected_cells_for_size = set()
    
    # Define tiers of how many rows/cols to exclude from edges
    # Tier 1: exclude 3 rows/cols from each edge (if grid is large enough)
    # Tier 2: exclude 2 rows/cols
    # Tier 3: exclude 1 row/col
    # Tier 4: any cell (if all preferred tiers are exhausted)
    
    excluded_rows_tier1, excluded_cols_tier1 = set(), set()
    if grid_r > 7 and grid_c > 7: # Grid must be at least 8x8 for 3-cell exclusion to leave a center
        excluded_rows_tier1 = {0, 1, 2, grid_r-3, grid_r-2, grid_r-1}
        excluded_cols_tier1 = {0, 1, 2, grid_c-3, grid_c-2, grid_c-1}
    
    excluded_rows_tier2, excluded_cols_tier2 = set(), set()
    if grid_r > 4 and grid_c > 4: # Grid at least 5x5 for 2-cell exclusion
        excluded_rows_tier2 = {0, 1, grid_r-2, grid_r-1}
        excluded_cols_tier2 = {0, 1, grid_c-2, grid_c-1}
        
    excluded_rows_tier3, excluded_cols_tier3 = set(), set()
    if grid_r > 1 and grid_c > 1: # Grid at least 2x2 for 1-cell exclusion
        excluded_rows_tier3 = {0, grid_r-1}
        excluded_cols_tier3 = {0, grid_c-1}

    def get_available_cells(excluded_r, excluded_c, previously_selected):
        potential_cells = []
        for r_idx in range(grid_r):
            if r_idx in excluded_r: continue
            for c_idx in range(grid_c):
                if c_idx in excluded_c: continue
                potential_cells.append((r_idx, c_idx))
        
        available_new_cells = [cell for cell in potential_cells if cell not in previously_selected]
        return available_new_cells, potential_cells # Return new ones, and all potentials in this tier

    # Determine which tier to start with based on grid size
    current_excluded_r, current_excluded_c = excluded_rows_tier3, excluded_cols_tier3
    tier_name_start = "Tier 3 (1-cell edge exclusion)"
    if grid_r > 4 and grid_c > 4:
        current_excluded_r, current_excluded_c = excluded_rows_tier2, excluded_cols_tier2
        tier_name_start = "Tier 2 (2-cell edge exclusion)"
    if grid_r > 7 and grid_c > 7:
        current_excluded_r, current_excluded_c = excluded_rows_tier1, excluded_cols_tier1
        tier_name_start = "Tier 1 (3-cell edge exclusion)"

    # Define tiers to try in order of preference
    search_tiers = []
    if (current_excluded_r, current_excluded_c) == (excluded_rows_tier1, excluded_cols_tier1):
        search_tiers = [
            (excluded_rows_tier1, excluded_cols_tier1, "Tier 1"),
            (excluded_rows_tier2, excluded_cols_tier2, "Tier 2 Fallback"),
            (excluded_rows_tier3, excluded_cols_tier3, "Tier 3 Fallback")
        ]
    elif (current_excluded_r, current_excluded_c) == (excluded_rows_tier2, excluded_cols_tier2):
        search_tiers = [
            (excluded_rows_tier2, excluded_cols_tier2, "Tier 2"),
            (excluded_rows_tier3, excluded_cols_tier3, "Tier 3 Fallback")
        ]
    else: # Default to Tier 3 or smaller grids
         search_tiers = [(excluded_rows_tier3, excluded_cols_tier3, "Tier 3")]
         
    # Try each tier
    for ex_r, ex_c, tier_desc_name in search_tiers:
        available_fresh_cells, all_potential_in_tier = get_available_cells(ex_r, ex_c, previously_selected_cells_for_size)
        if available_fresh_cells:
            return random.choice(available_fresh_cells) # Found a fresh cell in this tier
        # If no fresh cells, but potential cells exist in this tier (meaning they were previously selected)
        if all_potential_in_tier: 
            print(f"  Warning: Reusing a cell from '{tier_desc_name}' for grid {grid_r}x{grid_c} as no fresh cells available in this tier.")
            return random.choice(all_potential_in_tier) # Pick a previously used one from this tier
            
    # If all preferred tiers are exhausted (e.g., very small grid or all cells used up)
    print(f"  Warning: All preferred tiers exhausted for selecting anomalous cell in {grid_r}x{grid_c}. Trying any available cell.")
    all_grid_cells = [(r,c) for r in range(grid_r) for c in range(grid_c)]
    available_any_fresh = [cell for cell in all_grid_cells if cell not in previously_selected_cells_for_size]
    if available_any_fresh:
        return random.choice(available_any_fresh)
    if all_grid_cells: # If no fresh cells at all, reuse any cell
        print(f"  CRITICAL WARNING: Reusing *any* cell (all cells previously selected) for grid {grid_r}x{grid_c}.")
        return random.choice(all_grid_cells)
        
    print(f"  ERROR: Cannot select any cell for grid {grid_r}x{grid_c}. This should not happen.")
    return None # Should be unreachable if grid_r/c > 0


# --- Main Patterned Grid Dataset Generation Function ---
def create_grid_dataset(quality_scale=5.0, svg_size=800):
    """
    Generates "notitle" datasets for Dice and Tally patterned grids.
    Each "group" corresponds to a (grid_size, anomalous_cell_pair) combination,
    and for each group, 4 image types are generated (Dice-Remove, Dice-Replace, Tally-Remove, Tally-Add).
    """
    # Create directory structures for Dice and Tally outputs
    dice_output_dirs = grid_utils.create_directory_structure(DICE_TOPIC_ID)
    tally_output_dirs = grid_utils.create_directory_structure(TALLY_TOPIC_ID)
    
    # For temp files, we can use a shared one or per-type.
    # grid_utils.create_directory_structure makes per-type temp dirs.
    # We might want a general temp dir if processing steps are shared before splitting.
    # For now, assume SVG->PNG is per type. Let's use dice_output_dirs["temp_dir"] as a generic.
    # It might be cleaner if generate_single_action_image_and_meta handles its own temp dir usage.
    shared_temp_dir_base = "temp_patterned_grid_output" # A common base for temp files
    os.makedirs(shared_temp_dir_base, exist_ok=True)


    print(f"  Starting Patterned Grid ('{DICE_TOPIC_ID}' & '{TALLY_TOPIC_ID}') 'notitle' dataset generation...")
    
    # Determine grid sizes to use for the NUM_ACTION_GROUPS
    possible_grid_dims = list(range(MIN_GRID_DIM, MAX_GRID_DIM + 1))
    num_unique_dims = len(possible_grid_dims)
    # Assign a grid size (rows=cols) to each of the NUM_ACTION_GROUPS, cycling through possible_grid_dims
    grid_sizes_for_groups = [(possible_grid_dims[i % num_unique_dims], possible_grid_dims[i % num_unique_dims]) 
                             for i in range(NUM_ACTION_GROUPS)]

    all_dice_metadata_entries = []
    all_tally_metadata_entries = []
    # Keep track of selected anomalous cells for each grid size to try to avoid reuse for *different groups* of same size
    selected_anomalous_cells_by_size = {} 

    # Total images: NUM_ACTION_GROUPS * 4 actions (Dice-Rem, Dice-Rep, Tally-Rem, Tally-Add) * len(PIXEL_SIZES)
    total_images_to_generate = NUM_ACTION_GROUPS * 4 * len(PIXEL_SIZES)
    progress_bar = tqdm(total=total_images_to_generate, desc="Generating Patterned Grids", unit="image", ncols=100)

    for group_idx in range(NUM_ACTION_GROUPS):
        current_group_id = group_idx + 1
        grid_r_for_group, grid_c_for_group = grid_sizes_for_groups[group_idx]
        grid_size_tuple_key = (grid_r_for_group, grid_c_for_group)

        # Select one anomalous cell for this entire group (will be used for all 4 actions)
        previously_selected_for_this_size = selected_anomalous_cells_by_size.get(grid_size_tuple_key, set())
        anomalous_cell_for_group = select_anomalous_cell_for_grid(grid_r_for_group, grid_c_for_group, previously_selected_for_this_size)
        
        if anomalous_cell_for_group is None:
            print(f"  FATAL: Could not select anomalous cell for Group {current_group_id} (size {grid_r_for_group}x{grid_c_for_group}). Skipping group.")
            progress_bar.update(4 * len(PIXEL_SIZES)) # Skip all images for this group
            continue
        
        # Record the selected cell to try to avoid re-selecting it for the *next group* of the same size
        if grid_size_tuple_key not in selected_anomalous_cells_by_size:
            selected_anomalous_cells_by_size[grid_size_tuple_key] = set()
        selected_anomalous_cells_by_size[grid_size_tuple_key].add(anomalous_cell_for_group)

        # Determine target shape index for Dice Remove/Replace actions within this group
        # This index will be the same for both Dice Remove and Dice Replace for consistency.
        standard_count_in_anomalous_cell = get_pattern_count_for_cell(
            anomalous_cell_for_group[0], anomalous_cell_for_group[1], 
            7, grid_r_for_group, grid_c_for_group
        )
        # If standard count is 0, index is irrelevant but set to 0. If >0, pick valid index.
        target_dice_shape_index = random.randint(0, standard_count_in_anomalous_cell - 1) if standard_count_in_anomalous_cell > 0 else 0

        # Define the 4 actions for this group (using the same anomalous_cell_for_group and target_dice_shape_index)
        actions_for_this_group = [
            # Dice Pattern Grid Actions
            {'grid_type_id': DICE_TOPIC_ID, 'shape_to_draw': 'circle', 'action_verb': 'remove', 
             'exception_config': {'operation': 'remove', 'shape_index_if_dice': target_dice_shape_index}},
            {'grid_type_id': DICE_TOPIC_ID, 'shape_to_draw': 'circle', 'action_verb': 'replace', 
             'exception_config': {'operation': 'replace', 
                                  'new_shape_type': random.choice(['square', 'triangle', 'star']), # Randomly pick replacement shape
                                  'shape_index_if_dice': target_dice_shape_index}},
            # Tally Pattern Grid Actions
            {'grid_type_id': TALLY_TOPIC_ID, 'shape_to_draw': 'tally', 'action_verb': 'remove',
             'exception_config': {'operation': 'remove'}}, # Tally remove doesn't use shape_index
            {'grid_type_id': TALLY_TOPIC_ID, 'shape_to_draw': 'tally', 'action_verb': 'add',
             'exception_config': {'operation': 'add'}},    # Tally add doesn't use shape_index
        ]

        for action_config in actions_for_this_group:
            # Determine correct output directories (dice or tally)
            current_output_dirs = dice_output_dirs if action_config['grid_type_id'] == DICE_TOPIC_ID else tally_output_dirs
            
            # Generate images and metadata for this specific action config
            metadata_for_this_action = _generate_single_patterned_grid_action_images_and_meta(
                current_group_id,
                action_config['grid_type_id'], # DICE_TOPIC_ID or TALLY_TOPIC_ID
                action_config['action_verb'],  # "remove", "replace", "add"
                current_output_dirs,           # Specific output dirs for dice/tally
                action_config['shape_to_draw'],# "circle" or "tally"
                "black",                       # base_shape_color
                PIXEL_SIZES, quality_scale, svg_size,
                grid_r_for_group, grid_c_for_group,
                anomalous_cell_for_group,      # (r,c) tuple
                action_config['exception_config'], # Details of the anomaly
                progress_bar,
                shared_temp_dir_base # Pass the shared temp dir base
            )
            
            if action_config['grid_type_id'] == DICE_TOPIC_ID:
                all_dice_metadata_entries.extend(metadata_for_this_action)
            else: # TALLY_TOPIC_ID
                all_tally_metadata_entries.extend(metadata_for_this_action)
                
    progress_bar.close()
    if progress_bar.n < progress_bar.total:
        print(f"  Warning: Patterned Grids generation progress ({progress_bar.n}/{progress_bar.total}) indicates some images may have been skipped.")

    # --- Save Metadata for Dice and Tally separately ---
    print(f"\n  Saving 'notitle' metadata for {DICE_TOPIC_ID}...")
    grid_utils.write_metadata_files(all_dice_metadata_entries, dice_output_dirs, DICE_TOPIC_ID)
    
    print(f"\n  Saving 'notitle' metadata for {TALLY_TOPIC_ID}...")
    grid_utils.write_metadata_files(all_tally_metadata_entries, tally_output_dirs, TALLY_TOPIC_ID)

    # --- Final Summary ---
    # (Count actual files and print summary as in other generators)
    dice_img_count = 0; tally_img_count = 0
    try:
        if os.path.exists(dice_output_dirs["notitle_img_dir"]):
            dice_img_count = len([f for f in os.listdir(dice_output_dirs["notitle_img_dir"]) if f.endswith('.png')])
        if os.path.exists(tally_output_dirs["notitle_img_dir"]):
            tally_img_count = len([f for f in os.listdir(tally_output_dirs["notitle_img_dir"]) if f.endswith('.png')])
    except Exception as e: print(f"  Warning: Could not count final images: {e}")

    print(f"\n  --- Patterned Grids 'notitle' Generation Summary ---")
    print(f"  Actual Dice Patterned Grid images: {dice_img_count}")
    print(f"  Actual Tally Patterned Grid images: {tally_img_count}")
    print(f"  Total images expected for both: {total_images_to_generate}")
    print(f"  Dice metadata entries: {len(all_dice_metadata_entries)}")
    print(f"  Tally metadata entries: {len(all_tally_metadata_entries)}")
    
    # Cleanup shared temp dir
    try:
        if os.path.exists(shared_temp_dir_base):
            shutil.rmtree(shared_temp_dir_base)
            print(f"  Cleaned up shared temporary directory: {shared_temp_dir_base}")
    except Exception as e:
        print(f"  Warning: Failed to clean up shared temp directory {shared_temp_dir_base}: {e}")
    print(f"  Patterned Grids 'notitle' dataset generation finished.")


def _generate_single_patterned_grid_action_images_and_meta(
    group_id_val, grid_type_identifier, action_verb_str,
    output_dirs_for_type, # Specific dirs for dice or tally
    shape_type_str, shape_color_str,
    pixel_sizes_list, png_quality_val, svg_render_size_hint,
    num_rows, num_cols, anomalous_cell_tuple, # (r,c) of anomalous cell
    anomaly_details_dict, # {'operation': 'remove', 'shape_index_if_dice': 0} etc.
    progress_bar_obj,
    temp_dir_base_path): # Base path for temp files for this run

    # (This function is similar to generate_single_action_image from original, adapted for this structure)
    # (It now takes output_dirs_for_type to save directly to dice/tally specific locations)

    generated_metadata_list = []
    anom_r, anom_c = anomalous_cell_tuple
    
    # Cell name like "A1", "C5" for use in prompts
    display_col_label = string.ascii_uppercase[anom_c % 26]
    if anom_c >= 26: display_col_label = string.ascii_uppercase[(anom_c // 26) -1] + display_col_label
    display_cell_name = f"{display_col_label}{anom_r + 1}"

    standard_count_at_anomalous_cell = get_pattern_count_for_cell(anom_r, anom_c, 7, num_rows, num_cols)
    
    # Determine modified count based on anomaly_details_dict
    modified_count_at_anomalous_cell = standard_count_at_anomalous_cell
    operation_performed = anomaly_details_dict.get('operation')
    if operation_performed == 'remove':
        if standard_count_at_anomalous_cell > 0: modified_count_at_anomalous_cell -=1
    elif operation_performed == 'add': # Tally only
        modified_count_at_anomalous_cell += 1
    elif operation_performed == 'replace': # Dice only, count of shapes in cell doesn't change
        modified_count_at_anomalous_cell = standard_count_at_anomalous_cell 
    modified_count_at_anomalous_cell = max(0, modified_count_at_anomalous_cell)

    # Prompts (Q1, Q2 ask about count in anomalous cell; Q3 asks if cell matches standard count)
    prompt_item_name = "circles" if grid_type_identifier == DICE_TOPIC_ID else "lines"
    prompts_q1_q2 = [
        f"How many {prompt_item_name} are there in cell {display_cell_name}? Answer with a number in curly brackets, e.g., {{9}}.",
        f"Count the {prompt_item_name} in cell {display_cell_name}. Answer with a number in curly brackets, e.g., {{9}}."
    ]
    ground_truth_q1_q2 = str(modified_count_at_anomalous_cell)
    expected_bias_q1_q2 = str(standard_count_at_anomalous_cell)
    
    prompt_q3 = f"Does cell {display_cell_name} contain {standard_count_at_anomalous_cell} {prompt_item_name}? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    # Q3 Ground truth: if modified_count != standard_count, then "No". If they are same (e.g. replace didn't change count), then "Yes".
    # However, for this task, Q3's intent is to check if the *visual appearance* matches standard.
    # A "replace" action changes appearance even if count is same. So, Q3 GT should be "No" for all our actions.
    ground_truth_q3 = "No" 
    expected_bias_q3 = "Yes" # Bias is to assume cell matches pattern

    group_id_str_fmt = f"{group_id_val:03d}"
    grid_dims_str = f"{num_rows}x{num_cols}"
    sanitized_action_str = sanitize_filename(action_verb_str)
    sanitized_cell_name_str = sanitize_filename(display_cell_name)

    # Ensure the specific temp directory for this grid type exists if using per-type temp
    # For simplicity, using one temp_dir_base_path passed in.
    
    for px_size in pixel_sizes_list:
        # Filename construction
        # e.g., dice_patterned_grid_001_remove_C5_px384.png
        img_basename = f"{grid_type_identifier}_{group_id_str_fmt}_{sanitized_action_str}_{sanitized_cell_name_str}_px{px_size}.png"
        
        temp_png_path = os.path.join(temp_dir_base_path, img_basename) # Save to a common temp path
        final_png_path = os.path.join(output_dirs_for_type["notitle_img_dir"], img_basename)

        # SVG parameters
        label_fs_for_svg = max(50, int(px_size / 20.0)) # Dynamic label font size
        min_cell_svg_size = 30
        # Estimate cell size for SVG based on svg_render_size_hint and grid dimensions
        est_max_dim_for_svg = max(num_rows, num_cols, 1) * 1.1 
        calc_cell_svg_size = max(1, int(svg_render_size_hint / est_max_dim_for_svg if est_max_dim_for_svg > 0 else svg_render_size_hint))
        actual_cell_svg_size = max(min_cell_svg_size, calc_cell_svg_size)
        actual_cell_svg_spacing = max(3, actual_cell_svg_size // 12)

        svg_content = generate_patterned_grid_svg(
            num_rows, num_cols, 7, # Pattern type 7
            actual_cell_svg_size, actual_cell_svg_spacing,
            shape_type_str, shape_color_str,
            anomalous_cell_tuple, anomaly_details_dict,
            True, label_fs_for_svg
        )
        if not svg_content: progress_bar_obj.update(1); continue

        current_png_scale = png_quality_val * (px_size / float(svg_render_size_hint if svg_render_size_hint > 0 else px_size))
        if not grid_utils.svg_to_png_direct(svg_content, temp_png_path, scale=current_png_scale, output_size=px_size):
            progress_bar_obj.update(1); continue
        
        try:
            os.makedirs(os.path.dirname(final_png_path), exist_ok=True)
            shutil.copy2(temp_png_path, final_png_path)
        except Exception as e:
            print(f"  ERROR copying {temp_png_path} to {final_png_path}: {e}")
            progress_bar_obj.update(1); continue
        
        progress_bar_obj.update(1) # Successful image

        # Metadata
        img_path_rel = os.path.join("images", img_basename).replace("\\", "/")
        # Topic name for metadata, e.g. "Dice Patterned Grid"
        metadata_topic_name = grid_type_identifier.replace('_', ' ').title() 
        
        common_meta_payload = {
            "group_id": group_id_val, "action_type": action_verb_str,
            "grid_type": grid_type_identifier, # "dice_patterned_grid" or "tally_patterned_grid"
            "grid_size": grid_dims_str,
            "anomalous_cell_algebraic": display_cell_name,
            "standard_count_in_cell": standard_count_at_anomalous_cell,
            "modified_count_in_cell": modified_count_at_anomalous_cell,
            "pattern_id_used": 7,
            "pixel_size": px_size,
            # Add anomaly_details_dict directly, or unpack its relevant keys
            "anomaly_operation": anomaly_details_dict.get('operation'),
        }
        if 'shape_index_if_dice' in anomaly_details_dict:
            common_meta_payload['anomaly_target_shape_index'] = anomaly_details_dict['shape_index_if_dice']
        if 'new_shape_type' in anomaly_details_dict: # For replace
            common_meta_payload['anomaly_replaced_with_shape'] = anomaly_details_dict['new_shape_type']


        for q_idx, prompt_txt in enumerate(prompts_q1_q2):
            q_lbl = f"Q{q_idx+1}"; meta_id_str = f"{grid_type_identifier}_{group_id_str_fmt}_{sanitized_action_str}_{sanitized_cell_name_str}_px{px_size}_{q_lbl}"
            generated_metadata_list.append({
                "ID": meta_id_str, "image_path": img_path_rel, "topic": metadata_topic_name,
                "prompt": prompt_txt, "ground_truth": ground_truth_q1_q2, "expected_bias": expected_bias_q1_q2,
                "with_title": False, "type_of_question": q_lbl, "pixel": px_size,
                "metadata": common_meta_payload.copy()
            })
        
        meta_id_q3_str = f"{grid_type_identifier}_{group_id_str_fmt}_{sanitized_action_str}_{sanitized_cell_name_str}_px{px_size}_Q3"
        generated_metadata_list.append({
            "ID": meta_id_q3_str, "image_path": img_path_rel, "topic": metadata_topic_name,
            "prompt": prompt_q3, "ground_truth": ground_truth_q3, "expected_bias": expected_bias_q3,
            "with_title": False, "type_of_question": "Q3", "pixel": px_size,
            "metadata": common_meta_payload.copy()
        })
    return generated_metadata_list


if __name__ == '__main__':
    print("Testing Patterned Grid Generator directly...")
    # This requires grid_utils.py to be importable from parent or same dir
    # And will create output directories in the current working directory.
    if not os.path.exists("vlms-are-biased-notitle"): os.makedirs("vlms-are-biased-notitle")
    create_grid_dataset(quality_scale=5.0, svg_size=800)
    print("\nDirect test of Patterned Grid Generator complete.")