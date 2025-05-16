# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import string
import numpy as np
import os
import random
import json
import re
import sys
import PIL.Image
from PIL import ImageDraw, ImageFont
import io
import csv
import pandas as pd
import time
from tqdm import tqdm
import cairosvg
import textwrap
import math # For star shape in draw_shape
import shutil # For copying files

# --- Global Constants ---
PIXEL_SIZES = [384, 768, 1152]
NUM_ACTION_GROUPS = 14 # Number of unique ID groups (and cell selections)
ACTIONS_PER_GROUP = 4 # Dice-Remove, Dice-Replace, Tally-Remove, Tally-Add
MIN_GRID_DIM = 6
MAX_GRID_DIM = 12

# --- Helper Functions ---
def sanitize_filename(name):
    """Removes potentially problematic characters for filenames/foldernames."""
    name = str(name)
    name = name.replace(' ', '_').replace(':', '_').replace('/', '_').replace('\\', '_')
    name = re.sub(r'[^\w\-]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name if name else "sanitized_empty"

def svg_to_png_direct(svg_content, output_path, scale=2.0, output_size=768):
    """Convert SVG content directly to PNG file without saving SVG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=output_path,
            scale=scale,
            background_color="white", # Ensure background is white
            output_width=output_size,
            output_height=output_size
        )
        return True
    except Exception as e:
        print(f"Error converting SVG to PNG ({output_path}): {e}")
        if 'failed to load external entity' in str(e).lower():
             print("Hint: Check if SVG content is valid or if file paths within SVG are correct.")
        elif 'cairosvg' in str(e).lower():
             print("Hint: Ensure cairosvg and its dependencies (like Cairo) are correctly installed.")
        elif not svg_content or not svg_content.strip():
            print("Hint: SVG content appears to be empty.")
        return False

# --- Grid Generation Functions ---
def get_pattern_count(row, col, pattern_type, n_rows, n_cols):
    """Calculate how many shapes should be in a cell based on pattern type."""
    if n_cols <= 0: return 0
    if pattern_type == 7:
        distance_from_edge = min(col, n_cols - 1 - col)
        count = distance_from_edge + 1
        return int(count)
    else:
        print(f"Warning: Unknown pattern_type {pattern_type}. Returning 1.")
        return 1


def generate_grid_svg(rows, cols, pattern_type=7, cell_size=80, cell_spacing=10,
                      shape_type='circle', shape_color='black',
                      exception_cell=None, exception_change=None,
                      exception_shape_index=0, show_labels=True,
                      label_fontsize=12):
    """Generate a grid SVG with specified pattern and optional exception."""
    label_margin_factor = 2.0
    left_margin = (label_fontsize * label_margin_factor * 1.5) if show_labels else cell_spacing
    top_margin = cell_spacing
    bottom_margin = (label_fontsize * label_margin_factor * 1.2) if show_labels else cell_spacing
    grid_width = cols * (cell_size + cell_spacing) - cell_spacing
    grid_height = rows * (cell_size + cell_spacing) - cell_spacing
    total_width = grid_width + left_margin + cell_spacing
    total_height = grid_height + top_margin + bottom_margin
    svg = f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">\n'
    svg += f'<rect width="{total_width}" height="{total_height}" fill="white"/>\n' # White background

    # Draw Column Labels
    if show_labels:
        label_y_pos = top_margin + grid_height + bottom_margin * 0.6
        for c in range(cols):
            col_label = string.ascii_uppercase[c % 26]
            if c >= 26: col_label = string.ascii_uppercase[(c // 26) - 1] + col_label
            label_x_pos = left_margin + c * (cell_size + cell_spacing) + cell_size / 2
            svg += f'<text x="{label_x_pos}" y="{label_y_pos}" text-anchor="middle" dominant-baseline="middle" '
            svg += f'font-family="Arial, sans-serif" font-size="{label_fontsize}" fill="black" font-weight="bold">{col_label}</text>\n'

    # Draw Row Labels
    if show_labels:
         label_x_pos = left_margin * 0.4
         for r in range(rows):
            row_label = str(r + 1)
            label_y_pos = top_margin + r * (cell_size + cell_spacing) + cell_size / 2
            svg += f'<text x="{label_x_pos}" y="{label_y_pos}" text-anchor="middle" dominant-baseline="middle" '
            svg += f'font-family="Arial, sans-serif" font-size="{label_fontsize}" fill="black" font-weight="bold">{row_label}</text>\n'

    # Draw Grid Cells and Shapes
    for r in range(rows):
        for c in range(cols):
            cell_x = left_margin + c * (cell_size + cell_spacing)
            cell_y = top_margin + r * (cell_size + cell_spacing)
            svg += f'<rect x="{cell_x}" y="{cell_y}" width="{cell_size}" height="{cell_size}" fill="none" stroke="black" stroke-width="1"/>\n'
            shapes_count_standard = get_pattern_count(r, c, pattern_type, rows, cols)
            is_exception = exception_cell and r == exception_cell[0] and c == exception_cell[1]
            effective_exception_change = None
            effective_exception_index = exception_shape_index

            if is_exception and exception_change:
                 if 'remove' in exception_change and exception_change['remove']:
                     # Check validity for remove (count>0 and index valid IF applicable, tally doesn't use index)
                     is_valid_remove = shapes_count_standard > 0
                     if shape_type != 'tally': # Dice/other shapes need valid index
                         is_valid_remove = is_valid_remove and (0 <= effective_exception_index < shapes_count_standard)
                     if is_valid_remove:
                         effective_exception_change = {'remove': True}
                     else: is_exception = False # Invalidate exception if remove isn't possible
                 elif 'add' in exception_change and exception_change['add']:
                     effective_exception_change = {'add': True} # Add is always conceptually possible
                 elif 'modify' in exception_change and exception_change['modify']:
                     # Check validity for modify (count>0 and index valid)
                     if shapes_count_standard > 0 and 0 <= effective_exception_index < shapes_count_standard:
                         effective_exception_change = {'modify': True,
                                                       'type': exception_change.get('type', shape_type),
                                                       'color': exception_change.get('color', shape_color)}
                     else: is_exception = False # Invalidate exception if modify isn't possible

            # Draw if standard count > 0 OR adding to an empty cell
            should_draw = shapes_count_standard > 0 or (is_exception and effective_exception_change and 'add' in effective_exception_change)
            if should_draw:
                if shape_type == 'tally':
                    svg += draw_tally_marks(cell_x, cell_y, cell_size, shapes_count_standard,
                                            is_exception, effective_exception_index, # Pass index, though not used by tally draw
                                            effective_exception_change, shape_color)
                else: # Dice, etc.
                     svg += draw_shapes(cell_x, cell_y, cell_size, shapes_count_standard,
                                        shape_type, shape_color,
                                        is_exception, effective_exception_index, # Pass the specific index
                                        effective_exception_change)
    svg += '</svg>'
    return svg

def draw_tally_marks(x, y, cell_size, original_value, is_exception, exception_index, exception_change, color='black'):
    """Draw realistic tally marks, applying add/remove exception logic."""
    svg = ''
    padding = cell_size * 0.15
    line_width = max(1.5, cell_size * 0.035)
    line_height = cell_size - 2 * padding
    available_width = cell_size - 2 * padding

    # Determine the final number of lines
    final_value = original_value
    if is_exception and exception_change:
        # Tally only supports 'remove' and 'add' exceptions
        if 'remove' in exception_change and original_value > 0:
            final_value -= 1
        elif 'add' in exception_change:
            final_value += 1
    final_value = max(0, final_value)

    if final_value <= 0: return svg

    # Layout Parameters
    line_internal_spacing = available_width * 0.1
    spacing_after_group = line_internal_spacing * 1.5
    group_width = 4 * line_internal_spacing

    # Generate Lines
    lines_to_draw = []
    current_x = x + padding
    num_full_groups = final_value // 5
    num_remaining_singles = final_value % 5

    # Draw full groups
    for g in range(num_full_groups):
        group_start_x = current_x
        for i in range(4):
            line_x = group_start_x + i * line_internal_spacing
            lines_to_draw.append({'type': 'vertical', 'x1': line_x, 'y1': y + padding, 'x2': line_x, 'y2': y + padding + line_height})
        diag_x1 = group_start_x - line_internal_spacing * 0.2
        diag_x2 = group_start_x + 3 * line_internal_spacing + line_internal_spacing * 0.2
        diag_y1 = y + padding + line_height * 0.1
        diag_y2 = y + padding + line_height * 0.9
        lines_to_draw.append({'type': 'diagonal', 'x1': diag_x1, 'y1': diag_y1, 'x2': diag_x2, 'y2': diag_y2})
        current_x = group_start_x + group_width + spacing_after_group

    # Draw remaining singles
    singles_start_x = current_x
    for i in range(num_remaining_singles):
        line_x = singles_start_x + i * line_internal_spacing
        lines_to_draw.append({'type': 'vertical', 'x1': line_x, 'y1': y + padding, 'x2': line_x, 'y2': y + padding + line_height})
        current_x = line_x + line_internal_spacing

    # Draw lines to SVG
    for line in lines_to_draw:
        svg += f'<line x1="{line["x1"]:.2f}" y1="{line["y1"]:.2f}" x2="{line["x2"]:.2f}" y2="{line["y2"]:.2f}" '
        svg += f'stroke="{color}" stroke-width="{line_width:.2f}" stroke-linecap="round"/>\n'
    return svg


def draw_shapes(x, y, cell_size, original_count, shape_type, color, is_exception, exception_index, exception_change):
    """Draw shapes, applying exception logic using the provided index."""
    svg = ''
    padding = cell_size * 0.15
    available_size = cell_size - (2 * padding)
    base_shape_size = available_size * 0.3
    max_shapes_before_shrink = 6

    # Determine count for scaling based on final number of shapes
    count_for_scaling = original_count
    valid_exception = False
    if is_exception and exception_change:
        if 'remove' in exception_change and original_count > 0 and 0 <= exception_index < original_count:
             count_for_scaling -= 1
             valid_exception = True
        elif 'add' in exception_change:
            count_for_scaling += 1
            valid_exception = True
        elif 'modify' in exception_change and original_count > 0 and 0 <= exception_index < original_count:
            valid_exception = True # Count doesn't change for scaling
    count_for_scaling = max(1, count_for_scaling)

    # Calculate shape size based on final count
    scale_factor = min(1.0, math.sqrt(max_shapes_before_shrink / count_for_scaling)) if count_for_scaling > max_shapes_before_shrink else 1.0
    shape_size = max(5, base_shape_size * scale_factor)

    # Get base positions based on original count
    positions = get_dice_positions(original_count, x, y, cell_size, padding, shape_size)
    initial_shapes = [{'index': i, 'x': px, 'y': py, 'size': shape_size, 'type': shape_type, 'color': color}
                      for i, (px, py) in enumerate(positions)]

    final_shapes_to_draw = []

    # Apply Exception Logic ONLY if it's a valid exception for this cell/index
    if is_exception and valid_exception:
        if 'remove' in exception_change:
            final_shapes_to_draw = [s for s in initial_shapes if s['index'] != exception_index]
        elif 'modify' in exception_change:
            for shape in initial_shapes:
                if shape['index'] == exception_index:
                    shape['type'] = exception_change.get('type', shape_type)
                    shape['color'] = exception_change.get('color', color)
                final_shapes_to_draw.append(shape)
        elif 'add' in exception_change:
            final_shapes_to_draw.extend(initial_shapes)
            next_pos = get_next_dice_position(original_count, positions, x, y, cell_size, padding, shape_size)
            if next_pos:
                added_shape = {'index': original_count, 'x': next_pos[0], 'y': next_pos[1], 'size': shape_size,
                               'type': exception_change.get('add_type', shape_type),
                               'color': exception_change.get('add_color', color)}
                final_shapes_to_draw.append(added_shape)
        else: # Should not happen if valid_exception is True
            final_shapes_to_draw.extend(initial_shapes)
    else: # No exception or invalid exception
        final_shapes_to_draw.extend(initial_shapes)

    # Draw the final shapes
    for shape in final_shapes_to_draw:
        svg += draw_shape(shape['x'], shape['y'], shape['size'], shape['type'], shape['color'])
    return svg

def get_dice_positions(count, x, y, cell_size, padding, shape_size):
    """Get positions for shapes in standard dice pattern or grid."""
    if count <= 0: return []
    draw_area_x, draw_area_y = x + padding, y + padding
    draw_area_width = max(0, cell_size - 2 * padding)
    draw_area_height = max(0, cell_size - 2 * padding)
    center_x, center_y = draw_area_x + draw_area_width / 2, draw_area_y + draw_area_height / 2
    pos_c = (center_x - shape_size / 2, center_y - shape_size / 2)
    pos_tl = (draw_area_x, draw_area_y)
    pos_tr = (draw_area_x + draw_area_width - shape_size, draw_area_y)
    pos_bl = (draw_area_x, draw_area_y + draw_area_height - shape_size)
    pos_br = (draw_area_x + draw_area_width - shape_size, draw_area_y + draw_area_height - shape_size)
    pos_cl = (draw_area_x, center_y - shape_size / 2)
    pos_cr = (draw_area_x + draw_area_width - shape_size, center_y - shape_size / 2)

    patterns = {
        1: [pos_c], 2: [pos_tl, pos_br], 3: [pos_tl, pos_c, pos_br],
        4: [pos_tl, pos_tr, pos_bl, pos_br], 5: [pos_tl, pos_tr, pos_c, pos_bl, pos_br],
        6: [pos_tl, pos_tr, pos_cl, pos_cr, pos_bl, pos_br]
    }
    if count in patterns: return patterns[count]
    else: # Grid layout
        positions = []
        grid_cols = max(1, int(np.ceil(np.sqrt(count))))
        grid_rows = max(1, int(np.ceil(count / grid_cols)))
        if grid_cols > 1 and grid_rows * (grid_cols - 1) >= count:
            grid_cols -= 1; grid_rows = max(1, int(np.ceil(count / grid_cols)))
        spacing_x = (draw_area_width - shape_size) / (grid_cols - 1) if grid_cols > 1 else 0
        spacing_y = (draw_area_height - shape_size) / (grid_rows - 1) if grid_rows > 1 else 0
        grid_w = (grid_cols - 1) * spacing_x + shape_size if grid_cols > 0 else 0
        grid_h = (grid_rows - 1) * spacing_y + shape_size if grid_rows > 0 else 0
        start_x = draw_area_x + (draw_area_width - grid_w) / 2
        start_y = draw_area_y + (draw_area_height - grid_h) / 2
        item_idx = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                if item_idx < count:
                    px = start_x + c * spacing_x if grid_cols > 1 else start_x
                    py = start_y + r * spacing_y if grid_rows > 1 else start_y
                    positions.append((px, py)); item_idx += 1
                else: break
            if item_idx >= count: break
        return positions

def get_next_dice_position(count, existing_positions, x, y, cell_size, padding, shape_size):
    """Get the next logical position for adding a shape."""
    next_count = count + 1
    all_positions_next = get_dice_positions(next_count, x, y, cell_size, padding, shape_size)
    if not all_positions_next:
         center_x, center_y = x + cell_size / 2, y + cell_size / 2
         print(f"Warning: Could not generate positions for {next_count} shapes. Defaulting to center.")
         return (center_x - shape_size / 2, center_y - shape_size / 2)
    existing_pos_set = set((round(p[0], 2), round(p[1], 2)) for p in existing_positions)
    for pos in all_positions_next:
        rounded_pos = (round(pos[0], 2), round(pos[1], 2))
        if rounded_pos not in existing_pos_set: return pos
    print(f"Warning: Could not find distinct next position for count {next_count}. Using last generated.")
    return all_positions_next[-1]

def draw_shape(x, y, size, shape_type, color):
    """Draw a single shape."""
    svg = ''
    center_x, center_y, radius = x + size / 2, y + size / 2, size / 2
    if size <= 0: return ""
    draw_radius, draw_side = max(0.1, radius * 0.9), max(0.1, size * 0.9)

    if shape_type == 'circle':
        svg += f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="{draw_radius:.2f}" fill="{color}"/>\n'
    elif shape_type == 'square':
        sq_x, sq_y = center_x - draw_side / 2, center_y - draw_side / 2
        svg += f'<rect x="{sq_x:.2f}" y="{sq_y:.2f}" width="{draw_side:.2f}" height="{draw_side:.2f}" fill="{color}"/>\n'
    elif shape_type == 'triangle':
        h = max(0.1, draw_radius * 1.5); w = h / (math.sqrt(3)/2) * 1.0
        p1 = (center_x, center_y - draw_radius)
        p2 = (center_x - w / 2, center_y + draw_radius * 0.5)
        p3 = (center_x + w / 2, center_y + draw_radius * 0.5)
        svg += f'<polygon points="{p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}" fill="{color}"/>\n'
    elif shape_type == 'star':
        outer_r = max(0.1, radius * 0.95); inner_r = outer_r * 0.38; pts = 5
        svg += '<polygon points="'
        all_pts = []
        for i in range(pts * 2):
            curr_r = outer_r if i % 2 == 0 else inner_r
            angle = math.pi * i / pts - (math.pi / 2)
            px, py = center_x + curr_r * math.cos(angle), center_y + curr_r * math.sin(angle)
            all_pts.append(f"{px:.2f},{py:.2f}")
        svg += " ".join(all_pts); svg += f'" fill="{color}"/>\n'
    else:
        print(f"Warning: Unknown shape_type '{shape_type}'. Fallback circle.");
        svg += f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="{max(0.1, radius * 0.5):.2f}" fill="black"/>\n'
    return svg

# --- Selection of Actions ---
def select_edge_cell(rows, cols, previously_selected=None):
    """Select a cell avoiding edges/near-edges."""
    if previously_selected is None: previously_selected = set()
    ur1, uc1, ur2, uc2, ur3, uc3 = set(), set(), set(), set(), set(), set()
    if rows > 7 and cols > 7: ur1={0,1,2,rows-3,rows-2,rows-1}; uc1={0,1,2,cols-3,cols-2,cols-1}
    if rows > 4 and cols > 4: ur2={0,1,rows-2,rows-1}; uc2={0,1,cols-2,cols-1}
    if rows > 1 and cols > 1: ur3={0,rows-1}; uc3={0,cols-1}
    def get_available(ur, uc, prev):
        pot = [(r,c) for r in range(rows) for c in range(cols) if r not in ur and c not in uc]
        avail = [cell for cell in pot if cell not in prev]; return avail, pot
    cur_ur, cur_uc = ur3, uc3
    if rows > 4 and cols > 4: cur_ur, cur_uc = ur2, uc2
    if rows > 7 and cols > 7: cur_ur, cur_uc = ur1, uc1
    tiers = [(cur_ur, cur_uc, "Primary")]
    if (cur_ur, cur_uc) == (ur1, uc1): tiers.extend([(ur2, uc2, "Fallback T2"), (ur3, uc3, "Fallback T3")])
    elif (cur_ur, cur_uc) == (ur2, uc2): tiers.append((ur3, uc3, "Fallback T3"))
    for ur, uc, name in tiers:
        avail, pot = get_available(ur, uc, previously_selected)
        if avail: return random.choice(avail)
        if pot: print(f"Warning: Reusing '{name}' cell for {rows}x{cols}."); return random.choice(pot)
    print(f"Warning: Trying Tier 4 (any) selection for {rows}x{cols}.")
    all_c = [(r,c) for r in range(rows) for c in range(cols)]
    avail_any = [cell for cell in all_c if cell not in previously_selected]
    if avail_any: return random.choice(avail_any)
    if all_c: print(f"CRITICAL WARNING: Reusing *any* cell for {rows}x{cols}."); return random.choice(all_c)
    print(f"ERROR: Cannot select cell for {rows}x{cols}."); return None

# --- Create Directory Structure Function ---
def create_directory_structure():
    """Creates the directory structure."""
    base = "vlms-are-biased-notitle"
    dirs_to_create = {
        "notitle_base_dir": base,
        "notitle_dice_dir": os.path.join(base, "dice"),
        "notitle_dice_img_dir": os.path.join(base, "dice", "images"),
        "notitle_tally_dir": os.path.join(base, "tally"),
        "notitle_tally_img_dir": os.path.join(base, "tally", "images"),
        "temp_dir": "temp_grid_output"
    }
    print("Creating directories...")
    for name, path in dirs_to_create.items():
        try:
            os.makedirs(path, exist_ok=True); print(f"  Ensured: {path}")
        except OSError as e: print(f"ERROR creating {path}: {e}"); raise
    return dirs_to_create

# --- Single Action Generation Function ---
def generate_single_action_image(group_id, grid_type, action_type,
                               dirs, shape_type, shape_color,
                               pixel_sizes, quality_scale, svg_size,
                               rows, cols, cell, target_shape_index,
                               progress_bar):
    """
    Generates images and metadata for a single action within a group.
    Modification details are stored directly in metadata.
    """
    metadata_entries = []
    row, col = cell
    display_row, display_col = row + 1, string.ascii_uppercase[col % 26]
    if col >= 26: display_col = string.ascii_uppercase[(col // 26) - 1] + display_col
    cell_name = f"{display_col}{display_row}"

    standard_count = get_pattern_count(row, col, 7, rows, cols)
    # exception_change holds the modification details (e.g., {'remove': True} or {'modify': True, 'type': 'square'})
    exception_change, modified_count, is_action_valid = None, standard_count, False
    action_desc_raw = f"Group {group_id}, Action: {grid_type} {action_type}"
    shape_index_to_use = target_shape_index

    # --- Apply Action Logic & Check Validity ---
    if action_type == 'remove':
        action_desc_raw += f" index {shape_index_to_use} in {cell_name}"
        is_valid_remove = standard_count > 0
        if grid_type != 'tally': is_valid_remove &= (0 <= shape_index_to_use < standard_count)
        if is_valid_remove:
            modified_count = standard_count - 1
            exception_change = {'remove': True}; is_action_valid = True # Details: remove=True
        else: action_desc_raw += " (no op - invalid)"
    elif action_type == 'replace': # Dice only
        action_desc_raw += f" index {shape_index_to_use} in {cell_name}"
        if standard_count > 0 and 0 <= shape_index_to_use < standard_count:
            modified_count = standard_count - 1 # Count - 1
            target_shape = random.choice([s for s in ['square', 'triangle', 'star'] if s != shape_type])
            # Details: modify=True, type=target_shape, color=shape_color
            exception_change = {'modify': True, 'type': target_shape, 'color': shape_color}
            action_desc_raw += f" with {target_shape}"; is_action_valid = True
        else: action_desc_raw += " (no op - invalid)"
    elif action_type == 'add': # Tally only
        action_desc_raw += f" to {cell_name}"
        modified_count = standard_count + 1
        exception_change = {'add': True}; shape_index_to_use = -1; is_action_valid = True # Details: add=True
    else: print(f"Error: Unknown action '{action_type}' G{group_id}"); progress_bar.update(len(pixel_sizes)); return []

    if not is_action_valid:
        print(f"Warning: Action '{action_type}' invalid G{group_id} C{cell_name} (std={standard_count}, idx={shape_index_to_use}). Skipping.")
        progress_bar.update(len(pixel_sizes)); return []
    modified_count = max(0, modified_count)

    # --- Prompts & Answers (Unchanged) ---
    prompts, gt_prompt12, bias_prompt12, gt_prompt3, bias_prompt3 = [], "", "", "", ""
    gt_prompt3 = "No"  # Always No
    bias_prompt3 = "Yes"  # Always Yes
    prompt_shape = "circles" if grid_type == 'dice' else "lines"
    prompt1 = f"How many {prompt_shape} are there in cell {cell_name}? Answer with a number in curly brackets, e.g., {{9}}."
    prompt2 = f"Count the {prompt_shape} in cell {cell_name}. Answer with a number in curly brackets, e.g., {{9}}."
    prompt3 = f"Does cell {cell_name} contain {standard_count} {prompt_shape}? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    prompts = [prompt1, prompt2, prompt3]
    gt_prompt12 = str(modified_count); 
    bias_prompt12 = str(standard_count)

    # --- Generate Images & Metadata Loop ---
    group_id_str = f"{group_id:03d}"
    grid_size_str = f"{rows}x{cols}"
    sanitized_action = sanitize_filename(action_type)
    cell_name_sanitized = sanitize_filename(cell_name)

    generation_successful_for_all_res = True
    for pixel_size in pixel_sizes:
         if not generation_successful_for_all_res: break # Skip remaining sizes if one failed

         filename_base = f"{grid_type}_{group_id_str}_{sanitized_action}_{cell_name_sanitized}_notitle_px{pixel_size}"
         filename = filename_base + ".png"
         temp_path = os.path.join(dirs["temp_dir"], filename)
         final_dir = dirs[f"notitle_{grid_type}_img_dir"]
         metadata_relative_path = f"images/{filename}"
         final_path = os.path.join(final_dir, filename)

         # Font & SVG params (Unchanged)
         label_fs = max(50, int(pixel_size / 20))
         min_cs, est_dim = 30, max(rows, cols, 1) * 1.1
         calc_cs = max(1, svg_size) // max(1, int(est_dim))
         cell_sz = max(min_cs, calc_cs)
         cell_sp = max(3, cell_sz // 12)

         # Generate SVG (Unchanged)
         svg_content = None
         try:
             svg_content = generate_grid_svg(rows, cols, 7, cell_sz, cell_sp, shape_type, shape_color,
                                             cell, exception_change, shape_index_to_use, True, label_fs)
         except Exception as e: print(f"SVG Error G{group_id} F:{filename}: {e}"); progress_bar.update(1); generation_successful_for_all_res = False; continue

         # Convert SVG->PNG & Copy (Unchanged)
         this_res_success = False
         adj_scale = quality_scale * (pixel_size / 768.0)
         if svg_to_png_direct(svg_content, temp_path, adj_scale, pixel_size):
             try:
                 os.makedirs(final_dir, exist_ok=True)
                 shutil.copy2(temp_path, final_path)
                 progress_bar.update(1); this_res_success = True

                 # <<< --- MODIFIED Metadata Block --- >>>
                 # Base metadata
                 common_meta = {
                     "group_id": group_id,
                     "action": action_type,
                     "grid_type": grid_type,
                     "grid_size": grid_size_str,
                     "modified_cell": cell_name,
                     "standard_count": standard_count,
                     "modified_count": modified_count,
                     "target_shape_index": shape_index_to_use if action_type in ['remove','replace'] else None,
                     "pattern_type": 7,
                     # "modification_details": exception_change, # REMOVED this line
                     "action_description": action_desc_raw,
                     "pixel": pixel_size
                 }
                 # Directly add keys from exception_change (if it exists)
                 if exception_change:
                     for key, value in exception_change.items():
                         common_meta[key] = value
                 # <<< --- End MODIFIED Metadata Block --- >>>


                 # Append metadata entries for prompts (Unchanged logic, uses modified common_meta)
                 for p_idx, prompt in enumerate(prompts):
                     gt = gt_prompt12 if p_idx < 2 else gt_prompt3
                     bias = bias_prompt12 if p_idx < 2 else bias_prompt3
                     meta_id = f"{grid_type}_{group_id_str}_{sanitized_action}_notitle_px{pixel_size}_prompt{p_idx+1}"
                     try: # Added try-except around append for robustness
                         metadata_entries.append({
                             "ID": meta_id, "image_path": metadata_relative_path,
                             "topic": f"{grid_type.capitalize()} Pattern", "prompt": prompt,
                             "ground_truth": gt, "expected_bias": bias,
                             "type_of_question": f"Q{p_idx+1}", "with_title": False,
                             "pixel": pixel_size,
                             "metadata": common_meta.copy() # Uses the modified common_meta
                         })
                     except Exception as meta_e:
                          print(f"Metadata Append Error G{group_id} F:{filename} P{p_idx+1}: {meta_e}")
                          # Potentially set generation_successful_for_all_res = False here if needed

             except Exception as e: print(f"Copy/Meta Error G{group_id} F:{final_path}: {e}"); this_res_success=False; generation_successful_for_all_res=False
         else: progress_bar.update(1); generation_successful_for_all_res = False # SVG->PNG failed
         if not this_res_success: generation_successful_for_all_res = False # Ensure flag is set if any step failed

    # Return only if all resolutions were successful (or modify logic if partial success is okay)
    # if not generation_successful_for_all_res:
    #     return [] # Discard partial results for this action if any size failed
    return metadata_entries

# --- Main Dataset Generation Function ---
def create_grid_dataset(quality_scale=5.0, svg_size=800):
    """Generates grid dataset with 14 groups, each with 4 actions."""
    dirs = create_directory_structure()
    print("===================================================================================")
    print("=== Grid Dataset Generator (Dice & Tally, 6x6-12x12) - GROUPED ACTIONS V3 ===")
    print(f"    - Group IDs: {NUM_ACTION_GROUPS} (001 to {NUM_ACTION_GROUPS:03d})")
    print(f"    - Actions per Group: {ACTIONS_PER_GROUP} (Dice:[Remove, Replace], Tally:[Remove, Add])")
    print(f"    - Targeting: Same cell per Group ID; Same index for Dice Remove/Replace.")
    print(f"    - Resolutions: {PIXEL_SIZES}")
    print(f"    - Output: NO-TITLE images; filenames like dice_001_remove...")
    total_images = NUM_ACTION_GROUPS * ACTIONS_PER_GROUP * len(PIXEL_SIZES)
    print(f"    - Total Expected Images: {total_images}")
    print("===================================================================================")

    # Grid sizes for groups
    target_dims = list(range(MIN_GRID_DIM, MAX_GRID_DIM + 1))
    num_unique = len(target_dims)
    assigned_sizes = [(target_dims[i % num_unique], target_dims[i % num_unique]) for i in range(NUM_ACTION_GROUPS)]

    all_dice_meta, all_tally_meta, selected_cells = [], [], {}
    progress = tqdm(total=total_images, desc="Processing Groups", unit="image", ncols=100)

    for group_idx in range(NUM_ACTION_GROUPS):
        group_id = group_idx + 1
        rows, cols = assigned_sizes[group_idx]
        grid_key = (rows, cols)
        prev_sel = selected_cells.get(grid_key, set())
        cell = select_edge_cell(rows, cols, prev_sel)
        if cell is None: print(f"FATAL: No cell for G{group_id} ({rows}x{cols}). Skip."); progress.update(ACTIONS_PER_GROUP * len(PIXEL_SIZES)); continue
        if grid_key not in selected_cells: selected_cells[grid_key] = set()
        selected_cells[grid_key].add(cell)

        # Determine target index for Dice actions (Remove/Replace) in this group
        std_count_dice = get_pattern_count(cell[0], cell[1], 7, rows, cols)
        dice_idx = random.randint(0, std_count_dice - 1) if std_count_dice > 0 else -1

        # Define the 4 actions for this group
        group_actions = [
            {'type': 'dice', 'action': 'remove', 'shape': 'circle', 'index': dice_idx},
            {'type': 'dice', 'action': 'replace', 'shape': 'circle', 'index': dice_idx},
            {'type': 'tally', 'action': 'remove', 'shape': 'tally', 'index': -1}, # Tally remove uses count logic, not index
            {'type': 'tally', 'action': 'add', 'shape': 'tally', 'index': -1},    # Tally add uses count logic, not index
        ]

        # Run generation for each action
        for cfg in group_actions:
            meta = generate_single_action_image(group_id, cfg['type'], cfg['action'], dirs, cfg['shape'], 'black',
                                                PIXEL_SIZES, quality_scale, svg_size, rows, cols, cell, cfg['index'], progress)
            if cfg['type'] == 'dice': all_dice_meta.extend(meta)
            else: all_tally_meta.extend(meta)

    progress.close()
    if progress.n < progress.total: print(f"Warning: Progress bar {progress.n}/{progress.total}. Check logs.")

    # Write Metadata
    print("\n--- Writing Metadata ---")
    try:
        if all_dice_meta: save_metadata_files(all_dice_meta, dirs["notitle_dice_dir"], "dice_notitle")
        if all_tally_meta: save_metadata_files(all_tally_meta, dirs["notitle_tally_dir"], "tally_notitle")
        combined = all_dice_meta + all_tally_meta
        if combined: save_metadata_files(combined, dirs["notitle_base_dir"], "combined_notitle")
    except Exception as e: print(f"ERROR writing metadata: {e}"); import traceback; traceback.print_exc()

    # Final Summary
    print("\n--- Summary ---")
    final_dice, final_tally = [], []
    try:
        if os.path.exists(dirs["notitle_dice_img_dir"]): final_dice = [f for f in os.listdir(dirs["notitle_dice_img_dir"]) if f.endswith('.png')]
        if os.path.exists(dirs["notitle_tally_img_dir"]): final_tally = [f for f in os.listdir(dirs["notitle_tally_img_dir"]) if f.endswith('.png')]
    except OSError as e: print(f"Warning: Could not count final files: {e}")
    total_actual = len(final_dice) + len(final_tally)
    print(f"Generated dice images: {len(final_dice)}")
    print(f"Generated tally images: {len(final_tally)}")
    print(f"Generated {total_actual} total PNG images (Expected: {total_images}).")
    print(f"Generated metadata entries:")
    print(f"  - Dice: {len(all_dice_meta)}")
    print(f"  - Tally: {len(all_tally_meta)}")
    print(f"\nDirectory structure created in: {dirs['notitle_base_dir']}")

    # Cleanup Temp
    try:
        if os.path.exists(dirs["temp_dir"]): shutil.rmtree(dirs["temp_dir"]); print(f"\nTemp directory {dirs['temp_dir']} cleaned up.")
    except Exception as e: print(f"Warning: Failed cleanup '{dirs['temp_dir']}': {e}")

def save_metadata_files(metadata_list, output_dir, base_name):
     """Saves metadata list to JSON and CSV files."""
     if not metadata_list: print(f"  No metadata for {base_name}, skipping."); return
     try: os.makedirs(output_dir, exist_ok=True)
     except OSError as e: print(f"  ERROR creating {output_dir}: {e}"); return

     # JSON
     json_path = os.path.join(output_dir, f"{base_name}_metadata.json")
     try:
         with open(json_path, 'w', encoding='utf-8') as f: json.dump(metadata_list, f, indent=4, ensure_ascii=False)
         print(f"  Wrote JSON: {json_path} ({len(metadata_list)} entries)")
     except (IOError, TypeError) as e: print(f"  ERROR writing JSON {json_path}: {e}")

     # CSV
     csv_path = os.path.join(output_dir, f"{base_name}_metadata.csv")
     try:
         flat_meta = []
         for entry in metadata_list:
             flat = entry.copy(); meta_dict = flat.pop('metadata', {})
             for k, v in meta_dict.items(): flat[f'meta_{k}'] = json.dumps(v) if isinstance(v, (dict, list)) else v
             flat_meta.append(flat)
         if not flat_meta: print(f"  No data for CSV {csv_path}."); return
         df = pd.DataFrame(flat_meta)
         pref_cols = ['ID', 'image_path', 'topic', 'prompt', 'ground_truth', 'expected_bias', 'type_of_question', 'with_title']
         meta_cols = sorted([c for c in df.columns if c.startswith('meta_')])
         ordered_cols = []; remaining = list(df.columns)
         for col in pref_cols + meta_cols:
             if col in remaining: ordered_cols.append(col); remaining.remove(col)
         ordered_cols.extend(sorted(remaining))
         df = df[ordered_cols]
         df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8')
         print(f"  Wrote CSV: {csv_path} ({len(metadata_list)} entries)")
     except Exception as e: print(f"  ERROR writing CSV {csv_path}: {e}"); import traceback; traceback.print_exc()

# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate grid dataset with GROUPED actions sharing same Group ID/cell/index.')
    parser.add_argument('--quality', type=float, default=5.0, help='PNG quality scale (default: 5.0)')
    parser.add_argument('--svg-size', type=int, default=800, help='SVG base size hint (default: 800)')
    args = parser.parse_args()

    print("Starting Grid Dataset Generation...")
    start_time = time.time()
    try:
        create_grid_dataset(quality_scale=args.quality, svg_size=args.svg_size)
    except Exception as main_err:
        print(f"\n!!! FATAL ERROR: {main_err} !!!"); import traceback; traceback.print_exc()
    finally:
        end_time = time.time()
        print(f"\nGeneration finished in {end_time - start_time:.2f} seconds.")
        print("========================= Complete ==========================")