# generators/xiangqi_board_generator.py
# -*- coding: utf-8 -*-
"""
Xiangqi (Chinese Chess) Board Generator - Generates "notitle" images of Xiangqi boards
with variations in rows/columns, and includes General/Advisor pieces.
"""
import os
import shutil
import svgwrite
from tqdm import tqdm
import sys
import matplotlib.font_manager as fm # For CJK font detection

import grid_utils # Common utilities for board generators

# --- Constants for Xiangqi Board ---
STANDARD_BOARD_ROWS = 10 # Standard Xiangqi board has 10 horizontal lines (ranks 0-9)
STANDARD_BOARD_COLS = 9  # Standard Xiangqi board has 9 vertical lines (files 0-8)
BOARD_TYPE_NAME = "Xiangqi Board"
BOARD_ID = "xiangqi_board"
PIXEL_SIZES = [384, 768, 1152]

# Piece characters for drawing (subset needed for Generals/Advisors)
XIANGQI_PIECES_CHARS = {
    'k': '將', 'a': '士', # Black
    'K': '帥', 'A': '仕'  # Red
}

# --- CJK Font Detection ---
def detect_cjk_fonts():
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        cjk_keywords = ['simsun', 'uming', 'ukai', 'mingliu', 'pmingliu', 'dfkai-sb', 'noto sans cjk', 'noto serif cjk', 'source han sans', 'source han serif', 'microsoft yahei', 'msyh', 'dengxian', 'simhei', 'fangsong', 'kaiti', 'wenquanyi', 'wqy', 'ar pl', 'nanum', 'malgun gothic', 'gulim', 'batang', 'dotum', 'meiryo', 'ms gothic', 'ms mincho', 'hiragino', 'heiti tc', 'songti sc']
        normalized_available_fonts = {f.lower().replace(" ", ""): f for f in available_fonts}
        cjk_fonts_found = set()
        for norm_font_name, original_font_name in normalized_available_fonts.items():
            if any(keyword in norm_font_name for keyword in cjk_keywords):
                cjk_fonts_found.add(original_font_name)
        return sorted(list(cjk_fonts_found))
    except Exception as e:
        print(f"  Warning: Error detecting CJK fonts: {e}"); return []

def get_best_cjk_font():
    detected_fonts = detect_cjk_fonts()
    if not detected_fonts: print("  No CJK fonts detected by font manager."); return None
    preferred_order = ['SimSun', 'Noto Sans CJK SC', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Source Han Sans SC', 'Arial Unicode MS']
    normalized_preferred = {p.lower().replace(" ", ""): p for p in preferred_order}
    for norm_pref, original_pref in normalized_preferred.items():
        for font_path_name in detected_fonts: # font_path_name is what matplotlib returns
            if norm_pref in font_path_name.lower().replace(" ", ""):
                print(f"  -> Using CJK font for Xiangqi Board: '{font_path_name}' (matched '{original_pref}')")
                return font_path_name
    fallback_font = detected_fonts[0]
    print(f"  Warning: No preferred CJK font for Xiangqi Board. Using fallback: '{fallback_font}'")
    return fallback_font

# --- XiangqiBoardGrid Class (Manages Dimensions, River, and Palace Pieces) ---
class XiangqiBoardGrid:
    def __init__(self, rows=STANDARD_BOARD_ROWS, cols=STANDARD_BOARD_COLS):
        self.rows = max(2, rows) 
        self.cols = max(1, cols) 
        self.river_row_idx = 4 # River is between rank 4 and 5 (0-indexed)
        self.river_row_idx = max(0, min(self.river_row_idx, self.rows - 2))
        
        self.pieces = {} # Stores {(file_idx, rank_idx): 'piece_symbol'}
        self._setup_palace_pieces()

    def _to_square_tuple(self, file_idx, rank_idx):
        if 0 <= file_idx < self.cols and 0 <= rank_idx < self.rows:
            return (file_idx, rank_idx)
        return None

    def set_piece_at_coords(self, file_idx, rank_idx, piece_symbol):
        sq_tuple = self._to_square_tuple(file_idx, rank_idx)
        if sq_tuple:
            self.pieces[sq_tuple] = piece_symbol

    def _setup_palace_pieces(self):
        self.pieces.clear()
        
        # Standard positions require at least 9 columns and 10 rows.
        if self.cols >= 9 and self.rows >= 10:
            # Black pieces (rank 0)
            black_rank = 0
            self.set_piece_at_coords(4, black_rank, 'k') # Black General
            self.set_piece_at_coords(3, black_rank, 'a') # Black Advisor
            self.set_piece_at_coords(5, black_rank, 'a') # Black Advisor
            
            # Red pieces (rank 9 on a 10-row board)
            red_rank = self.rows - 1 # Should be 9 for standard
            self.set_piece_at_coords(4, red_rank, 'K') # Red General
            self.set_piece_at_coords(3, red_rank, 'A') # Red Advisor
            self.set_piece_at_coords(5, red_rank, 'A') # Red Advisor
        elif self.cols >= 5 and self.rows >= 1: # Fallback for smaller boards (simplified placement)
            center_file_idx = (self.cols - 1) // 2
            self.set_piece_at_coords(center_file_idx, 0, 'k')
            self.set_piece_at_coords(center_file_idx, self.rows - 1, 'K')
            if center_file_idx > 0: # Check if space for advisors
                self.set_piece_at_coords(center_file_idx - 1, 0, 'a')
                self.set_piece_at_coords(center_file_idx - 1, self.rows - 1, 'A')
            if center_file_idx < self.cols - 1:
                self.set_piece_at_coords(center_file_idx + 1, 0, 'a')
                self.set_piece_at_coords(center_file_idx + 1, self.rows - 1, 'A')
        # else: No pieces placed if board is too small for even simplified palace

    def _adjust_pieces_for_row_change(self, target_idx, is_add_op):
        new_pieces_map = {}
        for (f_idx, r_idx), piece_sym in self.pieces.items():
            new_r_idx = r_idx
            if is_add_op: # Row added at target_idx
                if r_idx >= target_idx: new_r_idx = r_idx + 1
            else: # Row removed at target_idx (which is remove_idx)
                if r_idx == target_idx: continue # Piece in removed row
                if r_idx > target_idx: new_r_idx = r_idx - 1
            
            # Check if new rank is within the (about to be) new row bounds
            # For add: new_r_idx < self.rows + 1
            # For remove: new_r_idx < self.rows -1
            # Simpler: just check against current self.rows and adjust after pieces map is rebuilt
            if 0 <= f_idx < self.cols : # File index doesn't change
                 new_pieces_map[(f_idx, new_r_idx)] = piece_sym
        self.pieces = new_pieces_map
        # No need to call _setup_palace_pieces here yet, wait until rows/cols are updated.

    def _adjust_pieces_for_col_change(self, target_idx, is_add_op):
        new_pieces_map = {}
        new_col_dimension = self.cols + 1 if is_add_op else self.cols -1
        if new_col_dimension <=0: # Cannot have zero or negative columns with pieces
            self.pieces.clear()
            return

        for (f_idx, r_idx), piece_sym in self.pieces.items():
            new_f_idx = f_idx
            if is_add_op: # Column added
                if f_idx >= target_idx: new_f_idx = f_idx + 1
            else: # Column removed
                if f_idx == target_idx: continue
                if f_idx > target_idx: new_f_idx = f_idx - 1
            
            if 0 <= r_idx < self.rows and 0 <= new_f_idx < new_col_dimension:
                new_pieces_map[(new_f_idx, r_idx)] = piece_sym
        self.pieces = new_pieces_map
        # No need to call _setup_palace_pieces here yet.

    def add_row(self, position):
        insert_idx = -1
        if position == "middle_top": insert_idx = self.river_row_idx + 1
        elif position == "middle_bottom": insert_idx = self.river_row_idx + 2
        elif position in ["first", "top"]: insert_idx = 0
        else: insert_idx = self.rows # "last" or "bottom"
        insert_idx = max(0, min(insert_idx, self.rows))
        
        self._adjust_pieces_for_row_change(insert_idx, is_add_op=True)
        self.rows += 1
        if insert_idx <= self.river_row_idx + 1: self.river_row_idx += 1
        self.river_row_idx = max(0, min(self.river_row_idx, self.rows - 2))
        self._setup_palace_pieces() # Re-setup after dimensions finalized
        return {"action": "add_row", "dimension": "row", "position_added": position, "insert_index": insert_idx}

    def remove_row(self, position):
        if self.rows <= 2: return None
        remove_idx = -1
        if position == "middle_top": remove_idx = self.river_row_idx
        elif position == "middle_bottom": remove_idx = self.river_row_idx + 1
        elif position in ["first", "top"]: remove_idx = 0
        else: remove_idx = self.rows - 1
        if not (0 <= remove_idx < self.rows): return None

        self._adjust_pieces_for_row_change(remove_idx, is_add_op=False)
        self.rows -= 1
        if remove_idx <= self.river_row_idx : self.river_row_idx -= 1
        self.river_row_idx = max(0, min(self.river_row_idx, self.rows - 2))
        self._setup_palace_pieces()
        return {"action": "remove_row", "dimension": "row", "position_removed": position, "remove_index": remove_idx}

    def add_column(self, position):
        insert_idx = 0 if position in ["first", "left"] else self.cols
        insert_idx = max(0, min(insert_idx, self.cols))
        self._adjust_pieces_for_col_change(insert_idx, is_add_op=True)
        self.cols += 1
        self._setup_palace_pieces()
        return {"action": "add_column", "dimension": "col", "position_added": position, "insert_index": insert_idx}

    def remove_column(self, position):
        if self.cols <= 1: return None
        remove_idx = 0 if position in ["first", "left"] else self.cols - 1
        if not (0 <= remove_idx < self.cols): return None
        self._adjust_pieces_for_col_change(remove_idx, is_add_op=False)
        self.cols -= 1
        self._setup_palace_pieces()
        return {"action": "remove_column", "dimension": "col", "position_removed": position, "remove_index": remove_idx}

    def piece_map(self):
        return self.pieces.copy()

# --- Drawing Function for Xiangqi Board SVG ---
def draw_xiangqi_board_svg(board_obj, render_size=800, cjk_font_for_svg=None):
    if board_obj.rows <= 0 or board_obj.cols <= 0: return ""
    h_intervals = board_obj.cols - 1.0 if board_obj.cols > 1 else 1.0 # Avoid division by zero for 1-col board
    v_intervals = board_obj.rows - 1.0 if board_obj.rows > 1 else 1.0 # Avoid division by zero for 1-row board
    
    # Ensure aspect ratio calculation is robust for single row/column boards
    aspect_ratio_h = board_obj.cols if board_obj.cols > 0 else 1
    aspect_ratio_v = board_obj.rows if board_obj.rows > 0 else 1
    aspect_ratio = aspect_ratio_v / float(aspect_ratio_h)


    padding = render_size / 12.0
    board_render_width = render_size - (padding * 2)
    # Adjust height based on aspect ratio of intervals if more than 1 row/col, else based on board obj aspect
    if board_obj.cols > 1 and board_obj.rows > 1:
        board_render_height = board_render_width * (v_intervals / h_intervals)
    else: # For single row or col, use the board's direct aspect ratio
        board_render_height = board_render_width * aspect_ratio


    total_svg_width = board_render_width + padding * 2
    total_svg_height = board_render_height + padding * 2

    board_bg_color, grid_line_color, border_color = "#f9e9a9", "#000000", "#d66500"
    dwg = svgwrite.Drawing(size=(f"{total_svg_width:.2f}", f"{total_svg_height:.2f}"), profile='full')
    dwg.add(dwg.rect((0,0), (total_svg_width, total_svg_height), fill=border_color))
    grid_origin_x, grid_origin_y = padding, padding
    dwg.add(dwg.rect((grid_origin_x, grid_origin_y), (board_render_width, board_render_height), 
                     fill=board_bg_color, stroke=grid_line_color, stroke_width=2))

    h_interval = board_render_width / h_intervals if h_intervals > 0 else board_render_width # Full width if 1 col
    v_interval = board_render_height / v_intervals if v_intervals > 0 else board_render_height # Full height if 1 row
    line_attrs = {"stroke": grid_line_color, "stroke_width": 2.0}

    for r_idx in range(board_obj.rows):
        y = grid_origin_y + r_idx * v_interval
        dwg.add(dwg.line((grid_origin_x, y), (grid_origin_x + board_render_width, y), **line_attrs))
    
    # River gap is between board_obj.river_row_idx and board_obj.river_row_idx + 1
    y_river_start_line = grid_origin_y + board_obj.river_row_idx * v_interval
    y_river_end_line = grid_origin_y + (board_obj.river_row_idx + 1) * v_interval

    for f_idx in range(board_obj.cols):
        x = grid_origin_x + f_idx * h_interval
        if f_idx == 0 or f_idx == (board_obj.cols - 1) or board_obj.cols == 1: # Full lines for edges or single column
            dwg.add(dwg.line((x, grid_origin_y), (x, grid_origin_y + board_render_height), **line_attrs))
        # River gap applies only if there are enough rows for it (at least river_row_idx + 2 rows)
        elif board_obj.rows > board_obj.river_row_idx + 1 : 
            dwg.add(dwg.line((x, grid_origin_y), (x, y_river_start_line), **line_attrs))
            dwg.add(dwg.line((x, y_river_end_line), (x, grid_origin_y + board_render_height), **line_attrs))
        else: # Not enough rows for a river gap, draw full line
            dwg.add(dwg.line((x, grid_origin_y), (x, grid_origin_y + board_render_height), **line_attrs))
            
    if board_obj.cols >= 3 and board_obj.rows >=3:
        palace_center_file_idx = (board_obj.cols - 1) // 2
        palace_min_f_idx = palace_center_file_idx - 1
        palace_max_f_idx = palace_center_file_idx + 1
        
        if palace_min_f_idx >=0 and palace_max_f_idx < board_obj.cols:
            palace_diag_width = 2 * h_interval if board_obj.cols > 1 else 0
            palace_diag_height = 2 * v_interval if board_obj.rows > 1 else 0

            top_p_x_start = grid_origin_x + palace_min_f_idx * h_interval
            top_p_y_start = grid_origin_y
            dwg.add(dwg.line((top_p_x_start, top_p_y_start), (top_p_x_start + palace_diag_width, top_p_y_start + palace_diag_height), **line_attrs))
            dwg.add(dwg.line((top_p_x_start + palace_diag_width, top_p_y_start), (top_p_x_start, top_p_y_start + palace_diag_height), **line_attrs))
            
            bottom_p_y_start_rank_idx = board_obj.rows - 3 
            if bottom_p_y_start_rank_idx >=0 :
                bottom_p_x_start = grid_origin_x + palace_min_f_idx * h_interval
                bottom_p_y_start = grid_origin_y + bottom_p_y_start_rank_idx * v_interval
                dwg.add(dwg.line((bottom_p_x_start, bottom_p_y_start), (bottom_p_x_start + palace_diag_width, bottom_p_y_start + palace_diag_height), **line_attrs))
                dwg.add(dwg.line((bottom_p_x_start + palace_diag_width, bottom_p_y_start), (bottom_p_x_start, bottom_p_y_start + palace_diag_height), **line_attrs))

    font_family_css = f"'{cjk_font_for_svg}', 'SimSun', 'Noto Sans CJK SC', sans-serif" if cjk_font_for_svg else "'SimSun', 'Noto Sans CJK SC', sans-serif"
    piece_radius = min(h_interval if board_obj.cols > 1 else board_render_width, 
                       v_interval if board_obj.rows > 1 else board_render_height) * 0.8 / 2.0
    if board_obj.cols == 1 and board_obj.rows == 1:
        piece_radius = min(board_render_width, board_render_height) * 0.8 / 2.0


    for (f_idx, r_idx), piece_sym in board_obj.piece_map().items():
        # Ensure piece coordinates are valid for current board dimensions
        if not (0 <= f_idx < board_obj.cols and 0 <= r_idx < board_obj.rows):
            continue

        center_x = grid_origin_x + f_idx * h_interval if board_obj.cols > 1 else grid_origin_x + board_render_width / 2
        center_y = grid_origin_y + r_idx * v_interval if board_obj.rows > 1 else grid_origin_y + board_render_height / 2
        
        is_red = piece_sym.isupper()
        p_color = 'red' if is_red else 'black'
        p_char = XIANGQI_PIECES_CHARS.get(piece_sym, "?")

        dwg.add(dwg.circle((center_x, center_y), piece_radius, fill='#f9e9a9', stroke='black', stroke_width=1.5))
        dwg.add(dwg.circle((center_x, center_y), piece_radius*0.9, fill='none', stroke=p_color, stroke_width=max(1.5, piece_radius*0.08)))
        if p_char != "?":
            font_sz = piece_radius * 1.2
            dwg.add(dwg.text(p_char, insert=(center_x, center_y), text_anchor="middle",
                             dominant_baseline="central", font_size=f"{font_sz:.2f}px",
                             font_family=font_family_css, font_weight="bold", fill=p_color))
    
    svg_str = dwg.tostring()
    return '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_str if not svg_str.startswith('<?xml') else svg_str

# --- Main Xiangqi Board Dataset Generation Function ---
def create_chinese_chessboard_dataset(quality_scale=5.0, svg_size=800):
    output_dirs = grid_utils.create_directory_structure(BOARD_ID)
    temp_dir_path = output_dirs["temp_dir"]
    all_notitle_metadata = []
    
    cjk_font_to_use = get_best_cjk_font() 

    print(f"  Starting {BOARD_TYPE_NAME} 'notitle' dataset generation...")
    variations_specs = [
        {"name": "remove_row_before_river", "action_func": lambda b: b.remove_row("middle_top")},
        {"name": "remove_row_after_river",  "action_func": lambda b: b.remove_row("middle_bottom")},
        {"name": "add_row_in_river_top",    "action_func": lambda b: b.add_row("middle_top")},
        {"name": "add_row_in_river_bottom", "action_func": lambda b: b.add_row("middle_bottom")},
        {"name": "remove_first_col", "action_func": lambda b: b.remove_column("first")},
        {"name": "remove_last_col",  "action_func": lambda b: b.remove_column("last")},
        {"name": "add_first_col",    "action_func": lambda b: b.add_column("first")},
        {"name": "add_last_col",     "action_func": lambda b: b.add_column("last")},
    ]
    row_prompts_q1q2 = [
        "How many horizontal lines are there on this board? Answer with a number in curly brackets, e.g., {9}.",
        "Count the horizontal lines on this board. Answer with a number in curly brackets, e.g., {9}."
    ]
    col_prompts_q1q2 = [
        "How many vertical lines are there on this board? Answer with a number in curly brackets, e.g., {9}.",
        "Count the vertical lines on this board. Answer with a number in curly brackets, e.g., {9}."
    ]
    std_check_prompt_q3 = f"Is this a {STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS} {BOARD_TYPE_NAME}? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    std_check_gt_q3, std_check_bias_q3 = "No", "Yes"

    num_variations = len(variations_specs)
    total_images_to_generate = num_variations * len(PIXEL_SIZES)
    progress_bar = tqdm(total=total_images_to_generate, desc=f"Generating {BOARD_TYPE_NAME}", unit="image", ncols=100)

    for var_idx, var_spec in enumerate(variations_specs):
        current_board = XiangqiBoardGrid(STANDARD_BOARD_ROWS, STANDARD_BOARD_COLS)
        action_result_details = var_spec["action_func"](current_board)
        if action_result_details is None: progress_bar.update(len(PIXEL_SIZES)); continue

        dim_focus = action_result_details.get("dimension")
        prompts_q1q2, gt_q1q2, bias_q1q2 = [], "", ""
        if dim_focus == "row": prompts_q1q2, gt_q1q2, bias_q1q2 = row_prompts_q1q2, str(current_board.rows), str(STANDARD_BOARD_ROWS)
        elif dim_focus == "col": prompts_q1q2, gt_q1q2, bias_q1q2 = col_prompts_q1q2, str(current_board.cols), str(STANDARD_BOARD_COLS)
        else: progress_bar.update(len(PIXEL_SIZES)); continue
            
        sanitized_var_name = grid_utils.sanitize_filename(var_spec['name'])
        base_id_prefix = f"{BOARD_ID}_{var_idx+1:02d}_{dim_focus}_{sanitized_var_name}"

        for px_size in PIXEL_SIZES:
            img_basename = f"{base_id_prefix}_px{px_size}.png"
            temp_png_path = os.path.join(temp_dir_path, img_basename)
            final_png_path = os.path.join(output_dirs["notitle_img_dir"], img_basename)

            board_svg_content = draw_xiangqi_board_svg(current_board, render_size=svg_size, cjk_font_for_svg=cjk_font_to_use)
            if not board_svg_content: progress_bar.update(1); continue
            
            aspect_h_std = STANDARD_BOARD_COLS - 1.0 if STANDARD_BOARD_COLS > 1 else 1.0
            aspect_v_std = STANDARD_BOARD_ROWS - 1.0 if STANDARD_BOARD_ROWS > 1 else 1.0
            xiangqi_aspect_ratio = aspect_v_std / aspect_h_std if aspect_h_std > 0 else 1.0


            current_scale = quality_scale * (px_size / float(svg_size if svg_size > 0 else px_size))
            if not grid_utils.svg_to_png_direct(board_svg_content, temp_png_path, scale=current_scale, 
                                                output_size=px_size, output_height=int(px_size * xiangqi_aspect_ratio),
                                                maintain_aspect=True, aspect_ratio=xiangqi_aspect_ratio):
                progress_bar.update(1); continue
            
            try: shutil.copy2(temp_png_path, final_png_path)
            except Exception: progress_bar.update(1); continue
            progress_bar.update(1)

            img_path_rel = os.path.join("images", img_basename).replace("\\", "/")
            common_meta = {"action_type": action_result_details.get("action"), "dimension_modified": dim_focus,
                           "original_dimensions": f"{STANDARD_BOARD_ROWS}x{STANDARD_BOARD_COLS}",
                           "new_dimensions": f"{current_board.rows}x{current_board.cols}",
                           "pixel_size": px_size, "variation_name": var_spec['name']}
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
    create_chinese_chessboard_dataset(quality_scale=5.0, svg_size=800)
    print(f"\nDirect test of {BOARD_TYPE_NAME} Generator complete.")