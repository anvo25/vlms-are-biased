"""
Generates the "notitle" Xiangqi (Chinese Chess) Pieces dataset.
Modifies a standard Xiangqi starting position by removing or replacing one piece.
"""
import os
import random
import shutil
import svgwrite # For drawing Xiangqi board
from tqdm import tqdm
import sys
import matplotlib.font_manager as fm # For CJK font detection

from utils import (sanitize_filename, svg_to_png_direct,
                   save_metadata_files) 

# --- Global Constants ---
PIXEL_SIZES = [384, 768, 1152]
NUM_ACTION_GROUPS = 12
ACTIONS_PER_GROUP = 2
TOPIC_NAME = "xiangqi_pieces"

XIANGQI_BOARD_WIDTH = 9
XIANGQI_BOARD_HEIGHT = 10

XIANGQI_PIECES_CHARS = {
    'k': '將', 'a': '士', 'e': '象', 'h': '馬', 'r': '車', 'c': '砲', 'p': '卒',
    'K': '帥', 'A': '仕', 'E': '相', 'H': '馬', 'R': '俥', 'C': '炮', 'P': '兵'
}
XIANGQI_PIECE_FULL_NAMES = {
    'k': 'Black_General',  'a': 'Black_Advisor',  'e': 'Black_Elephant',
    'h': 'Black_Horse',    'r': 'Black_Chariot',  'c': 'Black_Cannon',   'p': 'Black_Soldier',
    'K': 'Red_General',    'A': 'Red_Advisor',    'E': 'Red_Elephant',
    'H': 'Red_Horse',      'R': 'Red_Chariot',    'C': 'Red_Cannon',     'P': 'Red_Soldier'
}
XIANGQI_PIECE_TYPE_NAMES_FOR_PROMPT = {
    'k': "General", 'a': "Advisor", 'e': "Elephant", 'h': "Horse",
    'r': "Chariot", 'c': "Cannon", 'p': "Soldier"
}

# --- CJK Font Detection ---
def detect_cjk_fonts():
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        cjk_keywords = [
            'simsun', 'uming', 'ukai', 'mingliu', 'pmingliu', 'dfkai-sb', 
            'noto sans cjk', 'noto serif cjk', 'source han sans', 'source han serif',
            'microsoft yahei', 'msyh', 'dengxian', 'simhei', 'fangsong', 'kaiti', 
            'wenquanyi', 'wqy', 'ar pl', 'nanum', 'malgun gothic', 'gulim', 'batang', 
            'dotum', 'meiryo', 'ms gothic', 'ms mincho', 'hiragino', 'heiti tc', 'songti sc'
        ]
        normalized_available_fonts = {f.lower().replace(" ", ""): f for f in available_fonts}
        cjk_fonts_found = set()
        for norm_font_name, original_font_name in normalized_available_fonts.items():
            if any(keyword in norm_font_name for keyword in cjk_keywords):
                cjk_fonts_found.add(original_font_name)
        return sorted(list(cjk_fonts_found))
    except Exception as e:
        print(f"  Warning: Error detecting CJK fonts: {e}")
        return []

def get_best_cjk_font():
    detected_fonts = detect_cjk_fonts()
    if not detected_fonts:
        print("  No CJK fonts detected by font manager for Xiangqi Pieces.")
        return None
    preferred_order = [
        'SimSun', 'Noto Sans CJK SC', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 
        'Source Han Sans SC', 'Noto Serif CJK SC', 'Source Han Serif SC', 
        'Arial Unicode MS'
    ]
    normalized_preferred = {p.lower().replace(" ", ""): p for p in preferred_order}
    for norm_pref, original_pref in normalized_preferred.items():
        for font_path_name in detected_fonts:
            if norm_pref in font_path_name.lower().replace(" ", ""):
                print(f"  -> Using CJK font for Xiangqi Pieces: '{font_path_name}' (matched '{original_pref}')")
                return font_path_name
    if detected_fonts:
        fallback_font = detected_fonts[0]
        print(f"  Warning: No preferred CJK font for Xiangqi Pieces. Using fallback: '{fallback_font}'")
        return fallback_font
    print("  Warning: No CJK fonts available for Xiangqi Pieces. Characters may not render correctly.")
    return None

# --- XiangqiBoard Class ---
class XiangqiBoard:
    def __init__(self):
        self.board_pieces = {} 
        self.setup_initial_position()

    def _to_square_index(self, file_idx, rank_idx):
        if 0 <= file_idx < XIANGQI_BOARD_WIDTH and 0 <= rank_idx < XIANGQI_BOARD_HEIGHT:
            return rank_idx * XIANGQI_BOARD_WIDTH + file_idx
        return None

    def set_piece_at_coords(self, file_idx, rank_idx, piece_symbol):
        square_idx = self._to_square_index(file_idx, rank_idx)
        if square_idx is not None:
            self.board_pieces[square_idx] = piece_symbol

    def set_piece_at_index(self, square_idx, piece_symbol):
        if 0 <= square_idx < (XIANGQI_BOARD_WIDTH * XIANGQI_BOARD_HEIGHT):
            if piece_symbol is None: 
                if square_idx in self.board_pieces:
                    del self.board_pieces[square_idx]
            else:
                self.board_pieces[square_idx] = piece_symbol
        else:
            print(f"  Warning (XiangqiBoard - Pieces): Attempted to set piece at invalid square index {square_idx}.")

    def get_piece_at_index(self, square_idx): 
        return self.board_pieces.get(square_idx)
        
    def setup_initial_position(self):
        self.board_pieces.clear()
        # Red pieces (uppercase) - ranks 9 down to 6 (0-indexed from Black's side)
        self.set_piece_at_coords(0, 9, 'R'); self.set_piece_at_coords(1, 9, 'H'); self.set_piece_at_coords(2, 9, 'E');
        self.set_piece_at_coords(3, 9, 'A'); self.set_piece_at_coords(4, 9, 'K'); self.set_piece_at_coords(5, 9, 'A');
        self.set_piece_at_coords(6, 9, 'E'); self.set_piece_at_coords(7, 9, 'H'); self.set_piece_at_coords(8, 9, 'R');
        self.set_piece_at_coords(1, 7, 'C'); self.set_piece_at_coords(7, 7, 'C');
        for f_idx in range(0, 9, 2): self.set_piece_at_coords(f_idx, 6, 'P')

        # Black pieces (lowercase) - ranks 0 up to 3
        self.set_piece_at_coords(0, 0, 'r'); self.set_piece_at_coords(1, 0, 'h'); self.set_piece_at_coords(2, 0, 'e');
        self.set_piece_at_coords(3, 0, 'a'); self.set_piece_at_coords(4, 0, 'k'); self.set_piece_at_coords(5, 0, 'a');
        self.set_piece_at_coords(6, 0, 'e'); self.set_piece_at_coords(7, 0, 'h'); self.set_piece_at_coords(8, 0, 'r');
        self.set_piece_at_coords(1, 2, 'c'); self.set_piece_at_coords(7, 2, 'c');
        for f_idx in range(0, 9, 2): self.set_piece_at_coords(f_idx, 3, 'p')
            
    def piece_map(self): 
        return self.board_pieces.copy()
        
    def copy(self): 
        new_board = XiangqiBoard() 
        new_board.board_pieces = self.board_pieces.copy()
        return new_board

# --- Drawing Function (Identical to the one in xiangqi_board_generator) ---
def draw_xiangqi_board_svg(board_obj, render_size=800, show_coords=False, cjk_font_override=None):
    h_intervals = XIANGQI_BOARD_WIDTH - 1.0 if XIANGQI_BOARD_WIDTH > 1 else 1.0
    v_intervals = XIANGQI_BOARD_HEIGHT - 1.0 if XIANGQI_BOARD_HEIGHT > 1 else 1.0
    aspect_ratio = v_intervals / h_intervals if h_intervals > 0 else (XIANGQI_BOARD_HEIGHT / float(XIANGQI_BOARD_WIDTH) if XIANGQI_BOARD_WIDTH > 0 else 1.0)

    padding = render_size / 12.0
    board_render_width = render_size - (padding * 2)
    board_render_height = board_render_width * aspect_ratio
    total_svg_width = board_render_width + padding * 2
    total_svg_height = board_render_height + padding * 2

    board_bg_color, grid_line_color, border_color = "#f9e9a9", "#000000", "#d66500"
    dwg = svgwrite.Drawing(size=(f"{total_svg_width:.2f}", f"{total_svg_height:.2f}"), profile='full', debug=False)
    dwg.add(dwg.rect((0,0), (total_svg_width, total_svg_height), fill=border_color))
    grid_origin_x, grid_origin_y = padding, padding
    dwg.add(dwg.rect((grid_origin_x, grid_origin_y), (board_render_width, board_render_height), fill=board_bg_color, stroke=grid_line_color, stroke_width=2))

    h_interval = board_render_width / h_intervals if h_intervals > 0 else board_render_width
    v_interval = board_render_height / v_intervals if v_intervals > 0 else board_render_height
    line_attrs = {"stroke": grid_line_color, "stroke_width": 2.0}

    for r_idx in range(XIANGQI_BOARD_HEIGHT):
        y = grid_origin_y + r_idx * v_interval
        dwg.add(dwg.line((grid_origin_x, y), (grid_origin_x + board_render_width, y), **line_attrs))
    
    river_rank_idx = 4 # River is between rank 4 and 5
    y_river_start_line = grid_origin_y + river_rank_idx * v_interval
    y_river_end_line = grid_origin_y + (river_rank_idx + 1) * v_interval

    for f_idx in range(XIANGQI_BOARD_WIDTH):
        x = grid_origin_x + f_idx * h_interval
        if f_idx == 0 or f_idx == (XIANGQI_BOARD_WIDTH - 1) or XIANGQI_BOARD_WIDTH == 1:
            dwg.add(dwg.line((x, grid_origin_y), (x, grid_origin_y + board_render_height), **line_attrs))
        elif XIANGQI_BOARD_HEIGHT > river_rank_idx + 1:
            dwg.add(dwg.line((x, grid_origin_y), (x, y_river_start_line), **line_attrs))
            dwg.add(dwg.line((x, y_river_end_line), (x, grid_origin_y + board_render_height), **line_attrs))
        else:
            dwg.add(dwg.line((x, grid_origin_y), (x, grid_origin_y + board_render_height), **line_attrs))
            
    if XIANGQI_BOARD_WIDTH >=5 and XIANGQI_BOARD_HEIGHT >=3: # Min for basic palace lines
        palace_diag_w = 2 * h_interval
        palace_diag_h = 2 * v_interval
        # Black's Palace (top)
        top_p_x_start = grid_origin_x + 3 * h_interval # File 3
        top_p_y_start = grid_origin_y # Rank 0
        dwg.add(dwg.line((top_p_x_start, top_p_y_start), (top_p_x_start + palace_diag_w, top_p_y_start + palace_diag_h), **line_attrs))
        dwg.add(dwg.line((top_p_x_start + palace_diag_w, top_p_y_start), (top_p_x_start, top_p_y_start + palace_diag_h), **line_attrs))
        # Red's Palace (bottom)
        bottom_p_x_start = grid_origin_x + 3 * h_interval # File 3
        bottom_p_y_start = grid_origin_y + (XIANGQI_BOARD_HEIGHT - 1 - 2) * v_interval # Rank 7 for 10-row board
        dwg.add(dwg.line((bottom_p_x_start, bottom_p_y_start), (bottom_p_x_start + palace_diag_w, bottom_p_y_start + palace_diag_h), **line_attrs))
        dwg.add(dwg.line((bottom_p_x_start + palace_diag_w, bottom_p_y_start), (bottom_p_x_start, bottom_p_y_start + palace_diag_h), **line_attrs))
    
    font_family_css = f"'{cjk_font_override}', 'SimSun', 'Noto Sans CJK SC', sans-serif" if cjk_font_override else "'SimSun', 'Noto Sans CJK SC', sans-serif"
    piece_radius = min(h_interval, v_interval) * 0.8 / 2.0
    if XIANGQI_BOARD_WIDTH == 1 and XIANGQI_BOARD_HEIGHT == 1:
        piece_radius = min(board_render_width, board_render_height) * 0.8 / 2.0

    for square_idx, piece_sym in board_obj.piece_map().items():
        f_idx = square_idx % XIANGQI_BOARD_WIDTH
        r_idx = square_idx // XIANGQI_BOARD_WIDTH
        center_x = grid_origin_x + f_idx * h_interval if XIANGQI_BOARD_WIDTH > 1 else grid_origin_x + board_render_width / 2
        center_y = grid_origin_y + r_idx * v_interval if XIANGQI_BOARD_HEIGHT > 1 else grid_origin_y + board_render_height / 2
        
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

# --- Action Selection Logic ---
def select_xiangqi_action_groups(board, num_groups=NUM_ACTION_GROUPS):
    # (Identical to the corrected version in previous response)
    occupied_indices = list(board.piece_map().keys())
    if len(occupied_indices) < num_groups:
        print(f"  FATAL ERROR: Not enough pieces ({len(occupied_indices)}) on Xiangqi board for {num_groups} groups.")
        return None

    print(f"  Selecting {num_groups} target squares for Xiangqi modifications...")
    target_indices = random.sample(occupied_indices, num_groups)
    
    action_groups = []
    all_xiangqi_piece_symbols = list(XIANGQI_PIECES_CHARS.keys()) 

    for group_id_counter, sq_idx in enumerate(target_indices, 1):
        original_piece_symbol = board.get_piece_at_index(sq_idx)
        if not original_piece_symbol: continue 

        remove_action = {'action_type': 'Remove', 'target_square_index': sq_idx, 
                         'original_piece_symbol': original_piece_symbol, 'new_piece_symbol': None}

        original_is_red = original_piece_symbol.isupper()
        # Filter for pieces of the same color but different type
        possible_new_symbols = [
            psym for psym in all_xiangqi_piece_symbols 
            if (psym.isupper() == original_is_red) and \
               (XIANGQI_PIECES_CHARS.get(psym.lower()) != XIANGQI_PIECES_CHARS.get(original_piece_symbol.lower())) 
        ]
        
        if not possible_new_symbols:
            print(f"  Warning: No suitable replacement found for {original_piece_symbol} at square {sq_idx} (Group {group_id_counter}). Skipping Replace.")
            replace_action = None
        else:
            new_replacement_symbol = random.choice(possible_new_symbols)
            replace_action = {'action_type': 'Replace', 'target_square_index': sq_idx,
                              'original_piece_symbol': original_piece_symbol, 'new_piece_symbol': new_replacement_symbol}
        
        if remove_action and replace_action:
            action_groups.append({'group_id': group_id_counter, 'target_square_index': sq_idx,
                                 'actions_in_group': [remove_action, replace_action]})
        else:
            print(f"  Warning: Skipping Group {group_id_counter} for Xiangqi due to missing action definition.")
            
    if len(action_groups) < num_groups:
        print(f"  Warning: Only {len(action_groups)} valid Xiangqi action groups defined (requested {num_groups}).")
    print(f"  Defined {len(action_groups)} action groups for Xiangqi Pieces.")
    return action_groups

# --- Single Action Generation Function (Notitle Only) ---
def generate_single_xiangqi_notitle_image_and_meta(
    group_id, action_details, base_board_state,
    output_dirs_struct, pixel_sizes_list, png_quality_scale, svg_render_size,
    progress_bar_updater, cjk_font_for_svg, topic_name_constant=TOPIC_NAME):
    # (Identical to the corrected version in previous response)
    generated_metadata_for_action = []
    action_verb = action_details['action_type']
    square_idx = action_details['target_square_index']
    original_piece_sym = action_details['original_piece_symbol']
    new_piece_sym_if_replace = action_details['new_piece_symbol']

    if not original_piece_sym:
        progress_bar_updater.update(len(pixel_sizes_list)); return []

    file_char = chr(ord('a') + (square_idx % XIANGQI_BOARD_WIDTH))
    rank_num = (square_idx // XIANGQI_BOARD_WIDTH) + 1 # 1-indexed rank for display
    square_algebraic_name = f"{file_char}{rank_num}"
    
    original_piece_fullname = XIANGQI_PIECE_FULL_NAMES.get(original_piece_sym, "Unknown")
    sanitized_orig_piece_name = sanitize_filename(original_piece_fullname)
    group_id_formatted = f"{group_id:03d}"
    action_verb_lower = action_verb.lower()

    piece_added_fullname = "N/A"
    piece_added_type_for_prompt = "N/A"

    current_modified_board = base_board_state.copy()
    if action_verb == 'Remove':
        current_modified_board.set_piece_at_index(square_idx, None)
    elif action_verb == 'Replace' and new_piece_sym_if_replace:
        current_modified_board.set_piece_at_index(square_idx, new_piece_sym_if_replace)
        piece_added_fullname = XIANGQI_PIECE_FULL_NAMES.get(new_piece_sym_if_replace, "Unknown")
        piece_added_type_for_prompt = XIANGQI_PIECE_TYPE_NAMES_FOR_PROMPT.get(new_piece_sym_if_replace.lower(), "Unknown")
    elif action_verb == 'Replace' and not new_piece_sym_if_replace:
        progress_bar_updater.update(len(pixel_sizes_list)); return []

    prompts_q1_q2, ground_truth_q1_q2, expected_bias_q1_q2 = [], "", ""
    standard_total_xiangqi_pieces = 32 
    standard_xiangqi_type_counts = { "General": 2, "Advisor": 4, "Elephant": 4, "Horse": 4, 
                                     "Chariot": 4, "Cannon": 4, "Soldier": 10 }

    if action_verb == 'Remove':
        current_total_pieces = len(current_modified_board.piece_map())
        ground_truth_q1_q2 = str(current_total_pieces)
        prompts_q1_q2 = [
            "How many xiangqi pieces are there on this board? Answer with a number in curly brackets, e.g., {9}.",
            "Count the xiangqi pieces on this board. Answer with a number in curly brackets, e.g., {9}."
        ]
        expected_bias_q1_q2 = str(standard_total_xiangqi_pieces)
    elif action_verb == 'Replace':
        count_of_added_type = sum(1 for piece_symbol_on_board in current_modified_board.piece_map().values() 
                                  if piece_symbol_on_board.lower() == new_piece_sym_if_replace.lower())
        ground_truth_q1_q2 = str(count_of_added_type)
        prompts_q1_q2 = [
            "How many {piece_added_type_for_prompt} pieces are there on this board? Answer with a number in curly brackets, e.g., {9}.",
            "Count the {piece_added_type_for_prompt} pieces on this board. Answer with a number in curly brackets, e.g., {9}."
        ]
        expected_bias_q1_q2 = str(standard_xiangqi_type_counts.get(piece_added_type_for_prompt, "0"))

    prompt_q3 = "Is this the Xiangqi starting position? Answer in curly brackets, e.g., {Yes} or {No}."
    ground_truth_q3, expected_bias_q3 = "No", "Yes"

    for px_size in pixel_sizes_list:
        if action_verb == 'Remove':
            piece_mod_info_str = f"{sanitized_orig_piece_name}_at_{square_algebraic_name}"
        else: 
            sanitized_added_piece_name = sanitize_filename(piece_added_fullname)
            piece_mod_info_str = f"{sanitized_orig_piece_name}_to_{sanitized_added_piece_name}_at_{square_algebraic_name}"
        
        img_basename = f"{topic_name_constant}_{group_id_formatted}_{action_verb_lower}_{piece_mod_info_str}_notitle_px{px_size}.png"
        final_png_path = os.path.join(output_dirs_struct["img_dirs"]['notitle'], img_basename)
        
        board_svg_content = draw_xiangqi_board_svg(current_modified_board, render_size=svg_render_size, 
                                                 show_coords=False, cjk_font_override=cjk_font_for_svg)
        
        aspect_h_std = XIANGQI_BOARD_WIDTH - 1.0 if XIANGQI_BOARD_WIDTH > 1 else 1.0
        aspect_v_std = XIANGQI_BOARD_HEIGHT - 1.0 if XIANGQI_BOARD_HEIGHT > 1 else 1.0
        xiangqi_aspect_ratio = aspect_v_std / aspect_h_std if aspect_h_std > 0 else 1.0
        
        scale_for_conversion = png_quality_scale * (px_size / 768.0)
        os.makedirs(os.path.dirname(final_png_path), exist_ok=True)
        
        if not svg_to_png_direct(board_svg_content, final_png_path, scale_for_conversion, 
                                 px_size, output_height=int(px_size * xiangqi_aspect_ratio),
                                 maintain_aspect=True, aspect_ratio=xiangqi_aspect_ratio):
            print(f"  ERROR: Failed PNG for {img_basename}. Skipping res.")
            progress_bar_updater.update(1); continue
        
        progress_bar_updater.update(1)
        
        common_metadata_payload = {
             "group_id": group_id, "action_type": action_verb,
             "modified_square_algebraic": square_algebraic_name,
             "piece_removed_info": original_piece_fullname,
             "piece_added_info": piece_added_fullname,
             "pixel_size": px_size
        }
        image_path_relative = os.path.join("images", img_basename).replace("\\", "/")

        for q_idx, current_prompt_text in enumerate(prompts_q1_q2):
            q_type_label = f"Q{q_idx + 1}"
            meta_id = f"{topic_name_constant}_{group_id_formatted}_{action_verb_lower}_{piece_mod_info_str}_notitle_px{px_size}_{q_type_label}"
            generated_metadata_for_action.append({
                 "ID": meta_id, "image_path": image_path_relative, "topic": "Xiangqi Pieces",
                 "prompt": current_prompt_text, "ground_truth": ground_truth_q1_q2,
                 "expected_bias": expected_bias_q1_q2, "with_title": False,
                 "type_of_question": q_type_label, "pixel": px_size,
                 "metadata": common_metadata_payload.copy()
            })
        meta_id_q3 = f"{topic_name_constant}_{group_id_formatted}_{action_verb_lower}_{piece_mod_info_str}_notitle_px{px_size}_Q3"
        generated_metadata_for_action.append({
             "ID": meta_id_q3, "image_path": image_path_relative, "topic": "Xiangqi Pieces",
             "prompt": prompt_q3, "ground_truth": ground_truth_q3, "expected_bias": expected_bias_q3,
             "with_title": False, "type_of_question": "Q3", "pixel": px_size,
             "metadata": common_metadata_payload.copy()
        })
    return generated_metadata_for_action

# --- Main Dataset Generation Function (Notitle Only) ---
def create_chinesechess_notitle_dataset(dirs, quality_scale=5.0, svg_size=800): # Name from main.py
    current_topic_name = TOPIC_NAME
    print(f"  Starting {current_topic_name.replace('_',' ').title()} 'notitle' dataset generation...")
    total_images_to_generate = NUM_ACTION_GROUPS * ACTIONS_PER_GROUP * len(PIXEL_SIZES)
    print(f"    - Target Groups: {NUM_ACTION_GROUPS}, Actions per Group: {ACTIONS_PER_GROUP}")
    print(f"    - Resolutions: {PIXEL_SIZES}, Total images expected: {total_images_to_generate}")

    all_notitle_metadata_entries = []
    base_xiangqi_board = XiangqiBoard()
    
    cjk_font_to_use_in_svg = get_best_cjk_font() 
    if not cjk_font_to_use_in_svg:
        print("  Proceeding without a specific CJK font for SVG; rendering quality may vary.")

    selected_action_groups = select_xiangqi_action_groups(base_xiangqi_board, NUM_ACTION_GROUPS)
    if not selected_action_groups or len(selected_action_groups) < NUM_ACTION_GROUPS:
         print(f"  FATAL ERROR: Could not define {NUM_ACTION_GROUPS} Xiangqi action groups. Generation aborted.")
         return

    progress_bar = tqdm(total=total_images_to_generate, 
                        desc=f"Generating Xiangqi Pieces", unit="image", ncols=100)

    for group_definition in selected_action_groups:
        current_group_id = group_definition['group_id']
        for action_spec_details in group_definition['actions_in_group']:
            metadata_for_this_action = generate_single_xiangqi_notitle_image_and_meta(
                current_group_id, action_spec_details, base_xiangqi_board,
                dirs, PIXEL_SIZES, quality_scale, svg_size, progress_bar,
                cjk_font_to_use_in_svg, 
                current_topic_name
            )
            all_notitle_metadata_entries.extend(metadata_for_this_action)
    
    progress_bar.close()
    if progress_bar.n < progress_bar.total:
         print(f"  Warning: Xiangqi Pieces generation progress ({progress_bar.n}/{progress_bar.total}) indicates some images may have been skipped. Check logs.")

    print(f"\n  Saving 'notitle' metadata for {current_topic_name}...")
    save_metadata_files(
        all_notitle_metadata_entries, 
        dirs["meta_dirs"]['notitle'], 
        f"{current_topic_name}_notitle"
    )

    print(f"\n  --- Xiangqi Pieces 'notitle' Generation Summary ---")
    final_image_files_count = 0
    try:
        img_output_dir = dirs["img_dirs"]['notitle']
        if os.path.exists(img_output_dir):
            final_image_files_count = len([f for f in os.listdir(img_output_dir) if f.endswith('.png')])
    except OSError as e: print(f"  Warning: Could not count final images: {e}")
    print(f"  Actual 'notitle' images generated: {final_image_files_count} (Expected: {total_images_to_generate})")
    print(f"  Total 'notitle' metadata entries created: {len(all_notitle_metadata_entries)}")
    
    try:
        temp_dir_to_clean = dirs.get("temp_dir")
        if temp_dir_to_clean and os.path.exists(temp_dir_to_clean):
            shutil.rmtree(temp_dir_to_clean)
            print(f"  Cleaned up temporary directory: {temp_dir_to_clean}")
    except Exception as e: print(f"  Warning: Failed temp cleanup {temp_dir_to_clean}: {e}")
    print(f"  Xiangqi Pieces 'notitle' dataset generation finished.")

if __name__ == '__main__':
    print("Testing Xiangqi Pieces Generator directly...")
    # Need to import create_directory_structure from the main utils.py for this test
    from utils import create_directory_structure as util_create_dirs
    if not os.path.exists("vlms-are-biased-notitle"): 
        os.makedirs("vlms-are-biased-notitle")
    
    test_output_dirs = util_create_dirs(TOPIC_NAME, title_types_to_create=['notitle'])
    print(f"  Test output directories for 'notitle':")
    print(f"    Images: {test_output_dirs['img_dirs']['notitle']}")
    print(f"    Metadata: {test_output_dirs['meta_dirs']['notitle']}")
    print(f"    Temp: {test_output_dirs['temp_dir']}")
        
    create_chinesechess_notitle_dataset(dirs=test_output_dirs, quality_scale=5.0, svg_size=800)
    print("\nDirect test of Xiangqi Pieces Generator complete.")