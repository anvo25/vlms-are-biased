# -*- coding: utf-8 -*-
import os
import random
import shutil
import svgwrite
from tqdm import tqdm
import sys
import matplotlib.font_manager as fm
from utils import (sanitize_filename, svg_to_png_direct, add_title_to_png, 
                   save_metadata_files, TITLE_TYPES, load_title)

# --- Constants for Chinese Chess ---
BOARD_WIDTH = 9
BOARD_HEIGHT = 10
SQUARES = [i for i in range(BOARD_WIDTH * BOARD_HEIGHT)]
PIXEL_SIZES = [384, 768, 1152]  # Standard resolutions
NUM_ACTION_GROUPS = 12  # Number of unique ID groups / target squares
ACTIONS_PER_GROUP = 2  # Remove, Replace

# Xiangqi piece types with Chinese characters
XIANGQI_PIECES = {
    'k': '將', 'a': '士', 'e': '象', 'h': '馬', 'r': '車', 'c': '砲', 'p': '卒',  # Black
    'K': '帥', 'A': '仕', 'E': '相', 'H': '馬', 'R': '俥', 'C': '炮', 'P': '兵'   # Red
}

# Full piece names for metadata
PIECE_NAMES = {
    'k': 'Black General',  'a': 'Black Advisor',  'e': 'Black Elephant',
    'h': 'Black Knight',    'r': 'Black Chariot', 'c': 'Black Cannon', 'p': 'Black Soldier',
    'K': 'Red General',    'A': 'Red Advisor',   'E': 'Red Elephant',
    'H': 'Red Knight',      'R': 'Red Chariot',   'C': 'Red Cannon',   'P': 'Red Soldier'
}

# Map lowercase simple type to full name for standard counts
PIECE_TYPE_NAMES_STD = {
    'k': "General", 'a': "Advisor", 'e': "Elephant", 'h': "Knight",
    'r': "Chariot", 'c': "Cannon", 'p': "Soldier"
}

# --- Helper Functions ---
def detect_cjk_fonts():
    """Find available CJK fonts on the system"""
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        # Broaden search terms significantly
        cjk_keywords = ['simsun', 'uming', 'ukai', 'mingliu', 'pmingliu', 'dfkai-sb', 
                        'noto sans cjk', 'noto serif cjk', 'yahei', 'dengxian', 'simhei', 
                        'fangsong', 'kaiti', 'source han', 'wqy', 'ar pl', 'nanum', 
                        'malgun', 'meiryo', 'msgothic', 'msmincho', 'gulim', 'batang', 
                        'dotum', 'applegothic', 'hiragino', 'heiti', 'songti', 'stsong', 'stkaiti']
        cjk_fonts = sorted(list(set([f for f in available_fonts 
                                    if any(name in f.lower().replace(" ","") 
                                          for name in cjk_keywords)])))
        return cjk_fonts
    except Exception as e:
        print(f"Warning: Error detecting fonts using matplotlib: {e}")
        return []

def get_best_cjk_font():
    """Returns the best available CJK font for rendering Chinese characters"""
    cjk_fonts = detect_cjk_fonts()
    # Refined preferred order
    preferred_order = ['SimSun', 'Noto Sans CJK SC', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 
                       'Noto Serif CJK SC', 'Source Han Sans SC', 'Source Han Serif SC', 
                       'Noto Sans CJK TC', 'Noto Sans CJK JP']
    for preferred in preferred_order:
        for font in cjk_fonts:
            if preferred.lower().replace(" ","") in font.lower().replace(" ",""):
                print(f"Info: Found preferred CJK font: {font}")
                return font
    # Fallback
    if cjk_fonts:
        print(f"Warning: Preferred CJK font not found, using first detected: {cjk_fonts[0]}")
        return cjk_fonts[0]
    else:
        print("Warning: No CJK fonts detected.")
        return None

def piece_to_name(piece_symbol):
    """Convert piece symbol to standard name"""
    return PIECE_NAMES.get(piece_symbol, 'Unknown')

def square_name(square):
    """Convert square index to Chinese Chess file/rank notation (a1-i10)"""
    file = square % BOARD_WIDTH
    rank = square // BOARD_WIDTH
    return f"{chr(ord('a') + file)}{rank + 1}"

# --- Chinese Chess Board Class ---
class XiangqiBoard:
    """Simple Chinese Chess board representation"""
    def __init__(self):
        self.board = {}
        self.setup_initial_position()

    def setup_initial_position(self):
        """Set up the standard initial position"""
        self.board = {}
        # Black (lowercase) - Ranks 0-4 (Bottom half visually)
        self.set_piece(0, 0, 'r'); self.set_piece(1, 0, 'h'); self.set_piece(2, 0, 'e');
        self.set_piece(3, 0, 'a'); self.set_piece(4, 0, 'k'); self.set_piece(5, 0, 'a');
        self.set_piece(6, 0, 'e'); self.set_piece(7, 0, 'h'); self.set_piece(8, 0, 'r');
        self.set_piece(1, 2, 'c'); self.set_piece(7, 2, 'c');
        for file in range(0, 9, 2): self.set_piece(file, 3, 'p')
        # Red (uppercase) - Ranks 5-9 (Top half visually)
        for file in range(0, 9, 2): self.set_piece(file, 6, 'P')
        self.set_piece(1, 7, 'C'); self.set_piece(7, 7, 'C');
        self.set_piece(0, 9, 'R'); self.set_piece(1, 9, 'H'); self.set_piece(2, 9, 'E');
        self.set_piece(3, 9, 'A'); self.set_piece(4, 9, 'K'); self.set_piece(5, 9, 'A');
        self.set_piece(6, 9, 'E'); self.set_piece(7, 9, 'H'); self.set_piece(8, 9, 'R');

    def _to_square(self, file, rank):
        if 0 <= file < BOARD_WIDTH and 0 <= rank < BOARD_HEIGHT: 
            return rank * BOARD_WIDTH + file
        return None

    def set_piece(self, file, rank, piece):
        square = self._to_square(file, rank)
        if square is not None: 
            self.board[square] = piece

    def set_piece_at(self, square, piece):
        if piece is None: 
            self.remove_piece_at(square)
        elif 0 <= square < (BOARD_WIDTH * BOARD_HEIGHT): 
            self.board[square] = piece

    def get_piece_at(self, square): 
        return self.board.get(square)
        
    def remove_piece_at(self, square):
        if square in self.board: 
            del self.board[square]
            
    def piece_map(self): 
        return self.board
        
    def copy(self): 
        new_board = XiangqiBoard() 
        new_board.board = self.board.copy()
        return new_board

# --- Drawing Function ---
def draw_xiangqi_board(board, size=800, show_coordinates=False):
    """Generate SVG image of a Xiangqi board using svgwrite."""
    h_intervals, v_intervals = BOARD_WIDTH - 1.0, BOARD_HEIGHT - 1.0
    aspect_ratio = v_intervals / h_intervals
    
    # Increase padding to prevent pieces from touching edges
    padding = size / 12  # Increased padding (was size/30)
    
    # Calculate board dimensions with the increased padding
    board_width_px = size - (padding * 2)
    board_height_px = board_width_px * aspect_ratio
    
    total_width = board_width_px + padding * 2
    total_height = board_height_px + padding * 2
    
    board_color="#f9e9a9"
    grid_color="#000000"
    border_color="#d66500"
    
    # Create the SVG drawing
    dwg = svgwrite.Drawing(size=(f"{total_width:.2f}", f"{total_height:.2f}"), profile='full', debug=False)
    
    # Add border
    dwg.add(dwg.rect((0, 0), (total_width, total_height), fill=border_color))
    
    # Add board background
    grid_start_x, grid_start_y = padding, padding
    dwg.add(dwg.rect((grid_start_x, grid_start_y), (board_width_px, board_height_px), 
                     fill=board_color, stroke=grid_color, stroke_width=2))

    # Calculate grid intervals
    h_interval, v_interval = board_width_px / h_intervals, board_height_px / v_intervals
    line_style = {"stroke": grid_color, "stroke_width": 2.0}  # Increased from 1.5

    # Draw horizontal lines
    for i in range(BOARD_HEIGHT):
        y = grid_start_y + i * v_interval
        dwg.add(dwg.line((grid_start_x, y), (grid_start_x + board_width_px, y), **line_style))
    
    # Draw vertical lines
    for i in range(BOARD_WIDTH):
        x = grid_start_x + i * h_interval
        if i == 0 or i == (BOARD_WIDTH - 1):  # Full edge lines
            dwg.add(dwg.line((x, grid_start_y), (x, grid_start_y + board_height_px), **line_style))
        else:  # Inner lines with river gap
            dwg.add(dwg.line((x, grid_start_y), (x, grid_start_y + 4 * v_interval), **line_style))
            dwg.add(dwg.line((x, grid_start_y + 5 * v_interval), (x, grid_start_y + board_height_px), **line_style))

    # Draw palaces (diagonal lines in the palace areas)
    palace_w, palace_h = 2 * h_interval, 2 * v_interval
    
    # Bottom palace (Black side, rank 0-2)
    p1_x, p1_y = grid_start_x + 3 * h_interval, grid_start_y
    dwg.add(dwg.line((p1_x, p1_y), (p1_x + palace_w, p1_y + palace_h), **line_style))
    dwg.add(dwg.line((p1_x + palace_w, p1_y), (p1_x, p1_y + palace_h), **line_style))
    
    # Top palace (Red side, rank 7-9)
    p2_x, p2_y = grid_start_x + 3 * h_interval, grid_start_y + 7 * v_interval
    dwg.add(dwg.line((p2_x, p2_y), (p2_x + palace_w, p2_y + palace_h), **line_style))
    dwg.add(dwg.line((p2_x + palace_w, p2_y), (p2_x, p2_y + palace_h), **line_style))

    # Draw point markers for cannon and soldier positions
    # marker_offset, marker_len = h_interval * 0.15, h_interval * 0.1
    # marker_style = {"stroke": grid_color, "stroke_width": 1.5}  # Increased from 1
    
    # # Points to mark with corner indicators
    # points_to_mark = [(1, 2), (7, 2), (1, 7), (7, 7)] + \
    #                  [(f, r) for f in range(0, 9, 2) for r in [3, 6]]
    
    # for file, rank in points_to_mark:
    #     x, y = grid_start_x + file * h_interval, grid_start_y + rank * v_interval
        
    #     # Left markers if not on left edge
    #     if file > 0:
    #         dwg.add(dwg.line((x-marker_offset-marker_len, y-marker_offset),(x-marker_offset, y-marker_offset), **marker_style))
    #         dwg.add(dwg.line((x-marker_offset, y-marker_offset-marker_len),(x-marker_offset, y-marker_offset), **marker_style))
    #         dwg.add(dwg.line((x-marker_offset-marker_len, y+marker_offset),(x-marker_offset, y+marker_offset), **marker_style))
    #         dwg.add(dwg.line((x-marker_offset, y+marker_offset+marker_len),(x-marker_offset, y+marker_offset), **marker_style))
        
    #     # Right markers if not on right edge
    #     if file < (BOARD_WIDTH - 1):
    #         dwg.add(dwg.line((x+marker_offset+marker_len, y-marker_offset),(x+marker_offset, y-marker_offset), **marker_style))
    #         dwg.add(dwg.line((x+marker_offset, y-marker_offset-marker_len),(x+marker_offset, y-marker_offset), **marker_style))
    #         dwg.add(dwg.line((x+marker_offset+marker_len, y+marker_offset),(x+marker_offset, y+marker_offset), **marker_style))
    #         dwg.add(dwg.line((x+marker_offset, y+marker_offset+marker_len),(x+marker_offset, y+marker_offset), **marker_style))

    # Font Selection for rendering Chinese characters
    best_font = get_best_cjk_font()
    font_families = [f"'{f}'" for f in ([best_font] if best_font else []) + detect_cjk_fonts()]
    font_families.extend(["'Noto Sans CJK SC'", "'Microsoft YaHei'", "'SimSun'", "sans-serif"])
    font_family_str = ", ".join(list(dict.fromkeys(font_families)))  # Unique, order preserved

    # Draw Pieces
    piece_radius = min(h_interval, v_interval) * 0.8 / 2  # Reduced from 0.85 to 0.8
    
    for square, piece_symbol in board.piece_map().items():
        file, rank = square % BOARD_WIDTH, square // BOARD_WIDTH
        x, y = grid_start_x + file * h_interval, grid_start_y + rank * v_interval
        is_red = piece_symbol.isupper()
        piece_color = 'red' if is_red else 'black'
        char_symbol = XIANGQI_PIECES.get(piece_symbol, "?")

        # Draw piece background (cream colored circle)
        dwg.add(dwg.circle(center=(x, y), r=piece_radius, fill='#f9e9a9', stroke='black', stroke_width=1.5))  # Increased from 1
        
        # Draw piece inner circle (colored ring)
        dwg.add(dwg.circle(center=(x, y), r=piece_radius * 0.9, fill='none', 
                            stroke=piece_color, stroke_width=max(1.5, piece_radius * 0.08)))  # Increased from 1
        
        # Draw the character on the piece
        if char_symbol != "?":
            font_size = piece_radius * 1.2
            dwg.add(dwg.text(char_symbol, insert=(x, y), text_anchor="middle",
                             dominant_baseline="central", font_size=f"{font_size:.2f}px",
                             font_family=font_family_str, font_weight="bold", fill=piece_color))

    # Ensure SVG has proper XML declaration
    svg_string = dwg.tostring()
    if '<?xml' not in svg_string: 
        svg_string = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_string
    
    return svg_string

# --- Define standard piece counts ---
def get_standard_piece_counts():
    """Returns standard counts for each Chinese Chess piece type"""
    return {"General": 2, "Advisor": 4, "Elephant": 4, "Knight": 4, "Chariot": 4, 
            "Cannon": 4, "Soldier": 10, "total": 32}

# --- Action Selection Logic (Grouped by Square) ---
def select_chinesechess_action_groups(board, num_groups=12):
    """Selects target squares and defines Remove/Replace actions for Xiangqi."""
    occupied = list(board.piece_map().keys())
    if len(occupied) < num_groups:
        print(f"Fatal Error: Only {len(occupied)} pieces, need {num_groups} for groups.")
        return None
    print(f"Selecting {num_groups} target squares from {len(occupied)} pieces...")
    target_squares = random.sample(occupied, num_groups)

    action_groups = []
    all_piece_types = list(XIANGQI_PIECES.keys())  # All possible symbols

    for group_id, sq in enumerate(target_squares, 1):
        original_piece = board.get_piece_at(sq)
        if not original_piece: 
            continue

        remove_action = {'action': 'Remove', 'square': sq, 'original_piece': original_piece, 'new_piece': None}

        old_type_lower = original_piece.lower()
        possible_new_symbols = [p for p in all_piece_types 
                               if p.lower() != old_type_lower 
                               and p.islower() == original_piece.islower()]

        if not possible_new_symbols:
            print(f"Warning: Cannot find replacement for {original_piece} at {square_name(sq)} G{group_id}. Skip Replace.")
            replace_action = None
        else:
            new_piece_symbol = random.choice(possible_new_symbols)
            replace_action = {'action': 'Replace', 'square': sq, 
                             'original_piece': original_piece, 'new_piece': new_piece_symbol}

        if remove_action and replace_action:
            action_groups.append({'group_id': group_id, 'target_square': sq, 
                                 'actions': [remove_action, replace_action]})
        else: 
            print(f"Warning: Skipping group {group_id} due to missing action.")

    if len(action_groups) < num_groups: 
        print(f"Warning: Only created {len(action_groups)}/{num_groups} valid groups.")
    print(f"Defined {len(action_groups)} action groups for Chinese Chess.")
    return action_groups

# --- Single Action Generation Function (Group Context) - NOTITLE ONLY ---
def generate_single_chinesechess_notitle_image(group_id, action_details, original_board,
                                     dirs, pixel_sizes, quality_scale, svg_size,
                                     progress_bar, topic_name="xiangqi_pieces"):
    """Generates NOTITLE images and metadata for a single Chinese Chess action within a group."""
    metadata_entries = {'notitle': []}
    
    action = action_details['action']
    square = action_details['square']
    original_piece_symbol = action_details['original_piece']
    new_piece_symbol = action_details['new_piece']

    if not original_piece_symbol: 
        print(f"Error (Gen): Orig piece missing G{group_id} Sq:{square_name(square)}. Skip.")
        progress_bar.update(len(pixel_sizes))
        return {'notitle': []}

    square_name_str = square_name(square)
    original_piece_name = piece_to_name(original_piece_symbol)
    sanitized_original_name = sanitize_filename(original_piece_name)
    group_id_str = f"{group_id:03d}"
    action_type_str = action.lower()
    piece_added_name, piece_type_name_prompt = "N/A", "Unknown"

    modified_board = original_board.copy()
    if action == 'Remove': 
        modified_board.remove_piece_at(square)
    elif action == 'Replace' and new_piece_symbol:
        modified_board.set_piece_at(square, new_piece_symbol)
        piece_added_name = piece_to_name(new_piece_symbol)
        piece_type_name_prompt = PIECE_TYPE_NAMES_STD.get(new_piece_symbol.lower(), "Unknown")
    elif action == 'Replace': 
        print(f"Error (Gen): Replace details invalid G{group_id}. Skip.")
        progress_bar.update(len(pixel_sizes))
        return {'notitle': []}

    prompts, ground_truth, expected_bias = [], "", ""
    standard_counts = get_standard_piece_counts()
    topic_name_display = "Xiangqi Pieces"
    
    if action == 'Remove':
        gt_count = len(modified_board.piece_map())
        ground_truth = str(gt_count)
        prompts = [f"How many xiangqi pieces are there on this board? Answer with a number in curly brackets, e.g., {{9}}.", 
                  f"Count the xiangqi pieces on this board. Answer with a number in curly brackets, e.g., {{9}}."]
        expected_bias = str(standard_counts["total"])
    elif action == 'Replace':
        count_type = sum(1 for p in modified_board.piece_map().values() 
                         if p.lower() == new_piece_symbol.lower())
        ground_truth = str(count_type)
        prompts = [f"How many {piece_type_name_prompt} pieces are there on this board? Answer with a number in curly brackets, e.g., {{9}}.", 
                  f"Count the {piece_type_name_prompt} pieces on this board. Answer with a number in curly brackets, e.g., {{9}}."]
        expected_bias = str(standard_counts.get(piece_type_name_prompt, '?'))

    std_pos_prompt = f"Is this the Xiangqi starting position? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    std_pos_truth = "No"
    std_pos_bias = "Yes"

    for pixel_size in pixel_sizes:
        if action == 'Remove': 
            piece_info = f"{sanitized_original_name.lower()}_at_{square_name_str}"
        else: 
            piece_info = f"{sanitized_original_name.lower()}_to_{sanitize_filename(piece_added_name).lower()}_at_{square_name_str}"

        base_filename_part = f"{topic_name}_{group_id_str}_{action_type_str}_{piece_info}"
        final_filename = f"{base_filename_part}_notitle_px{pixel_size}.png"
        final_path = os.path.join(dirs["img_dirs"]['notitle'], final_filename)
        
        # Create SVG content
        svg_content = draw_xiangqi_board(modified_board, size=svg_size)
        adj_scale = quality_scale * (pixel_size / 768.0)
        aspect_ratio = (BOARD_HEIGHT - 1.0) / (BOARD_WIDTH - 1.0)
        
        # Generate the notitle image directly to final location
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        if not svg_to_png_direct(svg_content, final_path, adj_scale, pixel_size, 
                                maintain_aspect=True, aspect_ratio=aspect_ratio):
            print(f"Error: Failed to generate PNG for {final_filename}")
            progress_bar.update(1)
            continue
        
        # Update progress
        progress_bar.update(1)
        
        # --- Metadata Generation ---
        common_meta = {
            "group_id": group_id,
            "action": action,
            "modified_square": square_name_str,
            "piece_removed": original_piece_name,
            "piece_added": piece_added_name,
            "pixel": pixel_size
        }
        
        # Add metadata for Q1, Q2 prompts
        for p_idx, prompt in enumerate(prompts):
            q_type = f"Q{p_idx+1}"
            meta_id = f"{topic_name}_{group_id_str}_{action_type_str}_notitle_px{pixel_size}_prompt{p_idx+1}"
            metadata_entries['notitle'].append({
                "ID": meta_id, 
                "image_path": os.path.join("images", final_filename),
                "topic": topic_name_display, 
                "prompt": prompt,
                "ground_truth": ground_truth, 
                "expected_bias": expected_bias,
                "type_of_question": q_type,
                "pixel": pixel_size,
                "metadata": common_meta.copy()
            })
        
        # Add Q3 (Standard Position) for notitle
        meta_id_q3 = f"{topic_name}_{group_id_str}_{action_type_str}_notitle_px{pixel_size}_prompt3"
        metadata_entries['notitle'].append({
            "ID": meta_id_q3, 
            "image_path": os.path.join("images", final_filename),
            "topic": topic_name_display, 
            "prompt": std_pos_prompt,
            "ground_truth": std_pos_truth, 
            "expected_bias": std_pos_bias,
            "type_of_question": "Q3",
            "pixel": pixel_size,
            "metadata": common_meta.copy()
        })

    return metadata_entries

# --- Main Dataset Generation Function - NOTITLE ONLY ---
def create_chinesechess_notitle_dataset(dirs, quality_scale=5.0, svg_size=800, topic_name="xiangqi_pieces"):
    """Generates Chinese Chess dataset with NOTITLE images only."""
    print("=======================================================================")
    print(f"=== {topic_name.capitalize()} Dataset Generator - NOTITLE ONLY ===")
    print(f"    - Group IDs: {NUM_ACTION_GROUPS} (001 to {NUM_ACTION_GROUPS:03d})")
    print(f"    - Actions per Group: {ACTIONS_PER_GROUP} (Remove, Replace on same square)")
    print(f"    - Resolutions: {PIXEL_SIZES}")
    total_expected_images = NUM_ACTION_GROUPS * ACTIONS_PER_GROUP * len(PIXEL_SIZES)
    print(f"    - Total Expected Images: {total_expected_images}")
    print("=======================================================================")

    all_metadata = {'notitle': []}
    board = XiangqiBoard()
    
    action_groups = select_chinesechess_action_groups(board, NUM_ACTION_GROUPS)
    if not action_groups or len(action_groups) < NUM_ACTION_GROUPS:
        print(f"Fatal Error: Could not select {NUM_ACTION_GROUPS} valid action groups. Exiting.")
        sys.exit(1)

    progress_total = len(action_groups) * ACTIONS_PER_GROUP * len(PIXEL_SIZES)
    progress = tqdm(total=progress_total, desc=f"Processing {topic_name} Groups", unit="image", ncols=100)

    for group_data in action_groups:
        group_id = group_data['group_id']
        for action_details in group_data['actions']:
            metadata_result = generate_single_chinesechess_notitle_image(
                group_id, action_details, board, dirs,
                PIXEL_SIZES, quality_scale, svg_size, progress, topic_name
            )
            # Merge metadata results
            all_metadata['notitle'].extend(metadata_result['notitle'])

    progress.close()
    if progress.n < progress.total:
        print(f"Warning: Progress bar {progress.n}/{progress.total}. Check logs.")

    # --- Write Metadata ---
    print("\n--- Writing Metadata ---")
    save_metadata_files(
        all_metadata['notitle'], 
        dirs["meta_dirs"]['notitle'], 
        f"{topic_name}_notitle"
    )

    # --- Final Summary ---
    print("\n--- Summary ---")
    final_files = []
    try:
        if os.path.exists(dirs["img_dirs"]['notitle']):
            final_files = [f for f in os.listdir(dirs["img_dirs"]['notitle']) 
                          if f.endswith('.png')]
    except OSError as e:
        print(f"Warning: Could not count final images: {e}")

    total_actual = len(final_files)
    print(f"Generated {total_actual} total NOTITLE PNG images (Expected: {total_expected_images}).")
    print(f"Generated {len(all_metadata['notitle'])} metadata entries")
    print(f"\nOutput structure generated.")

    # Cleanup Temp
    try:
        if os.path.exists(dirs["temp_dir"]):
            shutil.rmtree(dirs["temp_dir"])
            print(f"\nTemp directory {dirs['temp_dir']} cleaned up.")
    except Exception as e:
        print(f"Warning: Failed cleanup '{dirs['temp_dir']}': {e}")