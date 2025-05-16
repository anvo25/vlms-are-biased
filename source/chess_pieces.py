# -*- coding: utf-8 -*-
import chess
import chess.svg
import os
import random
import shutil
from tqdm import tqdm
import sys
from utils import (sanitize_filename, svg_to_png_direct, add_title_to_png, 
                   save_metadata_files, TITLE_TYPES, load_title)

# --- Global Constants ---
PIXEL_SIZES = [384, 768, 1152]
NUM_ACTION_GROUPS = 12  # Number of unique ID groups / target squares
ACTIONS_PER_GROUP = 2   # Remove, Replace

# --- Helper Functions ---
def piece_to_name(piece_symbol):
    """Convert chess piece symbol to standard name"""
    mapping = {
        'P': 'White_Pawn', 'N': 'White_Knight', 'B': 'White_Bishop',
        'R': 'White_Rook', 'Q': 'White_Queen', 'K': 'White_King',
        'p': 'Black_Pawn', 'n': 'Black_Knight', 'b': 'Black_Bishop',
        'r': 'Black_Rook', 'q': 'Black_Queen', 'k': 'Black_King',
    }
    return mapping.get(piece_symbol, 'Unknown')

# --- Define standard piece counts ---
def get_standard_piece_counts():
    """Returns standard counts for each piece type"""
    return {"Pawn": 16, "Knight": 4, "Bishop": 4, "Rook": 4, "Queen": 2, "King": 2, "total": 32}

# --- Action Selection Logic (Grouped by Square) ---
def select_chess_action_groups(board, num_groups=12):
    """
    Selects target squares and defines 'Remove' and 'Replace' actions for each square.
    """
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq)]
    if len(occupied) < num_groups:
        print(f"Fatal Error: Not enough pieces ({len(occupied)}) on the board to select {num_groups} unique target squares.")
        return None # Indicate failure

    print(f"Selecting {num_groups} target squares from {len(occupied)} occupied squares...")
    target_squares = random.sample(occupied, num_groups)

    action_groups = []
    all_piece_types = list(range(1, 7)) # P, N, B, R, Q, K (chess.PAWN etc.)

    for group_id, sq in enumerate(target_squares, 1):
        original_piece = board.piece_at(sq)
        if not original_piece: # Should not happen if selected from occupied
             print(f"Warning: No piece found at selected target square {chess.square_name(sq)} for Group {group_id}. Skipping group.")
             continue

        # Define Remove Action
        remove_action = {
            'action': 'Remove',
            'square': sq,
            'original_piece': original_piece,
            'new_piece': None
        }

        # Define Replace Action
        possible_new_types = [pt for pt in all_piece_types if pt != original_piece.piece_type]
        if not possible_new_types:
            print(f"Warning: Cannot find a different piece type to replace {original_piece} at {chess.square_name(sq)} for Group {group_id}. Skipping Replace action for this group.")
            replace_action = None # Indicate replace is not possible
        else:
            new_type = random.choice(possible_new_types)
            new_piece = chess.Piece(new_type, original_piece.color)
            replace_action = {
                'action': 'Replace',
                'square': sq,
                'original_piece': original_piece, # Keep track of what was originally there
                'new_piece': new_piece
            }

        # Only add group if both actions are defined (replace might fail in edge cases)
        if remove_action and replace_action:
            action_groups.append({
                'group_id': group_id,
                'target_square': sq,
                'actions': [remove_action, replace_action] # Store both actions
            })
        else:
             print(f"Warning: Could not define both actions for square {chess.square_name(sq)} in Group {group_id}. Skipping group.")

    if len(action_groups) < num_groups:
         print(f"Warning: Only able to define {len(action_groups)} valid action groups out of {num_groups} requested.")

    print(f"Defined {len(action_groups)} action groups.")
    return action_groups

# --- Single Action Generation Function (Refactored for Group Context) ---
def generate_single_chess_notitle_image(group_id, action_details, original_board,
                              dirs, pixel_sizes, quality_scale, svg_size,
                              progress_bar, topic_name="chess_pieces"):
    """
    Generates NOTITLE images and metadata for a single action (Remove or Replace).
    """
    metadata_entries = {'notitle': []}

    action = action_details['action']
    square = action_details['square']  # Target square for the group
    original_piece = action_details['original_piece']  # Piece originally at the square
    new_piece = action_details['new_piece']  # None for Remove, a Piece object for Replace

    if not original_piece:  # Should be caught earlier, but double-check
        print(f"Error (generate_single): Original piece missing for G{group_id} Sq:{chess.square_name(square)}. Skipping.")
        progress_bar.update(len(pixel_sizes))
        return {'notitle': []}

    # --- Prepare Details ---
    square_name = chess.square_name(square)
    original_piece_name_full = piece_to_name(str(original_piece))
    original_piece_display = original_piece_name_full.replace('_', ' ')
    sanitized_original_piece_name = sanitize_filename(original_piece_name_full)
    group_id_str = f"{group_id:03d}"
    action_type_str = action.lower()  # 'remove' or 'replace'

    piece_added_display = "N/A"
    piece_added_type_only = "N/A"  # Just 'Pawn', 'Knight' etc. for prompts

    # --- Perform Modification ---
    modified_board = original_board.copy()
    if action == 'Remove':
        modified_board.set_piece_at(square, None)
    elif action == 'Replace' and new_piece:
        modified_board.set_piece_at(square, new_piece)
        piece_added_name_full = piece_to_name(str(new_piece))
        piece_added_display = piece_added_name_full.replace('_', ' ')
        piece_added_type_only = piece_added_display.split(' ')[-1]  # Get 'Pawn' from 'White Pawn'
    elif action == 'Replace' and not new_piece:
         print(f"Error (generate_single): Replace action details invalid for G{group_id}. Skipping.")
         progress_bar.update(len(pixel_sizes))
         return {'notitle': []}

    # --- Define Prompts & Ground Truth ---
    prompts, ground_truth, expected_bias = [], "", ""
    standard_counts = get_standard_piece_counts()

    if action == 'Remove':
        gt_count = len(modified_board.piece_map())
        ground_truth = str(gt_count)
        prompts = ["How many chess pieces are there on this board? Answer with a number in curly brackets, e.g., {9}.",
                   "Count the chess pieces on this board. Answer with a number in curly brackets, e.g., {9}."]
        expected_bias = str(standard_counts["total"])
    elif action == 'Replace':
        piece_type_enum = new_piece.piece_type
        white_count = len(modified_board.pieces(piece_type_enum, chess.WHITE))
        black_count = len(modified_board.pieces(piece_type_enum, chess.BLACK))
        gt_count = white_count + black_count
        ground_truth = str(gt_count)
        prompts = [f"How many {piece_added_type_only} pieces are there on this board? Answer with a number in curly brackets, e.g., {{9}}.",
                   f"Count the {piece_added_type_only} pieces on this board. Answer with a number in curly brackets, e.g., {{9}}."]
        # Bias is the standard count for that piece type
        expected_bias = str(standard_counts.get(piece_added_type_only, "Unknown"))

    # Prompt 3: Standard position check (always No after modification)
    std_pos_prompt = "Is this the Chess starting position? Answer in curly brackets, e.g., {Yes} or {No}."
    std_pos_truth = "No"
    std_pos_bias = "Yes"  # Bias assumes it's the standard start

    # --- Generate Images and Metadata for each resolution ---
    for pixel_size in pixel_sizes:
        # Create detailed piece info string for filename
        if action == 'Remove':
            piece_info = f"{original_piece_display.lower().replace(' ', '')}_at_{square_name}"
        else:  # Replace
            piece_info = f"{original_piece_display.lower().replace(' ', '')}_to_{piece_added_display.lower().replace(' ', '')}_at_{square_name}"

        # Base filename for the notitle image
        base_filename_part = f"{topic_name}_{group_id_str}_{action_type_str}_{piece_info}"
        final_filename = f"{base_filename_part}_notitle_px{pixel_size}.png"
        final_path = os.path.join(dirs["img_dirs"]['notitle'], final_filename)
        
        # Create SVG content
        svg_content = chess.svg.board(board=modified_board, size=svg_size, style="filter:none;", coordinates=False)
        adj_scale = quality_scale * (pixel_size / 768.0)
        
        # Generate the notitle image directly to final location
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        if not svg_to_png_direct(svg_content, final_path, adj_scale, pixel_size):
            print(f"Error: Failed to generate PNG for {final_filename}")
            progress_bar.update(1)
            continue
        
        # Update progress
        progress_bar.update(1)
        
        # --- Metadata Generation ---
        common_meta = {
             "group_id": group_id,
             "action": action,
             "modified_square": square_name,
             "piece_removed": original_piece_display,
             "piece_added": piece_added_display,
             "pixel": pixel_size
        }
        
        # Add metadata for Q1, Q2 prompts
        for p_idx, prompt in enumerate(prompts):
            q_type = f"Q{p_idx+1}"
            meta_id = f"{topic_name}_{group_id_str}_{action_type_str}_notitle_px{pixel_size}_prompt{p_idx+1}"
            metadata_entries['notitle'].append({
                 "ID": meta_id, 
                 "image_path": os.path.join("images", final_filename),
                 "topic": "Chess Pieces", 
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
             "topic": "Chess Pieces", 
             "prompt": std_pos_prompt,
             "ground_truth": std_pos_truth, 
             "expected_bias": std_pos_bias,
             "type_of_question": "Q3",
             "pixel": pixel_size,
             "metadata": common_meta.copy()
        })

    return metadata_entries

# --- Main Dataset Generation Function (Notitle Only) ---
def create_chess_notitle_dataset(dirs, quality_scale=5.0, svg_size=800, topic_name="chess_pieces"):
    """
    Generates a chess dataset with NOTITLE images only.
    """
    print("=======================================================================")
    print(f"=== {topic_name.capitalize()} Dataset Generator - NOTITLE ONLY ===")
    print(f"    - Group IDs: {NUM_ACTION_GROUPS} (001 to {NUM_ACTION_GROUPS:03d})")
    print(f"    - Actions per Group: {ACTIONS_PER_GROUP} (Remove, Replace on same square)")
    print(f"    - Resolutions: {PIXEL_SIZES}")
    total_expected_images = NUM_ACTION_GROUPS * ACTIONS_PER_GROUP * len(PIXEL_SIZES)
    print(f"    - Total Expected Images: {total_expected_images}")
    print("=======================================================================")

    all_metadata = {'notitle': []}
    board = chess.Board()  # Standard starting board

    # --- Select Action Groups ---
    action_groups = select_chess_action_groups(board, NUM_ACTION_GROUPS)
    if action_groups is None or len(action_groups) != NUM_ACTION_GROUPS:
         print(f"Fatal Error: Could not select the required {NUM_ACTION_GROUPS} action groups. Exiting.")
         sys.exit(1)

    # --- Generation Loop (Iterate through Groups) ---
    progress_total = len(action_groups) * ACTIONS_PER_GROUP * len(PIXEL_SIZES)
    progress = tqdm(total=progress_total, desc=f"Processing {topic_name} Groups", unit="image", ncols=100)

    for group_data in action_groups:
        group_id = group_data['group_id']
        # Generate images for both actions (Remove, Replace) in the group
        for action_details in group_data['actions']:
            metadata_result = generate_single_chess_notitle_image(
                group_id, action_details, board, dirs,
                PIXEL_SIZES, quality_scale, svg_size, progress, topic_name
            )
            # Merge metadata results
            all_metadata['notitle'].extend(metadata_result['notitle'])

    progress.close()
    if progress.n < progress.total:
         print(f"Warning: Progress bar finished at {progress.n}/{progress.total}. Check logs for errors.")

    # --- Write Metadata ---
    print("\n--- Writing Metadata ---")
    save_metadata_files(
        all_metadata['notitle'], 
        dirs["meta_dirs"]['notitle'], 
        f"{topic_name}_notitle"
    )

    # --- Final Summary ---
    print("\n--- Summary ---")
    print(f"Dataset generation finished.")
    # Count actual files
    final_files = []
    try:
        if os.path.exists(dirs["img_dirs"]['notitle']):
            final_files = [f for f in os.listdir(dirs["img_dirs"]['notitle']) if f.endswith('.png')]
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