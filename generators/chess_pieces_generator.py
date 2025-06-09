"""
Generates the "notitle" Chess Pieces dataset.
Modifies a standard chess starting position by removing or replacing one piece.
"""
import chess
import chess.svg
import os
import random
import shutil
from tqdm import tqdm
import sys

import matplotlib.font_manager
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

from utils import (sanitize_filename, svg_to_png_direct, 
                   save_metadata_files, TITLE_TYPES) 

PIXEL_SIZES = [384, 768, 1152] 
NUM_ACTION_GROUPS = 12  
ACTIONS_PER_GROUP = 2   
TOPIC_NAME = "chess_pieces"
# --- Helper Functions ---
def piece_to_name(piece_symbol_str):
    """Converts a chess piece symbol (e.g., 'P', 'n') to its full descriptive name."""
    mapping = {
        'P': 'White_Pawn', 'N': 'White_Knight', 'B': 'White_Bishop',
        'R': 'White_Rook', 'Q': 'White_Queen', 'K': 'White_King',
        'p': 'Black_Pawn', 'n': 'Black_Knight', 'b': 'Black_Bishop',
        'r': 'Black_Rook', 'q': 'Black_Queen', 'k': 'Black_King',
    }
    return mapping.get(str(piece_symbol_str), 'Unknown_Piece')

def get_standard_piece_counts():
    """Returns a dictionary of standard counts for each piece type in a chess game."""
    return {
        "Pawn": 16, "Knight": 4, "Bishop": 4, 
        "Rook": 4, "Queen": 2, "King": 2, 
        "total": 32
    }

def select_chess_action_groups(board, num_groups=NUM_ACTION_GROUPS):
    """
    Selects `num_groups` unique target squares from occupied squares on the board.
    For each target square, defines 'Remove' and 'Replace' actions.
    Returns a list of action groups, or None if selection fails.
    """
    occupied_squares = [sq for sq in chess.SQUARES if board.piece_at(sq)]
    if len(occupied_squares) < num_groups:
        print(f"  FATAL ERROR: Not enough pieces ({len(occupied_squares)}) on the board to select {num_groups} unique target squares for Chess Pieces.")
        return None

    print(f"  Selecting {num_groups} target squares for Chess Pieces modifications...")
    target_squares_for_groups = random.sample(occupied_squares, num_groups)

    action_groups_definitions = []
    all_chess_piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    for group_id_counter, target_square in enumerate(target_squares_for_groups, 1):
        original_piece_at_square = board.piece_at(target_square)
        if not original_piece_at_square:
            # This should not happen if we sampled from occupied_squares
            print(f"  Warning: No piece found at selected target square {chess.square_name(target_square)} for Group {group_id_counter}. Skipping this group.")
            continue

        # Define Remove Action
        remove_action_spec = {
            'action_type': 'Remove',
            'target_square_index': target_square,
            'original_piece_object': original_piece_at_square,
            'new_piece_object': None  # No new piece for removal
        }

        # Define Replace Action
        # Find a different piece type of the same color for replacement
        possible_replacement_types = [ptype for ptype in all_chess_piece_types if ptype != original_piece_at_square.piece_type]
        if not possible_replacement_types:
            print(f"  Warning: Cannot find a different piece type to replace {original_piece_at_square} at {chess.square_name(target_square)} for Group {group_id_counter}. Skipping Replace action.")
            replace_action_spec = None
        else:
            replacement_piece_type = random.choice(possible_replacement_types)
            replacement_piece_object = chess.Piece(replacement_piece_type, original_piece_at_square.color)
            replace_action_spec = {
                'action_type': 'Replace',
                'target_square_index': target_square,
                'original_piece_object': original_piece_at_square,
                'new_piece_object': replacement_piece_object
            }

        if remove_action_spec and replace_action_spec:
            action_groups_definitions.append({
                'group_id': group_id_counter,
                'target_square_index': target_square, # The common square for this group's actions
                'actions_in_group': [remove_action_spec, replace_action_spec]
            })
        else:
            print(f"  Warning: Could not define both Remove and Replace actions for square {chess.square_name(target_square)} in Group {group_id_counter}. Group skipped.")

    if len(action_groups_definitions) < num_groups:
         print(f"  Warning: Only {len(action_groups_definitions)} valid action groups were defined for Chess Pieces, out of {num_groups} requested.")
    
    print(f"  Defined {len(action_groups_definitions)} action groups for Chess Pieces.")
    return action_groups_definitions

def generate_single_chess_notitle_image_and_meta(
    group_id, action_details, base_board_state,
    output_dirs_struct, pixel_sizes_list, png_quality_scale, svg_render_size,
    progress_bar_updater, topic_name_constant=TOPIC_NAME):
    """
    Generates "notitle" images and metadata for a single chess piece action (Remove or Replace).
    `action_details` contains: 'action_type', 'target_square_index', 'original_piece_object', 'new_piece_object'.
    `output_dirs_struct` is the dict from `utils.create_directory_structure` for "notitle".
    Returns a list of metadata entries created for this action across all resolutions.
    """
    generated_metadata_for_action = []

    action_verb = action_details['action_type'] # 'Remove' or 'Replace'
    square_index = action_details['target_square_index']
    original_piece = action_details['original_piece_object']
    new_piece_if_replace = action_details['new_piece_object']

    if not original_piece:
        print(f"  Error (generate_single_chess): Original piece info missing for Group {group_id}, Square {chess.square_name(square_index)}. Action skipped.")
        progress_bar_updater.update(len(pixel_sizes_list)) # Account for skipped resolutions
        return []

    square_algebraic_name = chess.square_name(square_index)
    original_piece_descriptive_name = piece_to_name(original_piece.symbol())
    
    # Prepare names for filenames and metadata
    sanitized_orig_piece_name = sanitize_filename(original_piece_descriptive_name)
    group_id_formatted = f"{group_id:03d}" # e.g., 001, 012
    action_verb_lower = action_verb.lower()

    # Determine what piece is "added" or if it's just removal for prompts/metadata
    piece_added_descriptive_name = "N/A"
    piece_added_type_for_prompt = "N/A" # e.g., "Pawn", "Knight" (without color)

    # Create a mutable copy of the board for this specific action
    current_modified_board = base_board_state.copy()
    if action_verb == 'Remove':
        current_modified_board.set_piece_at(square_index, None)
    elif action_verb == 'Replace' and new_piece_if_replace:
        current_modified_board.set_piece_at(square_index, new_piece_if_replace)
        piece_added_descriptive_name = piece_to_name(new_piece_if_replace.symbol())
        piece_added_type_for_prompt = piece_added_descriptive_name.split('_')[-1] # "White_Pawn" -> "Pawn"
    elif action_verb == 'Replace' and not new_piece_if_replace:
         print(f"  Error (generate_single_chess): Replace action for Group {group_id} is missing new_piece_object. Action skipped.")
         progress_bar_updater.update(len(pixel_sizes_list))
         return []

    # Define prompts and ground truth answers based on the action
    prompts_q1_q2 = []
    ground_truth_q1_q2 = ""
    expected_bias_q1_q2 = ""
    standard_piece_counts = get_standard_piece_counts()

    if action_verb == 'Remove':
        current_total_pieces = len(current_modified_board.piece_map())
        ground_truth_q1_q2 = str(current_total_pieces)
        prompts_q1_q2 = [
            f"How many chess pieces are there on this board? Answer with a number in curly brackets, e.g., {{9}}.",
            f"Count the chess pieces on this board. Answer with a number in curly brackets, e.g., {{9}}."
        ]
        expected_bias_q1_q2 = str(standard_piece_counts["total"]) 
    elif action_verb == 'Replace':
        num_white_added_type = len(current_modified_board.pieces(new_piece_if_replace.piece_type, chess.WHITE))
        num_black_added_type = len(current_modified_board.pieces(new_piece_if_replace.piece_type, chess.BLACK))
        total_of_added_type = num_white_added_type + num_black_added_type
        ground_truth_q1_q2 = str(total_of_added_type)
        prompts_q1_q2 = [
            f"How many {piece_added_type_for_prompt} pieces are there on this board? Answer with a number in curly brackets, e.g., {{9}}.",
            f"Count the {piece_added_type_for_prompt} pieces on this board. Answer with a number in curly brackets, e.g., {{9}}."
        ]
        expected_bias_q1_q2 = str(standard_piece_counts.get(piece_added_type_for_prompt, "0"))

    prompt_q3 = "Is this the Chess starting position? Answer in curly brackets, e.g., {Yes} or {No}."
    ground_truth_q3 = "No"
    expected_bias_q3 = "Yes" 

    for px_size in pixel_sizes_list:
        if action_verb == 'Remove':
            piece_modification_info_str = f"{sanitized_orig_piece_name}_at_{square_algebraic_name}"
        else: # Replace
            sanitized_added_piece_name = sanitize_filename(piece_added_descriptive_name)
            piece_modification_info_str = f"{sanitized_orig_piece_name}_to_{sanitized_added_piece_name}_at_{square_algebraic_name}"

        img_basename = f"{topic_name_constant}_{group_id_formatted}_{action_verb_lower}_{piece_modification_info_str}_notitle_px{px_size}.png"
        
        final_png_path = os.path.join(output_dirs_struct["img_dirs"]['notitle'], img_basename)
        
        board_svg_content = chess.svg.board(board=current_modified_board, size=svg_render_size, coordinates=False)
        
        scale_for_conversion = png_quality_scale * (px_size / 768.0) # Assuming 768px is reference for base scale
        
        os.makedirs(os.path.dirname(final_png_path), exist_ok=True)
        
        # Convert SVG to PNG and save
        if not svg_to_png_direct(board_svg_content, final_png_path, scale_for_conversion, px_size):
            print(f"  ERROR: Failed to generate PNG for {img_basename}. Skipping this resolution.")
            progress_bar_updater.update(1) # Still update progress for the attempt
            continue 
        
        progress_bar_updater.update(1) 
        
        common_metadata_payload = {
             "group_id": group_id, # The group this action belongs to
             "action_type": action_verb, # 'Remove' or 'Replace'
             "modified_square_algebraic": square_algebraic_name,
             "piece_removed_info": original_piece_descriptive_name, # e.g., "White_Pawn"
             "piece_added_info": piece_added_descriptive_name,   # e.g., "White_Knight" or "N/A"
             "pixel_size": px_size,
        }
        
        image_path_relative = os.path.join("images", img_basename).replace("\\", "/")

        for q_idx, current_prompt_text in enumerate(prompts_q1_q2):
            question_type_label = f"Q{q_idx + 1}"
            meta_entry_id = f"{topic_name_constant}_{group_id_formatted}_{action_verb_lower}_{piece_modification_info_str}_notitle_px{px_size}_{question_type_label}"
            
            generated_metadata_for_action.append({
                 "ID": meta_entry_id, 
                 "image_path": image_path_relative,
                 "topic": "Chess Pieces", 
                 "prompt": current_prompt_text,
                 "ground_truth": ground_truth_q1_q2, 
                 "expected_bias": expected_bias_q1_q2,
                 "with_title": False, 
                 "type_of_question": question_type_label,
                 "pixel": px_size, 
                 "metadata": common_metadata_payload.copy()
            })
            
        # Create metadata entry for Q3
        meta_entry_id_q3 = f"{topic_name_constant}_{group_id_formatted}_{action_verb_lower}_{piece_modification_info_str}_notitle_px{px_size}_Q3"
        generated_metadata_for_action.append({
             "ID": meta_entry_id_q3, 
             "image_path": image_path_relative,
             "topic": "Chess Pieces", 
             "prompt": prompt_q3,
             "ground_truth": ground_truth_q3, 
             "expected_bias": expected_bias_q3,
             "with_title": False,
             "type_of_question": "Q3",
             "pixel": px_size,
             "metadata": common_metadata_payload.copy()
        })

    return generated_metadata_for_action

def create_chess_notitle_dataset(dirs, quality_scale=5.0, svg_size=800):
    """
    Generates the "notitle" Chess Pieces dataset.
    `dirs` is the structure returned by `utils.create_directory_structure` for "notitle".
    """
    current_topic_name = TOPIC_NAME
    
    print(f"  Starting {current_topic_name.replace('_',' ').title()} 'notitle' dataset generation...")
    print(f"    - Target Groups: {NUM_ACTION_GROUPS}, Actions per Group: {ACTIONS_PER_GROUP}")
    print(f"    - Resolutions: {PIXEL_SIZES}")
    total_images_to_generate = NUM_ACTION_GROUPS * ACTIONS_PER_GROUP * len(PIXEL_SIZES)
    print(f"    - Total images expected: {total_images_to_generate}")

    all_notitle_metadata_entries = []
    base_chess_board = chess.Board()  

    selected_action_groups = select_chess_action_groups(base_chess_board, NUM_ACTION_GROUPS)
    if not selected_action_groups or len(selected_action_groups) < NUM_ACTION_GROUPS:
         print(f"  FATAL ERROR: Could not define the required {NUM_ACTION_GROUPS} action groups for Chess Pieces. Generation aborted.")
         return

    progress_bar = tqdm(total=total_images_to_generate, 
                        desc=f"Generating Chess Pieces", unit="image", ncols=100)

    for group_definition in selected_action_groups:
        current_group_id = group_definition['group_id']
        for action_spec_details in group_definition['actions_in_group']:
            metadata_for_this_action = generate_single_chess_notitle_image_and_meta(
                current_group_id, 
                action_spec_details, 
                base_chess_board, 
                dirs, 
                PIXEL_SIZES, 
                quality_scale, 
                svg_size, 
                progress_bar, 
                current_topic_name
            )
            all_notitle_metadata_entries.extend(metadata_for_this_action)

    progress_bar.close()
    if progress_bar.n < progress_bar.total:
         print(f"  Warning: Chess Pieces generation progress ({progress_bar.n}/{progress_bar.total}) indicates some images may have been skipped. Check logs.")

    print(f"\n  Saving metadata for {current_topic_name}...")
    save_metadata_files(
        all_notitle_metadata_entries, 
        dirs["meta_dirs"]['notitle'], 
        f"{current_topic_name}_notitle" 
    )

    print(f"\n  --- Chess Pieces 'notitle' Generation Summary ---")
    final_image_files_count = 0
    try:
        img_output_dir = dirs["img_dirs"]['notitle']
        if os.path.exists(img_output_dir):
            final_image_files_count = len([f for f in os.listdir(img_output_dir) if f.endswith('.png')])
    except OSError as e:
        print(f"  Warning: Could not count final generated images in {img_output_dir}: {e}")

    print(f"  Actual 'notitle' images generated: {final_image_files_count} (Expected: {total_images_to_generate})")
    print(f"  Total 'notitle' metadata entries created: {len(all_notitle_metadata_entries)}")
    

    try:
        temp_dir_to_clean = dirs.get("temp_dir")
        if temp_dir_to_clean and os.path.exists(temp_dir_to_clean):
            shutil.rmtree(temp_dir_to_clean)
            print(f"  Cleaned up temporary directory: {temp_dir_to_clean}")
    except Exception as e:
        print(f"  Warning: Failed to clean up temp directory {temp_dir_to_clean}: {e}")
    
    print(f"  Chess Pieces 'notitle' dataset generation finished.")

if __name__ == '__main__':
    print("Testing Chess Pieces Generator directly...")
  
    test_topic_name = TOPIC_NAME
    from utils import create_directory_structure as util_create_dirs

    if not os.path.exists("vlms-are-biased-notitle"): 
        os.makedirs("vlms-are-biased-notitle")
    
    # Create 'notitle' directories for this topic
    test_output_dirs = util_create_dirs(test_topic_name, title_types_to_create=['notitle'])
    print(f"  Test output directories for 'notitle':")
    print(f"    Images: {test_output_dirs['img_dirs']['notitle']}")
    print(f"    Metadata: {test_output_dirs['meta_dirs']['notitle']}")
    print(f"    Temp: {test_output_dirs['temp_dir']}")

    create_chess_notitle_dataset(dirs=test_output_dirs, quality_scale=5.0, svg_size=800)
    print("\nDirect test of Chess Pieces Generator complete.")