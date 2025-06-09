import os
import argparse
import time
import sys
import importlib


project_root = os.path.dirname(os.path.abspath(__file__))
generators_path = os.path.join(project_root, "generators")
if generators_path not in sys.path:
    sys.path.insert(0, generators_path)

from utils import create_directory_structure, sanitize_filename
from generators import chess_pieces_generator
from generators import xiangqi_pieces_generator
from generators import chess_board_generator
from generators import go_board_generator
from generators import xiangqi_board_generator
from generators import sudoku_board_generator
from generators import patterned_grid_generator
from generators import optical_illusion_generator

try:
    pyllusion_spec = importlib.util.find_spec("Pyllusion")
    if pyllusion_spec is None:
        pyllusion_spec = importlib.util.find_spec("Pyllusion.pyllusion")
    HAS_PYLLUSION = pyllusion_spec is not None
except ImportError:
    HAS_PYLLUSION = False

if not HAS_PYLLUSION:
    print("Warning: Pyllusion package not found within the 'generators' directory or system paths.")
    print("Optical illusion generation will be skipped if selected.")


def setup_initial_dirs():
    os.makedirs("vlms-are-biased-notitle", exist_ok=True)
    os.makedirs("vlms-are-biased-in_image_title", exist_ok=True) 
    os.makedirs("titles", exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='ViasTest "notitle" Dataset Generator')
    parser.add_argument('--all', action='store_true', help='Generate all available datasets.')
    
    parser.add_argument('--chess_pieces', action='store_true', help='Generate Chess Pieces dataset.')
    parser.add_argument('--xiangqi_pieces', action='store_true', help='Generate Xiangqi Pieces dataset.')
    parser.add_argument('--chess_board', action='store_true', help='Generate Chess Board dataset.')
    parser.add_argument('--go_board', action='store_true', help='Generate Go Board dataset.')
    parser.add_argument('--xiangqi_board', action='store_true', help='Generate Xiangqi Board dataset.')
    parser.add_argument('--sudoku_board', action='store_true', help='Generate Sudoku Board dataset.')
    parser.add_argument('--patterned_grids', action='store_true', help='Generate Dice and Tally Patterned Grid datasets.')
    parser.add_argument('--optical_illusions', action='store_true', help='Generate Optical Illusion datasets.')
    
    parser.add_argument('--illusion_type', type=str, default="all",
                        choices=optical_illusion_generator.ALL_ILLUSION_TYPES + ["all"],
                        help='Specify a single optical illusion type or "all". Only used with --optical_illusions. (default: all)')

    parser.add_argument('--quality', type=float, default=5.0, help='PNG quality scale for SVG conversion (default: 5.0)')
    parser.add_argument('--svg_size', type=int, default=800, help='Base SVG size hint (default: 800)')
    
    args = parser.parse_args()
        
    setup_initial_dirs()
    
    run_all = args.all
    tasks_to_run = {
        'chess_pieces': run_all or args.chess_pieces,
        'xiangqi_pieces': run_all or args.xiangqi_pieces,
        'chess_board': run_all or args.chess_board,
        'go_board': run_all or args.go_board,
        'xiangqi_board': run_all or args.xiangqi_board,
        'sudoku_board': run_all or args.sudoku_board,
        'patterned_grids': run_all or args.patterned_grids,
        'optical_illusions': run_all or args.optical_illusions,
    }

    if not any(tasks_to_run.values()):
        parser.print_help()
        print("\nNo dataset selected for generation. Use --all or specify individual datasets.")
        return
    
    start_time = time.time()
    print("========== ViasTest 'notitle' Dataset Generator ==========")
    
    # --- Chess Pieces Dataset ---
    if tasks_to_run['chess_pieces']:
        print_task_header("Chess Pieces Dataset")
        try:
            # Assuming TOPIC_NAME is defined in the generator module
            output_dirs = create_directory_structure(chess_pieces_generator.TOPIC_NAME, title_types_to_create=['notitle'])
            chess_pieces_generator.create_chess_notitle_dataset(
                dirs=output_dirs,
                quality_scale=args.quality,
                svg_size=args.svg_size
            )
        except Exception as e:
            print_task_error("chess pieces", e)
    
    # --- Xiangqi Pieces Dataset ---
    if tasks_to_run['xiangqi_pieces']:
        print_task_header("Xiangqi Pieces Dataset")
        try:
            print("Detecting CJK fonts for Xiangqi pieces...")
            # These functions in the generator module will print their own status
            xiangqi_pieces_generator.detect_cjk_fonts() 
            xiangqi_pieces_generator.get_best_cjk_font() 
            
            output_dirs = create_directory_structure(xiangqi_pieces_generator.TOPIC_NAME, title_types_to_create=['notitle'])
            xiangqi_pieces_generator.create_chinesechess_notitle_dataset(
                dirs=output_dirs,
                quality_scale=args.quality,
                svg_size=args.svg_size
            )
        except Exception as e:
            print_task_error("xiangqi pieces", e)
            
    # --- Chess Board Dataset ---
    if tasks_to_run['chess_board']:
        print_task_header("Chess Board Dataset")
        try:
            chess_board_generator.create_chess_board_dataset(
                quality_scale=args.quality, 
                svg_size=args.svg_size
            )
        except Exception as e:
            print_task_error("chess board", e)
            
    # --- Go Board Dataset ---
    if tasks_to_run['go_board']:
        print_task_header("Go Board Dataset")
        try:
            go_board_generator.create_go_board_dataset(
                quality_scale=args.quality, 
                svg_size=args.svg_size
            )
        except Exception as e:
            print_task_error("go board", e)

    # --- Xiangqi Board Dataset ---
    if tasks_to_run['xiangqi_board']:
        print_task_header("Xiangqi Board Dataset")
        try:
            xiangqi_board_generator.create_chinese_chessboard_dataset(
                quality_scale=args.quality, 
                svg_size=args.svg_size
            )
        except Exception as e:
            print_task_error("xiangqi board", e)

    # --- Sudoku Board Dataset ---
    if tasks_to_run['sudoku_board']:
        print_task_header("Sudoku Board Dataset")
        try:
            sudoku_board_generator.create_sudoku_board_dataset(
                quality_scale=args.quality, 
                svg_size=args.svg_size
            )
        except Exception as e:
            print_task_error("sudoku board", e)

    # --- Patterned Grids (Dice and Tally) Dataset ---
    if tasks_to_run['patterned_grids']:
        print_task_header("Patterned Grids (Dice & Tally) Dataset")
        try:
            patterned_grid_generator.create_grid_dataset(
                quality_scale=args.quality, 
                svg_size=args.svg_size
            )
        except Exception as e:
            print_task_error("patterned grids", e)

    # --- Optical Illusions Dataset ---
    if tasks_to_run['optical_illusions']:
        print_task_header("Optical Illusions Dataset")
        if HAS_PYLLUSION: 
            try:
                illusion_to_gen = None if args.illusion_type.lower() == "all" else args.illusion_type
                optical_illusion_generator.main(specific_illusion=illusion_to_gen)
            except Exception as e:
                print_task_error("optical illusions", e)
        else:
            print("ERROR: Pyllusion library not found or not accessible. Cannot generate optical illusions.")
            print("Ensure Pyllusion is correctly placed in 'generators/Pyllusion/' and the directory has an __init__.py if it's a package.")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nGeneration of 'notitle' datasets completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")
    print("====================== 'notitle' Generation Complete ======================")
    print("\nNext step: Run 'python add_titles.py --topic all' (or specific topics) to generate 'in_image_title' versions.")

def print_task_header(task_name):
    print(f"\n----- Generating {task_name} -----")

def print_task_error(task_name, error):
    print(f"\n!!! ERROR generating {task_name}: {error} !!!")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    main()