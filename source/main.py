#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ViasTest Dataset Generator
Generate datasets for testing visual bias in vision language models.
"""

import os
import argparse
import time
import sys
import importlib

def setup_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs("titles", exist_ok=True)
    os.makedirs("titles/animals", exist_ok=True)
    os.makedirs("titles/logos", exist_ok=True)
    os.makedirs("titles/flags", exist_ok=True)
    os.makedirs("titles/chess_pieces", exist_ok=True)
    os.makedirs("titles/xiangqi_pieces", exist_ok=True)
    os.makedirs("titles/chess_grid", exist_ok=True)
    os.makedirs("titles/go_grid", exist_ok=True)
    os.makedirs("titles/xiangqi_grid", exist_ok=True)
    os.makedirs("titles/sudoku_grid", exist_ok=True)
    os.makedirs("titles/dice", exist_ok=True)
    os.makedirs("titles/tally", exist_ok=True)
    os.makedirs("titles/optical_illusion", exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='ViasTest Dataset Generator')
    parser.add_argument('--all', action='store_true', help='Generate all datasets')
    parser.add_argument('--chess_pieces', action='store_true', help='Generate chess pieces dataset')
    parser.add_argument('--xiangqi_pieces', action='store_true', help='Generate xiangqi pieces dataset')
    parser.add_argument('--chess_grid', action='store_true', help='Generate chess grid dataset')
    parser.add_argument('--go_grid', action='store_true', help='Generate go grid dataset')
    parser.add_argument('--xiangqi_grid', action='store_true', help='Generate xiangqi grid dataset')
    parser.add_argument('--sudoku_grid', action='store_true', help='Generate sudoku grid dataset')
    parser.add_argument('--dice_tally', action='store_true', help='Generate dice/tally dataset')
    parser.add_argument('--illusion', action='store_true', help='Generate optical illusion dataset')
    parser.add_argument('--quality', type=float, default=5.0, help='PNG quality scale (default: 5.0)')
    parser.add_argument('--svg_size', type=int, default=800, help='SVG base size hint (default: 800)')
    
    args = parser.parse_args()
    
    # Check if Pyllusion is available for optical illusion generation
    has_pyllusion = importlib.util.find_spec("pyllusion") is not None
    
    # Setup directories
    setup_dirs()
    
    # If no specific task is selected, show help
    if not (args.all or args.chess_pieces or args.xiangqi_pieces or args.chess_grid or 
            args.go_grid or args.xiangqi_grid or args.sudoku_grid or 
            args.dice_tally or args.illusion):
        parser.print_help()
        return
    
    start_time = time.time()
    print("========== ViasTest Dataset Generator ==========")
    
    # Generate chess pieces dataset
    if args.all or args.chess_pieces:
        print("\n----- Generating Chess Pieces Dataset -----")
        try:
            from utils import create_directory_structure, TITLE_TYPES
            from chess_pieces import create_chess_notitle_dataset
            
            chess_dirs = create_directory_structure("chess_pieces")
            create_chess_notitle_dataset(
                dirs=chess_dirs,
                quality_scale=args.quality,
                svg_size=args.svg_size
            )
        except Exception as e:
            print(f"\n!!! ERROR generating chess pieces: {e} !!!")
            import traceback
            traceback.print_exc()
    
    # Generate xiangqi pieces dataset
    if args.all or args.xiangqi_pieces:
        print("\n----- Generating Xiangqi Pieces Dataset -----")
        try:
            from utils import create_directory_structure, TITLE_TYPES
            from xiangqi_pieces import create_chinesechess_notitle_dataset, detect_cjk_fonts, get_best_cjk_font
            
            print("Detecting fonts...")
            cjk_fonts = detect_cjk_fonts()
            if cjk_fonts:
                best_font = get_best_cjk_font()
                print(f"-> Will use '{best_font}' in SVG font-family.")
            else:
                print("-> WARNING: No CJK fonts detected. Rendering may fail or use fallbacks.")
                
            xiangqi_dirs = create_directory_structure("xiangqi_pieces")
            create_chinesechess_notitle_dataset(
                dirs=xiangqi_dirs,
                quality_scale=args.quality,
                svg_size=args.svg_size
            )
        except Exception as e:
            print(f"\n!!! ERROR generating xiangqi pieces: {e} !!!")
            import traceback
            traceback.print_exc()
    
    # Generate chess grid dataset
    if args.all or args.chess_grid:
        print("\n----- Generating Chess Grid Dataset -----")
        try:
            from chess_grid_generator import create_chess_board_dataset
            create_chess_board_dataset(quality_scale=args.quality, svg_size=args.svg_size)
        except Exception as e:
            print(f"\n!!! ERROR generating chess grid: {e} !!!")
            import traceback
            traceback.print_exc()
    
    # Generate go grid dataset
    if args.all or args.go_grid:
        print("\n----- Generating Go Grid Dataset -----")
        try:
            from go_grid_generator import create_go_board_dataset
            create_go_board_dataset(quality_scale=args.quality, svg_size=args.svg_size)
        except Exception as e:
            print(f"\n!!! ERROR generating go grid: {e} !!!")
            import traceback
            traceback.print_exc()
    
    # Generate xiangqi grid dataset
    if args.all or args.xiangqi_grid:
        print("\n----- Generating Xiangqi Grid Dataset -----")
        try:
            from xiangqi_grid_generator import create_chinese_chessboard_dataset
            create_chinese_chessboard_dataset(quality_scale=args.quality, svg_size=args.svg_size)
        except Exception as e:
            print(f"\n!!! ERROR generating xiangqi grid: {e} !!!")
            import traceback
            traceback.print_exc()
    
    # Generate sudoku grid dataset
    if args.all or args.sudoku_grid:
        print("\n----- Generating Sudoku Grid Dataset -----")
        try:
            from sudoku_grid_generator import create_sudoku_grid_dataset
            create_sudoku_grid_dataset(quality_scale=args.quality, svg_size=args.svg_size)
        except Exception as e:
            print(f"\n!!! ERROR generating sudoku grid: {e} !!!")
            import traceback
            traceback.print_exc()
    
    # Generate dice/tally dataset
    if args.all or args.dice_tally:
        print("\n----- Generating Dice/Tally Dataset -----")
        try:
            from dice_tally_generator import create_grid_dataset
            create_grid_dataset(quality_scale=args.quality, svg_size=args.svg_size)
        except Exception as e:
            print(f"\n!!! ERROR generating dice/tally: {e} !!!")
            import traceback
            traceback.print_exc()
    
    # Generate optical illusion dataset
    if args.all or args.illusion:
        print("\n----- Generating Optical Illusion Dataset -----")
        if has_pyllusion:
            try:
                from optical_illusion_generator import main as generate_illusions
                generate_illusions()
            except Exception as e:
                print(f"\n!!! ERROR generating optical illusions: {e} !!!")
                import traceback
                traceback.print_exc()
        else:
            print("ERROR: Pyllusion not found. Cannot generate optical illusions.")
            print("Please install Pyllusion using: pip install pyllusion")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nGeneration completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")
    print("====================== Complete ======================")

if __name__ == "__main__":
    main()