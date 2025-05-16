# -*- coding: utf-8 -*-
import os
import argparse
import time
import sys
from utils import create_directory_structure, TITLE_TYPES
from chess_pieces import create_chess_notitle_dataset
from xiangqi_pieces import create_chinesechess_notitle_dataset, detect_cjk_fonts, get_best_cjk_font

def create_default_titles():
    """Creates default title files if they don't exist"""
    topics = ["chess_pieces", "xiangqi_pieces"]
    
    default_titles = {
        "chess_pieces": {
            "notitle": "",
            "bias_amplifying": "Standard Chess starting position",
            "bias_mitigating": "Non-standard Chess starting position"
        },
        "xiangqi_pieces": {  # Corrected spelling
            "notitle": "",
            "bias_amplifying": "Standard Xiangqi starting position", 
            "bias_mitigating": "Non-standard Xiangqi starting position"
        }
    }
    
    for topic in topics:
        title_dir = os.path.join("titles", topic)
        os.makedirs(title_dir, exist_ok=True)
        
        for title_type in TITLE_TYPES:
            title_file = os.path.join(title_dir, f"{title_type}.txt")
            if not os.path.exists(title_file):
                with open(title_file, 'w', encoding='utf-8') as f:
                    f.write(default_titles[topic].get(title_type, ""))
                print(f"Created default title file: {title_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate chess/xiangqi datasets with multiple title types')
    parser.add_argument('--game', type=str, choices=['chess', 'xiangqi', 'both'], default='both',
                      help='Which game to generate dataset for (default: both)')
    parser.add_argument('--quality', type=float, default=10.0, 
                      help='PNG quality scale (default: 10.0)')
    parser.add_argument('--svg-size', type=int, default=800, 
                      help='SVG base size hint (default: 800)')
    args = parser.parse_args()
    
    print("========== Chess Pieces Dataset Generator ==========")
    print(f"Selected game(s): {args.game}")
    print(f"Quality setting: {args.quality}")
    print(f"SVG size: {args.svg_size}")
    print(f"Title types: {TITLE_TYPES}")
    print("==================================================")
    
    # Create default title files
    create_default_titles()
    
    start_time = time.time()
    
    try:
        # Generate Western Chess dataset
        if args.game in ['chess', 'both']:
            print("\n----- Generating Western Chess Dataset -----")
            chess_dirs = create_directory_structure("chess_pieces")
            create_chess_notitle_dataset(
                dirs=chess_dirs,
                quality_scale=args.quality,
                svg_size=args.svg_size
            )
            
        # Generate Chinese Chess dataset
        if args.game in ['xiangqi', 'both']:
            print("\n----- Generating Xiangqi Dataset (NOTITLE ONLY) -----")  # Changed
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
        print(f"\n!!! FATAL ERROR: {e} !!!")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nGeneration completed in {duration:.2f} seconds.")
        if duration > 60:
            minutes, seconds = divmod(duration, 60)
            print(f"That's {int(minutes)} minutes and {seconds:.2f} seconds.")
        print("====================== Complete ======================")

if __name__ == "__main__":
    main()