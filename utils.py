# utils.py
# -*- coding: utf-8 -*-
import os
import re
import json
import csv
import pandas as pd
import shutil
import PIL.Image
from PIL import ImageDraw, ImageFont
import cairosvg 
import textwrap

# --- Global Constants ---
TITLE_TYPES = ['notitle', 'in_image_title']

# --- Default Title Texts for 'in_image_title' if file is missing/empty ---
DEFAULT_IN_IMAGE_TITLES = {
    "chess_pieces": "Chess starting position", # Paper uses "Chess starting position"
    "xiangqi_pieces": "Xiangqi starting position",
    "chess_board": "Chess", # Paper uses "Chess"
    "sudoku_board": "Sudoku", # Paper uses "Sudoku"
    "go_board": "Go", # Paper uses "Go"
    "xiangqi_board": "Xiangqi",
    "ebbinghaus_illusion": "Ebbinghaus illusion",
    "mullerlyer_illusion": "Müller-Lyer illusion",
    "poggendorff_illusion": "Poggendorff illusion",
    "ponzo_illusion": "Ponzo illusion",
    "verticalhorizontal_illusion": "Vertical-Horizontal illusion",
    "zollner_illusion": "Zöllner illusion",
    "dice_patterned_grid": "Patterned Grid",
    "tally_patterned_grid": "Patterned Grid",
    # Add defaults for animals, logos, flags if they become active
    "animals": "", # Placeholder, no default title
    "logos": "", # Placeholder, no default title
    "flags": "", # Placeholder, no default title
}


# --- Helper Functions ---
def sanitize_filename(name):
    """Removes potentially problematic characters for filenames/foldernames."""
    name = str(name)
    name = re.sub(r'[\\/*?:"<>|\s]+', '_', name)
    name = re.sub(r'[^\w\-]+', '', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name if name else "sanitized_empty"

def svg_to_png_direct(svg_content, output_path, scale=2.0, output_size=768, output_height=None, maintain_aspect=True, aspect_ratio=None):
    """Convert SVG content directly to PNG file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    target_output_width = output_size
    target_output_height = output_height if output_height is not None else output_size

    if maintain_aspect and aspect_ratio:
        target_output_height = int(output_size * aspect_ratio)
    elif output_height is None: 
        target_output_height = output_size 

    try:
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=output_path,
            scale=scale,
            background_color="white",
            output_width=target_output_width,
            output_height=target_output_height
        )
        return True
    except Exception as e:
        print(f"Error converting SVG to PNG ({output_path}): {e}")
        return False

def add_title_to_png(png_path, titled_png_path, title_text, font_size_initial=48, max_width_chars=40):
    """Adds a title bar at the top of a PNG image with wrapped text."""
    try:
        with PIL.Image.open(png_path) as img:
            if img.mode != 'RGBA': 
                img = img.convert('RGBA')
            
            original_width, original_height = img.size
            font_scale_factor = original_width / 800.0 
            font_size = max(20, int(font_size_initial * font_scale_factor)) 
            adjusted_max_width_chars = int(max_width_chars * (font_size_initial / font_size)) if font_size > 0 else max_width_chars
            adjusted_max_width_chars = max(15, adjusted_max_width_chars) 
            wrapped_lines = textwrap.wrap(title_text, width=adjusted_max_width_chars, replace_whitespace=False, drop_whitespace=True)
            line_height_factor = 1.2 
            line_height = int(font_size * line_height_factor)
            text_block_height = len(wrapped_lines) * line_height
            padding_vertical = int(font_size * 0.5) 
            title_bar_height = text_block_height + (padding_vertical * 2)
            new_height = original_height + title_bar_height
            output_mode = 'RGB' if titled_png_path.lower().endswith(('.jpg', '.jpeg')) else 'RGBA'
            new_img = PIL.Image.new(output_mode, (original_width, new_height), (255, 255, 255, 255)) 
            new_img.paste(img, (0, title_bar_height), mask=img if img.mode == 'RGBA' else None)
            draw = ImageDraw.Draw(new_img)
            bold_fonts_try_order = ['arialbd.ttf', 'DejaVuSans-Bold.ttf', 'LiberationSans-Bold.ttf', 'arial.ttf', 'DejaVuSans.ttf']
            font = None
            for font_name in bold_fonts_try_order:
                try: 
                    font = ImageFont.truetype(font_name, font_size)
                    break 
                except IOError: 
                    continue 
            if font is None: 
                try: 
                    font = ImageFont.load_default() 
                    print(f"Warning: Truetype font not found for title on '{os.path.basename(titled_png_path)}'. Using default bitmap font. Title quality may be lower.")
                except Exception as font_err: 
                    print(f"CRITICAL FONT ERROR: Cannot load any font for '{os.path.basename(titled_png_path)}': {font_err}")
                    if png_path != titled_png_path: shutil.copy2(png_path, titled_png_path)
                    return False 
            current_y = padding_vertical 
            for line in wrapped_lines:
                try: 
                    bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = bbox[2] - bbox[0]
                except AttributeError: 
                    text_width, _ = draw.textsize(line, font=font) 
                x_position = max(0, (original_width - text_width) // 2) 
                text_color = (0, 0, 0, 255) if output_mode == 'RGBA' else (0, 0, 0) 
                draw.text((x_position, current_y), line, fill=text_color, font=font)
                current_y += line_height 
            save_format = 'PNG' if output_mode == 'RGBA' else 'JPEG'
            new_img.save(titled_png_path, format=save_format, quality=95)
            return True
    except FileNotFoundError: 
        print(f"Error adding title: Source PNG not found at {png_path}")
        return False
    except PIL.UnidentifiedImageError: 
        print(f"Error adding title: Cannot identify image file (may be corrupted) at {png_path}")
        return False
    except Exception as e: 
        print(f"An unexpected error occurred while adding title to {titled_png_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_title(topic_key, title_type):
    """
    Loads a title for a given topic and title type.
    `topic_key` should match keys in DEFAULT_IN_IMAGE_TITLES or be a sanitized topic name.
    If a .txt file exists in titles/<topic_sanitized>/<title_type>.txt, it's used.
    Otherwise, a default is returned. For 'notitle', it's always "".
    For 'in_image_title', it uses DEFAULT_IN_IMAGE_TITLES.
    """
    if title_type == "notitle":
        return ""

    topic_sanitized = sanitize_filename(topic_key.lower())
    title_dir = os.path.join("titles", topic_sanitized)
    title_file_path = os.path.join(title_dir, f"{title_type}.txt") # e.g., in_image_title.txt

    # Determine the default title content
    default_title_content = DEFAULT_IN_IMAGE_TITLES.get(topic_key, topic_key.replace('_', ' ').title())
    if title_type != "in_image_title": # Should not happen with current TITLE_TYPES
        default_title_content = topic_key.replace('_', ' ').title()


    try:
        if os.path.exists(title_file_path):
            with open(title_file_path, 'r', encoding='utf-8') as f:
                title_from_file = f.read().strip()
            if title_from_file: # Use file content if it's not empty
                return title_from_file
            else: # File exists but is empty, use the default
                print(f"  Note: Title file '{title_file_path}' is empty, using default title: '{default_title_content}'")
                return default_title_content
        else:
            # File does not exist, create it with the default content for 'in_image_title'
            if title_type == "in_image_title":
                os.makedirs(title_dir, exist_ok=True) # Ensure directory exists
                with open(title_file_path, 'w', encoding='utf-8') as f_write:
                    f_write.write(default_title_content)
                print(f"  Note: Title file '{title_file_path}' not found. Created with default title: '{default_title_content}'")
            return default_title_content
    except Exception as e:
        print(f"Error loading or creating title file {title_file_path} for {topic_key}/{title_type}: {e}")
        return default_title_content # Fallback to default

def create_topic_title_dirs(topic_name_sanitized):
    """
    Ensures the specific title directory for a topic exists (e.g., titles/chess_pieces/).
    It no longer creates default .txt files here; `load_title` handles that.
    """
    title_dir = os.path.join("titles", topic_name_sanitized)
    os.makedirs(title_dir, exist_ok=True)
    # Default file creation is now handled by load_title if file is missing

def create_directory_structure(topic_name_sanitized, title_types_to_create=None):
    """Creates the directory structure for output files for a given topic."""
    if title_types_to_create is None:
        title_types_to_create = TITLE_TYPES 

    dirs_info = {
        "base_dirs": {}, "img_dirs": {}, "meta_dirs": {},
        "temp_dir": f"temp_{topic_name_sanitized}_output"
    }
    
    os.makedirs("titles", exist_ok=True)
    create_topic_title_dirs(topic_name_sanitized) 

    for title_type in title_types_to_create:
        base_dir_for_type = f"vlms-are-biased-{title_type}"
        topic_base_dir = os.path.join(base_dir_for_type, topic_name_sanitized)
        dirs_info["base_dirs"][title_type] = topic_base_dir
        dirs_info["img_dirs"][title_type] = os.path.join(topic_base_dir, "images")
        dirs_info["meta_dirs"][title_type] = topic_base_dir 
        os.makedirs(dirs_info["img_dirs"][title_type], exist_ok=True)
    
    os.makedirs(dirs_info["temp_dir"], exist_ok=True)
    return dirs_info

def save_metadata_files(metadata_list, output_dir, base_filename_prefix):
    """Saves metadata list to JSON and CSV files."""
    if not metadata_list:
        print(f"  INFO: No metadata entries to save for {base_filename_prefix}.")
        return
    
    try: 
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e: 
        print(f"  ERROR: Could not create output directory {output_dir}: {e}")
        return

    json_filename = f"{base_filename_prefix}_metadata.json"
    json_path = os.path.join(output_dir, json_filename)
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        print(f"  Successfully wrote JSON: {json_path} ({len(metadata_list)} entries)")
    except (IOError, TypeError) as e:
        print(f"  ERROR writing JSON file {json_path}: {e}")

    csv_filename = f"{base_filename_prefix}_metadata.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    try:
        flat_meta_for_csv = []
        for entry in metadata_list:
            flat_entry = entry.copy()
            meta_dict = flat_entry.pop('metadata', {}) 
            if isinstance(meta_dict, dict):
                for k, v in meta_dict.items():
                    flat_entry[f'meta_{k}'] = json.dumps(v) if isinstance(v, (dict, list)) else v
            elif meta_dict is not None: 
                 flat_entry['meta_value'] = meta_dict
            flat_meta_for_csv.append(flat_entry)
            
        if not flat_meta_for_csv:
            print(f"  INFO: No data structure to write to CSV for {base_filename_prefix}.")
            return

        df = pd.DataFrame(flat_meta_for_csv)
        preferred_cols_order = [
            'ID', 'image_path', 'topic', 'prompt', 
            'ground_truth', 'expected_bias', 'with_title', 
            'type_of_question' 
        ]
        meta_cols_present = sorted([col for col in df.columns if col.startswith('meta_')])
        final_cols_ordered = []
        for col in preferred_cols_order:
            if col in df.columns:
                final_cols_ordered.append(col)
        final_cols_ordered.extend(meta_cols_present)
        remaining_cols = sorted([col for col in df.columns if col not in final_cols_ordered])
        final_cols_ordered.extend(remaining_cols)
        df = df[final_cols_ordered] 
        df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8')
        print(f"  Successfully wrote CSV: {csv_path} ({len(df)} rows)")
    except Exception as e:
        print(f"  ERROR writing CSV file {csv_path}: {e}")
        import traceback
        traceback.print_exc()