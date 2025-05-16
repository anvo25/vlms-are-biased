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
TITLE_TYPES = ['notitle', 'bias_amplifying', 'bias_mitigating']

# --- Helper Functions ---
def sanitize_filename(name):
    """Removes potentially problematic characters for filenames/foldernames."""
    name = str(name)
    name = re.sub(r'[^\w\-]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name if name else "sanitized_empty"

def svg_to_png_direct(svg_content, output_path, scale=2.0, output_size=768, maintain_aspect=True, aspect_ratio=1.0):
    """Convert SVG content directly to PNG file without saving SVG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        if maintain_aspect:
            output_height = int(output_size * aspect_ratio)
        else:
            output_height = output_size
            
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=output_path,
            scale=scale,
            background_color="white",
            output_width=output_size,
            output_height=output_height
        )
        return True
    except Exception as e:
        print(f"Error converting SVG to PNG ({output_path}): {e}")
        return False

def add_title_to_png(png_path, titled_png_path, title_text, font_size=48, max_width_chars=40):
    """Add title at the top of the image with line wrapping."""
    try:
        with PIL.Image.open(png_path) as img:
            if img.mode != 'RGBA': img = img.convert('RGBA')
            original_width, original_height = img.size
            
            # More aggressive scaling for smaller images
            if original_width <= 400:
                # For small images (like 384px), use smaller font and tighter wrapping
                dynamic_font_size = max(28, int(original_width / 28))
                max_width_chars = 40  # Shorter line width for small images
            elif original_width <= 800:
                # Medium images
                dynamic_font_size = max(55, int(original_width / (max_width_chars * 0.35)))
            else:
                # Large images
                dynamic_font_size = max(55, int(original_width / (max_width_chars * 0.35)))
                
            font_size = dynamic_font_size
            
            wrapped_lines = textwrap.wrap(title_text, width=max_width_chars, replace_whitespace=False, drop_whitespace=True)
                
            # Tighter line spacing for smaller images
            line_height = int(font_size * (1.0 if original_width <= 400 else 1.1))
                
            # Calculate required text height without padding
            text_height = len(wrapped_lines) * line_height
            
            # Add small equal padding to top and bottom (20% of font size)
            padding = int(font_size * 0.5)
            title_height = text_height + padding * 2
                
            new_height = original_height + title_height
            output_mode = 'RGB' if titled_png_path.lower().endswith(('.jpg', '.jpeg')) else 'RGBA'
            new_img = PIL.Image.new(output_mode, (original_width, new_height), (255, 255, 255, 255))
            new_img.paste(img, (0, title_height), mask=img if img.mode == 'RGBA' else None)
            draw = ImageDraw.Draw(new_img)
            
            # Try finding bold fonts, fallback to regular/default
            bold_fonts = ['arialbd.ttf','DejaVuSans-Bold.ttf','LiberationSans-Bold.ttf','arial.ttf','DejaVuSans.ttf']
            font = None
            for font_name in bold_fonts:
                try: font = ImageFont.truetype(font_name, font_size); break
                except IOError: continue
            if font is None:
                try: font = ImageFont.load_default(); print(f"Warning: Font not found for {titled_png_path}, using default.")
                except Exception as font_err: print(f"CRITICAL: Cannot load any font for {titled_png_path}: {font_err}"); return False
            
            # Start text at the padding position
            y_position = padding
            for line in wrapped_lines:
                try: bbox = draw.textbbox((0, 0), line, font=font); text_width = bbox[2] - bbox[0]
                except AttributeError: text_width, _ = draw.textsize(line, font=font) # Fallback
                x_position = max(0, (original_width - text_width) // 2)
                text_color = (0, 0, 0, 255) if output_mode == 'RGBA' else (0, 0, 0)
                draw.text((x_position, y_position), line, fill=text_color, font=font)
                y_position += line_height
            
            save_format = 'PNG' if output_mode == 'RGBA' else 'JPEG'
            new_img.save(titled_png_path, format=save_format, quality=95)
            return True
    except FileNotFoundError: print(f"Error adding title: Source PNG not found at {png_path}"); return False
    except PIL.UnidentifiedImageError: print(f"Error adding title: Cannot identify {png_path}"); return False
    except Exception as e: print(f"Error adding title to {titled_png_path}: {e}"); import traceback; traceback.print_exc(); return False

def load_title(topic, title_type):
    """Load a title from titles/{topic}/{title_type}.txt if available, or use default title."""
    title_dir = os.path.join("titles", topic)
    title_file = os.path.join(title_dir, f"{title_type}.txt")
    default_titles = {
        "notitle": "",  # No title
        "bias_amplifying": f"{topic.capitalize()}", # Simple name
        "bias_mitigating": f"{topic.capitalize()} (Modified)" # Mention of modification
    }
    
    try:
        if os.path.exists(title_file):
            with open(title_file, 'r', encoding='utf-8') as f:
                title = f.read().strip()
                return title if title else default_titles[title_type]
        else:
            return default_titles[title_type]
    except Exception as e:
        print(f"Error loading title for {topic}/{title_type}: {e}")
        return default_titles[title_type]

def create_directory_structure(topic_name, title_types=None):
    """Creates the directory structure for output files."""
    if title_types is None:
        title_types = TITLE_TYPES
        
    dirs_info = {
        "base_dirs": {},
        "img_dirs": {},
        "meta_dirs": {},
        "temp_dir": f"temp_{topic_name}_output"
    }
    
    # Ensure titles directory exists
    os.makedirs(os.path.join("titles", topic_name), exist_ok=True)
    
    # Create standard directories for each title type
    for title_type in title_types:
        base_dir = f"vlms-are-biased-{title_type}/{topic_name}"
        dirs_info["base_dirs"][title_type] = base_dir
        dirs_info["img_dirs"][title_type] = os.path.join(base_dir, "images")
        dirs_info["meta_dirs"][title_type] = base_dir
        
        # Create directories
        os.makedirs(dirs_info["img_dirs"][title_type], exist_ok=True)
        print(f"  Ensured: {dirs_info['img_dirs'][title_type]}")
    
    # Create temp directory
    os.makedirs(dirs_info["temp_dir"], exist_ok=True)
    print(f"  Ensured: {dirs_info['temp_dir']}")
    
    return dirs_info

def save_metadata_files(metadata_list, output_dir, base_name):
    """Saves metadata list to JSON and CSV files."""
    if not metadata_list:
        print(f"  No metadata for {base_name}, skipping.")
        return
    
    try: 
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e: 
        print(f"  ERROR creating {output_dir}: {e}")
        return

    # JSON
    json_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)
        print(f"  Wrote JSON: {json_path} ({len(metadata_list)} entries)")
    except (IOError, TypeError) as e:
        print(f"  ERROR writing JSON {json_path}: {e}")

    # CSV
    csv_path = os.path.join(output_dir, f"{base_name}_metadata.csv")
    try:
        flat_meta = []
        for entry in metadata_list:
            flat = entry.copy()
            meta_dict = flat.pop('metadata', {})
            for k, v in meta_dict.items():
                flat[f'meta_{k}'] = json.dumps(v) if isinstance(v, (dict, list)) else v
            flat_meta.append(flat)
            
        if not flat_meta:
            print(f"  No data for CSV {csv_path}.")
            return

        df = pd.DataFrame(flat_meta)
        # Define preferred column order - update if needed
        pref_cols = ['ID', 'image_path', 'topic', 'prompt', 'ground_truth', 'expected_bias', 
                     'pixel', 'title_type', 'type_of_question']
        meta_cols = sorted([c for c in df.columns if c.startswith('meta_')])
        
        ordered_cols = []
        remaining = list(df.columns)
        for col in pref_cols + meta_cols:
            if col in remaining:
                ordered_cols.append(col)
                remaining.remove(col)
        ordered_cols.extend(sorted(remaining)) # Add any other columns alphabetically

        df = df[ordered_cols]
        df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8')
        print(f"  Wrote CSV: {csv_path} ({len(metadata_list)} entries)")
    except Exception as e:
        print(f"  ERROR writing CSV {csv_path}: {e}")
        import traceback
        traceback.print_exc()

def create_default_titles(topics=None):
    """Creates default title files if they don't exist"""
    if topics is None:
        topics = ["chess_pieces", "xiangqi_pieces"]
    
    default_titles = {
        "chess_pieces": {
            "notitle": "",
            "bias_amplifying": "Standard Chess",
            "bias_mitigating": "Modified Chess Board"
        },
        "xiangqi_pieces": {
            "notitle": "",
            "bias_amplifying": "Xiangqi",  # Changed from "Chinese Chess"
            "bias_mitigating": "Modified Xiangqi Board"
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