# grid_utils.py
# -*- coding: utf-8 -*-
"""
Common utility functions for all grid counting board generators.
"""
import os
import re
import json
import shutil
import pandas as pd
import PIL.Image
from PIL import ImageDraw, ImageFont
import textwrap

try:
    from cairosvg import svg2png
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False
    print("WARNING: cairosvg not installed. Will use alternative methods for SVG conversion.")

def sanitize_filename(name):
    """Removes potentially problematic characters for filenames/foldernames."""
    name = str(name)
    # Replace spaces and common problematic chars with underscore
    name = re.sub(r'[\\/*?:"<>|\s]+', '_', name)
    # Remove any remaining non-alphanumeric (excluding underscore and hyphen)
    name = re.sub(r'[^\w\-]+', '', name)
    # Consolidate multiple underscores
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name if name else "sanitized_empty"

def create_directory_structure(board_type):
    """Creates the directory structure for saving files"""
    board_type_lower = board_type.lower().replace(" ", "")
    
    # With title directories
    # withtitle_img_dir = f"vlms-are-biased-withtitle/{board_type_lower}/images"
    # withtitle_json_dir = f"vlms-are-biased-withtitle/{board_type_lower}"
    # withtitle_csv_dir = f"vlms-are-biased-withtitle/{board_type_lower}"
    
    # No title directories
    notitle_img_dir = f"vlms-are-biased-notitle/{board_type_lower}/images"
    notitle_json_dir = f"vlms-are-biased-notitle/{board_type_lower}"
    notitle_csv_dir = f"vlms-are-biased-notitle/{board_type_lower}"
    
    # Create all directories
    directories = [
        # withtitle_img_dir, withtitle_json_dir, withtitle_csv_dir,
        notitle_img_dir, notitle_json_dir, notitle_csv_dir
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Also create a temp directory for intermediate processing
    temp_dir = f"temp_{board_type_lower}_output"
    os.makedirs(temp_dir, exist_ok=True)
    
    return {
        # "withtitle_img_dir": withtitle_img_dir,
        # "withtitle_json_dir": withtitle_json_dir, 
        # "withtitle_csv_dir": withtitle_csv_dir,
        "notitle_img_dir": notitle_img_dir,
        "notitle_json_dir": notitle_json_dir,
        "notitle_csv_dir": notitle_csv_dir,
        "temp_dir": temp_dir
    }

def svg_to_png_direct(svg_content, output_path, scale=2.0, output_size=768, output_height=None, maintain_aspect=True, aspect_ratio=None):
    """Convert SVG content directly to PNG file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if maintain_aspect and aspect_ratio:
        output_height = int(output_size * aspect_ratio)
    elif not output_height:
        output_height = output_size  # Default to square unless specified
    
    try:
        if HAS_CAIROSVG:
            # Optional: Save SVG for debugging
            # svg_temp_path = output_path.replace('.png', '.svg')
            # with open(svg_temp_path, 'w', encoding='utf-8') as f:
            #     f.write(svg_content)
            
            svg2png(
                bytestring=svg_content.encode('utf-8'),
                write_to=output_path,
                scale=scale,
                background_color="white",
                output_width=output_size,
                output_height=output_height
            )
            return True
        else:
            # Fallback implementation
            # ...
            pass
    except Exception as e:
        print(f"Error converting SVG to PNG ({output_path}): {e}")
        return False


# def add_title_to_png(png_path, titled_png_path, title_text, font_size=36):
#     """
#     Add bold title directly in the middle of the board without changing dimensions.
#     Uses a semi-transparent overlay to ensure text visibility.
#     """
#     try:
#         with PIL.Image.open(png_path) as img:
#             new_img = img.copy().convert('RGBA')
#             width, height = new_img.size
#             draw = ImageDraw.Draw(new_img)

#             # Try to use bold fonts, fall back to regular
#             bold_fonts = [
#                 'arialbd.ttf', 'Arial Bold.ttf', 'DejaVuSans-Bold.ttf',
#                 'Helvetica-Bold.ttf', 'Verdana-Bold.ttf', 'Georgia-Bold.ttf',
#                 'FreeSansBold.ttf', 'LiberationSans-Bold.ttf',
#                 'arial.ttf', 'DejaVuSans.ttf'  # Fallbacks
#             ]
#             font = None
#             for font_name in bold_fonts:
#                 try:
#                     font = ImageFont.truetype(font_name, font_size)
#                     break
#                 except IOError:
#                     continue

#             if font is None:
#                 try:
#                     font = ImageFont.load_default()
#                     print(f"Warning: No suitable bold/regular font found for {titled_png_path}, using default font")
#                 except Exception as font_err:
#                     print(f"Critical Error: Could not load any font. Cannot add title. Error: {font_err}")
#                     return False

#             # Get text size using textbbox for accuracy
#             try:
#                 bbox = draw.textbbox((0, 0), title_text, font=font)
#                 text_width = bbox[2] - bbox[0]
#                 text_height = bbox[3] - bbox[1]
#             except AttributeError:  # Fallback for older Pillow
#                 text_width, text_height_approx = draw.textsize(title_text, font=font)
#                 text_height = font_size  # Approximation

#             # Reduce font size if text is too wide
#             max_width = width * 0.8
#             while text_width > max_width and font_size > 15:
#                 font_size -= 2
#                 font = None  # Re-attempt finding bold font at new size
#                 for font_name in bold_fonts:
#                     try: 
#                         font = ImageFont.truetype(font_name, font_size)
#                         break
#                     except IOError: 
#                         continue
#                 if font is None:
#                     try: 
#                         font = ImageFont.load_default()
#                     except Exception: 
#                         print("Error: Failed loading default font during resize.")
#                         return False

#                 try:  # Recalculate size
#                     bbox = draw.textbbox((0, 0), title_text, font=font)
#                     text_width = bbox[2] - bbox[0]
#                     text_height = bbox[3] - bbox[1]
#                 except AttributeError:
#                     text_width, _ = draw.textsize(title_text, font=font)
#                     text_height = font_size

#             # Calculate position
#             x_position = (width - text_width) // 2
#             y_position = (height - text_height) // 2

#             # Semi-transparent overlay
#             padding = 20
#             overlay = PIL.Image.new('RGBA', new_img.size, (0, 0, 0, 0))
#             overlay_draw = ImageDraw.Draw(overlay)
#             rect_coords = [
#                 max(0, x_position - padding), max(0, y_position - padding),
#                 min(width, x_position + text_width + padding), min(height, y_position + text_height + padding)
#             ]
#             overlay_draw.rectangle(rect_coords, fill=(255, 255, 255, 200))  # White overlay

#             # Composite and draw text
#             new_img = PIL.Image.alpha_composite(new_img, overlay)
#             draw = ImageDraw.Draw(new_img)  # Recreate draw object
#             text_color = (0, 0, 0, 255)  # Black
            
#             # Draw text slightly offset for pseudo-bold effect
#             draw.text((x_position+1, y_position+1), title_text, fill=text_color, font=font)
#             draw.text((x_position, y_position), title_text, fill=text_color, font=font)

#             # Convert back to RGB if necessary (PNG supports RGBA)
#             if titled_png_path.lower().endswith(('.jpg', '.jpeg')):
#                 new_img = new_img.convert('RGB')

#             new_img.save(titled_png_path, format='PNG', quality=95)
#             return True

#     except Exception as e:
#         print(f"Error adding title to PNG ({titled_png_path}): {e}")
#         if "cannot open resource" in str(e) or "font not found" in str(e):
#             print("Hint: Make sure standard fonts like Arial or DejaVu Sans are installed.")
#         return False

def add_title_to_png(png_path, titled_png_path, title_text, font_size=36, max_width_chars=40):
    """
    Add bold title at the top of the image with line wrapping.
    Creates a header bar with the wrapped title at the top of the image.

    Args:
        png_path: Path to the source PNG
        titled_png_path: Path to save titled PNG
        title_text: Title text to add
        font_size: Base font size (will be adjusted for resolutions)
        max_width_chars: Maximum characters per line for wrapping
    """
    try:
        with PIL.Image.open(png_path) as img:
            if img.mode != 'RGBA':
                 img = img.convert('RGBA')
            original_width, original_height = img.size
            wrapped_lines = textwrap.wrap(title_text, width=max_width_chars, replace_whitespace=False, drop_whitespace=True)
            line_height = int(font_size * 1.5)
            title_height = int((len(wrapped_lines) * line_height) + font_size * 0.7)
            new_height = original_height + title_height
            output_mode = 'RGB' if titled_png_path.lower().endswith(('.jpg', '.jpeg')) else 'RGBA'
            new_img = PIL.Image.new(output_mode, (original_width, new_height), (255, 255, 255, 255)) # White background
            new_img.paste(img, (0, title_height), mask=img if img.mode == 'RGBA' else None)
            draw = ImageDraw.Draw(new_img)
            bold_fonts = [
                'arialbd.ttf', 'Arial Bold.ttf', 'DejaVuSans-Bold.ttf',
                'Helvetica-Bold.ttf', 'Verdana-Bold.ttf', 'Georgia-Bold.ttf',
                'FreeSansBold.ttf', 'LiberationSans-Bold.ttf',
                'arial.ttf', 'DejaVuSans.ttf' # Fallbacks
            ]
            font = None
            for font_name in bold_fonts:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except IOError: continue
            if font is None:
                try:
                    font = ImageFont.load_default()
                    print(f"Warning: No suitable truetype font found for {titled_png_path}. Using default bitmap font. Title quality may be lower.")
                except Exception as font_err:
                    print(f"CRITICAL ERROR: Could not load any font (Truetype or default). Cannot add title. Error: {font_err}")
                    img.save(titled_png_path, format='PNG' if output_mode == 'RGBA' else 'JPEG', quality=95)
                    return False
            y_position = int(font_size * 0.35)
            for line in wrapped_lines:
                try:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = bbox[2] - bbox[0]
                except AttributeError:
                     try:
                         text_width, _ = draw.textsize(line, font=font)
                     except TypeError:
                         text_width = len(line) * font_size * 0.6
                         print(f"Warning: Could not get accurate text width for '{line}'. Centering may be approximate.")
                x_position = max(0, (original_width - text_width) // 2)
                text_color = (0, 0, 0, 255) if output_mode == 'RGBA' else (0, 0, 0)
                draw.text((x_position, y_position), line, fill=text_color, font=font)
                y_position += line_height
            save_format = 'PNG' if output_mode == 'RGBA' else 'JPEG'
            new_img.save(titled_png_path, format=save_format, quality=95)
            return True
    except FileNotFoundError:
        print(f"Error: Source PNG not found at {png_path}")
        return False
    except PIL.UnidentifiedImageError:
         print(f"Error: Cannot identify image file (may be corrupted or wrong format) at {png_path}")
         return False
    except Exception as e:
        print(f"Error adding title to PNG ({titled_png_path}): {e}")
        import traceback
        traceback.print_exc()
        if "cannot open resource" in str(e).lower() or "font not found" in str(e).lower():
             print("Hint: Make sure you have standard fonts like Arial or DejaVu Sans installed, or Pillow can find its default font.")
        return False

def write_metadata_files(metadata_rows, dirs, board_id, is_with_title=True):
    """Write metadata to CSV and JSON files"""
    if not metadata_rows:
        return
    
    csv_filename = f"{board_id}_metadata.csv"
    json_filename = f"{board_id}_metadata.json"
    
    if is_with_title:
        csv_filepath = os.path.join(dirs["withtitle_csv_dir"], csv_filename)
        json_filepath = os.path.join(dirs["withtitle_json_dir"], json_filename)
    else:
        csv_filepath = os.path.join(dirs["notitle_csv_dir"], csv_filename)
        json_filepath = os.path.join(dirs["notitle_json_dir"], json_filename)
    
    try:
        # Write CSV
        df = pd.DataFrame(metadata_rows)
        df['metadata'] = df['metadata'].apply(json.dumps)
        df.to_csv(csv_filepath, index=False)
        
        # Write JSON
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_rows, f, indent=4, ensure_ascii=False)
            
        return True
    except Exception as e:
        print(f"ERROR writing metadata files: {e}")
        return False