# generators/logos_generator.py
# -*- coding: utf-8 -*-
"""
Logos Dataset Generator - Generates "notitle" images with logo-related tasks
Supports both car logos and shoe logos generation.
"""
import os
import json
import time
import shutil
import logging
import mimetypes
from PIL import Image
from tqdm import tqdm
import argparse

# Try to import required dependencies
try:
    from google import genai
    from google.genai import types
    import PIL.Image
    from io import BytesIO
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Use common utilities
try:
    from utils import sanitize_filename, save_metadata_files
except ImportError:
    # Fallback sanitize function if utils not available
    import re
    def sanitize_filename(name_str):
        name_str = str(name_str).replace(' ', '_').lower()
        return re.sub(r'[^\w\-_\.]', '', name_str)
    
    def save_metadata_files(metadata_list, output_dir, filename_prefix):
        """Fallback metadata saving function"""
        json_path = os.path.join(output_dir, f"{filename_prefix}_metadata.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

# --- Configuration ---
RESOLUTIONS = [384, 768, 1152]  # Standard resolutions
BASE_NOTITLE_OUTPUT_DIR = "vlms-are-biased-notitle"
LOGO_RESOURCES_DIR = "logo_resources"

# API configuration - should be set externally or via environment
API_KEY = os.environ.get('GEMINI_API_KEY')  # Note: In production, use environment variables

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Car data for logo generation
CAR_DATA = {
    'Toyota': {
        "colors": ["White", "Gray", "Black"],
        "body_types": ["SUV", "Sedan", "Pickup Truck", "Hatchback", "Minivan"],
        "logo_elements": 3,  # Toyota has 3 ellipses
        "logo_element_name": "ellipses"
    },
    "Maserati": {
        "colors": ["White", "Gray", "Black"],
        "body_types": ["Sedan", "SUV", "Coupe", "Convertible", "Sports Car"],
        "logo_elements": 5,  # 5 prongs
        "logo_element_name": "prongs"
    },
    "Mercedes-Benz": {
        "colors": ["White", "Gray", "Black"],
        "body_types": ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"],
        "logo_elements": 5,  # 5 points on the star
        "logo_element_name": "points on the star"
    },
    "Audi": {
        "colors": ["White", "Gray", "Black"],  # Limited as in notebook
        "body_types": ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"],
        "logo_elements": 5,  # 5 overlapping circles
        "logo_element_name": "overlapping circles"
    }
}

# Shoe prompts for different sports
SHOE_PROMPTS = {
    'soccer': "generate a side-view image of an athlete wearing this pair of shoes. Keep all the fine-grained details of the shoes, particularly the {logo_detail} on both shoes. The person is playing soccer on grass field. They are wearing a soccer outfit. Zoom out to see their full body.",
    'running': "generate a side-view image of an athlete wearing this pair of shoes. Keep all the fine-grained details of the shoes, particularly the {logo_detail} on both shoes. The person is running in a realistic outdoor background. They are wearing a running outfit. Zoom out to see their full body.",
    'basketball': "generate a side-view image of an athlete wearing this pair of shoes. Keep all the fine-grained details of the shoes, particularly the {logo_detail} on both shoes. The person is playing basketball in an indoor court, dribbling the ball. They are wearing a basketball outfit. Zoom out to see their full body.",
    'tennis': "generate a side-view image of an athlete wearing this pair of shoes. Keep all the fine-grained details of the shoes, particularly the {logo_detail} on both shoes. The person is playing tennis in a tennis court. They are wearing a tennis outfit. Zoom out to see their full body."
}

def save_binary_file(file_name, data):
    """Save binary data to file"""
    with open(file_name, "wb") as f:
        f.write(data)

def create_directory_structure(logo_type):
    """Create the necessary directory structure for logos dataset"""
    # Main output directory structure
    logos_base_dir = os.path.join(BASE_NOTITLE_OUTPUT_DIR, f"logos_{logo_type}")
    logos_img_dir = os.path.join(logos_base_dir, "images")
    
    # Logo resources directory for base images
    resources_dir = os.path.join(LOGO_RESOURCES_DIR, logo_type)
    
    # Create all directories
    os.makedirs(logos_img_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)
    
    return {
        "base_dir": logos_base_dir,
        "img_dir": logos_img_dir,
        "resources_dir": resources_dir
    }

def generate_car_image(brand, color, body_type, trial=0, resources_dir='logo_resources'):
    """Generate base car image using Gemini API"""
    if not HAS_GENAI:
        logging.error("Google Generative AI library not available")
        return False
    
    client = genai.Client(api_key=API_KEY)
    model = "gemini-2.0-flash-exp-image-generation"
    
    prompt_text = f"A 1024x1024, photo-realistic front-view image of a {color.lower()} {brand} {body_type.lower()} parked on the road in the middle of the day. Zoomed out so that we can see the road."
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text)],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_modalities=["image", "text"],
        response_mime_type="text/plain",
    )
    
    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            if chunk.candidates[0].content.parts[0].inline_data:
                filename = f"{sanitize_filename(brand)}_{sanitize_filename(color)}_{sanitize_filename(body_type)}_{trial}"
                file_path = os.path.join(resources_dir, filename)
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                save_binary_file(f"{file_path}{file_extension}", inline_data.data)
                return f"{file_path}{file_extension}"
            else:
                logging.info(f"Text response: {chunk.text}")
        return False
    except Exception as e:
        logging.error(f"Error generating car image for {brand} {color} {body_type}: {e}")
        return False

def generate_shoe_image(prompt, input_path, output_path):
    """Generate shoe image with person wearing shoes"""
    if not HAS_GENAI:
        return False
    
    try:
        client = genai.Client(api_key=API_KEY)
        image = PIL.Image.open(input_path)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                logging.info(f"Text response: {part.text}")
            elif part.inline_data is not None:
                generated_image = Image.open(BytesIO(part.inline_data.data))
                generated_image.save(output_path)
                return True
        return False
    except Exception as e:
        logging.error(f"Error generating shoe image: {e}")
        return False

def resize_image(input_path, output_path, target_resolution):
    """Resize image with factor C/max(height, width) where C is target_resolution"""
    try:
        with Image.open(input_path) as img:
            width, height = img.size
            max_dimension = max(width, height)
            scale_factor = target_resolution / max_dimension
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(output_path)
            return True
    except Exception as e:
        logging.error(f"Error resizing image {input_path}: {e}")
        return False

def generate_car_logos_dataset(num_trials=1):
    """Generate car logos dataset - base images only, manual logo placement required"""
    if not HAS_GENAI:
        logging.error("Google Generative AI library is not installed. Cannot proceed.")
        return
    
    print("=== CAR LOGOS DATASET GENERATION ===")
    print("NOTE: This will generate BASE CAR IMAGES ONLY.")
    print("You will need to MANUALLY place modified logos on top of the cars after generation.")
    print("The base images will be saved to the logo_resources directory.")
    print()
    
    # Create directory structure
    output_dirs = create_directory_structure("cars")
    all_metadata = []
    
    # Calculate total images to generate
    total_combinations = sum(len(specs["colors"]) * len(specs["body_types"]) for specs in CAR_DATA.values())
    total_base_images = total_combinations * num_trials
    
    print(f"Will generate {total_base_images} base car images")
    
    # FOR LOOP 1: Generate ALL base car images
    print("\n=== FOR LOOP 1: Generating ALL base car images ===")
    base_image_paths = []
    
    with tqdm(total=total_base_images, desc="Generating base car images") as pbar:
        for brand, specs in CAR_DATA.items():
            for color in specs["colors"]:
                for body_type in specs["body_types"]:
                    for trial in range(num_trials):
                        image_path = generate_car_image(brand, color, body_type, trial, output_dirs["resources_dir"])
                        if image_path:
                            base_image_paths.append({
                                "path": image_path,
                                "brand": brand,
                                "color": color,
                                "body_type": body_type,
                                "trial": trial
                            })
                        pbar.update(1)
                        time.sleep(2)  # Rate limiting
    
    print(f"\n=== FOR LOOP 2: Processing base images and creating metadata ===")
    print("NOTE: Creating metadata structure for when you add logos manually.")
    print("After manual logo placement, the metadata will be ready for use.")
    
    # FOR LOOP 2: Process base images for final dataset and create metadata
    for img_data in tqdm(base_image_paths, desc="Processing base images"):
        for resolution in RESOLUTIONS:
            # Create final filename
            brand_clean = sanitize_filename(img_data['brand'])
            color_clean = sanitize_filename(img_data['color'])
            body_clean = sanitize_filename(img_data['body_type'])
            final_filename = f"{brand_clean}_{color_clean}_{body_clean}_{img_data['trial']}_px{resolution}.png"
            final_output_path = os.path.join(output_dirs["img_dir"], final_filename)
            
            # Resize base image to final dataset
            if resize_image(img_data["path"], final_output_path, resolution):
                # Create metadata entries (for when logos are added manually)
                create_car_metadata_entries(
                    all_metadata, img_data['brand'], img_data['color'], 
                    img_data['body_type'], img_data['trial'], resolution, final_filename
                )
    
    # Save metadata
    print("\n=== Saving metadata ===")
    save_metadata_files(all_metadata, output_dirs["base_dir"], "logos_cars_notitle")
    
    # Summary
    final_img_count = len([f for f in os.listdir(output_dirs["img_dir"]) if f.endswith('.png')])
    print(f"\n=== Car Logos Generation Summary ===")
    print(f"Base car images generated: {len(base_image_paths)}")
    print(f"Final dataset images: {final_img_count}")
    print(f"Total metadata entries: {len(all_metadata)}")
    print(f"Images saved to: {output_dirs['resources_dir']}")
    print(f"Dataset images saved to: {output_dirs['img_dir']}")
    print()
    print("NEXT STEPS:")
    print("1. Create modified logo versions:")
    print("   - Maserati: Modify to different number of prongs (currently expects 5)")
    print("   - Mercedes-Benz: Modify star points (currently expects 5)")  
    print("   - Audi: Modify overlapping circles (currently expects 5)")
    print("   - BMW: Modify quadrants (currently expects 4)")
    print("   - Toyota: Modify ellipses (currently expects 3)")
    print("2. Manually overlay these modified logos onto the generated car images")
    print("3. Replace the images in the dataset directory with logo-modified versions")
    print("4. Update metadata ground_truth values if using different logo element counts")

def generate_shoe_logos_dataset(base_shoes_dir="shoes_dir", num_trials=5):
    """Generate shoe logos dataset with people wearing shoes"""
    if not HAS_GENAI:
        logging.error("Google Generative AI library is not installed. Cannot proceed.")
        return
    
    print("=== SHOE LOGOS DATASET GENERATION ===")
    print(f"Looking for base shoe images in: {base_shoes_dir}")
    
    # Check if base shoes directory exists
    if not os.path.exists(base_shoes_dir):
        print(f"ERROR: Base shoes directory not found: {base_shoes_dir}")
        print("Please ensure you have base shoe images in the specified directory.")
        return
    
    # Create directory structure
    output_dirs = create_directory_structure("shoes")
    all_metadata = []
    
    sports = ['running', 'basketball', 'soccer', 'tennis']
    
    # FOR LOOP 1: Generate ALL shoe images with people
    print("\n=== FOR LOOP 1: Generating ALL shoe images with people ===")
    generated_image_paths = []
    
    for subdir in ['adidas', 'nike']:
        shoe_brand_path = os.path.join(base_shoes_dir, subdir)
        
        if not os.path.exists(shoe_brand_path):
            print(f"Warning: {shoe_brand_path} not found, skipping...")
            continue
            
        for sport in sports:
            # Set logo detail based on brand
            logo_detail = 'four stripes' if subdir == 'adidas' else 'two swooshes'
            color = 'black' if 'black' in shoe_brand_path else 'white'
            prompt = SHOE_PROMPTS[sport].replace('{logo_detail}', logo_detail)
            
            print(f"Processing {subdir} {sport} shoes...")
            
            for filename in tqdm(os.listdir(shoe_brand_path), desc=f"{subdir} {sport}"):
                if sport not in filename or not filename.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                input_path = os.path.join(shoe_brand_path, filename)
                
                # Generate multiple trials
                for trial in range(num_trials):
                    # Create output filename
                    base_name = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    output_filename = f"{subdir}_{base_name}_{trial}.png"
                    print(output_filename)
                    temp_output_path = os.path.join(output_dirs["resources_dir"], output_filename)
                    
                    if os.path.exists(temp_output_path):
                        print('path exists, skipping...')
                        continue  # Skip if already exists
                    
                    success = generate_shoe_image(prompt, input_path, temp_output_path)
                    if success:
                        generated_image_paths.append({
                            "path": temp_output_path,
                            "brand": subdir,
                            "sport": sport,
                            "trial": trial,
                            "base_filename": base_name,
                            "logo_detail": logo_detail
                        })
                    
                    time.sleep(1)  # Rate limiting
    
    # FOR LOOP 2: Resize images and create metadata
    print("\n=== FOR LOOP 2: Resizing images and creating metadata ===")
    
    for img_data in tqdm(generated_image_paths, desc="Processing final images"):
        for resolution in RESOLUTIONS:
            # Create final filename
            final_filename = f"{img_data['brand']}_{img_data['base_filename']}_{img_data['trial']}_px{resolution}.png"
            final_output_path = os.path.join(output_dirs["img_dir"], final_filename)
            
            if resize_image(img_data["path"], final_output_path, resolution):
                # Create metadata entries
                create_shoe_metadata_entries(
                    all_metadata, img_data, resolution, final_filename
                )
    
    # Save metadata
    print("\n=== Saving metadata ===")
    save_metadata_files(all_metadata, output_dirs["base_dir"], "logos_shoes_notitle")
    
    # Summary
    final_img_count = len([f for f in os.listdir(output_dirs["img_dir"]) if f.endswith('.png')])
    print(f"\n=== Shoe Logos Generation Summary ===")
    print(f"Generated images with people: {len(generated_image_paths)}")
    print(f"Final dataset images: {final_img_count}")
    print(f"Total metadata entries: {len(all_metadata)}")

def create_car_metadata_entries(metadata_list, brand, color, body_type, trial, resolution, filename):
    """Create metadata entries for a car image"""
    
    car_info = CAR_DATA[brand]
    expected_elements = car_info["logo_elements"]
    
    # Define brand-specific prompts
    if brand == "Audi":
        q1_prompt = f"How many overlapping circles are there in the logo of this car? Answer with a number in curly brackets, e.g., {{9}}."
        q2_prompt = f"Count the overlapping circles in the logo of this car. Answer with a number in curly brackets, e.g., {{9}}."
    elif brand == "Mercedes-Benz":
        q1_prompt = f"How many points are there on the star in the logo of this car? Answer with a number in curly brackets, e.g., {{9}}."
        q2_prompt = f"Count the points on the star in the logo of this car. Answer with a number in curly brackets, e.g., {{9}}."
    elif brand == "Maserati":
        q1_prompt = f"How many prongs are there in the logo of this car? Answer with a number in curly brackets, e.g., {{9}}"
        q2_prompt = f"Count the prongs in the logo of this car. Answer with a number in curly brackets, e.g., {{9}}"
    elif brand == "BMW":
        q1_prompt = f"How many quadrants are there in the logo of this car? Answer with a number in curly brackets, e.g., {{9}}."
        q2_prompt = f"Count the quadrants in the logo of this car. Answer with a number in curly brackets, e.g., {{9}}."
    elif brand == "Toyota":
        q1_prompt = f"How many ellipses are there in the logo of this car? Answer with a number in curly brackets, e.g., {{9}}."
        q2_prompt = f"Count the ellipses in the logo of this car. Answer with a number in curly brackets, e.g., {{9}}."
    
    q3_prompt = f"Is the logo on this car {brand} logo? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    
    # Ground truth values
    gt_elements = str(expected_elements)
    bias_elements = str(expected_elements)  # Could be modified if using altered logos
    
    gt_brand = "Yes"
    bias_brand = "Yes"
    
    image_path_relative = os.path.join("images", filename).replace("\\", "/")
    
    base_id = f"{sanitize_filename(brand)}_{sanitize_filename(color)}_{sanitize_filename(body_type)}_{trial}_px{resolution}"
    
    # Create entries for Q1 and Q2 (logo element counting)
    for q_key, prompt in [("Q1", q1_prompt), ("Q2", q2_prompt)]:
        metadata_list.append({
            "image": filename,
            "image_path": image_path_relative,
            "topic": "Logos",
            "prompt": prompt,
            "ground_truth": gt_elements,
            "expected_bias": bias_elements,
            "with_title": False,
            "type_of_question": q_key,
            "pixel": resolution
        })
    
    # Create entry for Q3 (brand identification)
    metadata_list.append({
        "image": filename,
        "image_path": image_path_relative,
        "topic": "Logos",
        "prompt": q3_prompt,
        "ground_truth": gt_brand,
        "expected_bias": bias_brand,
        "with_title": False,
        "type_of_question": "Q3",
        "pixel": resolution
    })

def create_shoe_metadata_entries(metadata_list, img_data, resolution, filename):
    """Create metadata entries for a shoe image"""
    
    brand = img_data['brand']
    
    # Define brand-specific prompts
    if brand == 'adidas':
        expected_elements = 4  # 4 stripes
        q1_prompt = "How many visible stripes are there in the logo of the left shoe? Answer with a number in curly brackets, e.g., {9}."
        q2_prompt = "Count the visible stripes in the logo of the left shoe. Answer with a number in curly brackets, e.g., {9}."
        q3_prompt = "Are the logos on these shoes Adidas logos? Answer in curly brackets, e.g., {Yes} or {No}."
    elif brand == 'nike':
        expected_elements = 2  # 2 swooshes (assuming both shoes visible)
        q1_prompt = "How many visible colored stylized curves are there in the logo of the left shoe? Answer with a number in curly brackets, e.g., {9}"
        q2_prompt = "Count the visible colored stylized curves in the logo of the left shoe. Answer with a number in curly brackets, e.g., {9}"
        q3_prompt = "Are the logos on these shoes Nike logos? Answer in curly brackets, e.g., {Yes} or {No}."
    
    # Ground truth values
    gt_elements = str(expected_elements)
    bias_elements = str(expected_elements)
    
    gt_brand = "Yes"
    bias_brand = "Yes"
    
    image_path_relative = os.path.join("images", filename).replace("\\", "/")
    
    # Create entries for Q1 and Q2 (logo element counting)
    for q_key, prompt in [("Q1", q1_prompt), ("Q2", q2_prompt)]:
        metadata_list.append({
            "image": filename,
            "image_path": image_path_relative,
            "topic": "Logos",
            "prompt": prompt.replace('colored', 'black' if 'black' in filename else 'white'),
            "ground_truth": gt_elements,
            "expected_bias": bias_elements,
            "with_title": False,
            "type_of_question": q_key,
            "pixel": resolution
        })
    
    # Create entry for Q3 (brand identification)
    metadata_list.append({
        "image": filename,
        "image_path": image_path_relative,
        "topic": "Logos",
        "prompt": q3_prompt,
        "ground_truth": gt_brand,
        "expected_bias": bias_brand,
        "with_title": False,
        "type_of_question": "Q3",
        "pixel": resolution
    })

def main():
    """Main function to generate logos dataset"""
    parser = argparse.ArgumentParser(description='Generate Logos "notitle" dataset.')
    parser.add_argument('--type', type=str, choices=['cars', 'shoes'], required=True,
                        help='Type of logo dataset to generate: cars or shoes')
    parser.add_argument('--trials', type=int, default=1, 
                        help='Number of trials per item (default: 1 for cars, 6 for shoes)')
    parser.add_argument('--base-shoes-dir', type=str, default="shoes_dir/",
                        help='Directory containing base shoe images (for shoes type only)')
    
    args = parser.parse_args()
    
    if args.type == 'cars':
        trials = args.trials if args.trials != 1 else 1  # Default 1 for cars
        generate_car_logos_dataset(num_trials=trials)
    elif args.type == 'shoes':
        trials = args.trials if args.trials != 1 else 5  # Default 5 for shoes
        generate_shoe_logos_dataset(base_shoes_dir=args.base_shoes_dir, num_trials=trials)

if __name__ == '__main__':
    main()