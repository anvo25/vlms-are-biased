# generators/animals_generator.py
# -*- coding: utf-8 -*-
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
    # import google.generativeai as genai
    from google import genai
    from google.genai import types
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
ANIMAL_RESOURCES_DIR = "animal_resources"
TOPIC_NAME = "Animals"
BOARD_ID = "animals"

# API configuration - should be set externally or via environment
API_KEY = os.environ.get('GEMINI_API_KEY')  # Note: In production, use environment variables

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Animal definitions from the notebook
ANIMALS = [
    # {"name": "Horse", "original_legs": 4},
    {"name": "Zebra", "original_legs": 4},
    # {"name": "Donkey", "original_legs": 4},
    # {"name": "Mule", "original_legs": 4},
    # {"name": "Cow", "original_legs": 4},
    # {"name": "Bull", "original_legs": 4},
    # {"name": "Bison", "original_legs": 4},
    # {"name": "Buffalo", "original_legs": 4},
    # {"name": "Yak", "original_legs": 4},
    # {"name": "Water Buffalo", "original_legs": 4},
    # {"name": "Deer", "original_legs": 4},
    # {"name": "Elk", "original_legs": 4},
    # {"name": "Moose", "original_legs": 4},
    # {"name": "Reindeer", "original_legs": 4},
    # {"name": "Caribou", "original_legs": 4},
    # {"name": "Antelope", "original_legs": 4},
    # {"name": "Gazelle", "original_legs": 4},
    # {"name": "Giraffe", "original_legs": 4},
    # {"name": "Camel", "original_legs": 4},
    # {"name": "Dromedary Camel", "original_legs": 4},
    # {"name": "Bactrian Camel", "original_legs": 4},
    # {"name": "Llama", "original_legs": 4},
    # {"name": "Alpaca", "original_legs": 4},
    # {"name": "Goat", "original_legs": 4},
    # {"name": "Sheep", "original_legs": 4},
    # {"name": "Ibex", "original_legs": 4},
    # {"name": "Mountain Goat", "original_legs": 4},
    # {"name": "Pronghorn", "original_legs": 4},
    # {"name": "Bighorn Sheep", "original_legs": 4},
    # {"name": "Wild Boar", "original_legs": 4},
    # {"name": "Pig", "original_legs": 4},
    # {"name": "Warthog", "original_legs": 4},
    # {"name": "Coyote", "original_legs": 4},
    # {"name": "Lynx", "original_legs": 4},
    # {"name": "Bobcat", "original_legs": 4},
    # {"name": "Leopard", "original_legs": 4},
    # {"name": "Cheetah", "original_legs": 4},
    # {"name": "Tiger", "original_legs": 4},
    # {"name": "Lion", "original_legs": 4},
    # {"name": "Jaguar", "original_legs": 4},
    # {"name": "Puma", "original_legs": 4},
    # {"name": "Ocelot", "original_legs": 4},
    # {"name": "Serval", "original_legs": 4},
    # {"name": "Caracal", "original_legs": 4},
    # {"name": "Hyena", "original_legs": 4},
    # {"name": "Rabbit", "original_legs": 4},
    # {"name": "Hare", "original_legs": 4},
    # {"name": "Impala", "original_legs": 4},
    # {"name": "Springbok", "original_legs": 4},
    # {"name": "Kudu", "original_legs": 4},
    # {"name": "Eland", "original_legs": 4},
    # {"name": "Waterbuck", "original_legs": 4},
    # {"name": "Wildebeest", "original_legs": 4},
    # {"name": "Okapi", "original_legs": 4},
    # {"name": "Rhinoceros", "original_legs": 4},
    # {"name": "Hippopotamus", "original_legs": 4},
    # {"name": "African Elephant", "original_legs": 4},
    # {"name": "Asian Elephant", "original_legs": 4},
    # {"name": "Indian Rhinoceros", "original_legs": 4},
    # {"name": "Kangaroo", "original_legs": 2},
    # {"name": "Wallaby", "original_legs": 2},
    # {"name": "Gnu", "original_legs": 4},
    # {"name": "Maned Wolf", "original_legs": 4},
    # {"name": "Arctic Fox", "original_legs": 4},
    # {"name": "Red Fox", "original_legs": 4},
    # {"name": "Fennec Fox", "original_legs": 4},
    # {"name": "Gray Wolf", "original_legs": 4},
    # {"name": "Red Wolf", "original_legs": 4},
    # {"name": "Domestic Dog", "original_legs": 4},
    # {"name": "Domestic Cat", "original_legs": 4},
    # {"name": "Snow Leopard", "original_legs": 4},
    # {"name": "African Wild Dog", "original_legs": 4},
    # {"name": "Dingo", "original_legs": 4},
    # {"name": "Jackal", "original_legs": 4},
    # {"name": "Ostrich", "original_legs": 2},
    # {"name": "Emu", "original_legs": 2},
    # {"name": "Rhea", "original_legs": 2},
    # {"name": "Cassowary", "original_legs": 2},
    # {"name": "Flamingo", "original_legs": 2},
    # {"name": "Heron", "original_legs": 2},
    # {"name": "Stork", "original_legs": 2},
    # {"name": "Crane", "original_legs": 2},
    # {"name": "Egret", "original_legs": 2},
    # {"name": "Ibis", "original_legs": 2},
    # {"name": "Spoonbill", "original_legs": 2},
    # {"name": "Secretary Bird", "original_legs": 2},
    # {"name": "Pelican", "original_legs": 2},
    # {"name": "Turkey", "original_legs": 2},
    # {"name": "Chicken", "original_legs": 2},
    # {"name": "Rooster", "original_legs": 2},
    # {"name": "Duck", "original_legs": 2},
    # {"name": "Goose", "original_legs": 2},
    # {"name": "Swan", "original_legs": 2},
    # {"name": "Peacock", "original_legs": 2},
    # {"name": "Sandpiper", "original_legs": 2},
    # {"name": "Avocet", "original_legs": 2},
    # {"name": "Stilt", "original_legs": 2},
    # {"name": "Plover", "original_legs": 2},
    # {"name": "Lapwing", "original_legs": 2},
    # {"name": "Oystercatcher", "original_legs": 2}
]

def save_binary_file(file_name, data):
    """Save binary data to file"""
    with open(file_name, "wb") as f:
        f.write(data)

def create_directory_structure():
    """Create the necessary directory structure for animals dataset"""
    # Main output directory structure
    animals_base_dir = os.path.join(BASE_NOTITLE_OUTPUT_DIR, "animals")
    animals_img_dir = os.path.join(animals_base_dir, "images")
    
    # Animal resources directory for base images
    resources_dir = ANIMAL_RESOURCES_DIR
    
    # Create all directories
    os.makedirs(animals_img_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)
    
    return {
        "base_dir": animals_base_dir,
        "img_dir": animals_img_dir,
        "resources_dir": resources_dir
    }

def generate_base_animal_image(animal, trial=0, resources_dir='animal_resources'):
    """Generate base animal image using Gemini API"""
    if not HAS_GENAI:
        logging.error("Google Generative AI library not available")
        return False
    
    client = genai.Client(api_key=API_KEY)
    model = "gemini-2.0-flash-exp-image-generation"
    
    prompt_text = f"""Generate a clear, full-body, side-view image of a(n) {animal['name'].lower()} with {animal['original_legs']} legs that is walking in a real-world natural background. The {animal['original_legs']}-legged animal must look photo-realistic in nature. All {animal['original_legs']} legs must be clearly visible."""
    
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
                file_name = os.path.join(resources_dir, f"{sanitize_filename(animal['name'])}_{trial}")
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                save_binary_file(f"{file_name}{file_extension}", inline_data.data)
                return f"{file_name}{file_extension}"
            else:
                logging.info(f"Text response: {chunk.text}")
        return False
    except Exception as e:
        logging.error(f"Error generating base image for {animal['name']}: {e}")
        return False

def edit_animal_image(animal, base_image_path, edit_type, trial=0, resources_dir='animal_resources'):
    """Edit animal image to add or remove legs"""
    if not HAS_GENAI or not os.path.exists(base_image_path):
        return False
    
    client = genai.Client(api_key=API_KEY)
    model = "gemini-2.0-flash-exp-image-generation"
    
    if edit_type == "add_leg":
        target_legs = animal['original_legs'] + 1
        prompt_text = f"Edit this image: Add 1 more leg to the {animal['name'].lower()} so that it has {target_legs} legs in total. The 1 extra leg is in the middle of the body. The {target_legs}-legged {animal['name'].lower()} must be photo-realistic. All {target_legs} legs must be clearly visible."
        suffix = "5_legs"
    elif edit_type == "remove_leg":
        target_legs = animal['original_legs'] - 1
        prompt_text = f"Edit this image: Remove 1 leg from the {animal['name'].lower()} so that it has {target_legs} legs in total. The {target_legs}-legged {animal['name'].lower()} must be photo-realistic. All {target_legs} legs must be clearly visible."
        suffix = "3_legs"
    else:
        return False
    
    try:
        files = client.files.upload(file=base_image_path)
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=files.uri,
                        mime_type=files.mime_type,
                    ),
                    types.Part.from_text(text=prompt_text),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            response_modalities=["image", "text"],
            response_mime_type="text/plain",
        )
        
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            if chunk.candidates[0].content.parts[0].inline_data:
                # Create subdirectory for edited images
                edit_dir = os.path.join(resources_dir, f"_{suffix}")
                os.makedirs(edit_dir, exist_ok=True)
                
                file_name = os.path.join(edit_dir, f"{sanitize_filename(animal['name'])}_{trial}")
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                save_binary_file(f"{file_name}{file_extension}", inline_data.data)
                return f"{file_name}{file_extension}"
            else:
                logging.info(f"Text response: {chunk.text}")
        return False
    except Exception as e:
        logging.error(f"Error editing image for {animal['name']} ({edit_type}): {e}")
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

def generate_dataset(num_trials=5):
    """Main function to generate the animals dataset"""
    if not HAS_GENAI:
        logging.error("Google Generative AI library is not installed. Cannot proceed.")
        print("Please install Google Generative AI: pip install google-generativeai")
        return
    
    print(f"Starting {TOPIC_NAME} 'notitle' dataset generation...")
    
    # Create directory structure
    output_dirs = create_directory_structure()
    
    all_metadata = []
    
    # Calculate total images to generate
    total_base_images = len(ANIMALS) * num_trials
    total_edit_images = len(ANIMALS) * num_trials * 2  # add_leg and remove_leg
    total_final_images = total_edit_images * len(RESOLUTIONS)  # Only edited images in final dataset
    
    print(f"Will generate {total_base_images} base images, {total_edit_images} edited images")
    print(f"Final dataset will have {total_final_images} images across {len(RESOLUTIONS)} resolutions")
    
    # FOR LOOP 1: Generate ALL base animal images first
    print("\n=== FOR LOOP 1: Generating ALL base animal images ===")
    base_image_paths = {}
    
    total_base_operations = len(ANIMALS) * num_trials
    with tqdm(total=total_base_operations, desc="Generating base images") as pbar:
        for animal in ANIMALS:
            base_image_paths[animal['name']] = []
            for trial in range(num_trials):
                image_path = generate_base_animal_image(animal, trial, output_dirs["resources_dir"])
                if image_path:
                    base_image_paths[animal['name']].append(image_path)
                pbar.update(1)
                time.sleep(2)  # Rate limiting
    
    # FOR LOOP 2: Generate ALL edited images based on base images
    print("\n=== FOR LOOP 2: Generating ALL edited images based on base images ===")
    edited_image_paths = {"add_leg": {}, "remove_leg": {}}
    
    total_edit_operations = 0
    # Count available base images for progress bar
    for animal in ANIMALS:
        if animal['name'] in base_image_paths:
            total_edit_operations += len(base_image_paths[animal['name']]) * 2  # add_leg and remove_leg
    
    with tqdm(total=total_edit_operations, desc="Generating edited images") as pbar:
        for animal in ANIMALS:
            if animal['name'] not in base_image_paths or not base_image_paths[animal['name']]:
                continue
                
            edited_image_paths["add_leg"][animal['name']] = []
            edited_image_paths["remove_leg"][animal['name']] = []
            
            for trial, base_path in enumerate(base_image_paths[animal['name']]):
                # Add leg
                add_leg_path = edit_animal_image(animal, base_path, "add_leg", trial, output_dirs["resources_dir"])
                if add_leg_path:
                    edited_image_paths["add_leg"][animal['name']].append(add_leg_path)
                pbar.update(1)
                time.sleep(2)  # Rate limiting
                
                # Remove leg
                remove_leg_path = edit_animal_image(animal, base_path, "remove_leg", trial, output_dirs["resources_dir"])
                if remove_leg_path:
                    edited_image_paths["remove_leg"][animal['name']].append(remove_leg_path)
                pbar.update(1)
                time.sleep(2)  # Rate limiting
    
    # Step 3: Resize edited images and create metadata (ONLY for edited images)
    print("\n=== Step 3: Processing edited images and creating metadata ===")
    
    # Process ONLY edited images (not base images)
    for edit_type in ["add_leg", "remove_leg"]:
        for animal in tqdm(ANIMALS, desc=f"Processing {edit_type} images"):
            if (animal['name'] not in edited_image_paths[edit_type] or 
                not edited_image_paths[edit_type][animal['name']]):
                continue
                
            target_legs = animal['original_legs'] + (1 if edit_type == "add_leg" else -1)
            
            for trial, edited_path in enumerate(edited_image_paths[edit_type][animal['name']]):
                for resolution in RESOLUTIONS:
                    # Create filename
                    animal_name_clean = sanitize_filename(animal['name'])
                    filename = f"{animal_name_clean}_{edit_type}_{trial}_px{resolution}.png"
                    output_path = os.path.join(output_dirs["img_dir"], filename)
                    
                    if resize_image(edited_path, output_path, resolution):
                        # Create metadata entries ONLY for edited images
                        create_metadata_entries(
                            all_metadata, animal, edit_type, trial, resolution, filename,
                            target_legs, animal['original_legs']
                        )
    
    # Step 4: Save metadata
    print("\nStep 4: Saving metadata...")
    save_metadata_files(all_metadata, output_dirs["base_dir"], f"{BOARD_ID}_notitle")
    
    # Summary
    final_img_count = len([f for f in os.listdir(output_dirs["img_dir"]) if f.endswith('.png')])
    print(f"\n--- {TOPIC_NAME} 'notitle' Generation Summary ---")
    print(f"Base images generated in resources: {sum(len(paths) for paths in base_image_paths.values())}")
    print(f"Final edited images in dataset: {final_img_count}")
    print(f"Total metadata entries (edited images only): {len(all_metadata)}")
    print(f"{TOPIC_NAME} dataset generation complete!")

def create_metadata_entries(metadata_list, animal, modification_type, trial, resolution, filename, actual_legs, original_legs):
    """Create metadata entries for an animal image"""
    
    # Define questions
    questions = {
        "Q1": f"How many legs does this {animal['name'].lower()} have? Answer with a number in curly brackets, e.g., {{4}}.",
        "Q2": f"Count the legs on this {animal['name'].lower()}. Answer with a number in curly brackets, e.g., {{4}}.",
        "Q3": f"Is this an animal with {original_legs} legs? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    }
    
    # Ground truth and expected bias
    gt_legs = str(actual_legs)
    bias_legs = str(original_legs)  # Bias is to see the typical number
    
    gt_typical = "Yes" if actual_legs == original_legs else "No"
    bias_typical = "Yes"  # Bias is to assume typical
    
    image_path_relative = os.path.join("images", filename).replace("\\", "/")
    
    base_id = f"{sanitize_filename(animal['name'])}_{modification_type}_{trial}_px{resolution}"
    
    # Create entries for Q1 and Q2 (leg counting)
    for q_key in ["Q1", "Q2"]:
        metadata_list.append({
            "ID": f"{base_id}_{q_key}",
            "image_path": image_path_relative,
            "topic": TOPIC_NAME,
            "prompt": questions[q_key],
            "ground_truth": gt_legs,
            "expected_bias": bias_legs,
            "with_title": False,
            "type_of_question": q_key,
            "pixel": resolution,
            "metadata": {
                "animal_name": animal['name'],
                "original_legs": original_legs,
                "actual_legs": actual_legs,
                "modification_type": modification_type,
                "trial": trial,
                "resolution_px": resolution
            }
        })
    
    # Create entry for Q3 (typical number check)
    metadata_list.append({
        "ID": f"{base_id}_Q3",
        "image_path": image_path_relative,
        "topic": TOPIC_NAME,
        "prompt": questions["Q3"],
        "ground_truth": gt_typical,
        "expected_bias": bias_typical,
        "with_title": False,
        "type_of_question": "Q3",
        "pixel": resolution,
        "metadata": {
            "animal_name": animal['name'],
            "original_legs": original_legs,
            "actual_legs": actual_legs,
            "modification_type": modification_type,
            "trial": trial,
            "resolution_px": resolution
        }
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Animals "notitle" dataset.')
    parser.add_argument('--trials', type=int, default=1, 
                        help='Number of trials per animal (default: 5)')
    args = parser.parse_args()
    
    generate_dataset(num_trials=args.trials)