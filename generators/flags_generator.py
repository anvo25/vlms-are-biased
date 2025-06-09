
"""
Flag Dataset Generator - Generates "notitle" images with flag stripe counting tasks
Uses WikimediaSVGRetriever to download flag SVGs and LLM to modify stripe counts.
"""
import os
import json
import time
import logging
import argparse
import requests
from lxml import etree
from PIL import Image
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from cairosvg import svg2png
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False

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

# API configuration from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Flag data with original stripe counts
FLAG_DATA = [
    {"name": "Flag of the United States", "original_stripes": 13},
    # {"name": "Flag of Malaysia", "original_stripes": 14},
    # {"name": "Flag of Greece", "original_stripes": 9},
    # {"name": "Flag of Thailand", "original_stripes": 5},
    # {"name": "Flag of Liberia", "original_stripes": 11},
    # {"name": "Flag of Uruguay", "original_stripes": 9},
    # {"name": "Flag of Costa Rica", "original_stripes": 5},
    # {"name": "Flag of Togo", "original_stripes": 5},
    # {"name": "Flag of Suriname", "original_stripes": 5},
    # {"name": "Flag of Kiribati", "original_stripes": 6},
    # {"name": "Flag of the Gambia", "original_stripes": 5},
    # {"name": "Flag of Cuba", "original_stripes": 5},
    # {"name": "Flag of Puerto Rico", "original_stripes": 5},
    # {"name": "Flag of Ohio", "original_stripes": 5},
    # {"name": "Flag of Kenya", "original_stripes": 5},
    # {"name": "Flag of Mauritius", "original_stripes": 4},
    # {"name": "Flag of Uganda", "original_stripes": 6},
    # {"name": "Flag of Zimbabwe", "original_stripes": 7},
    # {"name": "Flag of the United Arab Emirates", "original_stripes": 4},
    # {"name": "Flag of Comoros", "original_stripes": 4},
    # {"name": "Flag of the Central African Republic", "original_stripes": 5},
]

class WikimediaSVGRetriever:
    """
    A robust SVG downloader from Wikimedia Commons using:
    - Semantic reranking via SentenceTransformer or OpenAI LLM
    - OpenSearch query suggestions
    - LLM-based query reformulation
    - SVG validation via lxml parser
    """

    def __init__(self, openai_api_key: str = None, reranker: str = "embedding",
                 model_name="msmarco-distilbert-base-v4", llm_model="gpt-4.1-mini"):
        """
        Args:
            openai_api_key (str): OpenAI API key for LLM-based methods.
            reranker (str): 'embedding' or 'llm' to choose reranking method.
            model_name (str): SentenceTransformer model.
            llm_model (str): OpenAI model used for LLM reformulation/ranking.
        """
        if openai_api_key and HAS_OPENAI:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = None
        self.llm_model = llm_model
        self.reranker = reranker
        if reranker == "embedding" and HAS_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
        os.makedirs("assets", exist_ok=True)

    def is_valid_svg(self, content: bytes) -> bool:
        """Check if content is a valid SVG."""
        try:
            parser = etree.XMLParser(resolve_entities=False, dtd_validation=False)
            root = etree.fromstring(content, parser=parser)
            return "svg" in root.tag.lower()
        except Exception:
            snippet = content[:3000].decode(errors="ignore").lower()
            return "<svg" in snippet and "<html" not in snippet

    def sanitize_filename(self, name: str) -> str:
        """Sanitize filename for safe saving."""
        return "".join(c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in name)

    def rerank_with_llm(self, query: str, titles: list) -> str:
        """Ask LLM to pick the best matching file title."""
        if not self.client:
            return "[0]"  # Fallback
            
        prompt = f"""You are a helpful assistant tasked with selecting the best SVG file name based on a user's search query.
Given the query: "A standard {query}"

Here are some candidate file titles:
{titles}

Reorder the list in terms of the best matches the user's intent.
You have to follow python index convention (e.g. 0, 1, 2, etc.)
Return the list of index ordered based on the relevance of each term by descending order (highest relevance first).
Do not return anything else.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": "You are a ranking assistant."},
                          {"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error in LLM reranking: {e}")
            return "[0]"

    def filter_with_llm(self, query: str, titles: list) -> str:
        """Ask LLM to filter out irrelevant titles."""
        if not self.client:
            return str(list(range(len(titles))))  # Return all indices
            
        prompt = f"""You are a helpful assistant tasked with retaining only the relevant SVG file names based on a user's search query.
Given the query: "A standard {query}"

Here are some candidate file titles:
{titles}

Filter out all titles that are irrelevant or have low relevance to the search query. Retain only the relevant titles.
You have to follow python index convention (e.g. 0, 1, 2, etc.)
Return the list of indexes of the titles that have high relevance to the search query.
Note: Prioritize precision. If none of the titles are relevant, return an empty list.
Do not return anything else.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": "You are a ranking assistant."},
                          {"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error in LLM filtering: {e}")
            return str(list(range(len(titles))))

    def retrieve_svg_with_ranking(self, search_term: str, save_path: str = None, max_candidates: int = 20):
        """
        Use the chosen reranker (LLM or embedding) to select and download the best SVG match.

        Args:
            search_term (str): Reformulated search query.
            save_path (str): Optional output filename.
            max_candidates (int): How many results to consider.
        """
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": f"{search_term} filetype:svg",
            "gsrlimit": max_candidates,
            "prop": "imageinfo",
            "iiprop": "url"
        }

        print(f"🔍 Searching Wikimedia for: '{search_term}'")
        try:
            response = requests.get(url, params=params, timeout=30)
            pages = response.json().get("query", {}).get("pages", {})
        except Exception as e:
            logging.error(f"Error searching Wikimedia: {e}")
            return None

        svg_candidates = [
            (page["title"], page["imageinfo"][0]["url"])
            for page in pages.values()
            if page["title"].lower().endswith(".svg")
        ]
        if not svg_candidates:
            print("❌ No SVG candidates found.")
            return None

        print('Filtering out irrelevant titles...')
        print(f'Old candidate list have {len(svg_candidates)} items')
        titles = [title for title, _ in svg_candidates]
        try:
            irrelevant_idx = self.filter_with_llm(search_term, titles)
            filtered_indices = eval(irrelevant_idx)
            titles = [titles[i] for i in filtered_indices]
            svg_candidates = [svg_candidates[i] for i in filtered_indices]
        except:
            logging.warning("Failed to filter, using all candidates")
            
        print(f'New candidate list have {len(svg_candidates)} items')
        if len(svg_candidates) == 0:
            print("❌ No relevant SVG candidates found after filtering.")
            return None

        print('Reranking the titles based on relevance...')
        if self.reranker == "llm" and self.client:
            candidates = [t.replace('File:', '').replace('.svg', '').strip().lower() for t in titles]
            try:
                top_title_idx = self.rerank_with_llm(search_term, candidates)
                sorted_candidates = [svg_candidates[i] for i in eval(top_title_idx)]
            except:
                sorted_candidates = svg_candidates
        elif self.model:
            query_embed = self.model.encode(f'standard {search_term.lower()}', convert_to_tensor=True)
            title_embeds = self.model.encode([' '.join(t.replace('File:', '').replace('.svg', '').strip().lower().split('_')) for t in titles], convert_to_tensor=True)
            scores = util.cos_sim(query_embed, title_embeds)[0]
            sorted_candidates = sorted(zip(scores, svg_candidates), key=lambda x: x[0], reverse=True)
            sorted_candidates = [ele[1] for ele in sorted_candidates]
        else:
            sorted_candidates = svg_candidates

        for (title, url) in sorted_candidates:
            print(f"🔎 Trying: {title}")
            try:
                headers = {
                    'User-Agent': 'VLMBiasDataset/1.0 (research@example.org) Python/3.x requests/2.x'
                }
                svg_response = requests.get(url, headers=headers, timeout=10)

                if svg_response.status_code == 200 and self.is_valid_svg(svg_response.content):
                    final_name = save_path or self.sanitize_filename(url.split("/")[-1])
                    full_path = os.path.join("assets", final_name)
                    with open(full_path, "wb") as f:
                        f.write(svg_response.content)
                    print(f"✅ Downloaded: {title} → {full_path}")
                    return svg_response.content
                else:
                    print(f"⚠️ Invalid SVG (probably error page): {url}")
            except Exception as e:
                print(f"⚠️ Error fetching {title}: {e}")

        print("❌ All candidate SVGs failed validation")
        return None

def modify_svg_feature(downloader, name_identifier: str, feature_instruction: str, model_type='openai'):
    """
    Retrieve base SVG, then ask the LLM to modify it according to a feature instruction.

    Args:
        downloader: WikimediaSVGRetriever instance
        name_identifier (str): The function name or label for the object (e.g., "Flag of United States")
        feature_instruction (str): Instruction to modify the SVG (e.g., "Change to have 12 stripes")
        model_type (str): 'openai' or 'anthropic'

    Returns:
        str: Modified SVG code as returned by the LLM.
    """
    name_readable = name_identifier
    svg_content = downloader.retrieve_svg_with_ranking(name_readable)
    
    if svg_content:
        print(f"✅ Found base SVG for modification: {name_readable}")
        svg_code = svg_content.decode(errors='ignore')
        if len(svg_code.split()) >= 1500:
            print("⚠️ SVG code length exceeds limit:", len(svg_code.split()))
            return None
    else:
        print(f"⚠️ No SVG found for modification: {name_readable} — cannot proceed.")
        return None

    # Build prompt
    prompt = f"""You are an expert in editing SVG image code.

Task: Modify the base SVG for a {name_readable} according to the following instruction:

Instruction: "{feature_instruction}"

Base SVG code:
{svg_code}

Instructions:
1. Modify the base SVG by adding or removing the mentioned feature (stripes, stars, etc) according to the instruction above.
2. Wrap the entire SVG in <code> </code>. Do not explain anything.
"""

    if model_type == 'anthropic' and HAS_ANTHROPIC and ANTHROPIC_API_KEY:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                system="You are an expert SVG image editor. You return modified code only.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            logging.error(f"Error with Anthropic API: {e}")
            return None
    elif model_type == 'openai' and HAS_OPENAI and OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SVG image editor. You return modified code only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error with OpenAI API: {e}")
            return None
    else:
        logging.error(f"Model type {model_type} not available or API key missing")
        return None

def convert_to_instruction(flag_name, original_stripes, stripe_change):
    """Convert flag modification to instruction text"""
    new_stripes = original_stripes + stripe_change
    new_stripes = max(0, new_stripes)  # Ensure non-negative
    
    country_name = flag_name.replace("Flag of the ", "").replace("Flag of ", "")
    
    instruction = f"The {flag_name} has {original_stripes} stripes. Update it to have {new_stripes} stripes"
    
    if stripe_change > 0:
        instruction += f" by adding {stripe_change} stripes to it."
    else:
        instruction += f" by removing {abs(stripe_change)} stripes from it."
    
    return instruction

def convert_svgs_to_pngs(input_dir: str, output_dir: str) -> None:
    """
    Convert all .svg files in input_dir to .png in output_dir.
    Skips any SVGs that raise errors during conversion.
    """
    if not HAS_CAIROSVG:
        logging.error("cairosvg not available, cannot convert SVGs to PNGs")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.svg'):
            continue

        in_path = os.path.join(input_dir, fname)
        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(output_dir, out_name)

        try:
            print(f'Converting {fname} → {out_name}')
            svg2png(url=in_path, write_to=out_path)
        except Exception as e:
            print(f'❌ Failed to convert {fname}: {e}')

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

def create_directory_structure():
    """Create the necessary directory structure for flags dataset"""
    flags_base_dir = os.path.join(BASE_NOTITLE_OUTPUT_DIR, "flags")
    flags_img_dir = os.path.join(flags_base_dir, "images")
    svg_resources_dir = "flag_resources"
    png_temp_dir = "flag_pngs_temp"
    
    # Create all directories
    os.makedirs(flags_img_dir, exist_ok=True)
    os.makedirs(svg_resources_dir, exist_ok=True)
    os.makedirs(png_temp_dir, exist_ok=True)
    
    return {
        "base_dir": flags_base_dir,
        "img_dir": flags_img_dir,
        "svg_dir": svg_resources_dir,
        "png_temp_dir": png_temp_dir
    }

def create_flag_metadata_entries(metadata_list, flag_name, original_stripes, modified_stripes, trial, resolution, filename):
    """Create metadata entries for a flag image"""
    
    # Extract country name for Q3
    country_name = flag_name.replace("Flag of the ", "").replace("Flag of ", "")
    
    # Define prompts
    q1_prompt = f"How many stripes are there in this flag? Answer with a number in curly brackets, e.g., {{9}}."
    q2_prompt = f"Count the stripes in this flag. Answer with a number in curly brackets, e.g., {{9}}."
    q3_prompt = f"Is this the flag of {country_name}? Answer in curly brackets, e.g., {{Yes}} or {{No}}."
    
    # Ground truth values
    gt_stripes = str(modified_stripes)
    bias_stripes = str(original_stripes)  # Bias might be toward original count
    
    # For Q3, if stripes are modified significantly, it might not be the "real" flag
    stripe_difference = abs(modified_stripes - original_stripes)
    if stripe_difference == 0:
        gt_flag = "Yes"
        bias_flag = "Yes"
    else:
        gt_flag = "No"  # Modified flag is not the real flag
        bias_flag = "Yes"  # Bias might be to assume it's still the real flag
    
    image_path_relative = os.path.join("images", filename).replace("\\", "/")
    
    # Create entries for Q1 and Q2 (stripe counting)
    for q_key, prompt in [("Q1", q1_prompt), ("Q2", q2_prompt)]:
        metadata_list.append({
            "image": filename,
            "image_path": image_path_relative,
            "topic": "Flags",
            "prompt": prompt,
            "ground_truth": gt_stripes,
            "expected_bias": bias_stripes,
            "with_title": False,
            "type_of_question": q_key,
            "pixel": resolution
        })
    
    # Create entry for Q3 (flag identification)
    metadata_list.append({
        "image": filename,
        "image_path": image_path_relative,
        "topic": "Flags",
        "prompt": q3_prompt,
        "ground_truth": gt_flag,
        "expected_bias": bias_flag,
        "with_title": False,
        "type_of_question": "Q3",
        "pixel": resolution
    })

def generate_flags_dataset(num_trials=3, stripe_changes=[-1], model_type='openai'):
    """Generate flags dataset with modified stripe counts"""
    
    # Check dependencies
    if model_type == 'openai' and not (HAS_OPENAI and OPENAI_API_KEY):
        logging.error("OpenAI library not available or API key missing. Cannot proceed.")
        return
    if model_type == 'anthropic' and not (HAS_ANTHROPIC and ANTHROPIC_API_KEY):
        logging.error("Anthropic library not available or API key missing. Cannot proceed.")
        return
    
    print("=== FLAGS DATASET GENERATION ===")
    print(f"Will generate flags with stripe modifications: {stripe_changes}")
    print(f"Number of trials per flag: {num_trials}")
    print(f"Using model: {model_type}")
    print()
    
    # Create directory structure
    output_dirs = create_directory_structure()
    all_metadata = []
    
    # Initialize downloader
    downloader = WikimediaSVGRetriever(
        openai_api_key=OPENAI_API_KEY,
        llm_model='gpt-4.1-mini',
        reranker='llm'
    )
    
    # Calculate total operations
    total_operations = len(FLAG_DATA) * len(stripe_changes) * num_trials
    print(f"Total flag modifications to generate: {total_operations}")
    
    # FOR LOOP 1: Generate ALL modified flag SVGs
    print("\n=== FOR LOOP 1: Generating modified flag SVGs ===")
    generated_svgs = []
    
    with tqdm(total=total_operations, desc="Generating flag SVGs") as pbar:
        for flag_info in FLAG_DATA:
            flag_name = flag_info["name"]
            original_stripes = flag_info["original_stripes"]
            
            for stripe_change in stripe_changes:
                instruction = convert_to_instruction(flag_name, original_stripes, stripe_change)
                new_stripe_count = original_stripes + stripe_change
                new_stripe_count = max(0, new_stripe_count)
                
                for trial in range(num_trials):
                    print(f"\nProcessing: {flag_name} (stripes: {original_stripes} → {new_stripe_count})")
                    print(f"Instruction: {instruction}")
                    
                    try:
                        output = modify_svg_feature(downloader, flag_name, instruction, model_type=model_type)
                        if output is None:
                            print("❌ Failed to generate SVG")
                            pbar.update(1)
                            continue
                        
                        # Extract SVG code from response
                        if '<code>' in output and '</code>' in output:
                            svg_code = output.split('<code>')[-1].split('</code>')[0].strip()
                        else:
                            svg_code = output.strip()
                        
                        # Create filename
                        country_clean = sanitize_filename(flag_name.replace("Flag of ", ""))
                        country_clean = country_clean.replace(' ', '_')
                        filename = f"{country_clean}_stripes_{new_stripe_count}_trial_{trial}.svg"
                        file_path = os.path.join(output_dirs["svg_dir"], filename)
                        
                        # Save SVG
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(svg_code)
                        
                        generated_svgs.append({
                            "path": file_path,
                            "flag_name": flag_name,
                            "original_stripes": original_stripes,
                            "modified_stripes": new_stripe_count,
                            "trial": trial,
                            "filename": filename
                        })
                        
                        print(f"✅ Generated: {filename}")
                        
                    except Exception as e:
                        print(f"❌ Error generating {flag_name}: {e}")
                    
                    pbar.update(1)
                    time.sleep(1)  # Rate limiting
    
    print(f"\n=== FOR LOOP 2: Converting SVGs to PNGs ===")
    # Convert SVGs to PNGs
    print('converting svg -> png')
    convert_svgs_to_pngs(output_dirs["svg_dir"], output_dirs["png_temp_dir"])
    
    print(f"\n=== FOR LOOP 3: Resizing images and creating metadata ===")
    # FOR LOOP 3: Resize images and create metadata
    successful_conversions = 0
    
    for svg_data in tqdm(generated_svgs, desc="Processing final images"):
        png_filename = svg_data["filename"].replace('.svg', '.png')
        png_temp_path = os.path.join(output_dirs["png_temp_dir"], png_filename)
        
        # Check if PNG conversion was successful
        if not os.path.exists(png_temp_path):
            print(f"⚠️ PNG not found for {svg_data['filename']}, skipping...")
            continue
        
        for resolution in RESOLUTIONS:
            # Create final filename
            final_filename = f"{svg_data['filename'].replace('.svg', '')}_px{resolution}.png"
            final_output_path = os.path.join(output_dirs["img_dir"], final_filename)
            
            if resize_image(png_temp_path, final_output_path, resolution):
                successful_conversions += 1
                # Create metadata entries
                create_flag_metadata_entries(
                    all_metadata, svg_data['flag_name'], svg_data['original_stripes'],
                    svg_data['modified_stripes'], svg_data['trial'], resolution, final_filename
                )
    
    # Save metadata
    print("\n=== Saving metadata ===")
    save_metadata_files(all_metadata, output_dirs["base_dir"], "flags_notitle")
    
    # Summary
    final_img_count = len([f for f in os.listdir(output_dirs["img_dir"]) if f.endswith('.png')])
    print(f"\n=== Flags Generation Summary ===")
    print(f"Flag SVGs generated: {len(generated_svgs)}")
    print(f"Successful PNG conversions: {successful_conversions}")
    print(f"Final dataset images: {final_img_count}")
    print(f"Total metadata entries: {len(all_metadata)}")
    print(f"Dataset saved to: {output_dirs['base_dir']}")

def main():
    """Main function to generate flags dataset"""
    parser = argparse.ArgumentParser(description='Generate Flags "notitle" dataset.')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials per flag modification (default: 3)')
    parser.add_argument('--stripe-changes', type=int, nargs='+', default=[1,-1],
                        help='List of stripe count changes (default: [1])')
    parser.add_argument('--model', type=str, choices=['openai', 'anthropic'], default='openai',
                        help='LLM model to use for SVG modification (default: openai)')
    
    args = parser.parse_args()
    
    generate_flags_dataset(
        num_trials=args.trials,
        stripe_changes=args.stripe_changes,
        model_type=args.model
    )

if __name__ == '__main__':
    main()