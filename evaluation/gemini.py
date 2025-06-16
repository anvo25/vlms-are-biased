import argparse
import json
import os
import time
import concurrent.futures
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
import threading

file_lock = threading.Lock()

def read_api_key(key_path):
    try:
        with open(key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key not found at {key_path}. Please make sure the file exists.")

def make_api_call(client, model, prompt, image_path, temperature=1):
    try:
        image = Image.open(image_path)
        
        for attempt in range(3):
            try:
                config = genai.types.GenerateContentConfig(
                    temperature=temperature
                )
                
                response = client.models.generate_content(
                    model=model,
                    contents=[image, prompt],
                    config=config
                )
                return response
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt < 2: 
                    print(f"Retrying in 60 seconds...")
                    time.sleep(60)
                else:
                    raise
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}. Please check the path.")

def ensure_dir_exists(directory):
    if directory:
        os.makedirs(os.path.dirname(directory), exist_ok=True)

def load_existing_results(output_file):
    """Load existing results if the file exists."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            return results
        except json.JSONDecodeError:
            print(f"Warning: {output_file} exists but is not valid JSON. Starting fresh.")
    return []

def get_processed_ids(results):
    """Get a set of IDs that have already been processed."""
    return {item['ID'] for item in results if 'ID' in item}

def get_output_filename(input_file, model_name):
    """Generate output filename based on input filename and model name,
    following the structure: results/model_name/full_notitle or full_withtitle"""
    input_basename = os.path.basename(input_file)
    input_name = os.path.splitext(input_basename)[0]
    
    if "notitle" in input_file.lower():
        title_type = "full_notitle"
    elif "withtitle" in input_file.lower():
        title_type = "full_withtitle"
    else:
        title_type = "full_notitle"
    
    return f"results/{model_name}/{title_type}/results_{input_name}.json"

def process_item(args):
    """Process a single item. This function is called by the thread pool."""
    client, model, item, temperature, current_results, output_file, prefix = args
    item_id = item['ID']
    print(f"Processing item {item_id}")
    
    try:
        prompt_text = item['prompt']
        if prefix:
            prompt_text = f"{prefix}\n{prompt_text}"
            
        response = make_api_call(
            client, 
            model, 
            prompt_text, 
            item['image_path'], 
            temperature
        )
        
        item_with_response = item.copy()
        
        item_with_response['processed_prompt'] = prompt_text
        
        response_dict = {
            "text": response.text,
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": response.text}]
                    },
                    "finish_reason": getattr(response, "finish_reason", "STOP"),
                    "safety_ratings": getattr(response, "safety_ratings", [])
                }
            ]
        }
        
        item_with_response['api_response'] = response_dict
        
        with file_lock:
            current_results.append(item_with_response)
            save_results_to_file(current_results, output_file)
            print(f"  → Saved progress for {item_id}")
            
        return item_with_response
        
    except Exception as e:
        print(f"Failed to process {item_id}: {e}")
        return None

def save_results_to_file(results, output_file):
    """Save results to the output file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results to file: {e}")
        try:
            backup_file = f"{output_file}.backup"
            with open(backup_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to backup file: {backup_file}")
        except:
            print("Failed to save to backup file as well. Results may be lost.")


def main():
    parser = argparse.ArgumentParser(description="Google Gemini API")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-preview-05-06", help="Gemini model to use")
    parser.add_argument("--input-file", type=str, help="Input JSON file path")
    parser.add_argument("--output-file", type=str, help="Output JSON file path (optional, will be auto-generated if not provided)")
    parser.add_argument("--api-key-path", type=str, default="api_key/gemini_key.txt", help="Path to Gemini API key file")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the model")
    parser.add_argument("--test", action="store_true", help="Test mode with only 2 samples")
    parser.add_argument("--concurrent", type=int, default=10, help="Number of samples to process concurrently")
    parser.add_argument("--prefix", default=None, type=str, help="Optional prefix to add to all prompts")
    
    args = parser.parse_args()
    
    model_parts = args.model.split('-')
    if len(model_parts) >= 2:
        if model_parts[0] == "gemini":
            if len(model_parts) >= 3 and model_parts[2] == "pro":
                model_folder_name = f"{model_parts[0]}-{model_parts[1]}-{model_parts[2]}"  
            else:
                model_folder_name = f"{model_parts[0]}-{model_parts[1]}" 
        else:
            model_folder_name = model_parts[0]
    else:
        model_folder_name = args.model
    
    output_file = args.output_file if args.output_file else get_output_filename(args.input_file, model_folder_name)
    
    ensure_dir_exists(output_file)
    
    api_key = read_api_key(args.api_key_path)
    client = genai.Client(api_key=api_key) 
    
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    if args.test:
        data = data[:2]
    
    results = load_existing_results(output_file)
    processed_ids = get_processed_ids(results)
    
    items_to_process = [item for item in data if item['ID'] not in processed_ids]
    
    print(f"Found {len(results)} existing results. {len(items_to_process)} items left to process.")
    print(f"Using output file: {output_file}")
    print(f"Processing with concurrency level: {args.concurrent}")
    if args.prefix:
        print(f"Using prefix: \"{args.prefix}\"")
    
    if not items_to_process:
        print("All items have already been processed. Nothing to do.")
        return
    
    successful_items = []
    failed_items = []
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            process_args = [
                (client, args.model, item, args.temperature, results, output_file, args.prefix)
                for item in items_to_process
            ]
            
            futures = [executor.submit(process_item, arg) for arg in process_args]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        successful_items.append(result['ID'])
                    else:
                        failed_items.append("Unknown item")
                except Exception as e:
                    print(f"Exception occurred during processing: {e}")
                    failed_items.append("Unknown item")
    except Exception as e:
        print(f"Exception in main processing loop: {e}")
    finally:
        print("Saving final results...")
        save_results_to_file(results, output_file)
        
        summary_file = f"{output_file}.summary.txt"
        try:
            with open(summary_file, 'w') as f:
                f.write(f"Processing Summary\n")
                f.write(f"=================\n")
                f.write(f"Model: {args.model}\n")
                if args.prefix:
                    f.write(f"Prefix: \"{args.prefix}\"\n")
                f.write(f"Temperature: {args.temperature}\n")
                f.write(f"Total items processed: {len(successful_items) + len(failed_items)}\n")
                f.write(f"Successfully processed: {len(successful_items)}\n")
                f.write(f"Failed to process: {len(failed_items)}\n")
                f.write(f"Total results saved: {len(results)}\n")
                
                if failed_items:
                    f.write("\nFailed items:\n")
                    for item in failed_items:
                        f.write(f"- {item}\n")
            print(f"Summary saved to {summary_file}")
        except Exception as e:
            print(f"Failed to write summary file: {e}")
    
    print(f"All done! Successfully processed {len(successful_items)} items, failed {len(failed_items)}.")
    print(f"Results saved to {output_file}")
    
    if failed_items:
        print(f"Some items failed to process. Consider running the script again to retry these items.")

if __name__ == "__main__":
    main()