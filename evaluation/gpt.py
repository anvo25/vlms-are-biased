import argparse
import json
import os
import time
import threading
from pathlib import Path
import base64
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

thread_local = threading.local()

def read_api_key(key_path):
    try:
        with open(key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key not found at {key_path}. Please make sure the file exists.")

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}. Please check the path.")

def get_client(api_key):
    if not hasattr(thread_local, 'client'):
        thread_local.client = OpenAI(api_key=api_key)
    return thread_local.client

def make_api_call(api_key, model, prompt, image_path, temperature=1, prefix=None):
    client = get_client(api_key)
    base64_image = encode_image_to_base64(image_path)
    
    if prefix:
        prompt = f"{prefix}\n{prompt}"
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            return response
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {image_path}: {e}")
            if attempt < 2: 
                print(f"Retrying in 60 seconds...")
                time.sleep(60)
            else:
                raise

def ensure_dir_exists(directory):
    if directory:
        Path(os.path.dirname(directory)).mkdir(parents=True, exist_ok=True)

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
    """Generate output filename based on input filename and model name."""
    input_basename = os.path.basename(input_file)
    input_name = os.path.splitext(input_basename)[0]
    return f"results/{model_name}/results_{input_name}.json"

def process_item(item, api_key, model, temperature, prefix=None):
    """Process a single item and return the result"""
    item_id = item['ID']
    image_path = item['image_path']
    
    try:
        response = make_api_call(
            api_key, 
            model, 
            item['prompt'], 
            image_path, 
            temperature,
            prefix
        )
        
        item_with_response = item.copy()
        item_with_response['api_response'] = response.model_dump()
        print(f"Successfully processed {item_id}")
        return item_with_response
    except Exception as e:
        print(f"Failed to process {item_id}: {e}")
        return None

def save_results(results, output_file, lock):
    """Thread-safe save results to file"""
    with lock:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="OpenAI API")
    parser.add_argument("--model", type=str, help="OpenAI model to use")
    parser.add_argument("--input-file", type=str, help="Input JSON file path")
    parser.add_argument("--output-file", type=str, help="Output JSON file path (optional, will be auto-generated if not provided)")
    parser.add_argument("--api-key-path", type=str, default="api_key/openai_key.txt", help="Path to OpenAI API key file")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the model")
    parser.add_argument("--test", action="store_true", help="Test mode with only 2 samples")
    parser.add_argument("--concurrency", type=int, default=300, help="Number of concurrent API calls")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix to add to all prompts")
    
    args = parser.parse_args()
    
    model_folder_name = args.model.split('-')[0]  
    if model_folder_name == "gpt":
        model_parts = args.model.split('-')
        if len(model_parts) > 1:
            model_folder_name = f"{model_parts[0]}-{model_parts[1]}" 
    
    output_file = args.output_file if args.output_file else get_output_filename(args.input_file, model_folder_name)
    
    ensure_dir_exists(output_file)
    
    api_key = read_api_key(args.api_key_path)
    
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    if args.test:
        data = data[:2]
    
    results = load_existing_results(output_file)
    processed_ids = get_processed_ids(results)
    
    print(f"Found {len(results)} existing results. Continuing from where we left off.")
    print(f"Using output file: {output_file}")
    print(f"Running with concurrency: {args.concurrency}")
    if args.prefix:
        print(f"Using prefix: {args.prefix}")
    
    results_lock = threading.Lock()
    save_lock = threading.Lock()
    
    items_to_process = [item for item in data if item['ID'] not in processed_ids]
    print(f"Found {len(items_to_process)} items to process out of {len(data)} total items.")
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_item = {
            executor.submit(
                process_item, 
                item, 
                api_key, 
                args.model, 
                args.temperature,
                args.prefix
            ): item for item in items_to_process
        }
        
        for i, future in enumerate(as_completed(future_to_item)):
            item = future_to_item[future]
            try:
                result = future.result()
                if result:
                    with results_lock:
                        results.append(result)
                        
                    if (i + 1) % 5 == 0 or i == len(items_to_process) - 1:
                        save_results(results, output_file, save_lock)
                        print(f"Saved progress ({len(results)}/{len(data)} items) to {output_file}")
            except Exception as exc:
                print(f"Item {item['ID']} generated an exception: {exc}")
    
    save_results(results, output_file, save_lock)
    print(f"All done! Processed {len(items_to_process)} items, {len(results)} total. Results saved to {output_file}")

if __name__ == "__main__":
    main()