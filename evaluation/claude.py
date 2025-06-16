import argparse
import json
import os
import time
import threading
from pathlib import Path
import base64
from anthropic import Anthropic
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

def get_image_media_type(image_path):
    ext = Path(image_path).suffix.lower()
    if ext == ".jpg" or ext == ".jpeg":
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        print(f"Warning: Unknown image type for {image_path}, defaulting to image/jpeg.")
        return "image/jpeg"

def get_client(api_key):
    if not hasattr(thread_local, 'client'):
        thread_local.client = Anthropic(api_key=api_key)
    return thread_local.client

def make_api_call(api_key, model, prompt, image_path, temperature=1.0, max_tokens=1024):
    client = get_client(api_key)
    base64_image = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )
            return response.model_dump()
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
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            return results
        except json.JSONDecodeError:
            print(f"Warning: {output_file} exists but is not valid JSON. Starting fresh.")
    return []

def get_processed_ids(results):
    return {item['ID'] for item in results if 'ID' in item}

def get_output_filename(input_file, model_name):
    input_basename = os.path.basename(input_file)
    input_name = os.path.splitext(input_basename)[0]
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    return f"results_single/claude/{safe_model_name}/results_{input_name}.json"

def process_item(item, api_key, model, temperature, max_tokens):
    """Process a single item and return the result"""
    item_id = item['ID']
    image_path = item['image_path']
    
    try:
        response_dict = make_api_call(
            api_key, 
            model, 
            item['prompt'], 
            image_path, 
            temperature,
            max_tokens
        )
        
        item_with_response = item.copy()
        item_with_response['api_response'] = response_dict
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
    parser = argparse.ArgumentParser(description="Claude API")
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet-20250219", help="Claude model to use")
    parser.add_argument("--input-file", type=str, help="Input JSON file path")
    parser.add_argument("--output-file", type=str, help="Output JSON file path (optional, will be auto-generated if not provided)")
    parser.add_argument("--api-key-path", type=str, default="api_key/claude_key.txt", help="Path to Claude API key file")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the model")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for the model response")
    parser.add_argument("--test", action="store_true", help="Test mode with only 2 samples")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent API calls")
    
    args = parser.parse_args()

    output_file = args.output_file if args.output_file else get_output_filename(args.input_file, args.model)
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
    
    results_lock = threading.Lock()
    save_lock = threading.Lock()
    
    items_to_process = [item for item in data if item['ID'] not in processed_ids]
    print(f"Found {len(items_to_process)} items to process out of {len(data)} total items.")
    
    total_processed_this_run = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_item = {
            executor.submit(
                process_item, 
                item, 
                api_key, 
                args.model, 
                args.temperature,
                args.max_tokens
            ): item for item in items_to_process
        }
        
        for i, future in enumerate(as_completed(future_to_item)):
            item = future_to_item[future]
            try:
                result = future.result()
                if result:
                    with results_lock:
                        results.append(result)
                        total_processed_this_run += 1
                    
                    if (i + 1) % 5 == 0 or i == len(items_to_process) - 1:
                        save_results(results, output_file, save_lock)
                        print(f"Saved progress ({len(results)}/{len(data)} items) to {output_file}")
            except Exception as exc:
                print(f"Item {item['ID']} generated an exception: {exc}")
    
    save_results(results, output_file, save_lock)
    print(f"All done! Processed {total_processed_this_run} new items, {len(results)} total. Results saved to {output_file}")

if __name__ == "__main__":
    main()