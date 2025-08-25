import json
import os
from typing import Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import random

# Configuration
class Config:
    HARMFUL_DATASET_PATH = "datasets/harmful_dataset.json"
    HARMLESS_DATASET_PATH = "datasets/harmless_similar_dataset.json"
    EXTENDED_HARMFUL_PATH = "datasets/answered_harmful_dataset.json"
    EXTENDED_HARMLESS_PATH = "datasets/answered_harmless_similar_dataset.json"
    MODEL_NAME = "<model-name>"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MAX_NEW_TOKENS = 512  # Increased for more natural responses
    TEMPERATURE = 0.8     # Slightly higher for more natural variation
    TOP_P = 0.9           
    SAVE_EVERY = 5        
    MAX_RETRIES = 3       # Reduced retries for more vanilla responses
    REGENERATE_MODE = "all"  # Options: "all" or "empty_only"

# Global registry for failed generations
failed_registry = {}

def is_empty_response(response: str) -> bool:
    return not response.strip()

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return []

def setup_model():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME, 
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return model, tokenizer

def generate_response(model, tokenizer, history, conv_id, turn_id, max_retries=Config.MAX_RETRIES):
    """Generate response with minimal intervention - keep responses as vanilla as possible"""
    # More conservative retry parameters to maintain vanilla responses
    retry_params = [
        (Config.TEMPERATURE, Config.TOP_P, Config.MAX_NEW_TOKENS),
        (0.9, 0.95, Config.MAX_NEW_TOKENS),  # Slightly more conservative
        (1.0, 0.85, Config.MAX_NEW_TOKENS + 100)  # Last resort with more tokens
    ]
    current_errors = []
    used_params = []
    
    for attempt in range(max_retries):
        try:
            # Tokenization
            try:
                chat_input = tokenizer.apply_chat_template(history, return_tensors="pt")
                if chat_input.shape[1] > tokenizer.model_max_length:
                    # Truncate context instead of failing completely
                    print(f"Input too long, truncating context for conversation {conv_id}, turn {turn_id}")
                    # Keep only the last few turns to fit in context
                    truncated_history = history[-3:]  # Keep last 3 exchanges
                    chat_input = tokenizer.apply_chat_template(truncated_history, return_tensors="pt")
                
                inputs = chat_input.to(Config.DEVICE)
                attention_mask = torch.ones_like(inputs)
            except Exception as e:
                error_msg = f"Tokenization error: {e}"
                print(error_msg)
                current_errors.append(error_msg)
                used_params.append(retry_params[min(attempt, len(retry_params)-1)])
                continue
            
            temp, tp, max_tokens = retry_params[min(attempt, len(retry_params)-1)]
            used_params.append((temp, tp, max_tokens))
            print(f"Conversation {conv_id}, Turn {turn_id}: Attempt {attempt+1} using temperature={temp}, top_p={tp}")
            
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        top_p=tp,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Slight penalty to avoid repetition
                    )
            except torch.cuda.OutOfMemoryError:
                error_msg = "CUDA out of memory during generation"
                print(error_msg)
                torch.cuda.empty_cache()
                time.sleep(2)
                current_errors.append(error_msg)
                continue
            except Exception as e:
                error_msg = f"Generation error: {e}"
                print(error_msg)
                current_errors.append(error_msg)
                continue
            
            raw_token_ids = outputs[0][inputs.shape[1]:]
            
            try:
                response = tokenizer.decode(raw_token_ids, skip_special_tokens=True)
            except Exception as e:
                error_msg = f"Decoding error: {e}"
                print(error_msg)
                current_errors.append(error_msg)
                continue
            
            # Only retry if response is completely empty - accept weird/unusual content
            if is_empty_response(response):
                error_msg = "Empty response generated."
                print(f"{error_msg} Retrying... (Attempt {attempt+1}/{max_retries})")
                current_errors.append(error_msg)
                time.sleep(1)
                continue
            
            print(f"Conversation {conv_id}, Turn {turn_id}: Successful generation")
            return response.strip()  # Only strip whitespace, keep all content
            
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(f"{error_msg} Retrying... (Attempt {attempt+1}/{max_retries})")
            current_errors.append(error_msg)
            time.sleep(1)
    
    print(f"Failed to generate response for conversation {conv_id}, turn {turn_id} after {max_retries} attempts")
    if conv_id not in failed_registry:
        failed_registry[conv_id] = {}
    failed_registry[conv_id][turn_id] = {
        "fail_count": max_retries,
        "errors": current_errors,
        "hyperparams": used_params
    }
    return ""  # Return empty string instead of None to continue conversation

def extend_dataset(dataset: List[Dict[str, Any]], model, tokenizer, output_path: str, 
                   existing_data=None) -> List[Dict[str, Any]]:
    if existing_data and Config.REGENERATE_MODE == "empty_only":
        extended_dataset = existing_data.copy()
        print(f"Loaded {len(extended_dataset)} existing conversations for filling empty responses")
    else:
        extended_dataset = []
    
    partial_path = output_path.replace('.json', '_partial.json')
    
    for idx, data_item in enumerate(dataset):
        print(f"Processing conversation {idx+1}/{len(dataset)}")
        
        # Extract conversation data from new structure
        conversation = data_item.get("conversation", {})
        is_harmful = data_item.get("is_harmful", False)
        source = data_item.get("source", "unknown")
        length = data_item.get("length", len(conversation))
        
        # Initialize extended item with metadata
        if Config.REGENERATE_MODE == "empty_only" and idx < len(extended_dataset):
            extended_item = extended_dataset[idx].copy()
        else:
            extended_item = {
                "conversation": {},
                "is_harmful": is_harmful,
                "source": source,
                "length": length
            }
        
        # Check if we need to process this conversation in empty_only mode
        if Config.REGENERATE_MODE == "empty_only" and idx < len(extended_dataset):
            has_empty_response = False
            for turn_key in conversation.keys():
                turn_data = extended_item["conversation"].get(turn_key, {})
                if not isinstance(turn_data, dict) or is_empty_response(turn_data.get("response", "")):
                    has_empty_response = True
                    break
            
            if not has_empty_response:
                print(f"Skipping conversation {idx+1} - no empty responses")
                continue
            else:
                print(f"Filling empty responses in conversation {idx+1}")
        
        # Process each turn in order
        sorted_turn_keys = sorted(conversation.keys(), key=int)
        
        for turn_key in sorted_turn_keys:
            turn_num = turn_key  # Keep original turn numbering
            
            # Skip if response already exists and is not empty (in empty_only mode)
            if (Config.REGENERATE_MODE == "empty_only" and 
                turn_num in extended_item["conversation"] and 
                isinstance(extended_item["conversation"][turn_num], dict) and
                "response" in extended_item["conversation"][turn_num] and
                not is_empty_response(extended_item["conversation"][turn_num]["response"])):
                continue
            
            user_prompt = conversation[turn_key]
            
            # Initialize turn data
            if turn_num not in extended_item["conversation"]:
                extended_item["conversation"][turn_num] = {}
            
            extended_item["conversation"][turn_num]["prompt"] = user_prompt
            
            # Build conversation history
            history = []
            for prev_turn_key in sorted_turn_keys:
                if int(prev_turn_key) >= int(turn_num):
                    break
                    
                prev_turn_data = extended_item["conversation"].get(prev_turn_key, {})
                if isinstance(prev_turn_data, dict):
                    history.append({"role": "user", "content": prev_turn_data.get("prompt", "")})
                    prev_response = prev_turn_data.get("response", "")
                    if prev_response:  # Include even weird responses to maintain conversation flow
                        history.append({"role": "assistant", "content": prev_response})
            
            # Add current user prompt
            history.append({"role": "user", "content": user_prompt})
            
            # Generate response - continue even if previous turns failed
            response = generate_response(model, tokenizer, history, conv_id=idx+1, turn_id=turn_num)
            extended_item["conversation"][turn_num]["response"] = response
            
            if response:
                print(f"Generated response for conversation {idx+1}, turn {turn_num}")
            else:
                print(f"Empty response for conversation {idx+1}, turn {turn_num} - continuing anyway")
        
        # Update or append the conversation
        if Config.REGENERATE_MODE == "empty_only" and idx < len(extended_dataset):
            extended_dataset[idx] = extended_item
        else:
            if idx < len(extended_dataset):
                extended_dataset[idx] = extended_item
            else:
                extended_dataset.append(extended_item)
        
        # Save periodically
        if (idx + 1) % Config.SAVE_EVERY == 0 or idx == len(dataset) - 1:
            save_dataset(extended_dataset, partial_path)
            print(f"Saved partial results up to conversation {idx+1}")
    
    # Final save
    save_dataset(extended_dataset, output_path)
    
    # Save failure registry
    try:
        with open("failed_generations_registry.json", "w", encoding="utf-8") as f:
            json.dump(failed_registry, f, ensure_ascii=False, indent=2)
        print("Failed generations registry saved to failed_generations_registry.json")
    except Exception as e:
        print(f"Error saving failed generations registry: {e}")
    
    return extended_dataset

def save_dataset(dataset: List[Dict[str, Any]], file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {file_path}")
    except Exception as e:
        print(f"Error saving dataset to {file_path}: {e}")

def load_existing_dataset(file_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing dataset from {file_path}: {e}")
    return None

def main():
    print(f"Using device: {Config.DEVICE}")
    print(f"Regeneration mode: {Config.REGENERATE_MODE}")
    
    harmful_dataset = load_dataset(Config.HARMFUL_DATASET_PATH)
    harmless_dataset = load_dataset(Config.HARMLESS_DATASET_PATH)
    if not harmful_dataset or not harmless_dataset:
        print("Failed to load datasets. Exiting.")
        return
    print(f"Loaded harmful dataset with {len(harmful_dataset)} conversations.")
    print(f"Loaded harmless dataset with {len(harmless_dataset)} conversations.")
    
    model, tokenizer = setup_model()
    print(f"Model {Config.MODEL_NAME} loaded successfully.")
    
    existing_harmful = None
    existing_harmless = None
    if Config.REGENERATE_MODE == "empty_only":
        existing_harmful = load_existing_dataset(Config.EXTENDED_HARMFUL_PATH)
        existing_harmless = load_existing_dataset(Config.EXTENDED_HARMLESS_PATH)
    
    print("\nProcessing harmful dataset...")
    extend_dataset(harmful_dataset, model, tokenizer, Config.EXTENDED_HARMFUL_PATH, existing_harmful)
    
    print("\nProcessing harmless dataset...")
    extend_dataset(harmless_dataset, model, tokenizer, Config.EXTENDED_HARMLESS_PATH, existing_harmless)
    
    print("\nAll datasets processed and saved successfully!")

if __name__ == "__main__":
    main()