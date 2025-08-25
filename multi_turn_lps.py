import json
import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import Trace, TraceDict
from tqdm import tqdm
from huggingface_hub import login
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import h5py
import logging
import sys
import os
from datetime import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set global font sizes
plt.rcParams.update({
    # 'font.family': 'sans-serif',
    # 'font.sans-serif': ['Arial'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Set up logging
def setup_logging(log_dir: Path, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Configure logging to file and console with different verbosity levels.
    
    Args:
        log_dir: Directory to store log files
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    log_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main logger
    logger = logging.getLogger('multi_turn_lps')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear any existing handlers
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Setup main file handler for general logs
    main_log_file = log_dir / f"main_{timestamp}.log"
    file_handler = logging.FileHandler(main_log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger, log_dir, timestamp

# Utility function to create specialized loggers for specific components
def get_component_logger(name, log_dir, timestamp):
    """Get a logger for a specific component with its own log file."""
    logger = logging.getLogger(f'multi_turn_lps.{name}')
    
    # Check if this logger already has handlers
    if not logger.handlers:
        # Create file handler for component-specific logs
        component_log_file = log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(component_log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def precision_at_k(y_true, y_scores, k):
    """
    Calculate precision at k: among the top k samples with highest scores,
    what percentage are true positives?
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted scores/probabilities
        k: Number of top samples to consider
    
    Returns:
        Precision@k score
    """
    if k <= 0 or len(y_true) == 0:
        return 0.0
    
    # Ensure k is not larger than the number of samples
    k = min(k, len(y_true))
    
    # Get indices of top k highest scores
    top_k_indices = np.argsort(y_scores)[-k:]
    
    # Calculate precision for those indices
    return np.mean([y_true[i] for i in top_k_indices])

def load_preprocessed_data(processed_data_path: Path) -> List[Dict]:
    """
    Load preprocessed conversation data from a JSON file.
    Updated to handle the new structure with jailbreak scores.
    """
    print(f"Loading preprocessed data from {processed_data_path}")
    
    with open(processed_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Validate new structure
    sample_conversation = data[0] if data else {}
    if "conversation" not in sample_conversation:
        raise ValueError("Invalid data structure: missing 'conversation' field")
    
    # Check if jailbreak scores are present
    has_jailbreak_scores = False
    if data:
        sample_conv = sample_conversation.get("conversation", {})
        for turn_data in sample_conv.values():
            if isinstance(turn_data, dict) and "jailbreak_score" in turn_data:
                has_jailbreak_scores = True
                break
    
    print(f"Dataset contains jailbreak scores: {has_jailbreak_scores}")
    
    # Print statistics for verification
    harmful_count = sum(1 for item in data if item.get("is_harmful", False))
    harmless_count = len(data) - harmful_count
    
    print(f"Loaded {len(data)} conversations: {harmful_count} harmful, {harmless_count} harmless")
    
    # Count by length and check jailbreak score statistics
    length_counts = {}
    jailbreak_score_stats = {"total": 0, "nonzero": 0, "failed": 0}
    
    for item in data:
        length = item.get("length", len(item.get("conversation", {})))
        if length not in length_counts:
            length_counts[length] = {"harmful": 0, "harmless": 0}
        
        if item.get("is_harmful", False):
            length_counts[length]["harmful"] += 1
        else:
            length_counts[length]["harmless"] += 1
        
        # Check jailbreak scores if present
        if has_jailbreak_scores:
            conversation = item.get("conversation", {})
            for turn_data in conversation.values():
                if isinstance(turn_data, dict) and "jailbreak_score" in turn_data:
                    score = turn_data["jailbreak_score"]
                    jailbreak_score_stats["total"] += 1
                    if score > 0:
                        jailbreak_score_stats["nonzero"] += 1
                    elif score == -1.0:
                        jailbreak_score_stats["failed"] += 1
    
    print("Conversation count by length:")
    for length, counts in sorted(length_counts.items()):
        print(f"  Length {length}: {counts['harmful']} harmful + {counts['harmless']} harmless")
    
    if has_jailbreak_scores and jailbreak_score_stats["total"] > 0:
        print("Jailbreak score statistics:")
        print(f"  Total turns with scores: {jailbreak_score_stats['total']}")
        print(f"  Nonzero scores: {jailbreak_score_stats['nonzero']} ({jailbreak_score_stats['nonzero']/jailbreak_score_stats['total']:.1%})")
        print(f"  Failed parsing (score=-1): {jailbreak_score_stats['failed']} ({jailbreak_score_stats['failed']/jailbreak_score_stats['total']:.1%})")
    
    return data

def create_stratified_test_set(metadata: List[Dict], test_percentage: float = 0.4):
    """
    Create a stratified test set based on harmfulness and conversation length.
    
    Args:
        metadata: List of conversation metadata
        test_percentage: Percentage of conversations to use for testing
    
    Returns:
        Set of conversation IDs selected for the test set
    """
    # Group conversations by harmfulness and length
    stratified_groups = {}
    
    for meta in metadata:
        is_harmful = meta["is_harmful"]
        length = meta.get("conversation_length", 0)  # Get conversation length with fallback
        key = (is_harmful, length)
        
        if key not in stratified_groups:
            stratified_groups[key] = []
            
        # Get conversation ID with fallback options
        conv_id = meta.get("conversation_id") or meta.get("id")
        if conv_id is None:
            conv_id = f"unknown_{len(stratified_groups[key])}"

        stratified_groups[key].append(conv_id)
    
    # Create test set with appropriate representation from each stratum
    test_set_ids = set()
    
    # Use fixed random seed for reproducibility
    np.random.seed(42)
    
    print("Test set selection by stratum:")
    for (is_harmful, length), conv_ids in stratified_groups.items():
        # Shuffle the conversation IDs
        np.random.shuffle(conv_ids)
        
        # Calculate how many to select for test set (minimum 1 if available)
        n_test = max(1, int(len(conv_ids) * test_percentage))
        n_test = min(n_test, len(conv_ids))  # Don't select more than available
        
        # Select the test set
        test_ids = set(conv_ids[:n_test])
        test_set_ids.update(test_ids)
        
        print(f"  {'Harmful' if is_harmful else 'Harmless'}, length {length}: "
              f"{len(test_ids)}/{len(conv_ids)} conversations selected")
    
    return test_set_ids

def setup_model(model_name: str, device: str):
    """Load the transformer model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Fix for padding token error
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set padding token to EOS token: {tokenizer.pad_token}")

    print(f"Model loaded: {model_name}")
    return model, tokenizer

def extract_activations(model, tokenizer, prompt_text: str, device: str, layer_num: int) -> np.ndarray:
    """Extract activation vectors for a specific layer."""
    with torch.no_grad():
        # Process the raw text
        inputs = tokenizer.encode(
            prompt_text, 
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        # Explicitly ensure inputs are Long type
        inputs = inputs.long()
        inputs = inputs.to(device)

        # Get specified layer
        layer = model.model.layers[layer_num]
        with Trace(layer) as cache:
            _ = model(input_ids=inputs)
            activation_vector = cache.output[0].detach().cpu().numpy()[:, -1:, :]

        del inputs
        torch.cuda.empty_cache()

    return activation_vector

def extract_multi_turn_activations(model, tokenizer, conversation: Dict, device: str, 
                                 layer_num: int, turn_id: int, logger=None) -> Dict:
    """
    Extract different types of activations for multi-turn analysis:
    1. Current turn only (immediate response)
    2. Previous turn only (immediate context) 
    3. Full conversation history (cumulative context)
    4. Turn-to-turn differences (evolution)
    """
    activations = {}
    
    # Get sorted turn keys
    turn_keys = sorted([int(k) for k in conversation.keys()])
    current_turn_key = turn_keys[turn_id] if turn_id < len(turn_keys) else turn_keys[-1]
    
    if logger:
        logger.debug(f"Extracting activations for turn {turn_id} (key={current_turn_key}), layer {layer_num}")
    
    # 1. Current turn only - just the immediate prompt
    current_turn_data = conversation.get(str(current_turn_key), {})
    current_prompt = current_turn_data.get("prompt", "")
    if current_prompt:
        activations['current_turn'] = extract_activations(model, tokenizer, current_prompt, device, layer_num)
        if logger:
            logger.debug(f"Extracted 'current_turn' activation with shape {activations['current_turn'].shape}")
    
    # 2. Previous turn only (if exists)
    if turn_id > 0:
        prev_turn_key = turn_keys[turn_id - 1]
        prev_turn_data = conversation.get(str(prev_turn_key), {})
        prev_prompt = prev_turn_data.get("prompt", "")
        if prev_prompt:
            activations['previous_turn'] = extract_activations(model, tokenizer, prev_prompt, device, layer_num)
            if logger:
                logger.debug(f"Extracted 'previous_turn' activation with shape {activations['previous_turn'].shape}")
    elif logger:
        logger.debug("No previous turn available (first turn)")
    
    # 3. Full conversation history up to current turn
    full_history = build_conversation_history(conversation, turn_id)
    if full_history:
        activations['full_history'] = extract_activations(model, tokenizer, full_history, device, layer_num)
        if logger:
            logger.debug(f"Extracted 'full_history' activation with shape {activations['full_history'].shape}")
    
    # 4. Previous conversation history (without current turn) for comparison
    if turn_id > 0:
        prev_history = build_conversation_history(conversation, turn_id - 1)
        if prev_history:
            activations['previous_history'] = extract_activations(model, tokenizer, prev_history, device, layer_num)
            if logger:
                logger.debug(f"Extracted 'previous_history' activation with shape {activations['previous_history'].shape}")
    elif logger:
        logger.debug("No previous history available (first turn)")
    
    # Log final activation types extracted
    if logger:
        logger.debug(f"Activation types extracted: {list(activations.keys())}")
    
    return activations


def extract_multi_layer_activations_batched(model, tokenizer, prompt_texts: List[str],
                                           device: str, layer_nums: List[int]) -> Dict[int, List[np.ndarray]]:
    """Extract activations for multiple layers in a single forward pass for a batch of prompts."""
    activations = {layer_num: [] for layer_num in layer_nums}

    with torch.no_grad():
        # Process multiple texts in one batch
        batch_inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)

        # Process each layer separately (still batched by prompts)
        for layer_num in layer_nums:
            # Get specified layer
            layer = model.model.layers[layer_num]

            # Use Trace (singular) for one layer at a time - with detailed output inspection
            with Trace(layer) as cache:
                _ = model(**batch_inputs)

                # For each sequence in the batch
                for i in range(len(prompt_texts)):
                    seq_len = torch.sum(batch_inputs.attention_mask[i]).item()

                    try:
                        # Get hidden states output - this is a tuple where the first element contains hidden states
                        output = cache.output

                        # Extract hidden states based on output type
                        hidden_states = None
                        if isinstance(output, tuple) and len(output) > 0:
                            hidden_states = output[0]  # First element is hidden states
                        elif hasattr(output, 'shape') and len(output.shape) >= 3:
                            hidden_states = output  # Output is directly hidden states

                        if hidden_states is None:
                            raise ValueError("Could not extract hidden states")

                        # Extract the full vector for the last token
                        last_token_hidden = hidden_states[i, seq_len-1, :]  # Keep the hidden dimension

                        # Verify we have the full vector
                        if not last_token_hidden.shape or len(last_token_hidden.shape) == 0:
                            raise ValueError(f"Expected activation shape (2048,), got {last_token_hidden.shape}")

                        # Convert to numpy
                        activation = last_token_hidden.detach().cpu().numpy()
                        activations[layer_num].append(activation)

                    except Exception as e:
                        print(f"Error extracting activation for layer {layer_num}, prompt {i}: {str(e)}")
                        # Skip this extraction rather than crashing
                        continue

    return activations

def build_conversation_history(conversation: Dict, up_to_turn: int) -> str:
    """
    Build conversation history up to a specific turn.
    """
    history_parts = []
    turn_keys = sorted([int(k) for k in conversation.keys()])
    
    for i in range(min(up_to_turn + 1, len(turn_keys))):
        turn_key = str(turn_keys[i])
        turn_data = conversation.get(turn_key, {})
        
        if isinstance(turn_data, dict):
            prompt = turn_data.get("prompt", "")
            response = turn_data.get("response", "")
            
            if prompt:
                history_parts.append(f"User: {prompt}")
            if response and i < up_to_turn:  # Don't include current turn's response
                history_parts.append(f"Assistant: {response}")
    
    return "\n".join(history_parts)

def compute_activation_features(activations: Dict, logger=None) -> np.ndarray:
    """
    Compute feature representations from multi-turn activations.
    Ensures consistent output dimensions even for first turns.
    """
    features = []
    
    # Basic activations
    if 'current_turn' in activations:
        current_turn = activations['current_turn']
        # Debug print the actual shape we're receiving
        if logger:
            logger.debug(f"Raw current_turn activation shape: {current_turn.shape}")
        features.append(current_turn)
    else:
        if logger:
            logger.warning("No current_turn activation available!")
        return np.array([])  # No features if current_turn missing
    
    if 'full_history' in activations:
        full_history = activations['full_history']
        if logger:
            logger.debug(f"Raw full_history activation shape: {full_history.shape}")
        features.append(full_history)
    else:
        if logger:
            logger.warning("No full_history activation available!")
        return np.array([])  # No features if full_history missing
    
    # Add padding for missing features to ensure consistent dimensions
    if 'previous_turn' in activations:
        prev_turn = activations['previous_turn']
        if logger:
            logger.debug(f"Raw previous_turn activation shape: {prev_turn.shape}")
        # Make sure shapes match before subtraction
        if prev_turn.shape == current_turn.shape:
            turn_diff = current_turn - prev_turn
            features.append(turn_diff)
        else:
            # If shapes don't match, reshape one to match the other
            if logger:
                logger.warning(f"Shape mismatch: current_turn {current_turn.shape} vs previous_turn {prev_turn.shape}")
            # Reshape previous_turn to match current_turn if possible
            reshaped_prev = prev_turn.reshape(current_turn.shape)
            turn_diff = current_turn - reshaped_prev
            features.append(turn_diff)
    else:
        # Add zero padding for turn difference
        features.append(np.zeros_like(current_turn))
        if logger:
            logger.debug(f"Added ZERO PADDING for turn-to-turn difference (shape: {np.zeros_like(current_turn).shape})")
    
    if 'previous_history' in activations and 'full_history' in activations:
        prev_history = activations['previous_history']
        if logger:
            logger.debug(f"Raw previous_history activation shape: {prev_history.shape}")
        # Make sure shapes match before subtraction
        if prev_history.shape == full_history.shape:
            history_diff = full_history - prev_history
            features.append(history_diff)
        else:
            # If shapes don't match, reshape one to match the other
            if logger:
                logger.warning(f"Shape mismatch: full_history {full_history.shape} vs previous_history {prev_history.shape}")
            # Reshape previous_history to match full_history if possible
            reshaped_prev_history = prev_history.reshape(full_history.shape)
            history_diff = full_history - reshaped_prev_history
            features.append(history_diff)
    else:
        # Add zero padding for history evolution
        features.append(np.zeros_like(current_turn))
        if logger:
            logger.debug(f"Added ZERO PADDING for history evolution (shape: {np.zeros_like(current_turn).shape})")
    
    # Ensure each feature has the correct dimensions - first fully flatten them all
    flattened_features = []
    for i, f in enumerate(features):
        # Handle different input shapes
        if len(f.shape) == 1:  # If (2048,)
            flattened = f.reshape(1, -1)
        elif len(f.shape) == 2:  # If (1, 2048)
            flattened = f
        elif len(f.shape) == 3:  # If (1, 1, 2048)
            flattened = f.reshape(1, -1)
        else:
            if logger:
                logger.warning(f"Unexpected shape for feature {i}: {f.shape}, attempting to flatten")
            flattened = f.reshape(1, -1)

        flattened_features.append(flattened)
        if logger:
            logger.debug(f"Feature {i} flattened shape: {flattened.shape}")

    # Check all features have the same dimension for hstack
    feature_dims = [f.shape[1] for f in flattened_features]
    if len(set(feature_dims)) > 1:
        if logger:
            logger.warning(f"Feature dimensions mismatch: {feature_dims}, using zero padding")
        # Find max dimension
        max_dim = max(feature_dims)
        # Pad all features to max dimension
        for i in range(len(flattened_features)):
            if flattened_features[i].shape[1] < max_dim:
                padding = np.zeros((1, max_dim - flattened_features[i].shape[1]))
                flattened_features[i] = np.hstack([flattened_features[i], padding])

    # Then concatenate horizontally
    result = np.hstack(flattened_features)

    if logger:
        logger.debug(f"Final concatenated feature vector shape: {result.shape}")
    
    return result

def process_conversations_optimized_batched(conversations: List[Dict],
                                          model,
                                          tokenizer,
                                          device: str,
                                          layers: List[int],
                                          output_dir: Path,
                                          batch_size: int = 8,
                                          use_layer_averaging: bool = False,
                                          avg_layer_name: str = 'layer_avg',
                                          has_jailbreak_scores: bool = False,
                                          strategic_layers: Dict[str, int] = None,
                                          memory_clear_freq: int = 50,
                                          logger=None) -> Dict:
    """
    Process conversations and store activations efficiently in an HDF5 file.
    Enhanced with batch processing for better GPU utilization.
    
    Args:
        conversations: List of conversation dictionaries
        model, tokenizer: The model and tokenizer to use
        device: The device to run inference on
        layers: List of ALL layer numbers to extract
        output_dir: Output directory
        batch_size: Number of prompts to process in a single batch
        use_layer_averaging: Whether to compute and store layer averages
        avg_layer_name: Name for the averaged layer
        has_jailbreak_scores: Whether the dataset includes jailbreak scores
        strategic_layers: Dict of strategic layer indices for averaging
        memory_clear_freq: How often to clear CUDA cache (in batches)
        logger: Optional logger for debug output
        
    Returns:
        Dictionary with conversation metadata
    """
    # Use main logger if none provided
    if logger is None:
        logger = logging.getLogger('multi_turn_lps')
        
    # Get component logger for extraction details
    extraction_logger = logging.getLogger('multi_turn_lps.extraction')
    
    logger.info(f"Processing {len(conversations)} conversations, extracting {len(layers)} layers")
    logger.info(f"Using batch size {batch_size} with memory clear frequency {memory_clear_freq}")

    # Create HDF5 file for storing all activations
    activations_file = output_dir / "activations.h5"
    metadata = []
    
    # If no strategic layers provided, use first, mid, last from the layers list
    if strategic_layers is None:
        total_layers = max(layers) + 1 if layers else 24
        strategic_layers = get_strategic_layers(total_layers)
    
    # Get the strategic layer indices that we'll actually use for averaging
    strategic_layer_indices = [idx for idx in strategic_layers.values() if idx in layers]
    logger.info(f"Using strategic layers for averaging: {strategic_layer_indices}")
    
    # Summary counts for logging
    successful_extractions = 0
    failed_extractions = 0
    harmful_count = 0
    harmless_count = 0
    
    with h5py.File(activations_file, 'w') as f:
        # Create groups for harmful and harmless conversations
        harmful_group = f.create_group("harmful")
        harmless_group = f.create_group("harmless")
        
        # Create progress bar
        pbar = tqdm(range(0, len(conversations), batch_size), desc="Processing batches", total=(len(conversations) + batch_size - 1) // batch_size)
        
        # Group conversations into batches for efficient processing
        for batch_start in pbar:  # Use the progress bar iterator directly
            batch_end = min(batch_start + batch_size, len(conversations))
            batch_conversations = conversations[batch_start:batch_end]
            actual_batch_size = len(batch_conversations)
            batch_meta = []

            # First pass: Create metadata and HDF5 groups for each conversation
            for batch_idx, conversation_data in enumerate(batch_conversations):
                conv_idx = batch_start + batch_idx

                # Extract from structure
                conversation = conversation_data.get("conversation", {})
                is_harmful = conversation_data.get("is_harmful", False)
                source = conversation_data.get("source", "unknown")
                original_length = conversation_data.get("length", len(conversation))

                # Update counters
                if is_harmful:
                    harmful_count += 1
                else:
                    harmless_count += 1

                # Choose the appropriate group
                group = harmful_group if is_harmful else harmless_group

                # Create a group for this conversation
                conv_group = group.create_group(f"conv_{conv_idx}")

                # Store conversation metadata as attributes
                conv_group.attrs["source"] = source
                conv_group.attrs["original_length"] = original_length

                # Get the actual turn keys and sort them
                turn_keys = sorted([int(k) for k in conversation.keys()])
                turn_count = len(turn_keys)

                # Create metadata
                conv_meta = {
                    "conversation_id": conv_idx,
                    "is_harmful": is_harmful,
                    "source": source,
                    "original_length": original_length,
                    "turn_count": turn_count,
                    "activations": {},
                    "jailbreak_scores": {}
                }

                batch_meta.append((conv_meta, conversation, turn_keys, conv_group))
            
            # Second pass: Process each turn for all conversations in the batch
            max_turns = max(len(meta[2]) for meta in batch_meta) if batch_meta else 0
            
            # Create turn groups first (just once per turn)
            turn_groups = {}  # Dictionary to store turn groups: {(batch_idx, turn_id): turn_group}
            for turn_id in range(max_turns):
                for batch_idx, (conv_meta, conversation, turn_keys, conv_group) in enumerate(batch_meta):
                    if turn_id >= len(turn_keys):
                        continue

                    current_turn_key = turn_keys[turn_id]

                    # Create turn group only once
                    try:
                        if f"turn_{turn_id}" not in conv_group:
                            turn_group = conv_group.create_group(f"turn_{turn_id}")
                        else:
                            turn_group = conv_group[f"turn_{turn_id}"]

                        # Store the turn group reference for later use
                        turn_groups[(batch_idx, turn_id)] = turn_group

                        # Get jailbreak score for this turn
                        key_str = str(current_turn_key)
                        jailbreak_score = 0.0
                        if has_jailbreak_scores and key_str in conversation:
                            turn_data = conversation.get(key_str, {})
                            if isinstance(turn_data, dict):
                                jailbreak_score = turn_data.get("jailbreak_score", 0.0)
                                # Handle -1.0 (failed parsing) by setting to 0.0
                                if jailbreak_score == -1.0:
                                    jailbreak_score = 0.0

                        conv_meta["jailbreak_scores"][str(turn_id)] = jailbreak_score
                        turn_group.attrs["jailbreak_score"] = jailbreak_score
                        turn_group.attrs["original_turn_key"] = current_turn_key
                    except Exception as e:
                        extraction_logger.error(f"Error creating turn group: {e}")

            for turn_id in range(max_turns):
                # For each layer, process all applicable conversations in a batch
                for layer_num in layers:
                    # Collect all prompts for this turn across all conversations in the batch
                    turn_data_by_type = {
                        'current_turn': [],
                        'previous_turn': [],
                        'full_history': [],
                        'previous_history': []
                    }

                    # Map to track which conversation each prompt belongs to
                    conv_mapping = {
                        'current_turn': [],
                        'previous_turn': [],
                        'full_history': [],
                        'previous_history': []
                    }

                    # Prepare all prompts for this turn across the batch
                    for batch_idx, (conv_meta, conversation, turn_keys, conv_group) in enumerate(batch_meta):
                        if turn_id >= len(turn_keys) or (batch_idx, turn_id) not in turn_groups:
                            continue

                        current_turn_key = turn_keys[turn_id]

                        # Create turn group
                        turn_group = turn_groups[(batch_idx, turn_id)]

                        # Get jailbreak score for this turn
                        key_str = str(current_turn_key)
                        jailbreak_score = 0.0
                        if has_jailbreak_scores and key_str in conversation:
                            turn_data = conversation.get(key_str, {})
                            if isinstance(turn_data, dict):
                                jailbreak_score = turn_data.get("jailbreak_score", 0.0)
                                # Handle -1.0 (failed parsing) by setting to 0.0
                                if jailbreak_score == -1.0:
                                    jailbreak_score = 0.0

                        conv_meta["jailbreak_scores"][str(turn_id)] = jailbreak_score
                        turn_group.attrs["jailbreak_score"] = jailbreak_score
                        turn_group.attrs["original_turn_key"] = current_turn_key

                        # Prepare 1: Current turn prompt
                        current_turn_data = conversation.get(str(current_turn_key), {})
                        current_prompt = current_turn_data.get("prompt", "")
                        if current_prompt:
                            turn_data_by_type['current_turn'].append(current_prompt)
                            conv_mapping['current_turn'].append((batch_idx, turn_group))

                        # Prepare 2: Previous turn prompt (if exists)
                        if turn_id > 0:
                            prev_turn_key = turn_keys[turn_id - 1]
                            prev_turn_data = conversation.get(str(prev_turn_key), {})
                            prev_prompt = prev_turn_data.get("prompt", "")
                            if prev_prompt:
                                turn_data_by_type['previous_turn'].append(prev_prompt)
                                conv_mapping['previous_turn'].append((batch_idx, turn_group))

                        # Prepare 3: Full conversation history
                        full_history = build_conversation_history(conversation, turn_id)
                        if full_history:
                            turn_data_by_type['full_history'].append(full_history)
                            conv_mapping['full_history'].append((batch_idx, turn_group))

                        # Prepare 4: Previous conversation history
                        if turn_id > 0:
                            prev_history = build_conversation_history(conversation, turn_id - 1)
                            if prev_history:
                                turn_data_by_type['previous_history'].append(prev_history)
                                conv_mapping['previous_history'].append((batch_idx, turn_group))

                    # Process each type of prompt in batch mode
                    for activation_type, prompts in turn_data_by_type.items():
                        if not prompts:  # Skip if no prompts of this type
                            continue

                        try:
                            # Extract activations for this batch of prompts
                            activations = extract_multi_layer_activations_batched(
                                model, tokenizer, prompts, device, [layer_num]
                            )

                            # Store the activations for each conversation
                            for idx, (batch_idx, turn_group) in enumerate(conv_mapping[activation_type]):
                                # Get the conversation metadata
                                conv_meta = batch_meta[batch_idx][0]

                                # Store activation
                                activation = activations[layer_num][idx]
                                # Ensure we're storing the full vector with proper dimensions
                                if len(activation.shape) == 1:  # If (2048,)
                                    activation_to_store = activation.reshape(1, -1)
                                else:
                                    activation_to_store = activation

                                dataset_name = f"layer_{layer_num}_{activation_type}"
                                turn_group.create_dataset(dataset_name, data=activation_to_store)
                                
                                # Store path in metadata
                                is_harmful = conv_meta["is_harmful"]
                                conv_idx = conv_meta["conversation_id"]
                                key = f"turn_{turn_id}_layer_{layer_num}_{activation_type}"
                                path = f"{'harmful' if is_harmful else 'harmless'}/conv_{conv_idx}/turn_{turn_id}/{dataset_name}"
                                conv_meta["activations"][key] = path

                                # Also store basic activation for backward compatibility
                                if activation_type == 'full_history':
                                    basic_dataset_name = f"layer_{layer_num}"
                                    turn_group.create_dataset(basic_dataset_name, data=activation.reshape(1, -1))

                                    # Store basic path in metadata
                                    basic_key = f"turn_{turn_id}_layer_{layer_num}"
                                    basic_path = f"{'harmful' if is_harmful else 'harmless'}/conv_{conv_idx}/turn_{turn_id}/{basic_dataset_name}"
                                    conv_meta["activations"][basic_key] = basic_path

                            successful_extractions += len(conv_mapping[activation_type])

                        except Exception as e:
                            error_msg = f"Error processing layer {layer_num}, turn {turn_id}, type {activation_type}: {str(e)}"
                            extraction_logger.error(error_msg)
                            failed_extractions += len(conv_mapping[activation_type])
                
                # Update progress bar
                pbar.set_description(f"Processing: {batch_start}/{len(conversations)} convs ({harmful_count}/{harmless_count} H/H)")
            
            # Add conversation metadata
            for conv_meta, _, _, _ in batch_meta:
                metadata.append(conv_meta)
            
            # Strategic memory clearing
            if (batch_start // batch_size) % memory_clear_freq == 0:
                torch.cuda.empty_cache()
                extraction_logger.debug(f"Cleared GPU memory cache after batch {batch_start // batch_size}")

    # Compute and store layer averages if requested
    if use_layer_averaging and strategic_layer_indices:
        logger.info(f"Computing layer averages using strategic layers: {strategic_layer_indices}")

        with h5py.File(activations_file, 'r+') as f:
            for harm_status in ['harmful', 'harmless']:
                group = f[harm_status]
                conv_groups = list(group.keys())

                for conv_id in conv_groups:
                    conv_group = group[conv_id]
                    turn_groups = list(conv_group.keys())

                    for turn_id in turn_groups:
                        turn_group = conv_group[turn_id]

                        # Average across strategic layers for each activation type
                        for activation_type in ['current_turn', 'previous_turn', 'full_history', 'previous_history', '']:
                            suffix = f"_{activation_type}" if activation_type else ""

                            # Collect activations from strategic layers
                            layer_activations = []
                            for layer_idx in strategic_layer_indices:
                                dataset_name = f"layer_{layer_idx}{suffix}"
                                if dataset_name in turn_group:
                                    layer_activations.append(turn_group[dataset_name][:])

                            # Compute average if we have activations
                            if layer_activations:
                                avg_activation = np.mean(layer_activations, axis=0)
                                avg_dataset_name = f"{avg_layer_name}{suffix}"

                                # Create the dataset
                                if avg_dataset_name in turn_group:
                                    del turn_group[avg_dataset_name]  # Replace if exists
                                turn_group.create_dataset(avg_dataset_name, data=avg_activation)

                                # Update metadata
                                if activation_type:  # Skip empty suffix (backward compatibility)
                                    conv_idx = int(conv_id.split('_')[1])
                                    turn_num = int(turn_id.split('_')[1])
                                    for meta in metadata:
                                        if meta['conversation_id'] == conv_idx:
                                            meta_key = f"turn_{turn_num}_{avg_layer_name}_{activation_type}"
                                            meta_path = f"{harm_status}/{conv_id}/{turn_id}/{avg_dataset_name}"
                                            meta['activations'][meta_key] = meta_path
    
    # Log summary statistics
    logger.info(f"Extraction complete: {successful_extractions} successful, {failed_extractions} failed")
    logger.info(f"Processed {harmful_count} harmful and {harmless_count} harmless conversations")
    
    # Save metadata separately for easier access
    with open(output_dir / "conversations_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Activations stored in {activations_file}")
    logger.info(f"Metadata saved to {output_dir / 'conversations_metadata.json'}")
    
    return {"metadata": metadata, "activations_file": str(activations_file)}

def analyze_activation_jailbreak_correlation(layer_data: Dict, output_dir: Path) -> Dict:
    """
    Analyze correlation between activation features and jailbreak scores.
    This helps determine if we should use all features or just some subset.
    """
    correlation_results = {}
    
    for layer_num, data_tuple in layer_data.items():
        if len(data_tuple) != 6:  # Skip if no jailbreak scores
            continue
            
        X_train, X_test, y_train, y_test, jailbreak_scores_train, jailbreak_scores_test = data_tuple
        
        # Combine train and test for correlation analysis
        X_combined = np.vstack([X_train, X_test])
        jailbreak_scores_combined = np.hstack([jailbreak_scores_train, jailbreak_scores_test])
        y_combined = np.hstack([y_train, y_test])
        
        print(f"Analyzing correlations for layer {layer_num}...")
        
        # 1. Correlation between jailbreak scores and harmful labels
        jailbreak_label_corr = np.corrcoef(jailbreak_scores_combined, y_combined)[0, 1]
        
        # 2. Correlation between activation features and jailbreak scores
        # Sample features to avoid memory issues (use every 10th feature)
        sample_indices = np.arange(0, X_combined.shape[1], max(1, X_combined.shape[1] // 100))
        X_sampled = X_combined[:, sample_indices]
        
        feature_jailbreak_corrs = []
        for i in range(X_sampled.shape[1]):
            try:
                corr = np.corrcoef(X_sampled[:, i], jailbreak_scores_combined)[0, 1]
                if not np.isnan(corr):
                    feature_jailbreak_corrs.append(abs(corr))
            except:
                continue
        
        # 3. Correlation between activation features and harmful labels
        feature_label_corrs = []
        for i in range(X_sampled.shape[1]):
            try:
                corr = np.corrcoef(X_sampled[:, i], y_combined)[0, 1]
                if not np.isnan(corr):
                    feature_label_corrs.append(abs(corr))
            except:
                continue
        
        # 4. PCA analysis to see feature redundancy
        try:
            from sklearn.decomposition import PCA
            pca = PCA()
            pca.fit(X_sampled)
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Find how many components explain 95% of variance
            n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        except:
            explained_variance_ratio = []
            n_components_95 = X_sampled.shape[1]
        
        # Store results
        layer_results = {
            "jailbreak_label_correlation": float(jailbreak_label_corr),
            "mean_feature_jailbreak_correlation": float(np.mean(feature_jailbreak_corrs)) if feature_jailbreak_corrs else 0.0,
            "max_feature_jailbreak_correlation": float(np.max(feature_jailbreak_corrs)) if feature_jailbreak_corrs else 0.0,
            "mean_feature_label_correlation": float(np.mean(feature_label_corrs)) if feature_label_corrs else 0.0,
            "max_feature_label_correlation": float(np.max(feature_label_corrs)) if feature_label_corrs else 0.0,
            "n_components_95_variance": int(n_components_95),
            "original_feature_count": int(X_combined.shape[1]),
            "feature_redundancy_ratio": float(n_components_95 / X_combined.shape[1]) if X_combined.shape[1] > 0 else 1.0,
            "jailbreak_score_stats": {
                "mean": float(jailbreak_scores_combined.mean()),
                "std": float(jailbreak_scores_combined.std()),
                "min": float(jailbreak_scores_combined.min()),
                "max": float(jailbreak_scores_combined.max()),
                "nonzero_ratio": float(np.mean(jailbreak_scores_combined > 0))
            }
        }
        
        correlation_results[str(layer_num)] = layer_results
        
        # Create visualizations
        layer_dir = output_dir / f"layer_{layer_num}"
        layer_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot 1: Correlation matrix heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = np.array([
            [1.0, jailbreak_label_corr, layer_results["max_feature_label_correlation"]],
            [jailbreak_label_corr, 1.0, layer_results["max_feature_jailbreak_correlation"]],
            [layer_results["max_feature_label_correlation"], layer_results["max_feature_jailbreak_correlation"], 1.0]
        ])
        
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    fmt='.3f',
                    cmap='coolwarm',
                    center=0,
                    xticklabels=['Harmful Labels', 'Jailbreak Scores', 'Best Activation Feature'],
                    yticklabels=['Harmful Labels', 'Jailbreak Scores', 'Best Activation Feature'])
        plt.title(f'Layer {layer_num}: Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(layer_dir / "correlation_matrix.png")
        plt.close()
        
        # Plot 2: Jailbreak score distribution by harmful/harmless
        plt.figure(figsize=(10, 6))
        harmful_scores = jailbreak_scores_combined[y_combined == 1]
        harmless_scores = jailbreak_scores_combined[y_combined == 0]
        
        plt.hist(harmless_scores, bins=30, alpha=0.7, label='Harmless', density=True)
        plt.hist(harmful_scores, bins=30, alpha=0.7, label='Harmful', density=True)
        plt.xlabel('Jailbreak Score')
        plt.ylabel('Density')
        plt.title(f'Layer {layer_num}: Jailbreak Score Distribution')
        plt.legend()
        plt.axvline(harmful_scores.mean(), color='red', linestyle='--', alpha=0.7, label=f'Harmful Mean: {harmful_scores.mean():.3f}')
        plt.axvline(harmless_scores.mean(), color='blue', linestyle='--', alpha=0.7, label=f'Harmless Mean: {harmless_scores.mean():.3f}')
        plt.tight_layout()
        plt.savefig(layer_dir / "jailbreak_score_distribution.png")
        plt.close()
        
        # Plot 3: PCA explained variance
        if len(explained_variance_ratio) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, min(51, len(explained_variance_ratio) + 1)), 
                    explained_variance_ratio[:50], 'bo-', markersize=4)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title(f'Layer {layer_num}: PCA Explained Variance (First 50 Components)')
            plt.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
            plt.axvline(n_components_95, color='red', linestyle='--', alpha=0.7, label=f'95% at Component {n_components_95}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(layer_dir / "pca_explained_variance.png")
            plt.close()
        
        print(f"Layer {layer_num} correlation analysis:")
        print(f"  Jailbreak-Label correlation: {jailbreak_label_corr:.4f}")
        print(f"  Max Feature-Jailbreak correlation: {layer_results['max_feature_jailbreak_correlation']:.4f}")
        print(f"  Feature redundancy: {layer_results['feature_redundancy_ratio']:.2%}")
        print(f"  Nonzero jailbreak scores: {layer_results['jailbreak_score_stats']['nonzero_ratio']:.2%}")
    
    # Save comprehensive correlation analysis
    with open(output_dir / "correlation_analysis.json", "w") as f:
        json.dump(correlation_results, f, indent=4)
    
    # Create summary comparison plot
    if correlation_results:
        create_correlation_summary_plot(correlation_results, output_dir)
    
    return correlation_results

def create_correlation_summary_plot(correlation_results: Dict, output_dir: Path):
    """Create summary plots comparing correlation across layers."""
    layers = sorted([int(k) for k in correlation_results.keys() if k.isdigit()])
    
    if not layers:
        return
    
    # Extract metrics for comparison
    jailbreak_label_corrs = [correlation_results[str(l)]["jailbreak_label_correlation"] for l in layers]
    max_feature_jailbreak_corrs = [correlation_results[str(l)]["max_feature_jailbreak_correlation"] for l in layers]
    max_feature_label_corrs = [correlation_results[str(l)]["max_feature_label_correlation"] for l in layers]
    redundancy_ratios = [correlation_results[str(l)]["feature_redundancy_ratio"] for l in layers]
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Jailbreak-Label correlations by layer
    ax1.plot(layers, jailbreak_label_corrs, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Layer Number')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('Jailbreak Score vs. Harmful Label Correlation by Layer')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Best feature correlations
    ax2.plot(layers, max_feature_jailbreak_corrs, 'go-', linewidth=2, markersize=6, label='Feature-Jailbreak')
    ax2.plot(layers, max_feature_label_corrs, 'ro-', linewidth=2, markersize=6, label='Feature-Label')
    ax2.set_xlabel('Layer Number')
    ax2.set_ylabel('Max Correlation Coefficient')
    ax2.set_title('Best Activation Feature Correlations by Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature redundancy
    ax3.plot(layers, redundancy_ratios, 'mo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Layer Number')
    ax3.set_ylabel('Redundancy Ratio (Components/Features)')
    ax3.set_title('Feature Redundancy by Layer (Lower = More Redundant)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='High Redundancy Threshold')
    ax3.legend()
    
    # Plot 4: Recommendation summary
    ax4.axis('off')
    
    # Generate recommendations based on analysis
    recommendations = []
    avg_jailbreak_corr = np.mean([abs(c) for c in jailbreak_label_corrs])
    avg_redundancy = np.mean(redundancy_ratios)
    
    if avg_jailbreak_corr > 0.7:
        recommendations.append("• Jailbreak scores are highly correlated with labels")
        recommendations.append("• Consider using jailbreak scores alone for initial experiments")
    elif avg_jailbreak_corr > 0.3:
        recommendations.append("• Jailbreak scores show moderate correlation with labels")
        recommendations.append("• Combine with activation features for best performance")
    else:
        recommendations.append("• Jailbreak scores show weak correlation with labels")
        recommendations.append("• Focus primarily on activation features")
    
    if avg_redundancy < 0.2:
        recommendations.append("• High feature redundancy detected")
        recommendations.append("• Use PCA or feature selection to reduce dimensionality")
    elif avg_redundancy < 0.5:
        recommendations.append("• Moderate feature redundancy")
        recommendations.append("• Consider light dimensionality reduction")
    else:
        recommendations.append("• Low feature redundancy")
        recommendations.append("• Can use features without major reduction")
    
    recommendation_text = "Recommendations:\n\n" + "\n".join(recommendations)
    ax4.text(0.1, 0.9, recommendation_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_summary.png")
    plt.close()

def determine_global_feature_strategy(correlation_results: Dict) -> Dict:
    """
    Determine ONE optimal feature strategy to use across ALL layers.
    This ensures fair comparison between layers.
    """
    if not correlation_results:
        return {
            "primary_features": "combined",
            "dimensionality_reduction": "feature_selection",
            "reasoning": ["Default strategy - no correlation analysis available"]
        }
    
    # Aggregate metrics across all layers
    all_jailbreak_label_corrs = []
    all_feature_jailbreak_corrs = []
    all_feature_label_corrs = []
    all_redundancy_ratios = []
    all_nonzero_ratios = []
    
    for layer_num, layer_results in correlation_results.items():
        if not layer_num.isdigit():
            continue
            
        all_jailbreak_label_corrs.append(abs(layer_results["jailbreak_label_correlation"]))
        all_feature_jailbreak_corrs.append(layer_results["max_feature_jailbreak_correlation"])
        all_feature_label_corrs.append(layer_results["max_feature_label_correlation"])
        all_redundancy_ratios.append(layer_results["feature_redundancy_ratio"])
        all_nonzero_ratios.append(layer_results["jailbreak_score_stats"]["nonzero_ratio"])
    
    # Calculate global averages
    avg_jailbreak_label_corr = np.mean(all_jailbreak_label_corrs)
    avg_feature_jailbreak_corr = np.mean(all_feature_jailbreak_corrs)
    avg_feature_label_corr = np.mean(all_feature_label_corrs)
    avg_redundancy = np.mean(all_redundancy_ratios)
    avg_nonzero_ratio = np.mean(all_nonzero_ratios)
    
    # Global decision logic
    strategy = {
        "primary_features": "combined",
        "dimensionality_reduction": "none",
        "reasoning": []
    }
    
    """# Decision rules based on GLOBAL statistics
    if avg_jailbreak_label_corr > 0.75:
        strategy["primary_features"] = "jailbreak_only"
        strategy["reasoning"].append(f"Jailbreak scores highly correlated globally (avg: {avg_jailbreak_label_corr:.3f})")
        
    elif avg_jailbreak_label_corr < 0.2 and avg_nonzero_ratio < 0.1:
        strategy["primary_features"] = "activations_only"
        strategy["reasoning"].append(f"Jailbreak scores weakly correlated globally (avg: {avg_jailbreak_label_corr:.3f}) and sparse (avg: {avg_nonzero_ratio:.1%})")
        
    elif avg_feature_label_corr > avg_jailbreak_label_corr * 2:
        strategy["primary_features"] = "activations_only"
        strategy["reasoning"].append(f"Activation features globally outperform jailbreak scores ({avg_feature_label_corr:.3f} vs {avg_jailbreak_label_corr:.3f})")
        
    else:
        strategy["reasoning"].append(f"Both feature types show moderate global correlation (jailbreak: {avg_jailbreak_label_corr:.3f}, activations: {avg_feature_label_corr:.3f})")
    
    # Dimensionality reduction decision
    if avg_redundancy < 0.2:
        strategy["dimensionality_reduction"] = "pca"
        strategy["reasoning"].append(f"High global feature redundancy (avg: {avg_redundancy:.2%})")
    elif avg_redundancy < 0.5:
        strategy["dimensionality_reduction"] = "feature_selection"
        strategy["reasoning"].append(f"Moderate global feature redundancy (avg: {avg_redundancy:.2%})")
    """
    strategy["primary_features"] = "combined"
    strategy["dimensionality_reduction"] = "pca"
    strategy["reasoning"].append("Default strategy - using combined features with PCA for dimensionality reduction")
    return strategy

def prepare_features_adaptively(X_train: np.ndarray, X_test: np.ndarray, 
                               jailbreak_scores_train: np.ndarray, jailbreak_scores_test: np.ndarray,
                               strategy: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features based on the determined strategy.
    """
    if strategy["primary_features"] == "jailbreak_only":
        # Use only jailbreak scores
        X_train_final = jailbreak_scores_train.reshape(-1, 1)
        X_test_final = jailbreak_scores_test.reshape(-1, 1)
        
    elif strategy["primary_features"] == "activations_only":
        # Use only activation features
        X_train_final = X_train
        X_test_final = X_test
        
    else:  # combined
        # Combine both feature types
        X_train_final = np.hstack([X_train, jailbreak_scores_train.reshape(-1, 1)])
        X_test_final = np.hstack([X_test, jailbreak_scores_test.reshape(-1, 1)])
    
    return X_train_final, X_test_final

def create_adaptive_pipeline(strategy: Dict, n_samples: int, n_features: int) -> Pipeline:
    """
    Create a machine learning pipeline based on the feature strategy.
    """
    steps = [('scaler', StandardScaler())]
    
    # Add dimensionality reduction if needed
    if strategy["dimensionality_reduction"] == "pca":
        # Use PCA to reduce to 95% of variance or max 100 components
        n_components = min(100, n_samples // 10, n_features)
        steps.append(('pca', PCA(n_components=n_components)))
        
    elif strategy["dimensionality_reduction"] == "feature_selection":
        # Use feature selection to keep top features
        k = min(50, n_samples // 20, n_features)
        steps.append(('feature_selection', SelectKBest(f_classif, k=k)))
    
    # Add classifier
    steps.append(('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')))
    
    return Pipeline(steps)

def prepare_data_for_probing_optimized(layers: List[int],
                                      metadata: List[Dict],
                                      activations_file: str,
                                      use_layer_averaging: bool = False,
                                      avg_layer_name: str = 'layer_avg',
                                      feature_strategy: Dict = None,
                                      logger=None) -> Dict[Union[int, str], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Prepare activation data for probe training by reading from the HDF5 file.
    Now supports different feature strategies using multi-turn activations.
    
    Args:
        layers: List of layer numbers to process
        metadata: List of conversation metadata dictionaries
        activations_file: Path to HDF5 file with activations
        use_layer_averaging: Whether to include the averaged layer
        avg_layer_name: Name for the averaged layer
        feature_strategy: Strategy dict with primary_features specification
        logger: Optional logger for debug output
    
    Returns:
        Dictionary mapping layer identifiers to tuples of (X_train, X_test, y_train, y_test, jailbreak_scores_train, jailbreak_scores_test)
    """
    # Use main logger if none provided
    if logger is None:
        logger = logging.getLogger('multi_turn_lps')
        
    # Get component logger for data preparation details
    data_prep_logger = logging.getLogger('multi_turn_lps.data_prep')
    
    layer_data = {}

    # Debug: Print a sample of metadata to help diagnose issues
    if metadata and len(metadata) > 0:
        sample_meta = metadata[0]
        logger.info(f"Sample metadata: keys={list(sample_meta.keys())}")
        if "activations" in sample_meta:
            logger.info(f"Sample activation keys: {list(sample_meta['activations'].keys())[:3]}...")

    # Create stratified test set
    test_conv_ids = create_stratified_test_set(metadata, test_percentage=0.4)
    logger.info(f"Split data: {len(test_conv_ids)} conversations in test set")
    
    # Mark which conversations are in test set in metadata
    for meta in metadata:
        conv_id = meta.get("conversation_id") or meta.get("id")
        meta['is_test_set'] = meta['conversation_id'] in test_conv_ids
    

    use_multi_turn = True
    logger.info("Using multi-turn activation features")
    
    summary_stats = {
        'total_harmful_samples': 0,
        'total_harmless_samples': 0,
        'samples_by_layer': {},
        'failed_loads': 0,
        'missing_features': 0
    }
    
    with h5py.File(activations_file, 'r') as f:
        # Debug: Print HDF5 structure to help diagnose issues
        logger.info(f"HDF5 keys at root level: {list(f.keys())}")
        if list(f.keys()):
            sample_layer = list(f.keys())[0]
            logger.info(f"Sample layer {sample_layer} keys: {list(f[sample_layer].keys())[:3]}...")

        # Process each regular layer
        for layer_num in layers:
            logger.info(f"Preparing data for layer {layer_num}...")
            data_prep_logger.debug(f"Starting data preparation for layer {layer_num}")
            
            harmful_activations = []
            harmless_activations = []
            harmful_conv_ids = []
            harmless_conv_ids = []
            jailbreak_scores = []
            
            summary_stats['samples_by_layer'][layer_num] = {
                'harmful': 0,
                'harmless': 0,
                'multi_turn_success': 0,
                'multi_turn_missing': 0,
                'basic_fallback': 0
            }
            
            # Collect activations for this layer
            for conv_meta in tqdm(metadata, desc=f"Processing layer {layer_num}"):
                is_harmful = conv_meta["is_harmful"]
                conv_id = conv_meta["conversation_id"]
                
                # For each turn, get the activation for this layer
                for turn_id in range(conv_meta.get("turn_count", 5)):
                    
                    if use_multi_turn:
                        # Try to load multi-turn features
                        activation_keys = [
                            f"turn_{turn_id}_layer_{layer_num}_current_turn",
                            f"turn_{turn_id}_layer_{layer_num}_full_history",
                            f"turn_{turn_id}_layer_{layer_num}_previous_turn",
                            f"turn_{turn_id}_layer_{layer_num}_previous_history"
                        ]
                        
                        multi_activations = {}
                        for key_suffix, activation_type in [
                            ("_current_turn", "current_turn"),
                            ("_full_history", "full_history"), 
                            ("_previous_turn", "previous_turn"),
                            ("_previous_history", "previous_history")
                        ]:
                            # Special case for layer_avg
                            if str(layer_num) == 'layer_avg':
                                key = f"turn_{turn_id}_{layer_num}{key_suffix}"
                            else:
                                key = f"turn_{turn_id}_layer_{layer_num}{key_suffix}"

                            if key in conv_meta.get("activations", {}):
                                path = conv_meta["activations"][key]
                                try:
                                    activation = f[path][()]
                                    multi_activations[activation_type] = activation
                                except Exception as e:
                                    data_prep_logger.debug(f"Error loading {key} for conv {conv_id}: {e}")
                                    summary_stats['failed_loads'] += 1
                                    continue
                        
                        if multi_activations:
                            # logger.info(f"Multi-activations found: {list(multi_activations.keys())}")
                            # logger.info(f"Activation shapes: {[a.shape for a in multi_activations.values()]}")
                            combined_features = compute_activation_features(multi_activations)
                            # logger.info(f"Combined feature shape: {combined_features.shape}")
                            if combined_features.size > 0:
                                flat_activation = combined_features.reshape(1, -1)
                                summary_stats['samples_by_layer'][layer_num]['multi_turn_success'] += 1
                            else:
                                summary_stats['samples_by_layer'][layer_num]['multi_turn_missing'] += 1
                                summary_stats['missing_features'] += 1
                                continue  # Skip if no valid features
                        else:
                            data_prep_logger.debug(f"No multi-turn data for conv {conv_id}, turn {turn_id}")
                            summary_stats['samples_by_layer'][layer_num]['multi_turn_missing'] += 1
                            continue  # Skip if no multi-turn data available
                    else:
                        # Use basic activation (backwards compatibility)
                        key = f"turn_{turn_id}_layer_{layer_num}"
                        if key not in conv_meta.get("activations", {}):
                            continue
                            
                        path = conv_meta["activations"][key]
                        
                        try:
                            # Extract activation from HDF5
                            activation = f[path][()]
                            flat_activation = activation.reshape(1, -1)
                            summary_stats['samples_by_layer'][layer_num]['basic_fallback'] += 1
                        except Exception as e:
                            data_prep_logger.debug(f"Error loading activation for conv {conv_id}, turn {turn_id}, layer {layer_num}: {e}")
                            summary_stats['failed_loads'] += 1
                            continue
                    
                    # Get jailbreak score
                    try:
                        turn_group = f[os.path.dirname(path)]
                        jailbreak_score = turn_group.attrs.get("jailbreak_score", 0.0)
                    except:
                        jailbreak_score = 0.0
                    
                    # Store based on harmfulness
                    if is_harmful:
                        harmful_activations.append(flat_activation)
                        harmful_conv_ids.append(conv_id)
                        summary_stats['samples_by_layer'][layer_num]['harmful'] += 1
                        summary_stats['total_harmful_samples'] += 1
                    else:
                        harmless_activations.append(flat_activation)
                        harmless_conv_ids.append(conv_id)
                        summary_stats['samples_by_layer'][layer_num]['harmless'] += 1
                        summary_stats['total_harmless_samples'] += 1
                        
                    # Store jailbreak score
                    jailbreak_scores.append(jailbreak_score)
            
            # Stack all activations
            if harmful_activations and harmless_activations:
                X_harmful = np.vstack(harmful_activations)
                X_harmless = np.vstack(harmless_activations)
                
                # Create labels
                y_harmful = np.ones(X_harmful.shape[0])
                y_harmless = np.zeros(X_harmless.shape[0])
                
                # Combine data
                X = np.vstack([X_harmful, X_harmless])
                y = np.hstack([y_harmful, y_harmless])
                conv_ids = np.hstack([harmful_conv_ids, harmless_conv_ids])
                jailbreak_scores = np.array(jailbreak_scores)
                
                # Get indices for train and test based on conversation IDs
                train_indices = [i for i, cid in enumerate(conv_ids) if cid not in test_conv_ids]
                test_indices = [i for i, cid in enumerate(conv_ids) if cid in test_conv_ids]
                
                X_train = X[train_indices]
                y_train = y[train_indices]
                jailbreak_scores_train = jailbreak_scores[train_indices]
                
                X_test = X[test_indices]
                y_test = y[test_indices]
                jailbreak_scores_test = jailbreak_scores[test_indices]
                
                # Store data for this layer - now including jailbreak scores
                layer_data[layer_num] = (X_train, X_test, y_train, y_test, jailbreak_scores_train, jailbreak_scores_test)
                
                feature_type = "multi-turn" if use_multi_turn else "basic"
                logger.info(f"Layer {layer_num} ({feature_type}): {X_harmful.shape[0]} harmful samples, "
                          f"{X_harmless.shape[0]} harmless samples")
                logger.info(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
                logger.info(f"  Feature dimensions: {X_train.shape[1]}")
                
                # Detailed log for debug purposes
                data_prep_logger.debug(f"Layer {layer_num} data shapes:")
                data_prep_logger.debug(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
                data_prep_logger.debug(f"  y_train: {y_train.shape}, y_test: {y_test.shape}")
                data_prep_logger.debug(f"  jailbreak_scores_train: {jailbreak_scores_train.shape}, jailbreak_scores_test: {jailbreak_scores_test.shape}")
    
    # Log summary statistics
    logger.info("Data preparation summary:")
    logger.info(f"  Total harmful samples: {summary_stats['total_harmful_samples']}")
    logger.info(f"  Total harmless samples: {summary_stats['total_harmless_samples']}")
    logger.info(f"  Failed loads: {summary_stats['failed_loads']}")
    logger.info(f"  Missing features: {summary_stats['missing_features']}")
    
    return layer_data

def train_linear_probes_with_baselines(layer_data: Dict, output_dir: Path, 
                                      correlation_results: Dict = None,
                                      logger=None) -> Dict:
    """
    Train linear probes with a GLOBAL strategy and include dummy baselines.
    """
    # Use main logger if none provided
    if logger is None:
        logger = logging.getLogger('multi_turn_lps')
        
    # Get component logger for training details
    train_logger = logging.getLogger('multi_turn_lps.training')
    
    # Determine global strategy
    global_strategy = determine_global_feature_strategy(correlation_results)
    
    logger.info(f"Global feature strategy: {global_strategy['primary_features']}")
    logger.info(f"Dimensionality reduction: {global_strategy['dimensionality_reduction']}")
    logger.info(f"Reasoning: {'; '.join(global_strategy['reasoning'])}")
    
    # Save global strategy
    with open(output_dir / "global_feature_strategy.json", "w") as f:
        json.dump(global_strategy, f, indent=4)
    
    probes = {}
    
    # First, train dummy baseline (same for all layers)
    logger.info("Training dummy baseline...")
    if layer_data:
        # Use first layer's data for dummy baseline
        first_layer = list(layer_data.keys())[0]
        first_data = layer_data[first_layer]
        
        if len(first_data) >= 4:
            _, _, y_train, y_test = first_data[:4]
            
            # Train different dummy strategies
            dummy_strategies = ['most_frequent', 'stratified', 'uniform']
            dummy_results = {}
            
            for strategy in dummy_strategies:
                train_logger.debug(f"Training dummy classifier with strategy: {strategy}")
                dummy_clf = DummyClassifier(strategy=strategy, random_state=42)
                dummy_clf.fit(np.zeros((len(y_train), 1)), y_train)  # Dummy features
                y_pred_dummy = dummy_clf.predict(np.zeros((len(y_test), 1)))
                
                dummy_results[strategy] = {
                    'accuracy': accuracy_score(y_test, y_pred_dummy),
                    'precision': precision_score(y_test, y_pred_dummy, zero_division=0),
                    'recall': recall_score(y_test, y_pred_dummy, zero_division=0),
                    'f1': f1_score(y_test, y_pred_dummy, zero_division=0)
                }
            
            # Save dummy results
            with open(output_dir / "dummy_baseline_results.json", "w") as f:
                json.dump(dummy_results, f, indent=4)
            
            probes['dummy_baseline'] = dummy_results
            
            logger.info("Dummy baseline results:")
            for strategy, metrics in dummy_results.items():
                logger.info(f"  {strategy}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    # Now train actual probes with GLOBAL strategy
    for layer_num, data_tuple in layer_data.items():
        logger.info(f"Training probe for layer {layer_num} with global strategy...")
        layer_dir = output_dir / f"layer_{layer_num}"
        layer_dir.mkdir(exist_ok=True, parents=True)
        
        # Apply global strategy consistently
        if len(data_tuple) == 6:  # With jailbreak scores
            X_train, X_test, y_train, y_test, jailbreak_scores_train, jailbreak_scores_test = data_tuple
            
            # Apply global feature strategy
            X_train_final, X_test_final = prepare_features_adaptively(
                X_train, X_test, jailbreak_scores_train, jailbreak_scores_test, global_strategy
            )
            
            train_logger.debug(f"Layer {layer_num}: Applied global strategy ({global_strategy['primary_features']})")
            train_logger.debug(f"  Input shapes: X_train={X_train.shape}, X_test={X_test.shape}")
            train_logger.debug(f"  Final shapes: X_train_final={X_train_final.shape}, X_test_final={X_test_final.shape}")
            
        else:  # Without jailbreak scores
            X_train, X_test, y_train, y_test = data_tuple
            X_train_final, X_test_final = X_train, X_test
            
            train_logger.debug(f"Layer {layer_num}: No jailbreak scores available")
            train_logger.debug(f"  Shapes: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # Create pipeline with global strategy
        pipeline = create_adaptive_pipeline(
            global_strategy, 
            n_samples=X_train_final.shape[0], 
            n_features=X_train_final.shape[1]
        )
        
        # Same parameter grid for all layers (fair comparison)
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0, 100.0]
        }
        
        if global_strategy["dimensionality_reduction"] == "pca":
            param_grid['pca__n_components'] = [50, 100] if X_train_final.shape[1] > 100 else [min(50, X_train_final.shape[1])]
        elif global_strategy["dimensionality_reduction"] == "feature_selection":
            max_k = min(X_train_final.shape[1], X_train_final.shape[0] // 10)
            param_grid['feature_selection__k'] = [max_k // 2, max_k] if max_k > 10 else [max_k]
        
        train_logger.debug(f"Layer {layer_num}: Grid search parameters: {param_grid}")
        
        # Train model
        logger.info(f"Layer {layer_num}: Starting grid search CV (this may take a while)...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=1)
        grid_search.fit(X_train_final, y_train)
        best_model = grid_search.best_estimator_
        
        train_logger.debug(f"Layer {layer_num}: Best parameters: {grid_search.best_params_}")
        
        # Evaluate
        y_pred = best_model.predict(X_test_final)
        y_proba = best_model.predict_proba(X_test_final)[:, 1]
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = 0.0
            
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc,
            'best_params': str(grid_search.best_params_),
            'global_strategy': global_strategy,
            'final_feature_count': X_train_final.shape[1]
        }
        
        # Save results
        with open(layer_dir / "model_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        probe_data = {
            'model': best_model,
            'model_metrics': metrics,
            'global_strategy': global_strategy
        }
        
        probes[layer_num] = probe_data
        
        logger.info(f"Layer {layer_num} - Features: {X_train_final.shape[1]}, F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")
    
    return probes

def analyze_features(probes: Dict[int, Dict[str, Any]], 
                    output_dir: Path):
    """
    Analyze which features contribute most to classification decisions.
    
    Args:
        probes: Dictionary of trained probes
        output_dir: Output directory for saving visualizations
    """
    for layer_num, probe_data in probes.items():
        model = probe_data.get('model')
        layer_dir = output_dir / f"layer_{layer_num}"
        
        # Extract the feature importance information
        try:
            # For LogisticRegression, coefficients represent feature importance
            pipeline_steps = {step[0]: step[1] for step in model.steps}
            classifier = pipeline_steps['classifier']
            
            if hasattr(classifier, 'coef_'):
                # Get the coefficients
                coefficients = classifier.coef_[0]
                
                # Get the indices of the selected features if using SelectKBest
                if 'feature_selection' in pipeline_steps:
                    selected_indices = pipeline_steps['feature_selection'].get_support(indices=True)
                    
                    # Create feature names based on indices
                    feature_names = [f"Feature_{i}" for i in selected_indices]
                else:
                    # If no feature selection, use all features
                    feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
                
                # Get absolute values for importance ranking
                importance = np.abs(coefficients)
                
                # Sort features by importance
                indices = np.argsort(importance)[::-1]
                
                # Plot top 20 features
                top_n = min(20, len(indices))
                top_indices = indices[:top_n]
                top_importance = importance[top_indices]
                top_names = [feature_names[i] for i in top_indices]
                top_coefficients = coefficients[top_indices]
                
                # Create a color map based on coefficient sign
                colors = ['red' if c < 0 else 'blue' for c in top_coefficients]
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(top_n), top_importance, color=colors)
                plt.yticks(range(top_n), top_names)
                plt.xlabel('Feature Importance (absolute coefficient value)')
                plt.ylabel('Feature')
                plt.title(f'Top {top_n} Important Features for Layer {layer_num}')
                
                # Add a legend for coefficient sign
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='blue', label='Positive coefficient (favors harmful)'),
                    Patch(facecolor='red', label='Negative coefficient (favors harmless)')
                ]
                plt.legend(handles=legend_elements)
                
                plt.tight_layout()
                plt.savefig(layer_dir / "feature_importance.png")
                plt.close()
                
                # Save feature importance to CSV
                feature_data = {
                    'Feature': feature_names,
                    'Coefficient': coefficients,
                    'Importance': importance
                }
                pd.DataFrame(feature_data).sort_values('Importance', ascending=False).to_csv(
                    layer_dir / "feature_importance.csv", index=False
                )
                
                print(f"Feature importance analysis completed for layer {layer_num}")
                
            else:
                print(f"Model for layer {layer_num} doesn't have coefficient attributes for feature importance")
        
        except Exception as e:
            print(f"Error analyzing features for layer {layer_num}: {e}")

def classify_conversations_with_voting(metadata: List[Dict], 
                                     probes: Dict[int, Dict[str, Any]],
                                     activations_file: str,
                                     config: Dict) -> List[Dict]:
    """
    Classify conversations using the trained probes with a voting system.
    Now supports multi-turn features based on the global strategy.
    """
    # Extract configuration parameters
    turn_threshold = config.get('turn_threshold', 0.8)
    min_turns_per_layer = config.get('min_turns_per_layer', 1)
    min_layers_for_harmful = config.get('min_layers_for_harmful', 2)
    avg_layer_name = config.get('avg_layer_name', 'layer_avg')
    
    # Determine if we should use multi-turn features
    use_multi_turn = True
    global_strategy = None
    
    # Get global strategy from any probe
    for layer_num, probe_data in probes.items():
        if layer_num != 'dummy_baseline' and 'model_metrics' in probe_data:
            global_strategy = probe_data['model_metrics'].get('global_strategy')
            break
    
    print(f"Classification using {'multi-turn' if use_multi_turn else 'basic'} features")
    print(f"Classification parameters: turn_threshold={turn_threshold}, "
          f"min_turns_per_layer={min_turns_per_layer}, "
          f"min_layers_for_harmful={min_layers_for_harmful}")
    
    classification_results = []
    
    with h5py.File(activations_file, 'r') as f:
        for conv_meta in tqdm(metadata, desc="Classifying conversations"):
            conv_id = conv_meta["conversation_id"]
            is_harmful = conv_meta["is_harmful"]
            conv_turn_count = conv_meta.get("turn_count", 5)
            
            conv_classification = {
                "conversation_id": conv_id,
                "ground_truth_harmful": is_harmful,
                "is_test_set": conv_meta.get("is_test_set", False),
                "layer_votes": {},
                "turn_probabilities": {},
                "conversation_length": conv_turn_count
            }
            
            # Track layers voting harmful PER TURN
            layers_voting_harmful_by_turn = {turn_id: set() for turn_id in range(conv_turn_count)}
            
            # Process each layer's probe separately
            for layer_num, probe_data in probes.items():
                if layer_num == 'dummy_baseline':
                    continue
                    
                model = probe_data.get('model')
                harmful_turns = []
                turn_probabilities = {}
                
                # Check if this probe model uses jailbreak scores
                global_strategy = probe_data.get('model_metrics', {}).get('global_strategy', {})
                uses_jailbreak = global_strategy.get('primary_features') in ['jailbreak_only', 'combined']
                
                # Check each turn
                for turn_id in range(conv_turn_count):
                    activation_data = None
                    
                    if use_multi_turn:
                        # Load multi-turn activations
                        multi_activations = {}
                        
                        for activation_type in ["current_turn", "full_history", "previous_turn", "previous_history"]:
                            if isinstance(layer_num, str) and layer_num == avg_layer_name:
                                key = f"turn_{turn_id}_{layer_num}_{activation_type}"
                            else:
                                key = f"turn_{turn_id}_layer_{layer_num}_{activation_type}"
                            
                            if key in conv_meta.get("activations", {}):
                                path = conv_meta["activations"][key]
                                try:
                                    activation = f[path][()]
                                    multi_activations[activation_type] = activation
                                except Exception as e:
                                    continue
                        
                        if multi_activations:
                            # Compute combined features
                            combined_features = compute_activation_features(multi_activations)
                            if combined_features.size > 0:
                                activation_data = combined_features.reshape(1, -1)
                    else:
                        # Load basic activation
                        if isinstance(layer_num, str) and layer_num == avg_layer_name:
                            key = f"turn_{turn_id}_{layer_num}"
                        else:
                            key = f"turn_{turn_id}_layer_{layer_num}"
                        
                        if key in conv_meta.get("activations", {}):
                            path = conv_meta["activations"][key]
                            try:
                                activation = f[path][()]
                                activation_data = activation.reshape(1, -1)
                            except Exception as e:
                                continue
                    
                    if activation_data is not None:
                        try:
                            # Add jailbreak score if model needs it
                            if uses_jailbreak:
                                turn_group = f[os.path.dirname(path)]
                                jailbreak_score = turn_group.attrs.get("jailbreak_score", 0.0)
                                activation_data = np.hstack([activation_data, np.array([[jailbreak_score]])])
                            
                            # Use the model to predict
                            if model:
                                probability = model.predict_proba(activation_data)[0][1]
                                turn_probabilities[str(turn_id)] = probability
                                
                                # Check if turn is harmful based on threshold
                                if probability > turn_threshold:
                                    harmful_turns.append(turn_id)
                                    
                        except Exception as e:
                            print(f"Error classifying conv {conv_id}, turn {turn_id}, layer {layer_num}: {e}")
                
                # Store results for this layer
                layer_key = str(layer_num)
                conv_classification["turn_probabilities"][layer_key] = turn_probabilities
                
                is_layer_harmful = len(harmful_turns) >= min_turns_per_layer
                
                conv_classification["layer_votes"][layer_key] = {
                    "harmful_turns": harmful_turns,
                    "vote": is_layer_harmful,
                    "first_detection_turn": min(harmful_turns) if harmful_turns else None,
                }
                
                # Update per-turn tracking
                if is_layer_harmful:
                    first_harmful_turn = min(harmful_turns) if harmful_turns else None
                    if first_harmful_turn is not None:
                        for turn in range(first_harmful_turn, conv_turn_count):
                            layers_voting_harmful_by_turn[turn].add(layer_key)
            
            # Determine earliest detection
            earliest_detection_turn = None
            for turn_id in range(conv_turn_count):
                if len(layers_voting_harmful_by_turn[turn_id]) >= min_layers_for_harmful:
                    earliest_detection_turn = turn_id
                    break
            
            # Final conversation classification
            harmful_layer_count = len(layers_voting_harmful_by_turn[conv_turn_count-1])
            
            conv_classification["harmful_layer_votes"] = harmful_layer_count
            conv_classification["conversation_detected_harmful"] = harmful_layer_count >= min_layers_for_harmful
            conv_classification["earliest_detection_turn"] = earliest_detection_turn
            
            classification_results.append(conv_classification)
    
    return classification_results

def calculate_metrics_for_subset(results: List[Dict]) -> Dict:
    """Helper function to calculate metrics for a subset of results"""
    if not results:
        return {
            "total_conversations": 0,
            "true_harmful_count": 0,
            "detected_harmful_count": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "avg_detection_turn": None,
            "detection_turn_distribution": {turn: 0 for turn in range(5)},
        }
    
    total_conversations = len(results)
    
    # Extract ground truth and predictions
    y_true = [r["ground_truth_harmful"] for r in results]
    y_pred = [r["conversation_detected_harmful"] for r in results]
    
    # We need probabilities for AUC-ROC and Precision@k
    # Extract probabilities by averaging the layer probabilities for each conversation
    probabilities = []
    for r in results:
        # Get avg probability across all layers that voted
        layer_probas = []
        for layer_num, layer_vote in r.get("layer_votes", {}).items():
            # Check if we have turn probabilities for this layer
            if layer_num in r.get("turn_probabilities", {}):
                # Average the probabilities across all turns
                turn_probas = [float(prob) for prob in r["turn_probabilities"][layer_num].values()]
                if turn_probas:
                    layer_probas.append(np.mean(turn_probas))
        
        # Average across all layers
        avg_proba = np.mean(layer_probas) if layer_probas else 0.5
        probabilities.append(avg_proba)

    # Calculate AUC-ROC if we have enough data points
    roc_auc = 0.0
    try:
        if len(set(y_true)) > 1:  # Need both classes to calculate AUC
            roc_auc = roc_auc_score(y_true, probabilities)
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC: {e}")

    # Calculate Precision@k for conversation-level predictions
    k_values = [5, 10, 20]
    precision_at_ks = {}
    for k in k_values:
        if len(y_true) >= k:
            precision_at_ks[f"precision_at_{k}"] = precision_at_k(y_true, probabilities, k)


    # Calculate classification metrics with zero_division=0
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0) 
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Get detection statistics
    harmful_conversations = sum(1 for r in results if r["conversation_detected_harmful"])
    true_harmful = sum(1 for r in results if r["ground_truth_harmful"])
    
    # Find earliest detection turns for true positives
    detection_turns = [
        r["earliest_detection_turn"]
        for r in results
        if r["ground_truth_harmful"] and r["conversation_detected_harmful"] and r["earliest_detection_turn"] is not None
    ]
    
    # Calculate detection distribution by turn
    detection_distribution = {turn: 0 for turn in range(5)}
    for turn in detection_turns:
        if turn is not None and 0 <= turn < 5:
            detection_distribution[turn] += 1
    
    # Compile metrics
    return {
        "total_conversations": total_conversations,
        "true_harmful_count": true_harmful,
        "detected_harmful_count": harmful_conversations,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        **precision_at_ks,  # Add all precision@k values
        "confusion_matrix": conf_matrix.tolist(),
        "avg_detection_turn": np.mean(detection_turns) if detection_turns else None,
        "detection_turn_distribution": detection_distribution,
    }

def calculate_metrics_by_length(results: List[Dict]) -> Dict:
    """
    Calculate accuracy metrics grouped by conversation length.
    """
    # Group results by conversation length
    by_length = {}
    for r in results:
        length = r.get("conversation_length", 5)  # Default to 5 if not specified
        if length not in by_length:
            by_length[length] = []
        by_length[length].append(r)
    
    # Calculate metrics for each length
    length_metrics = {}
    for length, length_results in by_length.items():
        if not length_results:
            continue
            
        y_true = [r["ground_truth_harmful"] for r in length_results]
        y_pred = [r["conversation_detected_harmful"] for r in length_results]
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        length_metrics[str(length)] = {
            "conversation_count": len(length_results),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix.tolist()
        }
    
    return length_metrics

def calculate_metrics_optimized(classification_results: List[Dict]) -> Dict:
    """
    Calculate metrics from classification results with ground truth comparison,
    separating metrics for training and test sets.
    
    Returns:
        Dictionary with metrics for test set, training set, and combined
    """
# Separate test and train results
    test_results = [r for r in classification_results if r.get('is_test_set', False)]
    train_results = [r for r in classification_results if not r.get('is_test_set', False)]
    
    print(f"Calculating metrics for {len(test_results)} test conversations and "
          f"{len(train_results)} training conversations")
    
    # Calculate basic metrics for each subset
    test_metrics = calculate_metrics_for_subset(test_results)
    train_metrics = calculate_metrics_for_subset(train_results)
    combined_metrics = calculate_metrics_for_subset(classification_results)
    
    # Calculate metrics by conversation length
    test_length_metrics = calculate_metrics_by_length(test_results)
    train_length_metrics = calculate_metrics_by_length(train_results)
    combined_length_metrics = calculate_metrics_by_length(classification_results)
    
    # Add length metrics to their respective metric sets
    test_metrics["metrics_by_length"] = test_length_metrics
    train_metrics["metrics_by_length"] = train_length_metrics
    combined_metrics["metrics_by_length"] = combined_length_metrics
    
    # Initialize layer metrics dictionaries
    test_layer_metrics = {}
    train_layer_metrics = {}
    combined_layer_metrics = {}
    
    # Calculate layer metrics, separate by train/test
    for r in classification_results:
        # Determine which metrics dict to use
        is_test = r.get('is_test_set', False)
        target_metrics = test_layer_metrics if is_test else train_layer_metrics
        
        # Process each layer's vote
        for layer_num, layer_vote in r.get("layer_votes", {}).items():
            if layer_num not in combined_layer_metrics:
                combined_layer_metrics[layer_num] = {"true_pos": 0, "false_pos": 0, "true_neg": 0, "false_neg": 0}
                test_layer_metrics[layer_num] = {"true_pos": 0, "false_pos": 0, "true_neg": 0, "false_neg": 0}
                train_layer_metrics[layer_num] = {"true_pos": 0, "false_pos": 0, "true_neg": 0, "false_neg": 0}
            
            # Get vote (harmful or not)
            vote = layer_vote.get("vote", False)
            
            # Update metrics
            if r["ground_truth_harmful"] and vote:
                combined_layer_metrics[layer_num]["true_pos"] += 1
                target_metrics[layer_num]["true_pos"] += 1
            elif not r["ground_truth_harmful"] and vote:
                combined_layer_metrics[layer_num]["false_pos"] += 1
                target_metrics[layer_num]["false_pos"] += 1
            elif not r["ground_truth_harmful"] and not vote:
                combined_layer_metrics[layer_num]["true_neg"] += 1
                target_metrics[layer_num]["true_neg"] += 1
            else:  # r["ground_truth_harmful"] and not vote
                combined_layer_metrics[layer_num]["false_neg"] += 1
                target_metrics[layer_num]["false_neg"] += 1
    
    # Calculate performance metrics for each layer set
    for metrics_dict in [combined_layer_metrics, test_layer_metrics, train_layer_metrics]:
        for layer, counts in metrics_dict.items():
            total = sum(counts.values())
            tp = counts["true_pos"]
            fp = counts["false_pos"]
            tn = counts["true_neg"]
            fn = counts["false_neg"]
            
            metrics_dict[layer]["accuracy"] = (tp + tn) / total if total > 0 else 0
            metrics_dict[layer]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics_dict[layer]["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics_dict[layer]["f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # Add layer metrics to their respective metric sets
    test_metrics["layer_metrics"] = test_layer_metrics
    train_metrics["layer_metrics"] = train_layer_metrics
    combined_metrics["layer_metrics"] = combined_layer_metrics
    
    return {
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "combined_metrics": combined_metrics
    }

def analyze_jailbreak_relationship(layer_data: Dict, probes: Dict, output_dir: Path):
    """
    Analyze the relationship between jailbreak scores and harmful content detection.
    """
    for layer_num, probe_data in probes.items():
        if layer_num not in layer_data:
            continue
            
        # Get data - handle both with and without jailbreak scores
        data_tuple = layer_data[layer_num]
        if len(data_tuple) == 6:  # Format with jailbreak scores
            _, X_test, _, y_test, _, jailbreak_scores_test = data_tuple
            
            model = probe_data['model']
            # For models trained with jailbreak scores, use X_test with jailbreak scores
            X_test_with_scores = np.hstack([X_test, jailbreak_scores_test.reshape(-1, 1)])
            
            # Get predictions
            y_pred = model.predict(X_test_with_scores)
            y_proba = model.predict_proba(X_test_with_scores)[:, 1]
            
            layer_dir = output_dir / f"layer_{layer_num}"
            
            # Plot jailbreak score vs. prediction probability
            plt.figure(figsize=(10, 6))
            plt.scatter(jailbreak_scores_test, y_proba, alpha=0.5, c=y_test, cmap='coolwarm')
            plt.xlabel('Jailbreak Score')
            plt.ylabel('Predicted Probability of Harmful Content')
            plt.title(f'Layer {layer_num}: Jailbreak Score vs. Prediction Probability')
            plt.colorbar(label='Actual Class (Harmful=1, Harmless=0)')
            plt.grid(alpha=0.3)
            plt.savefig(layer_dir / "jailbreak_vs_prediction.png")
            plt.close()
            
            # Calculate correlation
            correlation = np.corrcoef(jailbreak_scores_test, y_proba)[0, 1]
            print(f"Layer {layer_num}: Correlation between jailbreak scores and predicted probabilities: {correlation:.4f}")
            
            # Save correlation data
            with open(layer_dir / "jailbreak_correlation.json", "w") as f:
                json.dump({
                    "correlation": float(correlation),
                    "mean_jailbreak_score_harmful": float(jailbreak_scores_test[y_test == 1].mean()),
                    "mean_jailbreak_score_harmless": float(jailbreak_scores_test[y_test == 0].mean()),
                    "mean_prediction_high_jailbreak": float(y_proba[jailbreak_scores_test > 0.5].mean()),
                    "mean_prediction_low_jailbreak": float(y_proba[jailbreak_scores_test <= 0.5].mean()),
                }, f, indent=4)

def plot_detection_turn_distribution(metrics: Dict, output_dir: Path):
    """Plot the distribution of turns when harmful content was first detected."""
    plt.figure(figsize=(10, 6))
    turns = list(metrics["detection_turn_distribution"].keys())
    counts = list(metrics["detection_turn_distribution"].values())
    
    plt.bar(turns, counts)
    plt.xlabel("Turn Number")
    plt.ylabel("Number of Harmful Conversations Detected")
    plt.title("Turn at which Harmful Content was First Detected")
    plt.xticks(turns)
    plt.savefig(output_dir / "detection_turns.png")
    plt.close()

def plot_confusion_matrix(metrics: Dict, output_dir: Path):
    """Plot confusion matrix for conversation classification."""
    plt.figure(figsize=(8, 6))
    conf_matrix = np.array(metrics["confusion_matrix"])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Harmful', 'Harmful'],
                yticklabels=['Not Harmful', 'Harmful'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Conversation Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

def plot_layer_performance(metrics: Dict, output_dir: Path):
    """Plot performance metrics by layer."""
    if "layer_metrics" in metrics and metrics["layer_metrics"]:
        plt.figure(figsize=(12, 8))
        layers = list(metrics["layer_metrics"].keys())
        accuracies = [metrics["layer_metrics"][l]["accuracy"] for l in layers]
        precisions = [metrics["layer_metrics"][l]["precision"] for l in layers]
        recalls = [metrics["layer_metrics"][l]["recall"] for l in layers]
        f1s = [metrics["layer_metrics"][l]["f1"] for l in layers]
        
        width = 0.2
        x = np.arange(len(layers))
        
        plt.bar(x - 1.5*width, accuracies, width, label='Accuracy')
        plt.bar(x - 0.5*width, precisions, width, label='Precision')
        plt.bar(x + 0.5*width, recalls, width, label='Recall')
        plt.bar(x + 1.5*width, f1s, width, label='F1')
        
        plt.xlabel('Layer')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Layer (Test Set Only)')
        plt.xticks(x, layers)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "layer_performance.png")
        plt.close()

def plot_summary_metrics(metrics: Dict, output_dir: Path):
    """Plot summary metrics as a text figure."""
    roc_auc_text = f"ROC AUC: {metrics['roc_auc']:.4f}\n" if 'roc_auc' in metrics else ""
    precision_at_k_text = ""
    for k, val in metrics.items():
        if k.startswith('precision_at_') and isinstance(val, (int, float)):
            precision_at_k_text += f"{k}: {val:.4f}\n"

    plt.figure(figsize=(12, 6))
    plt.text(0.5, 0.5, 
            f"Total Conversations: {metrics['total_conversations']}\n"
            f"True Harmful: {metrics['true_harmful_count']}\n" 
            f"Detected Harmful: {metrics['detected_harmful_count']}\n"
            f"Accuracy: {metrics['accuracy']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall: {metrics['recall']:.4f}\n"
            f"F1 Score: {metrics['f1']:.4f}\n"
            f"{roc_auc_text}"
            f"{precision_at_k_text}"
            f"Average Detection Turn: {metrics['avg_detection_turn']:.2f}" if metrics['avg_detection_turn'] is not None else "Average Detection Turn: N/A",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=14)
    plt.axis('off')
    plt.savefig(output_dir / "summary_metrics.png")
    plt.close()


def plot_metrics_by_length(metrics: Dict, output_dir: Path):
    """Plot metrics broken down by conversation length."""
    if "metrics_by_length" not in metrics or not metrics["metrics_by_length"]:
        print("No metrics by length data available for plotting")
        return
    
    lengths = sorted(metrics["metrics_by_length"].keys(), key=lambda x: int(x))
    
    # Extract metrics for each length
    accuracies = [metrics["metrics_by_length"][length]["accuracy"] for length in lengths]
    precisions = [metrics["metrics_by_length"][length]["precision"] for length in lengths]
    recalls = [metrics["metrics_by_length"][length]["recall"] for length in lengths]
    f1s = [metrics["metrics_by_length"][length]["f1"] for length in lengths]
    counts = [metrics["metrics_by_length"][length]["conversation_count"] for length in lengths]
    
    # Create figure with two subplots - one for metrics, one for counts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    # Convert lengths to integers for x-axis
    x = [int(l) for l in lengths]
    
    # Plot metrics
    ax1.plot(x, accuracies, 'o-', label='Accuracy', linewidth=2)
    ax1.plot(x, precisions, 's-', label='Precision', linewidth=2)
    ax1.plot(x, recalls, '^-', label='Recall', linewidth=2)
    ax1.plot(x, f1s, 'd-', label='F1 Score', linewidth=2)
    
    ax1.set_xlabel('Conversation Length (turns)')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics by Conversation Length')
    ax1.set_xticks(x)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot counts as a bar chart
    ax2.bar(x, counts)
    ax2.set_xlabel('Conversation Length (turns)')
    ax2.set_ylabel('Count')
    ax2.set_title('Conversation Count by Length')
    ax2.set_xticks(x)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_by_length.png")
    plt.close()
    
    # Create a heatmap for confusion matrices by length
    ncols = 3  # Up to 3 confusion matrices per row
    nrows = (len(lengths) + ncols - 1) // ncols
    
    plt.figure(figsize=(15, 5 * nrows))
    
    for i, length in enumerate(lengths):
        plt.subplot(nrows, ncols, i+1)
        conf_matrix = np.array(metrics["metrics_by_length"][length]["confusion_matrix"])
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Harmful', 'Harmful'],
                    yticklabels=['Not Harmful', 'Harmful'])
        plt.title(f'Length {length} turns (n={counts[i]})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices_by_length.png")
    plt.close()

def plot_detection_by_turn_and_length(results: List[Dict], output_dir: Path):
    """
    Create a visualization showing when harmful conversations are detected
    based on their total length, including undetected conversations.
    """
    # Group by conversation length
    by_length = defaultdict(lambda: {"detected": [], "undetected": 0})
    
    # Process all harmful conversations
    for r in results:
        if r["ground_truth_harmful"]:  # Consider all harmful conversations
            length = r.get("conversation_length", 5)
            
            if r["conversation_detected_harmful"]:
                # Track detected harmful conversations
                detection_turn = r.get("earliest_detection_turn")
                if detection_turn is not None:
                    by_length[length]["detected"].append(detection_turn)
            else:
                # Count undetected harmful conversations
                by_length[length]["undetected"] += 1
    
    # Prepare data for plotting
    length_data = {}
    for length, data in by_length.items():
        detections = data["detected"]
        undetected = data["undetected"]
        
        # Count detections by turn
        turn_counts = Counter(detections)
        
        # Fill in zeros for missing turns
        all_turns = {i: turn_counts.get(i, 0) for i in range(length)}
        
        # Add undetected count
        length_data[length] = {"turns": all_turns, "undetected": undetected}
    
    # Only create plot if we have data
    if length_data:
        # Create a stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        lengths = sorted(length_data.keys())
        x = np.arange(len(lengths))
        width = 0.5
        
        # Create a color map for turns
        colors = plt.cm.viridis(np.linspace(0, 1, 6))  # 6 possible turns
        
        # Distinct color for undetected (red)
        undetected_color = 'red'
        
        bottom = np.zeros(len(lengths))
        
        # First stack all detected conversations by turn
        for turn in range(6):  # 0 to 5
            values = [length_data[length]["turns"].get(turn, 0) if turn < length else 0 for length in lengths]
            if any(values):  # Only plot if there's at least one non-zero value
                ax.bar(x, values, width, bottom=bottom, label=f'Turn {turn+1}', color=colors[turn])
                bottom += values
        
        # Then add undetected conversations at the top
        undetected_values = [length_data[length]["undetected"] for length in lengths]
        if any(undetected_values):
            ax.bar(x, undetected_values, width, bottom=bottom, label='Undetected', color=undetected_color)
        
        ax.set_ylabel('Number of Harmful Conversations')
        ax.set_title('Detection Turn by Conversation Length')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{l} turns' for l in lengths])
        ax.set_xlabel('Conversation Length')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(output_dir / "detection_by_length.png")
        plt.close()

def plot_detection_lead_time(results: List[Dict], output_dir: Path):
    """
    Create a visualization showing lead-time (N - detection_turn)
    for harmful conversations of varying lengths.
    """
    # Group by lead-time, track conversation length for stacking
    lead_time_data = defaultdict(lambda: defaultdict(int))  # {lead_time: {length: count}}
    undetected_counts = defaultdict(int)

    for r in results:
        if r["ground_truth_harmful"]:
            length = r.get("conversation_length", 5)

            if r["conversation_detected_harmful"]:
                detection_turn = r.get("earliest_detection_turn")
                if detection_turn is not None:
                    lead_time = length - detection_turn
                    lead_time_data[lead_time][length] += 1
            else:
                undetected_counts[length] += 1

    # Normalize lead-time bins to include all possibilities
    max_length = max([r.get("conversation_length", 5) for r in results], default=5)
    lead_times = list(range(1, max_length + 1))  # 1 (detected at last turn) up to N-1
    lead_times.append("Undetected")  # special bin

    # Prepare stacked data
    lengths = sorted(set([r.get("conversation_length", 5) for r in results]))
    stacked_values = {length: [] for length in lengths}

    for lt in lead_times:
        for length in lengths:
            if lt == "Undetected":
                count = undetected_counts[length]
            else:
                count = lead_time_data[lt].get(length, 0)
            stacked_values[length].append(count)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(lead_times))
    width = 0.5

    color_map = plt.cm.viridis(np.linspace(0, 1, len(lengths)))
    bottom = np.zeros(len(lead_times))

    for idx, length in enumerate(lengths):
        values = stacked_values[length]
        ax.bar(x, values, width, bottom=bottom, label=f'{length} turns', color=color_map[idx])
        bottom += values

    ax.set_ylabel('Number of Harmful Conversations')
    ax.set_title('Lead-Time Detection by Conversation Length')
    ax.set_xticks(x)
    ax.set_xticklabels([str(lt) if lt != "Undetected" else "Undetected" for lt in lead_times])
    ax.set_xlabel('Lead-Time (Turns Remaining Before Harm)')
    ax.legend(title="Conversation Length", loc='upper left', bbox_to_anchor=(1, 1))
    ax.text(0.02, 0.98, 
            f"Lead-time = Conversation Length - Detection Turn\n"
            f"Higher values = Earlier detection = Better anticipation", 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    plt.tight_layout()
    plt.savefig(output_dir / "lead_time_detection.png")
    plt.close()


def create_strategy_summary_plot(probes: Dict, output_dir: Path):
    """Create a summary visualization of the global feature strategy used."""
    if not probes or 'dummy_baseline' not in probes:
        return
    
    # Get global strategy from any probe (they all use the same strategy)
    global_strategy = None
    for layer_num, probe_data in probes.items():
        if layer_num != 'dummy_baseline' and 'model_metrics' in probe_data:
            global_strategy = probe_data['model_metrics'].get('global_strategy')
            break
    
    if not global_strategy:
        print("No global strategy found in probe data")
        return
    
    # Extract information
    primary_features = global_strategy.get('primary_features', 'unknown')
    dimensionality_reduction = global_strategy.get('dimensionality_reduction', 'none')
    reasoning = global_strategy.get('reasoning', [])
    
    # Get performance data
    layers = []
    f1_scores = []
    feature_counts = []
    
    for layer_num, probe_data in probes.items():
        if layer_num != 'dummy_baseline' and 'model_metrics' in probe_data:
            layers.append(str(layer_num))
            f1_scores.append(probe_data['model_metrics']['f1'])
            feature_counts.append(probe_data['model_metrics'].get('final_feature_count', 0))
    
    if not layers:
        return
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Global strategy summary
    ax1.axis('off')
    strategy_text = f"Global Feature Strategy\n\n"
    strategy_text += f"Primary Features: {primary_features}\n"
    strategy_text += f"Dimensionality Reduction: {dimensionality_reduction}\n\n"
    strategy_text += "Reasoning:\n" + "\n".join([f"• {reason}" for reason in reasoning])
    
    ax1.text(0.1, 0.9, strategy_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    ax1.set_title('Feature Strategy Used Across All Layers')
    
    # Plot 2: Performance by layer
    x = np.arange(len(layers))
    bars = ax2.bar(x, f1_scores, color='skyblue')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Performance by Layer (Same Strategy)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Feature count by layer
    ax3.bar(x, feature_counts, color='lightgreen')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Final Feature Count')
    ax3.set_title('Feature Count After Processing')
    ax3.set_xticks(x)
    ax3.set_xticklabels(layers)
    
    # Plot 4: Strategy effectiveness
    ax4.axis('off')
    
    # Calculate effectiveness metrics
    avg_f1 = np.mean(f1_scores)
    max_f1 = np.max(f1_scores)
    min_f1 = np.min(f1_scores)
    
    effectiveness_text = f"Strategy Effectiveness\n\n"
    effectiveness_text += f"Average F1: {avg_f1:.4f}\n"
    effectiveness_text += f"Best Layer F1: {max_f1:.4f}\n"
    effectiveness_text += f"Worst Layer F1: {min_f1:.4f}\n"
    effectiveness_text += f"F1 Range: {max_f1 - min_f1:.4f}\n\n"
    effectiveness_text += f"Features Used: {feature_counts[0] if feature_counts else 'Unknown'}\n"
    effectiveness_text += f"Strategy: {primary_features.replace('_', ' ').title()}"
    
    ax4.text(0.1, 0.9, effectiveness_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    ax4.set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig(output_dir / "global_strategy_summary.png")
    plt.close()

def create_performance_comparison_plot(probes: Dict, output_dir: Path):
    """
    Create comprehensive comparison including dummy baseline.
    """
    # Extract dummy baseline
    dummy_baseline = probes.get('dummy_baseline', {})
    best_dummy_f1 = 0.0
    if dummy_baseline:
        best_dummy_f1 = max(results['f1'] for results in dummy_baseline.values())
    
    # Extract layer results
    layers = []
    f1_scores = []
    accuracies = []
    
    for layer_num, probe_data in probes.items():
        if layer_num != 'dummy_baseline' and 'model_metrics' in probe_data:
            layers.append(str(layer_num))
            f1_scores.append(probe_data['model_metrics']['f1'])
            accuracies.append(probe_data['model_metrics']['accuracy'])
    
    if not layers:
        return
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: F1 scores with dummy baseline
    x = np.arange(len(layers))
    bars1 = ax1.bar(x, f1_scores, color='skyblue', label='Linear Probe')
    ax1.axhline(y=best_dummy_f1, color='red', linestyle='--', linewidth=2, label=f'Best Dummy Baseline ({best_dummy_f1:.3f})')
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score by Layer vs. Dummy Baseline')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Improvement over baseline
    improvements = [f1 - best_dummy_f1 for f1 in f1_scores]
    bars2 = ax2.bar(x, improvements, color=['green' if imp > 0 else 'red' for imp in improvements])
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('F1 Improvement over Baseline')
    ax2.set_title('Performance Improvement vs. Dummy Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                f'{imp:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison_with_baseline.png")
    plt.close()

def plot_results_optimized(metrics: Dict, results: List[Dict], output_dir: Path):
    """Create comprehensive visualizations for the analysis results."""
    plot_detection_turn_distribution(metrics, output_dir)
    plot_confusion_matrix(metrics, output_dir)
    plot_layer_performance(metrics, output_dir)
    plot_summary_metrics(metrics, output_dir)
    plot_metrics_by_length(metrics, output_dir)
    plot_detection_by_turn_and_length(results, output_dir)
    plot_detection_lead_time(results, output_dir)

def get_model_layer_count(model) -> int:
    """
    Automatically detect the number of layers in the model.
    """
    try:
        # For most transformer models, layers are in model.model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return len(model.model.layers)
        # Alternative for some model architectures
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return len(model.transformer.h)
        elif hasattr(model, 'layers'):
            return len(model.layers)
        else:
            print("Warning: Could not auto-detect layer count. Using default of 24.")
            return 24
    except Exception as e:
        print(f"Error detecting layer count: {e}. Using default of 24.")
        return 24

def get_strategic_layers(total_layers: int) -> Dict[str, int]:
    """
    Get strategic layer indices: first, middle, last.
    
    Args:
        total_layers: Total number of layers in the model
    
    Returns:
        Dictionary with layer identifiers and their indices
    """
    strategic_layers = {
        'first': 0,  # First layer (0-indexed)
        'mid': total_layers // 2,  # Middle layer
        'last': total_layers - 1  # Last layer (0-indexed)
    }
    
    print(f"Strategic layers for {total_layers}-layer model:")
    for name, idx in strategic_layers.items():
        print(f"  {name}: layer {idx}")
    
    return strategic_layers

def extract_layer_info_from_metadata(metadata: List[Dict]) -> List[Union[int, str]]:
    """
    Extract all unique layers from metadata's activations dictionary.

    Args:
        metadata: List of conversation metadata dictionaries

    Returns:
        List of unique layer identifiers (both numeric and string-based)
    """
    all_layers = set()

    # Scan through all conversations to find layer information
    for item in metadata:
        if 'activations' not in item:
            continue

        # Extract layer identifiers from activation keys
        for key in item['activations'].keys():
            if 'layer_' not in key:
                continue

            # Parse the layer identifier from keys like "turn_X_layer_Y_..."
            parts = key.split('_')
            for i, part in enumerate(parts):
                if part == 'layer' and i+1 < len(parts):
                    layer_id = parts[i+1]
                    if layer_id.isdigit():  # Numeric layer
                        all_layers.add(int(layer_id))
                    elif layer_id == 'avg':  # Layer average
                        all_layers.add('layer_avg')

    # Return sorted list with numeric layers first, then string layers
    numeric_layers = sorted([l for l in all_layers if isinstance(l, int)])
    string_layers = sorted([l for l in all_layers if isinstance(l, str)])

    return numeric_layers + string_layers

if __name__ == "__main__":
    # Setup configurations
    config = {
        # Path to preprocessed data (from dataset_stats.ipynb)
        'processed_data_path': None,

        'model_name': "meta-llama/Llama-3.2-1B-Instruct",
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'output_dir': Path("./results"),
        'seed': 42,
        
        # Auto-detect layers and use strategic subset for evaluation
        'extract_all_layers': False,  # Extract activations from all layers
        'evaluation_layers': 'strategic',  # Use strategic subset for evaluation
        
        # Layer averaging configuration
        'use_layer_averaging': True,
        'avg_layer_name': 'layer_avg',
        
        # Voting system parameters
        'turn_threshold': 0.8,
        'min_turns_per_layer': 1,
        'min_layers_for_harmful': 2,

        'has_jailbreak_scores': False,
        
        # Add logging configuration
        'log_level': 'INFO',
        'file_log_level': 'DEBUG',
    }
    
    # Create output directory if it doesn't exist
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = config['output_dir'] / "logs"
    logger, log_dir, timestamp = setup_logging(
        log_dir, 
        console_level=getattr(logging, config['log_level']),
        file_level=getattr(logging, config['file_log_level'])
    )
    
    # Create component loggers
    extraction_logger = get_component_logger('extraction', log_dir, timestamp)
    data_prep_logger = get_component_logger('data_prep', log_dir, timestamp)
    train_logger = get_component_logger('training', log_dir, timestamp)
    
    logger.info(f"Starting Multi-Turn Linear Probes analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Logs will be saved to {log_dir}")

    # Set random seed for reproducibility
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    logger.info(f"Using device: {config['device']}")
    
    # Update the path to point to your classified data
    config['processed_data_path'] = Path("datasets/evaluated_llama_answered_harmful_dataset.json")
    all_conversations_harmful = load_preprocessed_data(config['processed_data_path'])
    
    config['processed_data_path'] = Path("datasets/evaluated_llama_answered_harmless_dataset.json")
    all_conversations_harmless = load_preprocessed_data(config['processed_data_path'])
    
    # Combine both datasets
    all_conversations = all_conversations_harmful + all_conversations_harmless
    logger.info(f"Combined dataset: {len(all_conversations)} conversations")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model(config['model_name'], config['device'])
    
    # Auto-detect total layers and determine strategic layers
    total_layers = get_model_layer_count(model)
    strategic_layers = get_strategic_layers(total_layers)
    
    # Set up layer configuration
    if config['extract_all_layers']:
        # Extract activations from ALL layers
        all_layers = list(range(total_layers))
        logger.info(f"Extracting activations from all {total_layers} layers")
    else:
        # Extract only from strategic layers (fallback)
        all_layers = list(strategic_layers.values())
        logger.info(f"Extracting activations from strategic layers only: {all_layers}")
    
    # Set evaluation layers to strategic subset
    if config['evaluation_layers'] == 'strategic':
        config['layers'] = list(strategic_layers.values())
        logger.info(f"Evaluating strategic layers: {config['layers']}")
    else:
        # Fallback to original configuration
        config['layers'] = [1, 11, 23]
        logger.info(f"Using original layer configuration: {config['layers']}")
    
    # Log layer configuration summary
    logger.info(f"\n=== Layer Configuration ===")
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Total layers detected: {total_layers}")
    if config['extract_all_layers']:
        logger.info(f"Activations will be extracted from: ALL {total_layers} layers")
    else:
        logger.info(f"Activations will be extracted from: {list(strategic_layers.values())}")
    logger.info(f"Strategic layers: {strategic_layers}")
    logger.info(f"Evaluation layers: {config['layers']}")
    if config.get('use_layer_averaging', False):
        logger.info(f"Average layer '{config['avg_layer_name']}' computed from strategic layers: {list(strategic_layers.values())}")
    logger.info("=" * 40)

    # Process conversations to extract activations using optimized approach
    logger.info(f"Processing conversations to extract activations...")
    processing_result = process_conversations_optimized_batched(
        all_conversations, 
        model, 
        tokenizer, 
        config['device'], 
        all_layers,
        config['output_dir'],
        use_layer_averaging=config.get('use_layer_averaging', False),
        avg_layer_name=config.get('avg_layer_name', 'layer_avg'),
        has_jailbreak_scores=True,
        strategic_layers=strategic_layers,
        logger=logger  # Add main logger here
    )
    
    # Prepare data for probing - ONLY for strategic layers + average
    logger.info("Preparing data for probe training...")
    evaluation_layers = config['layers'].copy()
    if config.get('use_layer_averaging', False):
        evaluation_layers.append(config.get('avg_layer_name', 'layer_avg'))
    
    layer_data = prepare_data_for_probing_optimized(
        config['layers'],
        processing_result['metadata'],
        processing_result['activations_file'],
        use_layer_averaging=config.get('use_layer_averaging', False),
        avg_layer_name=config.get('avg_layer_name', 'layer_avg'),
        feature_strategy=None,  # Will be determined later
        logger=data_prep_logger  # Add data preparation logger
    )

    correlation_results = None
    if any(len(data_tuple) == 6 for data_tuple in layer_data.values()):
        logger.info("Performing correlation analysis...")
        correlation_results = analyze_activation_jailbreak_correlation(layer_data, config['output_dir'])
    else:
        logger.info("No jailbreak scores found - skipping correlation analysis")

    # Determine global strategy first
    global_strategy = determine_global_feature_strategy(correlation_results)
    logger.info(f"Determined global strategy: {global_strategy['primary_features']}")

    # Train linear probes with global strategy and baselines
    logger.info("Training linear probes with global strategy...")
    probes = train_linear_probes_with_baselines(
        layer_data,
        config['output_dir'],
        correlation_results=correlation_results,
        logger=train_logger  # Add training logger
    )    
    
    # Create comprehensive performance comparison
    logger.info("Creating performance visualizations...")
    create_performance_comparison_plot(probes, config['output_dir'])
    create_strategy_summary_plot(probes, config['output_dir'])

    # After training probes
    logger.info("Analyzing jailbreak relationships...")
    analyze_jailbreak_relationship(layer_data, probes, config['output_dir'])

    # Analyze feature importance
    logger.info("Analyzing feature importance...")
    analyze_features(probes, config['output_dir'])
    
    
    # Classify conversations using the trained probes
    logger.info("Classifying conversations...")
    classification_results = classify_conversations_with_voting(
        processing_result['metadata'],
        probes,
        processing_result['activations_file'],
        config
    )
    
    # Save classification results
    logger.info("Saving classification results...")
    pd.DataFrame(classification_results).to_csv(config['output_dir'] / "classification_results.csv", index=False)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics_results = calculate_metrics_optimized(classification_results)

    # Save all metrics
    with open(config['output_dir'] / "all_metrics.json", "w") as f:
        json.dump(metrics_results, f, indent=4)

    # Save individual metric files for backward compatibility
    with open(config['output_dir'] / "metrics.json", "w") as f:
        json.dump(metrics_results['combined_metrics'], f, indent=4)

    with open(config['output_dir'] / "test_metrics.json", "w") as f:
        json.dump(metrics_results['test_metrics'], f, indent=4)

    # Print summary statistics
    logger.info("\nResults Summary (TEST SET ONLY):")
    test_metrics = metrics_results['test_metrics']
    logger.info(f"Test conversations: {test_metrics['total_conversations']}")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
    
    for k, val in test_metrics.items():
        if k.startswith('precision_at_') and isinstance(val, (int, float)):
            logger.info(f"{k}: {val:.4f}")

    if test_metrics['avg_detection_turn'] is not None:
        logger.info(f"Average detection turn: {test_metrics['avg_detection_turn']:.2f}")
    else:
        logger.info("Average detection turn: N/A")
    
    # Generate plots
    logger.info("Generating visualization plots...")
    plot_results_optimized(test_metrics, classification_results, config['output_dir'])
    logger.info(f"Analysis complete. Results saved to {config['output_dir']}")