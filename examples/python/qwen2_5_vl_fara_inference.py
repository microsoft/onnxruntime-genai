"""
Qwen 2.5 VL Inference using FARA Dataset Format

This script uses the same sample format as the FARA NPU reference implementation,
loading samples from directories with sample.json and images.

Usage:
    python qwen2_5_vl_fara_inference.py --model_path <path_to_model> --data_path <path_to_data>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Error: PIL (Pillow) not found. Please install it: pip install Pillow")
    sys.exit(1)

try:
    import onnxruntime_genai as og
except ImportError:
    print("Error: onnxruntime_genai not found. Please install it first.")
    sys.exit(1)

# Import baseline image preprocessing
try:
    # Try to import from the baseline directory
    baseline_dir = Path(r"C:\Users\asonawane\Downloads\fara_npu_models_ver_1.0.1\fara_cua_vl_demo_inference")
    if baseline_dir.exists():
        sys.path.insert(0, str(baseline_dir))
        from image_utils import fetch_image_data
        print(f"✓ Using baseline image preprocessing from {baseline_dir}")
        USE_BASELINE_PREPROCESSING = True
    else:
        print(f"Warning: Baseline directory not found at {baseline_dir}")
        print("Will use og.Images processor instead")
        USE_BASELINE_PREPROCESSING = False
except ImportError as e:
    print(f"Warning: Could not import baseline image_utils: {e}")
    print("Will use og.Images processor instead")
    USE_BASELINE_PREPROCESSING = False


class FaraDatasetLoader:
    """Dataset loader compatible with FARA sample format."""
    
    def __init__(self, data_dir: str):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Path to data directory containing sample subdirectories
        """
        self.data_dir = Path(data_dir)
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from data directory."""
        samples = []
        
        # Check if data_dir itself is a single sample directory
        sample_json = self.data_dir / "sample.json"
        if sample_json.exists():
            sample_dirs = [self.data_dir]
        else:
            # Look for subdirectories containing sample.json
            sample_dirs = [d for d in self.data_dir.iterdir() 
                          if d.is_dir() and (d / "sample.json").exists()]
        
        for sample_dir in sample_dirs:
            try:
                with open(sample_dir / "sample.json", "r", encoding="utf-8") as f:
                    sample_data = json.load(f)
                
                if "conversation" not in sample_data or "images" not in sample_data:
                    print(f"Warning: Skipping {sample_dir}, missing required fields")
                    continue
                
                # Resolve image paths
                sample_data["images"] = [
                    str(sample_dir / img_name) 
                    for img_name in sample_data["images"]
                ]
                sample_data["file_name"] = sample_dir.name
                samples.append(sample_data)
                
            except Exception as e:
                print(f"Warning: Failed to load sample from {sample_dir}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample by index."""
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        return self.samples[idx]


def format_conversation_for_qwen(tokenizer, conversation: List[Dict], image_grid_thw: Optional[Tuple[int, int, int]] = None, spatial_merge_size: int = 2) -> Tuple[str, str]:
    """
    Convert FARA conversation format to Qwen prompt format using apply_chat_template.
    Matches the reference implementation exactly.
    
    Args:
        tokenizer: OnnxRuntime GenAI tokenizer with apply_chat_template support
        conversation: List of conversation turns with role and content
        image_grid_thw: Tuple of (temporal, height_patches, width_patches) for image grid
        spatial_merge_size: Merge size for patches (default: 2, meaning 2x2 merge)
        
    Returns:
        Tuple of (formatted_prompt, user_query)
    """
    import json
    
    # Find the last user query for reference
    user_query = None
    for message in reversed(conversation):
        if message['role'] == 'user':
            # In reference, user_query is the raw content, which can be a list or string
            user_query = message.get('content', '')
            if isinstance(user_query, list):
                # Extract text from content list
                for content in user_query:
                    if isinstance(content, dict) and content.get('type') == 'text':
                        user_query = content.get('text', '')
                        break
            break
    
    # Convert conversation to JSON string format expected by apply_chat_template
    # The tokenizer expects JSON string with the conversation array
    messages_json = json.dumps(conversation)
    
    # Use OnnxRuntime GenAI tokenizer's apply_chat_template
    # This matches: tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    prompt_text = tokenizer.apply_chat_template(messages_json, add_generation_prompt=True)
    
    # Apply get_image_padding_from_text logic: expand <|image_pad|> based on image_grid_thw
    if image_grid_thw is not None:
        merge_length = spatial_merge_size ** 2  # 2^2 = 4
        # Calculate number of patches: t * h * w / merge_length
        # For (1, 34, 58): 1 * 34 * 58 / 4 = 493
        num_patches = (image_grid_thw[0] * image_grid_thw[1] * image_grid_thw[2]) // merge_length
        
        # Replace each <|image_pad|> with the calculated number of tokens
        # Use placeholder approach like reference script
        prompt_text = prompt_text.replace("<|image_pad|>", "<|placeholder|>" * num_patches, 1)
        prompt_text = prompt_text.replace("<|placeholder|>", "<|image_pad|>")
    
    return prompt_text, user_query or "No query found"


def run_inference_on_sample(
    model: og.Model,
    processor,
    tokenizer,
    sample: Dict,
    max_length: int = 2048,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    use_baseline_embeddings: bool = False,
) -> str:
    """
    Run inference on a single FARA sample.
    
    Args:
        model: OnnxRuntime GenAI model
        processor: Multimodal processor
        tokenizer: OnnxRuntime GenAI tokenizer
        sample: Sample dictionary with conversation and images
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        
    Returns:
        Generated response text
    """
    # Load image
    image_path = sample['images'][0]
    print(f"Loading image: {image_path}")
    
    # Verify image exists
    if not Path(image_path).exists():
        return f"Error: Image file not found - {image_path}"
    
    # Choose preprocessing method based on availability
    if USE_BASELINE_PREPROCESSING:
        print("\n=== USING BASELINE IMAGE PREPROCESSING ===")
        try:
            # Use baseline's fetch_image_data function
            # IMPORTANT: Use resize_width=800, resize_height=480 to match baseline qwen_inference.py defaults
            image_data = fetch_image_data(image_path, resize_width=800, resize_height=480)
            pixel_values_np = image_data['pixel_values']
            image_grid_thw_np = image_data['image_grid_thw']
            
            print(f"✓ Baseline preprocessing complete")
            print(f"  pixel_values shape: {pixel_values_np.shape}")
            print(f"  image_grid_thw: {image_grid_thw_np}")
            print(f"  pixel_values stats: Min={pixel_values_np.min():.6f}, Max={pixel_values_np.max():.6f}, Mean={pixel_values_np.mean():.6f}")
            print(f"  First 10 values: {pixel_values_np.flatten()[:10].tolist()}")
            
            # Extract image_grid_thw as tuple for prompt formatting
            image_grid_thw_tuple = tuple(int(x) for x in image_grid_thw_np[0])
            print(f"  image_grid_thw_tuple: {image_grid_thw_tuple}")
            
            # Store in inputs dict for later use
            inputs = {
                'pixel_values': pixel_values_np,
                'image_grid_thw': image_grid_thw_np
            }
            
        except Exception as e:
            print(f"✗ Error in baseline preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: Baseline preprocessing failed - {e}"
    else:
        print("\n=== USING OG.IMAGES PREPROCESSING ===")
        # Load image using og.Images
        try:
            images = og.Images.open(image_path)
            print(f"Image loaded successfully")
        except Exception as e:
            print(f"Error loading image: {e}")
            return f"Error: Failed to load image - {e}"
        
        # First, create a temporary prompt to process the image and get image_grid_thw
        # We'll regenerate the proper prompt once we know how many image tokens we need
        temp_prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Question<|im_end|>\n"
        
        # Process images using og.Images - this will give us pixel_values and image_grid_thw
        inputs = processor(temp_prompt, images=images)
        
        # Debug: Print what tensors are in the inputs
        print(f"Available tensors in inputs: {list(inputs.keys())}")
        
        # Calculate the actual number of image pad tokens from image_grid_thw
        # Formula: image_grid_thw.prod() // (spatial_merge_size ** 2)
        # where spatial_merge_size = 2 for Qwen 2.5 VL
        image_grid_thw_tuple = None
        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw']
            print(f"image_grid_thw type: {type(image_grid_thw)}")
            # Try different ways to extract the values
            try:
                if hasattr(image_grid_thw, 'numpy'):
                    grid_array = image_grid_thw.numpy()
                elif hasattr(image_grid_thw, 'data'):
                    # og.Tensor has a data() method that returns numpy array
                    grid_array = np.array(image_grid_thw.data())
                else:
                    # Fallback: extract from string representation or use default
                    print(f"image_grid_thw object: {image_grid_thw}")
                    # Default for FARA: (1, 34, 58)
                    grid_array = np.array([[1, 34, 58]])
                
                print(f"image_grid_thw values: {grid_array}")
                image_grid_thw_tuple = tuple(int(x) for x in grid_array[0])  # Get first batch as tuple
                print(f"image_grid_thw_tuple: {image_grid_thw_tuple}")
            except Exception as e:
                print(f"Error extracting image_grid_thw: {e}")
                # Use default for FARA dataset
                image_grid_thw_tuple = (1, 34, 58)
                print(f"Using default image_grid_thw_tuple: {image_grid_thw_tuple}")
    
    # Now format the conversation with the image_grid_thw using tokenizer's apply_chat_template
    prompt, user_query = format_conversation_for_qwen(tokenizer, sample['conversation'], image_grid_thw=image_grid_thw_tuple)
    
    print("\n=== User Query ===")
    print(user_query)
    print("\n=== Full Prompt (first 500 chars) ===")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    # For pipeline models like qwen2_5_vl_pipeline:
    # 1. Tokenize the text prompt
    # 2. Process images separately (returns pixel_values and image_grid_thw)
    # 3. Use append_tokens with the tokenized prompt
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    print(f"\nTokenized prompt length: {len(input_ids)}")
    
    # Check for EOS tokens in the input
    eos_tokens = [151645, 151643]
    eos_count = sum(1 for token in input_ids if token in eos_tokens)
    print(f"EOS tokens in input: {eos_count} occurrences")
    
    # Count image pad tokens
    image_token_id = 151655
    image_token_count = sum(1 for token in input_ids if token == image_token_id)
    print(f"Image pad tokens (151655) in tokenized prompt: {image_token_count}")
    
    # Decode and print the prompt to compare with reference
    print("\n=== DECODED PROMPT (for comparison with reference) ===")
    decoded_prompt = tokenizer.decode(input_ids)
    print(decoded_prompt[:2000])  # First 2000 chars
    print(f"\n... (showing first 2000 of {len(decoded_prompt)} chars)")
    print("=== END DECODED PROMPT ===\n")
    
    # Save the prompt to a file for comparison
    with open("our_prompt.txt", "w", encoding="utf-8") as f:
        f.write(decoded_prompt)
    print(f"[DEBUG] Saved full decoded prompt to our_prompt.txt")
    
    # Find where image tokens are and print surrounding tokens
    print("=== TOKEN ANALYSIS ===")
    image_positions = [i for i, tid in enumerate(input_ids) if tid == 151655]
    if image_positions:
        first_img_pos = image_positions[0]
        last_img_pos = image_positions[-1]
        print(f"First image token at position: {first_img_pos}")
        print(f"Last image token at position: {last_img_pos}")
        print(f"Total image tokens: {len(image_positions)}")
        
        # Check positions in sliding windows
        print(f"\n=== Sliding Window Analysis (window_size=512) ===")
        for i, pos in enumerate([first_img_pos, last_img_pos]):
            chunk_idx = pos // 512
            within_chunk_pos = pos % 512
            print(f"Position {pos} → Chunk {chunk_idx}, Within-chunk position {within_chunk_pos}")
        
        print(f"\nTokens BEFORE first image (positions {first_img_pos-10} to {first_img_pos-1}):")
        print(f"  IDs: {input_ids[first_img_pos-10:first_img_pos]}")
        print(f"  Decoded: '{tokenizer.decode(input_ids[first_img_pos-10:first_img_pos])}'")
        print(f"Tokens AFTER last image (positions {last_img_pos+1} to {last_img_pos+11}):")
        print(f"  IDs: {input_ids[last_img_pos+1:last_img_pos+11]}")
        print(f"  Decoded: '{tokenizer.decode(input_ids[last_img_pos+1:last_img_pos+11])}'")
        
        # Verify all image token positions
        print(f"\n=== Verifying all {len(image_positions)} image token positions ===")
        print(f"First 10: {image_positions[:10]}")
        print(f"Last 10: {image_positions[-10:]}")
        print(f"Are they consecutive? {all(image_positions[i+1] - image_positions[i] == 1 for i in range(len(image_positions)-1))}")
    print("=== END TOKEN ANALYSIS ===\n")
    
    # For og.Images, re-process with the actual prompt to get correct inputs
    if not USE_BASELINE_PREPROCESSING:
        inputs = processor(prompt, images=images)
    # For baseline preprocessing, inputs already contain pixel_values and image_grid_thw
    
    # Convert input_ids to numpy array
    input_ids_array = np.array(input_ids, dtype=np.int32)
    
    # Set generation parameters
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    print(f"\n=== [OUR SCRIPT] Generation Parameters ===")
    print(f"max_length: {max_length}")
    print(f"temperature: {temperature}")
    print(f"top_k: {top_k}")
    print(f"top_p: {top_p}")
    print(f"===\n")
    
    # Generate response
    print("\n=== Assistant Response ===")
    generator = og.Generator(model, params)
    
    # For pipeline models, pass vision inputs via set_model_input before append_tokens
    # Convert og.Tensor to numpy array using as_numpy() method (for og.Images)
    # or use numpy arrays directly (for baseline preprocessing)
    
    # Option 1: Load baseline embeddings if requested
    if use_baseline_embeddings:
        print("\n[OPTION 1] Loading baseline vision embeddings...")
        baseline_embeddings = np.load("baseline_vision_embeddings.npy")
        print(f"[OPTION 1] Loaded embeddings shape: {baseline_embeddings.shape}")
        print(f"[OPTION 1] Min: {baseline_embeddings.min():.4f}, Max: {baseline_embeddings.max():.4f}")
        print(f"[OPTION 1] Mean: {baseline_embeddings.mean():.4f}, Std: {baseline_embeddings.std():.4f}")
        print("[OPTION 1] Setting baseline vision_embeddings as model input...")
        generator.set_model_input('vision_embeddings', baseline_embeddings)
        print("[OPTION 1] Baseline vision_embeddings set successfully")
    elif 'pixel_values' in inputs:
        print("\nSetting pixel_values as model input...")
        # Check if it's an og.Tensor or numpy array
        if hasattr(inputs['pixel_values'], 'as_numpy'):
            # og.Tensor from og.Images processor
            pixel_values_np = inputs['pixel_values'].as_numpy()
            print(f"  (converted from og.Tensor)")
        else:
            # Already numpy array from baseline preprocessing
            pixel_values_np = inputs['pixel_values']
            print(f"  (using baseline numpy array)")
        
        print(f"  pixel_values shape: {pixel_values_np.shape}")
        print(f"  pixel_values stats: Min={pixel_values_np.min():.6f}, Max={pixel_values_np.max():.6f}, Mean={pixel_values_np.mean():.6f}")
        generator.set_model_input('pixel_values', pixel_values_np)
        print("  ✓ pixel_values set successfully")
        
    if 'image_grid_thw' in inputs:
        print("\nSetting image_grid_thw as model input...")
        # Check if it's an og.Tensor or numpy array
        if hasattr(inputs['image_grid_thw'], 'as_numpy'):
            # og.Tensor from og.Images processor
            image_grid_thw_np = inputs['image_grid_thw'].as_numpy()
            print(f"  (converted from og.Tensor)")
        else:
            # Already numpy array from baseline preprocessing
            image_grid_thw_np = inputs['image_grid_thw']
            print(f"  (using baseline numpy array)")
        
        print(f"  image_grid_thw shape: {image_grid_thw_np.shape}")
        print(f"  image_grid_thw values: {image_grid_thw_np}")
        generator.set_model_input('image_grid_thw', image_grid_thw_np)
        print("  ✓ image_grid_thw set successfully")
    
    print(f"Appending {len(input_ids_array)} tokens to generator...")
    print(f"Input IDs array shape: {input_ids_array.shape}, dtype: {input_ids_array.dtype}")
    print(f"First 10 tokens: {input_ids_array[:10].tolist()}")
    print(f"Last 10 tokens: {input_ids_array[-10:].tolist()}")
    
    # Debug: Print tokens around image token positions
    print(f"\n=== Verifying tokens at image positions ===")
    print(f"Total tokens: {len(input_ids_array)}")
    if image_positions:
        first_pos = image_positions[0]
        last_pos = image_positions[-1]
        print(f"First image token at: {first_pos}")
        print(f"Last image token at: {last_pos}")
        if first_pos >= 5:
            print(f"Tokens before first image ({first_pos-5} to {first_pos-1}): {input_ids_array[first_pos-5:first_pos].tolist()}")
            print(f"First few image tokens ({first_pos} to {first_pos+9}): {input_ids_array[first_pos:first_pos+10].tolist()}")
        if last_pos < len(input_ids_array) - 10:
            print(f"Last few image tokens ({last_pos-9} to {last_pos}): {input_ids_array[last_pos-9:last_pos+1].tolist()}")
            print(f"Tokens after last image ({last_pos+1} to {last_pos+10}): {input_ids_array[last_pos+1:last_pos+11].tolist()}")
    print("===")
    
    try:
        print("Calling append_tokens...")
        generator.append_tokens(input_ids_array)
        print("✓ Tokens appended successfully")
    except Exception as e:
        print(f"✗ ERROR during append_tokens: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"
    
    try:
        is_done = generator.is_done()
        print(f"✓ Generator is_done: {is_done}")
    except Exception as e:
        print(f"✗ ERROR checking is_done: {e}")
        return f"Error: {e}"
    print("Starting generation loop...")
    output_tokens = []
    token_count = 0
    
    # Debug: Get initial logits before first token generation
    print("\n=== [OUR SCRIPT] PRE-GENERATION STATE ===")
    print(f"Generator is_done: {generator.is_done()}")
    print("===\n")
    
    while not generator.is_done():
        print(f"\n=== [OUR SCRIPT] Generating token {token_count + 1} ===")
        
        # Try to get logits before generate_next_token (may not be available)
        try:
            # This is just to see if we can inspect internal state
            print(f"About to call generate_next_token()...")
        except Exception as e:
            print(f"Cannot inspect pre-generation state: {e}")
        
        generator.generate_next_token()
        
        new_token = generator.get_next_tokens()[0]
        print(f"Token ID: {new_token}")
        output_tokens.append(new_token)
        token_count += 1
        
        # Decode and print token
        token_text = tokenizer.decode([new_token])
        print(f"Decoded: '{token_text}'")
        
        # For first few tokens, print more details
        if token_count <= 10:
            print(f"[Token {token_count}] ID={new_token}, text='{token_text}'")
            print(f"[Token {token_count}] Full output so far: '{tokenizer.decode(output_tokens)}'")
        
        print(token_text, end="", flush=True)
        
        # Safety check to prevent infinite loops
        if token_count > max_length:
            print(f"\n[Reached max_length limit of {max_length}]")
            break
    
    print(f"\n[Generated {token_count} tokens]")
    print(f"Final is_done: {generator.is_done()}")
    
    # Decode full output
    output_text = tokenizer.decode(output_tokens)
    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen 2.5 VL inference on FARA dataset format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Qwen 2.5 VL model directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to FARA data directory with sample.json files",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.000001,
        help="Sampling temperature (default: 0.000001, near-greedy like baseline)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.05,
        help="Top-p (nucleus) sampling parameter (default: 1.05, matching baseline)",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Run inference on specific sample index only (default: all samples)",
    )
    parser.add_argument(
        "--use_baseline_embeddings",
        action="store_true",
        help="Use baseline vision embeddings from baseline_vision_embeddings.npy instead of C++ vision pipeline (Option 1 debugging)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model_path).exists():
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not Path(args.data_path).exists():
        print(f"Error: Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    # Load dataset
    dataset = FaraDatasetLoader(args.data_path)
    
    if len(dataset) == 0:
        print("Error: No samples found in data directory")
        sys.exit(1)
    
    # Load model
    print(f"\nLoading Qwen 2.5 VL model from: {args.model_path}")
    try:
        model = og.Model(args.model_path)
        processor = model.create_multimodal_processor()
        tokenizer = og.Tokenizer(model)
        print(f"Model loaded successfully: {model.type}\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run inference
    try:
        if args.sample_idx is not None:
            # Run on specific sample
            if args.sample_idx < 0 or args.sample_idx >= len(dataset):
                print(f"Error: Sample index {args.sample_idx} out of range [0, {len(dataset)-1}]")
                sys.exit(1)
            
            sample = dataset[args.sample_idx]
            print(f"\n{'='*60}")
            print(f"Sample {args.sample_idx}: {sample['file_name']}")
            print(f"{'='*60}")
            
            output = run_inference_on_sample(
                model, processor, tokenizer, sample,
                args.max_length, args.temperature, args.top_k, args.top_p,
                args.use_baseline_embeddings
            )
            
            print(f"\n{'='*60}")
            print("Generation Complete")
            print(f"{'='*60}\n")
            
        else:
            # Run on all samples
            for idx in range(len(dataset)):
                sample = dataset[idx]
                print(f"\n{'='*60}")
                print(f"Sample {idx}/{len(dataset)-1}: {sample['file_name']}")
                print(f"{'='*60}")
                
                output = run_inference_on_sample(
                    model, processor, tokenizer, sample,
                    args.max_length, args.temperature, args.top_k, args.top_p,
                    args.use_baseline_embeddings
                )
                
                print(f"\n{'='*60}")
                print(f"Sample {idx} Complete")
                print(f"{'='*60}\n")
                
                # Ask to continue if multiple samples
                if idx < len(dataset) - 1:
                    continue_prompt = input("Continue to next sample? [Y/n]: ").strip().lower()
                    if continue_prompt and continue_prompt != 'y':
                        print("Stopping inference.")
                        break
        
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
