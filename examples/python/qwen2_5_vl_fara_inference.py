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


def format_conversation_for_qwen(conversation: List[Dict], image_grid_thw: Optional[Tuple[int, int, int]] = None, spatial_merge_size: int = 2) -> Tuple[str, str]:
    """
    Convert FARA conversation format to Qwen prompt format.
    
    This follows the chat template pattern from chat_template.jinja:
    - System: <|im_start|>system\\n{content}<|im_end|>\\n
    - User with image: <|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>\\n  
    - Then applies get_image_padding_from_text() logic to expand <|image_pad|>
    
    Args:
        conversation: List of conversation turns with role and content
        image_grid_thw: Tuple of (temporal, height_patches, width_patches) for image grid
        spatial_merge_size: Merge size for patches (default: 2, meaning 2x2 merge)
        
    Returns:
        Tuple of (formatted_prompt, user_query)
    """
    prompt_parts = []
    user_query = None
    
    # Find the last user query for reference
    for message in reversed(conversation):
        if message['role'] == 'user':
            # Extract text content from user message
            for content in message.get('content', []):
                if content.get('type') == 'text' or isinstance(content, dict) and 'text' in content:
                    user_query = content.get('text', '')
                    break
            break
    
    # Build conversation prompt following chat template
    for message in conversation:
        role = message['role']
        content_list = message.get('content', [])
        
        if role == 'system':
            prompt_parts.append("<|im_start|>system\n")
            for content in content_list:
                if isinstance(content, dict) and 'text' in content:
                    prompt_parts.append(content['text'])
            prompt_parts.append("<|im_end|>\n")
            
        elif role == 'user':
            prompt_parts.append("<|im_start|>user\n")
            
            # Check if there's an image
            has_image = any(
                (isinstance(c, dict) and c.get('type') == 'image') or 
                (isinstance(c, str) and c == 'image')
                for c in content_list
            )
            
            if has_image:
                # Add ONE <|image_pad|> token - will be expanded below
                prompt_parts.append("<|vision_start|><|image_pad|><|vision_end|>")
            
            # Add text content
            for content in content_list:
                if isinstance(content, dict) and content.get('type') == 'text':
                    prompt_parts.append(content['text'])
                elif isinstance(content, dict) and 'text' in content:
                    prompt_parts.append(content['text'])
            
            prompt_parts.append("<|im_end|>\n")
            
        elif role == 'assistant':
            prompt_parts.append("<|im_start|>assistant\n")
            for content in content_list:
                if isinstance(content, dict) and 'text' in content:
                    prompt_parts.append(content['text'])
                elif isinstance(content, str):
                    prompt_parts.append(content)
            prompt_parts.append("<|im_end|>\n")
    
    # Add generation prompt
    prompt_parts.append("<|im_start|>assistant\n")
    
    prompt_text = "".join(prompt_parts)
    
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
    import numpy as np
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
    
    # Now format the conversation with the image_grid_thw
    prompt, user_query = format_conversation_for_qwen(sample['conversation'], image_grid_thw=image_grid_thw_tuple)
    
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
    
    # Re-process with the actual prompt to get correct inputs
    inputs = processor(prompt, images=images)
    
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
    
    # Generate response
    print("\n=== Assistant Response ===")
    generator = og.Generator(model, params)
    
    # For pipeline models, pass vision inputs via set_model_input before append_tokens
    # Convert og.Tensor to numpy array using as_numpy() method
    if 'pixel_values' in inputs:
        print("Converting pixel_values to numpy...")
        pixel_values_np = inputs['pixel_values'].as_numpy()
        print(f"pixel_values shape: {pixel_values_np.shape}")
        print("Setting pixel_values as model input...")
        generator.set_model_input('pixel_values', pixel_values_np)
        print("pixel_values set successfully")
    if 'image_grid_thw' in inputs:
        print("Converting image_grid_thw to numpy...")
        image_grid_thw_np = inputs['image_grid_thw'].as_numpy()
        print(f"image_grid_thw shape: {image_grid_thw_np.shape}")
        print("Setting image_grid_thw as model input...")
        generator.set_model_input('image_grid_thw', image_grid_thw_np)
        print("image_grid_thw set successfully")
    
    print(f"Appending {len(input_ids_array)} tokens to generator...")
    print(f"Input IDs array shape: {input_ids_array.shape}, dtype: {input_ids_array.dtype}")
    print(f"First 10 tokens: {input_ids_array[:10].tolist()}")
    print(f"Last 10 tokens: {input_ids_array[-10:].tolist()}")
    
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
    while not generator.is_done():
        print(f"Generating token {token_count + 1}...")
        generator.generate_next_token()
        
        new_token = generator.get_next_tokens()[0]
        print(f"Got token: {new_token}")
        output_tokens.append(new_token)
        token_count += 1
        
        # Decode and print token
        token_text = tokenizer.decode([new_token])
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
        default=0.7,
        help="Sampling temperature (default: 0.7)",
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
        default=0.9,
        help="Top-p (nucleus) sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Run inference on specific sample index only (default: all samples)",
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
                args.max_length, args.temperature, args.top_k, args.top_p
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
                    args.max_length, args.temperature, args.top_k, args.top_p
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
