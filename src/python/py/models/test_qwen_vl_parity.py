import argparse
import torch
import numpy as np
import onnxruntime as ort
from transformers import Qwen2_5_VLForConditionalGeneration
from typing import Tuple, Dict, Any

# --- Configuration ---

# Set to torch.bfloat16 or torch.float16 based on your model export
# config.json shows "torch_dtype": "bfloat16", so we default to that.
TORCH_DTYPE = torch.bfloat16 

# Tolerances for numerical comparison. BF16/FP16 require higher tolerances.
RTOL = 1e-2
ATOL = 1e-2

# --- Helper Functions ---

def to_numpy(tensor):
    """Move tensor to CPU and convert to numpy."""
    return tensor.detach().cpu().numpy()

def get_ort_inputs_and_names(
    sess: ort.InferenceSession, 
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: torch.Tensor | None = None,
    num_layers: int = 0,
    num_kv_heads: int = 0,
    head_dim: int = 0,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """Creates the complete dictionary of inputs required by the ONNX model."""
    
    ort_inputs = {
        "inputs_embeds": to_numpy(inputs_embeds),
        "position_ids": to_numpy(position_ids),
        "attention_mask": to_numpy(attention_mask)
    }

    if past_key_values is None:
        # Prefill: Create dummy pasts with 0 sequence length
        batch_size = inputs_embeds.shape[0]
        past_shape = (batch_size, num_kv_heads, 0, head_dim)
        dummy_past = torch.empty(past_shape, dtype=TORCH_DTYPE, device=device)
        dummy_past_np = to_numpy(dummy_past)
        
        for i in range(num_layers):
            ort_inputs[f"past_key_values.{i}.key"] = dummy_past_np
            ort_inputs[f"past_key_values.{i}.value"] = dummy_past_np
    else:
        # Decode: Use the provided pasts (which are already numpy arrays)
        for i in range(num_layers):
            ort_inputs[f"past_key_values.{i}.key"] = past_key_values[i*2]
            ort_inputs[f"past_key_values.{i}.value"] = past_key_values[i*2 + 1]
            
    return ort_inputs

def sort_ort_present_outputs(
    ort_outputs: Dict[str, np.ndarray]
) -> list[np.ndarray]:
    """Gets the 'present' KV cache outputs from ORT in the correct layer order."""
    present_names = sorted(
        [name for name in ort_outputs.keys() if "present" in name],
        key=lambda n: (int(n.split('.')[1]), n.split('.')[2]) # Sort by layer, then key/value
    )
    return [ort_outputs[name] for name in present_names]

def compare_outputs(
    hf_logits: torch.Tensor,
    ort_logits: np.ndarray,
    hf_presents: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    ort_presents: list[np.ndarray],
    step_name: str
):
    """Compares logits and KV cache outputs using numpy."""
    
    print(f"--- Comparing {step_name} Logits ---")
    np.testing.assert_allclose(
        to_numpy(hf_logits), 
        ort_logits, 
        rtol=RTOL, 
        atol=ATOL
    )
    print("Logits: PASS")

    print(f"\n--- Comparing {step_name} KV Cache ---")
    # hf_presents is a tuple of tuples: ((k0, v0), (k1, v1), ...)
    # Flatten it to a list: [k0, v0, k1, v1, ...]
    hf_presents_list = [to_numpy(t) for layer_kv in hf_presents for t in layer_kv]
    
    assert len(hf_presents_list) == len(ort_presents), \
        f"HF presents count ({len(hf_presents_list)}) != ORT presents count ({len(ort_presents)})"

    for i in range(len(hf_presents_list)):
        layer = i // 2
        kv_type = "key" if i % 2 == 0 else "value"
        
        hf_tensor = hf_presents_list[i]
        ort_tensor = ort_presents[i]
        
        np.testing.assert_allclose(
            hf_tensor, 
            ort_tensor, 
            rtol=RTOL, 
            atol=ATOL
        )
    print(f"KV Cache (all {len(hf_presents_list)} tensors): PASS")
    print(f"\n✅ {step_name} Parity Test Passed!\n")


def test_parity(hf_model_name: str, cache_dir: str, onnx_model_path: str, use_gpu: bool):
    """
    Runs a two-step (prefill and decode) parity test between the Hugging Face
    and ONNX models.
    """
    
    print(f"Loading Hugging Face model: {hf_model_name}")
    print("This requires `trust_remote_code=True`.")
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    torch_dtype = TORCH_DTYPE if device == "cuda" else torch.float32 # CPU doesn't support BF16
    
    if device == "cpu" and TORCH_DTYPE == torch.bfloat16:
        print("Warning: CPU does not support bfloat16. Testing with float32.")
        
    
    hf_full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        cache_dir=cache_dir
    ).to(device).eval()
    
    # The ONNX model is *only* the language_model component
    hf_text_model = hf_full_model.language_model
    config = hf_text_model.config
    
    # Get model parameters
    BATCH_SIZE = 1
    PREFILL_LEN = 10
    DECODE_LEN = 1
    HIDDEN_SIZE = config.hidden_size
    NUM_LAYERS = config.num_hidden_layers
    NUM_KV_HEADS = config.num_key_value_heads
    HEAD_DIM = config.hidden_size // config.num_attention_heads
    
    print("\n--- Model Parameters ---")
    print(f"Device: {device}")
    print(f"DType: {torch_dtype}")
    print(f"Layers: {NUM_LAYERS}")
    print(f"Hidden Size: {HIDDEN_SIZE}")
    print(f"KV Heads: {NUM_KV_HEADS}")
    print(f"Head Dim: {HEAD_DIM}")
    print("------------------------\n")

    print(f"Loading ONNX model: {onnx_model_path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_model_path, providers=providers)
    
    # Get all ONNX output names
    output_names = [o.name for o in sess.get_outputs()]
    output_names_dict = {name: i for i, name in enumerate(output_names)}

    # =================================================================
    # 1. PREFILL STEP
    # =================================================================
    print("Running Prefill Step (Sequence Length = {PREFILL_LEN})...")
    
    # --- Create HF/Torch Inputs ---
    # We test the text model component directly, so we create random inputs_embeds.
    inputs_embeds_prefill = torch.rand(
        (BATCH_SIZE, PREFILL_LEN, HIDDEN_SIZE), 
        dtype=torch_dtype, 
        device=device
    )
    
    # Create 3D position_ids [3, B, S]
    # For a text-only test, all 3 dimensions (T, H, W) are identical.
    pos_ids_1d_prefill = torch.arange(PREFILL_LEN, device=device).expand(BATCH_SIZE, -1)
    position_ids_prefill = pos_ids_1d_prefill.unsqueeze(0).expand(3, -1, -1)
    
    # Attention mask for prefill is [B, S_total]
    attention_mask_prefill = torch.ones(
        (BATCH_SIZE, PREFILL_LEN), 
        dtype=torch.int64, 
        device=device
    )
    
    cache_position_prefill = torch.arange(PREFILL_LEN, device=device)
    
    # --- Create ONNX Inputs ---
    ort_inputs_prefill = get_ort_inputs_and_names(
        sess,
        inputs_embeds_prefill,
        position_ids_prefill,
        attention_mask_prefill,
        past_key_values=None, # This tells the helper to create dummy 0-len pasts
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        device=device
    )
    
    # --- Run Models ---
    with torch.no_grad():
        hf_outputs_prefill = hf_text_model(
            inputs_embeds=inputs_embeds_prefill,
            position_ids=position_ids_prefill,
            attention_mask=attention_mask_prefill,
            past_key_values=None,
            cache_position=cache_position_prefill,
            return_dict=True,
            use_cache=True
        )

    ort_outputs_prefill_list = sess.run(output_names, ort_inputs_prefill)
    ort_outputs_prefill = {name: ort_outputs_prefill_list[i] for i, name in enumerate(output_names)}

    # --- Compare Prefill ---
    hf_logits_prefill = hf_full_model.lm_head(hf_outputs_prefill.last_hidden_state)
    ort_logits_prefill = ort_outputs_prefill["logits"]
    
    hf_presents_prefill = hf_outputs_prefill.past_key_values.to_tuple()
    ort_presents_prefill = sort_ort_present_outputs(ort_outputs_prefill)

    compare_outputs(
        hf_logits_prefill,
        ort_logits_prefill,
        hf_presents_prefill,
        ort_presents_prefill,
        step_name="Prefill"
    )

    # =================================================================
    # 2. DECODE STEP
    # =================================================================
    print(f"Running Decode Step (Sequence Length = {DECODE_LEN})...")
    
    # --- Create HF/Torch Inputs ---
    inputs_embeds_decode = torch.rand(
        (BATCH_SIZE, DECODE_LEN, HIDDEN_SIZE), 
        dtype=torch_dtype, 
        device=device
    )
    
    # Position ID for the *new* token is just its index
    pos_ids_1d_decode = torch.tensor(
        [[PREFILL_LEN]], 
        dtype=torch.int64, 
        device=device
    )
    position_ids_decode = pos_ids_1d_decode.unsqueeze(0).expand(3, -1, -1)
    
    # Attention mask for decode is [B, S_total]
    attention_mask_decode = torch.ones(
        (BATCH_SIZE, PREFILL_LEN + DECODE_LEN), 
        dtype=torch.int64, 
        device=device
    )
    
    cache_position_decode = torch.tensor([PREFILL_LEN], device=device)
    
    # Use the KV cache from the HF prefill run
    hf_past_key_values = hf_outputs_prefill.past_key_values
    
    # --- Create ONNX Inputs ---
    # Use the KV cache from the ONNX prefill run
    ort_past_key_values = ort_presents_prefill
    
    ort_inputs_decode = get_ort_inputs_and_names(
        sess,
        inputs_embeds_decode,
        position_ids_decode,
        attention_mask_decode,
        past_key_values=ort_past_key_values,
        num_layers=NUM_LAYERS
    )
    
    # --- Run Models ---
    with torch.no_grad():
        hf_outputs_decode = hf_text_model(
            inputs_embeds=inputs_embeds_decode,
            position_ids=position_ids_decode,
            attention_mask=attention_mask_decode,
            past_key_values=hf_past_key_values,
            cache_position=cache_position_decode,
            return_dict=True,
            use_cache=True
        )

    ort_outputs_decode_list = sess.run(output_names, ort_inputs_decode)
    ort_outputs_decode = {name: ort_outputs_decode_list[i] for i, name in enumerate(output_names)}

    # --- Compare Decode ---
    hf_logits_decode = hf_full_model.lm_head(hf_outputs_decode.last_hidden_state)
    ort_logits_decode = ort_outputs_decode["logits"]
    
    hf_presents_decode = hf_outputs_decode.past_key_values.to_tuple()
    ort_presents_decode = sort_ort_present_outputs(ort_outputs_decode)
    
    compare_outputs(
        hf_logits_decode,
        ort_logits_decode,
        hf_presents_decode,
        ort_presents_decode,
        step_name="Decode"
    )

    print("="*30)
    print("🎉 All Parity Tests Passed! 🎉")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parity test for Qwen 2.5 VL ONNX model.")
    parser.add_argument(
        "--hf_model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Path or name of the Hugging Face model."
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        default="./qwen_fp32/model.onnx",
        help="Path to the exported ONNX model file."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./qwen2.5_vl_7b_instruct",
        help="Path to the cache directory."
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force running the test on CPU."
    )
    
    args = parser.parse_args()
    
    test_parity(
        hf_model_name=args.hf_model, 
        cache_dir=args.cache_dir,
        onnx_model_path=args.onnx_model, 
        use_gpu=not args.cpu
    )