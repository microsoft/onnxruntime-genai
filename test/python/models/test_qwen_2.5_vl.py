# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import argparse

import numpy as np
import onnxruntime as ort
import torch
from onnx import TensorProto
from transformers import Qwen2_5_VLForConditionalGeneration


def torch_dtype_to_onnx_tensor_proto(dtype: torch.dtype) -> int:
    """Maps torch.dtype to onnx.TensorProto.DataType"""
    if dtype == torch.float32:
        return TensorProto.FLOAT
    if dtype == torch.float16:
        return TensorProto.FLOAT16
    if dtype == torch.bfloat16:
        return TensorProto.BFLOAT16
    if dtype == torch.int64:
        return TensorProto.INT64
    if dtype == torch.int32:
        return TensorProto.INT32
    if dtype == torch.bool:
        return TensorProto.BOOL
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def to_numpy(tensor):
    """Move tensor to CPU and convert to numpy, handling bf16."""
    if tensor.dtype == torch.bfloat16:
        # NumPy doesn't support bfloat16, so cast to float32 first
        return tensor.detach().cpu().to(torch.float32).numpy()
    return tensor.detach().cpu().numpy()


def compare_outputs(
    hf_logits: torch.Tensor,
    ort_logits: torch.Tensor,  # Changed to torch.Tensor
    hf_presents: list[tuple[torch.Tensor, torch.Tensor]],
    ort_presents: list[torch.Tensor],  # Changed to list[torch.Tensor]
    step_name: str,
    rtol: float,
    atol: float,
):
    """Compares logits and KV cache outputs using numpy."""

    print(f"--- Comparing {step_name} Logits ---")

    # We can use to_numpy safely here because we'll compare fp32 vs fp32
    # or (bf16->fp32) vs (bf16->fp32)
    np.testing.assert_allclose(to_numpy(hf_logits), to_numpy(ort_logits), rtol=rtol, atol=atol)
    print("Logits: PASS")

    print(f"\n--- Comparing {step_name} KV Cache ---")
    # hf_presents is now a list of tuples: [(k0, v0), (k1, v1), ...]
    # Flatten it to a list: [k0, v0, k1, v1, ...]
    hf_presents_list = [t for layer_kv in hf_presents for t in layer_kv]

    assert len(hf_presents_list) == len(ort_presents), (
        f"HF presents count ({len(hf_presents_list)}) != ORT presents count ({len(ort_presents)})"
    )

    for i in range(len(hf_presents_list)):
        hf_tensor = hf_presents_list[i]
        ort_tensor = ort_presents[i]

        np.testing.assert_allclose(to_numpy(hf_tensor), to_numpy(ort_tensor), rtol=rtol, atol=atol)
    print(f"KV Cache (all {len(hf_presents_list)} tensors): PASS")
    print(f"\n✅ {step_name} Parity Test Passed!\n")


def ort_io_binding_helper(
    sess: ort.InferenceSession,
    input_tensors: dict[str, torch.Tensor],
    output_tensors: dict[str, torch.Tensor],
    device: str,
) -> None:
    """
    Binds torch tensors to an ONNX Runtime IOBinding object and runs the session.
    Tensors must be on the correct device (e.g., 'cuda:0').
    """
    bind = sess.io_binding()

    # Get device type and index for ORT
    ort_device = device.split(":")[0]
    ort_device_id = 0
    if ":" in device:
        ort_device_id = int(device.split(":")[1])

    for name, tensor in input_tensors.items():
        if not tensor.is_contiguous():
            raise RuntimeError(f"Input tensor {name} is not contiguous.")

        bind.bind_input(
            name,
            ort_device,
            ort_device_id,
            torch_dtype_to_onnx_tensor_proto(tensor.dtype),
            tensor.shape,
            tensor.data_ptr(),
        )

    for name, tensor in output_tensors.items():
        if not tensor.is_contiguous():
            raise RuntimeError(f"Output tensor {name} is not contiguous.")

        bind.bind_output(
            name,
            ort_device,
            ort_device_id,
            torch_dtype_to_onnx_tensor_proto(tensor.dtype),
            tensor.shape,
            tensor.data_ptr(),
        )

    sess.run_with_iobinding(bind)


def test_parity(
    hf_model_name: str,
    cache_dir: str,
    onnx_model_path: str,
    use_gpu: bool,
    use_bf16: bool,
    use_fp16: bool,
):
    """
    Runs a two-step (prefill and decode) parity test between the Hugging Face
    and ONNX models.
    """

    print(f"Loading Hugging Face model: {hf_model_name}")
    print("This requires `trust_remote_code=True`.")

    if not use_gpu:
        print("ERROR: This test script now requires a GPU (`--cpu` is not supported) due to IOBinding.")
        return

    device = "cuda:0"  # IOBinding needs the specific device ID

    if use_bf16:
        torch_dtype = torch.bfloat16
        # Standard BF16 tolerances
        rtol, atol = 2e-1, 1
    elif use_fp16:
        torch_dtype = torch.float16
        # Standard FP16 tolerances
        rtol, atol = 1e-1, 5e-1
    else:
        torch_dtype = torch.float32
        # Standard FP32 tolerances
        rtol, atol = 1e-1, 1e-1

    # The builder script (base.Model) upcasts logits to float32
    # ONLY when the io_dtype is bfloat16.
    # For FP16 or FP32, it keeps the original dtype.
    logits_dtype = torch.float32 if use_bf16 else torch_dtype

    print(f"Allocating ONNX logits output buffer with dtype: {logits_dtype}")

    hf_full_model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        .to(device)
        .eval()
    )

    # The ONNX model is *only* the language_model component
    hf_text_model = hf_full_model.language_model
    config = hf_text_model.config

    # Get model parameters
    batch_size = 1
    prefill_len = 10
    decode_len = 1
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    vocab_size = config.vocab_size

    print("\n--- Model Parameters ---")
    print(f"Device: {device}")
    print(f"DType: {torch_dtype}")
    print(f"RTOL: {rtol}, ATOL: {atol}")
    print(f"Layers: {num_layers}")
    print(f"Hidden Size: {hidden_size}")
    print(f"KV Heads: {num_kv_heads}")
    print(f"Head Dim: {head_dim}")
    print("------------------------\n")

    print(f"Loading ONNX model: {onnx_model_path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_model_path, providers=providers)

    # =================================================================
    # 1. PREFILL STEP
    # =================================================================
    print(f"Running Prefill Step (Sequence Length = {prefill_len})...")

    # --- Create HF/Torch Inputs ---
    # Use randn (normal distribution) scaled down for better stability in FP16
    # inputs_embeds are normally centered around 0, unlike rand which is [0, 1]
    inputs_embeds_prefill = (
        torch.randn((batch_size, prefill_len, hidden_size), dtype=torch_dtype, device=device) * 0.001
    )

    # Qwen2.5-VL uses 3D position IDs (temporal, height, width).
    # For text tokens, all three dimensions typically use the same sequence index.
    pos_ids_1d_prefill = torch.arange(prefill_len, device=device).expand(batch_size, -1)
    position_ids_prefill = pos_ids_1d_prefill.unsqueeze(0).expand(3, -1, -1).contiguous()

    attention_mask_prefill = torch.ones((batch_size, prefill_len), dtype=torch.int64, device=device)

    cache_position_prefill = torch.arange(prefill_len, device=device)

    # --- Create ONNX Input Tensors (on device) ---
    ort_inputs_prefill = {
        "inputs_embeds": inputs_embeds_prefill,
        "position_ids": position_ids_prefill,
        "attention_mask": attention_mask_prefill,
    }

    # Create dummy pasts with 0 sequence length
    past_shape = (batch_size, num_kv_heads, 0, head_dim)
    dummy_past = torch.empty(past_shape, dtype=torch_dtype, device=device)
    for i in range(num_layers):
        ort_inputs_prefill[f"past_key_values.{i}.key"] = dummy_past
        ort_inputs_prefill[f"past_key_values.{i}.value"] = dummy_past

    # --- Create ONNX Output Tensors (on device) ---
    ort_logits_prefill = torch.empty((batch_size, prefill_len, vocab_size), dtype=logits_dtype, device=device)
    ort_presents_prefill = []
    ort_outputs_prefill = {"logits": ort_logits_prefill}
    present_shape = (batch_size, num_kv_heads, prefill_len, head_dim)

    for i in range(num_layers):
        ort_present_k = torch.empty(present_shape, dtype=torch_dtype, device=device)
        ort_present_v = torch.empty(present_shape, dtype=torch_dtype, device=device)
        ort_outputs_prefill[f"present.{i}.key"] = ort_present_k
        ort_outputs_prefill[f"present.{i}.value"] = ort_present_v
        ort_presents_prefill.extend([ort_present_k, ort_present_v])

    # --- Run HF Model ---
    with torch.no_grad():
        hf_outputs_prefill = hf_text_model(
            inputs_embeds=inputs_embeds_prefill,
            position_ids=position_ids_prefill,
            attention_mask=attention_mask_prefill,
            past_key_values=None,
            cache_position=cache_position_prefill,
            return_dict=True,
            use_cache=True,
        )

    # --- Run ONNX Model with IOBinding ---
    ort_io_binding_helper(sess, ort_inputs_prefill, ort_outputs_prefill, device)

    # --- Compare Prefill ---
    hf_logits_prefill = hf_full_model.lm_head(hf_outputs_prefill.last_hidden_state)
    hf_presents_prefill = hf_outputs_prefill.past_key_values

    compare_outputs(
        hf_logits_prefill,
        ort_logits_prefill,  # This is the tensor we pre-allocated
        hf_presents_prefill,
        ort_presents_prefill,  # This is the list of tensors we pre-allocated
        step_name="Prefill",
        rtol=rtol,
        atol=atol,
    )

    # =================================================================
    # 2. DECODE STEP
    # =================================================================
    print(f"Running Decode Step (Sequence Length = {decode_len})...")

    # --- Create HF/Torch Inputs ---
    # Use randn (normal distribution) scaled down
    inputs_embeds_decode = torch.randn((batch_size, decode_len, hidden_size), dtype=torch_dtype, device=device) * 0.001

    # Position IDs continue from prefill length
    pos_ids_1d_decode = torch.tensor([[prefill_len]], dtype=torch.int64, device=device)
    position_ids_decode = pos_ids_1d_decode.unsqueeze(0).expand(3, -1, -1).contiguous()

    attention_mask_decode = torch.ones((batch_size, prefill_len + decode_len), dtype=torch.int64, device=device)

    cache_position_decode = torch.tensor([prefill_len], device=device)

    # Use the KV cache from the HF prefill run
    hf_past_key_values = hf_outputs_prefill.past_key_values

    # --- Create ONNX Input Tensors (on device) ---
    ort_inputs_decode = {
        "inputs_embeds": inputs_embeds_decode,
        "position_ids": position_ids_decode,
        "attention_mask": attention_mask_decode,
    }

    # Use the KV cache from the ONNX prefill run (these are already torch tensors)
    for i in range(num_layers):
        ort_inputs_decode[f"past_key_values.{i}.key"] = ort_presents_prefill[i * 2]
        ort_inputs_decode[f"past_key_values.{i}.value"] = ort_presents_prefill[i * 2 + 1]

    # --- Create ONNX Output Tensors (on device) ---
    # --- FIX: Logits from bf16 ONNX model are intentionally float32 for accuracy ---
    ort_logits_decode = torch.empty((batch_size, decode_len, vocab_size), dtype=logits_dtype, device=device)
    ort_presents_decode = []
    ort_outputs_decode = {"logits": ort_logits_decode}
    present_shape_decode = (
        batch_size,
        num_kv_heads,
        prefill_len + decode_len,
        head_dim,
    )

    for i in range(num_layers):
        ort_present_k = torch.empty(present_shape_decode, dtype=torch_dtype, device=device)
        ort_present_v = torch.empty(present_shape_decode, dtype=torch_dtype, device=device)
        ort_outputs_decode[f"present.{i}.key"] = ort_present_k
        ort_outputs_decode[f"present.{i}.value"] = ort_present_v
        ort_presents_decode.extend([ort_present_k, ort_present_v])

    # --- Run HF Model ---
    with torch.no_grad():
        hf_outputs_decode = hf_text_model(
            inputs_embeds=inputs_embeds_decode,
            position_ids=position_ids_decode,
            attention_mask=attention_mask_decode,
            past_key_values=hf_past_key_values,
            cache_position=cache_position_decode,
            return_dict=True,
            use_cache=True,
        )

    # --- Run ONNX Model with IOBinding ---
    ort_io_binding_helper(sess, ort_inputs_decode, ort_outputs_decode, device)

    # --- Compare Decode ---
    hf_logits_decode = hf_full_model.lm_head(hf_outputs_decode.last_hidden_state)
    hf_presents_decode = hf_outputs_decode.past_key_values

    compare_outputs(
        hf_logits_decode,
        ort_logits_decode,
        hf_presents_decode,
        ort_presents_decode,
        step_name="Decode",
        rtol=rtol,
        atol=atol,
    )

    print("=" * 30)
    print("🎉 All Parity Tests Passed! 🎉")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parity test for Qwen 2.5 VL ONNX model.")
    parser.add_argument(
        "--hf_model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Path or name of the Hugging Face model.",
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        required=True,
        help="Path to the exported ONNX model file.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./qwen2.5_vl_7b_instruct",
        help="Path to the cache directory.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force running the test on CPU (Not supported with IOBinding).",
    )

    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision.")

    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision.")

    args = parser.parse_args()

    if args.cpu and (args.bf16 or args.fp16):
        print("Warning: Cannot run bf16/fp16 on CPU. Forcing float32.")
        args.bf16 = False
        args.fp16 = False

    if args.cpu:
        print("Warning: CPU testing with IOBinding is not set up. Forcing GPU.")
        # This script is now GPU-only

    test_parity(
        hf_model_name=args.hf_model,
        cache_dir=args.cache_dir,
        onnx_model_path=args.onnx_model,
        use_gpu=True,  # Forcing GPU
        use_bf16=args.bf16,
        use_fp16=args.fp16,
    )
