from onnxruntime import InferenceSession, OrtValue, SessionOptions, get_available_providers
import numpy as np
import torch


def test_paged_model():
  # Create a session with PagedModelExecutionProvider
  options = SessionOptions()
  session = InferenceSession("C:/Users/aciddelgado/onnxruntime-genai/test/test_models/phi3.5paged_7_14_25/model.onnx", options, providers=['CUDAExecutionProvider'])

  # Input params
  batch_size = 1
  total_sequence_length = 276
  sequence_length = 1
  num_tokens = 1
  max_num_blocks = 2
  num_blocks = 2
  block_size = 256
  num_heads = 32
  head_size = 96
  num_layers = 32
  vocab_size = 32064

  # Prepare input data
  input_ids = torch.tensor([79], dtype=torch.int64, device='cuda')
  block_table = torch.tensor([[1, 0]], dtype=torch.int32, device='cuda')
  cumulative_sequence_length = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
  past_seqlens = torch.tensor([total_sequence_length-1], dtype=torch.int32, device='cuda')
  keys, keys_nopage, values, values_nopage = [], [], [], []
  for _ in range(num_layers):
    torch_key = torch.randn(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device='cuda')
    torch_value = torch.randn(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device='cuda')
    keys.append(OrtValue.ortvalue_from_numpy(torch_key.detach().cpu().numpy(), "cuda", 0))
    values.append(OrtValue.ortvalue_from_numpy(torch_value.detach().cpu().numpy(), "cuda", 0))
    key_nopage = torch_key[block_table[0]].reshape(1, -1, num_heads, head_size)
    print(f"key_nopage shape: {key_nopage.shape}")
    value_nopage = torch_value[block_table[0]].reshape(1, -1, num_heads, head_size)
    key_nopage = key_nopage[:, :total_sequence_length]
    print(f"key_nopage after slicing shape: {key_nopage.shape}")
    key_nopage = key_nopage.reshape(batch_size, total_sequence_length, num_heads, head_size).permute(0, 2, 1, 3)
    print(f"key_nopage after reshape and permute shape: {key_nopage.shape}")
    value_nopage = value_nopage[:, :total_sequence_length]
    value_nopage = value_nopage.reshape(batch_size, total_sequence_length, num_heads, head_size).permute(0, 2, 1, 3)
    keys_nopage.append(OrtValue.ortvalue_from_numpy(key_nopage.detach().cpu().numpy(), "cuda", 0))
    values_nopage.append(OrtValue.ortvalue_from_numpy(value_nopage.detach().cpu().numpy(), "cuda", 0))
  
  # IO Bindings
  io_binding = session.io_binding()
  io_binding.bind_cpu_input("input_ids", input_ids.detach().cpu().numpy())
  io_binding.bind_cpu_input("block_table", block_table.detach().cpu().numpy())
  io_binding.bind_cpu_input("cumulative_sequence_length", cumulative_sequence_length.detach().cpu().numpy())
  io_binding.bind_cpu_input("past_seqlens", past_seqlens.detach().cpu().numpy())
  for i in range(num_layers):
    print(f"input pointer for past_key_values.{i}.key: {keys[i].data_ptr()}")
    print(f"input pointer for past_key_values.{i}.value: {values[i].data_ptr()}")
    io_binding.bind_input(f"past_key_values.{i}.key", "cuda", 0, np.float16, keys[i].shape(), keys[i].data_ptr())
    io_binding.bind_input(f"past_key_values.{i}.value", "cuda", 0, np.float16, values[i].shape(), values[i].data_ptr())
  io_binding.bind_output("logits")
  for i in range(num_layers):
    print(f"output pointer for present.{i}.key: {keys[i].data_ptr()}")
    print(f"output pointer for present.{i}.value: {values[i].data_ptr()}")
    io_binding.bind_ortvalue_output(f"present.{i}.key", keys[i])
    io_binding.bind_ortvalue_output(f"present.{i}.value", values[i])

  # Run inference
  outputs = session.run_with_iobinding(io_binding)
  outputs = io_binding.copy_outputs_to_cpu()
  # print(f"Outputs: {outputs}")

  # Check output shape
  logits = outputs[0]
  assert logits.shape == (num_tokens, vocab_size), f"Expected output shape {(num_tokens, vocab_size)}, but got {logits.shape}"

  print("Paged model test passed successfully.")

  session = InferenceSession("C:/Users/aciddelgado/onnxruntime-genai/test/test_models/phi-3.5-mini-12_05_24/gpu/gpu-int4-awq-block-128/model.onnx", options, providers=['CUDAExecutionProvider'])

  # Prepare input data for the second model
  input_ids = torch.tensor([[79]], dtype=torch.int64, device='cuda')
  attention_mask = torch.zeros(1, total_sequence_length, dtype=torch.int64, device='cuda')
  attention_mask[0, -1] = 1  # Set the last position to 1
  position_ids = torch.tensor([[total_sequence_length - 1]], dtype=torch.int64, device='cuda')
  
  io_binding = session.io_binding()
  io_binding.bind_cpu_input("input_ids", input_ids.detach().cpu().numpy())
  io_binding.bind_cpu_input("attention_mask", attention_mask.detach().cpu().numpy())
  io_binding.bind_cpu_input("position_ids", position_ids.detach().cpu().numpy())
  io_binding.bind_output("logits")
  for i in range(num_layers):
    io_binding.bind_input(f"past_key_values.{i}.key", "cuda", 0, np.float16, keys_nopage[i].shape(), keys_nopage[i].data_ptr())
    io_binding.bind_input(f"past_key_values.{i}.value", "cuda", 0, np.float16, values_nopage[i].shape(), values_nopage[i].data_ptr())
    io_binding.bind_ortvalue_output(f"present.{i}.key", keys_nopage[i])
    io_binding.bind_ortvalue_output(f"present.{i}.value", values_nopage[i])
  outputs = session.run_with_iobinding(io_binding)
  outputs = io_binding.copy_outputs_to_cpu()

  logits_nopage = outputs[0]
  assert logits_nopage.shape == (batch_size, sequence_length, vocab_size), f"Expected output shape {(batch_size, sequence_length, vocab_size)}, but got {logits_nopage.shape}"
  print("Second model test passed successfully.")

  # Compare logits and logits_nopage
  assert np.allclose(logits, logits_nopage[0], atol=1e-3), "Logits from paged model and nopaged model do not match."
  

if __name__ == "__main__":
  test_paged_model()
  print("All tests passed.")