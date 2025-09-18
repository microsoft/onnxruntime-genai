from onnxruntime import InferenceSession, OrtValue, SessionOptions, get_available_providers
import numpy as np
import torch


def test_paged_model():
  # Create a session with PagedModelExecutionProvider
  options = SessionOptions()
  session = InferenceSession("C:\\Users\\aciddelgado\\onnxruntime-genai\\test\\test_models\\phi-3.5-mini\\cuda-fp16-complete-paged\\model.onnx", options, providers=['CUDAExecutionProvider'])
  # session = InferenceSession("C:/Users/aciddelgado/onnxruntime-genai/test/test_models/phi3.5paged_8_11_25/model.onnx", options, providers=['CUDAExecutionProvider'])

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
  block_table = torch.tensor([[0, 1]], dtype=torch.int32, device='cuda')
  cumulative_sequence_length = torch.tensor([0, sequence_length], dtype=torch.int32, device='cuda')
  past_seqlens = torch.tensor([total_sequence_length-sequence_length], dtype=torch.int32, device='cuda')
  keys, keys_gqa, values, values_gqa = [], [], [], []
  for _ in range(num_layers):
    torch_key = torch.randn(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device='cuda')
    keys.append(OrtValue.ortvalue_from_numpy(torch_key.detach().cpu().numpy(), "cuda", 0))
    key_gqa = torch_key[[block_table[0]]].reshape(1, -1, num_heads, head_size) # TODO(aciddelgado): hack... bs 1 only
    key_gqa = key_gqa[:, :total_sequence_length]
    key_gqa = key_gqa.permute(0, 2, 1, 3)
    assert key_gqa.shape == (batch_size, num_heads, total_sequence_length, head_size), f"Expected shape {(batch_size, num_heads, total_sequence_length, head_size)}, but got {key_gqa.shape}"
    keys_gqa.append(OrtValue.ortvalue_from_numpy(key_gqa.detach().cpu().numpy(), "cuda", 0))
    torch_value = torch.randn(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device='cuda')
    values.append(OrtValue.ortvalue_from_numpy(torch_value.detach().cpu().numpy(), "cuda", 0))
    value_gqa = torch_value[[block_table[0]]].reshape(1, -1, num_heads, head_size)
    value_gqa = value_gqa[:, :total_sequence_length]
    value_gqa = value_gqa.permute(0, 2, 1, 3)
    assert value_gqa.shape == (batch_size, num_heads, total_sequence_length, head_size), f"Expected shape {(batch_size, num_heads, total_sequence_length, head_size)}, but got {value_gqa.shape}"
    values_gqa.append(OrtValue.ortvalue_from_numpy(value_gqa.detach().cpu().numpy(), "cuda", 0))

    # torch_key = torch.randn(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device='cuda')
    # torch_value = torch.randn(num_blocks, block_size, num_heads, head_size, dtype=torch.float16, device='cuda')
    # keys.append(OrtValue.ortvalue_from_numpy(torch_key.detach().cpu().numpy(), "cuda", 0))
    # values.append(OrtValue.ortvalue_from_numpy(torch_value.detach().cpu().numpy(), "cuda", 0))
    # key_gqa = torch_key[block_table[0]].reshape(1, -1, num_heads, head_size)
    # print(f"key_gqa shape: {key_gqa.shape}")
    # value_gqa = torch_value[block_table[0]].reshape(1, -1, num_heads, head_size)
    # key_gqa = key_gqa[:, :total_sequence_length]
    # print(f"key_gqa after slicing shape: {key_gqa.shape}")
    # key_gqa = key_gqa.reshape(batch_size, total_sequence_length, num_heads, head_size).permute(0, 2, 1, 3)
    # print(f"key_gqa after reshape and permute shape: {key_gqa.shape}")
    # value_gqa = value_gqa[:, :total_sequence_length]
    # value_gqa = value_gqa.reshape(batch_size, total_sequence_length, num_heads, head_size).permute(0, 2, 1, 3)
    # keys_gqa.append(OrtValue.ortvalue_from_numpy(key_gqa.detach().cpu().numpy(), "cuda", 0))
    # values_gqa.append(OrtValue.ortvalue_from_numpy(value_gqa.detach().cpu().numpy(), "cuda", 0))
  
  # IO Bindings
  io_binding = session.io_binding()
  io_binding.bind_cpu_input("input_ids", input_ids.detach().cpu().numpy())
  io_binding.bind_cpu_input("block_table", block_table.detach().cpu().numpy())
  io_binding.bind_cpu_input("cumulative_sequence_length", cumulative_sequence_length.detach().cpu().numpy())
  io_binding.bind_cpu_input("past_seqlens", past_seqlens.detach().cpu().numpy())
  for i in range(num_layers):
    io_binding.bind_input(f"past_key_values.{i}.key", "cuda", 0, np.float16, keys[i].shape(), keys[i].data_ptr())
    io_binding.bind_input(f"past_key_values.{i}.value", "cuda", 0, np.float16, values[i].shape(), values[i].data_ptr())
  logits_shape = (num_tokens, vocab_size)
  logits = torch.empty(logits_shape, dtype=torch.float16, device='cuda')
  logits_ortvalue = OrtValue.ortvalue_from_numpy(logits.detach().cpu().numpy(), "cuda", 0)
  io_binding.bind_ortvalue_output("logits", logits_ortvalue)
  for i in range(num_layers):
    io_binding.bind_ortvalue_output(f"present.{i}.key", keys[i])
    io_binding.bind_ortvalue_output(f"present.{i}.value", values[i])

  # Run inference
  outputs = session.run_with_iobinding(io_binding)
  outputs = io_binding.copy_outputs_to_cpu()

  # Check output shape
  logits = outputs[0]
  assert logits.shape == (num_tokens, vocab_size), f"Expected output shape {(num_tokens, vocab_size)}, but got {logits.shape}"

  print("Paged model ran successfully.")

  session = InferenceSession("C:\\Users\\aciddelgado\\onnxruntime-genai\\test\\test_models\\phi-3.5-mini\\cuda-fp16-complete\\model.onnx", options, providers=['CUDAExecutionProvider'])
  # session = InferenceSession("C:/Users/aciddelgado/onnxruntime-genai/test/test_models/phi-3.5-mini-12_05_24/gpu/gpu-int4-awq-block-128/model.onnx", options, providers=['CUDAExecutionProvider'])

  # Prepare input data for the second model
  input_ids = torch.tensor([[79]], dtype=torch.int64, device='cuda')
  attention_mask = torch.ones(batch_size, total_sequence_length, dtype=torch.int64, device='cuda')
  # attention_mask[0, -1] = 1  # Set the last position to 1
  # position_ids = torch.tensor([[total_sequence_length - 1]], dtype=torch.int64, device='cuda')
  
  io_binding = session.io_binding()
  io_binding.bind_cpu_input("input_ids", input_ids.detach().cpu().numpy())
  io_binding.bind_cpu_input("attention_mask", attention_mask.detach().cpu().numpy())
  # io_binding.bind_cpu_input("position_ids", position_ids.detach().cpu().numpy())
  io_binding.bind_output("logits")
  for i in range(num_layers):
    io_binding.bind_input(f"past_key_values.{i}.key", "cuda", 0, np.float16, keys_gqa[i].shape(), keys_gqa[i].data_ptr())
    io_binding.bind_input(f"past_key_values.{i}.value", "cuda", 0, np.float16, values_gqa[i].shape(), values_gqa[i].data_ptr())
    io_binding.bind_ortvalue_output(f"present.{i}.key", keys_gqa[i])
    io_binding.bind_ortvalue_output(f"present.{i}.value", values_gqa[i])
  outputs = session.run_with_iobinding(io_binding)
  outputs = io_binding.copy_outputs_to_cpu()

  logits_gqa = outputs[0]
  assert logits_gqa.shape == (batch_size, sequence_length, vocab_size), f"Expected output shape {(batch_size, sequence_length, vocab_size)}, but got {logits_gqa.shape}"
  print("GQA model passed successfully.")

  # Compare first present key between paged model and gqa model
  present_key_gqa = outputs[1]
  # print(f"Present key paged shape: {present_key_paged.shape}, Present key gqa shape: {present_key_gqa.shape}")
  # print(f"Difference between paged and gqa present key: {present_key_paged[1, total_sequence_length-block_size-1, 4, :50] - present_key_gqa[0, 4, total_sequence_length-1, :50]}")

  # Compare top logit indices
  top_logit_indices = np.argsort(logits, axis=1)[:, -32:][:, ::-1]
  top_logit_indices_gqa = np.argsort(logits_gqa[0], axis=1)[:, -32:][:, ::-1]
  print(f"Top logit indices paged: {top_logit_indices}")
  print(f"Top logit indices gqa: {top_logit_indices_gqa}")
  assert np.array_equal(top_logit_indices, top_logit_indices_gqa), "Top logit indices from paged model and gqa model do not match."

  # Compare logits and logits_gqa
  print(f"Logits - logits_gqa sample: {logits[0, :32] - logits_gqa[0, 0, :32]}")
  assert np.allclose(logits, logits_gqa[0], atol=1e-3), "Logits from paged model and gqa model do not match."

if __name__ == "__main__":
  test_paged_model()
  print("All tests passed.")