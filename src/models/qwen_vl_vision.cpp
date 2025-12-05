// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Qwen VL Vision pipeline implementation with optional QNN EP for vision attention stage.

#include "qwen_vl_vision.h"
#include "../generators.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace Generators {

QwenVisionPipeline::QwenVisionPipeline(OrtEnv& env,
                                       const std::string& patch_embed_model,
                                       const std::string& vision_attn_model,
                                       const std::string& patch_merger_model,
                                       int64_t spatial_merge_size,
                                       bool use_qnn_attn,
                                       const std::string& qnn_backend_path,
                                       int64_t patch_size,
                                       int64_t window_size)
    // Match declaration order to avoid MSVC C5038 warning-as-error
    : use_qnn_attn_(use_qnn_attn),
      qnn_backend_path_(qnn_backend_path),
      spatial_merge_size_(spatial_merge_size),
      patch_size_(patch_size),
      window_size_(window_size),
      env_(env) {
  // Convert std::string model paths to ORTCHAR_T for cross-platform (char or wchar_t)
  auto toOrtPath = [](const std::string& s) -> std::basic_string<ORTCHAR_T> {
    return std::basic_string<ORTCHAR_T>(s.begin(), s.end());
  };
  auto pe_path = toOrtPath(patch_embed_model);
  auto attn_path = toOrtPath(vision_attn_model);
  auto merger_path = toOrtPath(patch_merger_model);

  // Patch embed and patch merger sessions (CPU for now)
  patch_embed_session_ = OrtSession::Create(env_, pe_path.c_str(), nullptr);
  patch_merger_session_ = OrtSession::Create(env_, merger_path.c_str(), nullptr);

  if (use_qnn_attn_) {
    // Ensure QNN provider is available
    auto so = OrtSessionOptions::Create();

    so->SetIntraOpNumThreads(2).SetInterOpNumThreads(1);

    // QNN provider options
    std::unordered_map<std::string, std::string> qnn_options = {
        {"backend_path", qnn_backend_path_},
        {"htp_performance_mode", "burst"},
        {"htp_graph_finalization_optimization_mode", "3"},
        {"soc_model", "60"}};

    auto providers = Ort::GetAvailableProviders();
    bool has_qnn = std::find(providers.begin(), providers.end(), std::string("QNNExecutionProvider")) != providers.end();
    if (has_qnn) {
      const char* keys[] = {"backend_path", "htp_performance_mode", "htp_graph_finalization_optimization_mode", "soc_model"};
      const char* values[] = {qnn_backend_path_.c_str(), "burst", "3", "60"};
      so->AppendExecutionProvider("QNNExecutionProvider", keys, values, 4);
    } else {
      // Use registered QNN EP - use GenAI wrapper APIs
      auto ep_devices = GetOrtEnv().GetEpDevices();
      std::vector<const OrtEpDevice*> qnn_devices;
      qnn_devices.reserve(ep_devices.size());

      for (const auto* device : ep_devices) {
        if (device->Name() == "QNNExecutionProvider") {
          qnn_devices.push_back(device);
        }
      }

      if (qnn_devices.empty()) {
        throw std::runtime_error("QNNExecutionProvider requested for vision attention but not registered.");
      }
      so->AppendExecutionProvider_V2(GetOrtEnv(), qnn_devices, qnn_options);
    }

    vision_attn_session_ = OrtSession::Create(env_, attn_path.c_str(), so.get());
  } else {
    vision_attn_session_ = OrtSession::Create(env_, attn_path.c_str(), nullptr);
  }
}

std::unique_ptr<OrtValue> QwenVisionPipeline::CreateTensor(const float* data, size_t count, const std::vector<int64_t>& shape) const {
  auto memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::span<float> data_span(const_cast<float*>(data), count);
  std::span<const int64_t> shape_span(shape.data(), shape.size());
  return OrtValue::CreateTensor<float>(*memory_info, data_span, shape_span);
}

// Removed CreateEmptyTensor (previous implementation returned tensor with dangling backing store).

std::vector<float> QwenVisionPipeline::Run(const float* pixel_data, const std::vector<int64_t>& pixel_shape,
                                           const std::vector<int64_t>& grid_thw) {
  if (!patch_embed_session_ || !vision_attn_session_ || !patch_merger_session_) {
    throw std::runtime_error("Vision pipeline sessions not initialized");
  }

  // Calculate window indices dynamically if grid_thw provided
  if (!grid_thw.empty() && grid_thw.size() == 3) {
    wnd_idx_ = CalculateWindowIndex(grid_thw[0], grid_thw[1], grid_thw[2]);

    // Build reverse index (argsort)
    rev_idx_.resize(wnd_idx_.size());
    std::vector<std::pair<int64_t, size_t>> pairs;
    pairs.reserve(wnd_idx_.size());
    for (size_t i = 0; i < wnd_idx_.size(); ++i) pairs.emplace_back(wnd_idx_[i], i);
    std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return a.first < b.first; });
    for (size_t i = 0; i < pairs.size(); ++i) rev_idx_[i] = static_cast<int64_t>(pairs[i].second);
  }

  size_t pixel_count = 1;
  for (auto d : pixel_shape) pixel_count *= static_cast<size_t>(d);
  auto pixel_tensor = CreateTensor(pixel_data, pixel_count, pixel_shape);

  auto pe_in_name = patch_embed_session_->GetInputName(0);
  const char* pe_input_names[] = {pe_in_name.c_str()};
  OrtValue* pe_inputs[] = {pixel_tensor.get()};

  const int64_t num_patches = pixel_shape[1];
  const int64_t hidden_dim = 1280;
  std::vector<int64_t> pe_out_shape{num_patches, hidden_dim};
  pe_out_buf_.resize(num_patches * hidden_dim);
  auto pe_out_tensor = CreateTensor(pe_out_buf_.data(), pe_out_buf_.size(), pe_out_shape);

  auto pe_out_name = patch_embed_session_->GetOutputName(0);
  const char* pe_output_names[] = {pe_out_name.c_str()};
  OrtValue* pe_outputs[] = {pe_out_tensor.get()};

  patch_embed_session_->Run(nullptr, pe_input_names, pe_inputs, 1, pe_output_names, pe_outputs, 1);

  const int64_t seq_len = num_patches;
  const int64_t window_area = spatial_merge_size_ * spatial_merge_size_;
  const int64_t num_windows = seq_len / window_area;

  // Apply window reordering if indices available
  reordered_buf_.resize(seq_len * hidden_dim);

  if (!wnd_idx_.empty()) {
    // Validate window configuration
    if (seq_len % window_area != 0 || static_cast<int64_t>(wnd_idx_.size()) != num_windows) {
      throw std::runtime_error("Invalid window configuration for vision pipeline");
    }

    // Apply window reordering
    for (int64_t dst_w = 0; dst_w < num_windows; ++dst_w) {
      int64_t src_w = wnd_idx_[dst_w];
      if (src_w < 0 || src_w >= num_windows) throw std::runtime_error("wnd_idx value out of range");
      size_t offset_size = window_area * hidden_dim;
      std::memcpy(reordered_buf_.data() + dst_w * offset_size,
                  pe_out_buf_.data() + src_w * offset_size,
                  offset_size * sizeof(float));
    }
  } else {
    // No window reordering - use sequential order
    std::memcpy(reordered_buf_.data(), pe_out_buf_.data(), seq_len * hidden_dim * sizeof(float));
  }

  // Check if vision_attn session expects a different sequence length (fixed shape model)
  auto attn_input_info = vision_attn_session_->GetInputTypeInfo(0);
  auto& attn_input_tensor_info = attn_input_info->GetTensorTypeAndShapeInfo();
  auto attn_expected_shape = attn_input_tensor_info.GetShape();

  int64_t expected_seq_len = (attn_expected_shape.size() >= 2 && attn_expected_shape[0] > 0) ? attn_expected_shape[0] : seq_len;
  int64_t actual_seq_len = seq_len;  // Mutable copy for padding adjustments

  if (expected_seq_len != seq_len) {
    // Model expects fixed sequence length - need to pad or error
    if (expected_seq_len > seq_len) {
      // Pad the reordered buffer with zeros to match model's expected size
      reordered_buf_.resize(expected_seq_len * hidden_dim, 0.0f);
      actual_seq_len = expected_seq_len;  // Update actual_seq_len for subsequent operations
    } else {
      // Model expects smaller input - this is an error (image too large for fixed-shape model)
      throw std::runtime_error("Vision attention model input size mismatch");
    }
  }

  std::vector<int64_t> attn_shape{actual_seq_len, hidden_dim};
  auto attn_in_tensor = CreateTensor(reordered_buf_.data(), reordered_buf_.size(), attn_shape);
  auto attn_in_name = vision_attn_session_->GetInputName(0);
  const char* attn_input_names[] = {attn_in_name.c_str()};
  OrtValue* attn_inputs[] = {attn_in_tensor.get()};

  attn_out_buf_.resize(actual_seq_len * hidden_dim);
  auto attn_out_tensor = CreateTensor(attn_out_buf_.data(), attn_out_buf_.size(), attn_shape);
  auto attn_out_name = vision_attn_session_->GetOutputName(0);
  const char* attn_output_names[] = {attn_out_name.c_str()};
  OrtValue* attn_outputs[] = {attn_out_tensor.get()};

  vision_attn_session_->Run(nullptr, attn_input_names, attn_inputs, 1, attn_output_names, attn_outputs, 1);

  auto merger_in_tensor = CreateTensor(attn_out_buf_.data(), attn_out_buf_.size(), attn_shape);
  auto merger_in_name = patch_merger_session_->GetInputName(0);
  const char* merger_input_names[] = {merger_in_name.c_str()};
  OrtValue* merger_inputs[] = {merger_in_tensor.get()};

  const int64_t merged_seq_len = actual_seq_len / window_area;  // One token per window after merging
  const int64_t merged_hidden = 3584;
  std::vector<int64_t> merger_shape{merged_seq_len, merged_hidden};
  merger_out_buf_.resize(merged_seq_len * merged_hidden);
  auto merger_out_tensor = CreateTensor(merger_out_buf_.data(), merger_out_buf_.size(), merger_shape);
  auto merger_out_name = patch_merger_session_->GetOutputName(0);
  const char* merger_output_names[] = {merger_out_name.c_str()};
  OrtValue* merger_outputs[] = {merger_out_tensor.get()};

  patch_merger_session_->Run(nullptr, merger_input_names, merger_inputs, 1, merger_output_names, merger_outputs, 1);

  final_embeddings_buf_.resize(merger_out_buf_.size());

  if (!rev_idx_.empty()) {
    // Apply reverse reordering
    if (static_cast<int64_t>(rev_idx_.size()) != num_windows) {
      throw std::runtime_error("Vision pipeline reverse index size mismatch");
    }
    for (int64_t dst_w = 0; dst_w < num_windows; ++dst_w) {
      std::memcpy(final_embeddings_buf_.data() + dst_w * merged_hidden,
                  merger_out_buf_.data() + rev_idx_[dst_w] * merged_hidden,
                  merged_hidden * sizeof(float));
    }
  } else {
    // No reverse reordering - use sequential order
    std::memcpy(final_embeddings_buf_.data(), merger_out_buf_.data(),
                merger_out_buf_.size() * sizeof(float));
  }

  last_seq_len_ = merged_seq_len;
  last_hidden_size_ = merged_hidden;
  return final_embeddings_buf_;
}

// Calculate window indices dynamically based on grid dimensions
// Matches HuggingFace transformers implementation:
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L367
std::vector<int64_t> QwenVisionPipeline::CalculateWindowIndex(int64_t grid_t, int64_t grid_h, int64_t grid_w) {
  // Calculate LLM grid dimensions after spatial merging
  int64_t llm_grid_h = grid_h / spatial_merge_size_;
  int64_t llm_grid_w = grid_w / spatial_merge_size_;

  // Calculate window size at the merged resolution
  int64_t vit_merger_window_size = window_size_ / spatial_merge_size_ / patch_size_;

  // Calculate padding needed to fit into windows
  int64_t pad_h = (vit_merger_window_size - (llm_grid_h % vit_merger_window_size)) % vit_merger_window_size;
  int64_t pad_w = (vit_merger_window_size - (llm_grid_w % vit_merger_window_size)) % vit_merger_window_size;

  int64_t num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
  int64_t num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

  std::vector<int64_t> window_index;
  window_index.reserve(grid_t * llm_grid_h * llm_grid_w);

  // Create initial index grid
  std::vector<int64_t> index(grid_t * (llm_grid_h + pad_h) * (llm_grid_w + pad_w), -100);

  // Fill non-padded positions with sequential indices
  for (int64_t t = 0; t < grid_t; ++t) {
    for (int64_t h = 0; h < llm_grid_h; ++h) {
      for (int64_t w = 0; w < llm_grid_w; ++w) {
        int64_t idx = t * llm_grid_h * llm_grid_w + h * llm_grid_w + w;
        int64_t padded_idx = t * (llm_grid_h + pad_h) * (llm_grid_w + pad_w) + h * (llm_grid_w + pad_w) + w;
        index[padded_idx] = idx;
      }
    }
  }

  // Reshape into windows: (grid_t, num_windows_h, window_size, num_windows_w, window_size)
  // Then permute to (grid_t, num_windows_h, num_windows_w, window_size, window_size)
  // This groups patches by window instead of by spatial position
  for (int64_t t = 0; t < grid_t; ++t) {
    for (int64_t wh = 0; wh < num_windows_h; ++wh) {
      for (int64_t ww = 0; ww < num_windows_w; ++ww) {
        for (int64_t ph = 0; ph < vit_merger_window_size; ++ph) {
          for (int64_t pw = 0; pw < vit_merger_window_size; ++pw) {
            int64_t h = wh * vit_merger_window_size + ph;
            int64_t w = ww * vit_merger_window_size + pw;
            int64_t padded_idx = t * (llm_grid_h + pad_h) * (llm_grid_w + pad_w) + h * (llm_grid_w + pad_w) + w;

            // Only add non-padded indices
            if (index[padded_idx] != -100) {
              window_index.push_back(index[padded_idx]);
            }
          }
        }
      }
    }
  }

  return window_index;
}

}  // namespace Generators
