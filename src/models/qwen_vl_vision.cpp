// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Qwen VL Vision pipeline implementation with optional QNN EP for vision attention stage.

#include "qwen_vl_vision.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace Generators {

// Minimal .npy reader for 1D integer arrays.
// Only handles C-order, little-endian, shape (N,), for dtypes '<i4' or '<i8'.
std::vector<int64_t> Load1DNpyIndices(const std::string& file_path) {
  std::ifstream fin(file_path, std::ios::binary);
  if (!fin) throw std::runtime_error("Failed to open npy file: " + file_path);

  // Read magic string
  char magic[6];
  fin.read(magic, 6);
  if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("Invalid npy header (magic mismatch) for: " + file_path);
  }
  // Version
  unsigned char ver_major; unsigned char ver_minor;
  fin.read(reinterpret_cast<char*>(&ver_major), 1);
  fin.read(reinterpret_cast<char*>(&ver_minor), 1);
  uint16_t header_len_le;
  fin.read(reinterpret_cast<char*>(&header_len_le), 2); // little endian
  const uint16_t header_len = header_len_le;
  std::string header(header_len, '\0');
  fin.read(header.data(), header_len);

  auto find_field = [&](const std::string& key) {
    auto pos = header.find(key);
    if (pos == std::string::npos) return std::string();
    return header.substr(pos, header.size() - pos);
  };

  // dtype
  auto descr_pos = header.find("'descr':");
  if (descr_pos == std::string::npos) throw std::runtime_error("Missing 'descr' in npy header");
  auto descr_start = header.find("'", descr_pos + 8);
  auto descr_end = header.find("'", descr_start + 1);
  std::string dtype = header.substr(descr_start + 1, descr_end - descr_start - 1);
  bool is_int32 = (dtype == "<i4");
  bool is_int64 = (dtype == "<i8");
  if (!is_int32 && !is_int64) throw std::runtime_error("Unsupported dtype in npy (expected <i4 or <i8): " + dtype);

  auto shape_pos = header.find("'shape':");
  if (shape_pos == std::string::npos) throw std::runtime_error("Missing 'shape' in npy header");
  auto paren_start = header.find("(", shape_pos);
  auto paren_end = header.find(")", paren_start);
  std::string shape_str = header.substr(paren_start + 1, paren_end - paren_start - 1);
  // shape like "1234," or "1234" depending on version
  shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());
  if (shape_str.empty()) throw std::runtime_error("Empty shape in npy header");
  if (shape_str.back() == ',') shape_str.pop_back();
  int64_t N = std::stoll(shape_str);
  if (N <= 0) throw std::runtime_error("Invalid shape size in npy header");

  std::vector<int64_t> result;
  result.resize(static_cast<size_t>(N));

  if (is_int32) {
    std::vector<int32_t> tmp(N);
    fin.read(reinterpret_cast<char*>(tmp.data()), N * sizeof(int32_t));
    if (fin.gcount() != static_cast<std::streamsize>(N * sizeof(int32_t))) throw std::runtime_error("Unexpected EOF reading npy data");
    for (int64_t i = 0; i < N; ++i) result[static_cast<size_t>(i)] = static_cast<int64_t>(tmp[static_cast<size_t>(i)]);
  } else {
    fin.read(reinterpret_cast<char*>(result.data()), N * sizeof(int64_t));
    if (fin.gcount() != static_cast<std::streamsize>(N * sizeof(int64_t))) throw std::runtime_error("Unexpected EOF reading npy data");
  }
  return result;
}

QwenVisionPipeline::QwenVisionPipeline(OrtEnv& env,
                                       const std::string& patch_embed_model,
                                       const std::string& vision_attn_model,
                                       const std::string& patch_merger_model,
                                       int64_t spatial_merge_size,
                                       const std::string& wnd_idx_path,
                                       bool use_qnn_attn,
                                       const std::string& qnn_backend_path)
  // Match declaration order to avoid MSVC C5038 warning-as-error
  : use_qnn_attn_(use_qnn_attn),
    qnn_backend_path_(qnn_backend_path),
    spatial_merge_size_(spatial_merge_size),
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
    auto providers = Ort::GetAvailableProviders();
    bool has_qnn = std::find(providers.begin(), providers.end(), std::string("QNNExecutionProvider")) != providers.end();
    if (!has_qnn) {
      throw std::runtime_error("QNNExecutionProvider requested for vision attention but not available in this build");
    }
    auto so = OrtSessionOptions::Create();
    so->SetIntraOpNumThreads(2).SetInterOpNumThreads(1);
    // QNN provider options
    const char* keys[] = {"backend_path", "htp_performance_mode", "htp_graph_finalization_optimization_mode", "soc_model"};
    const char* values[] = { qnn_backend_path_.c_str(), "burst", "3", "60" };
    so->AppendExecutionProvider("QNNExecutionProvider", keys, values, 4);
    vision_attn_session_ = OrtSession::Create(env_, attn_path.c_str(), so.get());
  } else {
    vision_attn_session_ = OrtSession::Create(env_, attn_path.c_str(), nullptr);
  }

  wnd_idx_ = Load1DNpyIndices(wnd_idx_path);
  // Build reverse index (argsort)
  rev_idx_.resize(wnd_idx_.size());
  std::vector<std::pair<int64_t, size_t>> pairs;
  pairs.reserve(wnd_idx_.size());
  for (size_t i = 0; i < wnd_idx_.size(); ++i) pairs.emplace_back(wnd_idx_[i], i);
  std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b){ return a.first < b.first; });
  for (size_t i = 0; i < pairs.size(); ++i) rev_idx_[i] = static_cast<int64_t>(pairs[i].second);
}

std::unique_ptr<OrtValue> QwenVisionPipeline::CreateTensor(const float* data, size_t count, const std::vector<int64_t>& shape) const {
  auto memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::span<float> data_span(const_cast<float*>(data), count);
  std::span<const int64_t> shape_span(shape.data(), shape.size());
  return OrtValue::CreateTensor<float>(*memory_info, data_span, shape_span);
}

// Removed CreateEmptyTensor (previous implementation returned tensor with dangling backing store).

std::vector<float> QwenVisionPipeline::Run(const float* pixel_data, const std::vector<int64_t>& pixel_shape) {
  if (!patch_embed_session_ || !vision_attn_session_ || !patch_merger_session_) {
    throw std::runtime_error("Vision pipeline sessions not initialized");
  }
  // Create input tensor for patch embed
  size_t pixel_count = 1;
  for (auto d : pixel_shape) pixel_count *= static_cast<size_t>(d);
  
  auto pixel_tensor = CreateTensor(pixel_data, pixel_count, pixel_shape);
  
  const char* pe_input_names[] = {"pixel_values"};
  OrtValue* pe_inputs[] = { pixel_tensor.get() };

  // Compute expected output shape based on input
  // Input: [batch=1, num_patches, patch_dim]
  // Output: [num_patches, hidden_dim=1280]
  int64_t num_patches = pixel_shape[1];  // 1972
  int64_t hidden_dim = 1280;  // Qwen2.5-VL hidden dimension
  std::vector<int64_t> pe_out_shape_vec{num_patches, hidden_dim};
  size_t pe_out_count = static_cast<size_t>(num_patches * hidden_dim);
  
  std::vector<float> pe_out_buf(pe_out_count);
  auto pe_out_tensor = CreateTensor(pe_out_buf.data(), pe_out_count, pe_out_shape_vec);
  
  // Prepare output name
  auto pe_out_name_str = patch_embed_session_->GetOutputName(0);
  const char* pe_output_names[] = { pe_out_name_str.c_str() };
  OrtValue* pe_outputs[] = { pe_out_tensor.get() };

  patch_embed_session_->Run(nullptr, pe_input_names, pe_inputs, 1, pe_output_names, pe_outputs, 1);

  // Debug: Log patch_embed output
  float min_pe = pe_out_buf[0], max_pe = pe_out_buf[0], sum_pe = 0.0f;
  for (const auto& val : pe_out_buf) {
    min_pe = std::min(min_pe, val);
    max_pe = std::max(max_pe, val);
    sum_pe += val;
  }

  // hidden now in pe_out_buf with shape [seq_len, hidden_size]
  int64_t seq_len = pe_out_shape_vec[0];
  int64_t hidden_size = pe_out_shape_vec[1];
  int64_t window_area = spatial_merge_size_ * spatial_merge_size_;
  if (seq_len % window_area != 0) {
    throw std::runtime_error("Sequence length not divisible by spatial_merge_size^2 in vision pipeline");
  }
  int64_t num_windows = seq_len / window_area;
  // Reshape logically: [num_windows, window_area, hidden_size] then reorder by wnd_idx_
  if (static_cast<int64_t>(wnd_idx_.size()) != num_windows) {
    throw std::runtime_error("wnd_idx size does not match number of windows");
  }

  // Temporary buffer for reordered hidden
  std::vector<float> reordered(seq_len * hidden_size);
  // For each window index w: copy its window_area * hidden_size block in order
  for (int64_t dst_w = 0; dst_w < num_windows; ++dst_w) {
    int64_t src_w = wnd_idx_[dst_w];
    if (src_w < 0 || src_w >= num_windows) throw std::runtime_error("wnd_idx value out of range");
    // source offset in original flattened: src_w * window_area * hidden_size
    size_t src_offset = static_cast<size_t>(src_w) * static_cast<size_t>(window_area) * static_cast<size_t>(hidden_size);
    size_t dst_offset = static_cast<size_t>(dst_w) * static_cast<size_t>(window_area) * static_cast<size_t>(hidden_size);
    std::memcpy(reordered.data() + dst_offset, pe_out_buf.data() + src_offset,
                window_area * static_cast<size_t>(hidden_size) * sizeof(float));
  }

  float min_wnd = reordered[0], max_wnd = reordered[0], sum_wnd = 0.0f;
  for (const auto& val : reordered) {
    min_wnd = std::min(min_wnd, val);
    max_wnd = std::max(max_wnd, val);
    sum_wnd += val;
  }

  // Flatten reordered is still [seq_len, hidden_size]
  std::vector<int64_t> attn_in_shape{seq_len, hidden_size};
  auto attn_in_tensor = CreateTensor(reordered.data(), reordered.size(), attn_in_shape);
  const char* attn_input_names[] = {"hidden"};
  OrtValue* attn_inputs[] = { attn_in_tensor.get() };

  // Prepare attention output - shape should be same as input
  std::vector<int64_t> attn_out_shape_vec{seq_len, hidden_size};
  size_t attn_out_count = static_cast<size_t>(seq_len * hidden_size);
  std::vector<float> attn_out_buf(attn_out_count);
  auto attn_out_tensor = CreateTensor(attn_out_buf.data(), attn_out_count, attn_out_shape_vec);
  auto attn_out_name_str = vision_attn_session_->GetOutputName(0);
  const char* attn_output_names[] = { attn_out_name_str.c_str() };
  OrtValue* attn_outputs[] = { attn_out_tensor.get() };
  
  vision_attn_session_->Run(nullptr, attn_input_names, attn_inputs, 1, attn_output_names, attn_outputs, 1);

  float min_attn = attn_out_buf[0], max_attn = attn_out_buf[0], sum_attn = 0.0f;
  for (const auto& val : attn_out_buf) {
    min_attn = std::min(min_attn, val);
    max_attn = std::max(max_attn, val);
    sum_attn += val;
  }
  // Merger input (attention output)
  auto merger_in_tensor = CreateTensor(attn_out_buf.data(), attn_out_buf.size(), attn_out_shape_vec);
  const char* merger_input_names[] = {"hidden"};
  OrtValue* merger_inputs[] = { merger_in_tensor.get() };
  
  // Patch merger output shape: [seq_len / 4, 3584] 
  // The merger reduces spatial dimensions and projects to final vision hidden size
  int64_t merged_seq_len = seq_len / (spatial_merge_size_ * spatial_merge_size_);
  int64_t merged_hidden_size = 3584;  // Qwen2.5-VL final vision embedding dimension
  std::vector<int64_t> merger_out_shape_vec{merged_seq_len, merged_hidden_size};
  size_t merger_out_count = static_cast<size_t>(merged_seq_len * merged_hidden_size);
  std::vector<float> merger_out_buf(merger_out_count);
  auto merger_out_tensor = CreateTensor(merger_out_buf.data(), merger_out_count, merger_out_shape_vec);
  auto merger_out_name_str = patch_merger_session_->GetOutputName(0);
  const char* merger_output_names[] = { merger_out_name_str.c_str() };
  OrtValue* merger_outputs[] = { merger_out_tensor.get() };
  
  patch_merger_session_->Run(nullptr, merger_input_names, merger_inputs, 1, merger_output_names, merger_outputs, 1);

  float min_merger = merger_out_buf[0], max_merger = merger_out_buf[0], sum_merger = 0.0f;
  for (const auto& val : merger_out_buf) {
    min_merger = std::min(min_merger, val);
    max_merger = std::max(max_merger, val);
    sum_merger += val;
  }

  // Final reverse ordering using rev_idx_ (argsort of wnd_idx). Expect same number of windows mapping.
  // Merger output shape assumed [num_windows * window_area, hidden_size] or potentially [num_windows, hidden_size].
  // After merger, sequence length is reduced by spatial_merge_size^2
  if (merger_out_shape_vec.size() != 2) {
    throw std::runtime_error("Patch merger output must be rank-2");
  }
  int64_t final_seq_len = merger_out_shape_vec[0];  // 493 (merged)
  int64_t final_hidden = merger_out_shape_vec[1];     // 3584 (merged)
  
  // Validate final dimensions match expected after merging
  if (final_seq_len != merged_seq_len) {
    throw std::runtime_error("Unexpected final sequence length after merger");
  }
  if (final_hidden != merged_hidden_size) {
    throw std::runtime_error("Final hidden size mismatch after merger");
  }
  if (static_cast<int64_t>(rev_idx_.size()) != num_windows) {
    // Each window maps back; reorder at window granularity.
    throw std::runtime_error("rev_idx size does not match number of windows");
  }

  // Apply reverse indexing at merged window granularity
  // After merging, we have merged_seq_len tokens, one per original window
  std::vector<float> final_embeddings(merger_out_buf.size());
  for (int64_t dst_w = 0; dst_w < num_windows; ++dst_w) {
    int64_t src_w = rev_idx_[dst_w];
    // Each "window" in merged output is now just 1 token with merged_hidden_size features
    size_t src_offset = static_cast<size_t>(src_w) * static_cast<size_t>(final_hidden);
    size_t dst_offset = static_cast<size_t>(dst_w) * static_cast<size_t>(final_hidden);
    std::memcpy(final_embeddings.data() + dst_offset, merger_out_buf.data() + src_offset,
                static_cast<size_t>(final_hidden) * sizeof(float));
  }

  // Save final shape
  last_seq_len_ = final_seq_len;
  last_hidden_size_ = final_hidden;
  return final_embeddings; // shape: [final_seq_len=493, final_hidden=3584]
}

} // namespace Generators
