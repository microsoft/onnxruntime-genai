// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Fara VLM Vision pipeline implementation with optional QNN EP for vision attention stage.

#include "fara_vl_vision.h"
#include "../generators.h"

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
  unsigned char ver_major;
  unsigned char ver_minor;
  fin.read(reinterpret_cast<char*>(&ver_major), 1);
  fin.read(reinterpret_cast<char*>(&ver_minor), 1);
  uint16_t header_len_le;
  fin.read(reinterpret_cast<char*>(&header_len_le), 2);  // little endian
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

FaraVisionPipeline::FaraVisionPipeline(OrtEnv& env,
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
    auto so = OrtSessionOptions::Create();

    so->SetIntraOpNumThreads(2).SetInterOpNumThreads(1);
    // QNN provider options
    const char* keys[] = {"backend_path", "htp_performance_mode", "htp_graph_finalization_optimization_mode", "soc_model"};
    const char* values[] = { qnn_backend_path_.c_str(), "burst", "3", "60" };

    auto providers = Ort::GetAvailableProviders();
    bool has_qnn = std::find(providers.begin(), providers.end(), std::string("QNNExecutionProvider")) != providers.end();
    if (has_qnn) {
      so->AppendExecutionProvider("QNNExecutionProvider", keys, values, 4);
    }
    else {
      // Use registered QNN EP
      size_t num_devices = 0;
      const OrtEpDevice* const* device_ptrs = nullptr;
      Ort::GetEpDevices(&GetOrtEnv(), &device_ptrs, &num_devices);
      std::vector<const OrtEpDevice*> ep_devices_ptrs;
      ep_devices_ptrs.reserve(num_devices);
      for (size_t i = 0; i < num_devices; ++i) {
        if (Ort::api->EpDevice_EpName(device_ptrs[i]) == std::string("QNNExecutionProvider")) {
          ep_devices_ptrs.push_back(device_ptrs[i]);
        }
      }

      if (ep_devices_ptrs.empty()) {
        throw std::runtime_error("QNNExecutionProvider requested for vision attention but not registered.");
      } else {
        Ort::api->SessionOptionsAppendExecutionProvider_V2(
          so.get(),
          &GetOrtEnv(),
          ep_devices_ptrs.data(), ep_devices_ptrs.size(),
          keys, values, 4
        );
      }
    }

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
  std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return a.first < b.first; });
  for (size_t i = 0; i < pairs.size(); ++i) rev_idx_[i] = static_cast<int64_t>(pairs[i].second);
}

std::unique_ptr<OrtValue> FaraVisionPipeline::CreateTensor(const float* data, size_t count, const std::vector<int64_t>& shape) const {
  auto memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::span<float> data_span(const_cast<float*>(data), count);
  std::span<const int64_t> shape_span(shape.data(), shape.size());
  return OrtValue::CreateTensor<float>(*memory_info, data_span, shape_span);
}

// Removed CreateEmptyTensor (previous implementation returned tensor with dangling backing store).

std::vector<float> FaraVisionPipeline::Run(const float* pixel_data, const std::vector<int64_t>& pixel_shape) {
  if (!patch_embed_session_ || !vision_attn_session_ || !patch_merger_session_) {
    throw std::runtime_error("Vision pipeline sessions not initialized");
  }

  size_t pixel_count = 1;
  for (auto d : pixel_shape) pixel_count *= static_cast<size_t>(d);
  auto pixel_tensor = CreateTensor(pixel_data, pixel_count, pixel_shape);

  const char* pe_input_names[] = {"pixel_values"};
  OrtValue* pe_inputs[] = {pixel_tensor.get()};

  const int64_t num_patches = pixel_shape[1];
  const int64_t hidden_dim = 1280;
  std::vector<int64_t> pe_out_shape{num_patches, hidden_dim};
  std::vector<float> pe_out_buf(num_patches * hidden_dim);
  auto pe_out_tensor = CreateTensor(pe_out_buf.data(), pe_out_buf.size(), pe_out_shape);

  auto pe_out_name = patch_embed_session_->GetOutputName(0);
  const char* pe_output_names[] = {pe_out_name.c_str()};
  OrtValue* pe_outputs[] = {pe_out_tensor.get()};

  patch_embed_session_->Run(nullptr, pe_input_names, pe_inputs, 1, pe_output_names, pe_outputs, 1);

  const int64_t seq_len = num_patches;
  const int64_t window_area = spatial_merge_size_ * spatial_merge_size_;
  const int64_t num_windows = seq_len / window_area;

  if (seq_len % window_area != 0 || static_cast<int64_t>(wnd_idx_.size()) != num_windows) {
    throw std::runtime_error("Invalid window configuration for vision pipeline");
  }

  std::vector<float> reordered(seq_len * hidden_dim);
  for (int64_t dst_w = 0; dst_w < num_windows; ++dst_w) {
    int64_t src_w = wnd_idx_[dst_w];
    if (src_w < 0 || src_w >= num_windows) throw std::runtime_error("wnd_idx value out of range");
    size_t offset_size = window_area * hidden_dim;
    std::memcpy(reordered.data() + dst_w * offset_size,
                pe_out_buf.data() + src_w * offset_size,
                offset_size * sizeof(float));
  }

  std::vector<int64_t> attn_shape{seq_len, hidden_dim};
  auto attn_in_tensor = CreateTensor(reordered.data(), reordered.size(), attn_shape);
  const char* attn_input_names[] = {"hidden"};
  OrtValue* attn_inputs[] = {attn_in_tensor.get()};

  std::vector<float> attn_out_buf(seq_len * hidden_dim);
  auto attn_out_tensor = CreateTensor(attn_out_buf.data(), attn_out_buf.size(), attn_shape);
  auto attn_out_name = vision_attn_session_->GetOutputName(0);
  const char* attn_output_names[] = {attn_out_name.c_str()};
  OrtValue* attn_outputs[] = {attn_out_tensor.get()};

  vision_attn_session_->Run(nullptr, attn_input_names, attn_inputs, 1, attn_output_names, attn_outputs, 1);

  auto merger_in_tensor = CreateTensor(attn_out_buf.data(), attn_out_buf.size(), attn_shape);
  const char* merger_input_names[] = {"hidden"};
  OrtValue* merger_inputs[] = {merger_in_tensor.get()};

  const int64_t merged_seq_len = num_windows;  // One token per window after merging
  const int64_t merged_hidden = 3584;
  std::vector<int64_t> merger_shape{merged_seq_len, merged_hidden};
  std::vector<float> merger_out_buf(merged_seq_len * merged_hidden);
  auto merger_out_tensor = CreateTensor(merger_out_buf.data(), merger_out_buf.size(), merger_shape);
  auto merger_out_name = patch_merger_session_->GetOutputName(0);
  const char* merger_output_names[] = {merger_out_name.c_str()};
  OrtValue* merger_outputs[] = {merger_out_tensor.get()};

  patch_merger_session_->Run(nullptr, merger_input_names, merger_inputs, 1, merger_output_names, merger_outputs, 1);

  if (static_cast<int64_t>(rev_idx_.size()) != num_windows) {
    throw std::runtime_error("Vision pipeline reverse index size mismatch");
  }

  std::vector<float> final_embeddings(merger_out_buf.size());
  for (int64_t dst_w = 0; dst_w < num_windows; ++dst_w) {
    std::memcpy(final_embeddings.data() + dst_w * merged_hidden,
                merger_out_buf.data() + rev_idx_[dst_w] * merged_hidden,
                merged_hidden * sizeof(float));
  }

  last_seq_len_ = merged_seq_len;
  last_hidden_size_ = merged_hidden;
  return final_embeddings;
}

}  // namespace Generators
