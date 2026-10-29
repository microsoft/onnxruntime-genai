// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "generator/generators.h"
#include "model.h"
#include "multi_modal.h"
#include "io/token_type_ids.h"
#include "openvino_multi_modal.h"
#include "openvino_vision_merger.h"
#include "openvino_vision_merger_utils.h"
#include "qwen_vl_state.h"

namespace Generators {

std::vector<QwenImageSlice> PlanQwenImageSlices(std::span<const int64_t> grid_values,
                                                int64_t total_patches,
                                                int64_t merge_size) {
  if (grid_values.empty() || grid_values.size() % 3 != 0) {
    throw std::runtime_error("Qwen3.5 OpenVINO vision: image_grid_thw must contain one or more triplets");
  }
  if (total_patches < 0 || merge_size <= 0 ||
      merge_size > std::numeric_limits<int64_t>::max() / merge_size) {
    throw std::runtime_error("Qwen3.5 OpenVINO vision: invalid patch count or spatial_merge_size");
  }

  auto checked_multiply = [](int64_t lhs, int64_t rhs, const char* dimension) {
    if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs) {
      throw std::runtime_error(std::string("Qwen3.5 OpenVINO vision: grid dimension overflow for ") + dimension);
    }
    return lhs * rhs;
  };

  const size_t num_images = grid_values.size() / 3;
  std::vector<std::array<int64_t, 3>> grids;
  grids.reserve(num_images);
  int64_t total_grid_tokens = 0;
  int64_t total_hw = 0;
  int64_t max_grid_tokens = 0;
  bool all_temporal_dims_one = true;

  for (size_t i = 0; i < num_images; ++i) {
    const int64_t temporal = grid_values[i * 3];
    const int64_t height = grid_values[i * 3 + 1];
    const int64_t width = grid_values[i * 3 + 2];
    if (temporal <= 0 || height <= 0 || width <= 0) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: grid dimensions must be positive");
    }
    if (height % merge_size != 0 || width % merge_size != 0) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: image height and width must be divisible by spatial_merge_size");
    }

    const int64_t hw = checked_multiply(height, width, "height*width");
    const int64_t grid_tokens = checked_multiply(temporal, hw, "temporal*height*width");
    if (grid_tokens > std::numeric_limits<int64_t>::max() - total_grid_tokens ||
        hw > std::numeric_limits<int64_t>::max() - total_hw) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: aggregate grid size overflow");
    }
    total_grid_tokens += grid_tokens;
    total_hw += hw;
    max_grid_tokens = std::max(max_grid_tokens, grid_tokens);
    all_temporal_dims_one = all_temporal_dims_one && temporal == 1;
    grids.push_back({temporal, height, width});
  }

  const auto layout = ResolveQwenPatchLayout(total_patches, total_grid_tokens, total_hw, max_grid_tokens,
                                             static_cast<int64_t>(num_images), all_temporal_dims_one);
  if (layout.temporal_multiplier > 0) {
    throw std::runtime_error("Qwen3.5 OpenVINO vision does not support temporal-multiplier pixel_values layouts");
  }

  std::vector<QwenImageSlice> slices;
  slices.reserve(num_images);
  int64_t packed_offset = 0;
  for (size_t i = 0; i < num_images; ++i) {
    const auto [temporal, height, width] = grids[i];
    const int64_t num_patches = checked_multiply(temporal, checked_multiply(height, width, "height*width"),
                                                 "temporal*height*width");
    const int64_t patch_offset = layout.ImagePatchOffset(static_cast<int64_t>(i), packed_offset);
    if (patch_offset < 0 || patch_offset > total_patches || num_patches > total_patches - patch_offset) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: image patch slice exceeds pixel_values");
    }
    if (num_patches > std::numeric_limits<int64_t>::max() / num_patches) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: image attention mask size overflow");
    }
    slices.push_back({patch_offset, num_patches, temporal, height, width});
    packed_offset += num_patches;
  }
  return slices;
}

std::vector<int64_t> FindQwenImageTokenRows(DeviceSpan<int32_t> tokens, int32_t image_token_id) {
  std::vector<int64_t> rows;
  if (tokens.empty()) return rows;
  // The tokens may be device memory; CpuSpan alone would expose a stale or unallocated host mirror.
  const auto host_tokens = tokens.CopyDeviceToCpu();
  for (size_t i = 0; i < host_tokens.size(); ++i) {
    if (host_tokens[i] == image_token_id) rows.push_back(static_cast<int64_t>(i));
  }
  return rows;
}

size_t MergeImageFeaturesIntoEmbeddings(OrtValue& inputs_embeds,
                                        const OrtValue& image_features,
                                        std::span<const int64_t> target_token_rows) {
  const auto embeddings_info = inputs_embeds.GetTensorTypeAndShapeInfo();
  const auto vision_info = image_features.GetTensorTypeAndShapeInfo();
  if (embeddings_info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      vision_info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: both tensors must contain float elements");
  }

  const auto embeddings_shape = embeddings_info->GetShape();
  const auto vision_shape = vision_info->GetShape();
  if (embeddings_shape.empty() || vision_shape.empty()) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: tensors must have a hidden dimension");
  }

  // Hidden size is always the last dimension for both [batch, seq, hidden]/[seq, hidden] embeddings
  // and [num_image_tokens, hidden] image features.
  const int64_t embedding_dim = embeddings_shape.back();
  const int64_t vision_dim = vision_shape.back();
  if (embedding_dim <= 0 || vision_dim <= 0) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: hidden dimensions must be positive");
  }
  if (vision_dim != embedding_dim) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: hidden dimension mismatch - vision_dim=" +
                             std::to_string(vision_dim) + ", embedding_dim=" + std::to_string(embedding_dim));
  }

  const size_t embedding_element_count = embeddings_info->GetElementCount();
  const size_t vision_element_count = vision_info->GetElementCount();
  if (embedding_element_count % static_cast<size_t>(embedding_dim) != 0 ||
      vision_element_count % static_cast<size_t>(vision_dim) != 0) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: tensor element count is not divisible by hidden dimension");
  }
  const size_t embedding_rows = embedding_element_count / static_cast<size_t>(embedding_dim);
  const size_t num_vision_tokens = vision_element_count / static_cast<size_t>(vision_dim);
  const size_t rows_to_write = std::min(target_token_rows.size(), num_vision_tokens);

  for (size_t i = 0; i < rows_to_write; ++i) {
    const int64_t row = target_token_rows[i];
    if (row < 0 || static_cast<uint64_t>(row) >= embedding_rows) {
      throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: destination row " + std::to_string(row) +
                               " is outside embeddings row count " + std::to_string(embedding_rows));
    }
  }

  float* embeddings_data = inputs_embeds.GetTensorMutableData<float>();
  const float* vision_data = image_features.GetTensorData<float>();
  for (size_t consumed = 0; consumed < rows_to_write; ++consumed) {
    const int64_t row = target_token_rows[consumed];
    std::memcpy(embeddings_data + (row * embedding_dim),
                vision_data + (consumed * vision_dim),
                vision_dim * sizeof(float));
  }
  return rows_to_write;
}

namespace {

// Returns the OrtValue for a named extra input, or nullptr if absent.
OrtValue* FindExtraInput(const std::vector<ExtraInput>& extra_inputs, std::string_view name) {
  for (const auto& input : extra_inputs) {
    if (input.name == name) return input.tensor->GetOrtTensor();
  }
  return nullptr;
}

const OrtValue* PrepareVisionInput(OrtSession& session, const std::string& name, OrtValue& input,
                                   std::unique_ptr<OrtValue>& converted, DeviceInterface& device) {
  for (size_t i = 0; i < session.GetInputCount(); ++i) {
    if (session.GetInputName(i) != name) continue;
    const auto type = session.GetInputTypeInfo(i)->GetTensorTypeAndShapeInfo().GetElementType();
    if (input.GetTensorTypeAndShapeInfo()->GetElementType() == type) return &input;
    Cast(input, converted, device, type);
    return converted.get();
  }
  throw std::runtime_error("OpenVINO vision: session input not found: " + name);
}

void MergeAllImageFeatures(OrtValue& inputs_embeds,
                           const OrtValue& image_features,
                           std::span<const int64_t> image_token_rows) {
  const auto info = image_features.GetTensorTypeAndShapeInfo();
  const auto shape = info->GetShape();
  if (shape.empty() || shape.back() <= 0) {
    throw std::runtime_error("OpenVINO vision merger: image features must have a positive hidden dimension");
  }
  const size_t hidden_dim = static_cast<size_t>(shape.back());
  const size_t element_count = info->GetElementCount();
  if (element_count % hidden_dim != 0) {
    throw std::runtime_error("OpenVINO vision merger: image feature element count is not divisible by hidden dimension");
  }
  const size_t feature_rows = element_count / hidden_dim;
  if (image_token_rows.size() != feature_rows) {
    throw std::runtime_error("OpenVINO vision merger: image token count " + std::to_string(image_token_rows.size()) +
                             " does not match image feature row count " + std::to_string(feature_rows));
  }
  MergeImageFeaturesIntoEmbeddings(inputs_embeds, image_features, image_token_rows);
}

// Wraps a caller-owned buffer as a CPU tensor (a view; the buffer must outlive the tensor).
template <typename T>
std::unique_ptr<OrtValue> WrapCpuTensor(std::vector<T>& data, std::span<const int64_t> shape) {
  auto memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  return OrtValue::CreateTensor<T>(*memory_info, std::span<T>(data.data(), data.size()), shape);
}

// Gemma3: single vision session (pixel_values -> image_features). Image placeholder tokens are marked
// by token_type_ids == 1 (produced by GemmaImageProcessor); Gemma configs carry no image_token_id, so the
// token_type_ids channel is used to locate the image rows for host-side feature injection.
struct Gemma3OpenVINOVisionMerger : OpenVINOVisionMerger {
  explicit Gemma3OpenVINOVisionMerger(const OpenVINOMultiModalModel& model)
      : model_{model}, run_options_{OrtRunOptions::Create()} {
    if (model_.config_->model.vision.run_options) {
      for (const auto& [key, value] : *model_.config_->model.vision.run_options) {
        ApplyRunOption(*run_options_, key.c_str(), value.c_str());
      }
    }
  }

  void RunVision(const std::vector<ExtraInput>& extra_inputs) override;
  void Merge(OrtValue& inputs_embeds, DeviceSpan<int32_t> input_ids) override;

 private:
  const OpenVINOMultiModalModel& model_;
  std::unique_ptr<OrtRunOptions> run_options_;
  std::unique_ptr<OrtValue> image_features_;  // produced for the current prompt (null when no image)
  std::vector<int32_t> token_type_ids_;       // host copy of token_type_ids for the current prompt
};

void Gemma3OpenVINOVisionMerger::RunVision(const std::vector<ExtraInput>& extra_inputs) {
  image_features_.reset();
  token_type_ids_.clear();

  OrtValue* pixel_values = FindExtraInput(extra_inputs, model_.config_->model.vision.inputs.pixel_values);
  if (!pixel_values || !model_.ov_vision_session_) return;  // text-only prompt

  // Validate markers before spending work on vision inference.
  if (OrtValue* token_type_ids = FindExtraInput(extra_inputs, Config::Defaults::TokenTypeIdsName)) {
    token_type_ids_ = ReadTokenTypeIds(*token_type_ids);
  }

  const std::string& pixel_input_name = model_.config_->model.vision.inputs.pixel_values;
  std::unique_ptr<OrtValue> converted_pixels;
  const OrtValue* pixel_input = PrepareVisionInput(*model_.ov_vision_session_, pixel_input_name, *pixel_values,
                                                   converted_pixels, *model_.p_device_inputs_);
  const std::string& image_features_name = model_.config_->model.vision.outputs.image_features;
  const char* input_names[] = {pixel_input_name.c_str()};
  const OrtValue* input_values[] = {pixel_input};
  const char* output_names[] = {image_features_name.c_str()};

  auto outputs = model_.ov_vision_session_->Run(run_options_.get(), input_names, input_values, 1, output_names, 1);
  if (outputs[0]->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    Cast(*outputs[0], image_features_, *model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  } else {
    image_features_ = std::move(outputs[0]);
  }
}

void Gemma3OpenVINOVisionMerger::Merge(OrtValue& inputs_embeds, DeviceSpan<int32_t> input_ids) {
  if (!image_features_) return;  // text-only prompt

  // Only the token count is needed, so the token values are never read back to the host.
  const size_t prompt_tokens = input_ids.size();
  if (token_type_ids_.size() != prompt_tokens) {
    throw std::runtime_error("Gemma3 OpenVINO vision merger: token_type_ids has " +
                             std::to_string(token_type_ids_.size()) + " values but the prompt has " +
                             std::to_string(prompt_tokens) + " tokens");
  }

  // Gemma marks image placeholder tokens with token_type_ids == 1.
  std::vector<int64_t> image_token_rows;
  image_token_rows.reserve(token_type_ids_.size());
  for (size_t i = 0; i < token_type_ids_.size(); ++i) {
    if (token_type_ids_[i] == 1) image_token_rows.push_back(static_cast<int64_t>(i));
  }

  MergeAllImageFeatures(inputs_embeds, *image_features_, image_token_rows);
}

// ----------------------------------------------------------------------------
// Qwen3.5 3-stage vision (patch_embed -> pos -> merger). Host math mirrors OpenVINO GenAI's
// Qwen3-VL encoder (qwen3_vl / qwen2vl). All patch ordering is spatial-merge order,
// which the processor's PatchImage transform produces, so no host re-ordering of patch embeds.

constexpr float kRotaryTheta = 10000.0f;

// Bilinear interpolation indices/weights over a num_grid_per_side x num_grid_per_side position grid.
// indices [4, N] (i64) and weights [4, N] (f32) are corner-major: corner c, position p at c*N + p.
// Positions are enumerated in raw [t, h, w] order; the spatial-merge reorder happens later.
void GetPositionInterpolationIndicesAndWeights(int64_t t, int64_t h, int64_t w, int64_t num_grid_per_side,
                                               std::vector<int64_t>& indices, std::vector<float>& weights) {
  const int64_t num_positions = t * h * w;
  indices.resize(4 * num_positions);
  weights.resize(4 * num_positions);

  const int64_t grid_max = num_grid_per_side - 1;
  const float h_scale = h > 1 ? static_cast<float>(grid_max) / (h - 1) : 0.0f;
  const float w_scale = w > 1 ? static_cast<float>(grid_max) / (w - 1) : 0.0f;

  int64_t p = 0;
  for (int64_t ti = 0; ti < t; ++ti) {
    for (int64_t hi = 0; hi < h; ++hi) {
      const float h_idx = hi * h_scale;
      const int64_t h_floor = static_cast<int64_t>(h_idx);
      const int64_t h_ceil = std::min(h_floor + 1, grid_max);
      const float dh = h_idx - h_floor;
      const int64_t h_floor_row = h_floor * num_grid_per_side;
      const int64_t h_ceil_row = h_ceil * num_grid_per_side;

      for (int64_t wi = 0; wi < w; ++wi) {
        const float w_idx = wi * w_scale;
        const int64_t w_floor = static_cast<int64_t>(w_idx);
        const int64_t w_ceil = std::min(w_floor + 1, grid_max);
        const float dw = w_idx - w_floor;

        indices[0 * num_positions + p] = h_floor_row + w_floor;
        indices[1 * num_positions + p] = h_floor_row + w_ceil;
        indices[2 * num_positions + p] = h_ceil_row + w_floor;
        indices[3 * num_positions + p] = h_ceil_row + w_ceil;

        weights[0 * num_positions + p] = (1.0f - dh) * (1.0f - dw);
        weights[1 * num_positions + p] = (1.0f - dh) * dw;
        weights[2 * num_positions + p] = dh * (1.0f - dw);
        weights[3 * num_positions + p] = dh * dw;
        ++p;
      }
    }
  }
}

// Weighted sum of the 4 corner position embeddings ([4, N, D]) into [N, D], in raw [t, h, w] order.
std::vector<float> WeightedSumOverCorners(const float* pos_embeds, const std::vector<float>& weights,
                                          int64_t num_positions, int64_t embed_dim) {
  std::vector<float> weighted_sum(num_positions * embed_dim, 0.0f);
  for (int64_t corner = 0; corner < 4; ++corner) {
    for (int64_t pos = 0; pos < num_positions; ++pos) {
      const float weight = weights[corner * num_positions + pos];
      const float* src = pos_embeds + (corner * num_positions + pos) * embed_dim;
      float* dst = weighted_sum.data() + pos * embed_dim;
      for (int64_t d = 0; d < embed_dim; ++d) dst[d] += weight * src[d];
    }
  }
  return weighted_sum;
}

// Adds the interpolated position embeddings (raw [t, h, w] order) onto the patch embeddings, which are
// already laid out in spatial-merge order [t, h/m, w/m, m, m]. In-place add matching the OV reference.
void PermuteWithSpatialMergeAndAdd(const std::vector<float>& pos_embeds, float* hidden, int64_t t,
                                   int64_t h, int64_t w, int64_t merge_size, int64_t embed_dim) {
  const int64_t hw = h * w;
  const int64_t merge_h = h / merge_size;
  const int64_t merge_w = w / merge_size;
  int64_t dst_offset = 0;
  for (int64_t ti = 0; ti < t; ++ti) {
    for (int64_t mhi = 0; mhi < merge_h; ++mhi) {
      for (int64_t mwi = 0; mwi < merge_w; ++mwi) {
        for (int64_t shi = 0; shi < merge_size; ++shi) {
          for (int64_t swi = 0; swi < merge_size; ++swi) {
            const int64_t src_h = mhi * merge_size + shi;
            const int64_t src_w = mwi * merge_size + swi;
            const int64_t src_idx = ti * hw + src_h * w + src_w;
            const float* src = pos_embeds.data() + src_idx * embed_dim;
            float* dst = hidden + dst_offset * embed_dim;
            for (int64_t d = 0; d < embed_dim; ++d) dst[d] += src[d];
            ++dst_offset;
          }
        }
      }
    }
  }
}

// Vision rotary embeddings [N, rotary_dim], concatenating per-token h and w frequency halves. Height/width
// indices are generated in spatial-merge-block order, matching the patch layout.
std::vector<float> BuildRotaryPosEmb(int64_t t, int64_t h, int64_t w, int64_t merge_size,
                                     int64_t rotary_dim) {
  const int64_t half = rotary_dim / 2;
  std::vector<float> inv_freq(half);
  for (int64_t i = 0; i < half; ++i) {
    inv_freq[i] = 1.0f / std::pow(kRotaryTheta, static_cast<float>(i) / static_cast<float>(half));
  }

  const int64_t max_grid = std::max(h, w);
  std::vector<float> freqs(max_grid * half);
  for (int64_t pos = 0; pos < max_grid; ++pos) {
    for (int64_t j = 0; j < half; ++j) freqs[pos * half + j] = pos * inv_freq[j];
  }

  const int64_t num_positions = t * h * w;
  std::vector<float> rope(num_positions * rotary_dim);
  int64_t i = 0;
  for (int64_t ti = 0; ti < t; ++ti) {
    for (int64_t hb = 0; hb < h / merge_size; ++hb) {
      for (int64_t wb = 0; wb < w / merge_size; ++wb) {
        for (int64_t hs = 0; hs < merge_size; ++hs) {
          for (int64_t ws = 0; ws < merge_size; ++ws) {
            const int64_t hpos = hb * merge_size + hs;
            const int64_t wpos = wb * merge_size + ws;
            std::copy_n(freqs.data() + hpos * half, half, rope.data() + i * rotary_dim);
            std::copy_n(freqs.data() + wpos * half, half, rope.data() + i * rotary_dim + half);
            ++i;
          }
        }
      }
    }
  }
  return rope;
}

// Block-diagonal attention mask [1, N, N]: 0.0 inside each per-frame block, -inf elsewhere. A single
// image (t=1) yields one full-attention block. (Qwen3.5 does not use windowed attention.)
std::vector<float> BuildAttentionMask(int64_t t, int64_t h, int64_t w) {
  const int64_t num_positions = t * h * w;
  std::vector<float> mask(num_positions * num_positions, -std::numeric_limits<float>::infinity());
  const int64_t slice_len = h * w;
  for (int64_t frame = 0; frame < t; ++frame) {
    const int64_t start = frame * slice_len;
    const int64_t end = start + slice_len;
    for (int64_t row = start; row < end; ++row) {
      for (int64_t col = start; col < end; ++col) mask[row * num_positions + col] = 0.0f;
    }
  }
  return mask;
}

// Qwen3.5: 3-stage vision pipeline; image placeholder tokens are located by input_ids == image_token_id.
struct Qwen3_5OpenVINOVisionMerger : OpenVINOVisionMerger {
  explicit Qwen3_5OpenVINOVisionMerger(const OpenVINOMultiModalModel& model);

  void RunVision(const std::vector<ExtraInput>& extra_inputs) override;
  void Merge(OrtValue& inputs_embeds, DeviceSpan<int32_t> input_ids) override;

 private:
  const OpenVINOMultiModalModel& model_;

  OrtSession* patch_embed_session_{};
  OrtSession* pos_session_{};
  OrtSession* merger_session_{};
  std::unique_ptr<OrtRunOptions> patch_embed_run_options_, pos_run_options_, merger_run_options_;

  // Stage I/O names, resolved from the config pipeline stage inputs/outputs arrays.
  std::string patch_embed_input_, patch_embed_output_;
  std::string pos_input_, pos_output_;
  std::string merger_hidden_input_, merger_mask_input_, merger_rope_input_, merger_output_;

  int64_t spatial_merge_size_{2};
  int64_t rotary_dim_{};           // read from the merger's rotary_pos_emb input width
  int64_t num_grid_per_side_{48};  // sqrt(num_position_embeddings): 2304 -> 48 for qwen3_5

  std::unique_ptr<OrtValue> image_features_;  // produced for the current prompt (null when no image)
};

Qwen3_5OpenVINOVisionMerger::Qwen3_5OpenVINOVisionMerger(const OpenVINOMultiModalModel& model)
    : model_{model} {
  spatial_merge_size_ = model_.config_->model.vision.spatial_merge_size;

  auto find_stage = [&](const std::string& id) -> const Config::Model::Vision::PipelineModel* {
    for (const auto& stage : model_.config_->model.vision.pipeline) {
      if (stage.model_id == id) return &stage;
    }
    return nullptr;
  };
  const auto* patch_embed = find_stage("patch_embed");
  const auto* pos = find_stage("pos");
  const auto* merger = find_stage("merger");
  if (!patch_embed || !pos || !merger) {
    throw std::runtime_error("Qwen3.5 OpenVINO vision merger requires patch_embed, pos and merger pipeline stages.");
  }
  if (merger->inputs.size() < 3) {
    throw std::runtime_error("Qwen3.5 OpenVINO merger stage expects 3 inputs (hidden_states, attention_mask, rotary_pos_emb).");
  }

  auto create_run_options = [&](const Config::Model::Vision::PipelineModel& stage) {
    auto options = OrtRunOptions::Create();
    for (const auto& [key, value] : GetOpenVINOVisionStageRunOptions(*model_.config_, stage)) {
      ApplyRunOption(*options, key.c_str(), value.c_str());
    }
    return options;
  };
  patch_embed_run_options_ = create_run_options(*patch_embed);
  pos_run_options_ = create_run_options(*pos);
  merger_run_options_ = create_run_options(*merger);

  patch_embed_input_ = patch_embed->inputs.at(0);
  patch_embed_output_ = patch_embed->outputs.at(0);
  pos_input_ = pos->inputs.at(0);
  pos_output_ = pos->outputs.at(0);
  merger_hidden_input_ = merger->inputs.at(0);
  merger_mask_input_ = merger->inputs.at(1);
  merger_rope_input_ = merger->inputs.at(2);
  merger_output_ = merger->outputs.at(0);

  patch_embed_session_ = &model_.GetVisionPipelineSession("patch_embed");
  pos_session_ = &model_.GetVisionPipelineSession("pos");
  merger_session_ = &model_.GetVisionPipelineSession("merger");

  // Rotary width comes from the merger's rotary_pos_emb input (e.g. 32).
  for (size_t i = 0; i < merger_session_->GetInputCount(); ++i) {
    if (merger_session_->GetInputName(i) == merger_rope_input_) {
      auto shape = merger_session_->GetInputTypeInfo(i)->GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() >= 2 && shape[1] > 0) rotary_dim_ = shape[1];
      break;
    }
  }
  if (rotary_dim_ <= 0) {
    throw std::runtime_error("Qwen3.5 OpenVINO merger: could not resolve rotary_pos_emb width.");
  }
}

void Qwen3_5OpenVINOVisionMerger::RunVision(const std::vector<ExtraInput>& extra_inputs) {
  image_features_.reset();

  OrtValue* pixel_values = FindExtraInput(extra_inputs, model_.config_->model.vision.inputs.pixel_values);
  if (!pixel_values) return;  // text-only prompt

  auto pixel_info = pixel_values->GetTensorTypeAndShapeInfo();
  const auto pixel_shape = pixel_info->GetShape();
  if (pixel_shape.empty() || pixel_shape.back() <= 0) {
    throw std::runtime_error("Qwen3.5 OpenVINO vision: pixel_values must have a positive patch dimension");
  }
  const int64_t patch_dim = pixel_shape.back();
  const size_t pixel_element_count = pixel_info->GetElementCount();
  if (pixel_element_count % static_cast<size_t>(patch_dim) != 0 ||
      pixel_element_count / static_cast<size_t>(patch_dim) >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error("Qwen3.5 OpenVINO vision: invalid pixel_values patch count");
  }
  const int64_t total_patches = static_cast<int64_t>(pixel_element_count / static_cast<size_t>(patch_dim));

  OrtValue* grid_thw = FindExtraInput(extra_inputs, Config::Defaults::ImageGridThwName);
  if (!grid_thw) {
    grid_thw = FindExtraInput(extra_inputs, model_.config_->model.vision.inputs.image_grid_thw);
  }
  if (!grid_thw) throw std::runtime_error("Qwen3.5 OpenVINO vision: image_grid_thw input missing.");
  const auto grid_info = grid_thw->GetTensorTypeAndShapeInfo();
  const size_t grid_count = grid_info->GetElementCount();
  if (grid_count == 0 || grid_count % 3 != 0) {
    throw std::runtime_error("Qwen3.5 OpenVINO vision: image_grid_thw must contain one or more triplets");
  }
  if (grid_info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("Qwen3.5 OpenVINO vision: image_grid_thw must contain int64 values");
  }
  std::vector<int64_t> grid_values(grid_count);
  const int64_t* values = grid_thw->GetTensorData<int64_t>();
  std::copy_n(values, grid_count, grid_values.begin());
  const auto slices = PlanQwenImageSlices(grid_values, total_patches, spatial_merge_size_);

  // Keep CPU slicing in float32; stage feeds follow each session's declared type.
  std::unique_ptr<OrtValue> pixel_values_fp32;
  OrtValue* pixel_input = pixel_values;
  if (pixel_values->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    Cast(*pixel_values, pixel_values_fp32, *model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    pixel_input = pixel_values_fp32.get();
  }

  auto& allocator = model_.allocator_cpu_;
  const auto pixel_data = pixel_input->GetTensorMutableData<float>();
  auto cpu_memory = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<std::unique_ptr<OrtValue>> image_features;
  image_features.reserve(slices.size());
  int64_t merged_hidden_dim = 0;
  int64_t total_feature_rows = 0;

  for (const auto& slice : slices) {
    const size_t patch_offset = static_cast<size_t>(slice.patch_offset);
    const size_t num_patches = static_cast<size_t>(slice.num_patches);
    if (patch_offset > std::numeric_limits<size_t>::max() / static_cast<size_t>(patch_dim) ||
        num_patches > std::numeric_limits<size_t>::max() / static_cast<size_t>(patch_dim)) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: pixel_values slice size overflow");
    }
    const size_t patch_element_offset = patch_offset * static_cast<size_t>(patch_dim);
    const size_t patch_element_count = num_patches * static_cast<size_t>(patch_dim);
    if (patch_element_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: pixel_values slice byte size overflow");
    }
    const std::array<int64_t, 2> image_pixel_shape{slice.num_patches, patch_dim};
    auto image_pixel_values = OrtValue::CreateTensor(
        *cpu_memory, pixel_data + patch_element_offset, patch_element_count * sizeof(float), image_pixel_shape,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    const char* pe_in[] = {patch_embed_input_.c_str()};
    std::unique_ptr<OrtValue> converted_pixels;
    const OrtValue* pe_vals[] = {PrepareVisionInput(*patch_embed_session_, patch_embed_input_, *image_pixel_values,
                                                    converted_pixels, *model_.p_device_inputs_)};
    const char* pe_out[] = {patch_embed_output_.c_str()};
    auto hidden_states = std::move(patch_embed_session_->Run(patch_embed_run_options_.get(), pe_in, pe_vals, 1, pe_out, 1)[0]);
    std::unique_ptr<OrtValue> hidden_states_fp32;
    if (hidden_states->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      Cast(*hidden_states, hidden_states_fp32, *model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      hidden_states = std::move(hidden_states_fp32);
    }
    const auto hidden_info = hidden_states->GetTensorTypeAndShapeInfo();
    const auto hidden_shape = hidden_info->GetShape();
    if (hidden_shape.empty() || hidden_shape.back() <= 0) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: patch_embed output has no hidden dimension");
    }
    const int64_t embed_dim = hidden_shape.back();
    const size_t hidden_element_count = hidden_info->GetElementCount();
    if (hidden_element_count % static_cast<size_t>(embed_dim) != 0 ||
        hidden_element_count / static_cast<size_t>(embed_dim) != num_patches) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: patch_embed returned " +
                               std::to_string(hidden_element_count / static_cast<size_t>(embed_dim)) +
                               " rows for an image with " + std::to_string(slice.num_patches) + " patches");
    }
    float* hidden = hidden_states->GetTensorMutableData<float>();

    std::vector<int64_t> indices;
    std::vector<float> weights;
    GetPositionInterpolationIndicesAndWeights(slice.temporal, slice.height, slice.width, num_grid_per_side_,
                                              indices, weights);
    const std::array<int64_t, 2> indices_shape{4, slice.num_patches};
    auto indices_tensor = WrapCpuTensor(indices, indices_shape);
    const char* pos_in[] = {pos_input_.c_str()};
    const OrtValue* pos_vals[] = {indices_tensor.get()};
    const char* pos_out[] = {pos_output_.c_str()};
    auto pos_embeds = std::move(pos_session_->Run(pos_run_options_.get(), pos_in, pos_vals, 1, pos_out, 1)[0]);
    if (pos_embeds->GetTensorTypeAndShapeInfo()->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      std::unique_ptr<OrtValue> pos_embeds_fp32;
      Cast(*pos_embeds, pos_embeds_fp32, *model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      pos_embeds = std::move(pos_embeds_fp32);
    }
    const auto pos_info = pos_embeds->GetTensorTypeAndShapeInfo();
    if (pos_info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: pos output must contain float elements");
    }
    if (static_cast<size_t>(slice.num_patches) > std::numeric_limits<size_t>::max() / 4 ||
        static_cast<size_t>(embed_dim) >
            std::numeric_limits<size_t>::max() / (4 * static_cast<size_t>(slice.num_patches))) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: expected pos output size overflow");
    }
    const size_t expected_pos_elements = 4 * static_cast<size_t>(slice.num_patches) * static_cast<size_t>(embed_dim);
    if (pos_info->GetElementCount() != expected_pos_elements) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: pos output has " +
                               std::to_string(pos_info->GetElementCount()) + " elements; expected " +
                               std::to_string(expected_pos_elements));
    }
    auto weighted_sum = WeightedSumOverCorners(pos_embeds->GetTensorData<float>(), weights,
                                               slice.num_patches, embed_dim);
    PermuteWithSpatialMergeAndAdd(weighted_sum, hidden, slice.temporal, slice.height, slice.width,
                                  spatial_merge_size_, embed_dim);

    auto rope = BuildRotaryPosEmb(slice.temporal, slice.height, slice.width, spatial_merge_size_, rotary_dim_);
    auto mask = BuildAttentionMask(slice.temporal, slice.height, slice.width);
    const std::array<int64_t, 2> rope_shape{slice.num_patches, rotary_dim_};
    const std::array<int64_t, 3> mask_shape{1, slice.num_patches, slice.num_patches};
    auto rope_tensor = WrapCpuTensor(rope, rope_shape);
    auto mask_tensor = WrapCpuTensor(mask, mask_shape);

    const char* merger_in[] = {merger_hidden_input_.c_str(), merger_mask_input_.c_str(), merger_rope_input_.c_str()};
    std::unique_ptr<OrtValue> converted_hidden, converted_mask, converted_rope;
    const OrtValue* merger_vals[] = {
        PrepareVisionInput(*merger_session_, merger_hidden_input_, *hidden_states,
                           converted_hidden, *model_.p_device_inputs_),
        PrepareVisionInput(*merger_session_, merger_mask_input_, *mask_tensor,
                           converted_mask, *model_.p_device_inputs_),
        PrepareVisionInput(*merger_session_, merger_rope_input_, *rope_tensor,
                           converted_rope, *model_.p_device_inputs_)};
    const char* merger_out[] = {merger_output_.c_str()};
    auto features = std::move(merger_session_->Run(merger_run_options_.get(), merger_in, merger_vals, 3, merger_out, 1)[0]);
    if (features->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::unique_ptr<OrtValue> features_fp32;
      Cast(*features, features_fp32, *model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      features = std::move(features_fp32);
    }
    const auto features_info = features->GetTensorTypeAndShapeInfo();
    const auto features_shape = features_info->GetShape();
    if (features_shape.empty() || features_shape.back() <= 0) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: merger output has no hidden dimension");
    }
    const int64_t feature_hidden_dim = features_shape.back();
    const size_t feature_element_count = features_info->GetElementCount();
    if (feature_element_count % static_cast<size_t>(feature_hidden_dim) != 0) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: merger output element count is not divisible by hidden dimension");
    }
    const size_t feature_rows = feature_element_count / static_cast<size_t>(feature_hidden_dim);
    const int64_t expected_feature_rows =
        slice.num_patches / (spatial_merge_size_ * spatial_merge_size_);
    if (feature_rows != static_cast<size_t>(expected_feature_rows)) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: merger returned " + std::to_string(feature_rows) +
                               " rows; expected " + std::to_string(expected_feature_rows));
    }
    if (merged_hidden_dim != 0 && merged_hidden_dim != feature_hidden_dim) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: merger hidden dimension differs between images");
    }
    merged_hidden_dim = feature_hidden_dim;
    if (expected_feature_rows > std::numeric_limits<int64_t>::max() - total_feature_rows) {
      throw std::runtime_error("Qwen3.5 OpenVINO vision: total image feature row count overflow");
    }
    total_feature_rows += expected_feature_rows;
    image_features.push_back(std::move(features));
  }

  if (merged_hidden_dim <= 0 || total_feature_rows <= 0 ||
      static_cast<uint64_t>(total_feature_rows) >
          std::numeric_limits<size_t>::max() / static_cast<size_t>(merged_hidden_dim) ||
      static_cast<uint64_t>(total_feature_rows) * static_cast<size_t>(merged_hidden_dim) >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("Qwen3.5 OpenVINO vision: invalid concatenated feature shape");
  }
  const std::array<int64_t, 2> concatenated_shape{total_feature_rows, merged_hidden_dim};
  image_features_ = OrtValue::CreateTensor<float>(allocator, concatenated_shape);
  float* concatenated_data = image_features_->GetTensorMutableData<float>();
  size_t row_offset = 0;
  for (const auto& features : image_features) {
    const auto features_info = features->GetTensorTypeAndShapeInfo();
    const size_t rows = features_info->GetElementCount() / static_cast<size_t>(merged_hidden_dim);
    std::memcpy(concatenated_data + row_offset * static_cast<size_t>(merged_hidden_dim),
                features->GetTensorData<float>(), rows * static_cast<size_t>(merged_hidden_dim) * sizeof(float));
    row_offset += rows;
  }
}

void Qwen3_5OpenVINOVisionMerger::Merge(OrtValue& inputs_embeds, DeviceSpan<int32_t> input_ids) {
  if (!image_features_) return;  // text-only prompt

  // Qwen marks image placeholder tokens with input_ids == image_token_id. The canonical tokens are int32
  // even when the embedding graph declares int64 input_ids.
  const auto image_token_rows = FindQwenImageTokenRows(input_ids, model_.config_->model.image_token_id);
  MergeAllImageFeatures(inputs_embeds, *image_features_, image_token_rows);
}

}  // namespace

std::unique_ptr<OpenVINOVisionMerger> CreateOpenVINOVisionMerger(const OpenVINOMultiModalModel& model,
                                                                 const GeneratorParams& /*params*/) {
  const std::string& type = model.config_->model.type;
  if (type == "gemma3") {
    return std::make_unique<Gemma3OpenVINOVisionMerger>(model);
  }
  if (type == "qwen3_5" || type == "qwen3_5_moe") {
    return std::make_unique<Qwen3_5OpenVINOVisionMerger>(model);
  }
  throw std::runtime_error("No OpenVINO vision merger registered for model type: " + type);
}

}  // namespace Generators
