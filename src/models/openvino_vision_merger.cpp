// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "generator/generators.h"
#include "model.h"
#include "io/multi_modal_features.h"
#include "openvino_multi_modal.h"
#include "openvino_vision_merger.h"

namespace Generators {

namespace {

// Returns the OrtValue for a named extra input, or nullptr if absent.
OrtValue* FindExtraInput(const std::vector<ExtraInput>& extra_inputs, std::string_view name) {
  for (const auto& input : extra_inputs) {
    if (input.name == name) return input.tensor->GetOrtTensor();
  }
  return nullptr;
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
  explicit Gemma3OpenVINOVisionMerger(const OpenVINOMultiModalModel& model) : model_{model} {}

  void RunVision(const std::vector<ExtraInput>& extra_inputs) override;
  void Merge(OrtValue& inputs_embeds, const OrtValue& input_ids) override;

 private:
  const OpenVINOMultiModalModel& model_;
  std::unique_ptr<OrtValue> image_features_;  // produced for the current prompt (null when no image)
  std::vector<int32_t> token_type_ids_;       // host copy of token_type_ids for the current prompt
};

void Gemma3OpenVINOVisionMerger::RunVision(const std::vector<ExtraInput>& extra_inputs) {
  image_features_.reset();
  token_type_ids_.clear();

  OrtValue* pixel_values = FindExtraInput(extra_inputs, Config::Defaults::PixelValuesName);
  if (!pixel_values || !model_.ov_vision_session_) return;  // text-only prompt

  // Ensure pixel_values are float32 for the vision graph.
  std::unique_ptr<OrtValue> pixel_values_fp32;
  const OrtValue* pixel_input = pixel_values;
  if (pixel_values->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    Cast(*pixel_values, pixel_values_fp32, *model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    pixel_input = pixel_values_fp32.get();
  }

  const std::string& pixel_input_name = model_.config_->model.vision.inputs.pixel_values;
  const std::string& image_features_name = model_.config_->model.vision.outputs.image_features;
  const char* input_names[] = {pixel_input_name.c_str()};
  const OrtValue* input_values[] = {pixel_input};
  const char* output_names[] = {image_features_name.c_str()};

  auto outputs = model_.ov_vision_session_->Run(nullptr, input_names, input_values, 1, output_names, 1);
  image_features_ = std::move(outputs[0]);

  // Cache token_type_ids (int32) so Merge can locate the image rows after the text embedding runs.
  if (OrtValue* token_type_ids = FindExtraInput(extra_inputs, Config::Defaults::TokenTypeIdsName)) {
    const size_t count = token_type_ids->GetTensorTypeAndShapeInfo()->GetElementCount();
    const int32_t* data = token_type_ids->GetTensorData<int32_t>();
    token_type_ids_.assign(data, data + count);
  }
}

void Gemma3OpenVINOVisionMerger::Merge(OrtValue& inputs_embeds, const OrtValue& /*input_ids*/) {
  if (!image_features_) return;  // text-only prompt

  // Gemma marks image placeholder tokens with token_type_ids == 1.
  std::vector<int64_t> image_token_rows;
  image_token_rows.reserve(token_type_ids_.size());
  for (size_t i = 0; i < token_type_ids_.size(); ++i) {
    if (token_type_ids_[i] == 1) image_token_rows.push_back(static_cast<int64_t>(i));
  }

  MergeImageFeaturesIntoEmbeddings(inputs_embeds, *image_features_, image_token_rows);
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
  void Merge(OrtValue& inputs_embeds, const OrtValue& input_ids) override;

 private:
  const OpenVINOMultiModalModel& model_;

  std::unique_ptr<OrtSession> patch_embed_session_;
  std::unique_ptr<OrtSession> pos_session_;
  std::unique_ptr<OrtSession> merger_session_;

  // Stage I/O names, resolved from the config pipeline stage inputs/outputs arrays.
  std::string patch_embed_input_, patch_embed_output_;
  std::string pos_input_, pos_output_;
  std::string merger_hidden_input_, merger_mask_input_, merger_rope_input_, merger_output_;

  int64_t spatial_merge_size_{2};
  int64_t rotary_dim_{};        // read from the merger's rotary_pos_emb input width
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

  patch_embed_input_ = patch_embed->inputs.at(0);
  patch_embed_output_ = patch_embed->outputs.at(0);
  pos_input_ = pos->inputs.at(0);
  pos_output_ = pos->outputs.at(0);
  merger_hidden_input_ = merger->inputs.at(0);
  merger_mask_input_ = merger->inputs.at(1);
  merger_rope_input_ = merger->inputs.at(2);
  merger_output_ = merger->outputs.at(0);

  OrtSessionOptions* session_options = model_.vision_session_options_ov_.get();
  patch_embed_session_ = OrtSession::Create(model_.ort_env_, (model_.config_->config_path / fs::path(patch_embed->filename)).c_str(), session_options);
  pos_session_ = OrtSession::Create(model_.ort_env_, (model_.config_->config_path / fs::path(pos->filename)).c_str(), session_options);
  merger_session_ = OrtSession::Create(model_.ort_env_, (model_.config_->config_path / fs::path(merger->filename)).c_str(), session_options);

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

  // Ensure pixel_values are float32 for the vision graph.
  std::unique_ptr<OrtValue> pixel_values_fp32;
  const OrtValue* pixel_input = pixel_values;
  if (pixel_values->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    Cast(*pixel_values, pixel_values_fp32, *model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    pixel_input = pixel_values_fp32.get();
  }

  // Grid [t, h, w] from image_grid_thw (int64 or int32; last 3 values).
  OrtValue* grid_thw = FindExtraInput(extra_inputs, model_.config_->model.vision.inputs.image_grid_thw);
  if (!grid_thw) throw std::runtime_error("Qwen3.5 OpenVINO vision: image_grid_thw input missing.");
  int64_t t = 0, h = 0, w = 0;
  {
    auto info = grid_thw->GetTensorTypeAndShapeInfo();
    const size_t count = info->GetElementCount();
    if (count < 3) throw std::runtime_error("Qwen3.5 OpenVINO vision: image_grid_thw has fewer than 3 values.");
    if (info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const int64_t* g = grid_thw->GetTensorData<int64_t>();
      t = g[count - 3]; h = g[count - 2]; w = g[count - 1];
    } else {
      const int32_t* g = grid_thw->GetTensorData<int32_t>();
      t = g[count - 3]; h = g[count - 2]; w = g[count - 1];
    }
  }
  const int64_t num_patches = t * h * w;

  // Stage 1: patch_embed(pixel_values) -> hidden_states [N, D].
  const char* pe_in[] = {patch_embed_input_.c_str()};
  const OrtValue* pe_vals[] = {pixel_input};
  const char* pe_out[] = {patch_embed_output_.c_str()};
  auto hidden_states = std::move(patch_embed_session_->Run(nullptr, pe_in, pe_vals, 1, pe_out, 1)[0]);
  const int64_t embed_dim = hidden_states->GetTensorTypeAndShapeInfo()->GetShape().back();
  float* hidden = hidden_states->GetTensorMutableData<float>();

  // Stage 2: interpolated position embeddings, added onto hidden_states in spatial-merge order.
  std::vector<int64_t> indices;
  std::vector<float> weights;
  GetPositionInterpolationIndicesAndWeights(t, h, w, num_grid_per_side_, indices, weights);
  const int64_t indices_shape[] = {4, num_patches};
  auto indices_tensor = WrapCpuTensor(indices, indices_shape);
  const char* pos_in[] = {pos_input_.c_str()};
  const OrtValue* pos_vals[] = {indices_tensor.get()};
  const char* pos_out[] = {pos_output_.c_str()};
  auto pos_embeds = std::move(pos_session_->Run(nullptr, pos_in, pos_vals, 1, pos_out, 1)[0]);
  auto weighted_sum = WeightedSumOverCorners(pos_embeds->GetTensorData<float>(), weights, num_patches, embed_dim);
  PermuteWithSpatialMergeAndAdd(weighted_sum, hidden, t, h, w, spatial_merge_size_, embed_dim);

  // Stage 3: rotary embeddings + block-diagonal attention mask -> merger -> image features.
  auto rope = BuildRotaryPosEmb(t, h, w, spatial_merge_size_, rotary_dim_);
  auto mask = BuildAttentionMask(t, h, w);
  const int64_t rope_shape[] = {num_patches, rotary_dim_};
  const int64_t mask_shape[] = {1, num_patches, num_patches};
  auto rope_tensor = WrapCpuTensor(rope, rope_shape);
  auto mask_tensor = WrapCpuTensor(mask, mask_shape);

  const char* merger_in[] = {merger_hidden_input_.c_str(), merger_mask_input_.c_str(), merger_rope_input_.c_str()};
  const OrtValue* merger_vals[] = {hidden_states.get(), mask_tensor.get(), rope_tensor.get()};
  const char* merger_out[] = {merger_output_.c_str()};
  auto features = std::move(merger_session_->Run(nullptr, merger_in, merger_vals, 3, merger_out, 1)[0]);

  // The host merge is float-only; cast the merger output if the EP emitted a reduced-precision type.
  if (features->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    Cast(*features, image_features_, *model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  } else {
    image_features_ = std::move(features);
  }
}

void Qwen3_5OpenVINOVisionMerger::Merge(OrtValue& inputs_embeds, const OrtValue& input_ids) {
  if (!image_features_) return;  // text-only prompt

  // Qwen marks image placeholder tokens with input_ids == image_token_id. input_ids may be int32 or
  // int64 depending on the embedding graph, so dispatch on the element type.
  const int64_t image_token_id = model_.config_->model.image_token_id;
  const auto info = input_ids.GetTensorTypeAndShapeInfo();
  const size_t count = info->GetElementCount();

  std::vector<int64_t> image_token_rows;
  image_token_rows.reserve(count);
  if (info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* ids = input_ids.GetTensorData<int64_t>();
    for (size_t i = 0; i < count; ++i) {
      if (ids[i] == image_token_id) image_token_rows.push_back(static_cast<int64_t>(i));
    }
  } else {
    const int32_t* ids = input_ids.GetTensorData<int32_t>();
    for (size_t i = 0; i < count; ++i) {
      if (ids[i] == image_token_id) image_token_rows.push_back(static_cast<int64_t>(i));
    }
  }

  MergeImageFeaturesIntoEmbeddings(inputs_embeds, *image_features_, image_token_rows);
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
