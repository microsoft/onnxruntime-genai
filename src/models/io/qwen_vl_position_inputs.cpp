#include "qwen_vl_position_inputs.h"

#include "generator/generators.h"
#include "models/model.h"
#include "models/model_type.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace Generators {

namespace {
template <typename Func>
void DispatchOnType(ONNXTensorElementDataType type, Func&& func) {
  if (type == Ort::TypeToTensorType<int32_t>)
    func.template operator()<int32_t>();
  else
    func.template operator()<int64_t>();
}
}  // namespace

struct UpdatePositionIdsFunctor {
  OrtValue* position_ids;
  int base_pos;
  int64_t batch_size;
  int64_t seq_len;
  const std::vector<int64_t>& rope_deltas;

  template <typename T>
  void operator()() {
    auto* data = position_ids->GetTensorMutableData<T>();
    for (int64_t dim = 0; dim < 3; ++dim) {
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
          T delta = static_cast<T>(base_pos + rope_deltas[b]);
          T pos = static_cast<T>(s);
          data[dim * batch_size * seq_len + b * seq_len + s] = delta + pos;
        }
      }
    }
  }
};

struct InitPositionIdsFunctor {
  Qwen2VLPositionInputs* self;
  DeviceSpan<int32_t> next_tokens;
  std::array<int64_t, 3> position_ids_shape;

  template <typename T>
  void operator()() {
    self->CreateAndInitialize3DPositionIDs<T>(next_tokens, position_ids_shape);
  }
};

struct InitAttentionMaskFunctor {
  Qwen2VLPositionInputs* self;
  DeviceSpan<int32_t> next_tokens;
  std::array<int64_t, 2> attention_mask_shape;

  template <typename T>
  void operator()() {
    self->CreateAndInitializeAttentionMask<T>(next_tokens, attention_mask_shape);
  }
};

// Qwen2VLPositionInputs implementation
Qwen2VLPositionInputs::Qwen2VLPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk)
    : model_{model},
      state_{state},
      image_token_id_{model.config_->model.image_token_id},
      video_token_id_{model.config_->model.video_token_id},
      vision_start_token_id_{model.config_->model.vision_start_token_id},
      tokens_per_second_{model.config_->model.vision.tokens_per_second},
      spatial_merge_size_{model.config_->model.vision.spatial_merge_size} {
  has_mask_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.attention_mask);
  has_posid_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.position_ids);

  type_ = Ort::TypeToTensorType<int64_t>;  // Default to int64 for Qwen2VL
  if (has_mask_input_) {
    type_ = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);
  }

  if (has_posid_input_) {
    ONNXTensorElementDataType posid_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.position_ids);

    // Set up 3D position IDs shape: [3, batch_size, sequence_length]
    // The 3 dimensions represent temporal, height, and width for mrope
    position_ids_shape_[0] = 3;
    position_ids_shape_[1] = state_.params_->search.batch_size;
    position_ids_shape_[2] = 0;  // Will be set during first update

    position_ids_ = std::make_unique<Tensor>(model_.p_device_inputs_, posid_type);
  }
  if (has_mask_input_) {
    attention_mask_shape_[0] = state_.params_->search.batch_size;
    attention_mask_shape_[1] = 0;  // Will be set during first update
    attention_mask_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
  }
  const auto& history_name = model_.config_->model.decoder.inputs.rotary_position_ids;
  if (!history_name.empty()) {
    const auto shape = model_.session_info_.GetInputShape(history_name);
    if (!has_posid_input_ || type_ != Ort::TypeToTensorType<int64_t> ||
        model_.session_info_.GetInputDataType(history_name) != Ort::TypeToTensorType<int64_t> ||
        shape.size() != 3 || shape[0] != 3 || state_.params_->search.max_length <= 0) {
      throw std::runtime_error("Rotary history requires int64 [3, batch, history] and current T/H/W positions");
    }
    const std::array<int64_t, 3> history_shape{
        3, state_.params_->BatchBeamSize(), state_.params_->search.max_length};
    rotary_history_ = std::make_unique<RotaryPositionHistory>(
        history_shape[1], history_shape[2], model_.config_->model.context_length - 1);
    rotary_history_value_ = OrtValue::CreateTensor(
        model_.allocator_cpu_.GetInfo(), rotary_history_->Values().data(),
        rotary_history_->Values().size_bytes(), history_shape, Ort::TypeToTensorType<int64_t>);
    rotary_history_input_ = std::make_unique<Tensor>(model_.p_device_inputs_, Ort::TypeToTensorType<int64_t>);
    rotary_history_input_->CreateTensor(history_shape, true);
  }
  position_buffer_ = CreatePositionIdsBuffer(*state_.params_);
  mask_buffer_ = CreateAttentionMaskBuffer(model_, *state_.params_, DynamicMaskUpdate::FillOnes);
}

void Qwen2VLPositionInputs::SetGridTensors(const std::shared_ptr<Tensor>& image_grid_thw,
                                           const std::shared_ptr<Tensor>& video_grid_thw,
                                           const std::shared_ptr<Tensor>& second_per_grid_ts) {
  // Typical Qwen2-VL grids are far below this threshold (usually <= 10^3); this limit is
  // intentionally conservative and blocks pathological dimensions that can explode vision_len.
  const auto validate_grid_tensor = [&](const std::shared_ptr<Tensor>& grid_tensor, const char* tensor_name) {
    if (!grid_tensor) {
      return;
    }

    if (grid_tensor->GetType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      throw std::runtime_error(std::string(tensor_name) + " must be int64.");
    }

    ValidateQwen2VLGridTensorValues(grid_tensor->GetData<int64_t>(), grid_tensor->GetElementCount(), tensor_name);
  };

  validate_grid_tensor(image_grid_thw, "image_grid_thw");
  validate_grid_tensor(video_grid_thw, "video_grid_thw");
  if (second_per_grid_ts && second_per_grid_ts->GetType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("second_per_grid_ts must be float32.");
  }

  image_grid_thw_ = image_grid_thw;
  video_grid_thw_ = video_grid_thw;
  second_per_grid_ts_ = second_per_grid_ts;
}

void Qwen2VLPositionInputs::Add() {
  if (has_posid_input_) {
    AddPositionIDs();
  }
  if (has_mask_input_) {
    AddAttentionMask();
  }
  if (rotary_history_) {
    rotary_input_index_ = state_.inputs_.size();
    state_.inputs_.push_back(rotary_history_input_->GetOrtTensor());
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.rotary_position_ids.c_str());
  }
}

void Qwen2VLPositionInputs::AddPositionIDs() {
  posid_input_index_ = state_.inputs_.size();
  state_.inputs_.push_back(position_ids_->GetOrtTensor());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.position_ids.c_str());
}

void Qwen2VLPositionInputs::AddAttentionMask() {
  mask_input_index_ = state_.inputs_.size();
  state_.inputs_.push_back(attention_mask_->GetOrtTensor());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
}

template <typename T>
void Qwen2VLPositionInputs::CreateAndInitialize3DPositionIDs(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 3> shape) {
  // Replicates the logic from HuggingFace's `get_rope_index`
  // `shape` is [3, batch_size, seq_len] (before beam expansion)
  // `next_tokens` is [batch_size, seq_len]
  int64_t batch_size = shape[1];
  int64_t seq_len = shape[2];

  auto position_ids = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  auto* position_data = position_ids->GetTensorMutableData<T>();

  // Get spans for grid_thw tensors (on CPU)
  std::span<const int64_t> image_grid_thw_span;
  if (image_grid_thw_) {
    image_grid_thw_span = std::span(image_grid_thw_->GetData<int64_t>(), image_grid_thw_->GetElementCount());
  }

  std::span<const int64_t> video_grid_thw_span;
  if (video_grid_thw_) {
    video_grid_thw_span = std::span(video_grid_thw_->GetData<int64_t>(), video_grid_thw_->GetElementCount());
  }

  std::span<const float> second_per_grid_ts_span;
  if (second_per_grid_ts_) {
    // Qwen 2.5 processor outputs float32 for this
    if (second_per_grid_ts_->GetType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("second_per_grid_ts must be float32.");
    second_per_grid_ts_span = std::span(second_per_grid_ts_->GetData<float>(), second_per_grid_ts_->GetElementCount());
  }

  auto input_ids_span = next_tokens.CpuSpan();
  int image_index = 0;
  int video_index = 0;
  rope_deltas_.clear();

  for (int64_t b = 0; b < batch_size; ++b) {
    auto input_ids = input_ids_span.subspan(b * seq_len, seq_len);

    int64_t image_nums = 0;
    int64_t video_nums = 0;

    // Count images/videos for this batch item by checking the token *after* vision_start_token_id
    for (int64_t s = 0; s < seq_len - 1; ++s) {
      if (input_ids[s] == vision_start_token_id_) {
        if (input_ids[s + 1] == image_token_id_) {
          image_nums++;
        } else if (input_ids[s + 1] == video_token_id_) {
          video_nums++;
        }
      }
    }

    int64_t st = 0;
    int64_t remain_images = image_nums;
    int64_t remain_videos = video_nums;
    T st_idx = 0;
    T max_pos_for_batch = 0;

    for (int64_t k = 0; k < image_nums + video_nums; ++k) {
      int64_t ed_image = seq_len + 1;
      int64_t ed_video = seq_len + 1;

      // Find next image_token_id (after a vision_start_token_id)
      if (remain_images > 0) {
        for (int64_t s = st; s < seq_len - 1; ++s) {
          if (input_ids[s] == vision_start_token_id_ && input_ids[s + 1] == image_token_id_) {
            ed_image = s + 1;  // Point to the image_token_id
            break;
          }
        }
      }
      // Find next video_token_id (after a vision_start_token_id)
      if (remain_videos > 0) {
        for (int64_t s = st; s < seq_len - 1; ++s) {
          if (input_ids[s] == vision_start_token_id_ && input_ids[s + 1] == video_token_id_) {
            ed_video = s + 1;  // Point to the video_token_id
            break;
          }
        }
      }

      int64_t ed;
      int64_t t, h, w;
      float second_per_grid_t = 0.0f;

      if (ed_image < ed_video) {
        // Process image
        if (image_index * 3 + 2 >= image_grid_thw_span.size())
          throw std::runtime_error("Not enough image_grid_thw data for image tokens.");
        t = image_grid_thw_span[image_index * 3 + 0];
        h = image_grid_thw_span[image_index * 3 + 1];
        w = image_grid_thw_span[image_index * 3 + 2];
        second_per_grid_t = 0.0f;  // Images have 0 time delta
        image_index++;
        remain_images--;
        ed = ed_image;
      } else {
        // Process video
        if (video_index * 3 + 2 >= video_grid_thw_span.size())
          throw std::runtime_error("Not enough video_grid_thw data for video tokens.");
        t = video_grid_thw_span[video_index * 3 + 0];
        h = video_grid_thw_span[video_index * 3 + 1];
        w = video_grid_thw_span[video_index * 3 + 2];
        if (second_per_grid_ts_span.empty() || video_index >= second_per_grid_ts_span.size()) {
          second_per_grid_t = 1.0f;  // Default from Python
        } else {
          second_per_grid_t = second_per_grid_ts_span[video_index];
        }
        video_index++;
        remain_videos--;
        ed = ed_video;
      }

      int64_t llm_grid_t = t;
      int64_t llm_grid_h = h / spatial_merge_size_;
      int64_t llm_grid_w = w / spatial_merge_size_;

      // 1. Fill Text Part
      // Text runs from `st` up to `ed-1` (which is the <|vision_start|> token)
      int64_t text_len = ed - st;
      st_idx = (k > 0 || b > 0) ? max_pos_for_batch + 1 : 0;
      T current_pos = st_idx;

      for (int64_t s = 0; s < text_len; ++s) {
        int64_t current_token_idx = st + s;
        if (input_ids[current_token_idx] == model_.config_->model.pad_token_id) {
          position_data[0 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
          position_data[1 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
          position_data[2 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
        } else {
          position_data[0 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          position_data[1 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          position_data[2 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          max_pos_for_batch = current_pos;
          current_pos++;  // Only increment position for non-pad tokens
        }
      }

      // 2. Fill Vision Part
      st_idx = max_pos_for_batch + 1;
      ValidateQwen2VLVisionLengthFitsSequence(llm_grid_t, llm_grid_h, llm_grid_w, ed, seq_len);
      int64_t vision_len = llm_grid_t * llm_grid_h * llm_grid_w;

      for (int64_t s = 0; s < vision_len; ++s) {
        int64_t gt = s / (llm_grid_h * llm_grid_w);
        int64_t gh = (s / llm_grid_w) % llm_grid_h;
        int64_t gw = s % llm_grid_w;

        // Round to nearest integer for temporal position
        // Note: huggingface code use truncation/floor (time_tensor_long = time_tensor.long() when converting time coordinates.
        // This will cause slight deviation from the reference during parity comparsion.
        T t_pos = static_cast<T>(std::round(gt * second_per_grid_t * tokens_per_second_)) + st_idx;
        T h_pos = static_cast<T>(gh) + st_idx;
        T w_pos = static_cast<T>(gw) + st_idx;

        // Vision tokens are guaranteed not to be padding
        position_data[0 * batch_size * seq_len + b * seq_len + (ed + s)] = t_pos;
        position_data[1 * batch_size * seq_len + b * seq_len + (ed + s)] = h_pos;
        position_data[2 * batch_size * seq_len + b * seq_len + (ed + s)] = w_pos;
        max_pos_for_batch = std::max({max_pos_for_batch, t_pos, h_pos, w_pos});
      }
      st = ed + vision_len;  // New start is after the vision tokens
    }

    // 3. Fill Remaining Text Part
    if (st < seq_len) {
      st_idx = (max_pos_for_batch == 0 && st == 0) ? 0 : max_pos_for_batch + 1;
      int64_t text_len = seq_len - st;
      T current_pos = st_idx;
      for (int64_t s = 0; s < text_len; ++s) {
        int64_t current_token_idx = st + s;
        if (input_ids[current_token_idx] == model_.config_->model.pad_token_id) {
          position_data[0 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
          position_data[1 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
          position_data[2 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
        } else {
          position_data[0 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          position_data[1 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          position_data[2 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          max_pos_for_batch = current_pos;
          current_pos++;  // Only increment position for non-pad tokens
        }
      }
    }
    rope_deltas_.push_back(max_pos_for_batch + 1 - seq_len);
  }

  UpdateRotaryHistory(*position_ids, 0, state_.params_->search.num_beams);
  // Move tensor to GPU and expand by num_beams
  SetPositionIDs(std::move(position_ids), state_.params_->search.num_beams);
  position_ids_shape_[1] *= state_.params_->search.num_beams;

  // Expand rope_deltas_
  std::vector<int64_t> expanded_deltas;
  for (int64_t delta : rope_deltas_) {
    for (int b = 0; b < state_.params_->search.num_beams; ++b) {
      expanded_deltas.push_back(delta);
    }
  }
  rope_deltas_ = std::move(expanded_deltas);
}

template <typename T>
void Qwen2VLPositionInputs::CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape) {
  auto attention_mask = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  auto* mask_data = attention_mask->GetTensorMutableData<T>();
  auto input_ids_span = next_tokens.CpuSpan();
  int64_t batch_size = shape[0];
  int64_t seq_len = shape[1];

  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t s = 0; s < seq_len; ++s) {
      int64_t current_token_idx = b * seq_len + s;
      mask_data[current_token_idx] = (input_ids_span[current_token_idx] == model_.config_->model.pad_token_id)
                                         ? static_cast<T>(0)
                                         : static_cast<T>(1);
    }
  }

  mask_buffer_->Initialize(*attention_mask_, attention_mask, state_.params_->search.num_beams, attention_mask_shape_);
  state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
}

void Qwen2VLPositionInputs::Update3DPositionIDs(int base_pos) {
  // This is the generation step (decode)
  // base_pos is cache_position[0]
  auto position_ids = OrtValue::CreateTensor(model_.allocator_cpu_, position_ids_shape_, type_);
  int64_t batch_size = position_ids_shape_[1];  // This is already expanded (batch*beams)
  int64_t seq_len = position_ids_shape_[2];     // This will be 1 for generation

  if (rope_deltas_.size() != static_cast<size_t>(batch_size)) {
    throw std::runtime_error("rope_deltas size mismatch with batch_size * num_beams.");
  }

  DispatchOnType(type_, UpdatePositionIdsFunctor{position_ids.get(), base_pos, batch_size, seq_len, rope_deltas_});

  UpdateRotaryHistory(*position_ids, base_pos, 1);
  SetPositionIDs(std::move(position_ids), 1);  // Already expanded over beams.
}

void Qwen2VLPositionInputs::SetPositionIDs(std::unique_ptr<OrtValue> position_ids, int num_beams) {
  position_buffer_->Assign(*position_ids_, model_, position_ids, num_beams);
  state_.inputs_[posid_input_index_] = position_ids_->GetOrtTensor();
}

void Qwen2VLPositionInputs::UpdateAttentionMask(int total_length, int new_length) {
  mask_buffer_->Update(*attention_mask_, attention_mask_shape_, total_length, new_length);
  state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
}

void Qwen2VLPositionInputs::Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) {
  if (has_posid_input_) {
    position_ids_shape_[2] = new_length;
    if (rotary_source_) {
      UpdateFromRotarySource(total_length, new_length);
    } else if (is_first_update_) {
      DispatchOnType(type_, InitPositionIdsFunctor{this, next_tokens, position_ids_shape_});
    } else {
      Update3DPositionIDs(total_length - new_length);
    }
  }

  if (has_mask_input_) {
    if (is_first_update_) {
      attention_mask_shape_[1] = new_length;
      DispatchOnType(type_, InitAttentionMaskFunctor{this, next_tokens, attention_mask_shape_});
    } else {
      UpdateAttentionMask(total_length, new_length);
    }
  }

  is_first_update_ = false;
}

void Qwen2VLPositionInputs::RewindTo(size_t index) {
  if (index == 0) {
    is_first_update_ = true;
    position_ids_shape_[1] = state_.params_->search.batch_size;
    attention_mask_shape_[0] = state_.params_->search.batch_size;
    rope_deltas_.clear();
    return;
  }
  // Partial rewinds during generation retain the prompt's mRoPE deltas.
  if (has_posid_input_) {
    position_ids_shape_[2] = static_cast<int64_t>(index);
  }
  if (has_mask_input_) {
    mask_buffer_->Rewind(*attention_mask_, attention_mask_shape_, index);
  }
  if (rotary_history_) {
    rotary_history_->Forecast(index);
    PublishRotaryHistory();
  }
  if (index == 0) {
    is_first_update_ = true;
    position_ids_shape_[1] = state_.params_->search.batch_size;
    attention_mask_shape_[0] = state_.params_->search.batch_size;
    rope_deltas_.clear();
  }
}

void Qwen2VLPositionInputs::PublishRotaryHistory() {
  auto destination = rotary_history_input_->GetDeviceSpan<int64_t>();
  auto host = destination.CpuSpan();
  std::copy(rotary_history_->Values().begin(), rotary_history_->Values().end(), host.begin());
  destination.CopyCpuToDevice();
  state_.inputs_[rotary_input_index_] = rotary_history_input_->GetOrtTensor();
}

void Qwen2VLPositionInputs::UpdateRotaryHistory(OrtValue& positions, size_t start, size_t beams) {
  if (!rotary_history_) return;
  auto info = positions.GetTensorTypeAndShapeInfo();
  const auto shape = info->GetShape();
  rotary_history_->Update(
      {positions.GetTensorData<int64_t>(), info->GetElementCount()}, start, shape[2], beams);
  PublishRotaryHistory();
}

void Qwen2VLPositionInputs::UpdateFromRotarySource(int total_length, int new_length) {
  if (!rotary_history_ || total_length < new_length || new_length <= 0) {
    throw std::runtime_error("MTP rotary positions require a configured history and a valid step length");
  }
  auto info = rotary_source_->GetTensorTypeAndShapeInfo();
  const auto shape = info->GetShape();
  if (rotary_source_->GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU ||
      info->GetElementType() != Ort::TypeToTensorType<int64_t> ||
      shape.size() != 3 || shape[0] != 3 || shape[1] != state_.params_->BatchBeamSize() || shape[2] <= 0) {
    throw std::runtime_error("Main rotary position history must be a CPU int64 [3, batch, history] tensor");
  }
  rotary_history_->CopyFrom({rotary_source_->GetTensorData<int64_t>(), info->GetElementCount()}, shape[2]);
  if (total_length > state_.params_->search.max_length) {
    throw std::runtime_error("MTP step exceeds rotary history capacity");
  }
  position_ids_shape_[1] = shape[1];
  auto positions = OrtValue::CreateTensor(model_.allocator_cpu_, position_ids_shape_, Ort::TypeToTensorType<int64_t>);
  auto* data = positions->GetTensorMutableData<int64_t>();
  const size_t capacity = state_.params_->search.max_length;
  for (size_t row = 0; row < static_cast<size_t>(3 * shape[1]); ++row) {
    const auto source = rotary_history_->Values().subspan(row * capacity + total_length - new_length, new_length);
    std::copy(source.begin(), source.end(), data + row * new_length);
  }
  if (state_.params_->use_graph_capture && new_length <= state_.params_->max_graph_capture_length) {
    auto& buffer = captured_rotary_positions_[new_length];
    if (!buffer) {
      buffer = std::make_unique<Tensor>(model_.p_device_inputs_, Ort::TypeToTensorType<int64_t>);
      buffer->CreateTensor(position_ids_shape_, true);
    }
    auto destination = buffer->GetDeviceSpan<int64_t>();
    std::copy_n(positions->GetTensorData<int64_t>(), positions->GetTensorTypeAndShapeInfo()->GetElementCount(),
                destination.CpuSpan().begin());
    destination.CopyCpuToDevice();
    state_.inputs_[posid_input_index_] = buffer->GetOrtTensor();
  } else {
    position_ids_->ort_tensor_ = model_.ExpandInputs(positions, 1);
    state_.inputs_[posid_input_index_] = position_ids_->GetOrtTensor();
  }
  PublishRotaryHistory();
}

}  // namespace Generators
