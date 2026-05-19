#include "../generators.h"
#include "model.h"
#include "model_type.h"
#include "qwen_position_inputs.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace Generators {

struct FillMaskFunctor {
  OrtValue* attention_mask;
  int64_t total_size;

  template <typename T>
  void operator()() {
    auto* mask_data = attention_mask->GetTensorMutableData<T>();
    std::fill_n(mask_data, total_size, static_cast<T>(1));
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

struct RewindStaticMaskFunctor {
  Tensor* attention_mask;
  size_t batch_beam_size;
  size_t index;
  size_t max_len;

  template <typename T>
  void operator()() {
    auto typed_span = attention_mask->GetDeviceSpan<T>();
    auto cpu = typed_span.CpuSpan();
    for (size_t i = 0; i < batch_beam_size; i++) {
      std::fill_n(cpu.data() + i * max_len, index, static_cast<T>(1));
      std::fill_n(cpu.data() + i * max_len + index, max_len - index, static_cast<T>(0));
    }
    typed_span.CopyCpuToDevice();
  }
};

Qwen2VLPositionInputs::Qwen2VLPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> /*sequence_lengths_unk*/)
    : model_{model},
      state_{state},
      image_token_id_{model.config_->model.image_token_id},
      video_token_id_{model.config_->model.video_token_id},
      vision_start_token_id_{model.config_->model.vision_start_token_id},
      tokens_per_second_{model.config_->model.vision.tokens_per_second},
      spatial_merge_size_{model.config_->model.vision.spatial_merge_size} {
  has_mask_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.attention_mask);
  has_posid_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.position_ids);

  ONNXTensorElementDataType mask_type = Ort::TypeToTensorType<int64_t>;
  if (has_mask_input_) {
    mask_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);
  }
  ONNXTensorElementDataType posid_type = mask_type;
  if (has_posid_input_) {
    posid_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.position_ids);
    if (has_mask_input_ && posid_type != mask_type) {
      throw std::runtime_error("position_ids & attention_mask must have the same data type");
    }
  }
  type_ = posid_type;

  if (has_posid_input_) {
    position_ids_shape_[0] = 3;
    position_ids_shape_[1] = state_.params_->search.batch_size;
    position_ids_shape_[2] = 0;
    position_ids_ = std::make_unique<Tensor>(model_.p_device_inputs_, posid_type);
  }
  if (has_mask_input_) {
    attention_mask_shape_[0] = state_.params_->search.batch_size;
    attention_mask_shape_[1] = 0;
    attention_mask_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
  }
}

void Qwen2VLPositionInputs::SetGridTensors(const std::shared_ptr<Tensor>& image_grid_thw,
                                           const std::shared_ptr<Tensor>& video_grid_thw,
                                           const std::shared_ptr<Tensor>& second_per_grid_ts) {
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
  int64_t batch_size = shape[1];
  int64_t seq_len = shape[2];

  auto position_ids = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  auto* position_data = position_ids->GetTensorMutableData<T>();

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

      if (remain_images > 0) {
        for (int64_t s = st; s < seq_len - 1; ++s) {
          if (input_ids[s] == vision_start_token_id_ && input_ids[s + 1] == image_token_id_) {
            ed_image = s + 1;
            break;
          }
        }
      }
      if (remain_videos > 0) {
        for (int64_t s = st; s < seq_len - 1; ++s) {
          if (input_ids[s] == vision_start_token_id_ && input_ids[s + 1] == video_token_id_) {
            ed_video = s + 1;
            break;
          }
        }
      }

      int64_t ed;
      int64_t t, h, w;
      float second_per_grid_t = 0.0f;

      if (ed_image < ed_video) {
        if (image_index * 3 + 2 >= image_grid_thw_span.size())
          throw std::runtime_error("Not enough image_grid_thw data for image tokens.");
        t = image_grid_thw_span[image_index * 3 + 0];
        h = image_grid_thw_span[image_index * 3 + 1];
        w = image_grid_thw_span[image_index * 3 + 2];
        second_per_grid_t = 0.0f;
        image_index++;
        remain_images--;
        ed = ed_image;
      } else {
        if (video_index * 3 + 2 >= video_grid_thw_span.size())
          throw std::runtime_error("Not enough video_grid_thw data for video tokens.");
        t = video_grid_thw_span[video_index * 3 + 0];
        h = video_grid_thw_span[video_index * 3 + 1];
        w = video_grid_thw_span[video_index * 3 + 2];
        if (second_per_grid_ts_span.empty() || video_index >= second_per_grid_ts_span.size()) {
          second_per_grid_t = 1.0f;
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
          current_pos++;
        }
      }

      st_idx = max_pos_for_batch + 1;
      int64_t vision_len = llm_grid_t * llm_grid_h * llm_grid_w;
      for (int64_t s = 0; s < vision_len; ++s) {
        int64_t gt = s / (llm_grid_h * llm_grid_w);
        int64_t gh = (s / llm_grid_w) % llm_grid_h;
        int64_t gw = s % llm_grid_w;

        T t_pos = static_cast<T>(std::round(gt * second_per_grid_t * tokens_per_second_)) + st_idx;
        T h_pos = static_cast<T>(gh) + st_idx;
        T w_pos = static_cast<T>(gw) + st_idx;

        position_data[0 * batch_size * seq_len + b * seq_len + (ed + s)] = t_pos;
        position_data[1 * batch_size * seq_len + b * seq_len + (ed + s)] = h_pos;
        position_data[2 * batch_size * seq_len + b * seq_len + (ed + s)] = w_pos;
        max_pos_for_batch = std::max({max_pos_for_batch, t_pos, h_pos, w_pos});
      }
      st = ed + vision_len;
    }

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
          current_pos++;
        }
      }
    }
    rope_deltas_.push_back(max_pos_for_batch + 1 - seq_len);
  }

  position_ids_->ort_tensor_ = model_.ExpandInputs(position_ids, state_.params_->search.num_beams);
  position_ids_shape_[1] *= state_.params_->search.num_beams;
  state_.inputs_[posid_input_index_] = position_ids_->GetOrtTensor();

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

  if (ShouldUseStaticMaskHandling()) {
    InitializeStaticMask<T>(*attention_mask);
    return;
  }

  attention_mask_->ort_tensor_ = model_.ExpandInputs(attention_mask, state_.params_->search.num_beams);
  attention_mask_shape_[0] *= state_.params_->search.num_beams;
  state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
}

template <typename T>
void Qwen2VLPositionInputs::InitializeStaticMask(OrtValue& cpu_attention_mask) {
  attention_mask_shape_[0] *= state_.params_->search.num_beams;
  attention_mask_shape_[1] = state_.params_->search.max_length;
  attention_mask_->CreateTensor(attention_mask_shape_, true);

  auto output_span = attention_mask_->GetDeviceSpan<T>();

  auto input_shape = cpu_attention_mask.GetTensorTypeAndShapeInfo()->GetShape();
  auto batch_size = input_shape[0];
  auto prompt_length = input_shape[1];
  auto num_beams = state_.params_->search.num_beams;
  auto max_length = attention_mask_shape_[1];

  const T* input_data = cpu_attention_mask.GetTensorData<T>();
  auto cpu_mirror = output_span.CpuSpan();
  std::fill(cpu_mirror.begin(), cpu_mirror.end(), static_cast<T>(0));
  for (int64_t i = 0; i < batch_size; i++) {
    const T* src = input_data + i * prompt_length;
    for (int j = 0; j < num_beams; j++) {
      T* dst = cpu_mirror.data() + ((i * num_beams) + j) * max_length;
      std::copy_n(src, prompt_length, dst);
    }
  }
  output_span.CopyCpuToDevice();

  state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
}

template <typename T>
void Qwen2VLPositionInputs::Update3DPositionIDsInPlace(int base_pos) {
  int64_t batch_size = position_ids_shape_[1];
  int64_t seq_len = position_ids_shape_[2];

  auto dst_span = position_ids_->GetDeviceSpan<T>();
  auto cpu = dst_span.CpuSpan();
  T* data = cpu.data();
  for (int64_t dim = 0; dim < 3; ++dim) {
    for (int64_t b = 0; b < batch_size; ++b) {
      T delta = static_cast<T>(base_pos + rope_deltas_[b]);
      for (int64_t s = 0; s < seq_len; ++s) {
        data[dim * batch_size * seq_len + b * seq_len + s] = delta + static_cast<T>(s);
      }
    }
  }
  dst_span.CopyCpuToDevice();
}

void Qwen2VLPositionInputs::Update3DPositionIDs(int base_pos) {
  int64_t batch_size = position_ids_shape_[1];

  if (rope_deltas_.size() != static_cast<size_t>(batch_size)) {
    throw std::runtime_error("rope_deltas size mismatch with batch_size * num_beams.");
  }

  std::vector<int64_t> desired_shape(position_ids_shape_.begin(), position_ids_shape_.end());
  if (!position_ids_->ort_tensor_ || position_ids_->GetShape() != desired_shape) {
    position_ids_->CreateTensor(position_ids_shape_, ShouldUseStaticPositionIDHandling() && position_ids_shape_[2] == 1);
    state_.inputs_[posid_input_index_] = position_ids_->GetOrtTensor();
  }

  DispatchOnType(type_, [this, base_pos]<typename T>() { Update3DPositionIDsInPlace<T>(base_pos); });
}

void Qwen2VLPositionInputs::UpdateAttentionMask(int total_length, int new_length) {
  if (ShouldUseStaticMaskHandling()) {
    if (!model_.p_device_inputs_->UpdateAttentionMask(nullptr,
                                                      attention_mask_->GetMutableRawData(),
                                                      static_cast<int>(attention_mask_shape_[0]),
                                                      new_length,
                                                      total_length,
                                                      state_.params_->search.max_length,
                                                      true,
                                                      type_)) {
      auto attention_mask_span = attention_mask_->GetByteSpan();
      GetDeviceInterface(DeviceType::CPU)->UpdateAttentionMask(nullptr,
                                                               attention_mask_span.CopyDeviceToCpu().data(),
                                                               static_cast<int>(attention_mask_shape_[0]),
                                                               new_length,
                                                               total_length,
                                                               state_.params_->search.max_length,
                                                               true,
                                                               type_);
      attention_mask_span.CopyCpuToDevice();
    }
    return;
  }

  auto attention_mask = OrtValue::CreateTensor(model_.allocator_cpu_, attention_mask_shape_, type_);

  DispatchOnType(type_, FillMaskFunctor{attention_mask.get(), attention_mask_shape_[0] * attention_mask_shape_[1]});

  attention_mask_->ort_tensor_ = model_.ExpandInputs(attention_mask, 1);
  state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
}

bool Qwen2VLPositionInputs::ShouldUseStaticInputsForGraphReplay() const {
  return state_.params_->use_graph_capture ||
         state_.params_->IsPastPresentShareBufferEnabled(model_.config_->model.type);
}

bool Qwen2VLPositionInputs::ShouldUseStaticMaskHandling() const {
  return ShouldUseStaticInputsForGraphReplay();
}

bool Qwen2VLPositionInputs::ShouldUseStaticPositionIDHandling() const {
  return ShouldUseStaticInputsForGraphReplay();
}

void Qwen2VLPositionInputs::Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) {
  if (has_posid_input_) {
    position_ids_shape_[2] = new_length;
    if (is_first_update_) {
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
      if (!ShouldUseStaticMaskHandling()) {
        attention_mask_shape_[1] = total_length;
      }
      UpdateAttentionMask(total_length, new_length);
    }
  }

  is_first_update_ = false;
}

void Qwen2VLPositionInputs::RewindTo(size_t index) {
  if (has_posid_input_) {
    position_ids_shape_[2] = static_cast<int64_t>(index);
  }
  if (has_mask_input_) {
    if (ShouldUseStaticMaskHandling()) {
      size_t max_len = static_cast<size_t>(state_.params_->search.max_length);
      if (index > max_len) {
        throw std::runtime_error("Qwen2VLPositionInputs::RewindTo: index exceeds max_length.");
      }

      size_t batch_beam_size = static_cast<size_t>(attention_mask_shape_[0]);
      DispatchOnType(type_, RewindStaticMaskFunctor{attention_mask_.get(), batch_beam_size, index, max_len});
      return;
    }
    attention_mask_shape_[1] = static_cast<int64_t>(index);
  }
}

std::unique_ptr<PositionInputs> TryCreateQwenPositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths) {
  if (ModelType::IsQwenVLFamily(state.model_.config_->model.type)) {
    return std::make_unique<Qwen2VLPositionInputs>(state.model_, state, sequence_lengths);
  }
  return nullptr;
}

}  // namespace Generators
