// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "multi_modal_vision_model.h"

namespace Generators {

namespace {

RoamingArray<float> MakeDummy() {
  return RoamingArray<float>();
}

void Select(std::span<const int32_t> input_ids, OrtValue* hidden_states, OrtValue* visual_features,
            int32_t num_img_tokens, int32_t hidden_size, DeviceType device_type,
            cudaStream_t cuda_stream) {
  // Assme batch_size = 1
  constexpr int32_t min_input_id = -1000000000;
  constexpr int64_t expected_batch_size = 1;

  // Find the position in the input_ids that corresponds to the start of the image tokens.
  // Image tokens are represented by negative values in the input_ids.
  const int64_t sequence_length = input_ids.size();
  int32_t image_position_start{};
  for (int64_t idx = 0; idx < sequence_length; ++idx) {
    if (input_ids[idx] < 0 && input_ids[idx] > min_input_id) {
      image_position_start = idx;
      break;
    }
  }

  // Replace the positions in the hidden_states tensor that correspond to the image tokens
  // with the visual features tensor.
  const int32_t start_pos = image_position_start * hidden_size;
  const int32_t element_count = num_img_tokens * hidden_size;
  const int32_t hidden_states_element_count = static_cast<int32_t>(sequence_length) * hidden_size;

  switch (device_type) {
    case DeviceType::CPU: {
      auto target = cpu_span<float>(hidden_states->GetTensorMutableData<float>(), hidden_states_element_count)
                        .subspan(start_pos, element_count);
      auto source = cpu_span<float>(visual_features->GetTensorMutableData<float>(), element_count);
      std::copy(source.begin(), source.end(), target.begin());
      break;
    }
#if USE_CUDA
    case DeviceType::CUDA: {
      auto target = gpu_span<uint16_t>(hidden_states->GetTensorMutableData<uint16_t>(), hidden_states_element_count)
                        .subspan(start_pos, element_count);
      auto source = gpu_span<uint16_t>(visual_features->GetTensorMutableData<uint16_t>(), element_count);
      CudaCheck() == cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(),
                                     cudaMemcpyDeviceToDevice, cuda_stream);
      break;
    }
#endif
    default:
      throw std::runtime_error("Unsupported device type for Select.");
  }
}

int64_t GetNumImageTokens(const std::vector<GeneratorParams::Input>& extra_inputs,
                          const std::string& image_sizes_name) {
  std::shared_ptr<Tensor> image_sizes;
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == image_sizes_name) {
      image_sizes = extra_inputs[i].tensor;
      break;
    }
  }

  if (!image_sizes || !image_sizes->ort_tensor_) {
    // This prompt does not have any images.
    return 0;
  }

  if (image_sizes->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape() != std::vector<int64_t>{1, 2}) {
    throw std::runtime_error("image_sizes tensor must have 2 elements");
  }

  auto image_sizes_data = image_sizes->ort_tensor_->GetTensorMutableData<int64_t>();
  const int64_t h = image_sizes_data[0] / 336;
  const int64_t w = image_sizes_data[1] / 336;
  return ((h * w + 1) * 144) + 1 + ((h + 1) * 12);
}

std::unique_ptr<OrtValue> GetVisualFeatures(OrtAllocator& device_allocator, const SessionInfo& session_info,
                                            const std::string& visual_features_name, int32_t hidden_size,
                                            int64_t num_image_tokens) {
  constexpr int32_t batch_size = 1;
  if (!session_info.HasOutput(visual_features_name)) {
    throw std::runtime_error("Visual features output not found in the model");
  }

  auto type = session_info.GetOutputDataType(visual_features_name);

  std::vector<int64_t> shape = {batch_size, num_image_tokens, hidden_size};
  std::unique_ptr<OrtValue> visual_features;

  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      visual_features = OrtValue::CreateTensor<float>(device_allocator, shape);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      visual_features = OrtValue::CreateTensor<Ort::Float16_t>(device_allocator, shape);
      break;
    default:
      throw std::runtime_error("Unsupported data type for visual features: " + std::to_string(type));
  }

  return visual_features;
}

}  // namespace

MultiModalVisionModel::MultiModalVisionModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  embedding_session_ = OrtSession::Create(
      ort_env, (config_->config_path / config_->model.embedding.filename).c_str(), session_options_.get());
  vision_session_ = OrtSession::Create(
      ort_env, (config_->config_path / config_->model.vision.filename).c_str(), session_options_.get());
  decoder_session_ = OrtSession::Create(
      ort_env, (config_->config_path / config_->model.decoder.filename).c_str(), session_options_.get());

  InitDeviceAllocator(*decoder_session_);
  session_info_->Add(*embedding_session_);
  session_info_->Add(*vision_session_);
}

std::unique_ptr<State> MultiModalVisionModel::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<MultiModalPipelineState>(*this, sequence_lengths, params);
}

EmbeddingState::EmbeddingState(const MultiModalVisionModel& model, const GeneratorParams& params)
    : State{params},
      model_{model} {
  input_ids_.Add();
  inputs_embeds_.Add();
}

void EmbeddingState::UpdateInputsAndOutputs(RoamingArray<int32_t> next_tokens) {
  input_ids_.Update(next_tokens);
  inputs_embeds_.UpdateSequenceLength();
}

RoamingArray<float> EmbeddingState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  State::Run(*model_.embedding_session_, *model_.run_options_);

  return MakeDummy();
}

VisionState::VisionState(const MultiModalVisionModel& model, const GeneratorParams& params)
    : State{params},
      model_{model} {
  extra_inputs_.Add();
  num_image_tokens_ = GetNumImageTokens(params_->extra_inputs, model_.config_->model.vision.inputs.image_sizes);
  if (num_image_tokens_ > 0) {
    visual_features_ = GetVisualFeatures(*model_.allocator_device_, *model_.session_info_,
                                         model_.config_->model.vision.outputs.visual_features,
                                         params_->hidden_size, num_image_tokens_);
    output_names_.push_back(model_.config_->model.vision.outputs.visual_features.c_str());
    outputs_.push_back(visual_features_.get());
  }
}

RoamingArray<float> VisionState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  State::Run(*model_.vision_session_, *model_.run_options_);

  return MakeDummy();
}

DecoderState::DecoderState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params},
      model_{model},
      position_inputs_{model, *this, sequence_lengths} {
  inputs_embeds_.Add();
  position_inputs_.Add();
  logits_.Add();
  kv_cache_.Add();
}

RoamingArray<float> DecoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  State::Run(*model_.decoder_session_, *model_.run_options_);

  return logits_.Get();
}

void DecoderState::UpdateInputs(int current_length, RoamingArray<int32_t> beam_indices) {
  position_inputs_.Update(current_length);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
}

MultiModalPipelineState::MultiModalPipelineState(const MultiModalVisionModel& model,
                                                 RoamingArray<int32_t> sequence_lengths_unk,
                                                 const GeneratorParams& params)
    : State{params},
      model_{model},
      embedding_state_{std::make_unique<EmbeddingState>(model, params)},
      vision_state_{std::make_unique<VisionState>(model_, params)},
      decoder_state_{std::make_unique<DecoderState>(model_, sequence_lengths_unk, params)} {
}

RoamingArray<float> MultiModalPipelineState::Run(int current_length, RoamingArray<int32_t> next_tokens,
                                                 RoamingArray<int32_t> next_indices) {
  // Pipeline state defines the pipeline of the execution of the models
  // Prompt stage:
  //   - input_ids -> |embeddings_model| -> |inputs_embeds|
  //   - pixel_values, img_sizes -> |vision_model| -> |inputs_embeds|
  //   - inputs_embeds, visual_features -> |Select| -> |inputs_embeds|
  //   - inputs_embeds -> |decoder_model| -> |logits|
  // Generation stage:
  //   - input_ids -> |embeddings_model| -> |inputs_embeds|
  //   - inputs_embeds -> |decoder_model| -> |logits|
  if (is_prompt_) {
    embedding_state_->Run(current_length, next_tokens, next_indices);
    if (vision_state_->num_image_tokens_ > 0) {
      vision_state_->Run(current_length, next_tokens, next_indices);

      // Run the select logic
      Select(params_->input_ids, embedding_state_->inputs_embeds_.Get(),
             vision_state_->visual_features_.get(), vision_state_->num_image_tokens_,
             params_->hidden_size, params_->device_type, params_->cuda_stream);
    }

    decoder_state_->inputs_embeds_ = embedding_state_->inputs_embeds_;
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    is_prompt_ = false;
    vision_state_.reset();  // The vision state is no longer needed in generation stage

    return logits;
  }

  embedding_state_->UpdateInputsAndOutputs(next_tokens);
  decoder_state_->UpdateInputs(current_length, next_indices);

  embedding_state_->Run(current_length, next_tokens, next_indices);
  decoder_state_->inputs_embeds_ = embedding_state_->inputs_embeds_;
  return decoder_state_->Run(current_length, next_tokens, next_indices);
}

}  // namespace Generators
