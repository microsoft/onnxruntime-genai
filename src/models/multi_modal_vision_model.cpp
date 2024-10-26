// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "multi_modal_vision_model.h"

namespace Generators {

namespace {

int64_t GetNumImageTokens(const std::vector<GeneratorParams::Input>& extra_inputs,
                          const std::string& pixel_values_name,
                          const std::string& image_sizes_name) {
  std::shared_ptr<Tensor> pixel_values;
  std::shared_ptr<Tensor> image_sizes;
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == pixel_values_name) {
      pixel_values = extra_inputs[i].tensor;
    } else if (extra_inputs[i].name == image_sizes_name) {
      image_sizes = extra_inputs[i].tensor;
    }
  }

  if (!image_sizes || !image_sizes->ort_tensor_) {
    // This prompt does not have any images.
    return 0;
  }

  auto image_sizes_shape = image_sizes->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
  auto num_images = pixel_values->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape()[0];
  if (image_sizes_shape != std::vector<int64_t>{num_images, 2}) {
    std::string wrong_image_sizes_shape = "(";
    for (int i = 0; i < image_sizes_shape.size(); i++) {
      wrong_image_sizes_shape += std::to_string(image_sizes_shape[i]);
      std::string eos = (i != image_sizes_shape.size() - 1) ? ", " : ")";
      wrong_image_sizes_shape += eos;
    }
    throw std::runtime_error("image_sizes tensor must be of shape (num_images, 2), got " + wrong_image_sizes_shape);
  }

  auto image_sizes_data = image_sizes->ort_tensor_->GetTensorMutableData<int64_t>();
  int64_t num_image_tokens = 0;
  for (int i = 0; i < num_images; i++) {
    int64_t h = image_sizes_data[i * num_images] / 336;
    int64_t w = image_sizes_data[i * num_images + 1] / 336;
    num_image_tokens += ((h * w + 1) * 144) + 1 + ((h + 1) * 12);
  }
  return num_image_tokens;
}

}  // namespace

MultiModalVisionModel::MultiModalVisionModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  // The embedding and vision models don't support graph capture because of control flow nodes, so disable graph capture for them
  auto vision_session_options = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *vision_session_options, true, true);

  auto embedding_session_options = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *embedding_session_options, true, true);

  embedding_session_ = OrtSession::Create(
      ort_env, (config_->config_path / fs::path(config_->model.embedding.filename)).c_str(), embedding_session_options.get());
  vision_session_ = OrtSession::Create(
      ort_env, (config_->config_path / fs::path(config_->model.vision.filename)).c_str(), vision_session_options.get());
  decoder_session_ = OrtSession::Create(
      ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  InitDeviceAllocator(*decoder_session_);
  session_info_->Add(*embedding_session_);
  session_info_->Add(*vision_session_);
}

std::unique_ptr<State> MultiModalVisionModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<MultiModalPipelineState>(*this, sequence_lengths, params);
}

EmbeddingState::EmbeddingState(const MultiModalVisionModel& model, const GeneratorParams& params, const int64_t num_image_tokens)
    : State{params, model},
      model_{model},
      num_image_tokens_{num_image_tokens} {
  input_ids_.Add();
  image_features_.Add();
  inputs_embeds_.Add();
}

void EmbeddingState::UpdateInputsAndOutputs(DeviceSpan<int32_t> next_tokens) {
  input_ids_.Update(next_tokens);
  image_features_.Update();
}

DeviceSpan<float> EmbeddingState::Run(int current_length, DeviceSpan<int32_t> next_tokens, DeviceSpan<int32_t> next_indices) {
  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  State::Run(*model_.embedding_session_, batch_size);

  return {};
}

VisionState::VisionState(const MultiModalVisionModel& model, const GeneratorParams& params, const int64_t num_image_tokens)
    : State{params, model},
      model_{model},
      num_image_tokens_{num_image_tokens} {
  extra_inputs_.Add();
  image_features_.Add();
}

DeviceSpan<float> VisionState::Run(int current_length, DeviceSpan<int32_t> next_tokens, DeviceSpan<int32_t> next_indices) {
  const int num_images = static_cast<int>(inputs_[0]->GetTensorTypeAndShapeInfo()->GetShape()[0]);
  State::Run(*model_.vision_session_, num_images);

  return {};
}

DecoderState::DecoderState(const MultiModalVisionModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params, const CapturedGraphInfo* captured_graph_info)
    : State{params, model},
      model_{model},
      captured_graph_info_{captured_graph_info},
      position_inputs_{model, *this, sequence_lengths} {
  inputs_embeds_.Add();
  position_inputs_.Add();
  logits_.Add();
  kv_cache_.Add();
}

DeviceSpan<float> DecoderState::Run(int current_length, DeviceSpan<int32_t> next_tokens, DeviceSpan<int32_t> next_indices) {
  int batch_size = static_cast<int>(inputs_embeds_.GetShape()[0]);
  State::Run(*model_.decoder_session_, batch_size);
  return logits_.Get();
}

void DecoderState::UpdateInputsAndOutputs(int current_length, DeviceSpan<int32_t> beam_indices) {
  position_inputs_.Update(current_length);
  kv_cache_.Update(beam_indices, current_length);
  logits_.Update();
  inputs_embeds_.UpdateSequenceLength();
}

MultiModalPipelineState::MultiModalPipelineState(const MultiModalVisionModel& model,
                                                 DeviceSpan<int32_t> sequence_lengths_unk,
                                                 const GeneratorParams& params)
    : State{params, model},
      model_{model},
      num_image_tokens_{GetNumImageTokens(params_->extra_inputs, model_.config_->model.vision.inputs.pixel_values, model_.config_->model.vision.inputs.image_sizes)},
      captured_graph_info_{model.GetCapturedGraphPool()->ReserveCapturedGraph(model, params)} {
  embedding_state_ = std::make_unique<EmbeddingState>(model, params, num_image_tokens_);
  vision_state_ = std::make_unique<VisionState>(model_, params, num_image_tokens_);
  decoder_state_ = std::make_unique<DecoderState>(model_, sequence_lengths_unk, params, captured_graph_info_.get());
}

DeviceSpan<float> MultiModalPipelineState::Run(int current_length, DeviceSpan<int32_t> next_tokens,
                                               DeviceSpan<int32_t> next_indices) {
  // Pipeline state defines the pipeline of the execution of the models
  // Prompt stage:
  //   - pixel_values, image_sizes -> |vision_model| -> image_features
  //   - input_ids, image_features -> |embeddings_model| -> inputs_embeds
  //   - inputs_embeds -> |decoder_model| -> logits
  // Generation stage:
  //   - input_ids, image_features -> |embeddings_model| -> inputs_embeds
  //   - inputs_embeds -> |decoder_model| -> logits
  if (is_prompt_) {
    if (num_image_tokens_ > 0) {
      vision_state_->Run(current_length, next_tokens, next_indices);
    }
    embedding_state_->image_features_.ReuseImageFeaturesBuffer(vision_state_->image_features_);
    embedding_state_->inputs_embeds_.ReuseEmbeddingsBuffer(decoder_state_->inputs_embeds_);
    embedding_state_->Run(current_length, next_tokens, next_indices);

    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    is_prompt_ = false;
    vision_state_.reset();  // The vision state is no longer needed in generation stage

    return logits;
  }

  embedding_state_->UpdateInputsAndOutputs(next_tokens);
  decoder_state_->UpdateInputsAndOutputs(current_length, next_indices);

  embedding_state_->inputs_embeds_.ReuseEmbeddingsBuffer(decoder_state_->inputs_embeds_);
  embedding_state_->Run(current_length, next_tokens, next_indices);

  return decoder_state_->Run(current_length, next_tokens, next_indices);
}

}  // namespace Generators
