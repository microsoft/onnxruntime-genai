// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "multi_modal.h"

namespace Generators {

namespace {

int64_t GetNumImageTokens(const std::vector<GeneratorParams::Input>& extra_inputs) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::NumImageTokens) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const int64_t* num_image_tokens_data = extra_inputs[i].tensor->ort_tensor_->GetTensorData<int64_t>();
      return std::accumulate(num_image_tokens_data,
                             num_image_tokens_data + extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount(),
                             0LL);
    }
  }

  return 0;
}

int64_t GetNumAudioTokens(const std::vector<GeneratorParams::Input>& extra_inputs,
                          const std::string& audio_sizes_name) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == audio_sizes_name) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      auto type_and_shape_info = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto element_count = type_and_shape_info->GetElementCount();
      if (type_and_shape_info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* audio_sizes_data = extra_inputs[i].tensor->ort_tensor_->GetTensorData<int64_t>();
        return std::accumulate(audio_sizes_data, audio_sizes_data + element_count, 0LL);
      } else {
        throw std::runtime_error("Unsupported data type " + std::to_string(static_cast<int64_t>(type_and_shape_info->GetElementType())) + " for audio_sizes tensor. Only int64 is supported.");
      }
    }
  }

  return 0;
}

}  // namespace

MultiModalLanguageModel::MultiModalLanguageModel(std::unique_ptr<Config> config, OrtEnv& ort_env, bool vision, bool speech)
    : Model(std::move(config)) {
  // The non-decoder models don't support graph capture because of control flow nodes, so disable graph capture for them
  if (vision) {
    auto vision_session_options = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *vision_session_options, true, true);
    vision_session_ = OrtSession::Create(
        ort_env, (config_->config_path / fs::path(config_->model.vision.filename)).c_str(), vision_session_options.get());
  }

  if (speech) {
    auto speech_session_options = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *speech_session_options, true, true);
    speech_session_ = OrtSession::Create(
        ort_env, (config_->config_path / fs::path(config_->model.speech.filename)).c_str(), speech_session_options.get());
  }

  auto embedding_session_options = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *embedding_session_options, true, true);

  embedding_session_ = OrtSession::Create(
      ort_env, (config_->config_path / fs::path(config_->model.embedding.filename)).c_str(), embedding_session_options.get());
  decoder_session_ = OrtSession::Create(
      ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  InitDeviceAllocator(*decoder_session_);
  session_info_->Add(*embedding_session_);
  if (speech) {
    session_info_->Add(*speech_session_);
  }
  if (vision) {
    session_info_->Add(*vision_session_);
  }
}

std::unique_ptr<State> MultiModalLanguageModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<MultiModalPipelineState>(*this, sequence_lengths, params);
}

VisionState::VisionState(const MultiModalLanguageModel& model, const GeneratorParams& params, const int64_t num_image_tokens)
    : State{params, model},
      model_{model},
      num_image_tokens_{num_image_tokens} {
  extra_inputs_.Add(model_.vision_session_->GetInputNames());
  image_features_.Add();
}

DeviceSpan<float> VisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  const int num_images = static_cast<int>(inputs_[0]->GetTensorTypeAndShapeInfo()->GetShape()[0]);
  State::Run(*model_.vision_session_, num_images);
  return {};
}

SpeechState::SpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params, const int64_t num_audio_tokens)
    : State{params, model},
      model_{model},
      num_audio_tokens_{num_audio_tokens} {
  extra_inputs_.Add(model_.speech_session_->GetInputNames());
  audio_features_.Add();
}

DeviceSpan<float> SpeechState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  const int num_audios = static_cast<int>(inputs_[0]->GetTensorTypeAndShapeInfo()->GetShape()[0]);
  State::Run(*model_.speech_session_, num_audios);
  return {};
}

EmbeddingState::EmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params, const int64_t num_image_tokens, const int64_t num_audio_tokens)
    : State{params, model},
      model_{model},
      num_image_tokens_{num_image_tokens},
      num_audio_tokens_{num_audio_tokens} {
  input_ids_.Add();
  if (model_.vision_session_) {
    image_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Input,  // Optional model input
                                                           model_.config_->model.embedding.inputs.image_features,
                                                           num_image_tokens_);
    image_features_->Add();
  }
  if (model_.speech_session_) {
    audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Input,  // Optional model input
                                                           model_.config_->model.embedding.inputs.audio_features,
                                                           num_audio_tokens_);
    audio_features_->Add();
  }
  inputs_embeds_.Add();
}

void EmbeddingState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, bool is_prompt) {
  input_ids_.Update(next_tokens);
  if (model_.vision_session_) image_features_->Update(is_prompt);
  if (model_.speech_session_) audio_features_->Update(is_prompt);
}

DeviceSpan<float> EmbeddingState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  State::Run(*model_.embedding_session_, batch_size);
  return {};
}

DecoderState::DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params, const CapturedGraphInfo* captured_graph_info)
    : State{params, model},
      model_{model},
      captured_graph_info_{captured_graph_info},
      position_inputs_{model, *this, sequence_lengths} {
  inputs_embeds_.Add();
  position_inputs_.Add();
  logits_.Add();
  kv_cache_.Add();
}

DeviceSpan<float> DecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  int batch_size = static_cast<int>(inputs_embeds_.GetShape()[0]);
  State::Run(*model_.decoder_session_, batch_size);
  return logits_.Get();
}

void DecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices) {
  int batch_size = static_cast<int>(inputs_embeds_.GetShape()[0]);
  size_t new_length = next_tokens.size() / batch_size;
  position_inputs_.Update(next_tokens, total_length, static_cast<int>(new_length));
  kv_cache_.Update(beam_indices, total_length);
  logits_.Update(next_tokens, new_length);
  inputs_embeds_.UpdateSequenceLength(new_length);
}

MultiModalPipelineState::MultiModalPipelineState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      num_image_tokens_{GetNumImageTokens(params_->extra_inputs)},
      num_audio_tokens_{GetNumAudioTokens(params_->extra_inputs, model_.config_->model.speech.inputs.audio_sizes)},
      captured_graph_info_{model.GetCapturedGraphPool()->ReserveCapturedGraph(model, params)},
      adapters_{std::make_shared<Adapters>(&model_)} {
  if (model_.vision_session_) {
    vision_state_ = std::make_unique<VisionState>(model_, params, num_image_tokens_);
  }
  if (model_.speech_session_) {
    speech_state_ = std::make_unique<SpeechState>(model_, params, num_audio_tokens_);
  }
  embedding_state_ = std::make_unique<EmbeddingState>(model, params, num_image_tokens_, num_audio_tokens_);
  decoder_state_ = std::make_unique<DecoderState>(model_, sequence_lengths, params, captured_graph_info_.get());

  if (vision_state_ != nullptr && model_.config_->model.vision.adapter_filename.has_value() && num_image_tokens_ > 0) {
    const auto lora_adapter = (model_.config_->config_path / fs::path(*model_.config_->model.vision.adapter_filename));
    std::string lora_adapter_str = lora_adapter.string();  // Returns UTF-8 encoded string on Windows
    adapters_->LoadAdapter(lora_adapter_str.c_str(), vision_adapter_name_);
    decoder_state_->SetActiveAdapter(adapters_.get(), vision_adapter_name_);
  } else if (speech_state_ != nullptr && model_.config_->model.speech.adapter_filename.has_value() && num_audio_tokens_ > 0) {
    const auto lora_adapter = (model_.config_->config_path / fs::path(*model_.config_->model.speech.adapter_filename));
    std::string lora_adapter_str = lora_adapter.string();  // Returns UTF-8 encoded string on Windows
    adapters_->LoadAdapter(lora_adapter_str.c_str(), speech_adapter_name_);
    decoder_state_->SetActiveAdapter(adapters_.get(), speech_adapter_name_);
  }
}

DeviceSpan<float> MultiModalPipelineState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  // Pipeline state defines the pipeline of the execution of the models
  // Prompt stage:
  //   - pixel_values, [image_attention_mask], image_sizes -> |vision_model| -> image_features
  //   - audio_embeds, audio_sizes, audio_projection_mode -> |audio_model| -> audio_features
  //   - input_ids, image_features, audio_features -> |embeddings_model| -> inputs_embeds
  //   - inputs_embeds -> |decoder_model| -> logits
  // Generation stage:
  //   - input_ids, image_features, audio_features -> |embeddings_model| -> inputs_embeds
  //   - inputs_embeds -> |decoder_model| -> logits

  embedding_state_->UpdateInputsOutputs(next_tokens, is_prompt_);
  decoder_state_->UpdateInputsOutputs(next_tokens, current_length, next_indices);

  if (is_prompt_) {
    if (num_image_tokens_ > 0 && vision_state_) {
      vision_state_->Run(current_length, next_tokens, next_indices);
    }
    if (num_audio_tokens_ > 0 && speech_state_) {
      speech_state_->Run(current_length, next_tokens, next_indices);
    }
    if (vision_state_) embedding_state_->image_features_->ReuseFeaturesBuffer(vision_state_->image_features_);
    if (speech_state_) embedding_state_->audio_features_->ReuseFeaturesBuffer(speech_state_->audio_features_);
    embedding_state_->inputs_embeds_.ReuseEmbeddingsBuffer(decoder_state_->inputs_embeds_);
    embedding_state_->Run(current_length, next_tokens, next_indices);

    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    is_prompt_ = false;
    if (vision_state_) vision_state_.reset();  // The vision state is no longer needed in generation stage
    if (speech_state_) speech_state_.reset();  // The speech state is no longer needed in generation stage

    return logits;
  }

  embedding_state_->inputs_embeds_.ReuseEmbeddingsBuffer(decoder_state_->inputs_embeds_);
  embedding_state_->Run(current_length, next_tokens, next_indices);
  return decoder_state_->Run(current_length, next_tokens, next_indices);
}

OrtValue* MultiModalPipelineState::GetInput(const char* name) {
  if (vision_state_) {
    // Check if input name is in vision state's inputs
    for (size_t i = 0; i < vision_state_->input_names_.size(); i++) {
      if (std::strcmp(vision_state_->input_names_[i], name) == 0) {
        return vision_state_->inputs_[i];
      }
    }
  }

  if (speech_state_) {
    // Check if input name is in speech state's inputs
    for (size_t i = 0; i < speech_state_->input_names_.size(); i++) {
      if (std::strcmp(speech_state_->input_names_[i], name) == 0) {
        return speech_state_->inputs_[i];
      }
    }
  }

  // Check if input name is in embedding state's inputs
  for (size_t i = 0; i < embedding_state_->input_names_.size(); i++) {
    if (std::strcmp(embedding_state_->input_names_[i], name) == 0) {
      return embedding_state_->inputs_[i];
    }
  }

  // Check if input name is in decoder state's inputs
  for (size_t i = 0; i < decoder_state_->input_names_.size(); i++) {
    if (std::strcmp(decoder_state_->input_names_[i], name) == 0) {
      return decoder_state_->inputs_[i];
    }
  }

  return State::GetInput(name);
};

OrtValue* MultiModalPipelineState::GetOutput(const char* name) {
  if (vision_state_) {
    // Check if output name is in vision state's outputs
    for (size_t i = 0; i < vision_state_->output_names_.size(); i++) {
      if (std::strcmp(vision_state_->output_names_[i], name) == 0) {
        return vision_state_->outputs_[i];
      }
    }
  }

  if (speech_state_) {
    // Check if output name is in speech state's outputs
    for (size_t i = 0; i < speech_state_->output_names_.size(); i++) {
      if (std::strcmp(speech_state_->output_names_[i], name) == 0) {
        return speech_state_->outputs_[i];
      }
    }
  }

  // Check if output name is in embedding state's outputs
  for (size_t i = 0; i < embedding_state_->output_names_.size(); i++) {
    if (std::strcmp(embedding_state_->output_names_[i], name) == 0) {
      return embedding_state_->outputs_[i];
    }
  }

  // Check if output name is in decoder state's outputs
  for (size_t i = 0; i < decoder_state_->output_names_.size(); i++) {
    if (std::strcmp(decoder_state_->output_names_[i], name) == 0) {
      return decoder_state_->outputs_[i];
    }
  }

  return State::GetOutput(name);
};

}  // namespace Generators
