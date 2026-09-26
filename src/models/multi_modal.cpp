// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "multi_modal.h"
#include "models/io/lfm2_audio_output.h"

#include <algorithm>
#include <cstring>

namespace Generators {

MultiModalLanguageModel::MultiModalLanguageModel(std::unique_ptr<Config> config, OrtEnv& ort_env, bool vision, bool speech)
    : Model(std::move(config)) {
  ValidateMultiModalSessionDevices(*config_, p_device_->GetType(), p_device_inputs_->GetType());

  // The non-decoder models don't support graph capture because of control flow nodes, so disable graph capture for them
  if (vision) {
    vision_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(config_->model.vision.session_options.has_value() ? config_->model.vision.session_options.value() : config_->model.decoder.session_options, *vision_session_options_, true, /*disable_graph_capture=*/true);
    vision_session_ = CreateSession(ort_env, config_->model.vision.filename, vision_session_options_.get());
  }

  if (speech) {
    speech_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(config_->model.speech.session_options.has_value() ? config_->model.speech.session_options.value() : config_->model.decoder.session_options, *speech_session_options_, true, /*disable_graph_capture=*/true);
    speech_session_ = CreateSession(ort_env, config_->model.speech.filename, speech_session_options_.get());
  }

  embedding_session_options_ = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(config_->model.embedding.session_options.has_value() ? config_->model.embedding.session_options.value() : config_->model.decoder.session_options, *embedding_session_options_, true, /*disable_graph_capture=*/true);

  embedding_session_ = CreateSession(ort_env, config_->model.embedding.filename, embedding_session_options_.get());
  decoder_session_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());

  const auto& audio_output = config_->model.audio_output;
  if (!audio_output.depthformer.filename.empty() || !audio_output.embedding.filename.empty()) {
    if (audio_output.depthformer.filename.empty() || audio_output.embedding.filename.empty()) {
      throw std::runtime_error("model.audio_output needs both depthformer.filename and embedding.filename.");
    }
    // With speech output these tokens hand the turn to the depthformer; as stop tokens they would end
    // it there instead, which is what a text-only export needs and this one must not have.
    for (const int token : {audio_output.audio_start_token_id, audio_output.text_end_token_id}) {
      if (std::find(config_->model.eos_token_id.begin(), config_->model.eos_token_id.end(), token) !=
          config_->model.eos_token_id.end()) {
        throw std::runtime_error("model.eos_token_id holds " + std::to_string(token) +
                                 ", which model.audio_output uses to start speech. Remove it from eos_token_id.");
      }
    }
    depthformer_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(audio_output.depthformer.session_options.has_value() ? audio_output.depthformer.session_options.value() : config_->model.decoder.session_options, *depthformer_session_options_, true, /*disable_graph_capture=*/true);
    depthformer_session_ = CreateSession(ort_env, audio_output.depthformer.filename, depthformer_session_options_.get());
    audio_embedding_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(audio_output.embedding.session_options.has_value() ? audio_output.embedding.session_options.value() : config_->model.decoder.session_options, *audio_embedding_session_options_, true, /*disable_graph_capture=*/true);
    audio_embedding_session_ = CreateSession(ort_env, audio_output.embedding.filename, audio_embedding_session_options_.get());
  }

  session_info_.Add(*decoder_session_);
  session_info_.Add(*embedding_session_);
  if (speech) {
    session_info_.Add(*speech_session_);
  }
  if (vision) {
    session_info_.Add(*vision_session_);
  }
}

std::unique_ptr<State> MultiModalLanguageModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<MultiModalPipelineState>(*this, sequence_lengths, params);
}

MultiModalPipelineState::MultiModalPipelineState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      adapters_{std::make_shared<Adapters>(&model_)} {
  if (model_.vision_session_) {
    vision_state_ = CreateVisionState(model_, params);
  }
  if (model_.speech_session_) {
    speech_state_ = CreateSpeechState(model_, params);
  }
  embedding_state_ = CreateEmbeddingState(model_, params);
  decoder_state_ = CreateDecoderState(model_, sequence_lengths, params);
  if (model_.depthformer_session_) {
    audio_output_ = std::make_unique<Lfm2AudioOutput>(model_, params);
  }

  if (vision_state_ != nullptr && model_.config_->model.vision.adapter_filename.has_value() && num_image_tokens_ > 0) {
    const auto lora_adapter = (model_.config_->config_path / fs::path(*model_.config_->model.vision.adapter_filename)).string();
    adapters_->LoadAdapter(lora_adapter.c_str(), vision_adapter_name_);
    decoder_state_->SetActiveAdapter(adapters_.get(), vision_adapter_name_);
  } else if (speech_state_ != nullptr && model_.config_->model.speech.adapter_filename.has_value() && num_audio_tokens_ > 0) {
    const auto lora_adapter = (model_.config_->config_path / fs::path(*model_.config_->model.speech.adapter_filename)).string();
    adapters_->LoadAdapter(lora_adapter.c_str(), speech_adapter_name_);
    decoder_state_->SetActiveAdapter(adapters_.get(), speech_adapter_name_);
  }
}

void MultiModalPipelineState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  num_image_tokens_ = vision_state_ ? vision_state_->GetNumImageTokens(extra_inputs) : 0;
  num_audio_tokens_ = speech_state_ ? speech_state_->GetNumAudioTokens(extra_inputs) : 0;
  num_images_ = vision_state_ ? vision_state_->GetImageFeatureBatchSize(extra_inputs) : 0;

  if (model_.vision_session_) {
    vision_state_->SetExtraInputs(extra_inputs, num_images_, num_image_tokens_);
  }
  if (model_.speech_session_) {
    speech_state_->SetExtraInputs(extra_inputs, num_audio_tokens_);
  }
  embedding_state_->SetExtraInputs(num_images_, num_image_tokens_, num_audio_tokens_);

  // Hand any image/video grid metadata tensors to the decoder's position inputs. This is a no-op
  // for position-input implementations that don't use them.
  std::shared_ptr<Tensor> img_grid, vid_grid, sec_grid;
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::ImageGridThwName) {
      img_grid = input.tensor;
    } else if (input.name == "video_grid_thw") {
      vid_grid = input.tensor;
    } else if (input.name == "second_per_grid_ts") {
      sec_grid = input.tensor;
    }
  }
  if (img_grid || vid_grid) {
    decoder_state_->GetPositionInputs().SetGridTensors(img_grid, vid_grid, sec_grid);
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

  if (audio_output_) {
    audio_output_->BeginStep(next_tokens, is_prompt_);
  }
  embedding_state_->UpdateInputsOutputs(next_tokens, is_prompt_);

  // Prefill chunking (search.chunk_size): during the prompt stage the decoder can process the
  // prompt embeddings in several smaller runs to bound peak memory usage.
  const auto& chunk_size_opt = params_->search.chunk_size;
  const size_t num_tokens = next_tokens.size();
  const bool has_multimodal_content = num_image_tokens_ != 0 || num_audio_tokens_ != 0;
  const bool chunk_prefill = is_prompt_ && chunk_size_opt.has_value() && chunk_size_opt.value() > 0 &&
                             num_tokens > chunk_size_opt.value() && decoder_state_->SupportsPrefillChunking(has_multimodal_content);

  if (chunk_prefill) {
    decoder_state_->PrepareEmbeddingsForPrefill(num_tokens);
  } else {
    decoder_state_->UpdateInputsOutputs(next_tokens, current_length, next_indices);
  }

  if (is_prompt_) {
    if (num_image_tokens_ > 0 && vision_state_) {
      vision_state_->Run(current_length, next_tokens, next_indices);
    }
    if (num_audio_tokens_ > 0 && speech_state_) {
      speech_state_->Run(current_length, next_tokens, next_indices);
    }
    if (vision_state_) {
      embedding_state_->image_features_->ReuseFeaturesBuffer(*vision_state_->image_features_);
    }
    if (speech_state_ && num_audio_tokens_ > 0) {
      speech_state_->ReuseFeaturesBuffer(*embedding_state_->audio_features_);
    } else if (embedding_state_->audio_features_) {
      embedding_state_->audio_features_->AllocateEmptyFeatures();
    }
    embedding_state_->ReuseBuffersInDecoder(*decoder_state_);
    embedding_state_->Run(current_length, next_tokens, next_indices);

    auto logits = chunk_prefill
                      ? decoder_state_->RunPrefillWithChunking(current_length, next_tokens, next_indices, chunk_size_opt.value())
                      : decoder_state_->Run(current_length, next_tokens, next_indices);

    is_prompt_ = false;
    if (vision_state_) vision_state_.reset();  // The vision state is no longer needed in generation stage
    if (speech_state_) speech_state_.reset();  // The speech state is no longer needed in generation stage

    return audio_output_ ? SampleAudioOrText(logits) : logits;
  }

  embedding_state_->ReuseBuffersInDecoder(*decoder_state_);
  if (audio_output_ && audio_output_->HasPendingFrame()) {
    // The token is only the placeholder of the last audio frame: the decoder takes the frame itself,
    // and the embedding model would look for audio features to put in the placeholder's place.
    audio_output_->WritePendingFrame(*decoder_state_->GetInputsEmbeds().Get());
  } else {
    embedding_state_->Run(current_length, next_tokens, next_indices);
  }
  auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
  return audio_output_ ? SampleAudioOrText(logits) : logits;
}

DeviceSpan<float> MultiModalPipelineState::SampleAudioOrText(DeviceSpan<float> logits) {
  if (!audio_output_->InAudio()) {
    return logits;
  }
  const std::string& configured = model_.config_->model.decoder.outputs.hidden_states;
  const std::string name = configured.empty() ? std::string(Config::Defaults::HiddenStatesName) : configured;
  OrtValue* hidden_states = decoder_state_->GetOutput(name.c_str());
  if (!hidden_states) {
    throw std::runtime_error("Speech output needs the decoder's \"" + name +
                             "\" output. Build the decoder with --extra_options include_hidden_states=true.");
  }
  return audio_output_->SampleFrame(*hidden_states);
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
  if (audio_output_ && std::strcmp(name, "audio_codes") == 0) {
    return audio_output_->GetAudioCodes();
  }

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
