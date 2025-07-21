// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "whisper.h"

namespace Generators {

WhisperModel::WhisperModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  // If provider options were explicitly added for the encoder, create session options from this config.
  if (!config_->model.encoder.session_options.provider_options.empty()) {
    encoder_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(config_->model.encoder.session_options, *encoder_session_options_, true, false);
  }

  session_encoder_ = CreateSession(ort_env, config_->model.encoder.filename,
                                   encoder_session_options_ ? encoder_session_options_.get() : session_options_.get());
  session_decoder_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());

  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_encoder_);
}

std::unique_ptr<State> WhisperModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<WhisperState>(*this, params, sequence_lengths);
}

AudioEncoderState::AudioEncoderState(const WhisperModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void AudioEncoderState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  // Add audio features
  audio_features_ = std::make_unique<AudioFeatures>(*this, model_.config_->model.encoder.inputs.audio_features, extra_inputs);
  audio_features_->Add();

  // Verify that the frame size is expected
  const int num_frames = static_cast<int>(audio_features_->GetShape()[2]);
  if (num_frames != GetNumFrames()) {
    throw new std::runtime_error("Whisper uses num_frames = 3000. The provided inputs have num_frames = " + std::to_string(num_frames));
  }

  // Add encoder hidden states
  auto hidden_states_shape = std::array<int64_t, 3>{params_->BatchBeamSize(), GetNumFrames() / 2, model_.config_->model.encoder.hidden_size};
  hidden_states_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), hidden_states_shape, audio_features_->GetType());
  outputs_.push_back(hidden_states_.get());
  output_names_.push_back(model_.config_->model.encoder.outputs.hidden_states.c_str());
}

DeviceSpan<float> AudioEncoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  State::Run(*model_.session_encoder_);
  return {};
}

WhisperDecoderState::WhisperDecoderState(const WhisperModel& model, const GeneratorParams& params, const int num_frames)
    : State{params, model},
      model_{model},
      num_frames_{num_frames} {
  input_ids_.Add();
  logits_.Add();
  kv_cache_.Add();

  // Add past sequence length
  if (HasPastSequenceLengthInput()) {
    auto past_sequence_length_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.past_sequence_length);
    auto past_sequence_length_shape = std::array<int64_t, 1>{1};
    past_sequence_length_ = OrtValue::CreateTensor(GetDeviceInterface(DeviceType::CPU)->GetAllocator(), past_sequence_length_shape, past_sequence_length_type);
    auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
    *data = 0;

    input_names_.push_back(model_.config_->model.decoder.inputs.past_sequence_length.c_str());
    inputs_.push_back(past_sequence_length_.get());
  }

  // Add cache indirection
  if (HasCacheIndirectionInput()) {
    auto cache_indirection_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.cache_indirection);
    auto cache_indirection_shape = std::array<int64_t, 3>{params_->search.batch_size, params_->search.num_beams, params_->search.max_length};
    cache_indirection_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cache_indirection_shape, cache_indirection_type);
    cache_indirection_index_ = inputs_.size();

    input_names_.push_back(model_.config_->model.decoder.inputs.cache_indirection.c_str());
    inputs_.push_back(cache_indirection_.get());

    ByteWrapTensor(*model_.p_device_inputs_, *cache_indirection_).Zero();
  }

  output_cross_qk_name_ = ComposeKeyValueName(model_.config_->model.decoder.outputs.output_cross_qk_names, 0);
}

DeviceSpan<float> WhisperDecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  // Add output QK on first run
  if (first_run_ && model_.session_info_.HasOutput(output_cross_qk_name_)) {
    output_cross_qk_type_ = model_.session_info_.GetOutputDataType(output_cross_qk_name_);
    output_cross_qk_shape_ = std::array<int64_t, 4>{params_->BatchBeamSize(), model_.config_->model.decoder.num_attention_heads, current_length, num_frames_ / 2};
    output_cross_qk_index_ = outputs_.size();

    for (int i = 0; i < model_.config_->model.decoder.num_hidden_layers; i++) {
      output_cross_qk_.emplace_back(OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), output_cross_qk_shape_, output_cross_qk_type_));
      output_cross_qk_names_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.output_cross_qk_names, i));

      output_names_.emplace_back(output_cross_qk_names_.back().c_str());
      outputs_.emplace_back(output_cross_qk_.back().get());
    }
  }

  State::Run(*model_.session_decoder_);
  return logits_.Get();
}

void WhisperDecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int current_length, bool first_update) {
  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  size_t new_length = next_tokens.size() / batch_size;
  input_ids_.Update(next_tokens);
  kv_cache_.Update(beam_indices, current_length);
  logits_.Update(next_tokens, first_run_ ? current_length : new_length);

  // Return early if this method is just initializing the above OrtValue objects and not updating them
  if (first_run_) {
    return;
  }

  if (past_sequence_length_) {
    auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
    *data = current_length - 1;
  }

  if (cache_indirection_ && params_->search.num_beams > 1 && !first_update) {
    // Only update after having run one pass through the decoder with past KV caches
    auto beam_indices_span = beam_indices.Span();
    if (beam_indices_span.empty()) {
      auto beam_indices_cpu = beam_indices.CpuSpan();
      std::iota(beam_indices_cpu.begin(), beam_indices_cpu.end(), 0);
      beam_indices.CopyCpuToDevice();
    }
    std::unique_ptr<OrtValue> new_cache_indirection;
    auto cache_indirection_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.cache_indirection);
    auto cache_indirection_shape = std::array<int64_t, 3>{params_->search.batch_size, params_->search.num_beams, params_->search.max_length};
    new_cache_indirection = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cache_indirection_shape, cache_indirection_type);

    model_.p_device_inputs_->UpdateCacheIndirection(new_cache_indirection->GetTensorMutableData<int32_t>(),
                                                    cache_indirection_->GetTensorData<int32_t>(),
                                                    beam_indices_span.data(),
                                                    params_->search.batch_size,
                                                    params_->search.num_beams,
                                                    static_cast<int>(new_length),  // sequence length in input ids & logits during each forward pass
                                                    params_->search.max_length,    // max sequence length
                                                    current_length);               // total sequence length after N iterations (prompt's sequence length + number of generated tokens)

    cache_indirection_ = std::move(new_cache_indirection);
    inputs_[cache_indirection_index_] = cache_indirection_.get();
  }

  if (output_cross_qk_.size() && output_cross_qk_shape_[2] != 1) {
    // Resize output QKs from (batch_size, num_heads, sequence_length, total_sequence_length) for audio processing
    // to (batch_size, num_heads, 1, total_sequence_length) for token generation
    output_cross_qk_shape_[2] = 1;
    for (int i = 0; i < model_.config_->model.decoder.num_hidden_layers; i++) {
      output_cross_qk_[i] = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), output_cross_qk_shape_, output_cross_qk_type_);
      outputs_[output_cross_qk_index_ + i] = output_cross_qk_[i].get();
    }
  }
}

WhisperState::WhisperState(const WhisperModel& model, const GeneratorParams& params, DeviceSpan<int32_t> sequence_lengths_unk)
    : State{params, model},
      model_{model} {
  encoder_state_ = std::make_unique<AudioEncoderState>(model, params);
  cross_cache_ = std::make_unique<CrossCache>(*this, encoder_state_->GetNumFrames() / 2);
  encoder_state_->AddCrossCache(cross_cache_);
  decoder_state_ = std::make_unique<WhisperDecoderState>(model, params, encoder_state_->GetNumFrames());
  decoder_state_->AddCrossCache(cross_cache_);

  transpose_k_cache_buffer_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_cache_->GetShape(), cross_cache_->GetType());
}

void WhisperState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  encoder_state_->SetExtraInputs(extra_inputs);

  // Check if alignment heads input exists
  void* alignment_heads_input = nullptr;
  for (const auto& [name, value] : extra_inputs) {
    if (name == Generators::Config::Defaults::AlignmentHeadsName) {
      alignment_heads_input = value.get();
    }
  }
  // Add alignment heads
  if (decoder_state_->output_cross_qk_.size() && alignment_heads_input != nullptr) {
    auto alignment_heads = std::move(reinterpret_cast<Tensor*>(alignment_heads_input)->ort_tensor_);
    if (model_.p_device_inputs_->GetType() == DeviceType::CPU) {
      alignment_heads_ = std::move(alignment_heads);
    } else {
      auto alignment_heads_type_and_shape_info = alignment_heads->GetTensorTypeAndShapeInfo();
      auto alignment_heads_type = alignment_heads_type_and_shape_info->GetElementType();  // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
      auto alignment_heads_shape = alignment_heads_type_and_shape_info->GetShape();
      alignment_heads_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), alignment_heads_shape, alignment_heads_type);

      // Since alignment_heads is a user input, we need to copy from CPU to GPU
      auto alignment_heads_src = ByteWrapTensor(*model_.p_device_inputs_, *alignment_heads);
      auto alignment_heads_dest = ByteWrapTensor(*model_.p_device_inputs_, *alignment_heads_);
      alignment_heads_dest.CopyFrom(alignment_heads_src);

      auto cross_qk_search_buffer_shape = std::array<int64_t, 4>{params_->BatchBeamSize(), alignment_heads_shape[0], params_->search.max_length, encoder_state_->GetNumFrames() / 2};
      cross_qk_search_buffer_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_qk_search_buffer_shape, decoder_state_->output_cross_qk_type_);

      // Allocate GPU buffer for storing output_cross_qk_{i} pointers
      cross_qk_ptrs_ = model_.p_device_inputs_->Allocate<void*>(model_.config_->model.decoder.num_hidden_layers);
    }
  }
}

void WhisperState::TransposeKCaches(std::vector<std::unique_ptr<OrtValue>>& kv_caches) {
  // Transpose attention K caches for `DecoderMaskedMultiHeadAttention` kernel (done on CUDA only)
  auto kv_cache_info = kv_caches[0]->GetTensorTypeAndShapeInfo();
  auto kv_cache_type = kv_cache_info->GetElementType();
  if (model_.p_device_inputs_->GetType() != DeviceType::CUDA || !(decoder_state_->UsesDecoderMaskedMHA())) {
    return;
  }

  auto kv_cache_dims = kv_cache_info->GetShape();
  auto kv_cache_element_size = Ort::SizeOf(kv_cache_type);
  auto temp_span = ByteWrapTensor(*model_.p_device_inputs_, *transpose_k_cache_buffer_);

  /* Use pre-allocated temporary buffer since we need to reformat the `K` caches for
   * `DecoderMaskedMultiHeadAttention` and we need some extra memory to do so.
   *
   * Since the self attention K caches are of size (batch_size, num_heads, past_sequence_length, head_size),
   * the cross attention K caches are of size (batch_size, num_heads, num_frames / 2, head_size), and
   * past_sequence_length <= max_sequence_length < num_frames / 2, we have pre-allocated a temporary buffer that is the
   * size of a cross attention K cache. This lets us use the same temporary buffer for both
   * the self attention and cross attention K caches.
   */

  // Transpose attention K caches for `DecoderMaskedMultiHeadAttention` kernel
  for (int i = 0; i < kv_caches.size(); i += 2) {
    auto dest_span = ByteWrapTensor(*model_.p_device_inputs_, *kv_caches[i]);

    // Treat the 'K' caches as if they are of shape [B, N, max_length, head_size / x, x]
    // and transpose each 'K' cache into [B, N, head_size / x, max_length, x], where x = 16 / sizeof(T)
    int chunk_size = static_cast<int>(16 / kv_cache_element_size);
    if (chunk_size != 4 && chunk_size != 8) {
      throw std::runtime_error("ReorderPastStatesKernelLauncher only supports float32 or float16 precision");
    }

    // Copy the original 'K' caches to a temporary buffer in order to
    // use the destination buffer to store the transposed 'K' caches
    temp_span.subspan(0, dest_span.size()).CopyFrom(dest_span);

    // Transpose each 'K' cache
    model_.p_device_inputs_->ReorderPastStates(kv_caches[i]->GetTensorMutableRawData(),
                                               transpose_k_cache_buffer_->GetTensorMutableRawData(),
                                               static_cast<int32_t>(kv_cache_dims[0]),
                                               static_cast<int32_t>(kv_cache_dims[1]),
                                               static_cast<int32_t>(kv_cache_dims[2]),
                                               static_cast<int32_t>(kv_cache_dims[3]),
                                               chunk_size);
  }
}

template <typename T>
void WhisperState::UpdateCrossQKSearchBuffer(int current_length) {
  auto output_cross_qk_size = decoder_state_->output_cross_qk_.size();
  if (output_cross_qk_size && alignment_heads_ && model_.p_device_inputs_->GetType() == DeviceType::CUDA) {
    // Collect a GPU array of T* pointers from the list of OrtValues to pass to the kernel
    auto cross_qk_ptrs_cpu = cross_qk_ptrs_.CpuSpan();
    for (int i = 0; i < output_cross_qk_size; i++) {
      cross_qk_ptrs_cpu[i] = decoder_state_->output_cross_qk_[i]->GetTensorMutableData<T>();
    }
    cross_qk_ptrs_.CopyCpuToDevice();

    model_.p_device_inputs_->CopyCrossQK(cross_qk_search_buffer_->GetTensorMutableData<T>(),
                                         cross_qk_ptrs_.Span().data(),
                                         current_length - (first_run_ ? prompt_length_ : 1),
                                         params_->BatchBeamSize(),
                                         model_.config_->model.decoder.num_hidden_layers,
                                         static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[1]),
                                         static_cast<int32_t>(alignment_heads_->GetTensorTypeAndShapeInfo()->GetShape()[0]),
                                         alignment_heads_->GetTensorData<int32_t>(),
                                         static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[3]),
                                         params_->search.max_length,
                                         first_run_ ? prompt_length_ : 1);
  }
}

template <typename T>
void WhisperState::FinalizeCrossQK(int current_length) {
  if (decoder_state_->output_cross_qk_.size() && alignment_heads_ && decoder_state_->cache_indirection_ && model_.p_device_inputs_->GetType() == DeviceType::CUDA) {
    // Instantiate final output for cross QKs
    auto num_alignment_heads = alignment_heads_->GetTensorTypeAndShapeInfo()->GetShape()[0];
    auto cross_qk_shape = std::array<int64_t, 5>{params_->search.batch_size, params_->search.num_return_sequences, num_alignment_heads, current_length - 1, encoder_state_->GetNumFrames() / 2};
    cross_qk_final_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_qk_shape, decoder_state_->output_cross_qk_type_);

    model_.p_device_inputs_->FinalizeCrossQK(current_length - 1,
                                             prompt_length_,
                                             static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[0]),
                                             params_->search.num_beams,
                                             params_->search.max_length,
                                             static_cast<int32_t>(num_alignment_heads),
                                             static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[3]),
                                             cross_qk_search_buffer_->GetTensorData<T>(),
                                             cross_qk_final_->GetTensorMutableData<T>(),
                                             params_->search.num_return_sequences,
                                             decoder_state_->cache_indirection_->GetTensorData<int32_t>());
  }
}

DeviceSpan<float> WhisperState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (encoder_state_->first_run_) {
    // Run encoder
    encoder_state_->Run(current_length, next_tokens, next_indices);

    // Initialize inputs and outputs for decoder
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);

    // Run decoder-init
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    if (first_run_ && decoder_state_->model_.session_info_.HasOutput(decoder_state_->output_cross_qk_name_)) {
      prompt_length_ = static_cast<int>(decoder_state_->output_cross_qk_shape_[2]);
    }

    if (decoder_state_->output_cross_qk_type_ == Ort::TypeToTensorType<Ort::Float16_t>) {
      UpdateCrossQKSearchBuffer<Ort::Float16_t>(current_length);
    } else {
      UpdateCrossQKSearchBuffer<float>(current_length);
    }

    return logits;
  }

  if (first_run_ && decoder_state_->UsesDecoderMaskedMHA()) {
    // Transpose the K caches only when the else branch is run for the first time.
    // Otherwise the GetOutput(present_key_{self/cross}_{i}) method returns transposed K caches.
    TransposeKCaches(cross_cache_->GetValues());
    TransposeKCaches(decoder_state_->kv_cache_.GetPresents());
  }

  // Update inputs and outputs for decoder
  decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);

  // Run decoder-with-past
  auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

  if (decoder_state_->output_cross_qk_type_ == Ort::TypeToTensorType<Ort::Float16_t>) {
    UpdateCrossQKSearchBuffer<Ort::Float16_t>(current_length);
  } else {
    UpdateCrossQKSearchBuffer<float>(current_length);
  }

  first_run_ = false;
  return logits;
}

void WhisperState::Finalize(int current_length) {
  if (decoder_state_->output_cross_qk_.size() && alignment_heads_) {
    if (decoder_state_->output_cross_qk_type_ == Ort::TypeToTensorType<Ort::Float16_t>) {
      FinalizeCrossQK<Ort::Float16_t>(current_length);
    } else {
      FinalizeCrossQK<float>(current_length);
    }
  }
}

OrtValue* WhisperState::GetInput(const char* name) {
  // Check if input name is in encoder state's inputs
  for (size_t i = 0; i < encoder_state_->input_names_.size(); i++) {
    if (std::strcmp(encoder_state_->input_names_[i], name) == 0) {
      return encoder_state_->inputs_[i];
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

OrtValue* WhisperState::GetOutput(const char* name) {
  // Check if output name is in encoder state's outputs
  for (size_t i = 0; i < encoder_state_->output_names_.size(); i++) {
    if (std::strcmp(encoder_state_->output_names_[i], name) == 0) {
      return encoder_state_->outputs_[i];
    }
  }

  // Check if output name is in decoder state's outputs
  for (size_t i = 0; i < decoder_state_->output_names_.size(); i++) {
    if (std::strcmp(decoder_state_->output_names_[i], name) == 0) {
      // Note: K caches will be transposed when returned
      return decoder_state_->outputs_[i];
    }
  }

  // cross_qk_final_ is an onnxruntime-genai maintained buffer that
  // is not part of the model's outputs, so we need to check for it here.
  if (std::strcmp("cross_qk", name) == 0) {
    return cross_qk_final_.get();
  }

  // cross_qk_search_buffer_ is an onnxruntime-genai maintained buffer that
  // is not part of the model's outputs, so we need to check for it here.
  if (std::strcmp("cross_qk_search", name) == 0) {
    return cross_qk_search_buffer_.get();
  }

  return State::GetOutput(name);
};

}  // namespace Generators
