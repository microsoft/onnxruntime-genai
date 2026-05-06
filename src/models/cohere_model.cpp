// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "cohere_model.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <map>

namespace Generators {

CohereEncoderState::CohereEncoderState(const WhisperModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void CohereEncoderState::SetOrReplaceInput(const char* name,
                                           std::unique_ptr<OrtValue>& source,
                                           std::unique_ptr<OrtValue>& storage) {
  // Move to encoder device.
  storage = model_.ExpandInputs(source, params_->search.num_beams);
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (std::strcmp(input_names_[i], name) == 0) {
      inputs_[i] = storage.get();
      return;
    }
  }
  input_names_.push_back(name);
  inputs_.push_back(storage.get());
}

// Update num_frames_ and (re)create hidden_states_ output for the current
// audio_features_ tensor. Returns the input slot index of the audio features.
void CohereEncoderState::UpdateForCurrentAudio() {
  auto shape_info = audio_features_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  int audio_stride = model_.config_->model.encoder.audio_stride;

  // Encoder downsamples its time axis to T_enc frames. We store num_frames_ = 2*T_enc
  // so that num_frames_/2 = T_enc matches Whisper's "frames per 2 samples" convention,
  // letting CrossCache and WhisperDecoderState reuse Whisper's sizing logic unchanged.
  // This approach enables Cohere and other future models to reuse the same decoder and cross-attention cache logic as Whisper.
  if (audio_stride > 0 && shape.size() == 2) {
    // Raw audio [batch, samples]: encoder strides by audio_stride (+1 for the trailing partial frame).
    int T_enc = static_cast<int>(shape[1]) / audio_stride + 1;
    num_frames_ = T_enc * 2;
  } else if (audio_stride > 0 && shape.size() == 3) {
    // Mel input [batch, n_mels, T_mel]: encoder downsamples mel time axis by 8.
    int T_mel = static_cast<int>(shape[2]);
    int T_enc = (T_mel - 1) / 8 + 1;
    num_frames_ = T_enc * 2;
  } else {
    throw std::runtime_error(
        "Cohere Transcribe requires audio_stride > 0 in config. "
        "Got audio_stride=" +
        std::to_string(audio_stride) +
        " with input rank=" + std::to_string(shape.size()));
  }

  // Pre-allocate hidden states output if the model has it. Models that emit
  // cross-KV cache outputs instead don't expose hidden_states; skip in that case.
  if (model_.session_info_.HasOutput(model_.config_->model.encoder.outputs.hidden_states)) {
    auto hidden_states_shape = std::array<int64_t, 3>{params_->BatchBeamSize(), GetNumFrames() / 2, model_.config_->model.encoder.hidden_size};
    hidden_states_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), hidden_states_shape,
                                            shape_info->GetElementType());
  }
}

void CohereEncoderState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  const std::string& audio_name = model_.config_->model.encoder.inputs.audio_features;
  for (const auto& [name, value] : extra_inputs) {
    if (name == audio_name) {
      SetOrReplaceInput(audio_name.c_str(), value->ort_tensor_, audio_features_);
    } else if (name == "mel_length") {
      SetOrReplaceInput("mel_length", value->ort_tensor_, mel_length_);
    }
  }
  if (audio_features_ == nullptr) {
    throw std::runtime_error("audio_features must be provided via SetInputs");
  }

  UpdateForCurrentAudio();

  // Add encoder hidden states output if the model has it
  if (hidden_states_) {
    outputs_.push_back(hidden_states_.get());
    output_names_.push_back(model_.config_->model.encoder.outputs.hidden_states.c_str());
  }
}

void CohereEncoderState::SetChunkAudioFeatures(std::shared_ptr<Tensor> audio_features_tensor, std::shared_ptr<Tensor> mel_length_tensor) {
  const std::string& audio_name = model_.config_->model.encoder.inputs.audio_features;
  SetOrReplaceInput(audio_name.c_str(), audio_features_tensor->ort_tensor_, audio_features_);

  UpdateForCurrentAudio();

  if (mel_length_tensor) {
    SetOrReplaceInput("mel_length", mel_length_tensor->ort_tensor_, mel_length_);
  }

  // Update hidden states output pointer if applicable
  if (hidden_states_) {
    for (size_t i = 0; i < output_names_.size(); ++i) {
      if (std::string(output_names_[i]) == model_.config_->model.encoder.outputs.hidden_states) {
        outputs_[i] = hidden_states_.get();
        break;
      }
    }
  }

  // Mark encoder as needing to run again
  first_run_ = true;
}

DeviceSpan<float> CohereEncoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.encoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.encoder.run_options.value());
  }
  State::Run(*model_.session_encoder_);
  return {};
}

CohereState::CohereState(const WhisperModel& model, const GeneratorParams& params, DeviceSpan<int32_t> sequence_lengths)
    : State{params, model},
      model_{model} {
  encoder_state_ = std::make_unique<CohereEncoderState>(model, params);
  // decoder_state_ and cross_cache_ created in SetExtraInputs after num_frames is known
}

void CohereState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  // Extract and store chunk data before passing to encoder
  std::vector<ExtraInput> encoder_inputs;
  std::map<int, std::shared_ptr<Tensor>> indexed_mels;
  std::map<int, std::shared_ptr<Tensor>> indexed_mel_lengths;

  for (const auto& [name, tensor] : extra_inputs) {
    if (name == "cohere_chunk_count") {
      total_chunks_ = static_cast<int>(tensor->ort_tensor_->GetTensorData<int64_t>()[0]);
      current_chunk_ = 0;
    } else if (name.substr(0, 13) == "cohere_chunk_" && name.find("mel_length") == std::string::npos) {
      // Parse index from cohere_chunk_N
      int idx = std::stoi(name.substr(13));
      indexed_mels[idx] = tensor;
    } else if (name.substr(0, 24) == "cohere_chunk_mel_length_") {
      int idx = std::stoi(name.substr(24));
      indexed_mel_lengths[idx] = tensor;
    } else {
      encoder_inputs.push_back({name, tensor});
    }
  }

  // Store chunks in sorted order
  for (auto& [idx, tensor] : indexed_mels)
    chunk_mels_.push_back(tensor);
  for (auto& [idx, tensor] : indexed_mel_lengths)
    chunk_mel_lengths_.push_back(tensor);

  encoder_state_->SetExtraInputs(encoder_inputs);

  // Now that num_frames is known, create the decoder and wire it to the encoder.
  RebuildDecoderForCurrentChunk();
}

void CohereState::RebuildDecoderForCurrentChunk() {
  int num_frames = encoder_state_->GetNumFrames();
  decoder_state_ = std::make_unique<WhisperDecoderState>(model_, *params_, num_frames);

  if (encoder_state_->HasCrossKVCacheOutputs()) {
    // Dividing by 2 such that we can directly reuse Whisper decoder.
    cross_cache_ = std::make_unique<CrossCache>(*this, num_frames / 2);
    encoder_state_->AddCrossCache(cross_cache_);
    decoder_state_->AddCrossCache(cross_cache_);
  } else {
    decoder_state_->inputs_.push_back(encoder_state_->GetHiddenStates());
    decoder_state_->input_names_.push_back(model_.config_->model.decoder.inputs.encoder_hidden_states.c_str());
  }
}

bool CohereState::AdvanceToNextChunk() {
  if (!HasMoreChunks()) return false;

  size_t chunk_idx = current_chunk_;
  auto& next_mel = chunk_mels_[chunk_idx];
  auto& next_mel_length = chunk_mel_lengths_[chunk_idx];

  encoder_state_->SetChunkAudioFeatures(next_mel, next_mel_length);

  // Only clear encoder outputs when the encoder emits cross-KV cache outputs
  // that need to be rebuilt for the new chunk. In the hidden-states path, the
  // decoder depends on encoder hidden_states remaining registered as an output.
  if (encoder_state_->HasCrossKVCacheOutputs()) {
    encoder_state_->outputs_.clear();
    encoder_state_->output_names_.clear();
  }

  RebuildDecoderForCurrentChunk();

  // Reset state for new chunk
  first_run_ = true;
  current_chunk_++;
  return true;
}

// --- Boundary cleanup at chunk seams ----------------------------------------
//
// When we concatenate the token streams of two consecutive audio chunks the
// junction can render badly because the model treats every chunk start as a
// fresh sequence. Two empirically-observed problems:
//
//  1) The model emits byte-fallback tokens for C0 control bytes at the very
//     start of every non-first chunk. We have observed token id=11 (byte
//     \v, 0x0B) and id=13 (byte \r, 0x0D) being emitted as the first two
//     tokens of chunk 2..N. The byte-level streaming detokenizer either
//     swallows the next real token's leading space (giving "scamsIn")
//     or, in the case of \r, overwrites the previous chunk's tail when
//     printed to a terminal.
//
//  2) Once those leading control bytes are removed, the next "real" token
//     may or may not carry its own leading space depending on what the
//     model decided the chunk should start with. We need to inspect it and
//     inject a single space iff it doesn't already begin with whitespace.
//
// IDs 0..15 are byte-fallback tokens for the C0 control byte range
// (0x00..0x0F: NUL, \t, \n, \v, \f, \r, ...). They never carry semantic
// content; stripping them at the chunk seam is always safe. IDs 16..255
// are byte-fallback for the rest of the byte range (printable ASCII,
// high bytes) which can appear as legitimate content, so we must NOT
// strip those. IDs 256+ are real BPE subword tokens (e.g. id=5467 = " Come").

namespace {

// Tokens with id < this are byte-fallback for C0 control bytes; strip them
// at chunk seams before deciding whether to add a connecting space.
constexpr int32_t kSpecialTokenIdMax = 16;

// Returns true if the rendered text of `token_id` does not begin with
// whitespace, i.e. a connecting space must be injected before it at a
// chunk seam to avoid "scamsIn" / "thisInto" artifacts.
bool TokenLacksLeadingSpace(int32_t token_id, const Tokenizer& tokenizer) {
  std::string s = tokenizer.Decode(std::span<const int32_t>(&token_id, 1));
  if (s.empty()) return true;
  unsigned char c0 = static_cast<unsigned char>(s[0]);
  return !(c0 == ' ' || c0 == '\t' || c0 == '\n' || c0 == '\r');
}

}  // namespace

void CohereState::CommitChunkText(const std::vector<int32_t>& chunk_tokens,
                                  bool /*is_final*/, const Tokenizer& tokenizer) {
  if (chunk_tokens.empty()) return;

  const bool is_seam = !committed_tokens_.empty();
  size_t i = 0;

  // For non-first chunks: skip leading C0 control byte-fallback tokens
  // (id < 16). The model regularly emits id=11 (\v) and id=13 (\r) at chunk
  // start; if left in, the byte-level detokenizer renders them literally and
  // \r in particular overwrites the previous chunk's tail in any terminal.
  // Then, if the first real token does not carry its own leading space,
  // inject one so the seam reads ". This" instead of ".This".
  if (is_seam) {
    while (i < chunk_tokens.size() && chunk_tokens[i] >= 0 && chunk_tokens[i] < kSpecialTokenIdMax) {
      ++i;
    }
    if (i < chunk_tokens.size() && TokenLacksLeadingSpace(chunk_tokens[i], tokenizer)) {
      std::vector<int32_t> space_tokens = tokenizer.Encode(" ");
      committed_tokens_.insert(committed_tokens_.end(), space_tokens.begin(), space_tokens.end());
    }
  }

  committed_tokens_.insert(committed_tokens_.end(), chunk_tokens.begin() + i, chunk_tokens.end());
}

DeviceSpan<int32_t> CohereState::GetCommittedSpan() const {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  return cpu->WrapMemory<int32_t>(
      std::span<int32_t>(const_cast<int32_t*>(committed_tokens_.data()), streamed_tokens_count_));
}

Tokenizer& CohereState::GetOrCreateTokenizer() {
  if (!tokenizer_) {
    tokenizer_ = model_.CreateTokenizer();
  }
  return *tokenizer_;
}

DeviceSpan<float> CohereState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (encoder_state_->IsFirstRun()) {
    encoder_state_->Run(current_length, next_tokens, next_indices);
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);
    return decoder_state_->Run(current_length, next_tokens, next_indices);
  }

  decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);
  auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
  first_run_ = false;
  return logits;
}

OrtValue* CohereState::GetInput(const char* name) {
  for (size_t i = 0; i < encoder_state_->input_names_.size(); i++) {
    if (std::strcmp(encoder_state_->input_names_[i], name) == 0)
      return encoder_state_->inputs_[i];
  }
  for (size_t i = 0; i < decoder_state_->input_names_.size(); i++) {
    if (std::strcmp(decoder_state_->input_names_[i], name) == 0)
      return decoder_state_->inputs_[i];
  }
  return State::GetInput(name);
}

OrtValue* CohereState::GetOutput(const char* name) {
  for (size_t i = 0; i < encoder_state_->output_names_.size(); i++) {
    if (std::strcmp(encoder_state_->output_names_[i], name) == 0)
      return encoder_state_->outputs_[i];
  }
  for (size_t i = 0; i < decoder_state_->output_names_.size(); i++) {
    if (std::strcmp(decoder_state_->output_names_[i], name) == 0)
      return decoder_state_->outputs_[i];
  }
  return State::GetOutput(name);
}

std::unique_ptr<State> CohereModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<CohereState>(*this, params, sequence_lengths);
}

}  // namespace Generators
