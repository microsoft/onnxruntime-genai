// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "cohere_model.h"

#include <algorithm>
#include <cctype>
#include <map>

namespace Generators {

CohereEncoderState::CohereEncoderState(const WhisperModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

// Update num_frames_ and (re)create hidden_states_ output for the current
// audio_features_ tensor. Returns the input slot index of the audio features.
void CohereEncoderState::UpdateForCurrentAudio() {
  auto shape_info = audio_features_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  int audio_stride = model_.config_->model.encoder.audio_stride;

  if (audio_stride > 0 && shape.size() == 2) {
    // Raw audio input [batch, samples]
    int T_enc = static_cast<int>(shape[1]) / audio_stride + 1;
    num_frames_ = T_enc * 2;  // *2 so GetNumFrames()/2 == T_enc for CrossCache
  } else if (audio_stride > 0 && shape.size() == 3) {
    // Mel input [batch, n_mels, T_mel]
    int T_mel = static_cast<int>(shape[2]);
    int T_enc = (T_mel - 1) / 8 + 1;
    num_frames_ = T_enc * 2;
  } else {
    throw std::runtime_error("Cohere Transcribe requires audio_stride > 0 in config. "
                             "Got audio_stride=" + std::to_string(audio_stride) +
                             " with input rank=" + std::to_string(shape.size()));
  }

  if (model_.session_info_.HasOutput(model_.config_->model.encoder.outputs.hidden_states)) {
    auto hidden_states_shape = std::array<int64_t, 3>{params_->BatchBeamSize(), GetNumFrames() / 2, model_.config_->model.encoder.hidden_size};
    hidden_states_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), hidden_states_shape,
                                            shape_info->GetElementType());
  }
}

void CohereEncoderState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  const std::string& audio_name = model_.config_->model.encoder.inputs.audio_features;

  // Add audio features (CPU -> encoder device).
  for (const auto& [name, value] : extra_inputs) {
    if (name == audio_name) {
      audio_features_ = model_.ExpandInputs(value->ort_tensor_, params_->search.num_beams);
      break;
    }
  }
  if (audio_features_ == nullptr) {
    throw std::runtime_error("audio_features must be provided via SetInputs");
  }

  input_names_.push_back(audio_name.c_str());
  inputs_.push_back(audio_features_.get());

  UpdateForCurrentAudio();

  // Add mel_length input if provided. Route through ExpandInputs so it ends
  // up on the same device as the encoder (CUDA copies CPU->device here).
  for (const auto& [name, value] : extra_inputs) {
    if (name == "mel_length") {
      mel_length_ = model_.ExpandInputs(value->ort_tensor_, params_->search.num_beams);
      input_names_.push_back("mel_length");
      inputs_.push_back(mel_length_.get());
      break;
    }
  }

  // Add encoder hidden states output if the model has it
  if (hidden_states_) {
    outputs_.push_back(hidden_states_.get());
    output_names_.push_back(model_.config_->model.encoder.outputs.hidden_states.c_str());
  }
}

void CohereEncoderState::SetChunkAudioFeatures(std::shared_ptr<Tensor> audio_features_tensor, std::shared_ptr<Tensor> mel_length_tensor) {
  // Replace audio features input. Copy CPU -> encoder device via ExpandInputs.
  audio_features_ = model_.ExpandInputs(audio_features_tensor->ort_tensor_, params_->search.num_beams);

  UpdateForCurrentAudio();

  const std::string& audio_name = model_.config_->model.encoder.inputs.audio_features;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (std::string(input_names_[i]) == audio_name) {
      inputs_[i] = audio_features_.get();
      break;
    }
  }

  // Replace mel_length (CPU -> device)
  if (mel_length_tensor) {
    mel_length_ = model_.ExpandInputs(mel_length_tensor->ort_tensor_, params_->search.num_beams);
    for (size_t i = 0; i < input_names_.size(); ++i) {
      if (std::string(input_names_[i]) == "mel_length") {
        inputs_[i] = mel_length_.get();
        break;
      }
    }
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

// --- CohereState ---

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
// Chunks are emitted by the processor without audio overlap. The model tends
// to terminate each chunk with sentence-final punctuation even when the audio
// continues mid-thought, so naive concatenation produces seams like
//     "...payment related scams. In the UK come from..."
// This is how the original PyTorch model works as well and how it was trained.
// However, to avoid this and minimize the impact we apply the following heuristic
// Strip trailing whitespace + punctuation + special tokens
// from each non-final chunk before committing. Punctuation set is
// multilingual (ASCII + smart-quotes + ellipsis/em-dash).
//
// IDs 0..15 are this tokenizer's structural/control tokens plus single-byte
// fallback tokens. In practice the model emits id=11 (\v) and id=13 (\r) at
// the start of every non-first chunk, and they make the streaming
// detokenizer collapse the leading space of the next real token (so
// "...scams" + "<\v><\r> In the UK..." renders as "scamsIn the UK"). The
// remaining ids in this range are control tokens (BOS/EOS/pad,
// <|startoftranscript|>, <|pnc|>/<|nopnc|>, <|itn|>/<|noitn|>, timestamp,
// diarize, spkchange, audioseparator) which would render as literal
// "<|...|>" markup if streamed. IDs 16+ are categorical tags (emotion,
// language, etc.) which the model wouldn't normally emit mid-decode.
// We treat id < 16 as "always strip" at chunk seams (both leading and
// trailing) to keep the streamed text clean.

namespace {

constexpr int32_t kSpecialTokenIdMax = 16;  // IDs 0..15 are structural/control tokens; see comment above.

// Single-token decode helper. Returns the textual rendering of one token.
std::string DecodeOne(const Tokenizer& tokenizer, int32_t token_id) {
  return tokenizer.Decode(std::span<const int32_t>(&token_id, 1));
}

// Returns true if a space token must be inserted before `tokens` so that, when
// concatenated with the previously-committed text, the seam reads with a
// single space between words. We make this decision by decoding the first
// non-special token of the chunk and checking whether its rendered text
// already begins with whitespace. If it does, the streamer will already emit
// the gap and we must NOT add another space (else double space). If it does
// not (e.g. control tokens were emitted at chunk start that swallow the
// leading space, or the first real token is something like "19"), we need
// to add one.
bool DoINeedToAddSpace(const std::vector<int32_t>& tokens, const Tokenizer& tokenizer) {
  size_t first = 0;
  while (first < tokens.size() && tokens[first] >= 0 && tokens[first] < kSpecialTokenIdMax) {
    ++first;
  }
  if (first >= tokens.size()) return false;
  std::string s = DecodeOne(tokenizer, tokens[first]);
  if (s.empty()) return true;
  unsigned char c0 = static_cast<unsigned char>(s[0]);
  return !(c0 == ' ' || c0 == '\t' || c0 == '\n' || c0 == '\r');
}

}  // namespace

void CohereState::CommitChunkText(const std::vector<int32_t>& chunk_tokens,
                                  bool is_final, const Tokenizer& tokenizer) {
  if (chunk_tokens.empty()) return;

  std::vector<int32_t> cleaned(chunk_tokens);

  // For non-first chunks: first drop any leading special/control tokens
  // (id < 16). The model regularly emits id=11 (\v) and id=13 (\r) at chunk
  // start; if left in, the byte-level detokenizer renders them literally and
  // \r in particular overwrites the previous chunk's tail in any terminal.
  // After stripping, decide whether the first real token will already render
  // with a leading space; if not, insert a single space token so the seam
  // reads as ". This" instead of ".This".
  if (!committed_tokens_.empty()) {
    size_t first = 0;
    while (first < cleaned.size() && cleaned[first] >= 0 && cleaned[first] < kSpecialTokenIdMax) {
      ++first;
    }
    cleaned.erase(cleaned.begin(), cleaned.begin() + first);
    if (DoINeedToAddSpace(cleaned, tokenizer)) {
      std::vector<int32_t> space_tokens = tokenizer.Encode(" ");
      cleaned.insert(cleaned.begin(), space_tokens.begin(), space_tokens.end());
    }
  }

  committed_tokens_.insert(committed_tokens_.end(), cleaned.begin(), cleaned.end());
}


DeviceSpan<int32_t> CohereState::GetCommittedSpan() const {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  return cpu->WrapMemory<int32_t>(
      std::span<int32_t>(const_cast<int32_t*>(committed_tokens_.data()), streamed_tokens_count_));
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

// --- CohereModel ---

std::unique_ptr<State> CohereModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<CohereState>(*this, params, sequence_lengths);
}

}  // namespace Generators
