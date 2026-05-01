// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "cohere_model.h"

#include <algorithm>
#include <cctype>
#include <map>

namespace Generators {

// --- CohereEncoderState ---

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

  // Now that num_frames is known, create the decoder
  decoder_state_ = std::make_unique<WhisperDecoderState>(model_, *params_, encoder_state_->GetNumFrames());

  // Create cross cache
  if (encoder_state_->HasCrossKVCacheOutputs()) {
    cross_cache_ = std::make_unique<CrossCache>(*this, encoder_state_->GetNumFrames() / 2);
    encoder_state_->AddCrossCache(cross_cache_);
    decoder_state_->AddCrossCache(cross_cache_);
    transpose_k_cache_buffer_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_cache_->GetShape(), cross_cache_->GetType());
  }

  if (!encoder_state_->HasCrossKVCacheOutputs()) {
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

  int new_num_frames = encoder_state_->GetNumFrames();

  encoder_state_->outputs_.clear();
  encoder_state_->output_names_.clear();

  decoder_state_ = std::make_unique<WhisperDecoderState>(model_, *params_, new_num_frames);

  if (encoder_state_->HasCrossKVCacheOutputs()) {
    cross_cache_ = std::make_unique<CrossCache>(*this, new_num_frames / 2);
    encoder_state_->AddCrossCache(cross_cache_);
    decoder_state_->AddCrossCache(cross_cache_);
    transpose_k_cache_buffer_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_cache_->GetShape(), cross_cache_->GetType());
  }

  if (!encoder_state_->HasCrossKVCacheOutputs()) {
    decoder_state_->inputs_.push_back(encoder_state_->GetHiddenStates());
    decoder_state_->input_names_.push_back(model_.config_->model.decoder.inputs.encoder_hidden_states.c_str());
  }

  // Reset state for new chunk
  first_run_ = true;
  current_chunk_++;
  return true;
}

void CohereState::SaveChunkTokens(const int32_t* tokens, size_t count) {
  completed_chunk_tokens_.emplace_back(tokens, tokens + count);
}

static std::string StripAsciiWhitespace(const std::string& s) {
  // Mirrors Python str.strip(): strips ASCII whitespace (space, \t, \n, \r, \v, \f).
  static const char* ws = " \t\n\r\v\f";
  const size_t start = s.find_first_not_of(ws);
  if (start == std::string::npos) return {};
  const size_t end = s.find_last_not_of(ws);
  return s.substr(start, end - start + 1);
}

// --- Boundary cleanup at chunk seams ----------------------------------------
//
// Chunks are emitted by the processor without audio overlap. The model tends
// to terminate each chunk with sentence-final punctuation even when the audio
// continues mid-thought, so naive concatenation produces seams like
//     "...payment related scams. In the UK come from..."
// To avoid this we strip trailing whitespace + punctuation + special tokens
// from each non-final chunk before committing. Punctuation set is
// multilingual (ASCII + Spanish + smart-quotes + ellipsis/em-dash).
//
// Special tokens for this tokenizer are the first 16 added_tokens (IDs 0..15
// are contiguous), so we treat token id < 16 as "always strip".

namespace {

constexpr int32_t kSpecialTokenIdMax = 16;  // IDs 0..15 are added_tokens (special).

// All UTF-8 punctuation/symbol byte sequences we treat as "trailing junk" at
// chunk boundaries. Multilingual: ASCII sentence-end + Spanish inverted marks
// + smart quotes + ellipsis/dashes + CJK punctuation.
const std::vector<std::string>& BoundaryPunctSequences() {
  static const std::vector<std::string> kSeqs = {
      // ASCII
      ".", ",", "!", "?", ";", ":", "-", "\"", "'",
      // Multilingual
      "\xC2\xBF",        // ¿
      "\xC2\xA1",        // ¡
      "\xC2\xAB",        // «
      "\xC2\xBB",        // »
      // Smart quotes
      "\xE2\x80\x9C",    // “
      "\xE2\x80\x9D",    // ”
      "\xE2\x80\x98",    // ‘
      "\xE2\x80\x99",    // ’
      "\xE2\x80\x9E",    // „
      // Ellipsis & dashes
      "\xE2\x80\xA6",    // …
      "\xE2\x80\x94",    // —
      "\xE2\x80\x93",    // –
  };
  return kSeqs;
}

// Returns true if `s`, after ASCII whitespace strip, consists entirely of
// punctuation sequences from BoundaryPunctSequences(). An empty string also
// returns true (pure whitespace token).
bool IsPunctOrEmpty(const std::string& s) {
  std::string t = StripAsciiWhitespace(s);
  if (t.empty()) return true;
  const auto& seqs = BoundaryPunctSequences();
  size_t i = 0;
  while (i < t.size()) {
    bool matched = false;
    for (const auto& seq : seqs) {
      if (t.compare(i, seq.size(), seq) == 0) {
        i += seq.size();
        matched = true;
        break;
      }
    }
    if (!matched) return false;
  }
  return true;
}

// Single-token decode helper. Returns the textual rendering of one token.
std::string DecodeOne(const Tokenizer& tokenizer, int32_t token_id) {
  return tokenizer.Decode(std::span<const int32_t>(&token_id, 1));
}

// Drop trailing tokens that are special, whitespace-only, or pure punctuation.
// Operates in place.
void StripTrailingBoundaryTokens(std::vector<int32_t>& tokens, const Tokenizer& tokenizer) {
  while (!tokens.empty()) {
    int32_t id = tokens.back();
    if (id >= 0 && id < kSpecialTokenIdMax) {
      tokens.pop_back();
      continue;
    }
    std::string s = DecodeOne(tokenizer, id);
    if (IsPunctOrEmpty(s)) {
      tokens.pop_back();
      continue;
    }
    break;
  }
}

// Lowercase the first ASCII uppercase letter of a SINGLE token's decoded text.
// Touches only the first non-special token — never re-encodes the rest of the
// chunk, so no whitespace tokens or word-boundary markers are introduced at
// the seam.
//
// Note: SentencePiece's Encode prepends a word-boundary marker to its input.
// If we encoded " in" we'd get [▁, ▁in] — TWO tokens, decoding to "  in"
// (double space). So we strip the leading space before encoding: Encode("in")
// returns a single token whose decoded form is already " in".
void LowercaseFirstChunkToken(std::vector<int32_t>& tokens, const Tokenizer& tokenizer) {
  // Find first non-special token.
  size_t first = 0;
  while (first < tokens.size() && tokens[first] >= 0 && tokens[first] < kSpecialTokenIdMax) {
    ++first;
  }
  if (first >= tokens.size()) return;

  std::string text = DecodeOne(tokenizer, tokens[first]);
  // Lowercase the first ASCII uppercase letter.
  bool changed = false;
  for (size_t i = 0; i < text.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    if (c >= 'A' && c <= 'Z') {
      text[i] = static_cast<char>(c + ('a' - 'A'));
      changed = true;
      break;
    }
    if (std::isalnum(c)) break;  // already lowercase or non-ASCII alpha
  }
  if (!changed) return;

  // Strip leading ASCII whitespace before encoding to avoid the SentencePiece
  // double-boundary artifact described above.
  size_t lead = 0;
  while (lead < text.size() && std::isspace(static_cast<unsigned char>(text[lead]))) ++lead;
  if (lead > 0) text.erase(0, lead);
  if (text.empty()) return;

  std::vector<int32_t> repl = tokenizer.Encode(text.c_str());
  if (repl.empty()) return;
  tokens.erase(tokens.begin() + first);
  tokens.insert(tokens.begin() + first, repl.begin(), repl.end());
}

}  // namespace

void CohereState::CommitChunkText(const std::vector<int32_t>& chunk_tokens,
                                  bool is_final, const Tokenizer& tokenizer) {
  if (chunk_tokens.empty()) return;

  std::vector<int32_t> cleaned(chunk_tokens);

  // For non-first chunks, lowercase the first letter so the seam with the
  // previous (trailing-punct-stripped) chunk reads as one continuing sentence.
  if (!committed_tokens_.empty()) {
    LowercaseFirstChunkToken(cleaned, tokenizer);
  }

  if (is_final) {
    // Final chunk: commit everything as-is. (Trailing punctuation here is
    // legitimate end-of-utterance content and should be preserved.)
    committed_tokens_.insert(committed_tokens_.end(), cleaned.begin(), cleaned.end());
    return;
  }

  // Non-final chunk: strip trailing whitespace/punctuation/special tokens so
  // the seam with the next chunk reads naturally when streamed token-by-token.
  StripTrailingBoundaryTokens(cleaned, tokenizer);
  committed_tokens_.insert(committed_tokens_.end(), cleaned.begin(), cleaned.end());
}


DeviceSpan<int32_t> CohereState::GetCommittedSpan() const {
  size_t n = std::min(streamed_tokens_count_, committed_tokens_.size());
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  return cpu->WrapMemory<int32_t>(
      std::span<int32_t>(const_cast<int32_t*>(committed_tokens_.data()), n));
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
